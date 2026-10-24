#!/usr/bin/env python3
"""Independent six-class image accuracy benchmarking for Smart Sort.

Example:
 python evaluate_external.py --dataset /path/to/RealWaste --source realwaste \
   --model ../smart_sort_mobilenetv2_6class.pth --out results/realwaste

This is evaluation, not training. Do not train on the dataset you use for final testing.
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageOps

from ss_testing.core import CLASSES, load_checkpoint, predict, scan_dataset, duplicates_vs_training
from ss_testing.reporting import write_accuracy_report


def cli():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset',type=Path,required=True,help='Folder containing class subfolders')
    p.add_argument('--source',choices=['imagefolder','realwaste'],default='realwaste')
    p.add_argument('--model',type=Path,default=Path(__file__).resolve().parent.parent/'smart_sort_mobilenetv2_6class.pth')
    p.add_argument('--out',type=Path,default=Path('results/external'))
    p.add_argument('--max-per-class',type=int,default=0,help='0=entire dataset; otherwise stratified subset')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--train-dir',type=Path,help='Check exact-image duplicate overlap with original training images')
    p.add_argument('--actual-bin-csv',type=Path,help='Optional verified per-image city-specific ground truth; columns relative_path,actual_bin')
    p.add_argument('--robustness',action='store_true',help='Low-light, blur, JPEG and occlusion test')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    return p.parse_args()


def perturb(image,scenario):
    image=ImageOps.exif_transpose(image).convert('RGB')
    if scenario=='low_light':return ImageEnhance.Brightness(image).enhance(0.30)
    if scenario=='blur':return image.filter(ImageFilter.GaussianBlur(radius=3))
    if scenario=='jpeg_30':
        import io
        buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=30)
        buffer.seek(0);return Image.open(buffer).convert('RGB')
    if scenario=='occluded_center':
        image=image.copy();w,h=image.size
        ImageDraw.Draw(image).rectangle((int(w*.35),int(h*.35),int(w*.65),int(h*.65)),fill=(110,110,110))
        return image
    raise ValueError(scenario)


def main():
    args=cli()
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(min(4,max(1,torch.get_num_threads())))
    samples,missing=scan_dataset(args.dataset,args.source)
    total_by_class={c:sum(s.label==c for s in samples) for c in CLASSES}
    selected=[];random_gen=random.Random(args.seed)
    for label in CLASSES:
        pool=[s for s in samples if s.label==label]
        if args.max_per_class and len(pool)>args.max_per_class:
            pool=random_gen.sample(pool,args.max_per_class)
        selected+=pool
    selected.sort(key=lambda s:s.relative_path)
    overlap=[]
    if args.train_dir:
        overlap=duplicates_vs_training(selected,args.train_dir)
        selected=[s for s in selected if s.relative_path not in set(overlap)]
    city_bins={}
    if args.actual_bin_csv:
        with args.actual_bin_csv.open(newline='',encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row['actual_bin'] not in {'trash','recycling'}:raise ValueError('actual_bin must be trash or recycling')
                city_bins[row['relative_path'].replace('\\','/')]=row['actual_bin']
    model=load_checkpoint(args.model,args.device)
    print(f'Loaded {args.model.name}; evaluating {len(selected)} external images. Other ignored classes: {missing}.')
    rows=[];robust=[];unreadable=[]
    for index,s in enumerate(selected):
        try:
            with Image.open(s.path) as img:
                im=img.convert('RGB')
            start=time.perf_counter()
            result=predict(model,im,args.device)
            latency_ms=(time.perf_counter()-start)*1000
            rows.append({'path':str(s.path),'relative_path':s.relative_path,'label':s.label,
                         'predicted':result['material'],'predicted_bin':result['classification'],
                         'confidence':result['confidence'],'probabilities':json.dumps(result['probabilities']),
                         'latency_ms':latency_ms,'actual_bin':city_bins.get(s.relative_path)})
            if args.robustness:
                for scenario in ['low_light','blur','jpeg_30','occluded_center']:
                    r=predict(model,perturb(im,scenario),args.device)
                    robust.append({'relative_path':s.relative_path,'label':s.label,'scenario':scenario,
                                   'predicted':r['material'],'confidence':r['confidence'],
                                   'same_prediction_as_original':r['material']==result['material'],
                                   'correct':r['material']==s.label})
        except (OSError, ValueError, RuntimeError) as e:
            unreadable.append({'file':s.relative_path,'error':str(e)[:150]})
        if (index+1)%100==0: print(f'   {index+1}/{len(selected)} images ...',flush=True)
    metadata={'dataset':str(args.dataset.resolve()),'source':args.source,'model':str(args.model.resolve()),
              'original_dataset_counts':total_by_class,'missing_class_folders':missing,
              'subset_cap':args.max_per_class,'seed':args.seed,'complete_dataset':args.max_per_class==0,
              'unreadable_images':unreadable,'skipped_exact_duplicates':overlap,
              'train_overlap_status':('checked_no_exact_overlap' if args.train_dir else 'unknown_training_provenance'),
              'evaluation_note':'RealWaste origin distinct from TrashNet, but original .pth training provenance cannot be proven from the weight file. Exact hashes do not rule out modified duplicates. Iowa City policy not inferred from material alone.'}
    if len(rows)==0:raise RuntimeError('Could not read any evaluation images')
    report=write_accuracy_report(rows,args.out,metadata)
    if robust:
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from ss_testing.reporting import savefig
        f=pd.DataFrame(robust);f.to_csv(args.out/'robustness_predictions.csv',index=False)
        summary=f.groupby('scenario',sort=False).agg(accuracy=('correct','mean'),prediction_stability=('same_prediction_as_original','mean'),n=('correct','size')).reset_index()
        summary.to_csv(args.out/'robustness_summary.csv',index=False)
        x=np.arange(len(summary));w=.32
        plt.figure(figsize=(9,5));plt.bar(x-w/2,summary.accuracy,w,label='Accuracy');plt.bar(x+w/2,summary.prediction_stability,w,label='Stability')
        plt.xticks(x,list(summary.scenario));plt.ylim(0,1);plt.title('Real-image robustness under controlled perturbations');plt.ylabel('Share of images');plt.legend()
        savefig(args.out/'robustness.png')
        report['robustness_summary']=summary.to_dict(orient='records')
        (args.out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['n_evaluated','material_accuracy','current_code_bin_accuracy','engineering_75pct_gate']},indent=2))
    print('Results:',args.out.resolve())

if __name__=='__main__':main()
