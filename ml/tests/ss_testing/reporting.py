"""Actual-label evaluation visualizations; no fabricated accuracy values."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw
from .core import CLASSES, RULES


def wilson(correct: int, count: int, z=1.9599639845):
    if count==0:return [None,None]
    p=correct/count
    d=1+z*z/count
    m=(p+z*z/(2*count))/d
    a=(z/d)*math.sqrt(p*(1-p)/count+z*z/(4*count*count))
    return [round(m-a,4),round(m+a,4)]


def savefig(path):
    plt.tight_layout()
    plt.savefig(path,dpi=180,bbox_inches='tight',facecolor='white')
    plt.close()


def cmplot(actual, predicted, labels, title, path, normalize=False):
    cm = confusion_matrix(actual,predicted,labels=labels)
    cm_disp = cm.astype(float)
    if normalize:
        cm_disp = np.divide(cm_disp,cm.sum(1,keepdims=True),out=np.zeros_like(cm_disp),where=cm.sum(1,keepdims=True)!=0)*100
    plt.figure(figsize=(8.5,6.8))
    plt.imshow(cm_disp,cmap='Blues',vmin=0,vmax=(100 if normalize else None))
    plt.colorbar(label=('Row %' if normalize else 'Number of images'))
    plt.xticks(range(len(labels)),labels,rotation=35,ha='right')
    plt.yticks(range(len(labels)),labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val=cm_disp[i,j]
            plt.text(j,i,f'{val:.1f}%' if normalize else str(int(val)),ha='center',va='center',
                     color=('white' if val > 0.55*np.max(cm_disp) and val>0 else 'black'),fontsize=9)
    plt.xlabel('Predicted');plt.ylabel('Actual');plt.title(title)
    savefig(path)


def gallery(frame: pd.DataFrame, path: Path, correct: bool, max_images=12):
    subset=frame[frame['label'].eq(frame['predicted']) if correct else frame['label'].ne(frame['predicted'])]
    if subset.empty:return
    subset=subset.sort_values('confidence',ascending=correct).head(max_images)
    fig, axes=plt.subplots(3,4,figsize=(13,11))
    for ax in axes.flat:ax.axis('off')
    for ax,(_,r) in zip(axes.flat,subset.iterrows()):
        try:
            with Image.open(r['path']) as image:
                ax.imshow(ImageOps.exif_transpose(image).convert('RGB'))
            ax.set_title(f"True: {r['label']}\nPred: {r['predicted']}  ({r['confidence']:.0%})",fontsize=10)
        except Exception:ax.set_title('Image not available')
        ax.axis('off')
    fig.suptitle('Correct examples' if correct else 'Misclassified examples',fontsize=16)
    savefig(path)


def write_accuracy_report(rows, output: Path, metadata: dict):
    output.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows)
    if frame.empty:raise ValueError('No predictions to evaluate')
    frame.to_csv(output/'predictions.csv',index=False)
    actual=list(frame.label)
    predicted=list(frame.predicted)
    actual_bin=[RULES[c] for c in actual]
    predicted_bin=[RULES[c] for c in predicted]
    mat_acc=accuracy_score(actual,predicted)
    bin_acc=accuracy_score(actual_bin,predicted_bin)
    n=len(frame)
    labels_seen=sorted(set(actual))
    per_cls=classification_report(actual,predicted,labels=list(CLASSES),zero_division=0,output_dict=True)
    report={**metadata,'n_evaluated':n,'classes_present':labels_seen,
            'material_accuracy':round(float(mat_acc),5),
            'material_balanced_accuracy':round(float(balanced_accuracy_score(actual,predicted)),5),
            'material_accuracy_wilson95':wilson(sum(a==b for a,b in zip(actual,predicted)),n),
            'current_code_bin_accuracy':round(float(bin_acc),5),
            'current_code_bin_accuracy_wilson95':wilson(sum(a==b for a,b in zip(actual_bin,predicted_bin)),n),
            'six_class_report':per_cls}
    # Explicit independence assumptions and minimum sample size; not a certified field pass.
    eligible=(metadata.get('complete_dataset',False) and set(CLASSES)<=set(actual)
              and n>=100 and metadata.get('train_overlap_status')=='checked_no_exact_overlap')
    report['engineering_75pct_gate']={'status':('provisional_pass' if mat_acc>=.75 else 'provisional_fail') if eligible else 'not_certifiable',
          'why':'External-data score is not an Iowa City representative held-out field trial; independence and per-item policy require further validation.'}
    if 'actual_bin' in frame and frame.actual_bin.notna().any():
        valid=frame.actual_bin.isin(['trash','recycling'])
        r=frame[valid]
        if not r.empty:
            a=list(r.actual_bin); p=[RULES[x] for x in r.predicted]
            report['annotated_iowa_city_bin_accuracy']=round(float(accuracy_score(a,p)),5)
            report['annotated_iowa_city_bin_count']=len(r)
            report['annotated_iowa_city_bin_accuracy_wilson95']=wilson(sum(x==y for x,y in zip(a,p)),len(r))
    (output/'metrics.json').write_text(json.dumps(report,indent=2,default=float),encoding='utf-8')
    cmplot(actual,predicted,CLASSES,'Six-class confusion matrix (counts)',output/'material_confusion.png')
    cmplot(actual,predicted,CLASSES,'Six-class confusion matrix (row normalized)',output/'material_confusion_normalized.png',True)
    cmplot(actual_bin,predicted_bin,['recycling','trash'],'Current-rule disposal confusion matrix',output/'routing_confusion.png')
    names=list(CLASSES)
    f1=[per_cls[c]['f1-score'] for c in names]
    precision=[per_cls[c]['precision'] for c in names]
    recall=[per_cls[c]['recall'] for c in names]
    x=np.arange(len(names));w=0.24
    plt.figure(figsize=(10,5.5));plt.bar(x-w,precision,w,label='Precision');plt.bar(x,recall,w,label='Recall');plt.bar(x+w,f1,w,label='F1')
    plt.xticks(x,names);plt.ylim(0,1.05);plt.ylabel('Score');plt.title('Per-class classification performance');plt.legend()
    savefig(output/'per_class_scores.png')
    # Reliability diagram: confidence reported for material prediction, not bin confidence.
    conf=frame.confidence.astype(float).to_numpy()
    correct=(frame.label==frame.predicted).to_numpy().astype(float)
    edges=np.linspace(0,1,11);counts=[];avg_conf=[];avg_acc=[];ece=0.0
    for i in range(10):
        mask=(conf>=edges[i])&(conf<(edges[i+1]) if i<9 else conf<=edges[i+1])
        count=int(mask.sum());counts.append(count)
        if count:
            c=float(conf[mask].mean());a=float(correct[mask].mean());ece+=count/len(conf)*abs(c-a)
        else:c=a=np.nan
        avg_conf.append(c);avg_acc.append(a)
    report['expected_calibration_error_10bin']=round(ece,5)
    (output/'metrics.json').write_text(json.dumps(report,indent=2,default=float),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(7,6))
    ax.plot([0,1],[0,1],ls='--',color='gray',label='Ideal calibration')
    ax.plot([c for c in avg_conf if not np.isnan(c)], [a for a in avg_acc if not np.isnan(a)],marker='o',label='Observed bins')
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted confidence',ylabel='Observed accuracy',title=f'Reliability diagram (ECE={ece:.3f})')
    ax.legend();savefig(output/'calibration.png')
    if 'latency_ms' in frame:
        plt.figure(figsize=(9,5))
        plt.hist(frame.latency_ms,bins=min(35,max(5,int(np.sqrt(n)))),edgecolor='white')
        plt.axvline(5000,linestyle='--',color='red',label='5,000 ms limit')
        plt.xlabel('Measured inference latency (ms)');plt.ylabel('Images');plt.title('Latency distribution (excludes camera/UI)');plt.legend()
        savefig(output/'latency_distribution.png')
        report['inference_latency_ms']={k:round(float(np.percentile(frame.latency_ms,p)),2)
                                        for k,p in [('p50',50),('p95',95),('p99',99)]}
        (output/'metrics.json').write_text(json.dumps(report,indent=2,default=float),encoding='utf-8')
    gallery(frame,output/'correct_gallery.png',True)
    gallery(frame,output/'errors_gallery.png',False)
    return report
