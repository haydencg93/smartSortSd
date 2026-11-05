"""Independent 100-image Smart Sort benchmark: legacy 6-class and expanded models.

Loads the class list from <checkpoint>.classes.json for expanded MobileNetV2
checkpoints. Does NOT import or modify app.py, training code, or the checkpoint.
The original 100-image manifest covers six materials, not all eleven classes.
Its disposal labels were historically assigned using provisional Iowa City rules;
this benchmark does NOT certify University of Iowa item-level disposal accuracy.
"""
import argparse, base64, csv, hashlib, html, io, json, math, statistics, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# The historical 100-image test contains these SIX ground-truth material labels.
# The model under test can have more outputs; see load_checkpoint_classes().
CLASSES = ('cardboard','glass','metal','paper','plastic','trash')
BINS = ('recycling','trash','landfill')
LEGACY_CLASSES = list(CLASSES)
# Mirror material-only routing in the current Smart Sort app.py.
UIOWA_RULES = {
    'cardboard': 'recycling',
    'celluloseacetate': 'landfill',
    'glass': 'landfill',
    'metal': 'recycling',
    'paper': 'recycling',
    'plastic': 'recycling',
    'polystyrene': 'landfill',
    'rubber': 'landfill',
    'tetrapak': 'recycling',
    'textile': 'landfill',
    'trash': 'landfill',
}


def bin_for_comparison(value):
    """Treat old 'trash' and new 'landfill' as equivalent bin *names*.

    This is NOT proof that Iowa City and UIowa have identical item rules.
    """
    value = str(value).strip().lower()
    return 'landfill' if value == 'trash' else value


def load_checkpoint_classes(checkpoint_path, state_dict):
    """Use train_data.py's adjacent class sidecar; legacy six-class fallback."""
    if 'classifier.1.weight' not in state_dict:
        raise ValueError('Checkpoint has no MobileNetV2 classifier.1.weight')
    num_outputs = int(state_dict['classifier.1.weight'].shape[0])
    sidecar = checkpoint_path.with_suffix('.classes.json')
    if sidecar.is_file():
        data = json.loads(sidecar.read_text(encoding='utf-8-sig'))
        classes = data.get('classes') if isinstance(data, dict) else data
        if (not isinstance(classes, list) or len(classes) != num_outputs
                or len(set(classes)) != len(classes)
                or not all(isinstance(c, str) and c for c in classes)):
            raise ValueError(f'Class list is missing/invalid or does not match checkpoint: {sidecar}')
    elif num_outputs == len(LEGACY_CLASSES):
        classes = LEGACY_CLASSES.copy()
    else:
        raise ValueError(f'Expanded model requires its matching metadata file: {sidecar}')
    missing_rules = set(classes) - set(UIOWA_RULES)
    if missing_rules:
        raise ValueError(f'No basic disposal route for model classes: {sorted(missing_rules)}')
    return classes
STATUS_COLORS = {'Passed (Fully)':'#16833a','Passed (w/ Concern)':'#2563eb','Failed':'#dc2626','Ignored':'#eab308'}
HEADERS = ['id','image','expected_material','expected_bin','reviewed','reviewer','rule_source','notes','source_dataset','source_index','sha256']

def decide(pred, expected, pred_bin, expected_bin, confidence, elapsed_ms, limit_ms=2500, threshold=.75):
    if pred != expected or bin_for_comparison(pred_bin) != bin_for_comparison(expected_bin):
        return 'Failed', 'Incorrect material' if pred != expected else 'Incorrect provisional bin decision'
    concerns=[]
    if confidence < threshold: concerns.append('Confidence below 75%')
    if elapsed_ms > limit_ms: concerns.append('Processing over 2.5 seconds')
    return ('Passed (w/ Concern)', '; '.join(concerns)) if concerns else ('Passed (Fully)', '')

def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def read_manifest(path, img_root, training_hashes=None, exact=100):
    if not path.exists(): raise ValueError(f'Missing manifest {path}; run prepare_neoavara.py first')
    with open(path,encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    if len(rows)!=exact: raise ValueError(f'Expected exactly {exact} image rows; found {len(rows)}. No results have been produced.')
    counts=Counter(); ids=set(); seen_hashes=set()
    for i,r in enumerate(rows,1):
        prefix=f'Row {i}'
        if r.get('id') in ids or not r.get('id'): raise ValueError(f'{prefix}: missing/duplicate id')
        ids.add(r['id'])
        m=r.get('expected_material','').strip().lower()
        if m not in CLASSES: raise ValueError(f'{prefix}: invalid material {m!r}')
        counts[m]+=1
        if r.get('expected_bin','').strip().lower() not in BINS: raise ValueError(f'{prefix}: missing expected_bin decision')
        if r.get('reviewed','').strip().lower() not in ('yes','true','1','auto'): raise ValueError(f'{prefix}: label has no review or auto-label provenance')
        if not r.get('reviewer','').strip() or not r.get('rule_source','').strip() or not r.get('notes','').strip():
            raise ValueError(f'{prefix}: reviewer/auto source, rule_source and notes are required')
        rel=Path(r.get('image',''))
        if not str(rel) or rel.is_absolute() or '..' in rel.parts: raise ValueError(f'{prefix}: invalid image path')
        p=img_root/rel
        if not p.is_file(): raise ValueError(f'{prefix}: image does not exist: {p}')
        sha=file_hash(p)
        if sha in seen_hashes: raise ValueError(f'{prefix}: duplicate image detected')
        if training_hashes and sha in training_hashes: raise ValueError(f'{prefix}: image overlaps training hashes')
        seen_hashes.add(sha)
        if r.get('sha256') and r['sha256'] != sha: raise ValueError(f'{prefix}: source image hash mismatch')
        r['sha256']=sha
    for c in CLASSES:
        if counts[c]<10: raise ValueError(f'Class {c}: {counts[c]}, require at least 10')
    return rows

def load_training_hashes(path):
    if not path: return None
    with open(path,encoding='utf-8') as f:
        return {ln.strip().split(',')[0] for ln in f if len(ln.strip().split(',')[0])==64}

def picture_uri(path):
    from PIL import Image,ImageOps
    try:
        with Image.open(path) as original:
            img=ImageOps.exif_transpose(original).convert('RGB'); img.thumbnail((130,105))
            b=io.BytesIO(); img.save(b,format='JPEG',quality=72)
            return 'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()
    except Exception: return ''

def chart_uri(results, mode):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(14,3.7 if mode!='pie' else 4.6),dpi=130)
    try:
        xs=list(range(1,len(results)+1))
        if mode=='times':
            ax.bar(xs,[r['elapsed_ms'] for r in results],color=[STATUS_COLORS[r['status']] for r in results],width=.85)
            ax.axhline(2500,color='#ca8a04',linestyle='--',label='2.5-second goal'); ax.set_ylabel('Preprocess + inference (ms)');ax.set_xlabel('Test number');ax.legend()
        elif mode=='confidence':
            ax.bar(xs,[r['confidence']*100 for r in results],color=[STATUS_COLORS[r['status']] for r in results],width=.85)
            ax.axhline(75,color='#ca8a04',linestyle='--',label='75% threshold');ax.set_ylim(0,105);ax.set_ylabel('Confidence (%)');ax.set_xlabel('Test number');ax.legend()
        elif mode=='pie':
            counts=Counter(r['status'] for r in results)
            labels=[k for k in STATUS_COLORS if counts[k]]
            if labels: ax.pie([counts[k] for k in labels],labels=[f'{k}: {counts[k]}' for k in labels],colors=[STATUS_COLORS[k] for k in labels],autopct='%1.1f%%',startangle=90)
        ax.set_title({'times':'Response time for each image','confidence':'Confidence for each image','pie':'Test outcome distribution'}[mode]);fig.tight_layout()
        b=io.BytesIO();fig.savefig(b,format='png',bbox_inches='tight');return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
    finally: plt.close(fig)

def make_html(results, output, info):
    counts=Counter(r['status'] for r in results)
    avg=statistics.mean(r['elapsed_ms'] for r in results) if results else 0
    correct=counts['Passed (Fully)']+counts['Passed (w/ Concern)']
    acc=correct/len(results) if results else 0
    def e(v): return html.escape(str(v),quote=True)
    cards=''.join(f'<div class="stat" style="border-top:5px solid {color}"><span>{e(k)}</span><strong>{counts[k]}</strong></div>' for k,color in STATUS_COLORS.items())
    cards+=f'<div class="stat"><span>Average processing time</span><strong>{avg:.1f} ms</strong></div><div class="stat"><span>Both-correct accuracy</span><strong>{acc:.1%}</strong></div>'
    trs=[]
    for r in results:
        preview=picture_uri(Path(info['images_root'])/r['image'])
        image_html=f'<img src="{preview}" alt="Image {e(r["id"])}">' if preview else 'No preview'
        badge=f'<span class="status" style="background:{STATUS_COLORS[r["status"]]}">{e(r["status"])}</span>'
        trs.append(f'<tr data-status="{e(r["status"])}"><td>{e(r["id"])}</td><td>{image_html}<div>{e(r["image"])}</div></td><td>{e(r["expected_material"])} / {e(r["expected_bin"])}</td><td>{e(r["predicted_material"])} / {e(r["predicted_bin"])}</td><td>{r["elapsed_ms"]:.2f}</td><td>{r["confidence"]*100:.1f}%</td><td>{badge}<small>{e(r["reason"])}</small></td></tr>')
    charts=''.join(f'<section><img class="chart" src="{chart_uri(results,k)}" alt="{k} chart"></section>' for k in ('times','pie','confidence'))
    doc=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Smart Sort | 100-image Model Benchmark</title><style>
    body{{font:15px system-ui,-apple-system,sans-serif;max-width:1400px;margin:0 auto;padding:26px;background:#f4f7fb;color:#172437}}h1{{margin-bottom:2px}}.muted{{color:#526173}}.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:12px;margin:25px 0}}.stat,section{{background:white;padding:16px;border-radius:14px;box-shadow:0 2px 10px #18283b10}}.stat span{{display:block;font-size:13px;color:#526173}}.stat strong{{display:block;font-size:25px;margin-top:8px}}.charts{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.charts section:first-child{{grid-column:1/-1}}.chart{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #e5e8ef;text-align:left;vertical-align:middle}}th{{position:sticky;top:0;background:#eaf0f9}}td img{{width:100px;max-height:80px;object-fit:contain;border-radius:6px}}td div{{overflow-wrap:anywhere;max-width:160px}}.status{{color:white;border-radius:6px;padding:5px 7px;display:inline-block;font-size:12px;font-weight:bold;white-space:nowrap}}small{{display:block;color:#526173;margin-top:5px}}.scroll{{overflow:auto;max-height:780px}}select,input{{padding:9px;margin:8px 8px 16px 0}}@media(max-width:800px){{.charts{{grid-template-columns:1fr}}.charts section:first-child{{grid-column:auto}}}}</style></head><body>
    <h1>Smart Sort — 100-image ML Benchmark</h1><p class="muted">Measured: {e(info['created'])} · Device: {e(info['device'])} · Model: {e(info['checkpoint'])} · Preprocessing + model forward pass only (no camera/network/UI)</p>
    <div class="stats">{cards}</div><section style="border:2px solid #c68a18;background:#fff8dd"><b>Ground-truth limitation:</b> {sum(r.get('reviewed','').lower()=='auto' for r in results)} of {len(results)} bin labels were generated automatically from material categories, not independently verified for the photographed items. These outcomes use historical provisional Iowa City bin labels ('trash' and 'landfill' names normalized) and do NOT certify University of Iowa item-level disposal accuracy. The source data may overlap existing training images.</section><p><b>Pass rule:</b> Material and expected bin must both match. Correct result with confidence &lt;75% or time &gt;2500 ms = blue concern. Any wrong result or inference exception = red failure. Ignored is reserved for future functionality and is not used in this evaluation.</p>
    <p><b>Interpretation:</b> confidence is softmax prediction confidence, not probability of correctness. All 100 evaluated images count in the denominator. This benchmark cannot establish independence from training data without provenance checks; hash lists detect exact overlap only.</p>
    <div class="charts">{charts}</div><section><h2>All {len(results)} image tests</h2><label>Status filter: <select id="filter"><option value="all">All</option>{''.join(f'<option>{e(k)}</option>' for k in STATUS_COLORS)}</select></label><label>Search: <input id="q" placeholder="Image, class, test ID"></label><div class="scroll"><table><thead><tr><th>Test</th><th>Image used</th><th>Expected material / bin</th><th>Predicted material / bin</th><th>Time (ms)</th><th>Confidence</th><th>Status / reason</th></tr></thead><tbody>{''.join(trs)}</tbody></table></div></section>
    <script>const filter=document.querySelector('#filter'),q=document.querySelector('#q');function update(){{document.querySelectorAll('tbody tr').forEach(r=>{{r.style.display=(filter.value==='all'||filter.value===r.dataset.status)&&r.textContent.toLowerCase().includes(q.value.toLowerCase())?'':'none'}})}}filter.addEventListener('change',update);q.addEventListener('input',update);</script></body></html>'''
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(doc,encoding='utf-8')

def benchmark(args):
    import torch
    import torch.nn as nn
    from PIL import Image,ImageOps
    from torchvision import models,transforms
    if not args.checkpoint.is_file(): raise ValueError(f'Model checkpoint missing: {args.checkpoint}')
    hashes=load_training_hashes(args.training_hashes)
    rows=read_manifest(args.manifest,args.images,hashes)
    device=torch.device(args.device)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model_classes = load_checkpoint_classes(args.checkpoint, state)
    print(f'Testing {len(model_classes)}-class checkpoint: {args.checkpoint}')
    print('Model classes:', ', '.join(model_classes))
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(model_classes))
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    tform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    results=[]
    for i,r in enumerate(rows,1):
        t0=time.perf_counter()
        material='ERROR'; bin_pred='ERROR'; confidence=0.; error=''
        try:
            with Image.open(args.images/r['image']) as im:
                img=ImageOps.exif_transpose(im).convert('RGB')
                tensor=tform(img).unsqueeze(0).to(device)
            with torch.inference_mode():
                probs=torch.softmax(model(tensor),1)[0]
                confidence,idx=torch.max(probs,0)
                confidence=float(confidence.item());material=model_classes[int(idx.item())];bin_pred=UIOWA_RULES[material]
            if device.type=='cuda': torch.cuda.synchronize()
        except Exception as exc: error=f'Inference error: {type(exc).__name__}: {exc}'
        elapsed_ms=(time.perf_counter()-t0)*1000
        status,reason=decide(material,r['expected_material'],bin_pred,r['expected_bin'],confidence,elapsed_ms)
        if error: status='Failed';reason=error
        results.append({**r,'predicted_material':material,'predicted_bin':bin_pred,'confidence':confidence,'elapsed_ms':elapsed_ms,'status':status,'reason':reason})
        print(f'{i:03d}/100 {status:<19} {elapsed_ms:>9.1f}ms  {r["id"]}',flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    with open(args.output/'results.csv','w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    summary={'counts':dict(Counter(r['status'] for r in results)),'average_ms':statistics.mean(r['elapsed_ms'] for r in results),'accuracy_both':sum(r['status'].startswith('Passed') for r in results)/len(results),'material_accuracy':sum(r['predicted_material']==r['expected_material'] for r in results)/len(results),'checkpoint_classes':model_classes,'checkpoint_class_count':len(model_classes),'bin_comparison_note':'Legacy trash and landfill names are normalized for comparison. Auto-filled item-bin labels use historical provisional Iowa City assumptions, not independently verified UIowa rules.','total':len(results),'automatically_labeled':sum(r.get('reviewed','').lower()=='auto' for r in results),'label_certification':'provisional material policy; not verified University of Iowa item-level ground truth' if any(r.get('reviewed','').lower()=='auto' for r in results) else 'user-reviewed','threshold_ms':2500,'confidence_threshold':.75,'device':str(device),'created':datetime.now(timezone.utc).isoformat()}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    make_html(results,args.output/'SmartSort_100_Image_Report.html',{'images_root':str(args.images),'created':summary['created'],'device':str(device),'checkpoint':args.checkpoint.name})
    print('Saved',args.output/'SmartSort_100_Image_Report.html')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path('heldout_neoavara/manifest.csv'))
    p.add_argument('--images',type=Path,default=Path('heldout_neoavara/images'))
    p.add_argument('--checkpoint',type=Path,default=Path('../smart_sort_mobilenetv2_combined.pth'))
    p.add_argument('--training-hashes',type=Path,help='Optional plain text file: one SHA256 per training/validation image (strongly recommended)')
    p.add_argument('--output',type=Path,default=Path('results_100'))
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    benchmark(p.parse_args())
if __name__=='__main__': main()
