"""Download 100 NeoAivara images with automatically generated, PROVISIONAL bin labels.
Requires `pip install datasets pillow` and internet access on user's machine.
"""
import argparse,csv,hashlib,io,random
from collections import defaultdict
from pathlib import Path
from PIL import Image
from benchmark import CLASSES,HEADERS

# Policy for the standard Iowa City mixed recycling bin, not glass drop-off.
# This is a material-level assumption, NOT an item-by-item verified decision.
AUTO_BIN={'cardboard':'recycling','glass':'trash','metal':'recycling','paper':'recycling','plastic':'recycling','trash':'trash'}
CITY_RULE_URL='https://www.icgov.org/government/departments-and-divisions/public-works/resource-management/garbage'
LABELS={'cardboard':'cardboard','brown-glass':'glass','green-glass':'glass','white-glass':'glass','metal':'metal','paper':'paper','plastic':'plastic','trash':'trash'}

def prepare(out:Path,seed=20261012):
    from datasets import load_dataset
    if (out/'manifest.csv').exists(): raise FileExistsError('manifest.csv exists. Refusing to overwrite reviewed labels')
    print('Downloading NeoAivara/Waste_Classification_data. The source may overlap other public datasets.')
    ds=load_dataset('NeoAivara/Waste_Classification_data',split='train')
    names=ds.features['label'].names
    eligible=defaultdict(list)
    for index,label in enumerate(ds['label']):
        original=names[int(label)]
        if original in LABELS:eligible[LABELS[original]].append((index,original))
    rng=random.Random(seed)
    for c in CLASSES:
        if len(eligible[c])<17:raise RuntimeError(f'Only {len(eligible[c])} entries for {c}, cannot reserve 100 with 10 each')
        rng.shuffle(eligible[c])
    # 20 for each of four classes, 10 for two = exactly 100; each class has >=10.
    quotas={'cardboard':20,'glass':20,'metal':20,'paper':20,'plastic':10,'trash':10}
    dest=out/'images';dest.mkdir(parents=True,exist_ok=True)
    rows=[];hashes=set()
    for cls in CLASSES:
        selected=0
        for idx,original in eligible[cls]:
            try:
                item=ds[int(idx)]['image']
                if not isinstance(item,Image.Image): item=Image.open(io.BytesIO(item['bytes']))
                image=item.convert('RGB')
                b=io.BytesIO();image.save(b,format='JPEG',quality=95)
                raw=b.getvalue();sha=hashlib.sha256(raw).hexdigest()
                if sha in hashes:continue
                hashes.add(sha)
                filename=f'{cls}_{idx:06d}.jpg';(dest/filename).write_bytes(raw)
                rows.append(dict(id=f'T{len(rows)+1:03d}',image=filename,expected_material=cls,expected_bin=AUTO_BIN[cls],reviewed='auto',reviewer='automated-material-policy',rule_source=CITY_RULE_URL,notes=f'PROVISIONAL, inferred from source material category {original}. Does not verify item subtype, cleanliness or eligibility. Plastic especially requires item-level review.',source_dataset='NeoAivara/Waste_Classification_data',source_index=str(idx),sha256=sha))
                selected+=1
                if selected==quotas[cls]:break
            except Exception as e:
                print(f'Skipping {cls} #{idx}: {e}')
        if selected<quotas[cls]: raise RuntimeError(f'Cannot obtain {quotas[cls]} valid images of {cls}')
    with open(out/'manifest.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=HEADERS);w.writeheader();w.writerows(rows)
    print(f'Reserved {len(rows)} images at {out}. Auto-filled provisional expected_bin for all rows.')
    print('WARNING: bin labels are material-level assumptions, not image-verified Iowa City ground truth.')
    print('Verify item-level eligibility, provenance, and overlaps with all training datasets.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path('heldout_neoavara'));parser.add_argument('--seed',type=int,default=20261012);a=parser.parse_args();prepare(a.out,a.seed)
