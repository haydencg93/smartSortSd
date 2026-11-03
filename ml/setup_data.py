"""Build six-class Smart Sort ImageFolder data from HITL Recycling exports.

The CC0 HITL download is supplied through a publisher form; download it at
https://humansintheloop.org/resources/datasets/recycling-dataset/
Then run: python setup_data.py --hitl-archive path/to/recycling.zip --output data_6class

Accepted exports: Supervisely projects (meta.json, ann/*.json, img/*) or
COCO JSON (images, annotations, categories) with per-object MATERIAL tags.
Supports --cc0-manifest (manual photographs) and opt-in legacy MIT TrashNet.
No uncertain labels are guessed. Only six original labels are used.
"""
import argparse
import csv
import hashlib
import io
import json
import random
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageOps, ImageChops, ImageStat

CLASSES = ('cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash')
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
HITL_URL = 'https://humansintheloop.org/resources/datasets/recycling-dataset/'
# Deliberately conservative: "mixed", "other", composite packaging etc are ambiguous.
MATERIAL_MAP = {
    'cardboard': 'cardboard', 'corrugated cardboard': 'cardboard',
    'glass': 'glass', 'glass bottle': 'glass',
    'metal': 'metal', 'aluminium': 'metal', 'aluminum': 'metal', 'steel': 'metal', 'tin': 'metal',
    'paper': 'paper',
    'plastic': 'plastic', 'pet': 'plastic', 'hdpe': 'plastic',
    'trash': 'trash', 'general waste': 'trash', 'non recyclable': 'trash', 'non-recyclable': 'trash',
}

def normalize_label(value):
    if value is None: return None
    v = ' '.join(str(value).strip().lower().replace('_', ' ').replace('-', ' ').split())
    return MATERIAL_MAP.get(v)

def sha256_bytes(data): return hashlib.sha256(data).hexdigest()

def dhash(image):
    # 64-bit visual fingerprint for a conservative, near-identical photo check.
    im = ImageOps.grayscale(image).resize((9, 8))
    pixels = list(im.getdata())
    return sum((1 << (y*8+x)) for y in range(8) for x in range(8)
               if pixels[y*9+x] > pixels[y*9+x+1])

def tags_to_material(tags, meta_lookup=None):
    """Require a MATERIAL field. Don't use 'object' or 'brand' as material."""
    if isinstance(tags, dict):
        for k, value in tags.items():
            if str(k).strip().lower() == 'material':
                return normalize_label(value)
    if isinstance(tags, list):
        for tag in tags:
            if not isinstance(tag, dict): continue
            key = tag.get('name') or tag.get('tagName') or tag.get('key')
            if not key and meta_lookup:
                key = meta_lookup.get(str(tag.get('tagId', tag.get('id', ''))))
            if str(key).lower() == 'material':
                return normalize_label(tag.get('value', tag.get('text')))
    return None

def locate_image(root, candidate, parent=None):
    candidate = str(candidate).replace('\\', '/')
    name = Path(candidate).name
    if not name or '..' in Path(candidate).parts: return None
    variants = []
    if parent:
        variants += [parent / 'img' / name, parent / 'images' / name, parent / name]
    variants += [root / candidate]
    for v in variants:
        if v.is_file() and v.suffix.lower() in EXTENSIONS: return v
    return None

def extract_crop(path, rect=None):
    with Image.open(path) as opened:
        im = ImageOps.exif_transpose(opened).convert('RGB')
        if rect is not None:
            x0, y0, x1, y1 = (round(float(x)) for x in rect)
            x0, x1 = sorted((max(0, x0), min(im.width, x1)))
            y0, y1 = sorted((max(0, y0), min(im.height, y1)))
            if x1-x0 < 24 or y1-y0 < 24: return None
            im = im.crop((x0, y0, x1, y1))
        if min(im.size) < 24: return None
        return im

def supervisely_records(root, counters):
    metas = list(root.rglob('meta.json'))
    if not metas: return []
    # One meta.json may serve several project subsets.
    records = []
    for meta in metas:
        metadata = json.loads(meta.read_text(encoding='utf-8-sig'))
        lookup = {str(t.get('id')): t.get('name') for t in metadata.get('tag_metas', [])}
        ann_files = list(meta.parent.rglob('ann/*.json'))
        for ann_file in ann_files:
            ann = json.loads(ann_file.read_text(encoding='utf-8-sig'))
            # Supervisely's ann filename is image filename + '.json'.
            name = ann_file.name[:-5]
            img_path = locate_image(root, name, ann_file.parent.parent)
            if img_path is None:
                counters['image_missing'] += 1
                continue
            objects = ann.get('objects', [])
            if not objects:
                counters['no_objects'] += 1
            for obj in objects:
                material = tags_to_material(obj.get('tags', []), lookup)
                if not material:
                    # Image-level material is usable ONLY for a single annotated object.
                    if len(objects) == 1:
                        material = tags_to_material(ann.get('tags', []), lookup)
                if not material:
                    counters['unmapped_material'] += 1
                    continue
                points = obj.get('points', {}).get('exterior')
                if not points or len(points) < 2:
                    counters['missing_bbox'] += 1
                    continue
                xs, ys = [float(p[0]) for p in points], [float(p[1]) for p in points]
                rect = (min(xs), min(ys), max(xs), max(ys))
                records.append((img_path, material, rect, str(ann_file)))
    return records

def coco_records(root, counters):
    records = []
    for f in root.rglob('*.json'):
        if f.name == 'meta.json' or '/ann/' in f.as_posix(): continue
        try: doc = json.loads(f.read_text(encoding='utf-8-sig'))
        except (ValueError, UnicodeError): continue
        if not isinstance(doc, dict) or not all(k in doc for k in ('images', 'annotations', 'categories')): continue
        images = {im['id']: im for im in doc['images']}
        categories = {cat['id']: cat for cat in doc['categories']}
        anns_by_img = defaultdict(list)
        for a in doc['annotations']: anns_by_img[a.get('image_id')].append(a)
        for image_id, anns in anns_by_img.items():
            meta = images.get(image_id)
            if not meta: continue
            img_path = locate_image(root, meta.get('file_name', ''), f.parent)
            if img_path is None:
                matches = list(root.rglob(Path(meta.get('file_name','')).name))
                img_path = matches[0] if len(matches)==1 else None
            if img_path is None:
                counters['image_missing'] += 1; continue
            for ann in anns:
                cat = categories.get(ann.get('category_id'), {})
                material = (tags_to_material(ann.get('attributes', {}))
                            or tags_to_material(ann.get('tags', []))
                            or tags_to_material(cat.get('attributes', {})))
                if not material and len(anns)==1:
                    material = tags_to_material(meta.get('attributes', {}))
                # Only use category name if it actually denotes a supported material.
                material = material or normalize_label(cat.get('name'))
                if not material:
                    counters['unmapped_material'] += 1; continue
                bbox = ann.get('bbox')
                if not isinstance(bbox, list) or len(bbox) < 4:
                    counters['missing_bbox'] += 1; continue
                x, y, w, h = map(float, bbox[:4]); rect = (x,y,x+w,y+h)
                records.append((img_path,material,rect,str(f)))
    return records

def hitl_records(folder, counters):
    # Most HITL exports are Supervisely (publisher states annotations made in Supervisely).
    records = supervisely_records(folder, counters)
    if not records: records = coco_records(folder, counters)
    if not records:
        raise ValueError('No material-tagged Supervisely or COCO objects found. '
                         'Inspect the export and provide a sample annotation JSON; labels will not be guessed.')
    return records

def cc0_records(csv_path):
    csv_path = Path(csv_path).resolve()
    with csv_path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = {'path','label','license','source_url','rights_verified'}
        if not fields.issubset(reader.fieldnames or []): raise ValueError(f'Manifest requires {sorted(fields)}')
        for row in reader:
            lic = row['license'].upper().strip().replace(' ','-')
            if lic not in {'CC0','CC0-1.0','PUBLIC-DOMAIN'} or row['rights_verified'].lower().strip() not in ('yes','true','1'):
                raise ValueError('Non-CC0 or unverified rights in manifest')
            label = normalize_label(row['label'])
            if not label: raise ValueError(f"Unsupported label: {row['label']}")
            p = Path(row['path']); p = (p if p.is_absolute() else csv_path.parent / p).resolve()
            if not p.is_file() or p.suffix.lower() not in EXTENSIONS: raise ValueError(f'Missing image: {p}')
            with Image.open(p) as image: im = ImageOps.exif_transpose(image).convert('RGB')
            yield (im, label, str(row['source_url']).strip(), str(p))

def build(args):
    if not args.hitl_archive and not args.cc0_manifest and not args.include_trashnet:
        raise ValueError('Provide --hitl-archive, --cc0-manifest, or --include-trashnet')
    if args.output.exists() and not args.overwrite:
        raise ValueError(f'Output exists: {args.output}. Back it up and rerun with --overwrite')
    if not 0 < args.val_fraction < .5: raise ValueError('--val-fraction must be between 0 and 0.5')
    counters = Counter(); unique = {}; perceptual = {}; groups = defaultdict(list)
    with tempfile.TemporaryDirectory(prefix='smartsort_') as tmp:
        root = Path(tmp)
        if args.hitl_archive:
            archive = args.hitl_archive
            if archive.is_dir(): hitl_folder = archive
            else:
                if not zipfile.is_zipfile(archive): raise ValueError(f'Expected a ZIP or folder: {archive}')
                hitl_folder = root/'hitl'; hitl_folder.mkdir()
                with zipfile.ZipFile(archive) as z:
                    for member in z.infolist():
                        target = (hitl_folder/member.filename).resolve()
                        if not target.is_relative_to(hitl_folder.resolve()) or member.is_dir() and not member.filename.endswith('/'):
                            raise ValueError('Unsafe archive path')
                    z.extractall(hitl_folder)
            for path, label, rect, annotation in hitl_records(hitl_folder, counters):
                img = extract_crop(path, rect)
                if img is None:
                    counters['invalid_crops'] += 1; continue
                groups[str(path.resolve())].append((img, label, HITL_URL, annotation))
        if args.cc0_manifest:
            for img, label, source, path in cc0_records(args.cc0_manifest):
                groups[path].append((img, label, source, path))
        if args.include_trashnet:
            from huggingface_hub import hf_hub_download
            archive = hf_hub_download(repo_id='garythung/trashnet',filename='dataset-resized.zip',repo_type='dataset')
            with zipfile.ZipFile(archive) as z:
                for member in z.infolist():
                    if member.is_dir() or Path(member.filename).suffix.lower() not in EXTENSIONS: continue
                    label = Path(member.filename).parent.name
                    if label not in CLASSES: continue
                    with Image.open(io.BytesIO(z.read(member))) as img:
                        groups['trashnet/'+member.filename].append((img.convert('RGB'),label,'TrashNet (MIT, not CC0)',member.filename))
        # Assign all crops from the same original photo to the same split.
        for origin, items in sorted(groups.items()):
            for image, label, source, annotation in items:
                raw = io.BytesIO(); image.save(raw, format='PNG')
                pixel_bytes = raw.getvalue(); fingerprint = dhash(image)
                pixel_digest = sha256_bytes(pixel_bytes)
                if pixel_digest in unique:
                    if unique[pixel_digest]['label'] != label: counters['conflicting_duplicates'] += 1
                    else: counters['exact_duplicates'] += 1
                    continue
                # Only suppress near-identical images with Hamming distance <= 2.
                possible = [k for k in perceptual if (k ^ fingerprint).bit_count() <= 2]
                # A hash match alone is not enough (blank backgrounds collide).
                thumbnail = image.resize((32, 32))
                matches = []
                for p in possible:
                    for d in perceptual[p]:
                        candidate = unique[d]
                        other = Image.open(io.BytesIO(candidate['image'])).resize((32, 32))
                        rms = ImageStat.Stat(ImageChops.difference(thumbnail, other)).rms
                        if max(rms) <= 6.0:
                            matches.append(candidate)
                if matches:
                    if any(m['label'] != label for m in matches): counters['conflicting_near_duplicates'] += 1
                    else: counters['near_duplicates'] += 1
                    continue
                unique[pixel_digest] = {'image': pixel_bytes,'label':label,'origin':origin,'source':source,'annotation':annotation,'sha256':pixel_digest}
                perceptual.setdefault(fingerprint, []).append(pixel_digest)
        if not unique: raise ValueError(f'No usable images. Skipped: {dict(counters)}')
        counts = Counter(x['label'] for x in unique.values())
        missing = [c for c in CLASSES if counts[c] < 2]
        if missing:
            raise ValueError(f'Not enough CC0 examples for {missing}. HITL may not cover all six classes. '
                             f'Add own CC0 photos via --cc0-manifest. Counts: {dict(counts)}. Skipped: {dict(counters)}')
        group_by_label = defaultdict(list)
        for rec in unique.values(): group_by_label[rec['label']].append(rec)
        rng = random.Random(args.seed)
        # Group by origin, then reserve entire original images in one split.
        train = set(); val = set()
        origin_labels = defaultdict(set)
        for rec in unique.values(): origin_labels[rec['origin']].add(rec['label'])
        origins = sorted(origin_labels); rng.shuffle(origins)
        for origin in origins:
            scores = {label:sum(1 for r in unique.values() if r['origin']==origin and r['label']==label) for label in origin_labels[origin]}
            should_val = all(sum(1 for r in unique.values() if r['origin'] in val and r['label']==l)+scores[l] <= max(1, round(counts[l]*args.val_fraction)) for l in scores)
            (val if should_val else train).add(origin)
        if any(not any(r['origin'] in val and r['label']==c for r in unique.values()) or not any(r['origin'] in train and r['label']==c for r in unique.values()) for c in CLASSES):
            raise ValueError('Cannot produce nonempty train/val for all six classes with photo-group splitting. Add images.')
        stage = root/'ready'; metadata=[]
        for rec in unique.values():
            split = 'val' if rec['origin'] in val else 'train'
            folder = stage/split/rec['label']; folder.mkdir(parents=True, exist_ok=True)
            filename = rec['sha256']+'.png'; (folder/filename).write_bytes(rec['image'])
            metadata.append({k:rec[k] for k in ('label','origin','source','annotation','sha256')} | {'split':split,'file':filename})
        (stage/'source_manifest.json').write_text(json.dumps({'license_policy':'CC0 only' if not args.include_trashnet else 'MIXED: contains MIT TrashNet',
            'source':HITL_URL if args.hitl_archive else None, 'counts':dict(counts), 'skipped':dict(counters), 'records':metadata},indent=2),encoding='utf-8')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists(): shutil.rmtree(args.output)
        shutil.copytree(stage,args.output)
        print('Prepared:',len(metadata),'samples; class counts:',dict(counts))
        print('Deduplication:',dict(counters))
        print('Training data:',args.output)
        if args.include_trashnet: print('WARNING: contains MIT-licensed TrashNet; not CC0-only')

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--hitl-archive',type=Path,help='Publisher-provided HITL archive (.zip) or extracted directory')
    p.add_argument('--cc0-manifest',type=Path,help='Optional additional original/CC0 photographs CSV')
    p.add_argument('--include-trashnet',action='store_true',help='Explicit legacy MIT source; violates CC0-only condition')
    p.add_argument('--output',type=Path,default=Path('data_6class'))
    p.add_argument('--overwrite',action='store_true')
    p.add_argument('--val-fraction',type=float,default=.2)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    try: build(a)
    except (ValueError, FileNotFoundError) as exc: p.error(str(exc))

if __name__=='__main__': main()
