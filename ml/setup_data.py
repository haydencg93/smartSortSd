"""Centralize Smart Sort training sources in ONE raw_data directory.

  python setup_data.py                        # reprocess ALL registered sources
  python setup_data.py "New Dataset Folder"   # import/move new source and reprocess ALL
  python setup_data.py "data_6class" --license MIT --allow-non-cc0

A successful import MOVES the original source directory into raw_data/sources
(on the same drive), or copies and byte-verifies it when Windows denies rename. Nothing is permanently deleted: even unmapped HITL JSON
and images remain intact. Prepared images live in raw_data/train and raw_data/val.
The trained-model code reads these two folders directly.

Source license information is declarative, not independently verified. MIT
or unverified sources require the explicit --allow-non-cc0 opt-in. The HITL
Recycling/Litterati folder is known as CC0 based on its publisher's declaration.
"""
import argparse
import filecmp
import csv
import hashlib
import io
import math
import os
import json
import random
import re
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageOps, ImageChops, ImageStat

# Visual item classes; broad material names are intentionally not interpreted as
# recyclable containers. Old generic plastic/metal labels need item-level review.
CLASSES = ('battery', 'cardboard', 'carton', 'electronics', 'glass', 'light_bulb',
           'metal_can', 'paper', 'plastic_bag', 'plastic_container', 'rubber', 'trash')
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.jfif', '.webp', '.bmp', '.tif', '.tiff', '.gif'}
HITL_URL = 'https://humansintheloop.org/resources/datasets/recycling-dataset/'
# Conservative allowlist: do NOT map generic 'plastic', 'metal', 'aluminum',
# 'PET', 'other', 'mixed', 'paper pack', or 'cellulose acetate' to these classes.
# CLICK's 'Plastic Packet' / 'Silver Foil' are everyday landfill examples.
# License and image rights must be checked BEFORE importing a source.
MATERIAL_MAP = {
    'cardboard':'cardboard', 'corrugated cardboard':'cardboard',
    'paper':'paper', 'office paper':'paper', 'crumpled paper':'paper',
    'crumbled paper':'paper',
    'glass':'glass', 'glass bottle':'glass', 'brown glass':'glass',
    'green glass':'glass', 'white glass':'glass',
    'carton':'carton', 'tetrapak':'carton', 'tetra pak':'carton',
    'beverage carton':'carton', 'milk carton':'carton', 'juice carton':'carton',
    'plastic container':'plastic_container', 'plastic bottle':'plastic_container',
    'plastic cup':'plastic_container',
    'pet bottle':'plastic_container', 'hdpe bottle':'plastic_container',
    'metal can':'metal_can', 'aluminum can':'metal_can',
    'aluminium can':'metal_can', 'steel can':'metal_can', 'tin can':'metal_can',
    'food can':'metal_can', 'beverage can':'metal_can',
    'plastic bag':'plastic_bag', 'shopping bag':'plastic_bag',
    'grocery bag':'plastic_bag', 'carrier bag':'plastic_bag',
    'plastic shopping bag':'plastic_bag',
    'battery':'battery', 'batteries':'battery', 'battery waste':'battery', 'button battery':'battery',
    'rechargeable battery':'battery', 'lithium battery':'battery',
    'electronic device':'electronics', 'electronics':'electronics',
    'e waste':'electronics', 'electronic waste':'electronics',
    'cell phone':'electronics', 'mobile phone':'electronics', 'mobile':'electronics',
    'keyboard':'electronics', 'computer mouse':'electronics', 'mouse':'electronics',
    'pcb':'electronics', 'printed circuit board':'electronics',
    'laptop':'electronics', 'computer':'electronics',
    'light bulb':'light_bulb', 'lightbulb':'light_bulb',
    'lamp bulb':'light_bulb',
    'rubber':'rubber',
    'trash':'trash', 'miscellaneous trash':'trash', 'general waste':'trash',
    'litter':'trash', 'non recyclable':'trash', 'nonrecyclable':'trash',
    'plastic packet':'trash', 'plastic wrapper':'trash',
    'plastic film':'trash', 'plastic wrap':'trash',
    # Foil and foil trays are excluded until item-level policy is confirmed.
    'straw':'trash', 'plastic straw':'trash', 'plastic utensil':'trash',
    'paper towel':'trash', 'napkin':'trash', 'tissue':'trash',
    'styrofoam':'trash', 'polystyrene':'trash', 'expanded polystyrene':'trash',
    'thermocol':'trash', 'thermocol plate':'trash',
    'textile':'trash', 'textiles':'trash',
}

def normalize_label(value):
    if value is None: return None
    v = ' '.join(str(value).strip().lower().replace('_', ' ').replace('-', ' ').split())
    return MATERIAL_MAP.get(v)

def sha256_bytes(data): return hashlib.sha256(data).hexdigest()

def dhash(image):
    # 64-bit visual fingerprint for a conservative, near-identical photo check.
    im = ImageOps.grayscale(image).resize((9, 8))
    pixels = list(im.tobytes())
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

def extract_crop(path, rect=None, context_fraction=0.18):
    """Crop around a labeled object while preserving some scene background.

    The extra 18% per side reduces the white-background/isolated-item mismatch
    without treating every object in a busy photo as one material. No crop is
    made when the image comes from an already labeled ImageFolder directory.
    """
    with Image.open(path) as opened:
        im = ImageOps.exif_transpose(opened).convert('RGB')
        if rect is not None:
            x0, y0, x1, y1 = (float(x) for x in rect)
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            if x1-x0 < 24 or y1-y0 < 24: return None
            pad_x, pad_y = (x1-x0)*context_fraction, (y1-y0)*context_fraction
            x0=max(0, int(x0-pad_x)); y0=max(0, int(y0-pad_y))
            x1=min(im.width, int(x1+pad_x+0.99)); y1=min(im.height, int(y1+pad_y+0.99))
            if x1-x0 < 24 or y1-y0 < 24: return None
            im = im.crop((x0, y0, x1, y1))
        if min(im.size) < 24: return None
        return im

def supervisely_records(root, counters):
    """Read ds0/ann + ds0/img even when the export has no meta.json.

    Material labels must be explicit in the annotation tags. Numeric tag IDs
    are resolved only if a matching meta.json is available; never guess them.
    """
    ann_files = sorted(root.rglob('ann/*.json'))
    if not ann_files:
        return []

    meta_maps = {}
    for meta in root.rglob('meta.json'):
        try:
            data = json.loads(meta.read_text(encoding='utf-8-sig'))
            tag_metas = data.get('tag_metas', data.get('tagMetas', []))
            meta_maps[meta.parent.resolve()] = {
                str(tag.get('id')): tag.get('name')
                for tag in tag_metas if isinstance(tag, dict)
            }
        except (ValueError, UnicodeError):
            counters['unreadable_meta'] += 1

    records = []
    for ann_file in ann_files:
        counters['annotation_files'] += 1
        try:
            ann = json.loads(ann_file.read_text(encoding='utf-8-sig'))
        except (ValueError, UnicodeError):
            counters['invalid_annotations'] += 1
            continue
        if not isinstance(ann, dict):
            counters['invalid_annotations'] += 1
            continue
        name = ann_file.name[:-5]  # e.g. photo.jpg.json -> photo.jpg
        img_path = locate_image(root, name, ann_file.parent.parent)
        if img_path is None:
            counters['image_missing'] += 1
            continue
        lookup = {}
        for parent in ann_file.parents:
            found = meta_maps.get(parent.resolve())
            if found is not None:
                lookup = found
                break
        objects = ann.get('objects', [])
        if not objects:
            counters['no_objects'] += 1
        for obj in objects:
            if not isinstance(obj, dict):
                counters['invalid_objects'] += 1
                continue
            # Prefer an explicit per-object material tag. Some Supervisely exports
            # instead record a recognized material directly in classTitle.
            # Specific annotated object names beat generic materials, never
            # infer a plastic bottle from a plain "plastic" material tag.
            material = (normalize_label(obj.get('classTitle'))
                        or tags_to_material(obj.get('tags', []), lookup)
                        or tags_to_material(obj.get('attributes', {}), lookup))
            # An image-level material tag is safe only for one annotated object:
            # otherwise it could incorrectly label multiple distinct materials.
            if not material and len(objects) == 1:
                material = tags_to_material(ann.get('tags', []), lookup)
            if not material:
                counters['unmapped_material'] += 1
                continue
            points = obj.get('points', {}).get('exterior')
            if not points or len(points) < 2:
                counters['missing_bbox'] += 1
                continue
            try:
                xs, ys = [float(p[0]) for p in points], [float(p[1]) for p in points]
                rect = (min(xs), min(ys), max(xs), max(ys))
            except (ValueError, TypeError, IndexError):
                counters['invalid_bbox'] += 1
                continue
            records.append((img_path, material, rect, str(ann_file)))
            counters['recognized_objects'] += 1
    return records

def _coco_documents(root):
    """Find actual COCO annotations, not arbitrary JSON or single-class duplicates."""
    for f in sorted(root.rglob('*.json')):
        if f.name in ('meta.json', 'sources.json') or '/ann/' in f.as_posix():
            continue
        if 'single class' in f.as_posix().lower() or 'single_class' in f.as_posix().lower():
            continue
        try:
            doc=json.loads(f.read_text(encoding='utf-8-sig'))
        except (ValueError, UnicodeError):
            continue
        if (isinstance(doc,dict) and all(k in doc for k in ('images','annotations','categories'))
                and isinstance(doc['images'],list) and isinstance(doc['categories'],list)):
            yield f,doc


def _find_coco_image(root, json_file, file_name):
    """Resolve split-aware filenames; skip ambiguous same-named photos."""
    rel=Path(str(file_name).replace('\\','/'))
    if not rel.name or '..' in rel.parts:
        return None
    name=json_file.stem.lower()
    split=next((x for x in ('train','val','valid','test') if x in name),None)
    bases=[json_file.parent,json_file.parent.parent,root]
    candidates=[]
    for base in bases:
        candidates += [base/rel,base/'images'/rel]
        if split:
            candidates += [base/'images'/split/rel.name,base/split/rel.name]
    for c in candidates:
        if c.is_file() and c.suffix.lower() in EXTENSIONS: return c
    matches=[p for p in root.rglob(rel.name) if p.is_file() and p.suffix.lower() in EXTENSIONS]
    if split:
        matches=[p for p in matches if split in p.parts] or matches
    return matches[0] if len(matches)==1 else None


def coco_records(root,counters):
    records=[]
    for f,doc in _coco_documents(root):
        images={im.get('id'):im for im in doc['images'] if isinstance(im,dict)}
        categories={cat.get('id'):cat for cat in doc['categories'] if isinstance(cat,dict)}
        for ann in doc['annotations']:
            if not isinstance(ann,dict): continue
            meta=images.get(ann.get('image_id'))
            if not meta: continue
            cat=categories.get(ann.get('category_id'),{})
            # COCO class *name* must unambiguously describe a supported item.
            label=(normalize_label(cat.get('name'))
                   or tags_to_material(ann.get('attributes',{}))
                   or tags_to_material(ann.get('tags',[])))
            if not label:
                counters['unmapped_object_categories']+=1; continue
            bbox=ann.get('bbox')
            if not isinstance(bbox,(list,tuple)) or len(bbox)<4:
                counters['missing_bbox']+=1;continue
            try:
                x,y,w,h=map(float,bbox[:4]);rect=(x,y,x+w,y+h)
            except (TypeError,ValueError):
                counters['invalid_bbox']+=1;continue
            image=_find_coco_image(root,f,meta.get('file_name',''))
            if image is None:
                counters['image_missing']+=1;continue
            records.append((image,label,rect,str(f)))
    return records


def yolo_records(root,counters):
    """Read YOLO bounding boxes via each dataset's data.yaml class order.

    Do not fall back to single-class labels for datasets that include detailed
    multi-class annotations (for example CLICK). Ambiguous class names are
    intentionally excluded instead of counted as recyclable plastic/metal.
    """
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError('YOLO imports require PyYAML: pip install PyYAML') from exc
    configs=sorted(root.rglob('data.yaml'))+sorted(root.rglob('data.yml'))
    if not configs:
        raise ValueError('YOLO source missing data.yaml with original class names')
    records=[]
    detailed_seen=False
    for cfg in configs:
        if 'single class' in cfg.as_posix().lower(): continue
        data=yaml.safe_load(cfg.read_text(encoding='utf-8-sig'))
        if not isinstance(data,dict) or 'names' not in data:
            continue
        names=data['names']
        if isinstance(names,dict):
            names={int(k):v for k,v in names.items()}
        elif isinstance(names,list):
            names=dict(enumerate(names))
        else:
            raise ValueError(f'Invalid YOLO names field in {cfg}')
        if not any(normalize_label(n) for n in names.values()):
            continue
        # labels can be next to yaml or in nested multi-class/labels/ trees
        label_dirs=[cfg.parent/'labels']
        if not label_dirs[0].exists():
            label_dirs=[d for d in cfg.parent.rglob('labels') if d.is_dir()]
        for label_root in label_dirs:
            for label_file in sorted(label_root.rglob('*.txt')):
                rel=label_file.relative_to(label_root)
                split=next((a for a in rel.parts if a in ('train','val','valid','test')),None)
                candidates=[]
                for base in (cfg.parent,root,label_root.parent,label_root.parent.parent):
                    for ext in EXTENSIONS:
                        candidates.extend([base/'images'/rel.with_suffix(ext),
                                           base/'images'/(split or '')/(rel.stem+ext),
                                           base/rel.with_suffix(ext)])
                image=next((p for p in candidates if p.is_file()),None)
                if image is None:
                    matching=[p for p in root.rglob(rel.stem+'.*') if p.suffix.lower() in EXTENSIONS]
                    if split:
                        matching=[p for p in matching if split in p.parts] or matching
                    image=matching[0] if len(matching)==1 else None
                if image is None:
                    counters['image_missing']+=1;continue
                with Image.open(image) as im: width,height=im.size
                for line in label_file.read_text(encoding='utf-8-sig').splitlines():
                    parts=line.strip().split()
                    # YOLO detection: cls cx cy width height (5 tokens).
                    # YOLO segmentation: cls x1 y1 x2 y2 x3 y3 ... (odd tokens >= 7).
                    # Convert *valid* normalized polygons to bounding boxes for the
                    # existing classification cropper; never interpret arbitrary
                    # extra annotation fields as polygons.
                    if len(parts) != 5 and (len(parts) < 7 or len(parts) % 2 != 1):
                        counters['invalid_bbox'] += 1
                        continue
                    try:
                        idx = int(parts[0])
                        values = [float(value) for value in parts[1:]]
                    except ValueError:
                        counters['invalid_bbox'] += 1
                        continue
                    label = normalize_label(names.get(idx))
                    if not label:
                        counters['unmapped_object_categories'] += 1
                        continue
                    if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                        counters['invalid_bbox'] += 1
                        continue
                    if len(parts) == 5:
                        cx, cy, w, h = values
                        if not (w > 0 and h > 0):
                            counters['invalid_bbox'] += 1
                            continue
                        rect = ((cx - w/2) * width, (cy - h/2) * height,
                                (cx + w/2) * width, (cy + h/2) * height)
                    else:
                        xs, ys = values[::2], values[1::2]
                        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
                        if x1 <= x0 or y1 <= y0:
                            counters['invalid_bbox'] += 1
                            continue
                        rect = (x0 * width, y0 * height, x1 * width, y1 * height)
                        counters['converted_yolo_polygons'] += 1
                    records.append((image,label,rect,str(label_file)))
        detailed_seen=True
    if not detailed_seen:
        raise ValueError('No YOLO multi-class labels with a supported specific class')
    return records




# DWSD (Dense Waste Segmentation Dataset) official class IDs, as published in
# Data in Brief 59 (2025), 111340; 0 and 15 are background. ONLY the
# unambiguous object categories below are imported. Do not infer that any
# generic plastic/metal object qualifies as a campus-recyclable container.
# Keep DWSD's original Test split OUT of training for possible independent tests.
DWSD_ID_TO_CLASS = {
    1: 'plastic_container',  # plastic container
    2: 'plastic_container',  # plastic bottle
    3: 'trash',              # thermocol / foam
    6: 'glass',
    7: 'trash',              # thermocol plate
    9: 'paper',
    10: 'plastic_container', # plastic cup (object type only; cleanliness unknown)
}


def _dwsd_split_dirs(root):
    """Yield actual DWSD train/test Image+Mask pairs despite an outer DSWD wrapper."""
    root = Path(root)
    for split in sorted(root.rglob('*')):
        if not split.is_dir() or split.name.lower() not in ('train', 'test'):
            continue
        subs = {p.name.lower(): p for p in split.iterdir() if p.is_dir()}
        if 'image' in subs and 'mask' in subs:
            yield split.name.lower(), subs['image'], subs['mask']


def _dwsd_scalar_mask(path, counters):
    """Read 0..15 class-ID pixels without destroying P-mode palette indices.

    RGB files are allowed only when the three channels are identical. Colored
    visualization masks are NOT decoded by guessing their colors.
    """
    with Image.open(path) as im:
        im.load()
        if im.mode == 'L':
            out = im.copy()
        elif im.mode == 'P':
            out = Image.frombytes('L', im.size, im.tobytes())
        elif im.mode in ('RGB', 'RGBA'):
            red, green, blue = im.convert('RGB').split()
            if (ImageChops.difference(red, green).getbbox()
                    or ImageChops.difference(red, blue).getbbox()):
                counters['dwsd_unsupported_colored_masks'] += 1
                return None
            out = red
        else:
            counters['dwsd_unsupported_mask_modes'] += 1
            return None
    value_counts = out.getcolors(maxcolors=256)
    if value_counts is None or any(not 0 <= v <= 15 for count, v in value_counts):
        counters['dwsd_invalid_pixel_ids'] += 1
        return None
    return out, dict((value, count) for count, value in value_counts)


def dwsd_records(root, counters):
    """Use published DWSD semantic mask IDs to make conservative object crops.

    One bounding box per ID per original photo, not one full-scene label.
    A crop is accepted only when the named class is prominent in the box.
    Independent official Test images are reserved, not trained on.
    """
    records = []
    matched_train = False
    for split_name, image_dir, mask_dir in _dwsd_split_dirs(root):
        if split_name == 'test':
            counters['dwsd_reserved_test_masks'] += len(list(mask_dir.glob('*.png')))
            continue
        matched_train = True
        indices = defaultdict(list)
        for p in image_dir.rglob('*'):
            if p.is_file() and p.suffix.lower() in EXTENSIONS:
                suffix = re.search(r'(\d+)$', p.stem)
                if suffix:
                    indices[int(suffix.group(1))].append(p)
        for mask_file in sorted(mask_dir.glob('*.png')):
            suffix = re.search(r'(\d+)$', mask_file.stem)
            if not suffix:
                counters['dwsd_unpaired_masks'] += 1
                continue
            matched = indices.get(int(suffix.group(1)), [])
            if len(matched) != 1:
                counters['dwsd_unpaired_masks'] += 1
                continue
            image = matched[0]
            try:
                parsed = _dwsd_scalar_mask(mask_file, counters)
                if parsed is None:
                    continue
                mask, counts = parsed
                with Image.open(image) as img:
                    if img.size != mask.size:
                        counters['dwsd_image_mask_size_mismatch'] += 1
                        continue
            except (OSError, ValueError):
                counters['dwsd_unreadable_masks_or_images'] += 1
                continue
            for class_id, pixel_count in counts.items():
                if class_id in (0, 15):
                    continue
                output_class = DWSD_ID_TO_CLASS.get(class_id)
                if output_class is None:
                    counters['dwsd_unmapped_class_occurrences'] += 1
                    continue
                if pixel_count < 100:  # tiny isolated objects are not reliable crops
                    counters['dwsd_tiny_regions'] += 1
                    continue
                binary = mask.point([255 if n == class_id else 0 for n in range(256)])
                box = binary.getbbox()
                if not box:
                    continue
                width, height = box[2] - box[0], box[3] - box[1]
                if width < 24 or height < 24:
                    counters['dwsd_tiny_regions'] += 1
                    continue
                # Guard against union rectangles spanning numerous unrelated objects.
                if pixel_count / (width * height) < 0.12:
                    counters['dwsd_sparse_union_regions'] += 1
                    continue
                crop_counts = dict((v, n) for n, v in mask.crop(box).getcolors(256))
                foreground_pixels = sum(n for v, n in crop_counts.items() if v not in (0, 15))
                if foreground_pixels and pixel_count / foreground_pixels < .40:
                    counters['dwsd_mixed_category_regions'] += 1
                    continue
                records.append((image, output_class, box, str(mask_file)))
                counters['dwsd_accepted_mask_crops'] += 1
    if not matched_train:
        raise ValueError('DWSD import requires Train/Image and Train/Mask directories. '
                         'Do not import Test images as training data.')
    return records


def _labelme_documents(root):
    for f in sorted(root.rglob('*.json')):
        if f.name == 'meta.json' or '/ann/' in f.as_posix():
            continue
        try:
            doc=json.loads(f.read_text(encoding='utf-8-sig'))
        except (ValueError, UnicodeError):
            continue
        if (isinstance(doc,dict) and isinstance(doc.get('shapes'),list)
                and doc.get('imagePath')):
            yield f,doc


def labelme_records(root,counters):
    """Use LabelMe polygon/rectangle annotations, preserving image context."""
    records=[]
    for json_path,doc in _labelme_documents(root):
        name=str(doc['imagePath']).replace('\\','/')
        img_path=locate_image(root,name,json_path.parent)
        if img_path is None:
            matches=[p for p in root.rglob(Path(name).name)
                     if p.is_file() and p.suffix.lower() in EXTENSIONS]
            img_path=matches[0] if len(matches)==1 else None
        if img_path is None:
            counters['image_missing']+=1
            continue
        for shape in doc['shapes']:
            if not isinstance(shape,dict): continue
            label=normalize_label(shape.get('label'))
            if not label:
                counters['unmapped_object_categories']+=1
                continue
            points=shape.get('points')
            if not isinstance(points,list) or len(points)<2:
                counters['missing_bbox']+=1
                continue
            try:
                xs,ys=([float(p[0]) for p in points],
                       [float(p[1]) for p in points])
            except (ValueError,TypeError,IndexError):
                counters['invalid_bbox']+=1
                continue
            records.append((img_path,label,(min(xs),min(ys),max(xs),max(ys)),str(json_path)))
    return records

def hitl_records(folder, counters):
    # Most HITL exports are Supervisely (publisher states annotations made in Supervisely).
    records = supervisely_records(folder, counters)
    if not records: records = coco_records(folder, counters)
    if not records:
        raise ValueError('No material-tagged Supervisely or COCO objects found. '
                         f'Inspection counts: {dict(counters)}. '
                         'Please provide one sample ds0/ann/*.json; labels will not be guessed.')
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

def original_records(root, counters):
    """Yield image paths / labels from class folders, including old train/val.

    The original images are re-split together with HITL, so legacy validation
    images are NOT kept as a separate validation corpus and duplicates are
    removed across all sources before forming the new split.
    """
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f'Original image directory does not exist: {root}')
    split_roots = [root / split for split in ('train', 'val') if (root / split).is_dir()]
    search_roots = split_roots or [root]
    found = 0
    for base in search_roots:
        for class_folder in sorted(base.iterdir()):
            if not class_folder.is_dir():
                continue
            label = normalize_label(class_folder.name)
            if label is None:
                counters['unrecognized_original_class_folders'] += 1
                continue
            for image_path in sorted(class_folder.rglob('*')):
                if image_path.is_file() and image_path.suffix.lower() in EXTENSIONS:
                    found += 1
                    yield image_path, label
    if found == 0:
        raise ValueError(f'No labeled images found under {root}; expected train/<class>, '
                         'val/<class>, or <class> subfolders')


def audit_labels(folder, output):
    """Inventory raw Supervisely labels without loading a single image.

    This is useful when a supported material has zero mapped crops: it shows
    what the annotation JSON actually calls those objects. It never guesses
    that an unknown object is cardboard or trash.
    """
    sources = sorted(folder.rglob('ann/*.json'))
    if not sources:
        raise ValueError(f'No Supervisely ann/*.json files found below {folder}')
    all_class_titles = Counter()
    unmapped_class_titles = Counter()
    raw_material_values = Counter()
    tag_names = Counter()
    parsed = 0
    unreadable = 0
    object_count = 0
    unknown = 0
    for path in sources:
        try:
            ann = json.loads(path.read_text(encoding='utf-8-sig'))
            if not isinstance(ann, dict):
                unreadable += 1
                continue
        except (ValueError, UnicodeError):
            unreadable += 1
            continue
        parsed += 1
        for obj in ann.get('objects', []):
            if not isinstance(obj, dict):
                continue
            object_count += 1
            raw_title = str(obj.get('classTitle') or '<missing>')
            all_class_titles[raw_title] += 1
            tags = obj.get('tags', [])
            if isinstance(tags, list):
                for t in tags:
                    if not isinstance(t, dict):
                        continue
                    key = t.get('name') or t.get('tagName') or t.get('key') or ('tagId=' + str(t['tagId']) if 'tagId' in t else '<unknown>')
                    tag_names[str(key)] += 1
                    if str(key).strip().lower() == 'material':
                        raw_material_values[str(t.get('value', t.get('text', '<missing>')))] += 1
            elif isinstance(tags, dict):
                for k, v in tags.items():
                    tag_names[str(k)] += 1
                    if str(k).strip().lower() == 'material':
                        raw_material_values[str(v)] += 1
            attrs = obj.get('attributes', {})
            mat = tags_to_material(tags) or tags_to_material(attrs) or normalize_label(obj.get('classTitle'))
            if mat is None and len(ann.get('objects', [])) == 1:
                mat = tags_to_material(ann.get('tags', []))
            if mat is None:
                unknown += 1
                unmapped_class_titles[raw_title] += 1
    info = {
        'annotation_files': len(sources), 'parsed_annotation_files': parsed,
        'unreadable_annotation_files': unreadable, 'objects': object_count,
        'unmapped_by_current_label_mapping': unknown,
        'class_titles': dict(all_class_titles.most_common()),
        'unmapped_class_titles': dict(unmapped_class_titles.most_common()),
        'raw_material_tag_values': dict(raw_material_values.most_common()),
        'tag_keys': dict(tag_names.most_common()),
        'warning': 'Audit counts labels, not valid image crops. Unknown classes are not automatically classified as trash.',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(info, indent=2), encoding='utf-8')
    print('Label audit saved to', output)
    print('Unmapped class titles:', dict(unmapped_class_titles.most_common(30)))
    print('Raw material tags:', dict(raw_material_values.most_common(30)))
    return info



def infer_kind(folder):
    """Identify the existing import type, keeping COCO/YOLO out of ImageFolder."""
    if next(_dwsd_split_dirs(folder), None) is not None:
        return 'dwsd_masks'
    if next(folder.rglob('ann/*.json'),None) is not None:
        return 'supervisely'
    if next(_coco_documents(folder),None) is not None:
        return 'coco'
    if next(_labelme_documents(folder),None) is not None:
        return 'labelme'
    if (next(folder.rglob('data.yaml'),None) is not None
            or next(folder.rglob('data.yml'),None) is not None):
        return 'yolo'
    if any((folder/name).is_dir() for name in ('train','val')):
        return 'class_folders'
    if any(p.is_dir() and normalize_label(p.name) for p in folder.iterdir()):
        return 'class_folders'
    raise ValueError(f'Unsupported dataset structure in {folder}. Expected DWSD masks, COCO, YOLO, '
                     'LabelMe, Supervisely, or train/val/<class> directories.')

def license_for(folder, supplied, allow_non_cc0):
    """Record provenance; never infer license from anonymous class folders."""
    kind = infer_kind(folder)
    if supplied:
        license_name = supplied.upper().replace(' ', '-')
    elif kind == 'supervisely' and ('litterati' in folder.name.lower()
                                  or 'humans in the loop' in folder.name.lower()):
        license_name = 'CC0'  # Publisher-declared for this particular dataset.
    else:
        license_name = 'UNVERIFIED'
    if license_name not in ('CC0', 'CC0-1.0', 'PUBLIC-DOMAIN') and not allow_non_cc0:
        raise ValueError(f'{folder.name}: license {license_name} is not verified CC0. '
                         'For old MIT TrashNet data use --license MIT --allow-non-cc0; '
                         'for independently verified CC0 data use --license CC0.')
    return kind, license_name


def _source_id(folder):
    return folder.name.replace('\\', '_').replace('/', '_')


def _read_registered_sources(root, allow_non_cc0):
    sources_root = root / 'sources'
    registry = root / 'sources.json'
    existing_meta = {}
    if registry.is_file():
        info = json.loads(registry.read_text(encoding='utf-8'))
        existing_meta = {p['folder']: p for p in info.get('sources', [])}
    found = []
    if sources_root.is_dir():
        for directory in sorted(sources_root.iterdir()):
            if not directory.is_dir():
                continue
            meta = existing_meta.get(directory.name)
            if meta is None:
                raise ValueError(f'Registered source {directory} has no license metadata. '
                                 'Do not guess its provenance; import it through setup_data.py.')
            lic = meta.get('license', 'UNVERIFIED')
            if lic not in ('CC0', 'CC0-1.0', 'PUBLIC-DOMAIN'):
                # An explicitly accepted source stays accepted for future rebuilds.
                if not meta.get('non_cc0_opt_in', False):
                    raise ValueError(f'Missing non-CC0 opt-in record for {directory}')
            kind = infer_kind(directory)
            found.append({'path': directory, 'folder': directory.name,
                          'license': lic, 'kind': kind,
                          'source_url': meta.get('source_url', ''),
                          'non_cc0_opt_in': bool(meta.get('non_cc0_opt_in', False))})
    return found


def _unique_destination(root, name):
    safe = ''.join(c if c.isalnum() or c in ' _.-' else '_' for c in name).strip(' .')
    safe = safe or 'import'
    destination = root / safe
    n = 2
    while destination.exists():
        destination = root / f'{safe}_{n}'
        n += 1
    return destination


def _install_prepared(source, destination):
    """Publish a prepared entry, tolerating Windows USB directory-rename denials.

    Windows occasionally refuses directory renames on removable media even when
    there is no conflicting destination. In that case copy the staged directory
    into the now-vacant target and verify the copy. The rollback owner (_publish)
    restores the previous prepared data if this fails.
    """
    if destination.exists():
        raise FileExistsError(f'Prepared destination unexpectedly exists: {destination}')
    try:
        os.replace(source, destination)
        return
    except PermissionError as exc:
        # The source remains in staging. In particular, do NOT touch the
        # permanent original data below raw_data/sources.
        if destination.exists():
            raise PermissionError(
                f'Windows refused to replace {destination}; another process may '
                f'have opened or recreated the target folder. Close File Explorer, '
                f'Python trainers and virus-scanner windows, then retry.') from exc
        print(f'Windows blocked a rename for {destination.name}; '
              'using verified same-drive copy instead.', flush=True)

    if source.is_dir():
        shutil.copytree(source, destination)
        for original in source.rglob('*'):
            if original.is_file():
                copied = destination / original.relative_to(source)
                if not copied.is_file() or copied.stat().st_size != original.stat().st_size:
                    raise OSError(f'Prepared copy verification failed: {copied}')
    else:
        shutil.copy2(source, destination)
        if destination.stat().st_size != source.stat().st_size:
            raise OSError(f'Prepared copy verification failed: {destination}')


def _publish(stage, root):
    """Publish prepared data with rollback, preserving original sources.

    On Windows, renaming a staged nonempty directory to raw_data/val can return
    WinError 5 even after moving the previous directory to a backup. A verified
    copy fallback handles that specific removable-drive behavior.
    """
    backup = root / '.previous_prepared'
    if backup.exists():
        raise ValueError(
            f'Interrupted backup found at {backup}. Do NOT delete it: inspect '
            'its train/val folders before trying to publish again.')
    backup.mkdir()
    moved_old = []
    installed = []
    try:
        for name in ('train', 'val', 'source_manifest.json'):
            current = root / name
            if current.exists():
                os.replace(current, backup / name)
                moved_old.append(name)
        for name in ('train', 'val', 'source_manifest.json'):
            target = root / name
            # Record BEFORE attempting installation so a partially copied
            # folder is removed and old data can be rolled back.
            installed.append(name)
            _install_prepared(stage / name, target)
    except Exception as cause:
        recovery_errors = []
        for name in reversed(installed):
            target = root / name
            try:
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
            except OSError as error:
                recovery_errors.append(f'could not remove partial {name}: {error}')
        for name in moved_old:
            old = backup / name
            try:
                if (root / name).exists():
                    raise FileExistsError(f'Cannot restore over {root / name}')
                os.replace(old, root / name)
            except OSError as error:
                recovery_errors.append(f'could not restore original {name}: {error}')
        if recovery_errors:
            raise RuntimeError(
                f'Preparing new train/val failed ({cause}). Automatic rollback '
                f'was incomplete: {"; ".join(recovery_errors)}. The backup '
                f'is kept at {backup}; do NOT delete it.') from cause
        raise
    finally:
        if backup.exists() and not any(backup.iterdir()):
            backup.rmdir()
    if backup.exists():
        shutil.rmtree(backup)



def _archive_registered_source(source, destination):
    """Archive an imported source; fall back to a verified copy on WinError 5.

    The original remains untouched if Windows refuses a directory rename.
    A successful copy is retained under raw_data/sources and can be used by the
    existing registry and future rebuilds exactly like a moved source.
    """
    if destination.exists():
        raise FileExistsError(f'Destination already exists: {destination}. '
                             'Inspect it before retrying; no files were replaced.')
    try:
        os.rename(source, destination)
        print(f'Moved source into {destination}', flush=True)
        return
    except PermissionError as exc:
        if destination.exists():
            raise PermissionError(
                f'Windows refused to rename {source} because destination '
                f'{destination} exists or was created during the operation. '
                'Do not overwrite it; inspect the folder first.') from exc
        print(f'Windows denied directory rename of {source}; trying a verified '
              'copy instead. The original folder will be preserved.', flush=True)

    # A Windows handle on the source directory can block rename but often still
    # permits copy. Never delete source or overwrite an existing destination.
    try:
        shutil.copytree(source, destination)
        originals = {p.relative_to(source) for p in source.rglob('*') if p.is_file()}
        copied = {p.relative_to(destination) for p in destination.rglob('*') if p.is_file()}
        if not originals or originals != copied:
            raise OSError('Copied source inventory does not match original files')
        for relative in sorted(originals):
            if not filecmp.cmp(source / relative, destination / relative, shallow=False):
                raise OSError(f'Copied source differs from original: {relative}')
    except (OSError, ValueError) as exc:
        raise OSError(f'Copy/verification failed. Original remains at {source}. '
                      f'An incomplete copy MAY remain at {destination}; inspect '
                      'it before retrying. Registry and prepared data have not '
                      f'been published. Reason: {exc}') from exc
    print(f'Verified copied source into {destination}. '
          f'Original preserved at {source}; you may archive it separately '
          'after the full import succeeds.', flush=True)


def prepare(raw_data=Path('raw_data'), add=None, license_name=None,
            allow_non_cc0=False, seed=42, val_fraction=0.20,
            source_url='', min_class_images=25, min_class_origins=8, dry_run=False):
    """Build from all stored sources plus optional new source; move it only on success.

    Called from setup_data.py or train_data.py. No deletion of external source.
    """
    raw_data = Path(raw_data).resolve()
    if not 0 < val_fraction < 0.5:
        raise ValueError('Validation fraction must be greater than 0 and less than 0.5')
    if add:
        add = Path(add).resolve()
        if not add.is_dir():
            raise ValueError(f'New dataset directory does not exist: {add}')
        # One-time adoption of an old raw_data/dataset-resized folder is safe;
        # never let an import swallow raw_data itself or prepared output folders.
        legacy_child = add.parent == raw_data and add.name not in {
            'train', 'val', 'sources', '.previous_prepared'
        }
        if add == raw_data or add in raw_data.parents or (
                raw_data in add.parents and not legacy_child):
            raise ValueError('New dataset must be outside raw_data, except for a '
                             'direct legacy source folder such as raw_data/dataset-resized')
        kind, lic = license_for(add, license_name, allow_non_cc0)
        if raw_data.exists() and os.stat(add).st_dev != os.stat(raw_data).st_dev:
            raise ValueError('New dataset and raw_data must be on the same drive. '
                             'Move the source to the D: drive before importing; '
                             'this prevents risky cross-drive copy-and-delete behavior.')
    raw_data.mkdir(parents=True, exist_ok=True)
    sources_dir = raw_data / 'sources'
    sources_dir.mkdir(exist_ok=True)
    if add and os.stat(add).st_dev != os.stat(raw_data).st_dev:
        raise ValueError('New dataset must be on the same drive as raw_data for a safe move')
    registered = _read_registered_sources(raw_data, allow_non_cc0)
    upcoming = None
    if add:
        dest = _unique_destination(sources_dir, add.name)
        upcoming = {'path': add, 'folder': dest.name, 'license': lic,
                    'kind': kind, 'source_url': source_url,
                    'non_cc0_opt_in': bool(allow_non_cc0 and lic not in
                                            ('CC0', 'CC0-1.0', 'PUBLIC-DOMAIN'))}
    source_infos = registered + ([upcoming] if upcoming else [])
    if not source_infos:
        raise ValueError('No sources in raw_data/sources yet. Import your original '
                         'and HITL folders one time, for example:\n'
                         '  python setup_data.py ".\\data_6class" --license MIT --allow-non-cc0\n'
                         '  python setup_data.py ".\\Litterati Collection and Labeling Data" --license CC0')
    if min_class_images < 2 or min_class_origins < 2:
        raise ValueError('Minimum class images and origins must both be >= 2')
    counters = Counter()
    accepted_by_source = Counter()
    records = {}
    origin_records = defaultdict(list)
    class_counts = Counter()
    hash_buckets = defaultdict(set)
    # Stage on D: inside raw_data to avoid filling the C: drive.
    with tempfile.TemporaryDirectory(prefix='.setup_stage_', dir=raw_data) as tmp:
        stage = Path(tmp) / 'prepared'
        scratch = stage / '.unsplit'
        scratch.mkdir(parents=True)

        def add_image(image, label, origin, source, annotation):
            if image is None:
                counters['invalid_crops'] += 1
                return
            if label not in CLASSES:
                counters['unmapped_material'] += 1
                return
            image = image.convert('RGB')
            digest = hashlib.sha256(f'{image.width}x{image.height}:'.encode('ascii')
                                    + image.tobytes()).hexdigest()
            if digest in records:
                counters['conflicting_duplicates' if records[digest]['label'] != label
                         else 'exact_duplicates'] += 1
                return
            fp = dhash(image)
            thumb = image.resize((32, 32)).tobytes()
            parts = (fp & ((1 << 22)-1), (fp >> 22) & ((1 << 21)-1), fp >> 43)
            possible = set()
            for j, part in enumerate(parts):
                possible.update(hash_buckets[(j, part)])
            candidate_img = Image.frombytes('RGB', (32, 32), thumb)
            duplicate = None
            for other_digest in possible:
                old = records[other_digest]
                if (old['dhash'] ^ fp).bit_count() > 2:
                    continue
                other = Image.frombytes('RGB', (32, 32), old['thumbnail'])
                if max(ImageStat.Stat(ImageChops.difference(candidate_img, other)).rms) <= 6.0:
                    duplicate = old
                    break
            if duplicate:
                counters['conflicting_near_duplicates' if duplicate['label'] != label
                         else 'near_duplicates'] += 1
                return
            path = scratch / f'{digest}.jpg'
            image.save(path, format='JPEG', quality=92)
            records[digest] = {'label': label, 'origin': origin,
                               'source': source, 'annotation': annotation,
                               'sha256': digest, 'dhash': fp, 'thumbnail': thumb,
                               'path': path}
            origin_records[origin].append(digest)
            accepted_by_source[source] += 1
            class_counts[label] += 1
            for j, part in enumerate(parts):
                hash_buckets[(j, part)].add(digest)

        for info in source_infos:
            path = info['path']
            print(f'Reading {path} ({info["kind"]}, {info["license"]})', flush=True)
            if info['kind'] in ('supervisely', 'coco', 'yolo', 'labelme', 'dwsd_masks'):
                readers = {'supervisely':hitl_records, 'coco':coco_records, 'yolo':yolo_records,
                           'labelme':labelme_records, 'dwsd_masks':dwsd_records}
                for img_path, material, rect, annotation in readers[info['kind']](path, counters):
                    try:
                        crop = extract_crop(img_path, rect)
                    except (OSError, ValueError):
                        counters['unreadable_images'] += 1
                        continue
                    origin = f'{info["folder"]}/{img_path.relative_to(path).as_posix()}'
                    add_image(crop, material, origin,
                              f'{info["folder"]} ({info["license"]})',
                              str(Path(annotation).relative_to(path)))
            else:
                for img_path, material in original_records(path, counters):
                    try:
                        crop = extract_crop(img_path)
                    except (OSError, ValueError):
                        counters['unreadable_images'] += 1
                        continue
                    origin = f'{info["folder"]}/{img_path.relative_to(path).as_posix()}'
                    add_image(crop, material, origin,
                              f'{info["folder"]} ({info["license"]})',
                              str(img_path.relative_to(path)))

        if not records:
            raise ValueError(f'No recognizable images or annotations. Skipped: {dict(counters)}')
        if upcoming and accepted_by_source[f'{upcoming["folder"]} ({upcoming["license"]})'] == 0:
            raise ValueError('New source supplied zero unambiguous, unique training objects. '
                             'Inspect excluded class names/annotations; source was not moved.')
        by_label_origins = defaultdict(set)
        for rec in records.values():
            by_label_origins[rec['label']].add(rec['origin'])
        too_few = {c: {'images': class_counts[c], 'origins': len(by_label_origins[c])}
                   for c in CLASSES if class_counts[c] and
                   (class_counts[c] < min_class_images
                    or len(by_label_origins[c]) < min_class_origins)}
        for digest, rec in list(records.items()):
            if rec['label'] in too_few:
                rec['path'].unlink()
                records.pop(digest)
                origin_records[rec['origin']].remove(digest)
                class_counts[rec['label']] -= 1
        classes = sorted(c for c in CLASSES if class_counts[c] >= min_class_images)
        if len(classes) < 2:
            raise ValueError(f'At least two material classes need two distinct '
                             f'original images each. Available: {dict(class_counts)}; '
                             f'undercounted: {too_few}')

        rng = random.Random(seed)
        origins = sorted(origin for origin, digests in origin_records.items() if digests)
        rng.shuffle(origins)
        by_origin = {origin: Counter(records[d]['label'] for d in digests)
                     for origin, digests in origin_records.items() if digests}
        target = {label: max(1, round(class_counts[label] * val_fraction)) for label in classes}
        val_count = Counter()
        val_origins = set()
        for label in sorted(target, key=lambda c: class_counts[c]):
            while val_count[label] < target[label]:
                candidates = []
                for origin in origins:
                    if origin in val_origins or not by_origin[origin][label]:
                        continue
                    group = by_origin[origin]
                    if all(class_counts[c] - val_count[c] - count >= 1
                           for c, count in group.items()):
                        overflow = sum(max(0, val_count[c] + n - target[c])
                                       for c, n in group.items())
                        candidates.append((overflow, origin))
                if not candidates:
                    break
                chosen = min(candidates)[1]
                val_origins.add(chosen)
                val_count.update(by_origin[chosen])
        missing_val = [c for c in classes if not 0 < val_count[c] < class_counts[c]]
        if missing_val:
            raise ValueError(f'Cannot make leakage-safe train/val groups for {missing_val}; '
                             'add more independent photographs. Nothing was moved.')

        file_records = []
        for digest, rec in records.items():
            split = 'val' if rec['origin'] in val_origins else 'train'
            destination = stage / split / rec['label'] / f'{digest}.jpg'
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(rec['path'], destination)
            file_records.append({k: rec[k] for k in ('label', 'origin', 'source',
                                                     'annotation', 'sha256')}
                                | {'split': split, 'file': destination.name})
        scratch.rmdir()
        expected = len(file_records)
        actual = sum(1 for split in ('train', 'val')
                     for _ in (stage / split).rglob('*.jpg'))
        if actual != expected:
            raise ValueError(f'Output verification failed ({actual} != {expected}). '
                             'Source directory was not moved.')
        source_metadata = [
            {'folder': x['folder'], 'license': x['license'],
             'kind': x['kind'], 'source_url': x.get('source_url',''),
             'non_cc0_opt_in': x['non_cc0_opt_in']}
            for x in source_infos]
        manifest = {
            'classes': classes,
            'class_to_idx': {label: i for i, label in enumerate(classes)},
            'counts': {c: int(class_counts[c]) for c in classes},
            'source_count': len(source_infos),
            'license_policy': ('MIXED / not verified CC0: explicitly allowed'
                               if any(x['non_cc0_opt_in'] for x in source_infos)
                               else 'CC0 / public domain, as declared by sources'),
            'sources': source_metadata,
            'skipped_classes': too_few,
            'minimum_class_images': min_class_images,
            'minimum_independent_origins': min_class_origins,
            'dedup_and_skipped': dict(counters),
            'split_counts': {'train': dict(class_counts - val_count),
                             'val': dict(val_count)},
            'records': file_records,
        }
        (stage / 'source_manifest.json').write_text(json.dumps(manifest, indent=2),
                                                    encoding='utf-8')

        if dry_run:
            print('DRY RUN - source archives and current prepared training data NOT changed.')
            print('Candidate classes:', manifest['counts'])
            print('Skipped:', dict(counters))
            return manifest
        if upcoming:
            # An atomic SAME-DRIVE rename moves the entire raw archive into raw_data.
            # Crucially, this also preserves annotation JSONs/materials we skipped.
            source_destination = sources_dir / upcoming['folder']
            _archive_registered_source(add, source_destination)
        registry = raw_data / 'sources.json'
        new_registry = raw_data / '.sources_new.json'
        new_registry.write_text(json.dumps({'sources': source_metadata}, indent=2),
                                encoding='utf-8')
        # Keep metadata alongside moved sources even if publishing fails.
        os.replace(new_registry, registry)
        _publish(stage, raw_data)
        print(f'Prepared {expected} deduplicated images from {len(source_infos)} source(s).')
        print('Classes:', manifest['counts'])
        print('Duplicate/skip counts:', dict(counters))
        print('Saved train/val in:', raw_data)
        return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('new_dataset', nargs='?', type=Path,
                   help='New extracted HITL or labeled-class directory to import')
    p.add_argument('--add', type=Path, help='Alternative to positional new_dataset')
    p.add_argument('--raw-data', type=Path, default=Path('raw_data'),
                   help='ONE permanent storage directory; default raw_data')
    p.add_argument('--license', help='License of new source: CC0, PUBLIC-DOMAIN, MIT, etc')
    p.add_argument('--allow-non-cc0', action='store_true',
                   help='Explicitly accept an MIT/unverified original dataset')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-fraction', type=float, default=0.20)
    p.add_argument('--source-url', default='', help='Original URL for attribution in data provenance')
    p.add_argument('--min-class-images', type=int, default=25,
                   help='Skip classes with fewer independent usable crops (default 25)')
    p.add_argument('--min-class-origins', type=int, default=8,
                   help='Minimum different original photographs per active class')
    p.add_argument('--dry-run', action='store_true', help='Audit preparation without publishing or moving sources')
    # Compatibility with previous setup_data.py flags.
    p.add_argument('--hitl-archive', type=Path, help='Alias for new dataset DIRECTORY')
    p.add_argument('--original-data', type=Path, help='Alias for new class-folder dataset')
    p.add_argument('--original-license', help='Alias for --license when using --original-data')
    p.add_argument('--inspect-labels', action='store_true')
    p.add_argument('--audit-output', type=Path, default=Path('hitl_label_audit.json'))
    args = p.parse_args()
    sources = [x for x in (args.new_dataset, args.add, args.hitl_archive,
                           args.original_data) if x is not None]
    if len(sources) > 1:
        p.error('Supply only one source folder per import; sources accumulate in raw_data')
    add = sources[0] if sources else None
    if args.inspect_labels:
        if not add:
            p.error('--inspect-labels requires an annotation folder')
        audit_labels(add, args.audit_output)
        return
    try:
        prepare(raw_data=args.raw_data, add=add,
                license_name=args.license or args.original_license,
                allow_non_cc0=args.allow_non_cc0,
                seed=args.seed, val_fraction=args.val_fraction,
                source_url=args.source_url,
                min_class_images=args.min_class_images,
                min_class_origins=args.min_class_origins,
                dry_run=args.dry_run)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    main()
