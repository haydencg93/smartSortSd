"""Centralize Smart Sort training sources in ONE raw_data directory.

  python setup_data.py                        # reprocess ALL registered sources
  python setup_data.py "New Dataset Folder"   # import/move new source and reprocess ALL
  python setup_data.py "data_6class" --license MIT --allow-non-cc0

A successful import MOVES the original source directory into raw_data/sources
(on the same drive). Nothing is permanently deleted: even unmapped HITL JSON
and images remain intact. Prepared images live in raw_data/train and raw_data/val.
The trained-model code reads these two folders directly.

Source license information is declarative, not independently verified. MIT
or unverified sources require the explicit --allow-non-cc0 opt-in. The HITL
Recycling/Litterati folder is known as CC0 based on its publisher's declaration.
"""
import argparse
import csv
import hashlib
import io
import os
import json
import random
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageOps, ImageChops, ImageStat

CLASSES = ('cardboard', 'celluloseacetate', 'glass', 'metal', 'paper', 'plastic',
           'polystyrene', 'rubber', 'tetrapak', 'textile', 'trash')
# Match the image's actual suffix; preserve it in the annotation lookup (e.g. item.PNG.json).
# These are supported PIL input types, not output formats (prepared crops are saved as JPEG).
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.jfif', '.webp', '.bmp', '.tif', '.tiff', '.gif'}
HITL_URL = 'https://humansintheloop.org/resources/datasets/recycling-dataset/'
# Deliberately conservative: "mixed", "other", composite packaging etc are ambiguous.
MATERIAL_MAP = {
    'cardboard': 'cardboard', 'corrugated cardboard': 'cardboard',
    'glass': 'glass', 'glass bottle': 'glass',
    'metal': 'metal', 'aluminium': 'metal', 'aluminum': 'metal', 'steel': 'metal', 'tin': 'metal',
    'paper': 'paper',
    'plastic': 'plastic', 'pet': 'plastic', 'hdpe': 'plastic',
    'trash': 'trash', 'general waste': 'trash', 'non recyclable': 'trash', 'non-recyclable': 'trash',
    'celluloseacetate': 'celluloseacetate', 'cellulose acetate': 'celluloseacetate',
    'polystyrene': 'polystyrene', 'expanded polystyrene': 'polystyrene',
    'styrofoam': 'polystyrene', 'polystryene': 'polystyrene',
    'rubber': 'rubber', 'tetrapak': 'tetrapak', 'tetra pak': 'tetrapak',
    'textile': 'textile', 'textiles': 'textile',
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
            material = (tags_to_material(obj.get('tags', []), lookup)
                        or tags_to_material(obj.get('attributes', {}), lookup)
                        or normalize_label(obj.get('classTitle')))
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
    """Infer whether a directory is an annotation export or labeled class folders."""
    if next(folder.rglob('ann/*.json'), None) is not None:
        return 'supervisely'
    if any((folder / name).is_dir() for name in ('train', 'val')):
        return 'class_folders'
    if any(p.is_dir() and normalize_label(p.name) for p in folder.iterdir()):
        return 'class_folders'
    raise ValueError(f'Unsupported dataset structure in {folder}. Expected ds0/ann + '
                     'ds0/img, train/val/<material>, or <material> folders.')


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


def prepare(raw_data=Path('raw_data'), add=None, license_name=None,
            allow_non_cc0=False, seed=42, val_fraction=0.20):
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
                    'kind': kind,
                    'non_cc0_opt_in': bool(allow_non_cc0 and lic not in
                                            ('CC0', 'CC0-1.0', 'PUBLIC-DOMAIN'))}
    source_infos = registered + ([upcoming] if upcoming else [])
    if not source_infos:
        raise ValueError('No sources in raw_data/sources yet. Import your original '
                         'and HITL folders one time, for example:\n'
                         '  python setup_data.py ".\\data_6class" --license MIT --allow-non-cc0\n'
                         '  python setup_data.py ".\\Litterati Collection and Labeling Data" --license CC0')
    counters = Counter()
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
            class_counts[label] += 1
            for j, part in enumerate(parts):
                hash_buckets[(j, part)].add(digest)

        for info in source_infos:
            path = info['path']
            print(f'Reading {path} ({info["kind"]}, {info["license"]})', flush=True)
            if info['kind'] == 'supervisely':
                for img_path, material, rect, annotation in hitl_records(path, counters):
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
        by_label_origins = defaultdict(set)
        for rec in records.values():
            by_label_origins[rec['label']].add(rec['origin'])
        too_few = {c: {'images': class_counts[c], 'origins': len(by_label_origins[c])}
                   for c in CLASSES if class_counts[c] and
                   (class_counts[c] < 2 or len(by_label_origins[c]) < 2)}
        for digest, rec in list(records.items()):
            if rec['label'] in too_few:
                rec['path'].unlink()
                records.pop(digest)
                origin_records[rec['origin']].remove(digest)
                class_counts[rec['label']] -= 1
        classes = sorted(c for c in CLASSES if class_counts[c] >= 2)
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
             'kind': x['kind'], 'non_cc0_opt_in': x['non_cc0_opt_in']}
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
            'dedup_and_skipped': dict(counters),
            'split_counts': {'train': dict(class_counts - val_count),
                             'val': dict(val_count)},
            'records': file_records,
        }
        (stage / 'source_manifest.json').write_text(json.dumps(manifest, indent=2),
                                                    encoding='utf-8')

        if upcoming:
            # An atomic SAME-DRIVE rename moves the entire raw archive into raw_data.
            # Crucially, this also preserves annotation JSONs/materials we skipped.
            source_destination = sources_dir / upcoming['folder']
            os.rename(add, source_destination)
            print(f'Moved source into {source_destination}', flush=True)
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
                seed=args.seed, val_fraction=args.val_fraction)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    main()
