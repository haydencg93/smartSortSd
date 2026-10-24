"""Shared six-class inference and dataset adapters. Original project files remain untouched."""
from __future__ import annotations
import hashlib
import io
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms, models
from torch import nn

CLASSES = ('cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash')
RULES = {'cardboard': 'recycling', 'glass': 'trash', 'metal': 'recycling',
         'paper': 'recycling', 'plastic': 'recycling', 'trash': 'trash'}
REALWASTE_NAMES = {'Cardboard':'cardboard', 'Glass':'glass', 'Metal':'metal',
                   'Paper':'paper', 'Plastic':'plastic', 'Miscellaneous Trash':'trash'}
EXCLUDED_REALWASTE = {'Food Organics','Textile Trash','Vegetation'}
IMG_EXT = {'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}
PREPROCESS = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])

@dataclass(frozen=True)
class Sample:
    path: Path
    label: str
    relative_path: str


def load_checkpoint(path: Path, device: str='cpu') -> torch.nn.Module:
    """Only load trusted local weights; enforce exact six-class architecture."""
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
    weights = torch.load(str(path), map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    return model.to(device).eval()


def predict(model, img: Image.Image, device: str='cpu'):
    with torch.inference_mode():
        logits = model(PREPROCESS(ImageOps.exif_transpose(img).convert('RGB')).unsqueeze(0).to(device))
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    idx = int(probabilities.argmax())
    return {'material': CLASSES[idx], 'classification': RULES[CLASSES[idx]],
            'confidence': float(probabilities[idx]), 'probabilities': probabilities.tolist()}


def scan_dataset(root: Path, source: str) -> tuple[list[Sample], list[str]]:
    """Recognize a six-class ImageFolder or UCI RealWaste's nine folders; never guess labels."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset directory does not exist: {root}')
    mapping = dict(REALWASTE_NAMES) if source == 'realwaste' else {c:c for c in CLASSES}
    if source not in {'realwaste','imagefolder'}:
        raise ValueError('source must be realwaste or imagefolder')
    # Handle RealWaste.zip unpacked either as /RealWaste or /realwaste-main/RealWaste.
    roots = [root] + [p for p in root.rglob('*') if p.is_dir() and len(p.relative_to(root).parts) <= 3]
    hits = [p for p in roots if len({x.name for x in p.iterdir() if x.is_dir()} & set(mapping)) >= 1]
    if not hits:
        raise ValueError(f'Could not locate labeled folders for {source} below {root}')
    hits.sort(key=lambda p: (-len({x.name for x in p.iterdir() if x.is_dir()} & set(mapping)), len(p.relative_to(root).parts), str(p)))
    base = hits[0]
    samples = []
    missing=[]
    for source_label, standard_label in mapping.items():
        class_dir = base / source_label
        if not class_dir.is_dir():
            missing.append(standard_label)
            continue
        for path in sorted(class_dir.rglob('*')):
            if path.is_file() and path.suffix.lower() in IMG_EXT:
                samples.append(Sample(path, standard_label, str(path.relative_to(base))))
    if not samples:
        raise ValueError('No supported labeled images found')
    return samples, missing


def image_sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def duplicates_vs_training(samples: Iterable[Sample], train_root: Path) -> list[str]:
    """Exact file-duplicate identification. Doesn't imply near-duplicate detection."""
    originals = {image_sha256(p) for p in train_root.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXT}
    return [s.relative_path for s in samples if image_sha256(s.path) in originals]
