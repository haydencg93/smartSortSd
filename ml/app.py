"""Smart Sort MobileNetV2 API with configurable location-aware disposal rules.

GET /health and POST /classify retain their existing JSON response schemas.
Predicted class describes what the MODEL sees, not a guarantee an item is clean,
empty, safe to discard, or accepted by a campus collection program.

Location rules live in the *only* added repository file: rules/uiowa.json.
Additional locations can later be added under its ``locations`` key. The model
classifier is independent of the location-specific disposal decision.
"""
import io
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger('smart-sort-api')
OLD_CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = ('smart_sort_mobilenetv2_combined.pth'
                 if Path('smart_sort_mobilenetv2_combined.pth').is_file()
                 else 'smart_sort_mobilenetv2_6class.pth')
MODEL_PATH = os.getenv('SMARTSORT_MODEL_PATH', DEFAULT_MODEL)
RULES_PATH = Path(os.getenv('SMARTSORT_RULES_PATH', str(HERE / 'rules' / 'uiowa.json')))
MATERIAL_CLASSES = OLD_CLASSES.copy()
VALID_DISPOSALS = {'recycling', 'landfill', 'special_disposal', 'manual_review'}


def load_location_rules(path=RULES_PATH, location=None):
    """Reject malformed and missing configurations; never silently fall back."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('schema_version') != 1 or not isinstance(data.get('locations'), dict):
        raise ValueError(f'Invalid rules schema: {path}')
    chosen = location or os.getenv('SMARTSORT_LOCATION') or data.get('default_location')
    locations = data['locations']
    if chosen not in locations:
        raise ValueError(f'Unknown location {chosen!r}; supported locations: {sorted(locations)}')
    entries = locations[chosen].get('rules')
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f'No disposal rules for {chosen!r}')
    result = {}
    for label, entry in entries.items():
        if not isinstance(label, str) or not label or not isinstance(entry, dict):
            raise ValueError(f'Invalid rule entry: {label!r}')
        disposition = entry.get('disposal')
        if disposition not in VALID_DISPOSALS:
            raise ValueError(f'Invalid disposal {disposition!r} for {label!r}')
        result[label] = disposition
    return chosen, result


LOCATION, UIOWA_RULES = load_location_rules()
preprocess = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])
app = FastAPI(title='Smart Sort Classification API', version='1.3.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
model = None


class ClassificationResponse(BaseModel):
    classification: str
    confidence: float
    material: str
    latency_ms: float


def read_class_order(checkpoint_path, state_dict):
    sidecar = checkpoint_path.with_suffix('.classes.json')
    if 'classifier.1.weight' not in state_dict:
        raise ValueError('Checkpoint missing MobileNetV2 classifier.1.weight')
    num_outputs = state_dict['classifier.1.weight'].shape[0]
    if sidecar.is_file():
        data = json.loads(sidecar.read_text(encoding='utf-8'))
        classes = data.get('classes') if isinstance(data, dict) else data
        if (not isinstance(classes, list) or len(classes) != num_outputs
                or len(set(classes)) != len(classes)
                or not all(isinstance(label, str) and label for label in classes)):
            raise ValueError(f'Invalid class metadata in {sidecar}')
        missing = set(classes) - set(UIOWA_RULES)
        if missing:
            raise ValueError(f'No {LOCATION} routing rules for {sorted(missing)}')
        return classes
    if num_outputs == len(OLD_CLASSES):
        logger.warning('No classes.json found; assuming legacy six-class order for %s', checkpoint_path)
        return OLD_CLASSES.copy()
    raise ValueError(f'Expanded checkpoint needs label metadata: {sidecar}. '
                     'Keep the .classes.json alongside the model .pth file.')


@app.on_event('startup')
def load_model():
    global model, MATERIAL_CLASSES
    checkpoint = Path(MODEL_PATH)
    logger.info('Loading MobileNetV2 from %s for location %s', checkpoint, LOCATION)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    classes = read_class_order(checkpoint, state)
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, len(classes))
    m.load_state_dict(state, strict=True)
    m.eval()
    MATERIAL_CLASSES = classes
    model = m
    logger.info('Loaded %d classes: %s', len(classes), classes)


@app.get('/health')
def health_check():
    return {'status': 'ok', 'model_loaded': model is not None}


@app.post('/classify', response_model=ClassificationResponse)
async def classify(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded yet.')
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='Uploaded file must be an image.')
    started = time.perf_counter()
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(await file.read()))).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail='Could not decode image.')
    x = preprocess(image).unsqueeze(0)
    with torch.inference_mode():
        probabilities = F.softmax(model(x), dim=1)[0]
        confidence, idx = probabilities.max(dim=0)
    material = MATERIAL_CLASSES[int(idx)]
    disposition = UIOWA_RULES[material]
    # Optional abstention. Disabled by default to preserve existing API behavior;
    # threshold must be calibrated on held-out camera photographs before enabling.
    threshold = float(os.getenv('SMARTSORT_MIN_CONFIDENCE', '0'))
    if not 0 <= threshold <= 1:
        raise RuntimeError('SMARTSORT_MIN_CONFIDENCE must be between 0 and 1')
    if float(confidence) < threshold:
        disposition = 'manual_review'
    return ClassificationResponse(classification=disposition,
                                  confidence=round(float(confidence), 4), material=material,
                                  latency_ms=round((time.perf_counter()-started)*1000, 1))
