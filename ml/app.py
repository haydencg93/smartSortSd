"""Smart Sort MobileNetV2 API with labels read from the trained checkpoint metadata.

Retains the exact existing GET /health and POST /classify JSON formats.
No UI, camera-client, or test changes. Material-only routing is approximate;
object type and cleanliness are NOT determined by the material classifier.
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
# Prefer the newest trained model, but retain legacy six-class compatibility.
DEFAULT_MODEL = ('smart_sort_mobilenetv2_combined.pth'
                 if Path('smart_sort_mobilenetv2_combined.pth').is_file()
                 else 'smart_sort_mobilenetv2_6class.pth')
MODEL_PATH = os.getenv('SMARTSORT_MODEL_PATH', DEFAULT_MODEL)
MATERIAL_CLASSES = OLD_CLASSES.copy()

# Basic University of Iowa campus bin guidance (NOT Iowa City curbside rules):
# https://sustainability.uiowa.edu/what-bin
# https://sustainability.uiowa.edu/sites/sustainability.uiowa.edu/files/2025-02/recycle-poster-8.5x11-original-pics.pdf
# Clean, dry, empty cardboard, metal and plastic CONTAINERS, cartons and paper
# are accepted. Glass and Styrofoam do NOT belong in campus recycling bins.
# For uncertain materials, landfill is a conservative regular-bin fallback,
# NOT a claim that every object of that material is safe for landfill.
# Rubber TIRES, electronics, batteries, etc require separate handling.
# Reusable textile items should preferably be donated/reused, not landfilled.
UIOWA_RULES = {
    'cardboard': 'recycling',
    'celluloseacetate': 'landfill',
    'glass': 'landfill',
    'metal': 'recycling',
    'paper': 'recycling',
    'plastic': 'recycling',
    'polystyrene': 'landfill',
    'rubber': 'landfill',
    'tetrapak': 'recycling',       # Cartons when clean, dry, empty
    'textile': 'landfill',        # Reuse/donation is preferable, not regular recycling
    'trash': 'landfill',
}
POLICY_URL = 'https://sustainability.uiowa.edu/what-bin'
preprocess = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])
app = FastAPI(title='Smart Sort Classification API', version='1.2.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
model = None


class ClassificationResponse(BaseModel):
    classification: str
    confidence: float
    material: str
    latency_ms: float


def read_class_order(checkpoint_path, state_dict):
    sidecar = checkpoint_path.with_suffix('.classes.json')
    num_outputs = state_dict['classifier.1.weight'].shape[0]
    if sidecar.is_file():
        data = json.loads(sidecar.read_text(encoding='utf-8'))
        classes = data['classes'] if isinstance(data, dict) else data
        if not isinstance(classes, list) or len(classes) != num_outputs or len(set(classes)) != len(classes):
            raise ValueError(f'Invalid class metadata in {sidecar}')
        if any(label not in UIOWA_RULES for label in classes):
            raise ValueError(f'No University of Iowa routing rule for: {set(classes) - set(UIOWA_RULES)}')
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
    logger.info('Loading MobileNetV2 from %s', checkpoint)
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
    return ClassificationResponse(classification=UIOWA_RULES[material],
                                  confidence=round(float(confidence), 4), material=material,
                                  latency_ms=round((time.perf_counter()-started)*1000, 1))
