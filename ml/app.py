"""
Smart Sort Recycling Assistant - Cloud Inference API

Run locally for testing:
    python -m uvicorn app:app --host 0.0.0.0 --port 8000 --log-level debug

View docs:
    http://127.0.0.1:8000/docs

API JSON URL:
    http://127.0.0.1:8000/classify
"""
import io
import logging
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger('smart-sort-api')
MODEL_PATH = os.getenv('SMARTSORT_MODEL_PATH', 'smart_sort_mobilenetv2_6class.pth')
MATERIAL_CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
# Coarse material-only mapping for BASIC UIowa campus guidance.
# All assumed recyclable objects must actually be CLEAN, DRY, EMPTY and accepted.
# Material alone CANNOT determine disposal of plastic bags, foil, batteries etc.
UIOWA_RULES = {
    'cardboard': 'recycling',
    'metal': 'recycling',
    'paper': 'recycling',
    'plastic': 'recycling',
    'glass': 'trash',
    'trash': 'trash'
}
POLICY_URL = 'https://sustainability.uiowa.edu/what-bin'
preprocess = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])
app = FastAPI(title='Smart Sort Classification API', version='1.1.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
model = None


class ClassificationResponse(BaseModel):
    classification: str  # 'recycling' or 'trash' with this six-material model
    confidence: float
    material: str
    latency_ms: float


@app.on_event('startup')
def load_model():
    global model
    logger.info('Loading six-class MobileNetV2 from %s', MODEL_PATH)
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, len(MATERIAL_CLASSES))
    m.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True), strict=True)
    m.eval()
    model = m


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
