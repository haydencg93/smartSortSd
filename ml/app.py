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
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart-sort-api")

MODEL_PATH = "smart_sort_mobilenetv2_6class.pth"
MATERIAL_CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Iowa City disposal logic - glass is excluded from standard public recycling bins.
# Keep this in sync with iowa_city_rules in pc_proto_ml.py.
IOWA_CITY_RULES = {
    'cardboard': 'recycling',
    'metal': 'recycling',
    'paper': 'recycling',
    'plastic': 'recycling',
    'glass': 'trash',
    'trash': 'trash',
}

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

app = FastAPI(title="Smart Sort Classification API", version="1.0.0")

# For the prototype this is wide open. Once you know the exact origin(s)
# that will call this (dashboard domain, etc.) narrow allow_origins to that list.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


class ClassificationResponse(BaseModel):
    classification: str   # "recycling" or "trash" -- what the ESP32/LCD should display
    confidence: float      # 0.0 - 1.0
    material: str          # raw 6-class label, useful for logging/dashboard
    latency_ms: float      # server-side inference time only (not network time)


@app.on_event("startup")
def load_model():
    global model
    logger.info("Loading model weights from %s ...", MODEL_PATH)
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, len(MATERIAL_CLASSES))
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    m.load_state_dict(state_dict)
    m.eval()
    model = m
    logger.info("Model loaded and ready.")


@app.get("/health")
def health_check():
    """Cheap endpoint for uptime checks / load balancer health probes."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/classify", response_model=ClassificationResponse)
async def classify(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    start = time.time()

    raw_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probs, 0)

    material = MATERIAL_CLASSES[predicted_idx.item()]
    bin_decision = IOWA_CITY_RULES[material]
    latency_ms = (time.time() - start) * 1000

    logger.info(
        "material=%s confidence=%.3f bin=%s latency_ms=%.1f",
        material, confidence.item(), bin_decision, latency_ms,
    )

    return ClassificationResponse(
        classification=bin_decision,
        confidence=round(confidence.item(), 4),
        material=material,
        latency_ms=round(latency_ms, 1),
    )
