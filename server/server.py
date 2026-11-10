"""Smart Sort server.

Receives results from the scanners (PC barcode prototype, later the ESP32
camera), saves them in Supabase, and tells the sorter which way to send the
item: recycling -> "left", trash -> "right".

Endpoints
---------
GET  /health     Quick "is the server up?" check.
POST /api/scan   JSON from the barcode scanner after it has looked the item up.
POST /api/image  Raw JPEG bytes from the camera. Saved to Supabase Storage.

Run from the project root (smartSortSd/):
    uvicorn server.server:app --host 0.0.0.0 --port 8000
Docs:  http://<this-pc>:8000/docs
"""

import logging
from typing import Literal

from fastapi import Depends, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from server.db import SupabaseRepo

log = logging.getLogger("smartsort")

MAX_IMAGE_BYTES = 2_000_000      # reject photos larger than ~2 MB

# Which way the sorter moves each category. Change it here if the bins swap.
BIN_DIRECTION = {"recycling": "left", "trash": "right"}

app = FastAPI(title="Smart Sort")


def get_repo() -> SupabaseRepo:
    """Dependency so tests can replace the real database with a fake one."""
    return SupabaseRepo()


# ---------------------------------------------------------------- input shape


# Scan is the JSON the barcode scanner sends to /api/scan. It is validated by Pydantic.
class Scan(BaseModel):
    """The JSON the barcode scanner sends to /api/scan."""

    barcode: str = Field(min_length=1, max_length=128)
    item_name: str | None = Field(default=None, max_length=512)
    material: str | None = Field(default=None, max_length=128)
    category: Literal["recycling", "trash"] | None = None   # None = couldn't decide
    latency_ms: float = Field(default=0, ge=0)

    @field_validator("barcode", "item_name", "material", "category", mode="before")

    
    @classmethod
    def clean(cls, value, info):
        """Trim spaces, '' -> None, lowercase material/category."""
        if not isinstance(value, str):
            return value
        value = value.strip()
        if info.field_name in ("material", "category"):
            value = value.lower()
        if info.field_name == "barcode":
            return value  # an empty barcode is rejected by min_length
        return value or None


def error(status: int, message: str) -> JSONResponse:
    """Every error from this server looks like {"error": "message"}."""
    return JSONResponse(status_code=status, content={"error": message})


@app.exception_handler(RequestValidationError)
async def bad_input(_: Request, exc: RequestValidationError):
    first = exc.errors()[0]
    field = ".".join(str(part) for part in first["loc"] if part != "body")
    return error(400, f"{field}: {first['msg']}")


# ---------------------------------------------------------------- routes
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/scan", status_code=201)
def receive_scan(scan: Scan, repo=Depends(get_repo)):
    """Barcode already looked up by the scanner: log it and say which way to sort."""
    status = "success" if scan.category else "retry"
    direction = BIN_DIRECTION.get(scan.category)  # None if there's no category

    try:
        transaction_id = repo.log_scan(scan, status, direction)
    except Exception:
        log.exception("Supabase write failed")
        return error(503, "database unavailable")

    return {"accepted": True, "status": status,
            "transaction_id": transaction_id, "bin_direction": direction}


@app.post("/api/image", status_code=201)
async def receive_image(request: Request, repo=Depends(get_repo)):
    """Camera sends raw JPEG bytes (Content-Type: image/jpeg)."""
    if request.headers.get("content-type", "").split(";")[0].strip() != "image/jpeg":
        return error(415, "content type must be image/jpeg")

    image = await request.body()
    if not image or len(image) > MAX_IMAGE_BYTES:
        return error(413, "image is empty or too large")

    try:
        path = await run_in_threadpool(repo.save_image, image)  # blocking upload
    except Exception:
        log.exception("Supabase upload failed")
        return error(503, "storage unavailable")

    # TODO: classify the image here, then add a row to `transactions`
    # (with image_path=path) and fill predicted_category/confidence in `captures`.
    return {"accepted": True, "image_path": path}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)