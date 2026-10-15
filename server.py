"""Smart Sort server.

Purpose
------------
Receives results from the scanners (the PC barcode prototype and, later, the
ESP32 camera), saves them in a SQLite database, and tells the sorter which
way to send the item: recycling -> "left", trash -> "right".

Endpoints
---------
GET  /health     Quick "is the server up?" check.
POST /api/scan   JSON from the barcode scanner after it has looked the item up.
POST /api/image  Raw JPEG bytes from the camera. Saved to the uploads/ folder.

Run:   python server.py                 (needs: pip install fastapi uvicorn)
Docs:  http://<this-pc>:8000/docs 
"""

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------- settings
DB_PATH = "waste_sorter.db"      # SQLite file, created automatically
IMAGE_DIR = Path("uploads")      # where camera photos are saved
MAX_IMAGE_BYTES = 2_000_000      # reject photos larger than ~2 MB

# Which way the sorter moves each category. Change it here if the bins swap.
BIN_DIRECTION = {"recycling": "left", "trash": "right"}

app = FastAPI(title="Smart Sort")


# ---------------------------------------------------------------- database
def now() -> str:
    """Current UTC time as text, e.g. 2026-10-15T18:49:15+00:00."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    """Open the database. We open a fresh connection per request (simple and safe)."""
    return sqlite3.connect(DB_PATH, timeout=10)


def create_tables() -> None:
    """Create both tables if they don't exist yet (safe to run every start)."""
    # `closing` closes the connection afterwards; `with db` saves (commits) the changes.
    with closing(connect()) as db, db:
        # WAL mode: faster writes, and a dashboard can read while we write.
        db.execute("PRAGMA journal_mode=WAL")

        # `transactions` = one row per item that was actually sorted.
        # The dashboard reads this table.
        db.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                input_method TEXT NOT NULL,   -- 'barcode' or 'image'
                category TEXT NOT NULL,       -- 'recycling' or 'trash'
                confidence REAL,              -- only used for image scans
                latency_ms REAL NOT NULL,
                status TEXT NOT NULL,
                bin_direction TEXT,           -- 'left' or 'right'
                image_path TEXT,
                barcode TEXT,
                item_name TEXT,
                material TEXT
            )""")

        # `scanner_events` = log of EVERY barcode scan, including ones where
        # the item couldn't be identified (those have no category).
        db.execute("""
            CREATE TABLE IF NOT EXISTS scanner_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                barcode TEXT NOT NULL,
                item_name TEXT,
                material TEXT,
                category TEXT,
                latency_ms REAL NOT NULL,
                status TEXT NOT NULL
            )""")


# Runs once when the server starts.
create_tables()
IMAGE_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------- input shape
class Scan(BaseModel):
    """The JSON the barcode scanner sends to /api/scan.

    FastAPI checks every request against this. If something is wrong
    (missing barcode, negative latency, category not recycling/trash) the
    request is rejected with a 400 error before any of our code runs.
    """

    barcode: str = Field(min_length=1, max_length=128)          # required
    item_name: str | None = Field(default=None, max_length=512)
    material: str | None = Field(default=None, max_length=128)  # e.g. 'plastic', 'unknown'
    category: Literal["recycling", "trash"] | None = None       # !!! None = scanner couldn't decide
    latency_ms: float = Field(default=0, ge=0)                  # !!! how long the lookup took

    @field_validator("barcode", "item_name", "material", "category", mode="before")

    # Clean up text before
    @classmethod
    def clean(cls, value, info):
        """Tidy the text before it is checked: trim spaces, '' -> None, lowercase."""
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
    """Turn FastAPI's validation errors into our simple {"error": ...} format."""
    first = exc.errors()[0]
    field = ".".join(str(part) for part in first["loc"] if part != "body")
    return error(400, f"{field}: {first['msg']}")


# ---------------------------------------------------------------- routes
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/scan", status_code=201)
def receive_scan(scan: Scan):
    """Barcode already looked up by the scanner: log it and say which way to sort."""
    # No category means the scanner couldn't identify the material,
    # so we tell it to "retry" (the image camera is the fallback).
    status = "success" if scan.category else "retry"
    direction = BIN_DIRECTION.get(scan.category)  # None if there's no category
    transaction_id = None

    # Both inserts happen together: either both are saved or neither is.
    with closing(connect()) as db, db:

        # 1) Always log the scan, even failed lookups.
        db.execute(
            "INSERT INTO scanner_events "
            "(timestamp, barcode, item_name, material, category, latency_ms, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (now(), scan.barcode, scan.item_name, scan.material,
             scan.category, scan.latency_ms, status),
        )

        # 2) Only items that got sorted become a dashboard transaction.
        if scan.category:
            cursor = db.execute(
                "INSERT INTO transactions "
                "(timestamp, input_method, category, latency_ms, status, "
                " bin_direction, barcode, item_name, material) "
                "VALUES (?, 'barcode', ?, ?, ?, ?, ?, ?, ?)",
                (now(), scan.category, scan.latency_ms, status, direction,
                 scan.barcode, scan.item_name, scan.material),
            )
            transaction_id = cursor.lastrowid

    # The scanner reads "status" from this reply.
    return {"accepted": True, "status": status,
            "transaction_id": transaction_id, "bin_direction": direction}


# ---------------------------------------------------------------- image upload
@app.post("/api/image", status_code=201)

async def receive_image(request: Request):
    """Camera sends raw JPEG bytes (Content-Type: image/jpeg). Saved to uploads/."""
    # Only accept JPEG photos.
    if request.headers.get("content-type", "").split(";")[0].strip() != "image/jpeg":
        return error(415, "content type must be image/jpeg")

    # The request body IS the photo.
    image = await request.body()
    if not image or len(image) > MAX_IMAGE_BYTES:
        return error(413, "image is empty or too large")

    # Timestamped file name so photos never overwrite each other.
    path = IMAGE_DIR / f"{datetime.now():%Y%m%d-%H%M%S-%f}.jpg"
    path.write_bytes(image)

    # TODO: classify the image here, then add a row to `transactions`.
    
    return {"accepted": True, "image_path": str(path)}

# 
# Start the server when you run `python server.py`.
# 0.0.0.0 = reachable from other devices on your network (like the ESP32).
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)