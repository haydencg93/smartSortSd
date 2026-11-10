"""Supabase data-access layer. The only file that talks to Supabase.

Used by server.py (writes) and app/dashboard.py (reads). Keys are loaded from the
project root .env file and never hard-coded.
"""

import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from supabase import Client, create_client

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

BUCKET = os.environ.get("SUPABASE_BUCKET", "captures")


@lru_cache(maxsize=1)
def client() -> Client:
    """Create and cache a shared Supabase client on first use."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SECRET_KEY")

    if not url or not key:
        raise RuntimeError("Set SUPABASE_URL and SUPABASE_SECRET_KEY in .env")

    return create_client(url, key)


class SupabaseRepo:
    """Everything the server needs from the database. Tests swap in a fake."""

    def log_scan(self, scan, status: str, direction: str | None) -> int | None:
        """Log a barcode scan and return the new transaction id.

        Returns None when the item is unsorted.
        """
        # The database function log_scan() returns a single row with a single
        # column: the transaction id.
        result = client().rpc(
            "log_scan",
            {
                "p_barcode": scan.barcode,
                "p_item_name": scan.item_name,
                "p_material": scan.material,
                "p_category": scan.category,
                "p_latency_ms": scan.latency_ms,
                "p_status": status,
                "p_bin_direction": direction,
            },
        ).execute()
        return result.data

    def save_image(self, data: bytes) -> str:
        """Upload a JPEG to storage and index it in `captures`. Returns its path."""
        now = datetime.now(timezone.utc)
        path = f"{now:%Y/%m/%d}/{now:%H%M%S-%f}.jpg"

        storage = client().storage
        storage.from_(BUCKET).upload(
            path,
            data,
            {"content-type": "image/jpeg"},
        )

        client().table("captures").insert(
            {"storage_path": path, "size_bytes": len(data)}
        ).execute()
        return path


def fetch_transactions(page_size: int = 1000) -> list[dict]:
    """Fetch all transactions, oldest first.

    Supabase caps each request at 1000 rows, so we page through until the result
    set is exhausted.
    """
    rows: list[dict] = []
    start = 0

    while True:
        batch = (
            client()
            .table("transactions")
            .select("*")
            .order("id")
            .range(start, start + page_size - 1)
            .execute()
            .data
        )
        rows.extend(batch)

        if len(batch) < page_size:
            return rows

        start += page_size


def signed_url(path: str, seconds: int = 3600) -> str:
    """Return a temporary signed URL for a private image in the dashboard."""
    return client().storage.from_(BUCKET).create_signed_url(path, seconds)["signedURL"]