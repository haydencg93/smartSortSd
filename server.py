"""Local receiver for Smart Sort barcode scans.

Run with ``python server.py``. The scanner posts JSON to ``/api/scan`` and
the server stores classified transactions in ``waste_sorter.db``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sqlite3
from datetime import datetime, timezone
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit


MAX_REQUEST_BYTES = 16_384
CATEGORY_DIRECTIONS = {"recycling": "left", "trash": "right"}
DEFAULT_DB_PATH = "waste_sorter.db"


def initialize_database(db_path: str) -> None:
    """Create the scan log and dashboard-compatible transaction table."""
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                input_method TEXT NOT NULL,
                category TEXT NOT NULL,
                confidence REAL,
                latency_ms REAL NOT NULL,
                status TEXT NOT NULL,
                bin_direction TEXT,
                image_path TEXT,
                barcode TEXT,
                item_name TEXT,
                material TEXT
            )
            """
        )
        existing_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(transactions)")
        }
        for column in ("barcode", "item_name", "material"):
            if column not in existing_columns:
                connection.execute(f"ALTER TABLE transactions ADD COLUMN {column} TEXT")

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scanner_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                barcode TEXT NOT NULL,
                item_name TEXT,
                material TEXT,
                category TEXT,
                latency_ms REAL NOT NULL,
                status TEXT NOT NULL
            )
            """
        )


def _optional_text(payload: dict[str, Any], field: str, max_length: int) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    value = value.strip()
    if len(value) > max_length:
        raise ValueError(f"{field} is too long")
    return value or None


def record_scan(db_path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and persist one completed scanner lookup."""
    barcode = _optional_text(payload, "barcode", 128)
    if not barcode:
        raise ValueError("barcode is required")

    item_name = _optional_text(payload, "item_name", 512)
    material = _optional_text(payload, "material", 128)
    if material:
        material = material.lower()

    category = _optional_text(payload, "category", 32)
    if category:
        category = category.lower()
        if category not in CATEGORY_DIRECTIONS:
            raise ValueError("category must be 'recycling', 'trash', or null")

    latency = payload.get("latency_ms", 0)
    if isinstance(latency, bool) or not isinstance(latency, (int, float)):
        raise ValueError("latency_ms must be a non-negative number")
    latency_ms = float(latency)
    if not math.isfinite(latency_ms) or latency_ms < 0:
        raise ValueError("latency_ms must be a non-negative number")

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    status = "success" if category else "retry"
    transaction_id = None

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO scanner_events
                (timestamp, barcode, item_name, material, category, latency_ms, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, barcode, item_name, material, category, latency_ms, status),
        )
        if category:
            cursor = connection.execute(
                """
                INSERT INTO transactions
                    (timestamp, input_method, category, confidence, latency_ms,
                     status, bin_direction, image_path, barcode, item_name, material)
                VALUES (?, 'barcode', ?, NULL, ?, ?, ?, NULL, ?, ?, ?)
                """,
                (
                    timestamp,
                    category,
                    latency_ms,
                    status,
                    CATEGORY_DIRECTIONS[category],
                    barcode,
                    item_name,
                    material,
                ),
            )
            transaction_id = cursor.lastrowid

    return {
        "accepted": True,
        "status": status,
        "transaction_id": transaction_id,
    }


class ScanRequestHandler(BaseHTTPRequestHandler):
    db_path = DEFAULT_DB_PATH

    def __init__(self, request, client_address, server, db_path=DEFAULT_DB_PATH):
        self.db_path = db_path
        super().__init__(request, client_address, server)

    def _send_json(self, status_code: int, response: dict[str, Any]) -> None:
        body = json.dumps(response).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if urlsplit(self.path).path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/scan":
            self._send_json(404, {"error": "not found"})
            return
        if self.headers.get_content_type() != "application/json":
            self._send_json(415, {"error": "content type must be application/json"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._send_json(400, {"error": "a valid Content-Length header is required"})
            return
        if content_length < 1 or content_length > MAX_REQUEST_BYTES:
            self._send_json(413, {"error": "request body size is invalid"})
            return

        try:
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            result = record_scan(self.db_path, payload)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            self._send_json(400, {"error": str(error)})
            return
        except sqlite3.Error:
            logging.exception("Could not save scanner request")
            self._send_json(500, {"error": "could not save scan"})
            return

        self._send_json(201, result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Smart Sort scan receiver.")
    parser.add_argument(
        "--host",
        default=os.environ.get("SMARTSORT_HOST", "0.0.0.0"),
        help="interface to listen on (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SMARTSORT_PORT", "8000")),
        help="port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--db",
        default=os.environ.get("SMARTSORT_DB", DEFAULT_DB_PATH),
        help="SQLite database path (default: waste_sorter.db)",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    initialize_database(args.db)
    handler = partial(ScanRequestHandler, db_path=args.db)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Smart Sort receiver listening on %s:%s (database: %s)", args.host, args.port, args.db)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Stopping Smart Sort receiver")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
