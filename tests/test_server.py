import json
import sqlite3
import tempfile
import threading
import unittest
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from server import ScanRequestHandler, initialize_database


class ScanServerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "test.sqlite3")
        initialize_database(self.db_path)
        handler = partial(ScanRequestHandler, db_path=self.db_path)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        self.temp_dir.cleanup()

    def post_scan(self, payload):
        request = Request(
            f"{self.base_url}/api/scan",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            return response.status, json.loads(response.read())

    def test_classified_scan_is_available_to_dashboard(self):
        status, response = self.post_scan(
            {
                "barcode": "012345678905",
                "item_name": "Bottle",
                "material": "plastic",
                "category": "recycling",
                "latency_ms": 120.5,
            }
        )

        self.assertEqual(status, 201)
        self.assertEqual(response["status"], "success")
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT input_method, category, status, barcode, item_name, material "
                "FROM transactions"
            ).fetchone()
        self.assertEqual(
            row,
            ("barcode", "recycling", "success", "012345678905", "Bottle", "plastic"),
        )

    def test_unclassified_scan_is_logged_without_a_disposal_transaction(self):
        status, response = self.post_scan(
            {"barcode": "12345", "material": "unknown", "category": None}
        )

        self.assertEqual(status, 201)
        self.assertEqual(response["status"], "retry")
        self.assertIsNone(response["transaction_id"])
        with sqlite3.connect(self.db_path) as connection:
            event_count = connection.execute("SELECT COUNT(*) FROM scanner_events").fetchone()[0]
            transaction_count = connection.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        self.assertEqual(event_count, 1)
        self.assertEqual(transaction_count, 0)

    def test_missing_barcode_is_rejected(self):
        request = Request(
            f"{self.base_url}/api/scan",
            data=b'{"category":"trash"}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as error:
            urlopen(request)
        self.assertEqual(error.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
