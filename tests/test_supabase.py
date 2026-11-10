"""Live Supabase connection tests.

These talk to your REAL Supabase project (the one in .env). Every row and file
they create is tagged "TEST-..." and deleted again when each test finishes.
Because they touch the real database, they only run when you ask for them:

    RUN_LIVE=1 python -m unittest tests.test_supabase -v

Optional (enables the "public key is locked out" checks): add the publishable
key to .env as  SUPABASE_PUBLISHABLE_KEY=sb_publishable_...
"""

import os
import unittest
import uuid

import httpx
from supabase import create_client
from fastapi.testclient import TestClient

from server import db
from server.server import Scan, app

LIVE = os.environ.get("RUN_LIVE") == "1"
HAVE_KEYS = bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SECRET_KEY"))
PUBLISHABLE = os.environ.get("SUPABASE_PUBLISHABLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")

JPEG = b"\xff\xd8\xff\xe0smartsort-test-image\xff\xd9"   # tiny fake JPEG


@unittest.skipUnless(LIVE, "set RUN_LIVE=1 to run tests against the real Supabase project")
@unittest.skipUnless(HAVE_KEYS, "SUPABASE_URL / SUPABASE_SECRET_KEY missing from .env")
class SupabaseLiveTests(unittest.TestCase):
    def setUp(self):
        self.tag = f"TEST-{uuid.uuid4().hex[:12]}"   # unique marker for this test's rows
        self.paths = []                              # storage files to delete afterwards
        self.repo = db.SupabaseRepo()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        c = db.client()
        c.table("transactions").delete().eq("barcode", self.tag).execute()
        c.table("scanner_events").delete().eq("barcode", self.tag).execute()
        if self.paths:
            c.table("captures").delete().in_("storage_path", self.paths).execute()
            c.storage.from_(db.BUCKET).remove(self.paths)

    def rows(self, table, column="barcode", value=None):
        return db.client().table(table).select("*").eq(column, value or self.tag).execute().data

    # ------------------------------------------------------------ connection
    def test_all_tables_exist_and_are_reachable(self):
        for table in ("transactions", "scanner_events", "captures"):
            with self.subTest(table=table):
                result = db.client().table(table).select("id").limit(1).execute()
                self.assertIsInstance(result.data, list)

    def test_storage_bucket_exists(self):
        names = [b.name for b in db.client().storage.list_buckets()]
        self.assertIn(db.BUCKET, names)

    # ------------------------------------------------------------ scan logging
    def test_classified_scan_writes_event_and_transaction(self):
        scan = Scan(barcode=self.tag, item_name="Test bottle", material="plastic",
                    category="recycling", latency_ms=123.0)
        transaction_id = self.repo.log_scan(scan, "success", "left")

        self.assertIsInstance(transaction_id, int)
        events = self.rows("scanner_events")
        transactions = self.rows("transactions")
        self.assertEqual(len(events), 1)
        self.assertEqual(len(transactions), 1)
        row = transactions[0]
        self.assertEqual(row["id"], transaction_id)
        self.assertEqual((row["input_method"], row["category"], row["status"], row["bin_direction"]),
                         ("barcode", "recycling", "success", "left"))
        self.assertIsNotNone(row["timestamp"])          # stamped by the database

    def test_unclassified_scan_writes_event_only(self):
        scan = Scan(barcode=self.tag, material="unknown", category=None)
        transaction_id = self.repo.log_scan(scan, "retry", None)

        self.assertIsNone(transaction_id)
        self.assertEqual(len(self.rows("scanner_events")), 1)
        self.assertEqual(self.rows("transactions"), [])

    def test_database_rejects_invalid_category(self):
        """The CHECK constraint is the last line of defence behind server validation."""
        with self.assertRaises(Exception):
            db.client().table("transactions").insert({
                "input_method": "barcode", "category": "banana", "latency_ms": 1,
                "status": "success", "barcode": self.tag}).execute()

    def test_fetch_transactions_returns_new_row(self):
        scan = Scan(barcode=self.tag, category="trash", latency_ms=5)
        tid = self.repo.log_scan(scan, "success", "right")
        self.assertIn(tid, [r["id"] for r in db.fetch_transactions()])

    # ------------------------------------------------------------ images
    def test_image_upload_is_stored_indexed_and_readable(self):
        path = self.repo.save_image(JPEG)
        self.paths.append(path)

        self.assertTrue(path.endswith(".jpg"))
        rows = self.rows("captures", column="storage_path", value=path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["size_bytes"], len(JPEG))

        downloaded = db.client().storage.from_(db.BUCKET).download(path)
        self.assertEqual(downloaded, JPEG)

        url = db.signed_url(path, seconds=60)
        self.assertEqual(httpx.get(url).content, JPEG)

    def test_bucket_is_private(self):
        path = self.repo.save_image(JPEG)
        self.paths.append(path)
        public_url = f"{os.environ['SUPABASE_URL']}/storage/v1/object/public/{db.BUCKET}/{path}"
        self.assertNotEqual(httpx.get(public_url).status_code, 200)

    # ------------------------------------------------------------ whole chain
    def test_api_scan_end_to_end_through_real_database(self):
        """HTTP request -> FastAPI -> db.py -> Supabase (no fakes)."""
        response = TestClient(app).post("/api/scan", json={
            "barcode": self.tag, "item_name": "E2E item", "material": "glass",
            "category": "recycling", "latency_ms": 42})

        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertEqual((body["status"], body["bin_direction"]), ("success", "left"))
        self.assertEqual(self.rows("transactions")[0]["id"], body["transaction_id"])

    def test_api_image_end_to_end_through_real_storage(self):
        response = TestClient(app).post(
            "/api/image", content=JPEG, headers={"Content-Type": "image/jpeg"})

        self.assertEqual(response.status_code, 201)
        path = response.json()["image_path"]
        self.paths.append(path)
        self.assertEqual(db.client().storage.from_(db.BUCKET).download(path), JPEG)

    # ------------------------------------------------------------ security
    @unittest.skipUnless(PUBLISHABLE, "add SUPABASE_PUBLISHABLE_KEY to .env to run this")
    def test_public_key_cannot_read_or_call_anything(self):
        scan = Scan(barcode=self.tag, category="trash", latency_ms=1)
        self.repo.log_scan(scan, "success", "right")        # row exists (secret key)

        public = create_client(os.environ["SUPABASE_URL"], PUBLISHABLE)
        seen = public.table("transactions").select("*").eq("barcode", self.tag).execute().data
        self.assertEqual(seen, [])                          # RLS hides it

        with self.assertRaises(Exception):                  # execute revoked
            public.rpc("log_scan", {
                "p_barcode": self.tag, "p_item_name": None, "p_material": None,
                "p_category": "trash", "p_latency_ms": 1,
                "p_status": "success", "p_bin_direction": "right"}).execute()


if __name__ == "__main__":
    unittest.main()