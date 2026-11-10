import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from server.server import app, get_repo


class ScanServerTests(unittest.TestCase):
    def setUp(self):
        self.repo = Mock()
        self.repo.log_scan.return_value = 42
        app.dependency_overrides[get_repo] = lambda: self.repo
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_classified_scan_is_available_to_dashboard(self):
        response = self.client.post(
            "/api/scan",
            json={
                "barcode": "012345678905",
                "item_name": "Bottle",
                "material": "plastic",
                "category": "recycling",
                "latency_ms": 120.5,
            }
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["status"], "success")
        self.assertEqual(response.json()["transaction_id"], 42)
        self.assertEqual(response.json()["bin_direction"], "left")
        self.repo.log_scan.assert_called_once()
        scan, status, direction = self.repo.log_scan.call_args.args
        self.assertEqual(scan.barcode, "012345678905")
        self.assertEqual(scan.item_name, "Bottle")
        self.assertEqual(scan.material, "plastic")
        self.assertEqual(status, "success")
        self.assertEqual(direction, "left")

    def test_unclassified_scan_is_logged_without_a_disposal_transaction(self):
        self.repo.log_scan.return_value = None
        response = self.client.post(
            "/api/scan",
            json={"barcode": "12345", "material": "unknown", "category": None},
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["status"], "retry")
        self.assertIsNone(response.json()["transaction_id"])
        self.assertIsNone(response.json()["bin_direction"])
        self.repo.log_scan.assert_called_once()
        _, status, direction = self.repo.log_scan.call_args.args
        self.assertEqual(status, "retry")
        self.assertIsNone(direction)

    def test_missing_barcode_is_rejected(self):
        response = self.client.post(
            "/api/scan",
            json={"category": "trash"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("barcode", response.json()["error"])


if __name__ == "__main__":
    unittest.main()
