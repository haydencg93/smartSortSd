"""[AI] Smart Sort barcode prototype unit/integration-isolated tests.

Place this file at ml/tests/tests/test_barcode_ai.py, with pc_proto_barcode.py
at ml/pc_proto_barcode.py. No live web, webcam or display is used.
All outcomes are UNVERIFIED until pytest has actually been run.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def barcode_module():
    path = Path(__file__).resolve().parents[1] / "pc_proto_barcode.py"
    if not path.exists():
        pytest.fail(f"Barcode implementation not found at {path}; check project layout.")
    spec = importlib.util.spec_from_file_location("smart_sort_barcode_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("description, expected", [
    ("PET plastic bottle", "plastic"),
    ("glass jar", "glass"),
    ("aluminum container", "metal"),
    ("corrugated cardboard", "cardboard"),
    ("paper wrapper", "paper"),
    ("unrecognized material", None),
    ("", None),
    (None, None),
])
def test_ai_material_keyword_matching(barcode_module, description, expected):
    assert barcode_module.determine_material(description) == expected


def test_ai_material_case_insensitive(barcode_module):
    assert barcode_module.determine_material("GLASS JAR") == "glass"


def test_ai_disposal_mapping_current_prototype(barcode_module):
    assert barcode_module.iowa_city_rules == {
        "cardboard": "Recycling", "metal": "Recycling", "paper": "Recycling",
        "plastic": "Recycling", "glass": "Trash", "trash": "Trash",
    }


def fake_response(status_code, data):
    return SimpleNamespace(status_code=status_code, json=lambda: data)


def test_ai_off_success_and_upc_padding(barcode_module, monkeypatch):
    get = Mock(return_value=fake_response(200, {"status": 1, "product": {
        "product_name": "Test Bottle", "packaging_tags": ["plastic"],
    }}))
    monkeypatch.setattr(barcode_module.requests, "get", get)
    assert barcode_module.lookup_barcode("123456789012") == ("Test Bottle", "plastic")
    assert "/0123456789012.json" in get.call_args.args[0]
    assert get.call_count == 1
    assert get.call_args.kwargs["timeout"] == 2.5
    assert "User-Agent" in get.call_args.kwargs["headers"]


def test_ai_ean13_not_padded(barcode_module, monkeypatch):
    get = Mock(return_value=fake_response(200, {"status": 1, "product": {
        "product_name": "Paper Package", "packaging": "paper",
    }}))
    monkeypatch.setattr(barcode_module.requests, "get", get)
    assert barcode_module.lookup_barcode("0123456789012") == ("Paper Package", "paper")
    assert "/0123456789012.json" in get.call_args.args[0]


def test_ai_fallback_to_upc_when_off_not_found(barcode_module, monkeypatch):
    get = Mock(side_effect=[
        fake_response(200, {"status": 0}),
        fake_response(200, {"code": "OK", "items": [{"title": "Metal Can", "description": "aluminum can"}]}),
    ])
    monkeypatch.setattr(barcode_module.requests, "get", get)
    assert barcode_module.lookup_barcode("123456789012") == ("Metal Can", "metal")
    assert get.call_count == 2
    assert "upc=123456789012" in get.call_args.args[0]


def test_ai_fallback_when_off_material_unknown(barcode_module, monkeypatch):
    get = Mock(side_effect=[
        fake_response(200, {"status": 1, "product": {"product_name": "Known Product"}}),
        fake_response(200, {"code": "OK", "items": [{"title": "Glass Jar", "description": "glass"}]}),
    ])
    monkeypatch.setattr(barcode_module.requests, "get", get)
    assert barcode_module.lookup_barcode("123456789012") == ("Glass Jar", "glass")


def test_ai_unknown_material_preserves_off_name(barcode_module, monkeypatch):
    get = Mock(side_effect=[
        fake_response(200, {"status": 1, "product": {"product_name": "Named Item"}}),
        fake_response(200, {"code": "NOT_FOUND", "items": []}),
    ])
    monkeypatch.setattr(barcode_module.requests, "get", get)
    assert barcode_module.lookup_barcode("123456789012") == ("Named Item", "unknown")


def test_ai_no_match_returns_none(barcode_module, monkeypatch):
    get = Mock(side_effect=[fake_response(404, {}), fake_response(404, {})])
    monkeypatch.setattr(barcode_module.requests, "get", get)
    assert barcode_module.lookup_barcode("123456789012") == (None, None)


def test_ai_off_network_failure_falls_back(barcode_module, monkeypatch):
    get = Mock(side_effect=[
        barcode_module.requests.exceptions.Timeout("offline"),
        fake_response(200, {"code": "OK", "items": [{"title": "Paper Item", "description": "paper"}]}),
    ])
    monkeypatch.setattr(barcode_module.requests, "get", get)
    assert barcode_module.lookup_barcode("123456789012") == ("Paper Item", "paper")


def test_ai_both_networks_fail_without_crash(barcode_module, monkeypatch):
    get = Mock(side_effect=barcode_module.requests.exceptions.ConnectionError("offline"))
    monkeypatch.setattr(barcode_module.requests, "get", get)
    assert barcode_module.lookup_barcode("123456789012") == (None, None)
    assert get.call_count == 2


def camera_stubs(barcode_module, monkeypatch, frames, decoded, keys, clock=None):
    cap = Mock()
    cap.read.side_effect = [(True, object()) for _ in range(frames)]
    monkeypatch.setattr(barcode_module.cv2, "VideoCapture", Mock(return_value=cap))
    monkeypatch.setattr(barcode_module, "decode", Mock(side_effect=decoded))
    monkeypatch.setattr(barcode_module.cv2, "rectangle", Mock())
    put_text = Mock()
    monkeypatch.setattr(barcode_module.cv2, "putText", put_text)
    monkeypatch.setattr(barcode_module.cv2, "imshow", Mock())
    monkeypatch.setattr(barcode_module.cv2, "waitKey", Mock(side_effect=keys))
    monkeypatch.setattr(barcode_module.cv2, "destroyAllWindows", Mock())
    if clock is not None:
        monkeypatch.setattr(barcode_module.time, "time", Mock(side_effect=clock))
    return cap, put_text


def barcode_payload(value="0123456789012"):
    return SimpleNamespace(rect=(10, 30, 70, 25), data=value.encode("utf-8"))


def test_ai_scanner_recognized_item_displays_routing(barcode_module, monkeypatch):
    cap, put_text = camera_stubs(barcode_module, monkeypatch, 1, [[barcode_payload()]], [ord("q")])
    lookup = Mock(return_value=("Bottle", "plastic"))
    monkeypatch.setattr(barcode_module, "lookup_barcode", lookup)
    barcode_module.main()
    lookup.assert_called_once_with("0123456789012")
    assert "RECYCLING" in put_text.call_args.args[1]
    cap.release.assert_called_once()


def test_ai_scanner_unknown_prompts_camera(barcode_module, monkeypatch):
    _, put_text = camera_stubs(barcode_module, monkeypatch, 1, [[barcode_payload()]], [ord("q")])
    monkeypatch.setattr(barcode_module, "lookup_barcode", Mock(return_value=(None, None)))
    barcode_module.main()
    assert "Place item for camera" in put_text.call_args.args[1]


def test_ai_scanner_debounces_repeat_within_four_seconds(barcode_module, monkeypatch):
    camera_stubs(barcode_module, monkeypatch, 2,
                 [[barcode_payload()], [barcode_payload()]], [0, ord("q")],
                 [100.0, 100.1, 100.2, 102.0])
    lookup = Mock(return_value=("Glass Bottle", "glass"))
    monkeypatch.setattr(barcode_module, "lookup_barcode", lookup)
    barcode_module.main()
    assert lookup.call_count == 1


def test_ai_scanner_rescans_after_four_seconds(barcode_module, monkeypatch):
    camera_stubs(barcode_module, monkeypatch, 2,
                 [[barcode_payload()], [barcode_payload()]], [0, ord("q")],
                 [100.0, 100.1, 100.2, 105.0, 105.1, 105.2])
    lookup = Mock(return_value=("Glass Bottle", "glass"))
    monkeypatch.setattr(barcode_module, "lookup_barcode", lookup)
    barcode_module.main()
    assert lookup.call_count == 2


def test_ai_scanner_releases_camera_if_frame_read_fails(barcode_module, monkeypatch):
    cap = Mock()
    cap.read.return_value = (False, None)
    monkeypatch.setattr(barcode_module.cv2, "VideoCapture", Mock(return_value=cap))
    destroy = Mock()
    monkeypatch.setattr(barcode_module.cv2, "destroyAllWindows", destroy)
    barcode_module.main()
    cap.release.assert_called_once()
    destroy.assert_called_once()
