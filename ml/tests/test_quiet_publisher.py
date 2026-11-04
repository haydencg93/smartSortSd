"""Run with `python -m unittest discover -s tests -v` after extracting this ZIP.

Offline tests; no Kaggle login, GPU, or GitHub access required.
"""
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validate_model = load("quiet_validate", "ml/cloud/validate_model.py")
probe_model = load("quiet_probe", "ml/cloud/probe_kaggle_model.py")


class PublisherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from torchvision import models
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.m = cls.root / "model"
        cls.m.mkdir()
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(model.last_channel, len(validate_model.EXPECTED))
        torch.save(model.state_dict(), cls.m / "smart_sort_mobilenetv2_combined.pth")
        cls.digest = hashlib.sha256((cls.m / "smart_sort_mobilenetv2_combined.pth").read_bytes()).hexdigest()
        (cls.m / "smart_sort_mobilenetv2_combined.classes.json").write_text(json.dumps({"classes": list(validate_model.EXPECTED)}))
        (cls.m / "cloud_run.json").write_text(json.dumps({
            "model_sha256": cls.digest, "source_commit": "a" * 40,
            "dataset_ref": "haydengamble/smartsort-prepared"}))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def args(self):
        return SimpleNamespace(directory=str(self.m), previous=str(self.root / "missing.pth"),
                               min_accuracy=.85, min_macro_f1=.75,
                               min_glass_recall=.80, min_trash_recall=.80)

    def metrics(self, acc, f1, glass, trash):
        (self.m / "smart_sort_mobilenetv2_combined.metrics.json").write_text(json.dumps({
            "classes": list(validate_model.EXPECTED),
            "history": [{"epoch": 25, "val_accuracy": acc, "val_macro_f1": f1,
                         "val_recall_by_class": {"glass": glass, "trash": trash}}]}))

    def test_low_quality_is_successful_review_not_workflow_error(self):
        self.metrics(.772, .651, .743, .667)
        with tempfile.TemporaryDirectory() as d:
            with patch.dict(os.environ, {"GITHUB_OUTPUT": str(Path(d) / "out"),
                                             "GITHUB_STEP_SUMMARY": str(Path(d) / "summary")}):
                approved = validate_model.validate(self.args())
            self.assertFalse(approved)
            self.assertIn("approved=false", (Path(d) / "out").read_text())
            self.assertIn("REJECTED", (Path(d) / "summary").read_text())

    def test_high_quality_approved(self):
        self.metrics(.91, .88, .92, .90)
        with tempfile.TemporaryDirectory() as d:
            with patch.dict(os.environ, {"GITHUB_OUTPUT": str(Path(d) / "out")}):
                self.assertTrue(validate_model.validate(self.args()))
            self.assertIn("approved=true", (Path(d) / "out").read_text())

    def test_corrupt_model_still_errors_even_if_low_quality(self):
        self.metrics(.11, .12, .13, .14)
        bad_sha = json.loads((self.m / "cloud_run.json").read_text())
        bad_sha["model_sha256"] = "0" * 64
        (self.m / "cloud_run.json").write_text(json.dumps(bad_sha))
        try:
            with self.assertRaises(SystemExit):
                validate_model.validate(self.args())
        finally:
            bad_sha["model_sha256"] = self.digest
            (self.m / "cloud_run.json").write_text(json.dumps(bad_sha))

    def test_policy_and_model_identity_affect_review_key(self):
        f = self.m / "cloud_run.json"
        args = SimpleNamespace(provenance=str(f), kernel="haydengamble/smartsort-gpu-trainer",
                               previous=str(self.root / "missing.pth"),
                               min_accuracy=.85, min_macro_f1=.75, min_glass_recall=.80,
                               min_trash_recall=.80, auto_commit="false")
        base = probe_model.run(args)
        self.assertEqual(base, probe_model.run(args))
        args.min_accuracy = .90
        self.assertNotEqual(base, probe_model.run(args))
        args.min_accuracy = .85
        args.auto_commit = "true"
        self.assertNotEqual(base, probe_model.run(args))

    def test_workflow_gates_publishing_and_skips_repeat(self):
        import yaml
        with open(ROOT / ".github/workflows/smartsort-kaggle-publish.yml", encoding="utf-8") as f:
            workflow = yaml.load(f, Loader=yaml.BaseLoader)
        steps = workflow["jobs"]["publish"]["steps"]
        publication = [s for s in steps if s.get("name", "").startswith("Publish release")][0]
        self.assertIn("steps.verify.outputs.approved == 'true'", publication["if"])
        self.assertTrue(any(s.get("uses", "").startswith("actions/cache/restore") for s in steps))
        self.assertTrue(any(s.get("uses", "").startswith("actions/cache/save") for s in steps))
        self.assertTrue(any("Archive Kaggle model candidate" == s.get("name") for s in steps))
        self.assertIn("force_recheck", workflow["on"]["workflow_dispatch"]["inputs"])


if __name__ == "__main__":
    unittest.main()
