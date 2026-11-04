#!/usr/bin/env python3
"""Fail closed: check Kaggle outputs, provenance, class mapping and validation gates.

This is NOT an independent 100-image test. Gate thresholds should be raised or
supplemented with a real holdout test before enabling automatic deployment.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

EXPECTED = (
    "cardboard", "celluloseacetate", "glass", "metal", "paper", "plastic",
    "polystyrene", "rubber", "tetrapak", "textile", "trash"
)
BASENAME = "smart_sort_mobilenetv2_combined"


def validate(args):
    import torch
    from torchvision import models
    directory = Path(args.directory)
    paths = {"model": directory / f"{BASENAME}.pth",
             "classes": directory / f"{BASENAME}.classes.json",
             "metrics": directory / f"{BASENAME}.metrics.json",
             "provenance": directory / "cloud_run.json"}
    for name, p in paths.items():
        if not p.is_file():
            raise SystemExit(f"Missing required Kaggle output {name}: {p}")
    labels = json.loads(paths["classes"].read_text(encoding="utf-8"))["classes"]
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    run = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    if tuple(labels) != EXPECTED or metrics.get("classes") != labels:
        raise SystemExit("Unexpected 11-class order: refusing publication")
    digest = hashlib.sha256(paths["model"].read_bytes()).hexdigest()
    if run.get("model_sha256") != digest:
        raise SystemExit("Model SHA256 disagrees with Kaggle provenance file")
    source_commit = str(run.get("source_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit) or source_commit == "0" * 40:
        raise SystemExit("Run source_commit is invalid (requires actual GitHub commit SHA)")
    history = metrics.get("history", [])
    if not history:
        raise SystemExit("No validation history in metrics.json")
    best = max(history, key=lambda item: item["val_macro_f1"])
    f1, acc = float(best["val_macro_f1"]), float(best["val_accuracy"])
    recall = best.get("val_recall_by_class", {})
    print("Best validation epoch:", best["epoch"], "accuracy:", acc,
          "macro-F1:", f1, "glass recall:", recall.get("glass"),
          "trash recall:", recall.get("trash"))
    gates = [("accuracy", acc, args.min_accuracy),
             ("macro-F1", f1, args.min_macro_f1),
             ("glass recall", float(recall.get("glass", -1)), args.min_glass_recall),
             ("trash recall", float(recall.get("trash", -1)), args.min_trash_recall)]
    for name, actual, minimum in gates:
        if actual < minimum:
            raise SystemExit(f"Model failed {name} gate: {actual:.3f} < {minimum:.3f}. No publication.")
    # Verify that all state_dict keys/shapes match a fresh 11-output MobileNetV2.
    state = torch.load(paths["model"], map_location="cpu", weights_only=True)
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(labels))
    model.load_state_dict(state, strict=True)
    previous = Path(args.previous)
    changed = not previous.is_file() or hashlib.sha256(previous.read_bytes()).hexdigest() != digest
    tag = "smartsort-" + digest[:12]
    print("Verified checkpoint shape and mapping.", "Changed:", changed, "Release:", tag)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"changed={'true' if changed else 'false'}\n")
            f.write(f"tag={tag}\n")
            f.write(f"digest={digest}\n")
            f.write(f"source_commit={run['source_commit']}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", required=True)
    p.add_argument("--previous", default="ml/smart_sort_mobilenetv2_combined.pth")
    p.add_argument("--min-accuracy", type=float, default=.85)
    p.add_argument("--min-macro-f1", type=float, default=.75)
    p.add_argument("--min-glass-recall", type=float, default=.80)
    p.add_argument("--min-trash-recall", type=float, default=.80)
    validate(p.parse_args())
