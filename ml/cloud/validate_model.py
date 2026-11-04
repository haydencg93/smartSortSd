#!/usr/bin/env python3
"""Verify a Kaggle model and report quality-gate outcomes without treating rejection as an error.

Structural/provenance failures exit nonzero. A valid checkpoint that misses quality
thresholds exits zero, writes approved=false, and is never published.
Training-set validation is NOT an independent camera/100-image benchmark.
"""
import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

EXPECTED = (
    "cardboard", "celluloseacetate", "glass", "metal", "paper", "plastic",
    "polystyrene", "rubber", "tetrapak", "textile", "trash",
)
BASENAME = "smart_sort_mobilenetv2_combined"


def append_github_file(env_name, lines):
    path = os.environ.get(env_name)
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def validate(args):
    import torch
    from torchvision import models

    directory = Path(args.directory)
    paths = {
        "model": directory / f"{BASENAME}.pth",
        "classes": directory / f"{BASENAME}.classes.json",
        "metrics": directory / f"{BASENAME}.metrics.json",
        "provenance": directory / "cloud_run.json",
    }
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
    if not isinstance(history, list) or not history:
        raise SystemExit("No validation history in metrics.json")
    try:
        best = max(history, key=lambda item: float(item["val_macro_f1"]))
        f1 = float(best["val_macro_f1"])
        acc = float(best["val_accuracy"])
        recall = best["val_recall_by_class"]
        glass = float(recall["glass"])
        trash = float(recall["trash"])
        epoch = int(best["epoch"])
    except (TypeError, ValueError, KeyError, OverflowError) as exc:
        raise SystemExit(f"Malformed validation metrics: {exc}") from exc

    gates = [
        ("Accuracy", acc, args.min_accuracy),
        ("Macro-F1", f1, args.min_macro_f1),
        ("Glass recall", glass, args.min_glass_recall),
        ("Trash recall", trash, args.min_trash_recall),
    ]
    if any(not math.isfinite(actual) or not math.isfinite(minimum)
           for _, actual, minimum in gates):
        raise SystemExit("Non-finite validation metric or threshold: refusing publication")
    if any(not (0 <= actual <= 1 and 0 <= minimum <= 1)
           for _, actual, minimum in gates):
        raise SystemExit("Validation metrics/thresholds must be in [0, 1]")

    # Validate the checkpoint even when the metrics FAIL their gates. A broken
    # checkpoint is an actual workflow error, not a normal quality rejection.
    state = torch.load(paths["model"], map_location="cpu", weights_only=True)
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(labels))
    model.load_state_dict(state, strict=True)

    failed = [(name, actual, minimum) for name, actual, minimum in gates
              if actual < minimum]
    approved = not failed
    previous = Path(args.previous)
    changed = not previous.is_file() or hashlib.sha256(previous.read_bytes()).hexdigest() != digest
    tag = "smartsort-" + digest[:12]
    verdict = "APPROVED for automated publication gate" if approved else "REJECTED: quality gates unmet (normal review outcome)"

    print(f"Best validation epoch: {epoch}; {verdict}")
    for name, actual, minimum in gates:
        print(f"  {name}: {actual:.4f} (minimum {minimum:.4f}) -- {'PASS' if actual >= minimum else 'FAIL'}")
    print(f"Checkpoint/provenance verified; changed={changed}; release={tag}")
    if not approved:
        print("No publication. Workflow completed normally: valid candidate did not meet quality thresholds.")

    append_github_file("GITHUB_OUTPUT", [
        f"approved={'true' if approved else 'false'}",
        f"changed={'true' if changed else 'false'}",
        f"tag={tag}",
        f"digest={digest}",
        f"source_commit={source_commit}",
        "failed_gates=" + ", ".join(name for name, _, _ in failed),
    ])
    append_github_file("GITHUB_STEP_SUMMARY", [
        "## Smart Sort model review",
        f"**Result:** {verdict}",
        f"**Best validation epoch:** {epoch}  ",
        f"**Model digest:** `{digest}`  ",
        "",
        "| Quality metric | Actual | Required | Result |",
        "| --- | ---: | ---: | --- |",
        *[f"| {name} | {actual:.2%} | {minimum:.2%} | {'PASS' if actual >= minimum else 'REJECT'} |"
          for name, actual, minimum in gates],
        "",
        "Checkpoint structure, class order, and provenance were verified. "
        "Passing these metrics does not replace independent camera/held-out benchmarking.",
    ])
    return approved


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", required=True)
    p.add_argument("--previous", default="ml/smart_sort_mobilenetv2_combined.pth")
    p.add_argument("--min-accuracy", type=float, default=.85)
    p.add_argument("--min-macro-f1", type=float, default=.75)
    p.add_argument("--min-glass-recall", type=float, default=.80)
    p.add_argument("--min-trash-recall", type=float, default=.80)
    validate(p.parse_args())
