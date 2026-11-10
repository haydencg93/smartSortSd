#!/usr/bin/env python3
"""Create a stable cache key for a completed Kaggle run + publication policy.

Only run provenance (small JSON) must be downloaded before calling this script.
The full checkpoint is downloaded/verified only when the review has not been
cached. Cached reviews can be bypassed with workflow_dispatch force_recheck.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path


def run(args):
    run_data = json.loads(Path(args.provenance).read_text(encoding="utf-8"))
    model_sha = run_data.get("model_sha256")
    source_commit = run_data.get("source_commit")
    if not isinstance(model_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", model_sha):
        raise SystemExit("Invalid model_sha256 in Kaggle cloud_run.json")
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise SystemExit("Invalid source_commit in Kaggle cloud_run.json")

    validator = Path(__file__).with_name("validate_model.py")
    validator_sha = hashlib.sha256(validator.read_bytes()).hexdigest()
    previous = Path(args.previous)
    previous_sha = hashlib.sha256(previous.read_bytes()).hexdigest() if previous.is_file() else "none"

    policy = {
        "schema": 3,
        "kernel": args.kernel,
        "dataset_ref": run_data.get("dataset_ref"),
        "model_sha256": model_sha,
        "source_commit": source_commit,
        "validator_sha256": validator_sha,
        "previous_model_sha256": previous_sha,
        "thresholds": [args.min_accuracy, args.min_macro_f1,
                       args.min_glass_recall, args.min_trash_recall,
                       args.min_class_recall, args.max_low_confidence_rate],
        # A new training source or validation gate invalidates the cached review.
        "train_source_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "train_data.py").read_bytes()).hexdigest(),
        "setup_source_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "setup_data.py").read_bytes()).hexdigest(),
        "auto_commit": args.auto_commit.lower() == "true",
    }
    key_sha = hashlib.sha256(json.dumps(policy, sort_keys=True).encode("utf-8")).hexdigest()
    cache_key = "smartsort-model-review-v3-" + key_sha
    print(f"Latest completed Kaggle model: {model_sha[:16]}...; review cache key: {cache_key}")
    output = os.getenv("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as f:
            f.write(f"cache_key={cache_key}\n")
            f.write(f"model_sha={model_sha}\n")
    return cache_key


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provenance", required=True)
    p.add_argument("--kernel", required=True)
    p.add_argument("--previous", default="ml/smart_sort_mobilenetv2_combined.pth")
    p.add_argument("--min-accuracy", type=float, default=.85)
    p.add_argument("--min-macro-f1", type=float, default=.75)
    p.add_argument("--min-glass-recall", type=float, default=.80)
    p.add_argument("--min-trash-recall", type=float, default=.80)
    p.add_argument("--min-class-recall", type=float, default=.50)
    p.add_argument("--max-low-confidence-rate", type=float, default=.05)
    p.add_argument("--auto-commit", default="false")
    run(p.parse_args())
