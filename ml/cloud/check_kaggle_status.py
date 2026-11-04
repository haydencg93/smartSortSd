#!/usr/bin/env python3
"""Write ready=true/false to GITHUB_OUTPUT only for a completed Kaggle kernel."""
import argparse
import os
import re
import subprocess
from pathlib import Path


def run(ref):
    proc = subprocess.run(["kaggle", "kernels", "status", ref],
                          capture_output=True, text=True, check=True)
    print(proc.stdout)
    print(proc.stderr)
    match = re.search(r"\bstatus\s*[:=]\s*([a-z_]+)", proc.stdout, re.I)
    if not match:
        raise SystemExit("Could not recognize Kaggle status; inspect CLI output, do not assume completion.")
    status = match.group(1).lower()
    if status not in {"complete", "completed", "running", "pending", "queued", "error", "failed", "cancelled", "canceled"}:
        raise SystemExit(f"Unknown Kaggle status: {status}")
    ready = status in {"complete", "completed"}
    print(f"Kaggle status={status}; publish-ready={ready}")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"ready={'true' if ready else 'false'}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True)
    run(parser.parse_args().kernel)
