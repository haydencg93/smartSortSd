"""Template: built into one self-contained Kaggle script by build_kernel.py.

KAGGLE_CONFIG and EMBEDDED_SOURCES are provided by the builder. Do not push
this template by itself. Uses existing ml/train_data.py, ml/setup_data.py.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")
MODEL_NAME = "smart_sort_mobilenetv2_combined"


def prepared_root(parent):
    """Accept Kaggle's unzipped archive layout with zero, one, or two wrappers."""
    pending = [(parent, 0)]
    while pending:
        p, depth = pending.pop(0)
        if (p / "train").is_dir() and (p / "val").is_dir():
            return p
        if depth < 3 and p.is_dir():
            pending += [(child, depth + 1) for child in sorted(p.iterdir()) if child.is_dir()]
    return None


def sources_root(parent):
    pending = [(parent, 0)]
    while pending:
        p, depth = pending.pop(0)
        if (p / "sources").is_dir() and (p / "sources.json").is_file():
            return p
        if depth < 3 and p.is_dir():
            pending += [(child, depth + 1) for child in sorted(p.iterdir()) if child.is_dir()]
    return None


def unpack_if_necessary(parent, mode):
    probe = prepared_root if mode == "prepared" else sources_root
    found = probe(parent)
    if found:
        return found
    bundles = list(parent.rglob("smartsort_bundle.zip"))
    if len(bundles) != 1:
        raise RuntimeError(f"Could not find {mode} layout or exactly one smartsort_bundle.zip beneath {parent}")
    if mode == "raw":
        raise RuntimeError("Raw image archive not mounted expanded by Kaggle. "
                           "Unpacking the full 10+ GB archive into /kaggle/working may exceed saved disk. "
                           "Upload the raw sources as Kaggle-expanded directories, or use prepared mode.")
    extracted = WORK_ROOT / "_expanded_prepared"
    extracted.mkdir(exist_ok=True)
    # ZipFile.extractall can write paths; validate members first.
    with zipfile.ZipFile(bundles[0]) as z:
        for item in z.infolist():
            dest = (extracted / item.filename).resolve()
            if not dest.is_relative_to(extracted.resolve()):
                raise RuntimeError("Unsafe ZIP member path")
        z.extractall(extracted)
    found = probe(extracted)
    if not found:
        raise RuntimeError("Expanded archive did not contain train/ and val/")
    return found


def resolve_dataset_mount(dataset_ref, input_root=INPUT_ROOT):
    """Resolve Kaggle's flat or owner-qualified dataset mount paths.

    Kaggle may choose /kaggle/input/<slug>, /kaggle/input/<owner>/<slug>,
    or /kaggle/input/datasets/<owner>/<slug> depending on its input layout.
    Never guess a different dataset just because its name looks similar.
    """
    parts = dataset_ref.strip("/").split("/")
    if len(parts) != 2 or not all(parts):
        raise RuntimeError(f"Invalid Kaggle dataset reference: {dataset_ref!r}")
    owner, slug = parts
    candidates = (
        input_root / slug,
        input_root / owner / slug,
        input_root / "datasets" / owner / slug,
        input_root / "datasets" / slug,
    )
    for candidate in candidates:
        if candidate.is_dir():
            print(f"Using Kaggle dataset mount: {candidate}", flush=True)
            return candidate

    # List only nearby directories; do not enumerate thousands of images.
    visible = []
    queue = [(input_root, 0)]
    while queue and len(visible) < 50:
        base, depth = queue.pop(0)
        if not base.is_dir() or depth >= 3:
            continue
        try:
            dirs = sorted(p for p in base.iterdir() if p.is_dir())
        except OSError:
            continue
        for child in dirs:
            visible.append(str(child))
            queue.append((child, depth + 1))
            if len(visible) >= 50:
                break
    raise RuntimeError(
        f"Kaggle dataset {dataset_ref!r} is not mounted at any supported path. "
        f"Checked: {[str(p) for p in candidates]}. "
        f"Visible input directories: {visible or ['(none)']}. "
        "Verify the dataset is READY and the notebook has the dataset "
        "attached (kernel-metadata.json dataset_sources)."
    )


def main():
    config = KAGGLE_CONFIG  # provided by build_kernel.py
    src_dir = WORK_ROOT / "_smartsort_code"
    src_dir.mkdir(parents=True, exist_ok=True)
    for filename, source in EMBEDDED_SOURCES.items():
        (src_dir / filename).write_text(source, encoding="utf-8")
    dataset_dir = resolve_dataset_mount(config["dataset_ref"])
    # Fail rather than quietly burning the 12-hour GPU window on CPU.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available. Check Kaggle accelerator and GPU quota.")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    data_root = unpack_if_necessary(dataset_dir, config["mode"])
    if config["mode"] == "raw":
        # Mounted inputs are read-only. Setup's no--add mode only *reads* existing
        # registered originals; symlinks let it see them without copying 10+ GB.
        output_raw = WORK_ROOT / "raw_data"
        links = output_raw / "sources"
        links.mkdir(parents=True, exist_ok=True)
        registry = json.loads((data_root / "sources.json").read_text(encoding="utf-8"))
        expected = {entry["folder"] for entry in registry.get("sources", [])}
        actual = {p.name for p in (data_root / "sources").iterdir() if p.is_dir()}
        if not expected or expected != actual:
            raise RuntimeError("Kaggle raw source folders do not match sources.json")
        for item in (data_root / "sources").iterdir():
            if item.is_dir():
                (links / item.name).symlink_to(item, target_is_directory=True)
        shutil.copy2(data_root / "sources.json", output_raw / "sources.json")
        print("Preparing existing registered sources using read-only Kaggle mounts", flush=True)
        # Cloud-only adapter: setup_data.py's original_records() calls .resolve(),
        # so its source `path` must ALSO be the resolved mount root or relative_to()
        # will reject images. Do not edit the user's local setup_data.py.
        sys.path.insert(0, str(src_dir))
        import setup_data
        original_reader = setup_data._read_registered_sources
        def resolved_reader(root, allow_non_cc0):
            sources = original_reader(root, allow_non_cc0)
            for entry in sources:
                entry["path"] = entry["path"].resolve()
            return sources
        setup_data._read_registered_sources = resolved_reader
        setup_data.prepare(raw_data=output_raw)
        data_root = output_raw
    output = WORK_ROOT / f"{MODEL_NAME}.pth"
    # No --init-checkpoint: intentionally fresh ImageNet-pretrained model on all sources.
    cmd = [sys.executable, str(src_dir / "train_data.py"),
           "--data", str(data_root), "--output", str(output),
           "--epochs", str(config["epochs"]), "--batch-size", str(config["batch_size"])]
    print("Starting existing Smart Sort train_data.py:", cmd, flush=True)
    subprocess.run(cmd, check=True, cwd=WORK_ROOT)
    expected = [output, output.with_suffix(".classes.json"), output.with_suffix(".metrics.json")]
    if any(not p.is_file() for p in expected):
        raise RuntimeError("Training did not produce .pth + .classes.json + .metrics.json")
    record = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": config["source_commit"],
        "dataset_ref": config["dataset_ref"],
        "mode": config["mode"],
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "model_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "training_source_sha256": hashlib.sha256(EMBEDDED_SOURCES["train_data.py"].encode()).hexdigest(),
        "setup_source_sha256": hashlib.sha256(EMBEDDED_SOURCES["setup_data.py"].encode()).hexdigest(),
    }
    (WORK_ROOT / "cloud_run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    if config["mode"] == "raw":
        # This is the temporary working copy, not Kaggle's immutable input or D:.
        # Remove it so Kaggle saves only the model outputs, not rebuilt images.
        shutil.rmtree(WORK_ROOT / "raw_data")
    # If the prepared ZIP was not automatically expanded by Kaggle, the fallback
    # extraction is transient and must not be archived as a notebook output.
    if (WORK_ROOT / "_expanded_prepared").is_dir():
        shutil.rmtree(WORK_ROOT / "_expanded_prepared")
    shutil.rmtree(src_dir)
    print("Training finished. Kaggle outputs saved:", [p.name for p in expected], flush=True)
    print("Run provenance:", record, flush=True)


if __name__ == "__main__":
    main()
