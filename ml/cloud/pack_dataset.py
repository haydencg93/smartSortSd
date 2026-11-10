#!/usr/bin/env python3
"""Make a PRIVATE Kaggle upload bundle from Smart Sort's existing raw_data.

Only packs local files; does not upload, move, or delete originals. Everything is
streamed to the selected destination (use your D: flash drive, not full C:).
"""
import argparse
import json
import shutil
import zipfile
from pathlib import Path


def pack(args):
    raw = Path(args.raw_data).expanduser().resolve()
    dest = Path(args.destination).expanduser().resolve()
    if not raw.is_dir():
        raise SystemExit(f"Missing Smart Sort raw_data directory: {raw}")
    if dest == raw or raw in dest.parents:
        raise SystemExit("Destination must be outside raw_data (never package your output into itself).")
    if args.mode == "prepared":
        required = [raw / "train", raw / "val"]
        if not all(p.is_dir() for p in required):
            raise SystemExit("Missing raw_data/train or raw_data/val. Run setup_data.py locally first.")
        folders = ["train", "val"]
        extras = ["source_manifest.json"]
    else:
        required = [raw / "sources", raw / "sources.json"]
        if not required[0].is_dir() or not required[1].is_file():
            raise SystemExit("Raw mode requires raw_data/sources and raw_data/sources.json.")
        registry = json.loads((raw / "sources.json").read_text(encoding="utf-8"))
        names = {p.get("folder") for p in registry.get("sources", [])}
        actual = {p.name for p in (raw / "sources").iterdir() if p.is_dir()}
        if names != actual or not names:
            raise SystemExit("sources.json and sources/ disagree; fix source registration before uploading.")
        folders = ["sources"]
        extras = ["sources.json"]
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "smartsort_bundle.zip"
    meta_file = dest / "dataset-metadata.json"
    if archive.exists() or meta_file.exists():
        raise SystemExit("Output ZIP or metadata already exists. Use a new empty output folder to protect older uploads.")
    n_files = 0
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for folder in folders:
            for p in sorted((raw / folder).rglob("*")):
                if p.is_file() and not p.is_symlink():
                    z.write(p, arcname=p.relative_to(raw).as_posix())
                    n_files += 1
        for file in extras:
            p = raw / file
            if p.is_file():
                z.write(p, arcname=file)
                n_files += 1
    if n_files == 0:
        archive.unlink(missing_ok=True)
        raise SystemExit("Nothing to package")
    # The prepared dataset can now contain imported CC BY sources as well as
    # MIT and CC0. Describe what its OWN source manifest declares; never label
    # the combined archive as freely redistributable or CC0-only.
    provenance = raw / ('source_manifest.json' if args.mode == 'prepared' else 'sources.json')
    source_info = json.loads(provenance.read_text(encoding='utf-8')) if provenance.is_file() else {}
    declared = sorted({str(item.get('license', 'UNVERIFIED'))
                       for item in source_info.get('sources', []) if isinstance(item, dict)})
    license_summary = ', '.join(declared) if declared else 'not available'
    meta = {
        "title": args.title,
        "id": f"{args.owner}/{args.slug}",
        "licenses": [{"name": "other"}],
        "description": (
            f"PRIVATE Smart Sort ML dataset ({args.mode}). "
            f"Declared source licenses: {license_summary}. "
            "Contains mixed-license images; confirm each source's permitted use and "
            "attribution obligations before sharing. Per-source provenance and license "
            "declarations are in the source manifest included with this bundle. "
            "The combined archive is NOT CC0-only."
        )[:1000],
    }
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Packaged {n_files} files into {archive} ({archive.stat().st_size / 1024**3:.2f} GiB)")
    print(f"Metadata: {meta_file}")
    print("No dataset was uploaded and no input data was changed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--owner", required=True, help="Kaggle username")
    parser.add_argument("--slug", required=True, help="e.g. smart-sort-prepared")
    parser.add_argument("--mode", choices=["prepared", "raw"], default="prepared")
    parser.add_argument("--title", default="Smart Sort Private Training Images")
    pack(parser.parse_args())
