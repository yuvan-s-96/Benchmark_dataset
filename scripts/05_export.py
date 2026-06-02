"""
Step 5: Final Export
====================
Packages the completed benchmark into a clean, self-contained zip archive
ready for upload to Google Drive, OneDrive, or university filestore.

Usage:
    python 05_export.py \
        --annotated_json ../data/annotations/benchmark_annotated.json \
        --output_zip     ../benchmark_final_export.zip

What it includes:
    - benchmark_annotated.json
    - all mask PNG files referenced in the JSON
    - all style reference images referenced in the JSON
    - export_summary.txt with dataset statistics

What it excludes:
    - content_images/  (large, already public via COCO)
    - dropped_records.json
    - venv/, logs/, __pycache__/

Dependencies: stdlib only (json, zipfile, pathlib)
"""

import argparse
import json
import zipfile
from datetime import datetime
from pathlib import Path


def export(args):
    annotated_json = Path(args.annotated_json)
    if not annotated_json.exists():
        raise FileNotFoundError(
            f"Annotated JSON not found: {annotated_json}\n"
            "Run steps 1-4 first."
        )

    with open(annotated_json) as f:
        records = json.load(f)

    root = annotated_json.parent.parent   # data/
    zip_path = Path(args.output_zip)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Collect all files needed ──────────────────────────────────────────
    files_to_add: set[Path] = set()
    files_to_add.add(annotated_json)

    missing_masks:  list[str] = []
    missing_styles: list[str] = []

    for r in records:
        for region in r.get("regions", []):
            mask = root / region["mask_file"]
            if mask.exists():
                files_to_add.add(mask)
            else:
                missing_masks.append(str(mask))

            style = region.get("style_reference", "")
            if style and Path(style).exists():
                files_to_add.add(Path(style))
            elif style:
                missing_styles.append(style)

    if missing_masks:
        print(f"[warn] {len(missing_masks)} mask files missing — "
              "re-run Step 1 or check paths.")
    if missing_styles:
        print(f"[warn] {len(missing_styles)} style images missing — "
              "re-run download_wikiart.py.")

    # ── Summary stats ─────────────────────────────────────────────────────
    splits:        dict[str, int] = {}
    tags:          dict[str, int] = {}
    region_counts: list[int]      = []
    approved = sum(
        1 for r in records if r.get("annotation_status") == "approved"
    )
    rejected = sum(
        1 for r in records if r.get("annotation_status") == "rejected"
    )

    for r in records:
        key = r.get("split", "unset")
        splits[key] = splits.get(key, 0) + 1
        region_counts.append(r.get("num_regions", 0))
        for t in r.get("corner_case_tags", []):
            tags[t] = tags.get(t, 0) + 1

    summary = f"""Benchmark Export Summary
========================
Exported  : {datetime.now().strftime('%Y-%m-%d %H:%M')}
Source    : {annotated_json}

── Samples ──────────────────────────
Total     : {len(records)}
  Train   : {splits.get('train', 0)}
  Val     : {splits.get('val', 0)}
  Test    : {splits.get('test', 0)}
  Approved: {approved}
  Rejected: {rejected}

── Regions per sample ────────────────
  Min     : {min(region_counts) if region_counts else 0}
  Max     : {max(region_counts) if region_counts else 0}
  Mean    : {sum(region_counts)/len(region_counts):.1f if region_counts else 0}

── Corner cases ──────────────────────
  similar_entities : {tags.get('similar_entities', 0)}
  encompassed      : {tags.get('encompassed', 0)}
  background_heavy : {tags.get('background_heavy', 0)}

── Archive ───────────────────────────
  Files included : {len(files_to_add)}
  Missing masks  : {len(missing_masks)}
  Missing styles : {len(missing_styles)}
"""
    print(summary)

    # ── Write zip ─────────────────────────────────────────────────────────
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("export_summary.txt", summary)

        for fpath in sorted(files_to_add):
            try:
                arcname = fpath.relative_to(root)
            except ValueError:
                arcname = Path(fpath.name)
            zf.write(fpath, arcname)
            print(f"  + {arcname}")

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\nExport complete: {zip_path}  ({size_mb:.1f} MB)")
    print(
        "\nTo copy to your laptop run (in a NEW local terminal):\n"
        f"  scp yvs23@ogg.cs.bath.ac.uk:{zip_path.resolve()} .\n"
        "  # or with rsync (resumable):\n"
        f"  rsync -avz --progress yvs23@ogg.cs.bath.ac.uk:{zip_path.resolve()} ."
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Package the final benchmark into a portable zip archive."
    )
    p.add_argument(
        "--annotated_json",
        default="../data/annotations/benchmark_annotated.json",
        help="Path to benchmark_annotated.json (output of Step 4)"
    )
    p.add_argument(
        "--output_zip",
        default="../benchmark_final_export.zip",
        help="Where to write the zip archive"
    )
    return p.parse_args()


if __name__ == "__main__":
    export(parse_args())
