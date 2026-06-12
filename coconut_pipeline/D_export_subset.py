"""
Step D: Export Subset for Baseline Testing
==========================================
Packages the completed COCONut subset into a zip archive
ready to share with supervisor for baseline testing.

Usage:
    python3 D_export_subset.py \
        --annotated_json ../data/coconut_subset/annotations/benchmark_subset_final.json \
        --output_zip     ../coconut_subset_export.zip
"""

import argparse
import json
import zipfile
from datetime import datetime
from pathlib import Path


def export(args):
    ann_path = Path(args.annotated_json)
    if not ann_path.exists():
        raise FileNotFoundError(f"JSON not found: {ann_path}\nRun C_quality_control.py first.")

    with open(ann_path) as f:
        records = json.load(f)

    root     = ann_path.parent.parent
    zip_path = Path(args.output_zip)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    files_to_add   = {ann_path}
    missing_masks  = []
    missing_styles = []
    missing_images = []

    for r in records:
        img = root / r["image_file"]
        if img.exists(): files_to_add.add(img)
        else: missing_images.append(str(img))

        for region in r.get("regions", []):
            mask = root / region["mask_file"]
            if mask.exists(): files_to_add.add(mask)
            else: missing_masks.append(str(mask))

            style = region.get("style_reference", "")
            if style:
                sp = Path(style)
                if sp.exists(): files_to_add.add(sp)
                else: missing_styles.append(style)

    counts        = [r.get("num_regions", 0) for r in records]
    total_regions = sum(counts)
    splits        = {}
    tags          = {}
    for r in records:
        splits[r.get("split","?")] = splits.get(r.get("split","?"),0) + 1
        for t in r.get("corner_case_tags", []):
            tags[t] = tags.get(t, 0) + 1

    has_text = sum(1 for r in records for reg in r["regions"] if reg.get("instruction_text","").strip())
    has_ref  = sum(1 for r in records for reg in r["regions"] if reg.get("instruction_ref","").strip())

    summary = f"""COCONut Subset Export Summary
==============================
Exported  : {datetime.now().strftime('%Y-%m-%d %H:%M')}
Source    : COCONut-PanCap + WikiArt hybrid pipeline

Samples   : {len(records)}  (train={splits.get('train',0)} val={splits.get('val',0)} test={splits.get('test',0)})
Regions   : total={total_regions}  min={min(counts)}  max={max(counts)}  mean={sum(counts)/len(counts):.1f}
Tags      : {tags}

Instructions:
  instruction_text (text-based) : {has_text}/{total_regions}
  instruction_ref  (ref-based)  : {has_ref}/{total_regions}

Missing   : images={len(missing_images)}  masks={len(missing_masks)}  styles={len(missing_styles)}
Files     : {len(files_to_add)} included in archive
"""

    print(summary)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("export_summary.txt", summary)
        for fpath in sorted(files_to_add):
            try:    arcname = fpath.relative_to(root)
            except: arcname = Path(fpath.name)
            zf.write(fpath, arcname)
            print(f"  + {arcname}")

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\nExport  : {zip_path}  ({size_mb:.1f} MB)")
    print(f"\nCopy to laptop:")
    print(f'  scp "yvs23@ogg.cs.bath.ac.uk:{zip_path.resolve()}" "C:/Users/Yuvan Velkumar/Downloads/"')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--annotated_json", default="../data/coconut_subset/annotations/benchmark_subset_final.json")
    p.add_argument("--output_zip",     default="../coconut_subset_export.zip")
    return p.parse_args()


if __name__ == "__main__":
    export(parse_args())
