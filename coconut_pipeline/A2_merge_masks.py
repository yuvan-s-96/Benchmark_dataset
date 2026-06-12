"""
A2_merge_masks.py
=================
After running SAM2 on the COCONut images, merges the generated mask
paths back into coconut_stub.json so B_build_subset.py can use them.

Usage:
    python3 A2_merge_masks.py \
        --stub     ../data/coconut_subset/annotations/coconut_stub.json \
        --sam2_stub ../data/coconut_subset/annotations/masks_stub_sam2.json \
        --output   ../data/coconut_subset/annotations/coconut_stub_merged.json
"""

import argparse
import json
from pathlib import Path


def run(args):
    with open(args.stub) as f:
        coconut_records = json.load(f)

    with open(args.sam2_stub) as f:
        sam2_records = json.load(f)

    # Build lookup: image_id -> sam2 regions
    sam2_lookup = {r["image_id"]: r for r in sam2_records}

    merged  = []
    skipped = []

    for record in coconut_records:
        image_id = record["image_id"]
        sam2     = sam2_lookup.get(image_id)

        if not sam2 or not sam2.get("regions"):
            skipped.append(image_id)
            continue

        sam2_regions    = sam2["regions"]
        coconut_regions = record["regions"]

        # Pair SAM2 masks with COCONut region labels by index
        # SAM2 may have different count — take the minimum
        n = min(len(sam2_regions), len(coconut_regions))

        merged_regions = []
        for i in range(n):
            s = sam2_regions[i]
            c = coconut_regions[i]
            merged_regions.append({
                "mask_index":       i,
                "mask_file":        s["mask_file"],
                "region_label":     c["region_label"],
                "region_caption":   c["region_caption"],
                "area_fraction":    s["area_fraction"],
                "iou_score":        s["iou_score"],
                "style_name":       "",
                "style_reference":  "",
                "instruction_text": "",
                "instruction_ref":  "",
            })

        if len(merged_regions) < 2:
            skipped.append(image_id)
            continue

        record["regions"]     = merged_regions
        record["num_regions"] = len(merged_regions)
        merged.append(record)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged  : {len(merged)} samples")
    print(f"Skipped : {len(skipped)} (no SAM2 masks found)")
    print(f"Output  : {out_path}")

    if merged:
        r = merged[0]
        print(f"\nSample  : {r['image_id']}")
        for reg in r["regions"][:3]:
            print(f"  {reg['region_label']} — mask: {reg['mask_file']}")

    print(f"\nNext: python3 B_build_subset.py --stub {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stub",
                   default="../data/coconut_subset/annotations/coconut_stub.json")
    p.add_argument("--sam2_stub",
                   default="../data/coconut_subset/annotations/masks_stub_sam2.json")
    p.add_argument("--output",
                   default="../data/coconut_subset/annotations/coconut_stub_merged.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
