"""
E_mask_quality_iou.py — Mask Quality Evaluation (Best-Match IoU)
=================================================================
For each SAM2 mask, finds the best-matching COCONut panoptic segment
and computes IoU. This handles cases where explicit segment ID
correspondence is unavailable.

Approach: Hungarian-style best match
  For each SAM2 mask:
    1. Find all panoptic segments that overlap with the mask
    2. Pick the segment with highest IoU
    3. Record that IoU as the mask quality score

Usage:
    python3 E_mask_quality_iou.py \
        --stub        ../data/coconut_subset/annotations/coconut_stub_merged_auto.json \
        --stub_click  ../data/coconut_subset/annotations/coconut_stub_merged_click.json \
        --pan_json    ../data/content_images/annotations/panoptic_train2017.json \
        --pan_dir     ../data/content_images/annotations/panoptic_train2017 \
        --output      ../data/coconut_subset/annotations/mask_quality_iou.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_mask(path):
    return np.array(Image.open(path).convert("L")) > 127


def mask_iou(a, b):
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / union if union > 0 else 0.0


def decode_panoptic(pan_png_path):
    """
    Decode full panoptic PNG into a segment ID map.
    Returns: dict {segment_id: binary_mask}
    """
    pan = np.array(Image.open(pan_png_path).convert("RGB"))
    id_map = (pan[:,:,0].astype(np.int32) +
              pan[:,:,1].astype(np.int32) * 256 +
              pan[:,:,2].astype(np.int32) * 256 * 256)

    seg_ids = np.unique(id_map)
    seg_ids = seg_ids[seg_ids != 0]  # remove background

    return id_map, seg_ids


def best_match_iou(pred_mask, id_map, seg_ids):
    """
    Find the panoptic segment that best matches the predicted mask.
    Returns (best_iou, best_seg_id, precision, recall)
    """
    best_iou  = 0.0
    best_id   = -1
    best_prec = 0.0
    best_rec  = 0.0

    # Only check segments that have any overlap with pred_mask
    pred_bool    = pred_mask.astype(bool)
    overlapping  = np.unique(id_map[pred_bool])
    overlapping  = overlapping[overlapping != 0]

    for seg_id in overlapping:
        gt = (id_map == seg_id)
        iou = mask_iou(pred_bool, gt)
        if iou > best_iou:
            best_iou  = iou
            best_id   = int(seg_id)
            tp        = float((pred_bool & gt).sum())
            best_prec = tp / float(pred_bool.sum()) if pred_bool.sum() > 0 else 0.0
            best_rec  = tp / float(gt.sum())        if gt.sum()        > 0 else 0.0

    return best_iou, best_id, best_prec, best_rec


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_track(records, track_name, mask_key, pan_lookup,
                   pan_dir, cat_lookup, data_root):
    all_iou   = []
    results   = {}
    no_mask   = 0
    no_pan    = 0

    for record in tqdm(records, desc=f"Track {track_name}"):
        image_id = record["image_id"]
        pan_ann  = pan_lookup.get(image_id)

        if not pan_ann:
            no_pan += 1
            continue

        pan_png = pan_dir / pan_ann["file_name"]
        if not pan_png.exists():
            no_pan += 1
            continue

        # Decode panoptic PNG once per image
        id_map, seg_ids = decode_panoptic(pan_png)

        # Build category lookup for this image
        seg_cat = {s["id"]: cat_lookup.get(s["category_id"], "?")
                   for s in pan_ann["segments_info"]}

        results[image_id] = {"regions": []}

        for region in record["regions"]:
            mask_file  = region.get("mask_file", "")
            mask_index = region["mask_index"]
            label      = region["region_label"]

            if not mask_file:
                no_mask += 1
                continue

            mask_path = data_root / mask_file
            if not mask_path.exists():
                no_mask += 1
                continue

            pred_mask = load_mask(str(mask_path))

            # Resize panoptic map if needed
            if id_map.shape[:2] != pred_mask.shape[:2]:
                from PIL import Image as PILImage
                id_pil  = PILImage.fromarray(id_map.astype(np.int32))
                id_map_ = np.array(id_pil.resize(
                    (pred_mask.shape[1], pred_mask.shape[0]),
                    PILImage.NEAREST
                ))
            else:
                id_map_ = id_map

            iou, best_id, prec, rec = best_match_iou(pred_mask, id_map_, seg_ids)
            matched_cat = seg_cat.get(best_id, "?") if best_id > 0 else "no_match"

            all_iou.append(iou)
            results[image_id]["regions"].append({
                "mask_index":    mask_index,
                "region_label":  label,
                "best_match_iou": round(iou,  4),
                "precision":     round(prec, 4),
                "recall":        round(rec,  4),
                "matched_category": matched_cat,
                "area_fraction": region.get("area_fraction", 0.0),
            })

    # Summary stats
    arr = np.array(all_iou) if all_iou else np.array([0.0])
    summary = {
        "n":            len(all_iou),
        "mean_iou":     round(float(arr.mean()),   4),
        "median_iou":   round(float(np.median(arr)), 4),
        "min_iou":      round(float(arr.min()),    4),
        "max_iou":      round(float(arr.max()),    4),
        "above_0.5":    int((arr >= 0.5).sum()),
        "above_0.75":   int((arr >= 0.75).sum()),
        "no_mask":      no_mask,
        "no_panoptic":  no_pan,
    }
    return summary, results


def run(args):
    pan_json = Path(args.pan_json)
    pan_dir  = Path(args.pan_dir)

    if not pan_json.exists():
        print(f"[error] {pan_json} not found.")
        return

    print("Loading panoptic annotations...")
    with open(pan_json) as f:
        pan_data = json.load(f)

    pan_lookup = {str(a["image_id"]): a for a in pan_data["annotations"]}
    cat_lookup = {c["id"]: c["name"]   for c in pan_data["categories"]}
    print(f"Loaded {len(pan_lookup)} images\n")

    output = {}

    for track_name, stub_path in [
        ("auto",  args.stub),
        ("click", args.stub_click),
    ]:
        if not Path(stub_path).exists():
            print(f"[skip] {stub_path} not found")
            continue

        with open(stub_path) as f:
            records = json.load(f)

        data_root = Path(stub_path).parent.parent  # coconut_subset/

        summary, results = evaluate_track(
            records, track_name, "mask_file",
            pan_lookup, pan_dir, cat_lookup, data_root
        )
        output[track_name] = {"summary": summary, "per_image": results}

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print results
    for track_name in ["auto", "click"]:
        if track_name not in output:
            continue
        s = output[track_name]["summary"]
        label = "Track 1 — Auto grid" if track_name == "auto" else "Track 2 — Click/label-guided"
        print(f"\n{'='*50}")
        print(f"{label}")
        print(f"{'='*50}")
        print(f"  Regions evaluated : {s['n']}")
        print(f"  Mean IoU          : {s['mean_iou']}")
        print(f"  Median IoU        : {s['median_iou']}")
        print(f"  Min / Max         : {s['min_iou']} / {s['max_iou']}")
        print(f"  IoU >= 0.50       : {s['above_0.5']} / {s['n']}")
        print(f"  IoU >= 0.75       : {s['above_0.75']} / {s['n']}")
        print(f"  No mask file      : {s['no_mask']}")
        print(f"  No panoptic GT    : {s['no_panoptic']}")

    print(f"\nOutput: {args.output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stub",
                   default="../data/coconut_subset/annotations/coconut_stub_merged_auto.json")
    p.add_argument("--stub_click",
                   default="../data/coconut_subset/annotations/coconut_stub_merged_click.json")
    p.add_argument("--pan_json",
                   default="../data/content_images/annotations/panoptic_train2017.json")
    p.add_argument("--pan_dir",
                   default="../data/content_images/annotations/panoptic_train2017")
    p.add_argument("--output",
                   default="../data/coconut_subset/annotations/mask_quality_iou.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
