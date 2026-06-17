"""
E_mask_quality_iou.py — Mask Quality Evaluation
=================================================
Computes IoU between SAM2-generated masks and COCONut panoptic
ground truth masks for each region in the subset.

Two tracks evaluated:
  Track 1 — masks_auto   (SAM2 auto grid)
  Track 2 — masks_click  (SAM2 label-guided)

Requires:
  - data/coconut_subset/annotations/coconut_stub.json
  - data/coconut_subset/masks_auto/<id>/mask_NN.png
  - data/coconut_subset/masks_click/<id>/mask_NN.png
  - data/content_images/annotations/panoptic_train2017.json
  - data/content_images/annotations/panoptic_train2017/ (PNG files)

Usage:
    python3 E_mask_quality_iou.py \
        --stub      ../data/coconut_subset/annotations/coconut_stub.json \
        --pan_json  ../data/content_images/annotations/panoptic_train2017.json \
        --pan_dir   ../data/content_images/annotations/panoptic_train2017 \
        --masks_auto  ../data/coconut_subset/masks_auto \
        --masks_click ../data/coconut_subset/masks_click \
        --output    ../data/coconut_subset/annotations/mask_quality_iou.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# IoU utilities
# ─────────────────────────────────────────────────────────────────────────────

def mask_iou(pred, gt):
    """Compute IoU between two binary masks."""
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    inter = float((pred & gt).sum())
    union = float((pred | gt).sum())
    return inter / union if union > 0 else 0.0


def mask_precision(pred, gt):
    """What fraction of predicted pixels are correct."""
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    tp   = float((pred & gt).sum())
    return tp / float(pred.sum()) if pred.sum() > 0 else 0.0


def mask_recall(pred, gt):
    """What fraction of ground truth pixels are captured."""
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    tp   = float((pred & gt).sum())
    return tp / float(gt.sum()) if gt.sum() > 0 else 0.0


def decode_panoptic_gt(pan_png_path, seg_id):
    """
    Decode ground truth mask for a given segment ID from panoptic PNG.
    COCO panoptic format: segment_id = R + G*256 + B*256*256
    """
    pan = np.array(Image.open(pan_png_path).convert("RGB"))
    id_map = (pan[:,:,0].astype(np.int32) +
              pan[:,:,1].astype(np.int32) * 256 +
              pan[:,:,2].astype(np.int32) * 256 * 256)
    return (id_map == seg_id).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    stub_path = Path(args.stub)
    pan_json  = Path(args.pan_json)
    pan_dir   = Path(args.pan_dir)
    mask_dirs = {
        "auto":  Path(args.masks_auto),
        "click": Path(args.masks_click),
    }

    with open(stub_path) as f:
        records = json.load(f)

    # ── Load panoptic annotations ─────────────────────────────────────────
    print("Loading COCONut panoptic annotations...")
    if not pan_json.exists():
        print(f"[error] Panoptic JSON not found: {pan_json}")
        print("Download with:")
        print("  cd ~/Benchmark_dataset/data/content_images")
        print("  wget http://images.cocodataset.org/annotations/panoptic_train2017.zip")
        print("  unzip panoptic_train2017.zip")
        return

    with open(pan_json) as f:
        pan_data = json.load(f)

    # Build lookup: image_id (str) -> panoptic annotation entry
    pan_lookup = {
        str(ann["image_id"]): ann
        for ann in pan_data["annotations"]
    }
    print(f"Loaded {len(pan_lookup)} panoptic annotations")

    # ── Evaluate both tracks ──────────────────────────────────────────────
    results       = {}
    all_iou_auto  = []
    all_iou_click = []
    no_gt_count   = 0
    no_mask_count = 0

    data_root = stub_path.parent.parent  # coconut_subset/

    for record in tqdm(records, desc="Evaluating masks"):
        image_id = record["image_id"]
        pan_ann  = pan_lookup.get(image_id)

        if pan_ann is None:
            no_gt_count += 1
            continue

        # Panoptic PNG path
        pan_png = pan_dir / pan_ann["file_name"]
        if not pan_png.exists():
            no_gt_count += 1
            continue

        results[image_id] = {"regions": []}

        for region in record["regions"]:
            seg_ids    = region.get("region_ids", [])
            mask_index = region["mask_index"]
            label      = region["region_label"]

            if not seg_ids:
                continue

            # Get ground truth mask for this segment
            # If multiple IDs (e.g. [15,19,20] for "cars and trucks"), union them
            gt_mask = None
            for seg_id in seg_ids:
                try:
                    m = decode_panoptic_gt(pan_png, seg_id)
                    gt_mask = m if gt_mask is None else (gt_mask | m)
                except Exception:
                    continue

            if gt_mask is None or gt_mask.sum() == 0:
                continue

            region_result = {
                "mask_index":   mask_index,
                "region_label": label,
                "region_ids":   seg_ids,
                "gt_area":      float(gt_mask.sum()) / float(gt_mask.size),
                "tracks":       {}
            }

            # Evaluate each track
            for track_name, mask_dir in mask_dirs.items():
                mask_path = mask_dir / image_id / f"mask_{mask_index:02d}.png"

                if not mask_path.exists():
                    no_mask_count += 1
                    region_result["tracks"][track_name] = {
                        "iou": None, "precision": None, "recall": None,
                        "note": "mask file not found"
                    }
                    continue

                pred_mask = np.array(
                    Image.open(mask_path).convert("L")
                ) > 127

                # Resize gt to match pred if needed
                if gt_mask.shape != pred_mask.shape:
                    gt_pil   = Image.fromarray(gt_mask * 255)
                    gt_pil   = gt_pil.resize(
                        (pred_mask.shape[1], pred_mask.shape[0]),
                        Image.NEAREST
                    )
                    gt_mask_r = np.array(gt_pil) > 127
                else:
                    gt_mask_r = gt_mask.astype(bool)

                iou  = mask_iou(pred_mask, gt_mask_r)
                prec = mask_precision(pred_mask, gt_mask_r)
                rec  = mask_recall(pred_mask, gt_mask_r)

                region_result["tracks"][track_name] = {
                    "iou":       round(iou,  4),
                    "precision": round(prec, 4),
                    "recall":    round(rec,  4),
                }

                if track_name == "auto":
                    all_iou_auto.append(iou)
                else:
                    all_iou_click.append(iou)

            results[image_id]["regions"].append(region_result)

    # ── Summary ───────────────────────────────────────────────────────────
    def stats(vals):
        if not vals:
            return {"mean": 0, "median": 0, "min": 0, "max": 0, "n": 0}
        arr = np.array(vals)
        return {
            "mean":   round(float(arr.mean()), 4),
            "median": round(float(np.median(arr)), 4),
            "min":    round(float(arr.min()), 4),
            "max":    round(float(arr.max()), 4),
            "n":      len(vals),
            "above_0.5":  int((arr >= 0.5).sum()),
            "above_0.75": int((arr >= 0.75).sum()),
        }

    summary = {
        "auto_track":  stats(all_iou_auto),
        "click_track": stats(all_iou_click),
        "no_gt_found": no_gt_count,
        "no_mask_found": no_mask_count,
    }

    output = {
        "summary": summary,
        "per_image": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*50}")
    print("MASK QUALITY RESULTS")
    print(f"{'='*50}")
    print(f"\nTrack 1 — Auto grid:")
    s = summary["auto_track"]
    print(f"  Regions evaluated : {s['n']}")
    print(f"  Mean IoU          : {s['mean']}")
    print(f"  Median IoU        : {s['median']}")
    print(f"  Min / Max         : {s['min']} / {s['max']}")
    print(f"  IoU >= 0.50       : {s['above_0.5']} / {s['n']}")
    print(f"  IoU >= 0.75       : {s['above_0.75']} / {s['n']}")

    print(f"\nTrack 2 — Click/label-guided:")
    s = summary["click_track"]
    print(f"  Regions evaluated : {s['n']}")
    print(f"  Mean IoU          : {s['mean']}")
    print(f"  Median IoU        : {s['median']}")
    print(f"  Min / Max         : {s['min']} / {s['max']}")
    print(f"  IoU >= 0.50       : {s['above_0.5']} / {s['n']}")
    print(f"  IoU >= 0.75       : {s['above_0.75']} / {s['n']}")

    print(f"\nNo ground truth found : {no_gt_count} images")
    print(f"No mask file found    : {no_mask_count} regions")
    print(f"\nOutput: {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stub",
                   default="../data/coconut_subset/annotations/coconut_stub.json")
    p.add_argument("--pan_json",
                   default="../data/content_images/annotations/panoptic_train2017.json")
    p.add_argument("--pan_dir",
                   default="../data/content_images/annotations/panoptic_train2017")
    p.add_argument("--masks_auto",
                   default="../data/coconut_subset/masks_auto")
    p.add_argument("--masks_click",
                   default="../data/coconut_subset/masks_click")
    p.add_argument("--output",
                   default="../data/coconut_subset/annotations/mask_quality_iou.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
