"""
Step A: Download COCONut-PanCap Subset
=======================================
Downloads a small subset from COCONut-PanCap on HuggingFace.
Saves content images, masks (decoded from panoptic segmentation),
and region captions for each sample.

This is part of the HYBRID PIPELINE (COCONut track).
Do NOT modify the original scripts/ directory.

Usage:
    python3 A_download_coconut.py \
        --output_dir  ../data/coconut_subset \
        --num_samples 50

Dependencies:
    pip install datasets pillow tqdm numpy pycocotools
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def decode_panoptic_mask(pan_seg, seg_id):
    pan_arr    = np.array(pan_seg.convert("RGB"))
    pan_id_arr = (pan_arr[:, :, 0].astype(np.int32) +
                  pan_arr[:, :, 1].astype(np.int32) * 256 +
                  pan_arr[:, :, 2].astype(np.int32) * 256 * 256)
    return (pan_id_arr == seg_id).astype(bool)


def mask_area_fraction(mask):
    return float(mask.sum()) / float(mask.size)


def run(args):
    from datasets import load_dataset

    out_root = Path(args.output_dir)
    img_dir  = out_root / "images"
    mask_dir = out_root / "masks"
    ann_dir  = out_root / "annotations"
    for d in [img_dir, mask_dir, ann_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Loading COCONut-PanCap from HuggingFace (streaming)...")
    print(f"Target: {args.num_samples} samples\n")
    ds = load_dataset("xdeng77/coconut_pancap", split="train", streaming=True)

    records = []
    skipped = 0
    count   = 0
    pbar    = tqdm(ds, desc="Downloading", total=args.num_samples * 4)

    for item in pbar:
        if count >= args.num_samples:
            break

        pbar.set_postfix({"saved": count, "skipped": skipped})

        try:
            image    = item["image"].convert("RGB")
            W, H     = image.size
            image_id = str(item.get("image_id", f"coconut_{count:05d}"))
            caption  = item.get("caption", "")
            segments = item.get("segments_info", [])
            pan_seg  = item.get("panoptic_seg", None)

            if pan_seg is None or not segments or len(segments) < 2:
                skipped += 1
                continue

            min_pixels = int(0.02 * W * H)
            valid_segs = [s for s in segments if s.get("area", 0) >= min_pixels]
            if len(valid_segs) < 2:
                skipped += 1
                continue

            img_path = img_dir / f"{image_id}.jpg"
            image.save(img_path, quality=90)

            img_mask_dir = mask_dir / image_id
            img_mask_dir.mkdir(exist_ok=True)

            saved_regions = []
            for i, seg in enumerate(valid_segs[:6]):
                seg_id    = seg["id"]
                seg_label = seg.get("category_name", seg.get("label", f"region_{i}"))
                mask      = decode_panoptic_mask(pan_seg, seg_id)
                area      = mask_area_fraction(mask)
                if area < 0.02:
                    continue

                mask_fname = f"mask_{i:02d}.png"
                mask_path  = img_mask_dir / mask_fname
                Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)

                region_caption = (seg.get("caption") or
                                  seg.get("description") or
                                  caption[:200] or "")

                saved_regions.append({
                    "mask_index":       i,
                    "mask_file":        str(mask_path.relative_to(out_root)),
                    "region_label":     seg_label,
                    "region_caption":   region_caption,
                    "area_fraction":    float(area),
                    "iou_score":        1.0,
                    "style_name":       "",
                    "style_reference":  "",
                    "instruction_text": "",
                    "instruction_ref":  "",
                })

            if len(saved_regions) < 2:
                skipped += 1
                img_path.unlink(missing_ok=True)
                continue

            records.append({
                "image_id":    image_id,
                "image_file":  str(img_path.relative_to(out_root)),
                "width":       W,
                "height":      H,
                "source":      "coconut_pancap",
                "num_regions": len(saved_regions),
                "regions":     saved_regions,
            })
            count += 1

            stub_path = ann_dir / "coconut_stub.json"
            with open(stub_path, "w") as f:
                json.dump(records, f, indent=2)

        except Exception:
            skipped += 1
            continue

    pbar.close()
    stub_path = ann_dir / "coconut_stub.json"
    with open(stub_path, "w") as f:
        json.dump(records, f, indent=2)

    counts = [r["num_regions"] for r in records]
    print(f"\nSaved   : {count} samples")
    print(f"Skipped : {skipped}")
    if counts:
        print(f"Regions : min={min(counts)}  max={max(counts)}  mean={sum(counts)/len(counts):.1f}")
    print(f"Stub    : {stub_path}")
    print(f"\nNext: python3 B_build_subset.py --stub {stub_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir",  default="../data/coconut_subset")
    p.add_argument("--num_samples", type=int, default=50)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
