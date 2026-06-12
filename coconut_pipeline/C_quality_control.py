"""
Step C: Quality Control, Corner-Case Tagging & Split (COCONut track)
=====================================================================
Extended from original 03_quality_control.py.
Adds two new corner case tags: small_object and cluttered_scene.

Usage:
    python3 C_quality_control.py \
        --draft_json  ../data/coconut_subset/annotations/benchmark_subset.json \
        --output_json ../data/coconut_subset/annotations/benchmark_subset_final.json
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def load_mask(path):
    return np.array(Image.open(path).convert("L")) > 127


def mask_iou(m1, m2):
    inter = int((m1 & m2).sum())
    union = int((m1 | m2).sum())
    return inter / union if union > 0 else 0.0


def is_contained(inner, outer, threshold=0.85):
    if inner.sum() == 0:
        return False
    return float((inner & outer).sum()) / float(inner.sum()) >= threshold


def check_quality(record, root, min_regions=2, overlap_thresh=0.5):
    regions = record.get("regions", [])
    if len(regions) < min_regions:
        return False, f"too_few_regions:{len(regions)}"
    masks = []
    for r in regions:
        p = root / r["mask_file"]
        if not p.exists():
            return False, f"missing_mask:{p.name}"
        masks.append(load_mask(str(p)))
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            iou       = mask_iou(masks[i], masks[j])
            contained = (is_contained(masks[i], masks[j]) or
                         is_contained(masks[j], masks[i]))
            if iou > overlap_thresh and not contained:
                return False, f"excessive_overlap:{i}_{j}_iou{iou:.2f}"
    return True, "ok"


def tag_corner_cases(record, root):
    tags    = []
    regions = record["regions"]
    masks   = [load_mask(str(root / r["mask_file"])) for r in regions]

    labels    = [r.get("region_label", "").strip().lower() for r in regions]
    non_empty = [l for l in labels if l]
    if len(non_empty) != len(set(non_empty)):
        tags.append("similar_entities")

    for i in range(len(masks)):
        for j in range(len(masks)):
            if i != j and is_contained(masks[i], masks[j]):
                tags.append("encompassed")
                break
        if "encompassed" in tags:
            break

    areas = [float(m.sum()) / float(m.size) for m in masks]
    if areas and max(areas) > 0.50:
        tags.append("background_heavy")
    if any(a < 0.03 for a in areas):
        tags.append("small_object")
    if len(regions) >= 5:
        tags.append("cluttered_scene")

    return tags


def stratified_split(records, train_frac=0.70, val_frac=0.10, seed=42):
    rng     = random.Random(seed)
    buckets = {}
    for r in records:
        key = ",".join(sorted(r.get("corner_case_tags", []))) or "none"
        buckets.setdefault(key, []).append(r)
    train, val, test = [], [], []
    for group in buckets.values():
        rng.shuffle(group)
        n       = len(group)
        n_train = max(1, int(n * train_frac))
        n_val   = max(0, int(n * val_frac))
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


def run(args):
    with open(args.draft_json) as f:
        records = json.load(f)

    root    = Path(args.draft_json).parent.parent
    kept    = []
    dropped = []

    for record in tqdm(records, desc="QC + tagging"):
        keep, reason = check_quality(record, root, args.min_regions, args.overlap_iou_thresh)
        if not keep:
            record["drop_reason"] = reason
            dropped.append(record)
            continue
        record["corner_case_tags"] = tag_corner_cases(record, root)
        kept.append(record)

    print(f"\nKept    : {len(kept)} / {len(records)}  ({len(dropped)} dropped)")

    tag_counts = {}
    for r in kept:
        for t in r.get("corner_case_tags", []):
            tag_counts[t] = tag_counts.get(t, 0) + 1
    print("Tags    :", tag_counts)

    train, val, test = stratified_split(kept)
    for r in train: r["split"] = "train"
    for r in val:   r["split"] = "val"
    for r in test:  r["split"] = "test"

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(train + val + test, f, indent=2)

    dropped_path = out_path.parent / "dropped_records.json"
    with open(dropped_path, "w") as f:
        json.dump(dropped, f, indent=2)

    print(f"Final   : {out_path}")
    print(f"Split   : train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"Next    : python3 D_export_subset.py --annotated_json {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--draft_json",          default="../data/coconut_subset/annotations/benchmark_subset.json")
    p.add_argument("--output_json",         default="../data/coconut_subset/annotations/benchmark_subset_final.json")
    p.add_argument("--min_regions",         type=int,   default=2)
    p.add_argument("--overlap_iou_thresh",  type=float, default=0.5)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
