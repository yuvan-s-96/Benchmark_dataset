"""
A3_extract_clicks.py
====================
Track 2: Extracts click coordinates for each region label from COCO
instance annotations. Maps COCONut region labels to COCO category names,
finds matching annotations, and computes centroid click points.

This gives SAM2 semantically meaningful prompts aligned to the actual
described regions rather than a uniform grid.

Usage:
    python3 A3_extract_clicks.py \
        --stub          ../data/coconut_subset/annotations/coconut_stub.json \
        --coco_ann      ../data/content_images/annotations/instances_val2017.json \
        --output_clicks ../data/coconut_subset/annotations/clicks_coconut.json
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm


# Mapping from COCONut label words to COCO category names
LABEL_TO_COCO = {
    "person": "person", "people": "person", "man": "person", "woman": "person",
    "child": "person", "boy": "person", "girl": "person", "rider": "person",
    "car": "car", "cars": "car", "vehicle": "car",
    "truck": "truck", "trucks": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle", "motorbike": "motorcycle",
    "bicycle": "bicycle", "bike": "bicycle",
    "dog": "dog", "cat": "cat", "horse": "horse", "cow": "cow",
    "elephant": "elephant", "bear": "bear", "zebra": "zebra", "giraffe": "giraffe",
    "bird": "bird",
    "chair": "chair", "bench": "bench", "couch": "couch", "sofa": "couch",
    "table": "dining table", "desk": "dining table",
    "bottle": "bottle", "cup": "cup", "bowl": "bowl",
    "tree": "potted plant", "trees": "potted plant",
    "tv": "tv", "television": "tv", "monitor": "tv",
    "laptop": "laptop", "keyboard": "keyboard", "mouse": "mouse",
    "book": "book", "clock": "clock",
    "backpack": "backpack", "bag": "handbag",
    "umbrella": "umbrella",
    "sports ball": "sports ball", "ball": "sports ball",
    "kite": "kite", "frisbee": "frisbee",
    "skateboard": "skateboard", "surfboard": "surfboard",
    "tennis racket": "tennis racket",
}


def get_centroid(segmentation, bbox):
    """Get centroid from bbox as fallback (fast and reliable)."""
    x, y, w, h = bbox
    return [int(x + w / 2), int(y + h / 2)]


def run(args):
    # Load COCO annotations
    coco_ann_path = Path(args.coco_ann)
    if not coco_ann_path.exists():
        print(f"[warn] COCO annotations not found at {coco_ann_path}")
        print("Download with:")
        print("  cd ~/Benchmark_dataset/data/content_images")
        print("  wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
        print("  unzip annotations_trainval2017.zip")
        return

    print("Loading COCO annotations...")
    with open(coco_ann_path) as f:
        coco = json.load(f)

    # Build lookups
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    name_to_cat_id = {c["name"]: c["id"] for c in coco["categories"]}

    # image_id -> list of annotations
    img_anns = {}
    for ann in coco["annotations"]:
        img_id = str(ann["image_id"])
        img_anns.setdefault(img_id, []).append(ann)

    # Load coconut stub
    with open(args.stub) as f:
        records = json.load(f)

    clicks_dict = {}
    matched     = 0
    unmatched   = 0

    for record in tqdm(records, desc="Extracting clicks"):
        image_id = record["image_id"]
        anns     = img_anns.get(image_id, [])

        clicks = []
        for region in record["regions"]:
            label = region["region_label"].lower().strip()

            # Try to find matching COCO category
            coco_cat = None
            for word in label.split():
                if word in LABEL_TO_COCO:
                    coco_cat = LABEL_TO_COCO[word]
                    break
            if not coco_cat:
                # Try partial match
                for key, val in LABEL_TO_COCO.items():
                    if key in label:
                        coco_cat = val
                        break

            if coco_cat and coco_cat in name_to_cat_id:
                cat_id = name_to_cat_id[coco_cat]
                # Find annotation with matching category, largest area
                matching = [a for a in anns if a["category_id"] == cat_id]
                if matching:
                    best = max(matching, key=lambda a: a["area"])
                    pt   = get_centroid(best["segmentation"], best["bbox"])
                    clicks.append(pt)
                    matched += 1
                    continue

            # Fallback: use region index to place click in image grid
            W, H = record["width"], record["height"]
            idx  = region["mask_index"]
            grid = 3
            row  = idx // grid
            col  = idx  % grid
            pt   = [int(W * (col + 0.5) / grid), int(H * (row + 0.5) / grid)]
            clicks.append(pt)
            unmatched += 1

        clicks_dict[image_id] = clicks

    out_path = Path(args.output_clicks)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(clicks_dict, f, indent=2)

    print(f"\nMatched   : {matched} regions to COCO categories")
    print(f"Fallback  : {unmatched} regions used grid position")
    print(f"Output    : {out_path}")
    print(f"\nNext:")
    print(f"  python3 ../scripts/01_segment_regions_sam2.py \\")
    print(f"      --image_dir  ../data/coconut_subset/images \\")
    print(f"      --output_dir ../data/coconut_subset/masks_click \\")
    print(f"      --clicks_json {out_path} \\")
    print(f"      --mode click --no_resume")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stub",
                   default="../data/coconut_subset/annotations/coconut_stub.json")
    p.add_argument("--coco_ann",
                   default="../data/content_images/annotations/instances_train2017.json")
    p.add_argument("--output_clicks",
                   default="../data/coconut_subset/annotations/clicks_coconut.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
