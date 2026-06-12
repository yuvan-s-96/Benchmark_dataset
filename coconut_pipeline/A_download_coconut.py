"""
Step A: Download COCONut-PanCap Subset (fixed parser)
======================================================
Parses the narrative text format: <id: region label> embedded in captions.
Downloads content images from COCO and extracts region descriptions.

Since COCONut-PanCap does not provide separate mask images in this format,
we use the region labels from the narrative and generate masks using SAM2.
The captions provide rich region descriptions which is the key value-add.

Usage:
    python3 A_download_coconut.py \
        --output_dir  ../data/coconut_subset \
        --num_samples 50
"""

import argparse
import json
import re
import requests
from pathlib import Path

from PIL import Image
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Parse COCONut-PanCap narrative format
# ─────────────────────────────────────────────────────────────────────────────

def parse_narrative(txt):
    """
    Extract regions from narrative text.
    Format: <id: label> or <id1,id2: label> embedded in sentence.
    
    Returns list of dicts: {ids, label, context_sentence}
    """
    regions = []
    seen_labels = set()
    
    # Find all <...> tags
    pattern = r"<([\d,]+):\s*([^>]+)>"
    
    # Split text into sentences for context
    sentences = re.split(r"(?<=[.!?])\s+", txt)
    
    for sentence in sentences:
        for match in re.finditer(pattern, sentence):
            ids_str = match.group(1)
            label   = match.group(2).strip().lower()
            
            if label in seen_labels:
                continue
            seen_labels.add(label)
            
            ids = [int(x) for x in ids_str.split(",") if x.strip().isdigit()]
            
            # Use the sentence as the region caption
            # Strip all <> tags from the sentence for clean caption
            clean_sentence = re.sub(r"<[\d,]+:\s*([^>]+)>", r"\1", sentence).strip()
            
            regions.append({
                "ids":     ids,
                "label":   label,
                "caption": clean_sentence,
            })
    
    return regions


def download_coco_image(image_id, dest_path, session):
    """Download a COCO image by its ID."""
    # COCO val2017 image URL format
    fname = f"{int(image_id):012d}.jpg"
    url   = f"http://images.cocodataset.org/val2017/{fname}"
    try:
        r = session.get(url, timeout=20)
        if r.status_code == 200 and "image" in r.headers.get("content-type", ""):
            img = Image.open(__import__("io").BytesIO(r.content)).convert("RGB")
            img.save(dest_path, quality=90)
            return True
    except Exception:
        pass
    # Try train2017
    url = f"http://images.cocodataset.org/train2017/{fname}"
    try:
        r = session.get(url, timeout=20)
        if r.status_code == 200 and "image" in r.headers.get("content-type", ""):
            img = Image.open(__import__("io").BytesIO(r.content)).convert("RGB")
            img.save(dest_path, quality=90)
            return True
    except Exception:
        pass
    return False


def run(args):
    from datasets import load_dataset

    out_root = Path(args.output_dir)
    img_dir  = out_root / "images"
    ann_dir  = out_root / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    print("Loading COCONut-PanCap from HuggingFace (streaming)...")
    print(f"Target: {args.num_samples} samples\n")
    ds      = load_dataset("xdeng77/coconut_pancap", split="train", streaming=True)
    session = requests.Session()
    session.headers["User-Agent"] = "benchmark-research/1.0 (bath.ac.uk)"

    records = []
    skipped = 0
    count   = 0
    pbar    = tqdm(ds, desc="Processing", total=args.num_samples * 5)

    for item in pbar:
        if count >= args.num_samples:
            break

        pbar.set_postfix({"saved": count, "skipped": skipped})

        try:
            txt      = item.get("txt", "")
            key      = item.get("__key__", "")
            # key format: caption_train2017/000000208220
            image_id = key.split("/")[-1].lstrip("0") or "0"
            if not image_id:
                image_id = "0"

            if not txt or not image_id:
                skipped += 1
                continue

            # Parse regions from narrative
            regions = parse_narrative(txt)
            
            # Need at least 2 distinct regions
            if len(regions) < 2:
                skipped += 1
                continue

            # Download COCO image
            img_path = img_dir / f"{image_id}.jpg"
            if not img_path.exists():
                ok = download_coco_image(image_id, img_path, session)
                if not ok:
                    skipped += 1
                    continue

            # Get image dimensions
            try:
                img = Image.open(img_path)
                W, H = img.size
            except Exception:
                skipped += 1
                continue

            # Build region records (no masks yet — SAM2 will generate them)
            saved_regions = []
            for i, reg in enumerate(regions[:6]):
                saved_regions.append({
                    "mask_index":       i,
                    "mask_file":        "",   # filled by SAM2 step
                    "region_label":     reg["label"],
                    "region_caption":   reg["caption"],
                    "region_ids":       reg["ids"],
                    "area_fraction":    0.0,
                    "iou_score":        0.0,
                    "style_name":       "",
                    "style_reference":  "",
                    "instruction_text": "",
                    "instruction_ref":  "",
                })

            records.append({
                "image_id":    image_id,
                "image_file":  str(img_path.relative_to(out_root)),
                "narrative":   txt,
                "width":       W,
                "height":      H,
                "source":      "coconut_pancap",
                "num_regions": len(saved_regions),
                "regions":     saved_regions,
            })
            count += 1

            # Incremental save
            stub_path = ann_dir / "coconut_stub.json"
            with open(stub_path, "w") as f:
                json.dump(records, f, indent=2)

        except Exception as e:
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
    
    if records:
        print(f"\nSample:")
        r = records[0]
        print(f"  image_id : {r['image_id']}")
        for reg in r["regions"][:3]:
            print(f"  region   : {reg['region_label']}")
            print(f"    caption: {reg['region_caption'][:80]}")

    print(f"\nNote: mask_file fields are empty.")
    print(f"Next: run SAM2 segmentation to generate masks, then B_build_subset.py")
    print(f"  python3 ../scripts/01_segment_regions_sam2.py \\")
    print(f"      --image_dir  {img_dir} \\")
    print(f"      --output_dir {out_root}/masks \\")
    print(f"      --mode auto --no_resume")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir",  default="../data/coconut_subset")
    p.add_argument("--num_samples", type=int, default=50)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
