"""
Step 1: Regional Segmentation using SAM (Segment Anything Model)
================================================================
Produces per-region binary mask PNGs for each content image.
Region count is DYNAMIC — one region per click (click mode) or
auto-detected (auto mode), with no fixed upper cap.

Recommended model by hardware:
    RTX 3050 4 GB  →  facebook/sam-vit-base   (~1.5 GB VRAM)
    RTX 2080 8 GB  →  facebook/sam-vit-large  (~2.5 GB VRAM)
    A100 / V100    →  facebook/sam-vit-huge   (~4.5 GB VRAM)

Usage:
    # click mode (recommended — uses clicks.json from Step 0c)
    python 01_segment_regions.py \
        --image_dir  ../data/content_images \
        --output_dir ../data/masks \
        --mode       mixed \
        --clicks_json ../data/annotations/clicks.json \
        --sam_model_id facebook/sam-vit-large

    # auto mode (no clicks needed)
    python 01_segment_regions.py --mode auto --min_area 0.02

Crash-safe: writes masks_stub.json after every image.
Resumes automatically on rerun (--resume is on by default).

Dependencies:
    pip install transformers torch torchvision pillow tqdm numpy
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import SamModel, SamProcessor


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_mask(mask: np.ndarray, path: str) -> None:
    Image.fromarray((mask * 255).astype(np.uint8)).save(path)


def mask_area_fraction(mask: np.ndarray) -> float:
    return float(mask.sum()) / float(mask.size)


# ─────────────────────────────────────────────────────────────────────────────
# Automatic mask generation
# ─────────────────────────────────────────────────────────────────────────────

def auto_segment(image: Image.Image, model: SamModel,
                 processor: SamProcessor, device: str,
                 min_area: float = 0.02, max_masks: int = 8) -> list[dict]:
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    masks_t = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    iou_scores = outputs.iou_scores.cpu().squeeze()
    masks_np = masks_t[0].numpy().astype(bool)

    regions = []
    for i, (m, score) in enumerate(zip(masks_np, iou_scores)):
        area = mask_area_fraction(m)
        if area < min_area:
            continue
        regions.append({"mask_index": i, "iou_score": float(score),
                        "area_fraction": area, "mask": m})

    regions.sort(key=lambda x: x["area_fraction"], reverse=True)
    return regions[:max_masks]


# ─────────────────────────────────────────────────────────────────────────────
# Click-prompt segmentation — one mask per click, fully dynamic
# ─────────────────────────────────────────────────────────────────────────────

def click_segment(image: Image.Image, model: SamModel,
                  processor: SamProcessor, device: str,
                  click_points: list[list[int]]) -> list[dict]:
    regions = []
    for idx, pt in enumerate(click_points):
        inputs = processor(images=image, input_points=[[pt]],
                           return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        masks_t = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        scores = outputs.iou_scores[0, 0].cpu()
        best   = int(scores.argmax())
        mask   = masks_t[0][0][best].numpy().astype(bool)

        regions.append({"mask_index": idx, "click_point": pt,
                        "iou_score": float(scores[best]),
                        "area_fraction": mask_area_fraction(mask),
                        "mask": mask})
    return regions


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def process_directory(args):
    # Reduce VRAM fragmentation (helpful on 4–8 GB cards)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    if device == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("[warn] No GPU — SAM on CPU is ~30 s/image.")

    cache_dir = os.environ.get("HF_HOME", None)
    print(f"Loading: {args.sam_model_id}")
    model     = SamModel.from_pretrained(args.sam_model_id,
                                         cache_dir=cache_dir).to(device)
    processor = SamProcessor.from_pretrained(args.sam_model_id,
                                              cache_dir=cache_dir)

    image_dir  = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for ext in ["jpg", "jpeg", "png", "JPG", "PNG"]
        for p in image_dir.glob(f"*.{ext}")
    )
    print(f"Images : {len(image_paths)}")

    click_coords: dict[str, list] = {}
    if args.clicks_json and Path(args.clicks_json).exists():
        with open(args.clicks_json) as f:
            click_coords = json.load(f)
        print(f"Clicks : loaded for {len(click_coords)} images")
    elif args.mode in ("click", "mixed"):
        print("[info] No clicks.json found — auto mode will be used for all images.")

    # Resume support
    stub_path = output_dir.parent / "annotations" / "masks_stub.json"
    already_done: set[str] = set()
    records: list[dict] = []
    if stub_path.exists() and args.resume:
        with open(stub_path) as f:
            records = json.load(f)
        already_done = {r["image_id"] for r in records}
        print(f"Resume : skipping {len(already_done)} already-processed images")

    for img_path in tqdm(image_paths, desc="Segmenting"):
        stem = img_path.stem
        if stem in already_done:
            continue

        image       = load_image(str(img_path))
        img_out_dir = output_dir / stem
        img_out_dir.mkdir(exist_ok=True)

        if args.mode == "click" and stem in click_coords:
            pts = click_coords[stem]
            if not pts:
                tqdm.write(f"  [skip] {stem}: no clicks recorded.")
                continue
            regions = click_segment(image, model, processor, device, pts)
        elif args.mode == "auto":
            regions = auto_segment(image, model, processor, device,
                                   args.min_area, args.max_masks)
        else:   # mixed
            if stem in click_coords and click_coords[stem]:
                regions = click_segment(image, model, processor, device,
                                        click_coords[stem])
            else:
                regions = auto_segment(image, model, processor, device,
                                       args.min_area, args.max_masks)

        if not regions:
            tqdm.write(f"  [skip] {stem}: no usable regions found.")
            continue

        saved_masks = []
        for region in regions:
            fname = f"mask_{region['mask_index']:02d}.png"
            save_mask(region["mask"], str(img_out_dir / fname))
            entry = {
                "mask_file":     str((img_out_dir / fname).relative_to(output_dir.parent)),
                "mask_index":    region["mask_index"],
                "iou_score":     region["iou_score"],
                "area_fraction": region["area_fraction"],
                "region_label":  "",
                "style_reference": "",
                "instruction":   "",
            }
            if "click_point" in region:
                entry["click_point"] = region["click_point"]
            saved_masks.append(entry)

        records.append({
            "image_id":   stem,
            "image_file": str(img_path.relative_to(image_dir.parent)),
            "width":      image.width,
            "height":     image.height,
            "num_regions": len(saved_masks),
            "regions":    saved_masks,
        })

        # Incremental write — crash-safe
        stub_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stub_path, "w") as f:
            json.dump(records, f, indent=2)

    counts = [r["num_regions"] for r in records]
    print(f"\nDone. {len(records)} images processed.")
    if counts:
        print(f"Regions: min={min(counts)}  max={max(counts)}  "
              f"mean={sum(counts)/len(counts):.1f}")
    print(f"Stub   : {stub_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",     default="../data/content_images")
    p.add_argument("--output_dir",    default="../data/masks")
    p.add_argument("--sam_model_id",  default="facebook/sam-vit-large",
                   help="sam-vit-base (4GB), sam-vit-large (8GB), sam-vit-huge (A100)")
    p.add_argument("--mode", choices=["auto", "click", "mixed"], default="mixed")
    p.add_argument("--clicks_json",   default="../data/annotations/clicks.json")
    p.add_argument("--min_area",      type=float, default=0.02)
    p.add_argument("--max_masks",     type=int,   default=8)
    p.add_argument("--resume",        action="store_true", default=True)
    p.add_argument("--no_resume",     dest="resume", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    process_directory(parse_args())
