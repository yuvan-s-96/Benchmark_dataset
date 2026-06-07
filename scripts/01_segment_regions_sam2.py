import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def load_image(path):
    return Image.open(path).convert("RGB")


def save_mask(mask, path):
    m = np.squeeze(mask)
    Image.fromarray((m.astype(bool) * 255).astype("uint8")).save(path)


def mask_area_fraction(mask):
    return float(mask.sum()) / float(mask.size)


def auto_segment_sam2(predictor, image, min_area=0.02, max_masks=6):
    img_np = np.array(image)
    predictor.set_image(img_np)

    W, H = image.size
    grid_size = 4
    points = [
        [int(W * (col + 0.5) / grid_size), int(H * (row + 0.5) / grid_size)]
        for row in range(grid_size)
        for col in range(grid_size)
    ]

    seen_masks = []
    regions = []
    mask_index = 0

    for pt in points:
        point_coords = np.array([[pt]], dtype=np.float32)
        point_labels = np.array([[1]], dtype=np.int32)

        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        best = int(scores.argmax())
        mask = masks[best].astype(bool)

        area = mask_area_fraction(mask)
        if area < min_area or area > 0.95:
            continue

        duplicate = False
        for existing in seen_masks:
            inter = int((mask & existing).sum())
            union = int((mask | existing).sum())
            if union > 0 and inter / union > 0.5:
                duplicate = True
                break
        if duplicate:
            continue

        seen_masks.append(mask)
        regions.append({
            "mask_index":    mask_index,
            "iou_score":     float(scores[best]),
            "area_fraction": area,
            "mask":          mask,
        })
        mask_index += 1

        if len(regions) >= max_masks:
            break

    regions.sort(key=lambda x: x["area_fraction"], reverse=True)
    return regions


def click_segment_sam2(predictor, image, click_points):
    img_np = np.array(image)
    predictor.set_image(img_np)

    regions = []
    for idx, pt in enumerate(click_points):
        point_coords = np.array([[pt]], dtype=np.float32)
        point_labels = np.array([[1]], dtype=np.int32)

        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        best = int(scores.argmax())
        mask = masks[best].astype(bool)

        regions.append({
            "mask_index":    idx,
            "click_point":   pt,
            "iou_score":     float(scores[best]),
            "area_fraction": mask_area_fraction(mask),
            "mask":          mask,
        })
    return regions


def process_directory(args):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    if device == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"Loading: {args.checkpoint}")
    sam2_model = build_sam2(args.config, args.checkpoint, device=device)
    predictor  = SAM2ImagePredictor(sam2_model)
    print("SAM2 loaded.")

    image_dir  = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for ext in ["jpg", "jpeg", "png", "JPG", "PNG"]
        for p in image_dir.glob(f"*.{ext}")
    )
    print(f"Images : {len(image_paths)}")

    click_coords = {}
    if args.clicks_json and Path(args.clicks_json).exists():
        with open(args.clicks_json) as f:
            click_coords = json.load(f)
        print(f"Clicks : loaded for {len(click_coords)} images")

    stub_path = output_dir.parent / "annotations" / "masks_stub_sam2.json"
    already_done = set()
    records = []
    if stub_path.exists() and args.resume:
        with open(stub_path) as f:
            records = json.load(f)
        already_done = {r["image_id"] for r in records}
        print(f"Resume : skipping {len(already_done)} already-processed images")

    for img_path in tqdm(image_paths, desc="Segmenting (SAM2)"):
        stem = img_path.stem
        if stem in already_done:
            continue

        image       = load_image(str(img_path))
        img_out_dir = output_dir / stem
        img_out_dir.mkdir(exist_ok=True)

        try:
            if args.mode == "click" and stem in click_coords:
                pts = click_coords[stem]
                if not pts:
                    tqdm.write(f"  [skip] {stem}: no clicks.")
                    continue
                regions = click_segment_sam2(predictor, image, pts)
            elif args.mode == "auto":
                regions = auto_segment_sam2(predictor, image,
                                            args.min_area, args.max_masks)
            else:  # mixed
                if stem in click_coords and click_coords[stem]:
                    regions = click_segment_sam2(predictor, image,
                                                 click_coords[stem])
                else:
                    regions = auto_segment_sam2(predictor, image,
                                                args.min_area, args.max_masks)
        except Exception as e:
            tqdm.write(f"  [error] {stem}: {e}")
            continue

        if not regions:
            tqdm.write(f"  [skip] {stem}: no usable regions.")
            continue

        saved_masks = []
        for region in regions:
            fname = f"mask_{region[chr(109)+chr(97)+chr(115)+chr(107)+chr(95)+chr(105)+chr(110)+chr(100)+chr(101)+chr(120)]:02d}.png"
            save_mask(region["mask"], str(img_out_dir / fname))
            entry = {
                "mask_file":       str((img_out_dir / fname).relative_to(output_dir.parent)),
                "mask_index":      region["mask_index"],
                "iou_score":       region["iou_score"],
                "area_fraction":   region["area_fraction"],
                "region_label":    "",
                "style_reference": "",
                "instruction":     "",
            }
            if "click_point" in region:
                entry["click_point"] = region["click_point"]
            saved_masks.append(entry)

        records.append({
            "image_id":    stem,
            "image_file":  str(img_path.relative_to(image_dir.parent)),
            "width":       image.width,
            "height":      image.height,
            "num_regions": len(saved_masks),
            "sam_version": "sam2",
            "regions":     saved_masks,
        })

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
    p = argparse.ArgumentParser(
        description="SAM2 regional segmentation"
    )
    p.add_argument("--image_dir",   default="../data/test_images")
    p.add_argument("--output_dir",  default="../data/masks_sam2")
    p.add_argument("--checkpoint",  default="../checkpoints/sam2.1_hiera_large.pt")
    p.add_argument("--config",      default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--mode", choices=["auto", "click", "mixed"], default="mixed")
    p.add_argument("--clicks_json", default="../data/annotations/clicks.json")
    p.add_argument("--min_area",    type=float, default=0.02)
    p.add_argument("--max_masks",   type=int,   default=6)
    p.add_argument("--resume",      action="store_true", default=True)
    p.add_argument("--no_resume",   dest="resume", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    process_directory(parse_args())
