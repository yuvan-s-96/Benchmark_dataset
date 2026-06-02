import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import SamModel, SamProcessor, pipeline


def load_image(path):
    return Image.open(path).convert("RGB")


def save_mask(mask, path):
    m = np.squeeze(mask)
    Image.fromarray((m.astype(bool) * 255).astype("uint8")).save(path)


def mask_area_fraction(mask):
    return float(mask.sum()) / float(mask.size)


def auto_segment(image, model, processor, device, min_area=0.02, max_masks=8):
    # Use a grid of point prompts across the image to generate multiple masks
    W, H = image.size
    grid_size = 4  # 4x4 = 16 points
    points = []
    for row in range(grid_size):
        for col in range(grid_size):
            x = int(W * (col + 0.5) / grid_size)
            y = int(H * (row + 0.5) / grid_size)
            points.append([x, y])

    seen_masks = []
    regions = []
    mask_index = 0

    for pt in points:
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
        best = int(scores.argmax())
        mask = np.squeeze(masks_t[0][0][best].numpy()).astype(bool)

        area = mask_area_fraction(mask)
        if area < min_area or area > 0.95:
            continue

        # Skip if too similar to an existing mask (IoU > 0.5)
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
            "mask_index": mask_index,
            "iou_score": float(scores[best]),
            "area_fraction": area,
            "mask": mask,
        })
        mask_index += 1

        if len(regions) >= max_masks:
            break

    regions.sort(key=lambda x: x["area_fraction"], reverse=True)
    return regions


def click_segment(image, model, processor, device, click_points):
    regions = []
    for idx, pt in enumerate(click_points):
        inputs = processor(images=image, input_points=[[pt]], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        masks_t = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        scores = outputs.iou_scores[0, 0].cpu()
        best = int(scores.argmax())
        mask = np.squeeze(masks_t[0][0][best].numpy()).astype(bool)
        regions.append({
            "mask_index": idx,
            "click_point": pt,
            "iou_score": float(scores[best]),
            "area_fraction": mask_area_fraction(mask),
            "mask": mask,
        })
    return regions


def process_directory(args):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    if device == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    cache_dir = os.environ.get("HF_HOME", None)
    print(f"Loading: {args.sam_model_id}")
    model = SamModel.from_pretrained(args.sam_model_id, cache_dir=cache_dir).to(device)
    processor = SamProcessor.from_pretrained(args.sam_model_id, cache_dir=cache_dir)

    image_dir = Path(args.image_dir)
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

    stub_path = output_dir.parent / "annotations" / "masks_stub.json"
    already_done = set()
    records = []
    if stub_path.exists() and args.resume:
        with open(stub_path) as f:
            records = json.load(f)
        already_done = {r["image_id"] for r in records}
        print(f"Resume : skipping {len(already_done)} already-processed images")

    for img_path in tqdm(image_paths, desc="Segmenting"):
        stem = img_path.stem
        if stem in already_done:
            continue
        image = load_image(str(img_path))
        img_out_dir = output_dir / stem
        img_out_dir.mkdir(exist_ok=True)

        if args.mode == "click" and stem in click_coords:
            pts = click_coords[stem]
            if not pts:
                tqdm.write(f"  [skip] {stem}: no clicks.")
                continue
            regions = click_segment(image, model, processor, device, pts)
        elif args.mode == "auto":
            regions = auto_segment(image, model, processor, device, args.min_area, args.max_masks)
        else:
            if stem in click_coords and click_coords[stem]:
                regions = click_segment(image, model, processor, device, click_coords[stem])
            else:
                regions = auto_segment(image, model, processor, device, args.min_area, args.max_masks)

        if not regions:
            tqdm.write(f"  [skip] {stem}: no usable regions.")
            continue

        saved_masks = []
        for region in regions:
            fname = f"mask_{region['mask_index']:02d}.png"
            save_mask(region["mask"], str(img_out_dir / fname))
            entry = {
                "mask_file": str((img_out_dir / fname).relative_to(output_dir.parent)),
                "mask_index": region["mask_index"],
                "iou_score": region["iou_score"],
                "area_fraction": region["area_fraction"],
                "region_label": "",
                "style_reference": "",
                "instruction": "",
            }
            if "click_point" in region:
                entry["click_point"] = region["click_point"]
            saved_masks.append(entry)

        records.append({
            "image_id": stem,
            "image_file": str(img_path.relative_to(image_dir.parent)),
            "width": image.width,
            "height": image.height,
            "num_regions": len(saved_masks),
            "regions": saved_masks,
        })

        stub_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stub_path, "w") as f:
            json.dump(records, f, indent=2)

    counts = [r["num_regions"] for r in records]
    print(f"\nDone. {len(records)} images processed.")
    if counts:
        print(f"Regions: min={min(counts)}  max={max(counts)}  mean={sum(counts)/len(counts):.1f}")
    print(f"Stub   : {stub_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",    default="../data/test_images")
    p.add_argument("--output_dir",   default="../data/masks")
    p.add_argument("--sam_model_id", default="facebook/sam-vit-large")
    p.add_argument("--mode", choices=["auto", "click", "mixed"], default="mixed")
    p.add_argument("--clicks_json",  default="../data/annotations/clicks.json")
    p.add_argument("--min_area",     type=float, default=0.02)
    p.add_argument("--max_masks",    type=int,   default=6)
    p.add_argument("--resume",       action="store_true", default=True)
    p.add_argument("--no_resume",    dest="resume", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    process_directory(parse_args())
