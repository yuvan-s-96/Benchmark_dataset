"""
Step 7 — InstructPix2Pix downstream evaluation
================================================
Tests whether attention-guided prompt curation produces better
regional stylisation output.

Pipeline:
1. Load COCO image
2. Feed instruction to InstructPix2Pix
3. Apply COCONut segmentation mask to stylised output
4. Compute CLIP similarity between masked region and WikiArt reference
5. Also compute non-region CLIP (background preservation)

Compare:
- baseline_A vs lora_A_tmplA vs baseline_C vs baseline_E

Sources compared:
  baseline_A   — base Mistral, Template A (most grounded)
  lora_A_tmplA — LoRA-A fine-tuned, Template A
  baseline_C   — base Mistral, Template C (highest quality)
  baseline_E   — base Mistral, Template E (highest CLIP)
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline
import clip

REFUSAL_PHRASES = ["i'm an ai","i cannot","language model","as an ai","i'm unable"]

def is_refusal(text):
    return any(p in text.lower() for p in REFUSAL_PHRASES)

def load_mask(pan_dir, pan_json, image_id, mask_index):
    """Load COCONut panoptic mask using segment_lookup.json."""
    import json as _json
    
    # Load segment lookup (cached)
    if not hasattr(load_mask, "_lookup"):
        lookup_path = Path(pan_dir) / "segment_lookup.json"
        with open(lookup_path) as f:
            load_mask._lookup = _json.load(f)
    
    key = f"{image_id}_{mask_index}"
    info = load_mask._lookup.get(key)
    if info is None:
        return None
    
    seg_id = info["segment_id"]
    
    # Load panoptic PNG
    pan_path = Path(pan_dir) / f"{str(image_id).zfill(12)}.png"
    if not pan_path.exists():
        return None
    
    pan_img = np.array(Image.open(pan_path).convert("RGB"))
    segment_map = (pan_img[:,:,0].astype(np.int32) +
                   pan_img[:,:,1].astype(np.int32) * 256 +
                   pan_img[:,:,2].astype(np.int32) * 65536)
    
    mask = (segment_map == seg_id).astype(np.uint8) * 255
    return mask
def masked_clip_score(clip_model, clip_preprocess, stylised_img, orig_img,
                      mask, style_ref_path, device):
    """
    Compute two CLIP scores:
    1. region_clip: similarity between STYLISED target region and WikiArt style reference
       (higher = better stylisation in the right region)
    2. nonregion_clip: similarity between STYLISED background and ORIGINAL background
       (higher = background better preserved — we want this HIGH)
    """
    sty_arr  = np.array(stylised_img)
    orig_arr = np.array(orig_img)
    mask_3ch = np.stack([mask, mask, mask], axis=2) / 255.0
    inv_mask = 1.0 - mask_3ch

    # 1. Stylised target region
    sty_region = (sty_arr * mask_3ch).astype(np.uint8)
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any() or not cols.any():
        return None, None
    rmin, rmax = np.where(rows)[0][[0,-1]]
    cmin, cmax = np.where(cols)[0][[0,-1]]
    region_crop = Image.fromarray(sty_region[rmin:rmax+1, cmin:cmax+1])

    # 2. Background: stylised vs original (preservation)
    sty_bg  = (sty_arr  * inv_mask).astype(np.uint8)
    orig_bg = (orig_arr * inv_mask).astype(np.uint8)
    sty_bg_img  = Image.fromarray(sty_bg)
    orig_bg_img = Image.fromarray(orig_bg)

    # Style reference image
    style_ref = Image.open(style_ref_path).convert("RGB")

    with torch.no_grad():
        region_t   = clip_preprocess(region_crop).unsqueeze(0).to(device)
        style_t    = clip_preprocess(style_ref).unsqueeze(0).to(device)
        sty_bg_t   = clip_preprocess(sty_bg_img).unsqueeze(0).to(device)
        orig_bg_t  = clip_preprocess(orig_bg_img).unsqueeze(0).to(device)

        region_feat  = clip_model.encode_image(region_t)
        style_feat   = clip_model.encode_image(style_t)
        sty_bg_feat  = clip_model.encode_image(sty_bg_t)
        orig_bg_feat = clip_model.encode_image(orig_bg_t)

        region_feat  = region_feat  / region_feat.norm(dim=-1, keepdim=True)
        style_feat   = style_feat   / style_feat.norm(dim=-1, keepdim=True)
        sty_bg_feat  = sty_bg_feat  / sty_bg_feat.norm(dim=-1, keepdim=True)
        orig_bg_feat = orig_bg_feat / orig_bg_feat.norm(dim=-1, keepdim=True)

        # Region CLIP: stylised region vs style reference (quality of stylisation)
        region_clip = float((region_feat * style_feat).sum())

        # Non-region CLIP: stylised background vs original background (preservation)
        nonregion_clip = float((sty_bg_feat * orig_bg_feat).sum())

    return region_clip, nonregion_clip


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load InstructPix2Pix
    print("Loading InstructPix2Pix...")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    print("Pipeline loaded")

    # Load CLIP
    print("Loading CLIP ViT-B/32...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    print("CLIP loaded\n")

    # Load template results
    sources = {}
    with open(args.results_json) as f:
        tmpl = json.load(f)
    for src in ["A","C","E"]:
        sources[f"baseline_{src}"] = tmpl["per_template"][src]

    with open(args.lora_a_json) as f:
        lora_a = json.load(f)
    sources["lora_A_tmplA"] = lora_a["per_template"]["A"]

    with open(args.lora_h_json) as f:
        lora_h = json.load(f)
    sources["lora_H_tmplA"] = lora_h["per_template"]["A"]

    for src in ["H","F","B","D","G","I"]:
        sources[f"baseline_{src}"] = tmpl["per_template"][src]

    # Load sample indices
    with open(args.indices_json) as f:
        idx_data = json.load(f)
    sample_indices = idx_data["indices"][:args.n_sample]

    anchor = sources["baseline_A"]
    print(f"Evaluating {len(sample_indices)} regions x {len(sources)} sources")
    print(f"Estimated runtime: ~{len(sample_indices)*len(sources)*15/60:.0f} minutes\n")

    results = {}
    summary = {}

    for src_name, regions in sources.items():
        print(f"\n{'='*55}")
        print(f"Source: {src_name}")
        print(f"{'='*55}")

        src_results = []
        region_clips, nonregion_clips = [], []
        n_skipped = 0

        for idx in sample_indices:
            if idx >= len(regions):
                n_skipped += 1
                continue

            r = regions[idx]
            image_id   = r["image_id"]
            mask_index = r["mask_index"]
            style      = r["style_name"]
            instruction = r.get("instruction","")

            if is_refusal(instruction) or len(instruction.split()) < 5:
                n_skipped += 1
                continue

            # Truncate instruction to 77 tokens max (CLIP limit)
            instruction = " ".join(instruction.split()[:60])

            # Load image
            img_path = Path(args.img_dir) / f"{image_id}.jpg"
            if not img_path.exists():
                img_path = Path(args.img_dir) / f"{str(image_id).zfill(12)}.jpg"
            if not img_path.exists():
                n_skipped += 1
                continue

            orig_img = Image.open(img_path).convert("RGB")
            orig_img = orig_img.resize((512, 512))

            # Run InstructPix2Pix
            try:
                with torch.autocast("cuda"):
                    out = pipe(
                        instruction,
                        image=orig_img,
                        num_inference_steps=20,
                        image_guidance_scale=1.5,
                        guidance_scale=7.5,
                    )
                stylised = out.images[0]
            except Exception as e:
                print(f"    Generation failed: {e}")
                n_skipped += 1
                continue

            # Load mask
            mask = load_mask(args.pan_dir, args.pan_json,
                             image_id, mask_index)

            # Style reference
            style_refs = list(Path(args.style_ref_dir).glob(
                f"{style.replace(' ','-')}/*.jpg"))
            if not style_refs:
                style_refs = list(Path(args.style_ref_dir).glob(
                    f"{style}/*.jpg"))
            if not style_refs:
                n_skipped += 1
                continue

            style_ref = str(style_refs[0])

            if mask is not None:
                mask_resized = np.array(
                    Image.fromarray(mask).resize((512,512),
                    Image.NEAREST))
                rc, bc = masked_clip_score(clip_model, clip_preprocess,
                                          stylised, orig_img,
                                          mask_resized,
                                          style_ref, device)
            else:
                # Fallback: whole image CLIP
                rc, bc = masked_clip_score(clip_model, clip_preprocess,
                                          stylised, orig_img,
                                          np.ones((512,512), dtype=np.uint8)*255,
                                          style_ref, device)

            if rc is not None:
                region_clips.append(rc)
                nonregion_clips.append(bc if bc is not None else 0)
                src_results.append({
                    "image_id":    image_id,
                    "mask_index":  mask_index,
                    "style":       style,
                    "instruction": instruction[:100],
                    "region_clip": round(rc, 4),
                    "nonregion_clip": round(bc, 4) if bc else None,
                })
                print(f"    [{len(src_results):02d}] {r.get('region_label','')[:30]:<30} "
                      f"region_clip={rc:.4f}  nonregion_clip={bc:.4f}")

        n_scored = len(src_results)
        mean_rc = np.mean(region_clips) if region_clips else 0
        mean_bc = np.mean(nonregion_clips) if nonregion_clips else 0

        print(f"\n  Scored: {n_scored}  Skipped: {n_skipped}")
        print(f"  Region CLIP mean: {mean_rc:.4f}")
        print(f"  Background CLIP mean: {mean_bc:.4f}")

        results[src_name] = src_results
        summary[src_name] = {
            "region_clip_mean": round(float(mean_rc), 4),
            "nonregion_clip_mean":     round(float(mean_bc), 4),
            "n_scored":         n_scored,
            "n_skipped":        n_skipped,
        }

    # Final table
    print(f"\n{'='*65}")
    print("FINAL — Region-masked CLIP comparison")
    print(f"{'='*65}")
    print(f"\n  {'Source':<20} {'Region CLIP':>12} {'NonReg CLIP (higher=better preservation)':>10} {'n':>5}")
    print(f"  {'-'*50}")
    for src in sorted(summary, key=lambda s: -summary[s]["region_clip_mean"]):
        s = summary[src]
        print(f"  {src:<20} {s['region_clip_mean']:>12.4f} "
              f"{s['nonregion_clip_mean']:>10.4f} {s['n_scored']:>5}")

    out = {
        "model": "timbrooks/instruct-pix2pix",
        "clip_model": "ViT-B/32",
        "n_sample": len(sample_indices),
        "summary": summary,
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_json",
        default="../results/template_comparison_979.json")
    p.add_argument("--lora_a_json",
        default="../results/attention_lora_A.json")
    p.add_argument("--lora_h_json",
        default="../results/attention_lora_H.json")
    p.add_argument("--indices_json",
        default="../results/human_rating_indices.json")
    p.add_argument("--img_dir",
        default="../../data/coconut_subset/images")
    p.add_argument("--pan_dir",
        default="/mnt/fast1/yvs23/annotations/panoptic_train2017")
    p.add_argument("--pan_json",
        default="/mnt/fast1/yvs23/annotations/panoptic_train2017.json")
    p.add_argument("--style_ref_dir",
        default="../../data/style_references")
    p.add_argument("--output",
        default="../results/instruct_pix2pix_eval.json")
    p.add_argument("--n_sample", type=int, default=30)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
