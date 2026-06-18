"""
F_instruction_quality.py — Instruction Quality Evaluation
==========================================================
Evaluates instruction_text quality across both tracks using three
automated metrics — no human labelling needed.

Metrics:
  1. CLIP alignment    — cosine sim between instruction text embedding
                         and style reference image embedding
                         (higher = instruction matches the style)
  2. Word count        — length of instruction in words
                         (stub ~12w, GGUF ~25w)
  3. Label coverage    — does the instruction mention the region label?
                         (binary: 1 = yes, 0 = no)
  4. Visual specificity— count of visual descriptor words
                         (brushstroke, colour, texture, light, etc.)

Usage:
    python3 F_instruction_quality.py \
        --auto_json  ../data/coconut_subset/annotations/subset_auto_final_gguf.json \
        --click_json ../data/coconut_subset/annotations/subset_click_final_gguf.json \
        --output     ../data/coconut_subset/annotations/instruction_quality.json

Dependencies (already on ogg):
    transformers, torch, pillow
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


# ─────────────────────────────────────────────────────────────────────────────
# Visual descriptor vocabulary
# ─────────────────────────────────────────────────────────────────────────────

VISUAL_DESCRIPTORS = {
    # Brushwork
    "brushstroke","brushwork","stroke","impasto","gesture","gestural",
    # Colour
    "colour","color","hue","tone","palette","vivid","vibrant","muted",
    "pastel","saturated","monochromatic","chromatic","pigment",
    # Light
    "light","lighting","luminous","luminosity","shadow","contrast",
    "highlight","glow","radiance","shimmer",
    # Texture
    "texture","textured","grainy","smooth","rough","coarse",
    # Style-specific
    "pointillist","impressionist","cubist","baroque","expressionist",
    "abstract","geometric","organic","flowing","linear","angular",
    # Atmosphere
    "dreamy","dramatic","serene","atmospheric","depth","layer",
}


# ─────────────────────────────────────────────────────────────────────────────
# Metric functions
# ─────────────────────────────────────────────────────────────────────────────

def word_count(text):
    return len(text.strip().split()) if text.strip() else 0


def label_coverage(instruction, region_label):
    """Check if any significant word from region_label appears in instruction."""
    if not instruction or not region_label:
        return 0
    label_words = set(w.lower().strip(".,") for w in region_label.split()
                      if len(w) > 2)
    instr_lower = instruction.lower()
    return 1 if any(w in instr_lower for w in label_words) else 0


def visual_specificity(instruction):
    """Count visual descriptor words in instruction."""
    if not instruction:
        return 0
    words = set(w.lower().strip(".,;:!?") for w in instruction.split())
    return len(words & VISUAL_DESCRIPTORS)


@torch.no_grad()
def clip_alignment(model, processor, instruction, style_path, device):
    """
    CLIP cosine similarity between instruction text and style reference image.
    Higher = instruction is more semantically aligned with the style.
    """
    try:
        image = Image.open(style_path).convert("RGB")
        image.thumbnail((224, 224))

        inputs = processor(
            text=[instruction],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(device)

        outputs      = model(**inputs)
        text_emb     = outputs.text_embeds  / outputs.text_embeds.norm(dim=-1, keepdim=True)
        image_emb    = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        similarity   = (text_emb * image_emb).sum(dim=-1).item()
        return round(float(similarity), 4)

    except Exception as e:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_track(records, track_name, model, processor, device, data_root):
    results   = []
    wc_all    = []
    lc_all    = []
    vs_all    = []
    clip_all  = []

    for record in tqdm(records, desc=f"Track {track_name}"):
        for region in record["regions"]:
            instr  = region.get("instruction_text", "")
            label  = region.get("region_label",   "")
            style  = region.get("style_reference", "")

            wc = word_count(instr)
            lc = label_coverage(instr, label)
            vs = visual_specificity(instr)

            # CLIP alignment
            ca = None
            if style and Path(data_root / style).exists():
                ca = clip_alignment(model, processor, instr,
                                    str(data_root / style), device)

            results.append({
                "image_id":       record["image_id"],
                "mask_index":     region["mask_index"],
                "region_label":   label,
                "style_name":     region.get("style_name",""),
                "instruction":    instr,
                "word_count":     wc,
                "label_coverage": lc,
                "visual_specificity": vs,
                "clip_alignment": ca,
            })

            wc_all.append(wc)
            lc_all.append(lc)
            vs_all.append(vs)
            if ca is not None:
                clip_all.append(ca)

    summary = {
        "n": len(results),
        "word_count": {
            "mean":   round(float(np.mean(wc_all)),   2),
            "median": round(float(np.median(wc_all)), 2),
            "min":    int(min(wc_all)),
            "max":    int(max(wc_all)),
        },
        "label_coverage": {
            "mean":    round(float(np.mean(lc_all)), 3),
            "percent": round(float(np.mean(lc_all)) * 100, 1),
        },
        "visual_specificity": {
            "mean":   round(float(np.mean(vs_all)),   2),
            "median": round(float(np.median(vs_all)), 2),
            "max":    int(max(vs_all)),
        },
        "clip_alignment": {
            "mean":    round(float(np.mean(clip_all)),   4) if clip_all else None,
            "median":  round(float(np.median(clip_all)), 4) if clip_all else None,
            "min":     round(float(min(clip_all)),       4) if clip_all else None,
            "max":     round(float(max(clip_all)),       4) if clip_all else None,
            "n":       len(clip_all),
        },
    }
    return summary, results


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Loading CLIP (openai/clip-vit-base-patch32)...")

    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    print("CLIP loaded.\n")

    output = {}

    for track_name, json_path in [
        ("auto",  args.auto_json),
        ("click", args.click_json),
    ]:
        if not Path(json_path).exists():
            print(f"[skip] {json_path} not found")
            continue

        with open(json_path) as f:
            records = json.load(f)

        data_root = Path('.')

        summary, results = evaluate_track(
            records, track_name, model, processor, device, data_root)
        output[track_name] = {"summary": summary, "per_region": results}

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
        label = "Track 1 — Auto GGUF" if track_name == "auto" else "Track 2 — Click GGUF"
        print(f"\n{'='*55}")
        print(f"{label}")
        print(f"{'='*55}")
        print(f"  Regions evaluated   : {s['n']}")
        print(f"\n  Word count:")
        print(f"    Mean              : {s['word_count']['mean']}")
        print(f"    Median            : {s['word_count']['median']}")
        print(f"    Min / Max         : {s['word_count']['min']} / {s['word_count']['max']}")
        print(f"\n  Label coverage:")
        print(f"    Mentions region   : {s['label_coverage']['percent']}%")
        print(f"\n  Visual specificity:")
        print(f"    Mean descriptors  : {s['visual_specificity']['mean']}")
        print(f"    Max descriptors   : {s['visual_specificity']['max']}")
        print(f"\n  CLIP alignment (text vs style image):")
        c = s["clip_alignment"]
        if c["mean"] is not None:
            print(f"    Mean              : {c['mean']}")
            print(f"    Median            : {c['median']}")
            print(f"    Min / Max         : {c['min']} / {c['max']}")
            print(f"    Regions scored    : {c['n']}")
        else:
            print(f"    No style images found")

    print(f"\nOutput: {args.output}")
    print(f"\nNext: share instruction_quality.json with supervisors as part of meeting prep.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--auto_json",
                   default="../data/coconut_subset/annotations/subset_auto_final_gguf.json")
    p.add_argument("--click_json",
                   default="../data/coconut_subset/annotations/subset_click_final_gguf.json")
    p.add_argument("--output",
                   default="../data/coconut_subset/annotations/instruction_quality.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
