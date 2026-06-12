"""
Step B: Style Pairing + Dual Instruction Generation
=====================================================
Pairs each region with a WikiArt style reference and generates BOTH:
  instruction_text  — "Render the sky using impressionist brushstrokes."
  instruction_ref   — "Render the sky using the style of the reference image."

Usage:
    python3 B_build_subset.py \
        --stub        ../data/coconut_subset/annotations/coconut_stub.json \
        --style_dir   ../data/style_references \
        --output_json ../data/coconut_subset/annotations/benchmark_subset.json \
        --model stub   # or gemini / gpt4o

For Gemini (free): export GEMINI_API_KEY="your-key"
For GPT-4o:        export OPENAI_API_KEY="sk-..."
"""

import argparse
import base64
import json
import os
import random
import time
from io import BytesIO
from pathlib import Path

from PIL import Image
from tqdm import tqdm

STYLE_DESCRIPTIONS = {
    "impressionism":          "impressionist brushstrokes with soft colours and light",
    "post-impressionism":     "post-impressionist style with bold colours and expressive form",
    "baroque":                "baroque style with dramatic lighting and rich detail",
    "expressionism":          "expressionist style with distorted forms and vivid emotion",
    "cubism":                 "cubist style with geometric fragmentation",
    "fauvism":                "fauvist style with wild, non-naturalistic colours",
    "romanticism":            "romantic style with dramatic atmosphere and natural grandeur",
    "renaissance":            "Renaissance style with classical composition and realism",
    "watercolor":             "watercolour wash style with translucent layers",
    "art-nouveau":            "Art Nouveau style with flowing organic lines",
    "symbolism":              "symbolist style with dreamlike and mystical imagery",
    "pointillism":            "pointillist style using small distinct dots of colour",
    "abstract-expressionism": "abstract expressionist style with gestural brushwork",
    "pop-art":                "Pop Art style with bold outlines and flat bright colours",
    "minimalism":             "minimalist style with clean lines and reduced forms",
    "color-field-painting":   "colour field style with large areas of flat solid colour",
    "ukiyo-e":                "ukiyo-e woodblock print style",
    "art-deco":               "Art Deco style with geometric elegance",
    "neoclassicism":          "neoclassical style with ordered composition and clarity",
    "realism":                "realist style with faithful natural representation",
}

PROMPT_TEMPLATE = """You are given a region label, its visual description, and a style name.

Write a single natural-language style transfer instruction (1-2 sentences) that:
1. Identifies the region by its label (e.g. "the sky", "the person")
2. Describes the target style using specific visual terms
3. Is fluent, concise, and sounds like something a user would type

Region label: {label}
Region description: {caption}
Style: {style}

Return ONLY the instruction sentence. No preamble, no quotes."""


def load_style_index(style_dir):
    index = {}
    for d in sorted(Path(style_dir).iterdir()):
        if not d.is_dir():
            continue
        imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.png")))
        if imgs:
            index[d.name] = [str(p) for p in imgs]
    return index


def sample_style(style_index, exclude=None, seed=None):
    rng    = random.Random(seed)
    styles = [s for s in style_index if s not in (exclude or [])]
    if not styles:
        styles = list(style_index.keys())
    name = rng.choice(styles)
    return name, rng.choice(style_index[name])


def make_instruction_ref(region_label):
    label = region_label.strip() or "this region"
    return f"Render {label} using the style of the reference image."


def make_instruction_text_stub(region_label, style_name):
    label = region_label.strip() or "this region"
    desc  = STYLE_DESCRIPTIONS.get(style_name, f"{style_name} style")
    return f"Render {label} in {desc}."


def encode_b64(path, max_side=512):
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def make_instruction_text_gemini(label, caption, style_name, style_path, api_key):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model  = genai.GenerativeModel("gemini-1.5-flash")
    prompt = PROMPT_TEMPLATE.format(label=label, caption=caption, style=style_name)
    try:
        style_img = Image.open(style_path).convert("RGB")
        style_img.thumbnail((512, 512))
        return model.generate_content([prompt, style_img]).text.strip()
    except Exception as e:
        tqdm.write(f"    [gemini warn] {e}")
        return make_instruction_text_stub(label, style_name)


def make_instruction_text_gpt4o(label, caption, style_name, style_path, api_key):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    prompt = PROMPT_TEMPLATE.format(label=label, caption=caption, style=style_name)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{encode_b64(style_path)}"}},
            ]}],
            max_tokens=150,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        tqdm.write(f"    [gpt4o warn] {e}")
        return make_instruction_text_stub(label, style_name)


def run(args):
    with open(args.stub) as f:
        records = json.load(f)

    style_index = load_style_index(args.style_dir)
    if not style_index:
        raise FileNotFoundError(f"No style dirs under {args.style_dir}. Run download_wikiart.py first.")

    print(f"Styles  : {len(style_index)} categories, {sum(len(v) for v in style_index.values())} images")
    print(f"Samples : {len(records)}")
    print(f"Model   : {args.model}\n")

    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    if args.model == "gemini" and not gemini_key:
        print("[warn] GEMINI_API_KEY not set — using stub")
        args.model = "stub"
    if args.model == "gpt4o" and not openai_key:
        print("[warn] OPENAI_API_KEY not set — using stub")
        args.model = "stub"

    for record in tqdm(records, desc="Building"):
        used_styles = []
        for region in record["regions"]:
            style_name, style_path = sample_style(
                style_index, exclude=used_styles,
                seed=hash(record["image_id"] + str(region["mask_index"])))
            region["style_name"]      = style_name
            region["style_reference"] = style_path
            used_styles.append(style_name)

            label   = region.get("region_label", "this region")
            caption = region.get("region_caption", "")

            region["instruction_ref"] = make_instruction_ref(label)

            if args.model == "gemini":
                region["instruction_text"] = make_instruction_text_gemini(
                    label, caption, style_name, style_path, gemini_key)
                time.sleep(0.5)
            elif args.model == "gpt4o":
                region["instruction_text"] = make_instruction_text_gpt4o(
                    label, caption, style_name, style_path, openai_key)
                time.sleep(0.3)
            else:
                region["instruction_text"] = make_instruction_text_stub(label, style_name)

        record["composite_instruction_text"] = " ".join(r["instruction_text"] for r in record["regions"])
        record["composite_instruction_ref"]  = " ".join(r["instruction_ref"]  for r in record["regions"])
        record["num_regions"] = len(record["regions"])

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    counts = [r["num_regions"] for r in records]
    print(f"\nOutput  : {out_path}")
    print(f"Regions : min={min(counts)} max={max(counts)} mean={sum(counts)/len(counts):.1f}")
    print(f"Sample  : {records[0]['regions'][0]['instruction_text']}")
    print(f"Next    : python3 C_quality_control.py --draft_json {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stub",        default="../data/coconut_subset/annotations/coconut_stub.json")
    p.add_argument("--style_dir",   default="../data/style_references")
    p.add_argument("--output_json", default="../data/coconut_subset/annotations/benchmark_subset.json")
    p.add_argument("--model",       choices=["stub", "gemini", "gpt4o"], default="stub")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
