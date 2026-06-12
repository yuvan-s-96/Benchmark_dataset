"""
Step B (GGUF/Local): Style Pairing + Instruction Generation
============================================================
Uses local Mistral-7B GGUF model — no internet needed after download.
Runs on ogg GPU via llama-cpp-python.

Usage:
    python3 B_build_subset_gguf.py \
        --stub        ../data/coconut_subset/annotations/coconut_stub_merged_auto.json \
        --style_dir   ../data/style_references \
        --output_json ../data/coconut_subset/annotations/subset_auto_draft_gguf.json
"""

import argparse
import json
import random
from pathlib import Path
from tqdm import tqdm
from llama_cpp import Llama

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

PROMPT_TEMPLATE = """[INST] You are a style transfer assistant. Write a single natural-language instruction (1 sentence) for regional style transfer.

Region label: {label}
Region description: {caption}
Target style: {style}

Requirements:
- Identify the region by its label
- Describe how to apply the style visually and specifically
- Sound like something a user would naturally type
- Return ONLY the instruction sentence, nothing else

[/INST]"""


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


def make_instruction_ref(region_label, style_name=""):
    label = region_label.strip() or "this region"
    return f"Render {label} using the style of the reference image."

def make_instruction_ref_named(region_label, style_name):
    label = region_label.strip() or "this region"
    style = style_name.replace("-", " ").replace("_", " ").strip() or "the given"
    return f"Render {label} using the {style} style of the reference image."


def make_instruction_stub(region_label, style_name):
    label = region_label.strip() or "this region"
    desc  = STYLE_DESCRIPTIONS.get(style_name, f"{style_name} style")
    return f"Render {label} in {desc}."


def run(args):
    print(f"Loading model: {args.model_path}")
    llm = Llama(
        model_path=args.model_path,
        n_ctx=1024,
        n_gpu_layers=args.gpu_layers,
        verbose=False,
    )
    print("Model loaded.\n")

    with open(args.stub) as f:
        records = json.load(f)

    style_index = load_style_index(args.style_dir)
    print(f"Styles  : {len(style_index)} categories")
    print(f"Samples : {len(records)}\n")

    for record in tqdm(records, desc="Generating instructions"):
        used_styles = []

        for region in record["regions"]:
            if not region.get("style_name"):
                style_name, style_path = sample_style(
                    style_index, exclude=used_styles,
                    seed=hash(record["image_id"] + str(region["mask_index"])))
                region["style_name"]      = style_name
                region["style_reference"] = style_path
            else:
                style_name = region["style_name"]
            used_styles.append(style_name)

            label   = region.get("region_label",   "this region")
            caption = region.get("region_caption", "")

            region["instruction_ref"]       = make_instruction_ref(label)
            region["instruction_ref_named"] = make_instruction_ref_named(label, style_name)

            prompt = PROMPT_TEMPLATE.format(
                label=label,
                caption=caption[:200],
                style=style_name
            )
            try:
                result = llm(
                    prompt,
                    max_tokens=100,
                    stop=["</s>", "\n\n", "[INST]"],
                )
                text = result["choices"][0]["text"].strip()
                text = text.split("\n")[0].strip().strip('"').strip("'")
                region["instruction_text"] = text if text else make_instruction_stub(label, style_name)
            except Exception as e:
                tqdm.write(f"  [warn] {label}: {e}")
                region["instruction_text"] = make_instruction_stub(label, style_name)

        record["composite_instruction_text"] = " ".join(
            r["instruction_text"] for r in record["regions"])
        record["composite_instruction_ref"] = " ".join(
            r["instruction_ref"] for r in record["regions"])
        record["composite_instruction_ref_named"] = " ".join(
            r["instruction_ref_named"] for r in record["regions"])
        record["num_regions"] = len(record["regions"])

        # Incremental save
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)

    counts = [r["num_regions"] for r in records]
    print(f"\nOutput  : {args.output_json}")
    print(f"Regions : min={min(counts)} max={max(counts)} mean={sum(counts)/len(counts):.1f}")
    print(f"\nSample instructions:")
    for reg in records[0]["regions"][:3]:
        print(f"  [{reg['region_label']}]")
        print(f"    text: {reg['instruction_text']}")
        print(f"    ref : {reg['instruction_ref']}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stub",
                   default="../data/coconut_subset/annotations/coconut_stub_merged_auto.json")
    p.add_argument("--style_dir",
                   default="../data/style_references")
    p.add_argument("--output_json",
                   default="../data/coconut_subset/annotations/subset_auto_draft_gguf.json")
    p.add_argument("--model_path",
                   default="../models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    p.add_argument("--gpu_layers", type=int, default=35)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
