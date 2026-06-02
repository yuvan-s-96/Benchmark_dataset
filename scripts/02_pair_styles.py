"""
Step 2: Style-Reference Pairing & Instruction Generation
=========================================================
For each segmented region, assign a WikiArt style image and
auto-generate a natural-language instruction.
Handles variable region counts per image automatically.

Usage:
    python 02_pair_styles.py \
        --masks_stub  ../data/annotations/masks_stub.json \
        --style_dir   ../data/style_references \
        --output_json ../data/annotations/benchmark_draft.json \
        --instruction_model stub   # or 'gpt4v' (needs OPENAI_API_KEY)

Dependencies:
    pip install openai pillow tqdm requests
"""

import argparse
import base64
import json
import os
import random
from io import BytesIO
from pathlib import Path

from PIL import Image
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Style catalogue
# ─────────────────────────────────────────────────────────────────────────────

STYLE_CATALOGUE = {
    "classical_painting": [
        "baroque", "renaissance", "romanticism", "neoclassicism",
        "realism", "mannerism_late_renaissance",
    ],
    "modern_art": [
        "impressionism", "post_impressionism", "expressionism",
        "cubism", "fauvism", "art_nouveau_modern", "symbolism", "pointillism",
    ],
    "contemporary": [
        "abstract_expressionism", "pop_art", "minimalism",
        "color_field_painting", "new_realism",
    ],
    "asian_traditional": [
        "chinese_ink_wash", "japanese_ukiyo_e", "sumi_e", "chinese_watercolour",
    ],
    "illustration_graphic": [
        "art_deco", "comic_book", "stained_glass", "low_poly",
        "watercolor_illustration", "woodcut", "sketch_pencil", "oil_pastel",
    ],
}


def load_style_index(style_dir: str) -> dict[str, list[str]]:
    style_dir = Path(style_dir)
    index: dict[str, list[str]] = {}
    for d in sorted(style_dir.iterdir()):
        if not d.is_dir():
            continue
        imgs = sorted(
            list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png"))
        )
        if imgs:
            index[d.name] = [str(p) for p in imgs]
    return index


def sample_style(style_index: dict, exclude: list[str] | None = None,
                 seed: int | None = None) -> tuple[str, str]:
    rng    = random.Random(seed)
    styles = [s for s in style_index if s not in (exclude or [])]
    if not styles:
        styles = list(style_index.keys())   # wrap-around
    style_name = rng.choice(styles)
    return style_name, rng.choice(style_index[style_name])


# ─────────────────────────────────────────────────────────────────────────────
# MLLM instruction generation
# ─────────────────────────────────────────────────────────────────────────────

INSTRUCTION_PROMPT = """
You are given a content image and one or more regional segmentation masks.
For each mask a style-reference image is also provided.

Generate a concise natural-language instruction (1-2 sentences) that a user
would type to request this regional stylisation. The instruction must:
1. Identify the region by its visible semantic label (e.g. "the sky").
2. Specify the target style by name or a brief visual description.
3. Optionally mention boundary handling (e.g. "with a smooth transition").

Return ONLY a JSON list of strings, one per region, in region order.
Example: ["Apply Van Gogh swirling brushstrokes to the nearest volcano.",
          "Transfer the watercolour wash style to the background mountains."]
"""


def encode_b64(path: str, max_side: int = 512) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def generate_instructions_openai(content_path: str, mask_paths: list[str],
                                  style_paths: list[str]) -> list[str]:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    items: list[dict] = [
        {"type": "text", "text": INSTRUCTION_PROMPT},
        {"type": "text", "text": "Content image:"},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{encode_b64(content_path)}"}},
    ]
    for i, (mp, sp) in enumerate(zip(mask_paths, style_paths)):
        items += [
            {"type": "text", "text": f"Region {i+1} mask:"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{encode_b64(mp)}"}},
            {"type": "text", "text": f"Region {i+1} style reference:"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{encode_b64(sp)}"}},
        ]

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": items}],
        max_tokens=512,
    )
    raw = resp.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    result = json.loads(raw)
    if len(result) != len(mask_paths):
        raise ValueError(f"Expected {len(mask_paths)} instructions, got {len(result)}")
    return result


def generate_instructions_stub(regions: list[dict]) -> list[str]:
    return [f"Apply the reference style to region {r['mask_index'] + 1}."
            for r in regions]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    with open(args.masks_stub) as f:
        records: list[dict] = json.load(f)

    style_index = load_style_index(args.style_dir)
    if not style_index:
        raise FileNotFoundError(
            f"No style sub-directories found under {args.style_dir}. "
            "Run download_wikiart.py first."
        )
    print(f"Styles : {len(style_index)} categories, "
          f"{sum(len(v) for v in style_index.values())} images")

    data_root = Path(args.masks_stub).parent.parent

    for record in tqdm(records, desc="Pairing styles"):
        regions      = record["regions"]
        used_styles: list[str] = []

        for region in regions:
            style_name, style_path = sample_style(
                style_index, exclude=used_styles,
                seed=hash(record["image_id"] + str(region["mask_index"])),
            )
            region["style_name"]      = style_name
            region["style_reference"] = style_path
            used_styles.append(style_name)

        content_path = str(data_root / record["image_file"])
        mask_paths   = [str(data_root / r["mask_file"]) for r in regions]
        style_paths  = [r["style_reference"] for r in regions]

        if args.instruction_model == "gpt4v" and os.environ.get("OPENAI_API_KEY"):
            try:
                instructions = generate_instructions_openai(
                    content_path, mask_paths, style_paths)
            except Exception as e:
                tqdm.write(f"  [warn] GPT-4o failed for {record['image_id']}: {e}")
                instructions = generate_instructions_stub(regions)
        else:
            instructions = generate_instructions_stub(regions)

        for region, instr in zip(regions, instructions):
            region["instruction"] = instr

        record["composite_instruction"] = " ".join(
            r["instruction"] for r in regions)
        record["num_regions"] = len(regions)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    counts = [r["num_regions"] for r in records]
    print(f"\nDraft  : {out_path}")
    print(f"Regions: min={min(counts)}  max={max(counts)}  "
          f"mean={sum(counts)/len(counts):.1f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--masks_stub",  default="../data/annotations/masks_stub.json")
    p.add_argument("--style_dir",   default="../data/style_references")
    p.add_argument("--output_json", default="../data/annotations/benchmark_draft.json")
    p.add_argument("--instruction_model", choices=["gpt4v", "stub"], default="stub")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
