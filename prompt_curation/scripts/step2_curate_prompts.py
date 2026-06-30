"""
Step 2 — Prompt curation
========================
Runs 7 prompt templates through Mistral-7B and measures
attention mass on region label and style name tokens for each.
Compares templates to find which causes best grounding.

Templates:
  A — baseline (region + style + caption, brief)
  B — region first + explicit ONLY
  C — caption-grounded, exclusivity emphasis
  D — action-first + contrastive exclusion
  E — question-style
  F — chain-of-thought
  G — label repetition (label mentioned 3x)

Usage:
    export CUDA_VISIBLE_DEVICES=1
    python3 step2_curate_prompts.py \
        --json ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
        --output ../results/template_comparison_mistral.json \
        --max_regions 20
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Seven prompt templates
# ─────────────────────────────────────────────────────────────────────────────

def template_A(label, style, caption):
    return (
        f"[INST] You are a style transfer assistant. "
        f"Region: {label}. Style: {style}. "
        f"Scene caption: {caption}. "
        f"Write one instruction to apply this style to this region only. [/INST]"
    )

def template_B(label, style, caption):
    return (
        f"[INST] TARGET REGION: {label}\n"
        f"STYLE TO APPLY: {style}\n"
        f"SCENE: {caption}\n"
        f"Write a single instruction that applies {style} "
        f"ONLY to the {label}. Do not mention other regions. [/INST]"
    )

def template_C(label, style, caption):
    return (
        f"[INST] Scene description: {caption}\n"
        f"From this scene, focus exclusively on: {label}\n"
        f"Required artistic style: {style}\n"
        f"Write one instruction to stylise the {label} in {style} style. [/INST]"
    )

def template_D(label, style, caption):
    return (
        f"[INST] Apply {style} style to the {label} in this image.\n"
        f"Full scene: {caption}\n"
        f"Important: apply the style to {label} only. "
        f"Preserve all other regions unchanged.\n"
        f"Write the style transfer instruction. [/INST]"
    )

def template_E(label, style, caption):
    return (
        f"[INST] Image scene: {caption}\n"
        f"What single instruction would transfer {style} artistic style "
        f"specifically to the {label}, leaving everything else untouched? [/INST]"
    )

def template_F(label, style, caption):
    return (
        f"[INST] You are a style transfer assistant.\n"
        f"Scene: {caption}\n"
        f"Target region: {label}\n"
        f"Target style: {style}\n"
        f"Think step by step: first identify the {label} in the scene, "
        f"then describe how to apply {style} specifically to it.\n"
        f"Write one instruction. [/INST]"
    )

def template_G(label, style, caption):
    return (
        f"[INST] You are stylising ONE specific region: {label}.\n"
        f"Apply {style} to the {label} only.\n"
        f"Scene context: {caption}\n"
        f"Write a style transfer instruction for the {label}. [/INST]"
    )

def template_H(label, style, caption):
    return (
        f"[INST] {style} style transfer task.\n"
        f"What single instruction would apply {style} specifically to the {label}, "
        f"leaving everything else in the scene untouched?\n"
        f"The {label} is the only region to be stylised.\n"
        f"Scene context: {caption} [/INST]"
    )

def template_I(label, style, caption):
    return (
        f"[INST] Image scene: {caption}\n"
        f"Write a single image editing instruction that transfers {style} style "
        f"specifically to the {label} in the scene, "
        f"leaving all other regions completely unchanged.\n"
        f"The instruction must describe what to do to the {label} only, "
        f"using visual and stylistic language. [/INST]"
    )

TEMPLATES = {
    "A": template_A,
    "B": template_B,
    "C": template_C,
    "D": template_D,
    "E": template_E,
    "F": template_F,
    "G": template_G,
    "H": template_H,
    "I": template_I,
}


# ─────────────────────────────────────────────────────────────────────────────
# Attention extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_token_indices(tokenizer, prompt, text):
    all_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    candidates = [
        tokenizer.encode(text, add_special_tokens=False),
        tokenizer.encode(" " + text, add_special_tokens=False),
    ]
    for text_tokens in candidates:
        if not text_tokens:
            continue
        for i in range(len(all_tokens) - len(text_tokens) + 1):
            if all_tokens[i:i+len(text_tokens)] == text_tokens:
                return list(range(i, i + len(text_tokens)))
    return []


@torch.no_grad()
def extract_attention(model, tokenizer, prompt, label, style, device,
                      max_new_tokens=80):
    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=512,
    ).to(device)
    input_len = inputs["input_ids"].shape[1]

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        output_attentions=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    instruction = tokenizer.decode(
        output.sequences[0][input_len:], skip_special_tokens=True
    ).strip()

    step0     = output.attentions[0]
    att_stack = torch.stack([a[0] for a in step0], dim=0)
    att_mean  = att_stack.mean(dim=(0, 1))
    last_row  = att_mean[-1, :input_len]
    last_row  = last_row / (last_row.sum() + 1e-8)
    weights   = last_row.cpu().float().tolist()

    label_indices = get_token_indices(tokenizer, prompt, label)
    style_indices = get_token_indices(tokenizer, prompt, style)

    label_mass = sum(weights[i] for i in label_indices if i < len(weights))
    style_mass = sum(weights[i] for i in style_indices if i < len(weights))

    return {
        "instruction":  instruction,
        "label_mass":   label_mass,
        "style_mass":   style_mass,
        "label_indices": label_indices,
        "style_indices": style_indices,
        "weights":      weights,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading {args.model} in 4-bit...")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        quantization_config=bnb,
        device_map={"": device},
        attn_implementation="eager",
    )
    model.eval()
    print("Model loaded.\n")

    with open(args.json) as f:
        records = json.load(f)

    all_results = {}
    summary     = {}

    for tmpl_name, tmpl_fn in TEMPLATES.items():
        print(f"\n{'='*55}")
        print(f"Template {tmpl_name}")
        print(f"{'='*55}")

        results      = []
        label_masses = []
        style_masses = []
        n = 0

        for record in tqdm(records, desc=f"T{tmpl_name}"):
            caption = record.get("coconut_caption", "")
            for region in record["regions"]:
                if args.max_regions and n >= args.max_regions:
                    break

                label = region.get("region_label", "")
                style = region.get("style_name",   "")

                prompt = tmpl_fn(label, style, caption)
                res    = extract_attention(
                    model, tokenizer, prompt, label, style, device
                )

                results.append({
                    "image_id":    record["image_id"],
                    "mask_index":  region["mask_index"],
                    "region_label": label,
                    "style_name":  style,
                    "template":    tmpl_name,
                    "instruction": res["instruction"],
                    "label_attention_mass": res["label_mass"],
                    "style_attention_mass": res["style_mass"],
                    "prompt":      prompt,
                })
                label_masses.append(res["label_mass"])
                style_masses.append(res["style_mass"])
                n += 1

            if args.max_regions and n >= args.max_regions:
                break

        lm = np.array(label_masses)
        sm = np.array(style_masses)

        print(f"  Label mass mean   : {lm.mean():.4f}")
        print(f"  Style mass mean   : {sm.mean():.4f}")
        print(f"  Label mass median : {np.median(lm):.4f}")

        all_results[tmpl_name] = results
        summary[tmpl_name] = {
            "label_mass": {
                "mean":   round(float(lm.mean()),        4),
                "median": round(float(np.median(lm)),    4),
                "min":    round(float(lm.min()),         4),
                "max":    round(float(lm.max()),         4),
            },
            "style_mass": {
                "mean":   round(float(sm.mean()),        4),
                "median": round(float(np.median(sm)),    4),
                "min":    round(float(sm.min()),         4),
                "max":    round(float(sm.max()),         4),
            },
            "n": len(label_masses),
        }

    # ── Final ranking ─────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Template ranking — label attention mass mean")
    print(f"{'='*55}")
    ranked = sorted(summary.items(),
                    key=lambda x: x[1]["label_mass"]["mean"], reverse=True)
    for tmpl, s in ranked:
        bar = "█" * int(s["label_mass"]["mean"] * 500)
        print(f"  {tmpl}  label={s['label_mass']['mean']:.4f}  "
              f"style={s['style_mass']['mean']:.4f}  {bar}")

    best = ranked[0][0]
    print(f"\nBest template by label mass: {best}")

    out = {
        "model":        "mistral-7b-instruct-v0.2-transformers",
        "baseline_A": {
            "label_mass": 0.0114,
            "style_mass": 0.0089,
        },
        "summary":      summary,
        "best_label":   best,
        "per_template": all_results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nOutput: {args.output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2", help="HuggingFace model id")
    p.add_argument("--json",
        default="../../data/coconut_subset/annotations/prompt_curation_inputs.json")
    p.add_argument("--output",
        default="../results/template_comparison_mistral.json")
    p.add_argument("--max_regions", type=int, default=20,
        help="Limit for test run. Set 0 for all 229.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
