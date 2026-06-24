"""
Step 1 — Baseline attention extraction on Mistral-7B
=====================================================
Loads Mistral-7B-Instruct-v0.2 in 4-bit quantisation.
For each region in the dataset, feeds the current prompt template
and extracts attention maps from all layers and heads.
Computes attention mass on region label tokens vs all other tokens.
Saves per-region attention scores to attention_maps/baseline_mistral.json

Usage:
    export CUDA_VISIBLE_DEVICES=1
    python3 step1_extract_attention.py \
        --json ../../data/coconut_subset/annotations/subset_auto_final_gguf.json \
        --output ../attention_maps/baseline_mistral.json \
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
# Prompt template A — current baseline (same structure as GGUF)
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt_A(region_label, style_name, caption):
    return (
        f"[INST] You are a style transfer assistant. "
        f"Region: {region_label}. Style: {style_name}. "
        f"Scene caption: {caption}. "
        f"Write one instruction to apply this style to this region only. [/INST]"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Attention extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_label_token_indices(tokenizer, prompt, region_label):
    """
    Find which token positions correspond to the region label in the prompt.
    Returns list of token indices.
    """
    all_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    label_tokens = tokenizer.encode(region_label, add_special_tokens=False)

    # Sliding window match
    indices = []
    for i in range(len(all_tokens) - len(label_tokens) + 1):
        if all_tokens[i:i+len(label_tokens)] == label_tokens:
            indices = list(range(i, i + len(label_tokens)))
            break
    return indices


def extract_attention_score(model, tokenizer, prompt, region_label, device):
    """
    Run one forward pass with output_attentions=True.
    Returns:
        label_mass   — mean attention weight on region label tokens (0–1)
        total_tokens — total number of input tokens
        label_indices — which token positions are the region label
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
        )

    # outputs.attentions: tuple of (num_layers,) each (batch, heads, seq, seq)
    # Aggregate: mean over all layers and heads, take last token row
    # (last token attends to all previous — most informative for generation)
    attentions = outputs.attentions  # tuple of tensors
    # Stack: (layers, batch, heads, seq, seq)
    att_stack = torch.stack(attentions, dim=0)
    # Mean over layers and heads: (seq, seq)
    att_mean = att_stack[:, 0, :, :, :].mean(dim=(0, 1))
    # Last generated position attending to input
    last_row = att_mean[-1]  # shape (seq,)
    last_row = last_row / (last_row.sum() + 1e-8)

    seq_len = inputs["input_ids"].shape[1]
    label_indices = get_label_token_indices(
        tokenizer, prompt, region_label
    )

    if not label_indices:
        return None, seq_len, []

    label_mass = float(last_row[label_indices].sum().cpu())
    return label_mass, seq_len, label_indices


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Loading Mistral-7B-Instruct-v0.2 in 4-bit...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": device},
        attn_implementation="eager",
        output_attentions=True,
    )
    model.eval()
    print("Model loaded.\n")

    with open(args.json) as f:
        records = json.load(f)

    results = []
    n = 0

    for record in tqdm(records, desc="Images"):
        caption = record.get("coconut_caption", "")
        for region in record["regions"]:
            if args.max_regions and n >= args.max_regions:
                break

            label = region.get("region_label", "")
            style = region.get("style_name", "")
            mask  = region.get("mask_index", 0)
            img_id = record["image_id"]

            prompt = build_prompt_A(label, style, caption)

            label_mass, total_tokens, label_indices = extract_attention_score(
                model, tokenizer, prompt, label, device
            )

            results.append({
                "image_id":      img_id,
                "mask_index":    mask,
                "region_label":  label,
                "style_name":    style,
                "template":      "A",
                "model":         "mistral-7b-instruct-v0.2",
                "total_tokens":  total_tokens,
                "label_token_indices": label_indices,
                "label_attention_mass": label_mass,
                "prompt":        prompt,
            })
            n += 1

        if args.max_regions and n >= args.max_regions:
            break

    # Summary
    masses = [r["label_attention_mass"] for r in results
              if r["label_attention_mass"] is not None]
    print(f"\n{'='*50}")
    print(f"Regions processed   : {len(results)}")
    print(f"Label mass mean     : {np.mean(masses):.4f}")
    print(f"Label mass median   : {np.median(masses):.4f}")
    print(f"Label mass min/max  : {np.min(masses):.4f} / {np.max(masses):.4f}")
    print(f"{'='*50}")

    out = {
        "model":   "mistral-7b-instruct-v0.2",
        "template": "A",
        "summary": {
            "n": len(results),
            "label_attention_mass": {
                "mean":   round(float(np.mean(masses)), 4),
                "median": round(float(np.median(masses)), 4),
                "min":    round(float(np.min(masses)), 4),
                "max":    round(float(np.max(masses)), 4),
            }
        },
        "per_region": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nOutput: {args.output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json",
        default="../../data/coconut_subset/annotations/subset_auto_final_gguf.json")
    p.add_argument("--output",
        default="../attention_maps/baseline_mistral.json")
    p.add_argument("--max_regions", type=int, default=20,
        help="Limit regions for initial test run. Set to 0 for all.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
