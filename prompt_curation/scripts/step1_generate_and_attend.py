"""
Step 1 — Generate instructions + extract attention in one forward pass
======================================================================
Uses transformers Mistral-7B (not GGUF) so both outputs come from
the same model in the same compute path.

For each region:
  - Builds prompt template A (baseline)
  - Generates instruction text
  - Extracts attention maps from the generation forward pass
  - Computes label attention mass
  - Saves full token weights for heatmap visualisation

Output: attention_maps/baseline_mistral.json

Usage:
    export CUDA_VISIBLE_DEVICES=1
    python3 step1_generate_and_attend.py \
        --json ../../data/coconut_subset/annotations/subset_auto_final_gguf.json \
        --output ../attention_maps/baseline_mistral.json \
        --max_regions 0
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
# Prompt template A — baseline
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt_A(label, style, caption):
    return (
        f"[INST] You are a style transfer assistant. "
        f"Region: {label}. Style: {style}. "
        f"Scene caption: {caption}. "
        f"Write one instruction to apply this style to this region only. [/INST]"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Label token matching
# ─────────────────────────────────────────────────────────────────────────────

def get_label_indices(tokenizer, prompt, region_label):
    all_tokens   = tokenizer.encode(prompt, add_special_tokens=True)
    label_tokens = tokenizer.encode(region_label, add_special_tokens=False)
    for i in range(len(all_tokens) - len(label_tokens) + 1):
        if all_tokens[i:i+len(label_tokens)] == label_tokens:
            return list(range(i, i + len(label_tokens)))
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Single forward pass: generate + attend
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_and_attend(model, tokenizer, prompt, region_label, device,
                        max_new_tokens=80):
    """
    One forward pass:
      - Generates the instruction text
      - Returns attention weights over input tokens
      - Returns label token indices

    Attention strategy:
      Average over all layers and all heads,
      take the last input token's attention row
      (most informative for what the model attends to
       when starting to generate).
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    input_len = inputs["input_ids"].shape[1]

    # Generate with attention output
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        output_attentions=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode generated instruction (tokens after input)
    generated_ids = output.sequences[0][input_len:]
    instruction   = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Extract attention from the generation steps
    # output.attentions: tuple of steps, each step is tuple of layers
    # Each layer: (batch, heads, gen_seq, input_seq)
    # Use first generation step, average over layers and heads
    # -> attention from first new token back to all input tokens

    # step0: tuple of 32 layers, each (batch, heads, seq, seq)
    # seq = input_len + 1 (full causal self-attention)
    # Take last row of mean attention, slice to input_len only
    step0_atts = output.attentions[0]
    att_stack  = torch.stack([a[0] for a in step0_atts], dim=0)  # (layers,heads,seq,seq)
    att_mean   = att_stack.mean(dim=(0, 1))                        # (seq, seq)
    last_row   = att_mean[-1, :input_len]                          # (input_len,)
    last_row   = last_row / (last_row.sum() + 1e-8)
    att_weights = last_row.cpu().float().tolist()

    # Decode input tokens for visualisation
    input_ids  = inputs["input_ids"][0].tolist()
    tokens_decoded = [tokenizer.decode([i]) for i in input_ids]

    # Label token indices
    label_indices = get_label_indices(tokenizer, prompt, region_label)

    # Label attention mass
    label_mass = float(sum(att_weights[i] for i in label_indices
                           if i < len(att_weights)))

    return {
        "instruction":         instruction,
        "att_weights":         att_weights,
        "tokens_decoded":      tokens_decoded,
        "label_token_indices": label_indices,
        "label_attention_mass": label_mass,
        "n_input_tokens":      input_len,
    }


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
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2"
    )
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        quantization_config=bnb_config,
        device_map={"": device},
        attn_implementation="eager",
    )
    model.eval()
    print("Model loaded.\n")

    with open(args.json) as f:
        records = json.load(f)

    results  = []
    masses   = []
    n        = 0

    for record in tqdm(records, desc="Images"):
        caption = record.get("coconut_caption", "")
        for region in record["regions"]:
            if args.max_regions and n >= args.max_regions:
                break

            label = region.get("region_label", "")
            style = region.get("style_name",   "")

            prompt = build_prompt_A(label, style, caption)

            result = generate_and_attend(
                model, tokenizer, prompt, label, device
            )

            entry = {
                "image_id":            record["image_id"],
                "mask_index":          region["mask_index"],
                "region_label":        label,
                "style_name":          style,
                "template":            "A",
                "model":               "mistral-7b-instruct-v0.2-transformers",
                "prompt":              prompt,
                "instruction":         result["instruction"],
                "att_weights":         result["att_weights"],
                "tokens_decoded":      result["tokens_decoded"],
                "label_token_indices": result["label_token_indices"],
                "label_attention_mass": result["label_attention_mass"],
                "n_input_tokens":      result["n_input_tokens"],
            }
            results.append(entry)
            masses.append(result["label_attention_mass"])
            n += 1

            # Print sample output every 10 regions
            if n % 10 == 1:
                print(f"\n  [{n}] {label[:40]}")
                print(f"       mass={result['label_attention_mass']*100:.3f}%")
                print(f"       instr: {result['instruction'][:80]}...")

        if args.max_regions and n >= args.max_regions:
            break

    # Summary
    masses_arr = np.array(masses)
    print(f"\n{'='*55}")
    print(f"Regions processed     : {len(results)}")
    print(f"Label mass mean       : {masses_arr.mean():.4f}")
    print(f"Label mass median     : {np.median(masses_arr):.4f}")
    print(f"Label mass min/max    : {masses_arr.min():.4f} / {masses_arr.max():.4f}")
    print(f"{'='*55}")

    out = {
        "model":    "mistral-7b-instruct-v0.2-transformers",
        "template": "A",
        "summary": {
            "n": len(results),
            "label_attention_mass": {
                "mean":   round(float(masses_arr.mean()),   4),
                "median": round(float(np.median(masses_arr)), 4),
                "min":    round(float(masses_arr.min()),    4),
                "max":    round(float(masses_arr.max()),    4),
            }
        },
        "per_region": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nOutput: {args.output}")
    print("Next: run step1b_visualise_attention.py to generate heatmaps")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json",
        default="../../data/coconut_subset/annotations/subset_auto_final_gguf.json")
    p.add_argument("--output",
        default="../attention_maps/baseline_mistral.json")
    p.add_argument("--max_regions", type=int, default=5,
        help="Limit for test run. Set 0 for all 229.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
