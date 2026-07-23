"""
Step 5a-LoRA — Sink-corrected attention metric for LoRA models
================================================================
Applies the identical sink-correction formula from
step5_sink_corrected_metric.py to LoRA attention output
(per_template structure) instead of baseline per_region structure.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

STRUCTURAL_STRINGS = [
    "You are a style transfer assistant",
    "Write one instruction to apply this style to this region only",
    "Write a single image editing instruction",
    "leaving all other regions completely unchanged",
    "leaving everything else untouched",
    "leaving everything else in the scene untouched",
    "Think step by step",
    "first identify",
    "then describe how to apply",
    "style transfer task",
    "TARGET REGION",
    "STYLE TO APPLY",
]

def get_structural_indices(tokenizer, prompt):
    all_ids = tokenizer.encode(prompt, add_special_tokens=True)
    decoded = [tokenizer.decode([i]) for i in all_ids]
    n = len(all_ids)
    structural = set()
    structural.add(0)
    for i, tok in enumerate(decoded):
        stripped = tok.strip()
        if stripped in {".", ",", ":", ";", "!", "?", "[", "]",
                        "[INST]", "[/INST]", "INST", "/INST"}:
            structural.add(i)
        if len(stripped) == 0:
            structural.add(i)
    for struct_str in STRUCTURAL_STRINGS:
        candidates = [
            tokenizer.encode(struct_str, add_special_tokens=False),
            tokenizer.encode(" " + struct_str, add_special_tokens=False),
        ]
        for cand in candidates:
            if not cand:
                continue
            for start in range(n - len(cand) + 1):
                if all_ids[start:start+len(cand)] == cand:
                    for idx in range(start, start+len(cand)):
                        structural.add(idx)
    return structural

def compute_corrected_metrics(weights, label_indices, style_indices,
                               structural_indices, n_tokens):
    w = np.array(weights[:n_tokens], dtype=np.float64)
    label_raw = sum(w[i] for i in label_indices if i < n_tokens)
    style_raw = sum(w[i] for i in style_indices if i < n_tokens)
    semantic = [i for i in range(n_tokens) if i not in structural_indices]
    if not semantic:
        return label_raw, style_raw, 0.0, 0.0, 0.0
    w_sem = w[semantic]
    w_sem_sum = w_sem.sum()
    if w_sem_sum < 1e-10:
        return label_raw, style_raw, 0.0, 0.0, 0.0
    label_sem = np.array([w[i] for i in label_indices if i < n_tokens])
    style_sem = np.array([w[i] for i in style_indices if i < n_tokens])
    label_corrected = label_sem.sum() / w_sem_sum
    style_corrected = style_sem.sum() / w_sem_sum
    structural_valid = [i for i in structural_indices if i < n_tokens]
    structural_mass = w[structural_valid].sum() if structural_valid else 0.0
    return label_raw, style_raw, label_corrected, style_corrected, structural_mass

def run(args):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    print("Loading LoRA attention data...")
    with open(args.attention_json) as f:
        att_data = json.load(f)

    summary = {}
    per_template_out = {}

    for tmpl_name, regions in att_data["per_template"].items():
        raw_vals, corr_vals = [], []
        style_raw_vals, style_corr_vals = [], []
        structural_vals = []
        tmpl_results = []

        for region in tqdm(regions, desc=f"T{tmpl_name}"):
            prompt = region.get("prompt", "")
            weights = region.get("att_weights", [])
            label_indices = region.get("label_token_indices", [])
            style_indices = region.get("style_token_indices", [])
            n_tokens = region.get("n_input_tokens", len(weights))

            if not prompt or not weights:
                continue

            structural = get_structural_indices(tokenizer, prompt)
            lr, sr, lc, sc, sm = compute_corrected_metrics(
                weights, label_indices, style_indices, structural, n_tokens
            )

            raw_vals.append(lr)
            corr_vals.append(lc)
            style_raw_vals.append(sr)
            style_corr_vals.append(sc)
            structural_vals.append(sm)

            tmpl_results.append({
                "image_id": region.get("image_id"),
                "mask_index": region.get("mask_index"),
                "region_label": region.get("region_label"),
                "style_name": region.get("style_name"),
                "label_mass_raw": lr,
                "label_mass_corrected": lc,
                "style_mass_raw": sr,
                "style_mass_corrected": sc,
                "structural_mass": sm,
            })

        per_template_out[tmpl_name] = tmpl_results
        summary[tmpl_name] = {
            "label_mass_raw":       round(float(np.mean(raw_vals)), 4) if raw_vals else 0,
            "label_mass_corrected": round(float(np.mean(corr_vals)), 4) if corr_vals else 0,
            "style_mass_raw":       round(float(np.mean(style_raw_vals)), 4) if style_raw_vals else 0,
            "style_mass_corrected": round(float(np.mean(style_corr_vals)), 4) if style_corr_vals else 0,
            "structural_mass_mean": round(float(np.mean(structural_vals)), 4) if structural_vals else 0,
            "n": len(tmpl_results),
        }
        print(f"  T{tmpl_name}: raw={summary[tmpl_name]['label_mass_raw']*100:.3f}%  "
              f"corrected={summary[tmpl_name]['label_mass_corrected']*100:.3f}%  "
              f"n={summary[tmpl_name]['n']}")

    print(f"\n{'='*60}")
    print("TEMPLATE RANKING — sink-corrected label mass")
    print(f"{'='*60}")
    ranked = sorted(summary.items(), key=lambda t: t[1]["label_mass_corrected"], reverse=True)
    for tmpl, s in ranked:
        print(f"  {tmpl}: raw={s['label_mass_raw']*100:>7.3f}%  "
              f"corrected={s['label_mass_corrected']*100:>7.3f}%")

    out = {
        "lora": att_data.get("lora", att_data.get("adapter", "unknown")),
        "method": "sink-corrected attention mass (LoRA)",
        "summary": summary,
        "per_template": per_template_out,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.output}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attention_json", required=True)
    p.add_argument("--output", default="../results/sink_corrected_lora.json")
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())
