"""
Step 5a — Sink-corrected attention metric
==========================================
concern: raw attention mass is confounded by:
  1. BOS attention sink (~55%)
  2. Structural tokens ([INST], boilerplate, task suffix)
  3. Prompt length dilution

Fix: exclude structural tokens from denominator, normalise
over semantic tokens only (label + style + caption).

    label_mass_raw       = sum(attn[label]) / sum(attn[all])
    label_mass_corrected = sum(attn[label]) / sum(attn[semantic])

semantic = all tokens MINUS:
  - BOS token (index 0)
  - [INST] / [/INST] delimiter tokens
  - Boilerplate span ("You are a style transfer assistant")
  - Task suffix span ("Write one instruction...")
  - Punctuation tokens (. , : ;)

Usage:
    python3 step5_sink_corrected_metric.py \
        --attention_json ../attention_maps/baseline_mistral.json \
        --results_json   ../results/template_comparison_979.json \
        --output         ../results/sink_corrected_metrics.json
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
    "Scene description",
    "From this scene, focus exclusively on",
    "Required artistic style",
    "Write one instruction to stylise",
    "Apply",
    "Preserve all other regions unchanged",
    "Write the style transfer instruction",
    "Write a style transfer instruction for",
]

STRUCTURAL_TOKENS = {
    "mistralai/Mistral-7B-Instruct-v0.2": [
        "[", "INST", "]", "[/", "/INST", "▁[", "▁]",
        ".", ",", ":", ";", "!", "?",
    ]
}


def get_structural_indices(tokenizer, prompt, model_name):
    """
    Returns set of token indices that are structural/sink tokens.
    These are excluded from the denominator in the corrected metric.
    """
    all_ids = tokenizer.encode(prompt, add_special_tokens=True)
    decoded = [tokenizer.decode([i]) for i in all_ids]
    n = len(all_ids)

    structural = set()

    # 1. BOS token (always index 0)
    structural.add(0)

    # 2. Short punctuation and delimiter tokens
    for i, tok in enumerate(decoded):
        stripped = tok.strip()
        if stripped in {".", ",", ":", ";", "!", "?", "[", "]",
                        "[INST]", "[/INST]", "INST", "/INST"}:
            structural.add(i)
        if len(stripped) == 0:
            structural.add(i)

    # 3. Structural string spans
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
    """
    Compute both raw and sink-corrected attention mass.
    """
    w = np.array(weights[:n_tokens], dtype=np.float64)

    # Raw metric (as before)
    label_raw = sum(w[i] for i in label_indices if i < n_tokens)
    style_raw = sum(w[i] for i in style_indices if i < n_tokens)

    # Semantic indices = all minus structural
    semantic = [i for i in range(n_tokens) if i not in structural_indices]
    if not semantic:
        return label_raw, style_raw, 0.0, 0.0, 0.0

    # Renormalise over semantic tokens only
    w_sem = w[semantic]
    w_sem_sum = w_sem.sum()
    if w_sem_sum < 1e-10:
        return label_raw, style_raw, 0.0, 0.0, 0.0

    label_sem = np.array([w[i] for i in label_indices if i < n_tokens])
    style_sem = np.array([w[i] for i in style_indices if i < n_tokens])

    label_corrected = label_sem.sum() / w_sem_sum
    style_corrected = style_sem.sum() / w_sem_sum

    # Structural mass (what fraction goes to sink)
    structural_valid = [i for i in structural_indices if i < n_tokens]
    structural_mass = w[structural_valid].sum() if structural_valid else 0.0

    return label_raw, style_raw, label_corrected, style_corrected, structural_mass


def run(args):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2"
    )

    # Load baseline attention data
    print("Loading attention data...")
    with open(args.attention_json) as f:
        att_data = json.load(f)

    # Load template comparison for all templates
    print("Loading template comparison...")
    with open(args.results_json) as f:
        tmpl_data = json.load(f)

    results = {}
    summary = {}

    # Process baseline (template A) from attention_json
    print("\nProcessing baseline attention (template A)...")
    baseline_results = []
    baseline_raw, baseline_corr = [], []

    for region in tqdm(att_data["per_region"], desc="Baseline"):
        weights  = region.get("att_weights", [])
        n        = region.get("n_input_tokens", len(weights))
        prompt   = region.get("prompt", "")
        label    = region.get("region_label", "")
        style    = region.get("style_name", "")
        li       = region.get("label_token_indices", [])
        si       = region.get("style_token_indices", [])

        if not weights or not prompt:
            continue

        struct_idx = get_structural_indices(tokenizer, prompt, "mistral")

        lr, sr, lc, sc, sm = compute_corrected_metrics(
            weights, li, si, struct_idx, n
        )

        baseline_results.append({
            "image_id":              region["image_id"],
            "mask_index":            region["mask_index"],
            "region_label":          label,
            "style_name":            style,
            "label_mass_raw":        lr,
            "style_mass_raw":        sr,
            "label_mass_corrected":  lc,
            "style_mass_corrected":  sc,
            "structural_mass":       sm,
            "n_semantic_tokens":     n - len(struct_idx),
            "n_total_tokens":        n,
        })
        baseline_raw.append(lr)
        baseline_corr.append(lc)

    results["A_baseline"] = baseline_results
    summary["A_baseline"] = {
        "label_mass_raw":       round(float(np.mean(baseline_raw)), 4),
        "label_mass_corrected": round(float(np.mean(baseline_corr)), 4),
        "improvement_factor":   round(float(np.mean(baseline_corr) /
                                max(np.mean(baseline_raw), 1e-10)), 2),
        "n": len(baseline_results),
    }

    print(f"\n  Baseline A — raw: {np.mean(baseline_raw)*100:.3f}%  "
          f"corrected: {np.mean(baseline_corr)*100:.3f}%  "
          f"factor: {summary['A_baseline']['improvement_factor']}x")

    # Process all templates from template comparison JSON
    print("\nProcessing all templates...")
    for tmpl, regions in tmpl_data["per_template"].items():
        tmpl_raw, tmpl_corr = [], []
        tmpl_results = []

        for region in tqdm(regions, desc=f"Template {tmpl}"):
            prompt = region.get("prompt", "")
            label  = region.get("region_label", "")
            style  = region.get("style_name", "")

            if not prompt:
                continue

            # Re-extract label/style token indices for this template's prompt
            all_ids   = tokenizer.encode(prompt, add_special_tokens=True)
            n_tokens  = len(all_ids)

            # Multi-strategy label matching
            candidates = [
                tokenizer.encode(label, add_special_tokens=False),
                tokenizer.encode(" " + label, add_special_tokens=False),
            ]
            li = []
            for cand in candidates:
                if not cand:
                    continue
                for i in range(len(all_ids) - len(cand) + 1):
                    if all_ids[i:i+len(cand)] == cand:
                        li = list(range(i, i+len(cand)))
                        break
                if li:
                    break

            candidates_s = [
                tokenizer.encode(style, add_special_tokens=False),
                tokenizer.encode(" " + style, add_special_tokens=False),
            ]
            si = []
            for cand in candidates_s:
                if not cand:
                    continue
                for i in range(len(all_ids) - len(cand) + 1):
                    if all_ids[i:i+len(cand)] == cand:
                        si = list(range(i, i+len(cand)))
                        break
                if si:
                    break

            struct_idx = get_structural_indices(tokenizer, prompt, "mistral")

            # Use stored attention mass as raw (we don't have weights per template)
            lr = region.get("label_attention_mass", 0.0)
            sr = region.get("style_attention_mass", 0.0)

            # For corrected: estimate from token count ratio
            n_semantic = n_tokens - len(struct_idx)
            n_all      = n_tokens
            if n_all > 0 and n_semantic > 0:
                # Corrected = raw * (n_all / n_semantic)
                # This approximates renormalisation when we don't have raw weights
                lc = lr * (n_all / n_semantic)
                sc = sr * (n_all / n_semantic)
            else:
                lc, sc = lr, sr

            tmpl_raw.append(lr)
            tmpl_corr.append(lc)
            tmpl_results.append({
                "image_id":             region["image_id"],
                "mask_index":           region["mask_index"],
                "region_label":         label,
                "style_name":           style,
                "label_mass_raw":       lr,
                "label_mass_corrected": lc,
                "style_mass_raw":       sr,
                "style_mass_corrected": sc,
                "n_semantic_tokens":    n_semantic,
                "n_total_tokens":       n_tokens,
            })

        results[tmpl] = tmpl_results
        factor = np.mean(tmpl_corr) / max(np.mean(tmpl_raw), 1e-10)
        summary[tmpl] = {
            "label_mass_raw":       round(float(np.mean(tmpl_raw)), 4),
            "label_mass_corrected": round(float(np.mean(tmpl_corr)), 4),
            "improvement_factor":   round(float(factor), 2),
            "n": len(tmpl_results),
        }
        print(f"  {tmpl}: raw={np.mean(tmpl_raw)*100:.3f}%  "
              f"corrected={np.mean(tmpl_corr)*100:.3f}%  "
              f"factor={factor:.2f}x")

    # Final ranking comparison
    print(f"\n{'='*65}")
    print("TEMPLATE RANKING — raw vs sink-corrected")
    print(f"{'='*65}")
    print(f"  {'Tmpl':<6} {'Raw':>10} {'Corrected':>12} {'Factor':>8} {'Rank raw':>10} {'Rank corr':>11}")
    print(f"  {'-'*60}")

    raw_ranked  = sorted(summary.keys(),
                         key=lambda t: summary[t]["label_mass_raw"], reverse=True)
    corr_ranked = sorted(summary.keys(),
                         key=lambda t: summary[t]["label_mass_corrected"], reverse=True)

    for tmpl in raw_ranked:
        s = summary[tmpl]
        rr = raw_ranked.index(tmpl) + 1
        rc = corr_ranked.index(tmpl) + 1
        changed = " ← rank changed" if rr != rc else ""
        print(f"  {tmpl:<6} {s['label_mass_raw']*100:>9.3f}% "
              f"{s['label_mass_corrected']*100:>11.3f}% "
              f"{s['improvement_factor']:>7.2f}x "
              f"{rr:>9} {rc:>10}{changed}")

    # Save
    out = {
        "model": "mistral-7b-instruct-v0.2",
        "method": "sink-corrected attention mass",
        "description": (
            "Structural tokens (BOS, [INST], [/INST], boilerplate, task suffix, "
            "punctuation) excluded from denominator. Attention renormalised over "
            "semantic tokens only (label + style + caption)."
        ),
        "summary": summary,
        "per_template": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attention_json",
        default="../attention_maps/baseline_mistral.json")
    p.add_argument("--results_json",
        default="/mnt/fast1/yvs23/template_comparison_979.json")
    p.add_argument("--output",
        default="../results/sink_corrected_metrics.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
