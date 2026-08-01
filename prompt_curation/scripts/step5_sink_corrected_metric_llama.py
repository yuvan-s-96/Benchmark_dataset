"""
Step 5a — Sink-corrected attention metric, LLAMA VERSION
============================================================
Adapted from step5_sink_corrected_metric.py. Same STRUCTURAL_STRINGS list
(template content boilerplate is identical between Mistral/Llama versions).
Fixes:
  1. add_special_tokens=False (prompt already contains <|begin_of_text|>
     as literal text from apply_chat_template, matching how attention
     weights were actually extracted)
  2. Llama-specific structural tokens: <|begin_of_text|>, <|start_header_id|>,
     <|end_header_id|>, <|eot_id|>, "system", "user", "assistant" header
     labels, and the "Cutting Knowledge Date / Today Date" system preamble
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
    # Llama-specific system preamble (inserted by apply_chat_template)
    "Cutting Knowledge Date: December 2023",
    "Today Date: 26 Jul 2024",
]

LLAMA_SPECIAL_TOKEN_STRINGS = {
    "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
    "system", "user", "assistant",
}

def get_structural_indices_llama(tokenizer, prompt):
    # IMPORTANT: add_special_tokens=False -- prompt already contains
    # <|begin_of_text|> as literal text from apply_chat_template(), matching
    # exactly how attention weights were extracted in step1_full_weights_llama.py
    all_ids = tokenizer.encode(prompt, add_special_tokens=False)
    decoded = [tokenizer.decode([i]) for i in all_ids]
    n = len(all_ids)

    structural = set()

    # 1. BOS token (index 0 -- <|begin_of_text|> is the first real token here)
    structural.add(0)

    # 2. Short punctuation, delimiter, and Llama special-token/header tokens
    for i, tok in enumerate(decoded):
        stripped = tok.strip()
        if stripped in {".", ",", ":", ";", "!", "?", "[", "]"}:
            structural.add(i)
        if stripped in LLAMA_SPECIAL_TOKEN_STRINGS:
            structural.add(i)
        if len(stripped) == 0:
            structural.add(i)

    # 3. Structural string spans (template boilerplate + Llama system preamble)
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

def compute_corrected_metrics(weights, label_indices, style_indices, structural_indices, n_tokens):
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

    label_indices_sem = [i for i in label_indices if i in semantic]
    style_indices_sem = [i for i in style_indices if i in semantic]
    label_corrected = sum(w[i] for i in label_indices_sem) / w_sem_sum
    style_corrected = sum(w[i] for i in style_indices_sem) / w_sem_sum

    return label_raw, style_raw, label_corrected, style_corrected, w_sem_sum

def run(args):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    with open(args.attention_json) as f:
        d = json.load(f)

    TEMPLATES = list(d["per_template"].keys())
    summary = {}

    for t in TEMPLATES:
        regions = d["per_template"][t]
        raw_masses, corrected_masses = [], []
        for r in tqdm(regions, desc=f"T{t}"):
            structural_idx = get_structural_indices_llama(tokenizer, r["prompt"])
            n_tokens = r["n_input_tokens"]
            label_raw, style_raw, label_corr, style_corr, w_sem_sum = compute_corrected_metrics(
                r["att_weights"], r["label_token_indices"], r["style_token_indices"],
                structural_idx, n_tokens
            )
            raw_masses.append(label_raw)
            corrected_masses.append(label_corr)

        mean_raw = float(np.mean(raw_masses))
        mean_corrected = float(np.mean(corrected_masses))
        print(f"  T{t}: raw={mean_raw*100:.3f}%  corrected={mean_corrected*100:.3f}%  n={len(regions)}")
        summary[t] = {"label_mass_raw": mean_raw, "label_mass_corrected": mean_corrected, "n": len(regions)}

    print(f"\n{'='*60}\nTEMPLATE RANKING — sink-corrected label mass (Llama)\n{'='*60}")
    for t, s in sorted(summary.items(), key=lambda x: -x[1]["label_mass_corrected"]):
        print(f"  {t}: raw={s['label_mass_raw']*100:6.3f}%  corrected={s['label_mass_corrected']*100:6.3f}%")

    out = {"model": "meta-llama/Llama-3.1-8B-Instruct", "summary": summary}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.output}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attention_json", required=True)
    p.add_argument("--output", default="../results/sink_corrected_llama.json")
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())
