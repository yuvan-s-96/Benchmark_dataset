"""
Step 6 — LLM-as-judge using local LLaMA-3.1-8B-Instruct
No API required — runs on ogg GPU directly
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

RUBRIC_PROMPT = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

You are evaluating a style transfer instruction for image editing.

Region label: {label}
Target style: {style}
Instruction: {instruction}

Score this instruction on each dimension from 1 to 5. Be strict.

1. Regional specificity (1-5): does instruction name and isolate the target region?
2. Style fidelity (1-5): does it accurately describe the WikiArt style visually?
3. Visual descriptiveness (1-5): does it use concrete visual language?
4. Actionability (1-5): could a diffusion model act on this precisely?

Respond ONLY with valid JSON, nothing else:
{{"regional_specificity": <1-5>, "style_fidelity": <1-5>, "visual_descriptiveness": <1-5>, "actionability": <1-5>}}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{{"""

REFUSAL_PHRASES = ["i'm an ai", "i cannot", "language model", "as an ai"]


def is_refusal(text):
    return any(p in text.lower() for p in REFUSAL_PHRASES)


def score_instruction(model, tokenizer, device, label, style, instruction):
    if is_refusal(instruction) or len(instruction.split()) < 5:
        return None

    prompt = RUBRIC_PROMPT.format(
        label=label, style=style, instruction=instruction[:300]
    )

    try:
        inputs = tokenizer(prompt, return_tensors="pt",
                          truncation=True, max_length=768).to(device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=60, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=1.0,
            )

        text = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        text = "{" + text.split("{")[-1].split("}")[0] + "}"
        text = text.replace("```json","").replace("```","").strip()
        scores = json.loads(text)

        required = ["regional_specificity","style_fidelity",
                    "visual_descriptiveness","actionability"]
        if all(k in scores for k in required):
            if all(1 <= int(scores[k]) <= 5 for k in required):
                scores = {k: int(scores[k]) for k in required}
                scores["total"] = sum(scores[k] for k in required)
                return scores
    except Exception as e:
        print(f"    Parse error: {e} | text: {text[:80] if 'text' in dir() else 'N/A'}")
    return None


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)

    print("Loading LLaMA-3.1-8B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        quantization_config=bnb, device_map={"": device},
        attn_implementation="eager")
    model.eval()
    print("Model loaded\n")

    # Load instruction sources
    sources = {}
    with open(args.results_json) as f:
        tmpl_data = json.load(f)
    for tmpl in ["A","E","H","F","C"]:
        if tmpl in tmpl_data["per_template"]:
            sources[f"baseline_{tmpl}"] = tmpl_data["per_template"][tmpl]

    with open(args.lora_a_json) as f:
        lora_a = json.load(f)
    with open(args.lora_h_json) as f:
        lora_h = json.load(f)

    sources["lora_A_tmplA"] = lora_a["per_template"]["A"]
    sources["lora_H_tmplA"] = lora_h["per_template"]["A"]

    # Sample indices
    anchor = sources["baseline_A"]
    if args.indices_json:
        with open(args.indices_json) as f:
            idx_data = json.load(f)
        sample_indices = idx_data["indices"]
        print(f"Using pre-filtered indices: {len(sample_indices)} regions")
    else:
        random.seed(42)
        sample_indices = random.sample(range(len(anchor)),
                                       min(args.n_sample, len(anchor)))

    print(f"Evaluating {len(sample_indices)} regions x {len(sources)} sources\n")

    results = {}
    summary = {}

    for source_name, regions in sources.items():
        print(f"\n{'='*55}")
        print(f"Source: {source_name}")
        print(f"{'='*55}")

        source_results = []
        scores_by_dim = {
            "regional_specificity":[], "style_fidelity":[],
            "visual_descriptiveness":[], "actionability":[], "total":[],
        }
        n_skipped = 0

        for idx in sample_indices:
            if idx >= len(regions):
                n_skipped += 1
                continue

            r = regions[idx]
            label       = r.get("region_label","")
            style       = r.get("style_name","")
            instruction = r.get("instruction","")

            scores = score_instruction(model, tokenizer, device,
                                      label, style, instruction)

            if scores is None:
                n_skipped += 1
                continue

            source_results.append({
                "image_id":    r.get("image_id"),
                "region_label": label,
                "style_name":  style,
                "instruction": instruction,
                "scores":      scores,
            })
            for dim in scores_by_dim:
                scores_by_dim[dim].append(scores[dim])

        n_scored = len(source_results)
        print(f"  Scored: {n_scored}  Skipped: {n_skipped}")
        for dim, vals in scores_by_dim.items():
            if vals:
                print(f"  {dim:<28}: {np.mean(vals):.2f}")

        results[source_name] = source_results
        summary[source_name] = {
            dim: {"mean": round(float(np.mean(vals)),3),
                  "std":  round(float(np.std(vals)),3),
                  "n":    len(vals)}
            for dim, vals in scores_by_dim.items() if vals
        }

    print(f"\n{'='*65}")
    print("FINAL COMPARISON — LLM-as-judge (LLaMA-3.1-8B local)")
    print(f"{'='*65}")
    print(f"\n  {'Source':<20} {'RegSpec':>8} {'StyleFid':>9} "
          f"{'VisDesc':>8} {'Action':>8} {'Total':>7}")
    print(f"  {'-'*60}")
    for src, s in sorted(summary.items(),
                         key=lambda x: x[1].get("total",{}).get("mean",0),
                         reverse=True):
        rs = s.get("regional_specificity",{}).get("mean","—")
        sf = s.get("style_fidelity",{}).get("mean","—")
        vd = s.get("visual_descriptiveness",{}).get("mean","—")
        ac = s.get("actionability",{}).get("mean","—")
        tt = s.get("total",{}).get("mean","—")
        print(f"  {src:<20} {rs:>8} {sf:>9} {vd:>8} {ac:>8} {tt:>7}")

    out = {
        "model":    "meta-llama/Llama-3.1-8B-Instruct (local 4-bit)",
        "rubric":   ["regional_specificity","style_fidelity",
                     "visual_descriptiveness","actionability"],
        "n_sample": len(sample_indices),
        "summary":  summary,
        "results":  results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_json",
        default="../results/template_comparison_979.json")
    p.add_argument("--lora_a_json",
        default="../results/attention_lora_A.json")
    p.add_argument("--lora_h_json",
        default="../results/attention_lora_H.json")
    p.add_argument("--output",
        default="../results/llm_judge_scores.json")
    p.add_argument("--indices_json", default=None)
    p.add_argument("--n_sample", type=int, default=30)
    p.add_argument("--api_key", default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
