"""
Step 6 — LLM-as-judge instruction quality evaluation
======================================================
Uses Gemini 2.5 Flash to score generated style transfer instructions
on four dimensions with a fixed rubric.

Rubric (each 1-5):
  1. Regional specificity  — does instruction name and isolate the target region?
  2. Style fidelity        — does it accurately describe the WikiArt style?
  3. Visual descriptiveness — does it use concrete visual language?
  4. Actionability         — could a diffusion model act on this instruction?

Blind evaluation — judge sees instruction, region label, style name
but NOT which template or model produced it.

Usage:
    export GEMINI_API_KEY=your_key
    python3 step6_llm_judge.py \
        --results_json  ../results/template_comparison_979.json \
        --lora_a_json   ../results/attention_lora_A.json \
        --lora_h_json   ../results/attention_lora_H.json \
        --output        ../results/llm_judge_scores.json \
        --n_sample      50
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
from google import genai as genai_pkg


RUBRIC_PROMPT = """You are evaluating a style transfer instruction for image editing.

Region label: {label}
Target style: {style}
Instruction: {instruction}

Score this instruction on each of the following dimensions from 1 to 5.
Be strict and consistent. Use the full range.

1. Regional specificity (1-5)
   1 = does not mention the region at all
   3 = mentions the region but also affects other areas
   5 = clearly isolates the target region and excludes all others

2. Style fidelity (1-5)
   1 = does not reflect the target style at all
   3 = mentions the style name but no visual characteristics
   5 = accurately describes specific visual properties of the style

3. Visual descriptiveness (1-5)
   1 = purely abstract ("apply the style")
   3 = some visual language but generic
   5 = rich concrete visual language (brushstrokes, palette, texture, light)

4. Actionability (1-5)
   1 = vague — a diffusion model could not act on this
   3 = partially actionable but ambiguous
   5 = specific enough that a diffusion model could execute it precisely

Respond ONLY with a JSON object, no markdown, no explanation:
{{"regional_specificity": <1-5>, "style_fidelity": <1-5>, "visual_descriptiveness": <1-5>, "actionability": <1-5>}}"""


REFUSAL_PHRASES = [
    "i'm an ai", "i cannot", "language model",
    "as an ai", "i'm unable", "unfortunately"
]


def is_refusal(text):
    return any(p in text.lower() for p in REFUSAL_PHRASES)


def score_instruction(client, label, style, instruction, retries=3):
    """Score one instruction using Gemini. Returns dict of scores or None."""
    if is_refusal(instruction) or len(instruction.split()) < 5:
        return None

    prompt = RUBRIC_PROMPT.format(
        label=label, style=style, instruction=instruction
    )

    for attempt in range(retries):
        try:
            response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
            text = response.text.strip()
            # Clean JSON
            text = text.replace("```json", "").replace("```", "").strip()
            scores = json.loads(text)
            # Validate
            required = ["regional_specificity", "style_fidelity",
                        "visual_descriptiveness", "actionability"]
            if all(k in scores for k in required):
                if all(1 <= scores[k] <= 5 for k in required):
                    scores["total"] = sum(scores[k] for k in required)
                    return scores
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    Failed after {retries} attempts: {e}")
    return None


def run(args):
    # Configure Gemini
    api_key = os.environ.get("GEMINI_API_KEY") or args.api_key
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY environment variable or --api_key")

    client = genai_pkg.Client(api_key=api_key)
    print("Gemini 2.5 Flash loaded\n")

    # Load all instruction sources
    sources = {}

    # Baseline templates A–I
    with open(args.results_json) as f:
        tmpl_data = json.load(f)
    for tmpl in ["A","E","H","F","C"]:  # key templates only
        if tmpl in tmpl_data["per_template"]:
            sources[f"baseline_{tmpl}"] = tmpl_data["per_template"][tmpl]

    # LoRA fine-tuned models
    with open(args.lora_a_json) as f:
        lora_a = json.load(f)
    with open(args.lora_h_json) as f:
        lora_h = json.load(f)

    # Use template A instructions from fine-tuned models
    sources["lora_A_tmplA"] = lora_a["per_template"]["A"]
    sources["lora_H_tmplA"] = lora_h["per_template"]["A"]

    # Sample regions
    anchor = sources["baseline_A"]
    if args.indices_json:
        with open(args.indices_json) as f:
            idx_data = json.load(f)
        sample_indices = idx_data["indices"]
        print(f"Using pre-filtered indices: {len(sample_indices)} regions")
    else:
        random.seed(42)
        sample_indices = random.sample(range(len(anchor)), min(args.n_sample, len(anchor)))

    print(f"Evaluating {len(sample_indices)} regions × {len(sources)} sources")
    print(f"Total API calls: {len(sample_indices) * len(sources)}\n")

    results = {}
    summary = {}

    for source_name, regions in sources.items():
        print(f"\n{'='*55}")
        print(f"Source: {source_name}")
        print(f"{'='*55}")

        source_results = []
        scores_by_dim = {
            "regional_specificity": [],
            "style_fidelity": [],
            "visual_descriptiveness": [],
            "actionability": [],
            "total": [],
        }
        n_skipped = 0

        for idx in sample_indices:
            if idx >= len(regions):
                n_skipped += 1
                continue

            r = regions[idx]
            label       = r.get("region_label", "")
            style       = r.get("style_name", "")
            instruction = r.get("instruction", "")

            scores = score_instruction(client, label, style, instruction)

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

            # Rate limit — 15 RPM on free tier
            time.sleep(3)

        # Summary stats
        n_scored = len(source_results)
        print(f"  Scored: {n_scored}  Skipped: {n_skipped}")
        for dim, vals in scores_by_dim.items():
            if vals:
                print(f"  {dim:<28}: {np.mean(vals):.2f} ± {np.std(vals):.2f}")

        results[source_name]  = source_results
        summary[source_name]  = {
            dim: {
                "mean": round(float(np.mean(vals)), 3),
                "std":  round(float(np.std(vals)), 3),
                "n":    len(vals),
            }
            for dim, vals in scores_by_dim.items() if vals
        }

    # Final comparison table
    print(f"\n{'='*70}")
    print("FINAL COMPARISON — LLM-as-judge scores")
    print(f"{'='*70}")
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

    # Save
    out = {
        "model":   "gemini-2.0-flash",
        "rubric":  ["regional_specificity","style_fidelity",
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
    p.add_argument("--n_sample", type=int, default=50)
    p.add_argument("--indices_json", default=None,
        help="Pre-filtered region indices JSON")
    p.add_argument("--api_key", default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
