"""
Step 2e — CLIP alignment scoring across all templates
======================================================
Loads all template results from:
  - template_comparison_979.json  (templates A-H)
  - template_EI_comparison.json       (template I, skip duplicate E)

For each region × template:
  - Encodes generated instruction text with CLIP text encoder
  - Encodes ALL 50 WikiArt style reference images for that style
  - Computes mean cosine similarity (instruction vs style images)
  - Also computes max similarity (best matching reference)

Outputs:
  - results/clip_scores.json          full scores per region per template
  - figures/fig7_clip_comparison.png  bar chart comparison

Usage:
    export CUDA_VISIBLE_DEVICES=1
    python3 step2e_clip_scoring.py
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import clip
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


STYLE_REF_DIR = Path(
    "/homes/yvs23/Benchmark_dataset/data/style_references"
)
RESULTS_FILES = {
    "main": "../results/template_comparison_979.json",
    "EI":   "../results/template_EI_comparison.json",
}
# Template display order (ranked by label mass from step2)
TMPL_ORDER = ["E","A","C","B","D","H","G","F","I"]
TMPL_LABELS = {
    "E": "E — question-style",
    "A": "A — baseline",
    "C": "C — caption-grounded",
    "B": "B — region first",
    "D": "D — contrastive",
    "H": "H — hybrid",
    "G": "G — label repetition",
    "F": "F — chain-of-thought",
    "I": "I — explicit framing",
}


# ─────────────────────────────────────────────────────────────────────────────
# Load all template results, deduplicate E
# ─────────────────────────────────────────────────────────────────────────────

def load_all_results(files):
    """Returns dict: template -> list of region dicts"""
    combined = {}

    # Load main file (A-H)
    with open(files["main"]) as f:
        main = json.load(f)
    for tmpl, regions in main["per_template"].items():
        combined[tmpl] = regions

    # Load EI file — only take I (E is duplicate)
    with open(files["EI"]) as f:
        ei = json.load(f)
    combined["I"] = ei["per_template"]["I"]

    print(f"Templates loaded: {sorted(combined.keys())}")
    print(f"Regions per template: {len(list(combined.values())[0])}")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Pre-cache style reference image embeddings
# ─────────────────────────────────────────────────────────────────────────────

def cache_style_embeddings(model, preprocess, device):
    """
    For each style folder, encode all 50 images and store mean embedding.
    Returns dict: style_name -> (mean_embedding, all_embeddings)
    """
    print("\nCaching style reference embeddings...")
    style_cache = {}

    style_dirs = [d for d in STYLE_REF_DIR.iterdir()
                  if d.is_dir() and d.name != "style_references"]

    for style_dir in tqdm(sorted(style_dirs), desc="Styles"):
        style_name = style_dir.name
        imgs = sorted([f for f in style_dir.iterdir()
                       if f.suffix.lower() in (".jpg",".jpeg",".png")])
        if not imgs:
            continue

        embeddings = []
        for img_path in imgs:
            try:
                img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model.encode_image(img)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                embeddings.append(emb.cpu().float())
            except Exception:
                continue

        if embeddings:
            all_emb  = torch.cat(embeddings, dim=0)  # (n, 512)
            mean_emb = all_emb.mean(dim=0, keepdim=True)
            mean_emb = mean_emb / mean_emb.norm(dim=-1, keepdim=True)
            style_cache[style_name] = {
                "mean": mean_emb,
                "all":  all_emb,
            }

    print(f"Cached {len(style_cache)} styles")
    return style_cache


# ─────────────────────────────────────────────────────────────────────────────
# Score one instruction against a style
# ─────────────────────────────────────────────────────────────────────────────

def clip_score(model, tokenizer_fn, instruction, style_name,
               style_cache, device):
    """
    Returns mean and max cosine similarity between instruction
    and the style reference images.
    """
    if style_name not in style_cache:
        return None, None

    # Encode text
    text = clip.tokenize([instruction[:77]], truncate=True).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(text)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb.cpu().float()

    # Cosine similarity vs all style images
    all_emb = style_cache[style_name]["all"]  # (n, 512)
    sims    = (all_emb @ text_emb.T).squeeze()  # (n,)

    return float(sims.mean()), float(sims.max())


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — CLIP comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_clip_comparison(clip_summary, att_summary, out_path):
    """
    Side-by-side comparison: CLIP score vs label attention mass per template.
    """
    tmpls = [t for t in TMPL_ORDER if t in clip_summary]
    clip_means = [clip_summary[t]["clip_mean"] for t in tmpls]
    att_means  = [att_summary[t] * 100 for t in tmpls]
    labels     = [TMPL_LABELS.get(t, t) for t in tmpls]

    x  = np.arange(len(tmpls))
    w  = 0.38

    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - w/2, clip_means, w,
                    color=["#1D9E75" if t == "E" else
                           "#4A90D9" if t == "A" else "#B0BEC5"
                           for t in tmpls],
                    label="CLIP alignment (mean cosine sim)", alpha=0.9)
    bars2 = ax2.bar(x + w/2, att_means, w,
                    color=["#F6A623" if t == "E" else
                           "#7B8794" if t == "A" else "#CFD8DC"
                           for t in tmpls],
                    label="Label attention mass %", alpha=0.9)

    ax1.set_ylabel("CLIP cosine similarity", fontsize=10)
    ax2.set_ylabel("Label attention mass (%)", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax1.set_title(
        "CLIP alignment vs label attention mass per template\n"
        "Mistral-7B | 229 regions | green=E winner on attention | blue=baseline A",
        fontsize=11
    )

    # Annotate bars
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.001,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                 f"{h:.2f}%", ha="center", va="bottom", fontsize=7.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load CLIP
    print("Loading CLIP ViT-B/32...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # Load all template results
    all_results = load_all_results(RESULTS_FILES)

    # Cache style embeddings
    style_cache = cache_style_embeddings(model, preprocess, device)

    # Score all templates
    clip_scores  = {}  # template -> list of scores
    clip_summary = {}
    att_summary  = {}  # template -> mean label mass

    for tmpl in tqdm(TMPL_ORDER, desc="Templates"):
        if tmpl not in all_results:
            continue

        regions   = all_results[tmpl]
        scores    = []
        max_scores = []
        label_masses = []

        for r in regions:
            instr = r.get("instruction", "")
            style = r.get("style_name",  "")

            # Skip refusals
            is_refusal = any(p in instr.lower() for p in [
                "i'm an ai","i cannot","language model","as an ai"
            ])
            if is_refusal:
                continue

            mean_sim, max_sim = clip_score(
                model, None, instr, style, style_cache, device
            )
            if mean_sim is not None:
                scores.append(mean_sim)
                max_scores.append(max_sim)
                label_masses.append(r["label_attention_mass"])

        if scores:
            clip_scores[tmpl]  = scores
            clip_summary[tmpl] = {
                "clip_mean":   round(float(np.mean(scores)),     4),
                "clip_median": round(float(np.median(scores)),   4),
                "clip_max":    round(float(np.mean(max_scores)), 4),
                "n_scored":    len(scores),
                "n_skipped":   len(regions) - len(scores),
            }
            att_summary[tmpl] = float(np.mean(label_masses))

    # Print summary table
    print(f"\n{'='*65}")
    print("CLIP alignment summary — ranked by CLIP score")
    print(f"{'='*65}")
    print(f"  {'Tmpl':<6} {'CLIP mean':>11} {'CLIP max':>10} "
          f"{'Att mass':>10} {'Scored':>8} {'Skipped':>9}")
    print(f"  {'-'*60}")
    ranked = sorted(clip_summary.items(),
                    key=lambda x: x[1]["clip_mean"], reverse=True)
    for tmpl, s in ranked:
        print(f"  {tmpl:<6} {s['clip_mean']:>10.4f} {s['clip_max']:>10.4f} "
              f"{att_summary.get(tmpl,0)*100:>9.3f}% "
              f"{s['n_scored']:>8} {s['n_skipped']:>9}")

    # Save results
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = {
        "model":        "mistral-7b-instruct-v0.2-transformers",
        "clip_model":   "ViT-B/32",
        "summary":      clip_summary,
        "attention":    {t: round(v*100,4) for t,v in att_summary.items()},
    }
    json_path = Path("../results/clip_scores.json")
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nScores saved: {json_path}")

    # Figure 7
    fig_path = out_dir / "fig7_clip_comparison.png"
    plot_clip_comparison(clip_summary, att_summary, fig_path)

    print("\nDone. Copy to laptop:")
    print("  scp -r yvs23@ogg.cs.bath.ac.uk:~/Benchmark_dataset/"
          "prompt_curation/attention_maps/figures/ .")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output",
                   default="../attention_maps/figures/")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
