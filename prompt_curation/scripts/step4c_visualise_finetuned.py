"""
Step 4c — Visualise attention comparison before vs after fine-tuning
=====================================================================
Generates three figures:
  fig8_attention_before_after.png  — grouped heatmap: baseline vs LoRA-A vs LoRA-H
  fig9_variance_comparison.png     — CV before vs after per template
  fig10_improvement_scatter.png    — scatter: baseline vs fine-tuned per template
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


TMPLS = ["A","B","C","D","E","F","G","H","I"]
TMPL_LABELS = {
    "A": "A — baseline",
    "B": "B — region first",
    "C": "C — caption-grounded",
    "D": "D — contrastive",
    "E": "E — question-style",
    "F": "F — chain-of-thought",
    "G": "G — label repetition",
    "H": "H — hybrid",
    "I": "I — explicit framing",
}

BASELINE = {
    "A": {"mean": 0.00632, "cv": 0.668},
    "B": {"mean": 0.00489, "cv": 0.619},
    "C": {"mean": 0.00326, "cv": 1.169},
    "D": {"mean": 0.00502, "cv": 0.599},
    "E": {"mean": 0.00554, "cv": 1.409},
    "F": {"mean": 0.00252, "cv": 1.051},
    "G": {"mean": 0.00470, "cv": 0.565},
    "H": {"mean": 0.00558, "cv": 0.594},
    "I": {"mean": 0.00315, "cv": 1.129},
}


def load_results(path):
    with open(path) as f:
        d = json.load(f)
    return {t: d["summary"][t]["label_mass"] for t in TMPLS if t in d["summary"]}


def plot_before_after_bars(lora_a, lora_h, out_path):
    """
    Grouped bar chart: baseline vs LoRA-A vs LoRA-H per template.
    """
    x     = np.arange(len(TMPLS))
    w     = 0.26
    base  = [BASELINE[t]["mean"]*100 for t in TMPLS]
    a_val = [lora_a[t]["mean"]*100   for t in TMPLS]
    h_val = [lora_h[t]["mean"]*100   for t in TMPLS]

    fig, ax = plt.subplots(figsize=(14, 5))
    b1 = ax.bar(x - w,   base,  w, label="Baseline",
                color="#B0BEC5", alpha=0.9)
    b2 = ax.bar(x,       a_val, w, label="LoRA-A (fine-tuned on A)",
                color="#4A90D9", alpha=0.9)
    b3 = ax.bar(x + w,   h_val, w, label="LoRA-H (fine-tuned on H)",
                color="#1D9E75", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([TMPL_LABELS[t] for t in TMPLS],
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Label attention mass (%)", fontsize=10)
    ax.set_title(
        "Label attention mass before vs after LoRA fine-tuning\n"
        "Mistral-7B | 979 regions | all 9 templates",
        fontsize=11
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Annotate bars
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.2f}%", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def plot_variance_comparison(lora_a, lora_h, out_path):
    """
    CV before vs after — shows variance collapse (or lack thereof).
    """
    x      = np.arange(len(TMPLS))
    w      = 0.26
    base_cv = [BASELINE[t]["cv"]         for t in TMPLS]
    a_cv    = [lora_a[t]["cv"]           for t in TMPLS]
    h_cv    = [lora_h[t]["cv"]           for t in TMPLS]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w, base_cv, w, label="Baseline CV",
           color="#B0BEC5", alpha=0.9)
    ax.bar(x,     a_cv,    w, label="LoRA-A CV",
           color="#4A90D9", alpha=0.9)
    ax.bar(x + w, h_cv,    w, label="LoRA-H CV",
           color="#1D9E75", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([TMPL_LABELS[t] for t in TMPLS],
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Coefficient of Variation (lower = more consistent)", fontsize=10)
    ax.set_title(
        "Attention variance (CV) before vs after fine-tuning\n"
        "Lower CV = more invariant grounding — Yudi's hypothesis test",
        fontsize=11
    )
    ax.legend(fontsize=9)
    ax.axhline(y=np.mean(base_cv), color="#666", linewidth=1.5,
               linestyle="--", alpha=0.5, label=f"Baseline mean CV={np.mean(base_cv):.3f}")
    ax.axhline(y=np.mean(a_cv), color="#4A90D9", linewidth=1.5,
               linestyle=":", alpha=0.7, label=f"LoRA-A mean CV={np.mean(a_cv):.3f}")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def plot_improvement_scatter(lora_a, lora_h, out_path):
    """
    Scatter: baseline label mass (x) vs fine-tuned label mass (y).
    Points above diagonal = improved.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data, name, color in [
        (axes[0], lora_a, "LoRA-A", "#4A90D9"),
        (axes[1], lora_h, "LoRA-H", "#1D9E75"),
    ]:
        base_vals = [BASELINE[t]["mean"]*100 for t in TMPLS]
        ft_vals   = [data[t]["mean"]*100     for t in TMPLS]

        ax.scatter(base_vals, ft_vals, s=120, color=color,
                   zorder=3, edgecolors="white", linewidth=1.5)

        # Diagonal — points above = improved
        lim = max(max(base_vals), max(ft_vals)) * 1.1
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3, linewidth=1)
        ax.fill_between([0, lim], [0, lim], [0, 0],
                        alpha=0.05, color="red", label="worsened")
        ax.fill_between([0, lim], [lim, lim], [0, lim],
                        alpha=0.05, color="green", label="improved")

        # Label points
        for t, bv, fv in zip(TMPLS, base_vals, ft_vals):
            ax.annotate(t, (bv, fv),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=9, color=color)

        n_improved = sum(1 for b, f in zip(base_vals, ft_vals) if f > b)
        ax.set_xlabel("Baseline label mass (%)", fontsize=10)
        ax.set_ylabel("Fine-tuned label mass (%)", fontsize=10)
        ax.set_title(f"{name} — {n_improved}/9 templates improved\n"
                     f"Points above diagonal = improved after fine-tuning",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, linestyle="--")

    plt.suptitle(
        "Baseline vs fine-tuned label attention mass per template\n"
        "Mistral-7B | 979 regions",
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def run():
    lora_a = load_results("../results/attention_lora_A.json")
    lora_h = load_results("../results/attention_lora_H.json")
    out    = Path("../attention_maps/figures/")
    out.mkdir(parents=True, exist_ok=True)

    print("Generating fine-tuning comparison figures...")
    plot_before_after_bars(lora_a, lora_h,
                           out / "fig8_attention_before_after.png")
    plot_variance_comparison(lora_a, lora_h,
                             out / "fig9_variance_comparison.png")
    plot_improvement_scatter(lora_a, lora_h,
                             out / "fig10_improvement_scatter.png")
    print("\nDone. Copy to laptop:")
    print("  scp -r yvs23@ogg.cs.bath.ac.uk:~/Benchmark_dataset/"
          "prompt_curation/attention_maps/figures/ .")


if __name__ == "__main__":
    run()
