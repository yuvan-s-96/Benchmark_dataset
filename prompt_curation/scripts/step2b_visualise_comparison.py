"""
Step 2b — Visualise template comparison results
================================================
Generates three figures from template_comparison_mistral.json:

  fig4_template_heatmap.png     — attention by token group per template
  fig5_label_style_scatter.png  — label vs style mass trade-off
  fig6_per_region_improvement.png — template E vs A per region

Usage:
    python3 step2b_visualise_comparison.py \
        --results ../results/template_comparison_mistral.json \
        --output  ../attention_maps/figures/
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


TEMPLATES  = ["A","B","C","D","E","F","G","H"]
TEMPLATE_LABELS = {
    "A": "A — baseline",
    "B": "B — region first",
    "C": "C — caption-grounded",
    "D": "D — contrastive",
    "E": "E — question-style ★",
    "F": "F — chain-of-thought",
    "G": "G — label repetition",
    "H": "H — hybrid (E+G)",
}
COLORS = {
    "E": "#1D9E75",   # green — winner
    "A": "#4A90D9",   # blue — baseline
    "F": "#E53E3E",   # red — worst
}
DEFAULT_COLOR = "#718096"


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Template comparison heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_template_heatmap(summary, out_path):
    """
    Rows = templates A–H
    Columns = label mass | style mass | combined
    Shows relative improvement over baseline A
    """
    tmpl_order = ["E","A","C","B","D","H","G","F"]  # ranked by label mass

    label_means = [summary[t]["label_mass"]["mean"] * 100 for t in tmpl_order]
    style_means = [summary[t]["style_mass"]["mean"] * 100 for t in tmpl_order]
    combined    = [(l + s) / 2 for l, s in zip(label_means, style_means)]

    grid = np.array([label_means, style_means, combined]).T  # (8, 3)
    row_labels = [TEMPLATE_LABELS[t] for t in tmpl_order]
    col_labels = ["Label mass\n(COCO grounding)",
                  "Style mass\n(WikiArt grounding)",
                  "Combined\nmean"]

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(grid, aspect="auto", cmap="YlOrRd", vmin=0,
                   vmax=max(max(label_means), max(style_means)) * 1.1)

    plt.colorbar(im, ax=ax, shrink=0.6, label="Attention mass (%)")
    ax.set_xticks(range(3))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(tmpl_order)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax.set_title(
        "Template comparison — attention mass by grounding target\n"
        "Mistral-7B-Instruct-v0.2 | 979 regions | ranked by label mass",
        fontsize=11, pad=28
    )

    # Annotate cells
    for ri in range(len(tmpl_order)):
        for ci in range(3):
            val = grid[ri, ci]
            ax.text(ci, ri, f"{val:.2f}%",
                    ha="center", va="center", fontsize=9,
                    color="white" if val > grid.max() * 0.6 else "#333333")

    # Highlight winner row
    ax.add_patch(patches.Rectangle(
        (-0.5, -0.5), 3, 1,
        linewidth=2.5, edgecolor="#1D9E75",
        facecolor="none", zorder=3
    ))
    # Highlight baseline row
    ax.add_patch(patches.Rectangle(
        (-0.5, 0.5), 3, 1,
        linewidth=1.5, edgecolor="#4A90D9",
        facecolor="none", zorder=3, linestyle="--"
    ))

    ax.text(2.55, 0, "★ best label", fontsize=8, color="#1D9E75",
            va="center", ha="left")
    ax.text(2.55, 1, "baseline", fontsize=8, color="#4A90D9",
            va="center", ha="left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Label vs style scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_label_style_scatter(summary, out_path):
    """
    X = label mass, Y = style mass
    One point per template — shows trade-off
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for t in TEMPLATES:
        lm = summary[t]["label_mass"]["mean"] * 100
        sm = summary[t]["style_mass"]["mean"] * 100
        color = COLORS.get(t, DEFAULT_COLOR)
        size  = 180 if t in ("E", "A") else 120
        ax.scatter(lm, sm, s=size, color=color, zorder=3,
                   edgecolors="white", linewidth=1.5)
        ax.annotate(
            TEMPLATE_LABELS[t],
            (lm, sm),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8.5,
            color=color if t in COLORS else "#444444",
        )

    # Ideal quadrant arrow
    ax.annotate("", xy=(2.0, 1.8), xytext=(1.2, 1.0),
                arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=1.2))
    ax.text(2.0, 1.85, "ideal", fontsize=8, color="#aaaaaa", ha="center")

    ax.set_xlabel("Label attention mass % — COCO region grounding", fontsize=10)
    ax.set_ylabel("Style attention mass % — WikiArt style grounding", fontsize=10)
    ax.set_title(
        "Label vs style grounding trade-off across templates\n"
        "Mistral-7B | 979 regions | no template maximises both simultaneously",
        fontsize=11
    )
    ax.grid(True, alpha=0.3, linestyle="--")

    # Quadrant lines at baseline A values
    baseline_lm = summary["A"]["label_mass"]["mean"] * 100
    baseline_sm = summary["A"]["style_mass"]["mean"] * 100
    ax.axvline(x=baseline_lm, color="#4A90D9", linestyle="--",
               alpha=0.5, linewidth=1, label=f"Baseline A label={baseline_lm:.2f}%")
    ax.axhline(y=baseline_sm, color="#4A90D9", linestyle=":",
               alpha=0.5, linewidth=1, label=f"Baseline A style={baseline_sm:.2f}%")
    ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Per-region improvement E vs A
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_region_improvement(per_template, out_path):
    """
    For each of 979 regions: delta = H_label_mass - A_label_mass
    Sorted bar chart showing improvement is consistent, not driven by outliers
    """
    regions_A = {
        (r["image_id"], r["mask_index"]): r["label_attention_mass"]
        for r in per_template["A"]
    }
    regions_E = {
        (r["image_id"], r["mask_index"]): r["label_attention_mass"]
        for r in per_template["H"]
    }

    keys   = sorted(set(regions_A.keys()) & set(regions_E.keys()))
    deltas = [(regions_E[k] - regions_A[k]) * 100 for k in keys]
    deltas.sort()

    positive = sum(1 for d in deltas if d > 0)
    negative = sum(1 for d in deltas if d <= 0)

    colors = ["#1D9E75" if d > 0 else "#E53E3E" for d in deltas]

    fig, ax = plt.subplots(figsize=(14, 4))
    x = range(len(deltas))
    ax.bar(x, deltas, color=colors, width=1.0, linewidth=0)
    ax.axhline(y=0, color="black", linewidth=0.8)

    mean_delta = np.mean(deltas)
    ax.axhline(y=mean_delta, color="#1D9E75", linewidth=1.5,
               linestyle="--", label=f"Mean improvement: +{mean_delta:.3f}%")

    ax.set_xlabel("Regions (sorted by improvement)", fontsize=10)
    ax.set_ylabel("Label mass change (E − A) %", fontsize=10)
    ax.set_title(
        f"Per-region improvement: Template E vs Baseline A\n"
        f"{positive}/{len(deltas)} regions improved  |  "
        f"{negative}/{len(deltas)} regions worsened  |  "
        f"Mean delta: +{mean_delta:.3f}%",
        fontsize=11
    )
    ax.legend(fontsize=9)
    ax.set_xlim(-1, len(deltas))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    with open(args.results) as f:
        data = json.load(f)

    summary      = data["summary"]
    per_template = data["per_template"]
    out          = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("Generating figures...")
    plot_template_heatmap(summary, out / "fig4_template_heatmap.png")
    plot_label_style_scatter(summary, out / "fig5_label_style_scatter.png")
    plot_per_region_improvement(per_template, out / "fig6_per_region_improvement.png")

    print(f"\nAll figures saved to {out}/")
    print("Copy to laptop:")
    print("  scp -r yvs23@ogg.cs.bath.ac.uk:~/Benchmark_dataset/prompt_curation/attention_maps/figures/ .")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results",
        default="../results/template_comparison_mistral.json")
    p.add_argument("--output",
        default="../attention_maps/figures/")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
