"""
Step 1c — Attention heatmap figures for dissertation
=====================================================
Generates publication-quality matplotlib figures.

Figure 1: grouped attention bar chart — all 10 regions, token groups
Figure 2: individual full token heatmap per region
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def load_data(att_json, n_low=5, n_high=5):
    with open(att_json) as f:
        data = json.load(f)
    per_region = data["per_region"]
    sorted_r = sorted(per_region, key=lambda x: x["label_attention_mass"] or 0)
    return sorted_r[:n_low] + sorted_r[-n_high:], data["summary"]


def plot_comparison_grid(regions, out_path):
    """
    Grouped attention heatmap.
    Rows = regions, Columns = token groups.
    """
    group_names = [
        "BOS\n<s>",
        "opening\n[INST]...",
        "boilerplate\n(You are...)",
        "region\nlabel",
        "task text\n(Write one...)",
        "closing\n[/INST]",
    ]

    n_r = len(regions)
    n_g = len(group_names)
    grid = np.zeros((n_r, n_g))
    row_labels = []

    for ri, reg in enumerate(regions):
        weights = reg.get("att_weights", [])
        label_i = reg.get("label_token_indices", [])
        label   = reg["region_label"]
        mass    = reg["label_attention_mass"]
        n       = len(weights)

        short = (label[:32] + "…") if len(label) > 32 else label
        row_labels.append(f"{short}\n(mass={mass*100:.2f}%)")

        if not weights:
            continue

        # Group 0: BOS
        grid[ri, 0] = weights[0] * 100

        # Group 1: opening [INST] ] You are
        grid[ri, 1] = sum(weights[1:5]) * 100 if n > 4 else 0

        # Group 2: boilerplate "a style transfer assistant. Region :"
        pre_label = min(label_i[0] if label_i else 13, n)
        grid[ri, 2] = sum(weights[5:pre_label]) * 100 if pre_label > 5 else 0

        # Group 3: region label tokens
        grid[ri, 3] = sum(weights[i] for i in label_i if i < n) * 100

        # Group 4: task text (after label up to closing)
        post_label = (label_i[-1] + 1) if label_i else 14
        grid[ri, 4] = sum(weights[post_label:max(0,n-4)]) * 100 if n > post_label + 4 else 0

        # Group 5: closing [/INST]
        grid[ri, 5] = sum(weights[max(0,n-4):]) * 100 if n >= 4 else 0

    fig, ax = plt.subplots(figsize=(11, n_r * 0.85 + 2.0))
    im = ax.imshow(grid, aspect="auto", cmap="YlOrRd", vmin=0)

    cbar = plt.colorbar(im, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label("Attention weight (%)", fontsize=10)

    ax.set_xticks(range(n_g))
    ax.set_xticklabels(group_names, fontsize=9)
    ax.set_yticks(range(n_r))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax.set_title(
        "Attention distribution by token group — Mistral-7B baseline (template A)\n"
        "10 sampled regions from 979-region dataset",
        fontsize=11, pad=28
    )

    # Annotate cells
    for ri in range(n_r):
        for gi in range(n_g):
            val = grid[ri, gi]
            ax.text(gi, ri, f"{val:.1f}%",
                    ha="center", va="center", fontsize=8,
                    color="white" if val > 25 else "#333333")

    # Highlight region label column
    ax.add_patch(patches.Rectangle(
        (2.5, -0.5), 1, n_r,
        linewidth=2.5, edgecolor="#e53e3e",
        facecolor="none", zorder=3
    ))

    # Divider between LOW and HIGH groups
    ax.axhline(y=4.5, color="#4a90d9", linewidth=1.5, linestyle="--", alpha=0.6)
    ax.text(n_g - 0.4, 4.5, "LOW / HIGH", fontsize=8,
            color="#4a90d9", va="center", ha="right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_single_region(reg, out_path):
    """Full token heatmap for one region."""
    weights = reg.get("att_weights", [])
    tokens  = reg.get("tokens_decoded", [])
    label_i = reg.get("label_token_indices", [])
    label   = reg["region_label"]
    style   = reg["style_name"]
    mass    = reg["label_attention_mass"]

    if not weights or not tokens:
        return

    n = len(tokens)
    w = np.array(weights)
    # Scale excluding BOS
    non_bos = w[1:] if n > 1 else w
    vmax = float(non_bos.max()) if non_bos.size > 0 else 1.0

    cell_w = max(0.28, min(0.45, 12.0 / n))
    fig_w  = max(10, n * cell_w + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, 2.4))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for i, (tok, wi) in enumerate(zip(tokens, w)):
        is_bos   = (i == 0)
        is_label = (i in label_i)
        norm     = 0.0 if is_bos else min(float(wi) / (vmax + 1e-8), 1.0)

        if is_bos:
            fc = (0.82, 0.82, 0.80)
            ec = "#bbbbbb"
            lw = 0.5
        elif is_label:
            fc = (1.0 - norm * 0.55, 1.0 - norm * 0.25 + 0.05 * norm, 1.0 - norm * 0.9)
            ec = "#e53e3e"
            lw = 2.0
        else:
            fc = (1.0, 1.0 - norm * 0.82, 1.0 - norm)
            ec = "#cccccc"
            lw = 0.5

        rect = patches.FancyBboxPatch(
            (i - 0.42, 0.22), 0.84, 0.56,
            boxstyle="round,pad=0.02",
            facecolor=fc, edgecolor=ec, linewidth=lw,
        )
        ax.add_patch(rect)

        # Token label rotated
        disp = (tok[:5] + "…") if len(tok) > 5 else tok
        ax.text(i, 0.86, disp, ha="center", va="bottom",
                fontsize=6.5, fontfamily="monospace",
                rotation=45, color="#333333")

        # Percentage
        pct = float(wi) * 100
        if pct >= 1.0:
            ax.text(i, 0.50, f"{pct:.1f}%", ha="center", va="center",
                    fontsize=5.5,
                    color="#111111" if norm > 0.5 else "#555555")

        # Label marker
        if is_label:
            ax.text(i, 0.24, "▲", ha="center", va="top",
                    fontsize=5.5, color="#e53e3e")

    short = (label[:55] + "…") if len(label) > 55 else label
    fig.suptitle(
        f'region: "{short}" | style: {style} | label mass: {mass*100:.3f}%',
        fontsize=9, y=0.98
    )
    ax.text(0.0, 0.02,
            "▲ = label tokens   grey = BOS   scale excludes BOS",
            transform=ax.transAxes, fontsize=7, color="#888888")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def plot_token_bar(regions, out_path):
    """
    Horizontal bar chart comparing label attention mass across all 10 regions.
    Clean summary figure for the dissertation.
    """
    labels  = [(r["region_label"][:35] + "…") if len(r["region_label"]) > 35
               else r["region_label"] for r in regions]
    masses  = [r["label_attention_mass"] * 100 for r in regions]
    colors  = ["#e53e3e" if m < 0.5 else "#f6a623" if m < 1.5 else "#1D9E75"
               for m in masses]

    fig, ax = plt.subplots(figsize=(8, len(regions) * 0.55 + 1.5))
    bars = ax.barh(range(len(regions)), masses, color=colors, height=0.65)

    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Label attention mass (%)", fontsize=10)
    ax.set_title(
        "Label attention mass per region — Mistral-7B baseline\n"
        "Red < 0.5%   Orange 0.5–1.5%   Green > 1.5%",
        fontsize=11
    )
    ax.axvline(x=1.14, color="#4a90d9", linewidth=1.5,
               linestyle="--", label="mean=1.14%")
    ax.legend(fontsize=9)

    for i, (bar, val) in enumerate(zip(bars, masses)):
        ax.text(val + 0.02, i, f"{val:.3f}%",
                va="center", fontsize=8, color="#333333")

    ax.set_xlim(0, max(masses) * 1.25)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def run(args):
    print("Loading attention data...")
    regions, summary = load_data(args.attention_json)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Mean label mass: {summary['label_attention_mass']['mean']*100:.3f}%")

    print("\nFigure 1: grouped attention grid...")
    plot_comparison_grid(regions, out / "fig1_attention_group_grid.png")

    print("\nFigure 2: label mass bar chart...")
    plot_token_bar(regions, out / "fig2_label_mass_bars.png")

    print("\nFigure 3: individual token heatmaps...")
    for i, reg in enumerate(regions):
        tag = "low" if i < 5 else "high"
        fname = f"fig3_{tag}_{i:02d}_{reg['image_id']}.png"
        plot_single_region(reg, out / fname)

    print(f"\nAll figures in {out}/")
    print("Copy to laptop:")
    print(f"  scp -r yvs23@ogg.cs.bath.ac.uk:~/Benchmark_dataset/prompt_curation/attention_maps/figures/ .")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attention_json",
        default="../attention_maps/baseline_mistral.json")
    p.add_argument("--output",
        default="../attention_maps/figures/")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
