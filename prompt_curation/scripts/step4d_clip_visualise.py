"""
Step 4d — CLIP before vs after fine-tuning visualisation
=========================================================
Generates fig11_clip_before_after.png
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TMPLS = ["A","B","C","D","E","F","G","H","I"]
TMPL_LABELS = {
    "A": "A — baseline", "B": "B — region first",
    "C": "C — caption", "D": "D — contrastive",
    "E": "E — question", "F": "F — chain-of-thought",
    "G": "G — repetition", "H": "H — hybrid",
    "I": "I — explicit",
}

# Load data
with open("../results/clip_finetuned.json") as f:
    d = json.load(f)

baseline = {t: d["baseline"][t]["clip_mean"] for t in TMPLS if t in d["baseline"]}
lora_a   = {t: d["results"]["LoRA-A"][t]["clip_mean"] for t in TMPLS}
lora_h   = {t: d["results"]["LoRA-H"][t]["clip_mean"] for t in TMPLS}

x  = np.arange(len(TMPLS))
w  = 0.26

base_vals = [baseline.get(t, 0) for t in TMPLS]
a_vals    = [lora_a.get(t, 0)   for t in TMPLS]
h_vals    = [lora_h.get(t, 0)   for t in TMPLS]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ── Left: grouped bar chart ───────────────────────────────────────────────────
ax = axes[0]
b1 = ax.bar(x - w,   base_vals, w, label="Baseline",
            color="#B0BEC5", alpha=0.9)
b2 = ax.bar(x,       a_vals,    w, label="LoRA-A",
            color="#4A90D9", alpha=0.9)
b3 = ax.bar(x + w,   h_vals,    w, label="LoRA-H",
            color="#1D9E75", alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels([TMPL_LABELS[t] for t in TMPLS],
                   rotation=25, ha="right", fontsize=8.5)
ax.set_ylabel("CLIP cosine similarity", fontsize=10)
ax.set_title("CLIP alignment — baseline vs LoRA-A vs LoRA-H\n"
             "Mistral-7B | 979 regions | all 9 templates", fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(0.18, 0.235)
ax.grid(axis="y", alpha=0.3, linestyle="--")

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.0003,
                f"{h:.4f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

# ── Right: delta scatter ──────────────────────────────────────────────────────
ax2 = axes[1]
a_deltas = [(lora_a[t] - baseline[t]) * 1000 for t in TMPLS]
h_deltas = [(lora_h[t] - baseline[t]) * 1000 for t in TMPLS]

x2 = np.arange(len(TMPLS))
w2 = 0.38
ax2.bar(x2 - w2/2, a_deltas, w2, label="LoRA-A delta",
        color=["#4A90D9" if v >= 0 else "#E53E3E" for v in a_deltas], alpha=0.85)
ax2.bar(x2 + w2/2, h_deltas, w2, label="LoRA-H delta",
        color=["#1D9E75" if v >= 0 else "#F6A623" for v in h_deltas], alpha=0.85)

ax2.axhline(y=0, color="black", linewidth=0.8)
ax2.set_xticks(x2)
ax2.set_xticklabels([TMPL_LABELS[t] for t in TMPLS],
                    rotation=25, ha="right", fontsize=8.5)
ax2.set_ylabel("CLIP delta (×1000)", fontsize=10)
ax2.set_title("CLIP change after fine-tuning\n"
              "Positive = improved, negative = degraded", fontsize=11)
ax2.grid(axis="y", alpha=0.3, linestyle="--")

# Custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#4A90D9", label="LoRA-A improved"),
    Patch(facecolor="#E53E3E", label="LoRA-A degraded"),
    Patch(facecolor="#1D9E75", label="LoRA-H improved"),
    Patch(facecolor="#F6A623", label="LoRA-H degraded"),
]
ax2.legend(handles=legend_elements, fontsize=8, loc="lower right")

# Annotate deltas
for i, (da, dh) in enumerate(zip(a_deltas, h_deltas)):
    ax2.text(i - w2/2, da + (0.1 if da >= 0 else -0.3),
             f"{da:+.1f}", ha="center", va="bottom", fontsize=7)
    ax2.text(i + w2/2, dh + (0.1 if dh >= 0 else -0.3),
             f"{dh:+.1f}", ha="center", va="bottom", fontsize=7)

plt.suptitle("CLIP alignment before vs after LoRA fine-tuning — Mistral-7B",
             fontsize=13, y=1.02)
plt.tight_layout()
out = Path("../attention_maps/figures/fig11_clip_before_after.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# Print summary
print("\nCLIP summary:")
print(f"  LoRA-A improved: {sum(1 for d in a_deltas if d > 0)}/9 templates")
print(f"  LoRA-H improved: {sum(1 for d in h_deltas if d > 0)}/9 templates")
print(f"  LoRA-A mean delta: {np.mean(a_deltas):+.2f} ×10⁻³")
print(f"  LoRA-H mean delta: {np.mean(h_deltas):+.2f} ×10⁻³")
