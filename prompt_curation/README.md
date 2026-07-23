# Prompt Curation for Regional Style Transfer
**MSc Data Science Dissertation — University of Bath 2026**

---

## Current Status — Evaluation Pipeline Complete (LoRA Sink Correction Finishing)

Instructions were regenerated with per-template token limits (256/512/768 tokens
as required — see Section 14.1 of the progress document) to fix an 82%
truncation issue found in the original 80-token generation. All downstream
evaluations were rerun on the corrected instructions and are now **complete**:
causal ablation (base + 3 LoRA, with full Wilcoxon significance testing), true
sink-corrected attention mass (base model, all 9 templates), LLM-as-judge
(Claude + Gemini, with inter-rater reliability and a formal grounding
correlation), and InstructPix2Pix downstream evaluation (Region CLIP /
NonRegion CLIP / Gram / LPIPS / ArtFID). **Only the LoRA sink-corrected
attention mass is still running** (across all 6 GPUs).

A significant methodological fix was made during this process: attention-mass
Sink correction had been computed with an inconsistent method (Template A used
true weight renormalisation, Templates B–I used a cruder approximation). This
was found, fixed, and re-run for all 9 templates — see Section 14.2 of the
progress document for the full account, including the resulting downward
revision of a previously-reported correlation (r=0.857 → r=0.533, no longer
significant).

---

## Setup

```bash
# SSH to ogg
ssh yvs23@ogg.cs.bath.ac.uk

# Activate environment
source /mnt/fast1/yvs23/benchmark_env/bin/activate

# Set environment variables
export HF_HOME=/mnt/fast1/yvs23/hf_cache
export HF_HUB_DISABLE_XET=1
export CUDA_VISIBLE_DEVICES=1  # change per GPU (0-5)

cd ~/Benchmark_dataset/prompt_curation/scripts
```

---

## Pipeline Commands

### Step 1 — Baseline Attention, Template A only (DONE ✓ STANDS)
```bash
python3 step1_generate_and_attend.py \
    --json    ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output  ../attention_maps/baseline_mistral.json
```

### Step 1 (unified) — Raw Attention Weights, Any Template Subset, Optional LoRA (DONE ✓ for base model all 9; LoRA in progress)
```bash
python3 step1_full_weights.py \
    --output      ../attention_maps/OUTPUT_NAME.json \
    --templates   B,C,D            # omit for all 9
    --adapter     ../models/lora_A_v3/adapter   # omit for base model
```
Base model: run across GPUs 0–5, one or two templates per GPU (Templates D and
G are the slowest — pair them with fast templates). Runtime ~4–15 hours per
template depending on verbosity of the template's typical output.

### Step 2 — Prompt Curation, All 9 Templates (DONE ✓ — corrected token limits)
```bash
python3 step2_curate_prompts.py \
    --json        ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output      ../results/template_comparison_979_final.json \
    --max_regions 979
```
Token limits are now set per-template inside the script: 256 tokens for most
templates, 512 for D/E/F/H/I, 768 for A and G (which reliably trigger longer
generations from Mistral). Verify truncation rate is near-zero before trusting
downstream results:
```bash
python3 -c "
import json
d = json.load(open('../results/template_comparison_979_final.json'))
for t, regions in d['per_template'].items():
    complete = sum(1 for r in regions if r['instruction'].rstrip().endswith(('.','!','?','\"',')')))
    print(f'{t}: {complete}/{len(regions)} complete ({100*complete/len(regions):.0f}%)')
"
```

### Step 3 — LoRA Fine-Tuning (DONE ✓, one adapter per template A/C/H)
```bash
# GPU 1
python3 step3_finetune_lora.py --template A \
    --results_json ../results/template_comparison_979_final.json \
    --output ../models/lora_A_v3/ --epochs 3

# GPU 2
CUDA_VISIBLE_DEVICES=2 python3 step3_finetune_lora.py --template C \
    --results_json ../results/template_comparison_979_final.json \
    --output ../models/lora_C_v3/ --epochs 3

# GPU 3
CUDA_VISIBLE_DEVICES=3 python3 step3_finetune_lora.py --template H \
    --results_json ../results/template_comparison_979_final.json \
    --output ../models/lora_H_v3/ --epochs 3
```
Runtime: ~25 min each.

### Step 4 — Post Fine-Tuning Attention + Instructions (DONE ✓ — instructions fixed; raw weights via step1_full_weights.py, see above)
```bash
python3 step4_attention_finetuned.py \
    --adapter    ../models/lora_A_v3/adapter \
    --lora_name  lora_A_v3 \
    --base_model mistralai/Mistral-7B-Instruct-v0.2 \
    --output     ../results/attention_lora_A_v3_fixed.json
```
Repeat for `lora_C_v3` and `lora_H_v3`. This produces the corrected instruction
text and scalar attention mass per region; use `step1_full_weights.py` instead
if you need the raw per-token attention weight vector for true sink correction.

### Step 5a — Sink-Corrected Attention Mass (DONE ✓ — base model, all 9 templates, consistent true method)
```bash
python3 step5_sink_corrected_metric.py \
    --attention_json ../attention_maps/baseline_mistral_ALL9_full.json \
    --output         ../results/sink_corrected_ALL9_TRUE.json
```
For LoRA models (uses the same true weight-renormalisation method, adapted for
per-template LoRA data structure):
```bash
python3 step5_sink_corrected_lora.py \
    --attention_json ../attention_maps/lora_A_v3_full.json \
    --output         ../results/sink_corrected_lora_A_v3.json
```

### Step 5b — Causal Ablation Test (DONE ✓ STANDS — base + 3 LoRA, full Wilcoxon testing)
```bash
# Base model
python3 step5b_causal_test.py \
    --inputs_json ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output      ../results/causal_test.json \
    --n_sample    200

# LoRA variants
python3 step5b_causal_test.py \
    --inputs_json ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output      ../results/causal_test_lora_A_v3.json \
    --adapter     ../models/lora_A_v3/adapter \
    --n_sample    200
```
Runtime: ~40 hours each at n=200 (run in parallel across GPUs). A faster
confirmatory run (n=30, 512 tokens) validated the 80-token result is
conservative, not inflated — see `causal_test_lora_A_sanity512.json`.

Wilcoxon signed-rank significance testing (Template A vs each other template,
per-region BLEU-4 and Jaccard values, n=200):
```bash
python3 -c "
import json, numpy as np
from scipy.stats import wilcoxon
d = json.load(open('../results/causal_test.json'))
pt = d['per_template']
a_bleu = np.array([r['bleu4'] for r in pt['A']])
for t in 'BCDEFGHI':
    t_bleu = np.array([r['bleu4'] for r in pt[t]])
    stat, p = wilcoxon(a_bleu, t_bleu)
    print(f'A vs {t}: p={p:.2e}')
"
```

### Step 5c — Caption-Masked Causal Test (DONE ✓ STANDS)
```bash
python3 step5c_caption_masked_causal.py \
    --inputs_json ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output      ../results/causal_test_caption_masked.json \
    --templates   A,C,H,G,E \
    --n_sample    200
```

### Step 6 — LLM-as-Judge (DONE ✓ — Claude + Gemini, 30 regions × 7 sources)
```bash
python3 step6b_lora_inference.py \
    --adapter  ../models/lora_A_v3/adapter \
    --template A \
    --output   ../results/lora_A_v3_tmplA_instructions.json
```
Judge input (`judge_input_v3_final.json`) is built by combining 4 baseline
sources (A, C, E, H) and 3 LoRA-on-own-template sources, with the 25 source
images and ground-truth region/style labels, then submitted manually to
Claude and Gemini with the rubric in Section 14.5.2 of the progress document.
Inter-rater reliability, per-source scores, and the formal grounding
correlation are computed from the two returned JSON score sets (see
`judge_comparison_v3.json` and `judge_grounding_vs_causal_correlation.json`).

### Step 7 — InstructPix2Pix Downstream, 9×4 (DONE ✓ — base + 3 LoRA, all 9 templates)
```bash
# Base model
python3 step7_instruct_pix2pix.py \
    --results_json  ../results/template_comparison_979_final.json \
    --lora_a_json   ../results/attention_lora_A_v3_fixed.json \
    --pan_dir       /mnt/fast1/yvs23/coconut_panoptic \
    --pan_json      ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --indices_json  ../results/human_rating_indices.json \
    --output        ../results/instruct_pix2pix_final.json \
    --n_sample      30

# LoRA variants, all 9 templates each
python3 step7b_instruct_pix2pix_lora.py \
    --adapter      ../models/lora_A_v3/adapter \
    --inputs_json  ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --pan_dir      /mnt/fast1/yvs23/coconut_panoptic \
    --pan_json     ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output       ../results/pix2pix_lora_A_v3_all_templates.json \
    --templates    A,B,C,D,E,F,G,H,I \
    --n_sample     30
```
Note: InstructPix2Pix truncates each instruction to ~60 words before feeding
it to the model (CLIP's 77-token text encoder limit) — this is a controlled,
deliberate truncation, distinct from and unrelated to the earlier Mistral
generation truncation issue.

### Step 8 — ArtFID / Gram / LPIPS (DONE ✓ — base + 3 LoRA, all 9 templates)
```bash
python3 step8_artfid_gram.py \
    --results_json  ../results/template_comparison_979_final.json \
    --lora_a_json   ../results/attention_lora_A_v3_fixed.json \
    --lora_h_json   ../results/attention_lora_H_v3_fixed.json \
    --lora_c_json   ../results/attention_lora_C_v3_fixed.json \
    --indices_json  ../results/human_rating_indices.json \
    --img_dir       ../../data/coconut_subset/images \
    --pan_dir       /mnt/fast1/yvs23/coconut_panoptic \
    --style_ref_dir ../../data/style_references \
    --output        ../results/artfid_gram_final.json \
    --n_sample      30
```

---

## Check tmux Sessions

```bash
tmux ls
tmux capture-pane -t <session_name> -p | tail -10
```

---

## Key File Locations

```
~/Benchmark_dataset/
├── prompt_curation/
│   ├── scripts/                    All pipeline scripts
│   ├── results/                    All JSON results
│   ├── attention_maps/             Raw per-token attention weights, base + LoRA
│   ├── docs/                       prompt_curation_progress_v7.docx (master document)
│   └── models/
│       └── lora_A_v3/, lora_C_v3/, lora_H_v3/   fine-tuned on corrected instructions
├── data/
│   ├── coconut_subset/images/      COCO images
│   ├── coconut_subset/annotations/ COCONut annotations
│   └── style_references/           WikiArt reference images

/mnt/fast1/yvs23/
├── coconut_panoptic/
│   ├── 000000010909.png ... 000000548209.png  COCONut panoptic PNGs
│   ├── coconut_b_panoptic.json
│   └── segment_lookup.json         Region label → panoptic segment mapping
└── hf_cache/                       Mistral-7B, InstructPix2Pix
```

### Results File Status

| File | Status |
|---|---|
| `template_comparison_979_final.json` | ✓ DONE — corrected instructions, source of truth |
| `sink_corrected_ALL9_TRUE.json` | ✓ DONE — base model, all 9 templates, consistent true method |
| `causal_test.json` | ✓ DONE STANDS |
| `causal_test_lora_{A,C,H}_v3.json` | ✓ DONE — all 9 templates each |
| `causal_wilcoxon_A_vs_all.json` / `causal_wilcoxon_all_models.json` | ✓ DONE — 64/64 comparisons significant |
| `causal_test_caption_masked.json` | ✓ DONE STANDS |
| `attention_lora_{A,C,H}_v3_fixed.json` | ✓ DONE — corrected instructions + attention mass |
| `judge_scores_claude_v3.json` / `judge_scores_gemini_v3.json` | ✓ DONE |
| `judge_comparison_v3.json` | ✓ DONE — inter-rater r=0.741 |
| `judge_grounding_vs_causal_correlation.json` | ✓ DONE — r=-0.99 (Claude), r=-0.79 (Gemini) |
| `instruct_pix2pix_final.json` | ✓ DONE — base model, all 9 templates |
| `pix2pix_lora_{A,C,H}_v3_all_templates.json` | ✓ DONE — all 9 templates each |
| `artfid_gram_final.json` | ✓ DONE |
| `sink_corrected_lora_{A,C,H}_v3.json` | ⟳ RUNNING (across 6 GPUs) |

---

## Headline Results Summary

### Causal Grounding (base model + 3 LoRA variants, all statistically confirmed)

| Template | Base causal drop | LoRA-A | LoRA-C | LoRA-H |
|---|---|---|---|---|
| A | **66.0pp** | **65.0pp** | **56.5pp** | **65.0pp** |
| E | 24.0pp | 25.0pp | 24.0pp | 25.0pp |
| B, C, D, F, G, H, I | 0.0–2.0pp | 0.0–1.0pp | 0.0–0.5pp | -0.5–2.0pp |

Template A's advantage over every other template is significant across all 64
pairwise Wilcoxon tests (base + 3 LoRA × 8 comparison templates × BLEU-4 and
Jaccard), 60/64 at p<0.001.

### Sink-Corrected Attention Mass (base model, true consistent method)

| Template | Corrected mass | Correction factor |
|---|---|---|
| A | **6.17%** | 9.77× |
| G | 3.06% | 3.82× |
| B | 2.80% | 3.85× |
| H | 2.62% | 4.68× |
| E | 2.44% | 4.90× |
| D | 2.35% | 4.51× |
| C | 1.68% | 4.22× |
| F | 1.43% | 5.49× |
| I | 1.06% | 3.84× |

Attention mass vs causal drop correlation (base model, n=9 templates): **r=0.533,
p=0.139 — not significant** under the corrected, consistent method (previously
reported as r=0.857*, significant, under an inconsistent mixed method — see
progress document Section 14.2.2 for the full account).

### LLM-as-Judge (Claude + Gemini, 30 regions, 7 sources)

| Source | Claude | Gemini |
|---|---|---|
| baseline_C | 4.51 | 4.93 |
| lora_C_tmplC | 4.46 | 4.89 |
| baseline_A | 3.69 | 3.95 |
| baseline_H | 3.67 | 3.91 |
| lora_H_tmplH | 3.67 | 3.87 |
| baseline_E | 3.71 | 3.63 |
| lora_A_tmplA | 3.66 | 3.87 |

Inter-rater reliability: Pearson r=0.741, Spearman r=0.737 (both p<0.0001),
96.9% agreement within ±1 point.

**Judge grounding score is significantly negatively correlated with actual
causal grounding** (Claude: Pearson r=-0.993, p<0.0001; Gemini: r=-0.794,
p=0.033; n=7). The most causally grounded source (baseline_A) gets the lowest
judge grounding score; near-ungrounded sources get near-perfect judge
grounding scores. See progress document Section 14.5.6.

---

## Dataset

- 200 COCO images from COCONut-PanCap (`xdeng77/coconut_pancap` on HuggingFace)
- 979 segmented regions across 20 WikiArt styles
- COCONut panoptic masks from Kaggle (`xueqingdeng/coconut`)
- Segment lookup built by keyword matching region labels to COCONut categories

---



---

## Known Limitations

- Judge-grounding-vs-causal-grounding correlation (n=7 sources) is a striking
  effect size but a small sample — read as suggestive of a real phenomenon,
  not a definitive population estimate.
- LoRA sink-corrected attention mass is still running at time of writing.
- InstructPix2Pix downstream evaluation uses only the first ~60 words of each
  instruction (CLIP's 77-token text encoder limit) — an inherent, documented
  constraint of the CLIP-conditioned architecture.

---

*Last updated: July 2026*
