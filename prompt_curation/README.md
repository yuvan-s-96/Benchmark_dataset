# Prompt Curation for Regional Style Transfer

**MSc Data Science Dissertation — University of Bath 2026**


---

## ⚠ Current Status — Regeneration in Progress

Instructions are being regenerated with a **256-token limit** (previously 80 tokens caused 82% truncation, biasing comparisons towards Template C). After regeneration: LoRA models will be retrained, LLM judge and InstructPix2Pix 9×4 will be rerun.

**Attention mass and causal test results STAND** — they measure prompt properties, not generated instruction content.

---

## Overview

Attention-guided prompt curation for instruction-following LLMs in regional style transfer. Two-stage framework:

- **Stage 1** — Attention probe: compare nine prompt templates across 979 COCO regions using Mistral-7B-Instruct-v0.2. Identify which template most causally grounds the model on the label token.
- **Stage 2** — LoRA fine-tuning: fine-tune on instructions from the most grounded template. Evaluate downstream stylisation quality via InstructPix2Pix.

---

## Key Findings (Stable — Not Affected by Regeneration)

| Finding | Result | Status |
|---|---|---|
| Attention probe validated | Att mass vs causal drop r=0.804, p=0.0089 | ✓ STANDS |
| Template A most grounded | 68pp label drop, 6.172% corrected att mass | ✓ STANDS |
| Refusal elimination | E: 15.4%→1.0% (LoRA-A) without explicit suppression | ✓ STANDS |
| Caption confound confirmed | E +69pp delta, A +19.5pp, C/G genuine 0pp | ✓ STANDS |
| Background-region trade-off | Causal drop vs NonReg r=+0.812, p=0.008 | ✓ STANDS |
| LoRA-C best on attention mass | 9/9 templates improved (all p<1e-130***) | ✓ STANDS |
| FT degrades att-grounding corr | Base r=0.796* → LoRA-C r=0.463 ns | ✓ STANDS |
| C wins quality + downstream | Claude 17.50, Gemini 17.80, RC=0.5219 | ⚠ PENDING v2 |

---

## Pipeline

```
step1_generate_and_attend.py      Baseline attention (979 regions)        ✓ STANDS
step2_curate_prompts.py           9 templates × 979 regions               ⟳ REGENERATING (256 tokens)
step2e_clip_scoring.py            Per-region CLIP + bootstrap CI          PENDING v2
step3_finetune_lora.py            LoRA-A, LoRA-H, LoRA-C                  PENDING retrain
step4_attention_finetuned.py      Post-FT attention (3 LoRA variants)     PENDING v2
step5_sink_corrected_metric.py    Sink-corrected attention                 ✓ STANDS
step5b_causal_test.py             Causal label ablation (base + LoRA)     ✓ STANDS
step5c_caption_masked_causal.py   Caption confound test                   ✓ STANDS
step6_llm_judge.py                Claude + Gemini image-grounded          PENDING v2
step7_instruct_pix2pix.py         InstructPix2Pix 9×4 (correct masks)    PENDING v2
step7b_instruct_pix2pix_lora.py   LoRA variants downstream                PENDING v2
step8_artfid_gram.py              Gram/LPIPS/ArtFID                       PENDING v2
```

---

## Stable Results

### Template Comparison (979 regions, Mistral-7B, corrected attention mass)

| Template | Att mass (corr) | Causal drop | Caption confound | CLIP | Refusal% |
|---|---|---|---|---|---|
| A — baseline | **6.172%** | **68.0pp** | Partial (+19.5pp) | 0.2104 | 0% |
| E — question | 2.693% | 21.0pp* | Strong (+69pp) | **0.2263** | 15.4% |
| H — hybrid | 2.594% | 1.0pp | Weak (+5.5pp) | 0.2122 | 0% |
| C — caption | 1.352% | 0.0pp | None | 0.2061 | 0% |
| F — chain-of-thought | 1.368% | 0.0pp | None | 0.2103 | 0% |
| G — label repeat | 1.752% | 0.5pp | None | 0.2026 | 0% |

*E grounding is largely a caption confound — see caption-masked causal test.

Wilcoxon A vs all: p=8.84e-162 to p=2.65e-154 (all ***). Cohen's d 0.875–1.342.

### Background-Region Trade-Off (v1 — correct masks)

| Template | Region CLIP | NonRegion CLIP | Interpretation |
|---|---|---|---|
| F | 0.5238 | 0.7985 | Best stylisation, more leakage |
| A | 0.5185 | **0.8379** | Less stylisation, best preservation |
| C | 0.5219 | 0.7170 | Good stylisation, most leakage |

Causal drop vs NonReg CLIP: r=+0.812, p=0.008 (suggestive, n=9).
Att mass vs Region CLIP: r=-0.895, p=0.001 (suggestive, n=9).

### LoRA Attention Mass (stable — measures prompt grounding)

| Template | Base | LoRA-A | LoRA-H | LoRA-C |
|---|---|---|---|---|
| A | 0.632% | 0.679%*** | 0.599%*** | **0.847%***↑ |
| H | 0.558% | 0.529%*** | 0.386%***↓ | 0.638%*** |
| C | 0.326% | 0.374%*** | 0.340%*** | 0.441%*** |

LoRA-C improves 9/9 templates (all p<1e-130***). LoRA-H degrades H catastrophically (-0.173%, p=4.09e-158***).

---

## Environment

```bash
# On ogg
source /mnt/fast1/yvs23/benchmark_env/bin/activate
export HF_HOME=/mnt/fast1/yvs23/hf_cache
export HF_HUB_DISABLE_XET=1
export CUDA_VISIBLE_DEVICES=1  # or 0-5
cd ~/Benchmark_dataset/prompt_curation/scripts
```

---

## Key File Locations

```
~/Benchmark_dataset/
├── prompt_curation/
│   ├── scripts/           All pipeline scripts
│   ├── results/           All JSON results
│   └── models/lora_A/, lora_H/, lora_C/
├── data/
│   ├── coconut_subset/images/          COCO images
│   ├── coconut_subset/annotations/     COCONut annotations
│   └── style_references/               WikiArt reference images
└── README.md

/mnt/fast1/yvs23/
├── template_comparison_979_weights.json    62MB with att_weights (v1)
├── template_comparison_979_weights_v2.json 256-token version (regenerating)
├── coconut_panoptic/                       COCONut panoptic PNGs (25 images)
│   ├── 000000010909.png ... 000000548209.png
│   ├── coconut_b_panoptic.json
│   └── segment_lookup.json                 Region→segment mapping
└── hf_cache/                               Mistral-7B, InstructPix2Pix
```

### Results Files Status

```
prompt_curation/results/
├── template_comparison_979.json            v1 80-token — being replaced
├── template_comparison_979_v2.json         v2 256-token — regenerating
├── sink_corrected_proper.json              ✓ STANDS
├── causal_test.json                        ✓ STANDS
├── causal_test_lora_A/H/C.json            ✓ STANDS
├── causal_test_caption_masked.json         ✓ STANDS
├── statistical_tests.json                  ✓ STANDS
├── instruct_pix2pix_eval_full_v2.json      v1 correct masks — pending v2
├── pix2pix_lora_A/H/C_all_templates_v2.json v1 correct masks — pending v2
├── instruct_pix2pix_v2_complete.json       v1 correct masks — pending v2
├── artfid_gram_complete.json               v1 — pending v2
├── gemini_image_grounded.json              v1 — pending v2
├── three_way_judge_comparison_v2.json      v1 — pending v2
├── human_rating_indices.json               30 evaluation regions
└── judge_input_full.json                   10 sources, full instructions
```

---

## Regeneration Plan

```
Step 2 v2    Regenerate 979×9 (256 tokens)      ~7 hours    RUNNING
Truncation   Verify <5% truncation               After v2    PENDING
Step 3       Retrain LoRA-A/H/C                 75 min      PENDING
Step 4 v2    Post-FT attention                  9 hours     PENDING
Step 5b v2   Causal test new LoRA               40 hours    PENDING
Step 6 v2    LLM judge (Claude + Gemini)        2 hours     PENDING
Step 7 v2    InstructPix2Pix 9×4               4 hours     PENDING
Step 8 v2    ArtFID/Gram                        6 hours     PENDING
Human rating Session with Deblina + Yudi        After v2    POSTPONED
Writeup      Chapters 3/4/5                     After all   PENDING
```

---

## Dataset

- 200 COCO images from COCONut-PanCap (xdeng77/coconut_pancap)
- 979 segmented regions across 20 WikiArt styles (45–57 regions per style)
- COCONut panoptic masks downloaded from Kaggle (xueqingdeng/coconut)
- 225 empty-caption regions, 754 with full narrative captions

---

*Last updated: July 2026*
