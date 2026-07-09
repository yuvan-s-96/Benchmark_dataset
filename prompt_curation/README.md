# Prompt Curation for Regional Style Transfer

**MSc Data Science Dissertation — University of Bath 2026**

---

## ⚠ Current Status — Regeneration in Progress

Instructions are being regenerated with a **256-token limit** (previously 80 tokens caused 82% truncation). After regeneration: LoRA models retrained, LLM judge and InstructPix2Pix 9×4 rerun. **Attention mass and causal test results STAND.**

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

### Step 1 — Baseline Attention (DONE ✓ STANDS)
```bash
python3 step1_generate_and_attend.py \
    --json    ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output  ../results/step1_baseline_attention.json
```

### Step 2 — Prompt Curation 9 Templates (REGENERATING at 256 tokens)
```bash
python3 step2_curate_prompts.py \
    --json        ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output      ../results/template_comparison_979_v2.json \
    --max_regions 979
# Runtime: ~20 hours on 1 GPU. Check progress:
tmux capture-pane -t regen_256 -p | tail -5
```

### Step 2e — CLIP Scoring (PENDING v2)
```bash
python3 step2e_clip_scoring.py \
    --results_json ../results/template_comparison_979_v2.json \
    --output       ../results/clip_scores_v2.json
```

### Step 3 — LoRA Fine-Tuning (PENDING retrain on 256-token instructions)
```bash
# Run all three in parallel on separate GPUs
# GPU 1
python3 step3_finetune_lora.py --template A \
    --results_json ../results/template_comparison_979_v2.json \
    --output ../models/lora_A_v2/ --epochs 3

# GPU 2
CUDA_VISIBLE_DEVICES=2 python3 step3_finetune_lora.py --template H \
    --results_json ../results/template_comparison_979_v2.json \
    --output ../models/lora_H_v2/ --epochs 3

# GPU 3
CUDA_VISIBLE_DEVICES=3 python3 step3_finetune_lora.py --template C \
    --results_json ../results/template_comparison_979_v2.json \
    --output ../models/lora_C_v2/ --epochs 3
# Runtime: ~25 min each
```

### Step 4 — Post Fine-Tuning Attention (PENDING v2)
```bash
python3 step4_attention_finetuned.py \
    --adapter    ../models/lora_A_v2/adapter \
    --lora_name  lora_A \
    --base_model mistralai/Mistral-7B-Instruct-v0.2 \
    --output     ../results/attention_lora_A_v2.json
# Repeat for lora_H_v2 and lora_C_v2. Runtime: ~9 hours each.
```

### Step 5a — Sink-Corrected Attention (DONE ✓ STANDS)
```bash
python3 step5_sink_corrected_metric.py \
    --weights_json /mnt/fast1/yvs23/template_comparison_979_weights.json \
    --output       ../results/sink_corrected_proper.json
```

### Step 5b — Causal Test (DONE ✓ STANDS)
```bash
# Base model
python3 step5b_causal_test.py \
    --inputs_json ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output      ../results/causal_test.json \
    --n_sample    200

# LoRA variants (after retrain)
python3 step5b_causal_test.py \
    --inputs_json ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output      ../results/causal_test_lora_A_v2.json \
    --adapter     ../models/lora_A_v2/adapter \
    --n_sample    200
# Runtime: ~40 hours each. Run in parallel on separate GPUs.
```

### Step 5c — Caption-Masked Causal Test (DONE ✓ STANDS)
```bash
python3 step5c_caption_masked_causal.py \
    --inputs_json ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output      ../results/causal_test_caption_masked.json \
    --templates   A,C,H,G,E \
    --n_sample    200
```

### Step 6 — LLM Judge (PENDING v2)
```bash
# Prepare judge input from v2 instructions
python3 step6_llm_judge.py \
    --results_json  ../results/template_comparison_979_v2.json \
    --lora_a_json   ../results/attention_lora_A_v2.json \
    --indices_json  ../results/human_rating_indices.json \
    --output        ../results/llm_judge_scores_v2.json
# Then run Claude and Gemini via API
```

### Step 7 — InstructPix2Pix 9×4 (PENDING v2)
```bash
# Base model
python3 step7_instruct_pix2pix.py \
    --results_json  ../results/template_comparison_979_v2.json \
    --lora_a_json   ../results/attention_lora_A_v2.json \
    --pan_dir       /mnt/fast1/yvs23/coconut_panoptic \
    --pan_json      ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --indices_json  ../results/human_rating_indices.json \
    --output        ../results/instruct_pix2pix_eval_full_v3.json \
    --n_sample      30

# LoRA variants (after retrain)
python3 step7b_instruct_pix2pix_lora.py \
    --adapter      ../models/lora_A_v2/adapter \
    --inputs_json  ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --pan_dir      /mnt/fast1/yvs23/coconut_panoptic \
    --pan_json     ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output       ../results/pix2pix_lora_A_v2.json \
    --templates    A,B,C,D,E,F,G,H,I \
    --n_sample     30
# Repeat for lora_H_v2 and lora_C_v2. Runtime: ~1 hour each.
```

### Step 8 — ArtFID/Gram (PENDING v2)
```bash
python3 step8_artfid_gram.py \
    --results_json  ../results/template_comparison_979_v2.json \
    --lora_a_json   ../results/attention_lora_A_v2.json \
    --lora_h_json   ../results/attention_lora_H_v2.json \
    --lora_c_json   ../results/attention_lora_C_v2.json \
    --indices_json  ../results/human_rating_indices.json \
    --img_dir       ../../data/coconut_subset/images \
    --pan_dir       /mnt/fast1/yvs23/coconut_panoptic \
    --style_ref_dir ../../data/style_references \
    --output        ../results/artfid_gram_eval_v3.json \
    --n_sample      30
```

---

## Check tmux Sessions

```bash
tmux ls
tmux capture-pane -t regen_256 -p | tail -5
tmux capture-pane -t <session_name> -p | tail -5
```

---

## Key File Locations

```
~/Benchmark_dataset/
├── prompt_curation/
│   ├── scripts/                    All pipeline scripts
│   ├── results/                    All JSON results
│   └── models/
│       ├── lora_A/, lora_H/, lora_C/        v1 (80-token gold)
│       └── lora_A_v2/, lora_H_v2/, lora_C_v2/  v2 (256-token gold, pending)
├── data/
│   ├── coconut_subset/images/      COCO images
│   ├── coconut_subset/annotations/ COCONut annotations
│   └── style_references/           WikiArt reference images

/mnt/fast1/yvs23/
├── template_comparison_979_weights.json    v1 weights (80-token)
├── template_comparison_979_weights_v2.json v2 weights (256-token, pending)
├── coconut_panoptic/
│   ├── 000000010909.png ... 000000548209.png  COCONut panoptic PNGs
│   ├── coconut_b_panoptic.json
│   └── segment_lookup.json         Region label → panoptic segment mapping
└── hf_cache/                       Mistral-7B, InstructPix2Pix
```

### Results File Status

| File | Status |
|---|---|
| `sink_corrected_proper.json` | ✓ STANDS |
| `causal_test.json` | ✓ STANDS |
| `causal_test_lora_A/H/C.json` | ✓ STANDS |
| `causal_test_caption_masked.json` | ✓ STANDS |
| `statistical_tests.json` | ✓ STANDS |
| `template_comparison_979_v2.json` | ⟳ REGENERATING |
| `instruct_pix2pix_eval_full_v2.json` | ⚠ v1 correct masks, pending v2 instructions |
| `artfid_gram_complete.json` | ⚠ v1 pending v2 |
| `gemini_image_grounded.json` | ⚠ v1 pending v2 |

---

## Stable Results Summary

### Attention Mass (corrected, base model)

| Template | Att mass (corr) | Causal drop | Caption confound |
|---|---|---|---|
| A | **6.172%** | **68.0pp** | Partial (+19.5pp) |
| E | 2.693% | 21.0pp | Strong (+69pp) — caption confound |
| H | 2.594% | 1.0pp | Weak (+5.5pp) |
| C | 1.352% | 0.0pp | None — genuinely ungrounded |
| G | 1.752% | 0.5pp | None — genuinely ungrounded |

### Background-Region Trade-Off (v2 correct masks — robust)

| Correlation | r | p | Status |
|---|---|---|---|
| Att mass vs Region CLIP | -0.895 | 0.001** | Robust |
| Causal drop vs Region CLIP | -0.786 | 0.012* | Robust |
| Causal drop vs NonRegion CLIP | +0.812 | 0.008** | Robust |
| NonRegion CLIP vs ArtFID | -0.750 | 0.020* | Robust |

All n=9 templates — treat as suggestive.

---

## Dataset

- 200 COCO images from COCONut-PanCap (`xdeng77/coconut_pancap` on HuggingFace)
- 979 segmented regions across 20 WikiArt styles
- COCONut panoptic masks from Kaggle (`xueqingdeng/coconut`)
- Segment lookup built by keyword matching region labels to COCONut categories

---

*Last updated: July 2026*
