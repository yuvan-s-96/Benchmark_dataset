# Prompt Curation for Regional Style Transfer
**University of Bath MSc Data Science 2026**

---

## Environment

```bash
source ~/benchmark_env/bin/activate  # symlinked to /mnt/fast1/yvs23/benchmark_env
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/mnt/fast1/yvs23/hf_cache
export HF_HUB_DISABLE_XET=1
export TMPDIR=/mnt/fast1/yvs23/tmp
```

---

## Storage

```
~/Benchmark_dataset/prompt_curation/
├── scripts/           ← all pipeline scripts (--model arg on all)
├── attention_maps/    ← baseline_mistral.json, baseline_phi4.json, figures/, figures_phi4/
├── results/           ← all JSON results (large files symlinked to fast1)
├── models/
│   ├── lora_A/        ← Mistral LoRA-A adapter + training_summary.json
│   ├── lora_H/        ← Mistral LoRA-H adapter + training_summary.json
│   ├── phi4_lora_A/   ← Phi-4-mini LoRA-A (in progress)
│   └── phi4_lora_H/   ← Phi-4-mini LoRA-H (in progress)

/mnt/fast1/yvs23/
├── benchmark_env/              ← Python env (symlinked from ~/)
├── hf_cache/                   ← HuggingFace models (~21 GB)
├── template_comparison_979.json    ← Mistral full results (12 MB)
├── template_comparison_phi4.json   ← Phi-4-mini full results
└── annotations/panoptic_train2017/ ← COCONut PNGs
```

---

## Model support

All scripts accept `--model` argument (step1, step1b, step2) and `--base_model` (step3).

**Important:** Phi-4-mini and LLaMA-3.1 use BPE tokenization — different from Mistral's SentencePiece. The multi-strategy label matching fix is applied to all scripts:
```python
candidates = [
    tokenizer.encode(text, add_special_tokens=False),
    tokenizer.encode(" " + text, add_special_tokens=False),
]
```

---

## Step 1 — Generate instructions + extract attention

```bash
# Mistral-7B (DONE)
python3 step1_generate_and_attend.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --json ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output ../attention_maps/baseline_mistral.json \
    --max_regions 0

# Phi-4-mini (DONE)
python3 step1_generate_and_attend.py \
    --model microsoft/Phi-4-mini-instruct \
    --json ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output ../attention_maps/baseline_phi4.json \
    --max_regions 0

# LLaMA-3.1-8B (PENDING — needs HF access approval)
python3 step1_generate_and_attend.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --json ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output ../attention_maps/baseline_llama31.json \
    --max_regions 0
```

**Results:**
| Model | Label mass | Style mass | Runtime |
|---|---|---|---|
| Mistral-7B | 0.63% | 0.57% | ~9 hours |
| Phi-4-mini | 0.94% | 0.81% | 1h 51min |

---

## Step 1b — HTML visualisations

```bash
python3 step1b_visualise_attention.py \
    --model microsoft/Phi-4-mini-instruct \
    --attention_json ../attention_maps/baseline_phi4.json \
    --ann_json ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --pan_json /mnt/fast1/yvs23/annotations/panoptic_train2017.json \
    --pan_dir  /mnt/fast1/yvs23/annotations/panoptic_train2017 \
    --img_dir  ../../data/coconut_subset/images \
    --output   ../attention_maps/visualisations_phi4/ --n 10
```

---

## Step 1c — Heatmap figures

```bash
python3 step1c_heatmap_figure.py \
    --attention_json ../attention_maps/baseline_phi4.json \
    --output ../attention_maps/figures_phi4/
```

---

## Step 2 — Prompt curation (9 templates)

```bash
# Write output to fast1 to avoid home quota issues
python3 step2_curate_prompts.py \
    --model microsoft/Phi-4-mini-instruct \
    --json ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output /mnt/fast1/yvs23/template_comparison_phi4.json \
    --max_regions 0
```

**Results — Mistral-7B vs Phi-4-mini (template A):**
| Model | Label mass | CLIP | CV |
|---|---|---|---|
| Mistral-7B | 0.632% | 0.2104 | 0.668 |
| Phi-4-mini | 0.938% | 0.2028 | 0.963 |

**Key finding:** Template A wins on both models. Phi-4-mini higher label mass but lower CLIP and higher CV than Mistral.

---

## Step 2e — CLIP scoring

```bash
python3 step2e_clip_scoring.py \
    --main_results /mnt/fast1/yvs23/template_comparison_phi4.json \
    --ei_results   /mnt/fast1/yvs23/template_comparison_phi4.json \
    --output_json  ../results/clip_scores_phi4.json \
    --output       ../attention_maps/figures_phi4/
```

---

## Step 3 — LoRA fine-tuning

```bash
# Mistral LoRA-A (DONE)
python3 step3_finetune_lora.py \
    --template A \
    --base_model mistralai/Mistral-7B-Instruct-v0.2 \
    --results_json /mnt/fast1/yvs23/template_comparison_979.json \
    --output ../models/lora_A/ --epochs 3

# Phi-4-mini LoRA-A (RUNNING)
python3 step3_finetune_lora.py \
    --template A \
    --base_model microsoft/Phi-4-mini-instruct \
    --results_json /mnt/fast1/yvs23/template_comparison_phi4.json \
    --output ../models/phi4_lora_A/ --epochs 3

# Same pattern for LoRA-H (replace A with H, lora_A with lora_H)
```

---

## Step 4 — Attention extraction on fine-tuned models

```bash
python3 step4_attention_finetuned.py \
    --adapter   ../models/phi4_lora_A/adapter \
    --lora_name phi4_lora_A \
    --output    ../results/attention_phi4_lora_A.json
```

---

## Step 4b — CLIP on fine-tuned outputs

```bash
python3 step4b_clip_finetuned.py
# Update RESULTS_FILES in script to point to phi4 attention results
```

---

## Cross-model comparison summary

| Metric | Mistral-7B | Phi-4-mini | LLaMA-3.1 |
|---|---|---|---|
| Mean label mass | 0.455% | 0.575% | PENDING |
| Mean CV | 0.867 | 1.126 | PENDING |
| Mean CLIP | 0.2094 | 0.2030 | PENDING |
| Best template (label) | A | A | PENDING |
| Best template (CLIP) | E | E | PENDING |

---

## Troubleshooting

**Disk quota exceeded:**
```bash
# Write results directly to fast1
--output /mnt/fast1/yvs23/filename.json
# Then symlink back
ln -sf /mnt/fast1/yvs23/filename.json ~/Benchmark_dataset/prompt_curation/results/
```

**Label mass showing 0.000 (BPE tokenizer models):**
The multi-strategy fix is already in all scripts. If still failing, check:
```python
candidates = [
    tokenizer.encode(label, add_special_tokens=False),
    tokenizer.encode(" " + label, add_special_tokens=False),
]
```

**tmux for long runs:**
```bash
tmux new -s jobname     # start
# Ctrl+B then D          # detach
tmux attach -t jobname  # reattach
tmux capture-pane -t jobname -p | tail -10  # quick check
```

**Git push rejected:**
```bash
git pull origin main --no-rebase && git push
```
