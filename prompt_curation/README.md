# Prompt Curation for Regional Style Transfer
**University of Bath MSc Data Science 2026**

---

## Environment

```bash
source ~/benchmark_env/bin/activate
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/mnt/fast1/yvs23/hf_cache
export HF_HUB_DISABLE_XET=1
export TMPDIR=/mnt/fast1/yvs23/tmp
```

---

## Storage

```
~/Benchmark_dataset/prompt_curation/
├── scripts/          ← all pipeline scripts
├── attention_maps/   ← baseline_mistral.json, baseline_lora_A/H.json, visualisations, figures
├── results/          ← template_comparison_979.json, attention_lora_A/H.json, clip_*.json
├── models/lora_A/    ← LoRA-A adapter (~200 MB, not in git)
└── models/lora_H/    ← LoRA-H adapter (~200 MB, not in git)

/mnt/fast1/yvs23/
├── hf_cache/                    ← Mistral-7B (~14 GB)
├── annotations/panoptic_train2017/  ← COCONut PNGs (1.1 GB)
└── sam2.1_hiera_large.pt        ← SAM2 checkpoint (moved from home)
```

---

## Step 1 — Generate instructions + extract attention (DONE ✓)

```bash
python3 step1_generate_and_attend.py \
    --json ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output ../attention_maps/baseline_mistral.json \
    --max_regions 0
```

**Results:** 979 regions | label mass mean 0.63% | style mass mean 0.57%
BOS token ~55% | region label 0.09–3.82% | caption length major confound

---

## Step 1b — HTML visualisations (DONE ✓)

```bash
python3 step1b_visualise_attention.py \
    --attention_json ../attention_maps/baseline_mistral.json \
    --ann_json ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --pan_json /mnt/fast1/yvs23/annotations/panoptic_train2017.json \
    --pan_dir  /mnt/fast1/yvs23/annotations/panoptic_train2017 \
    --img_dir  ../../data/coconut_subset/images \
    --output   ../attention_maps/visualisations/ --n 10
```

---

## Step 1c — Heatmap figures (DONE ✓)

```bash
python3 step1c_heatmap_figure.py \
    --attention_json ../attention_maps/baseline_mistral.json \
    --output ../attention_maps/figures/
```

---

## Step 2 — Prompt curation, 9 templates (DONE ✓)

```bash
python3 step2_curate_prompts.py \
    --json ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output ../results/template_comparison_979.json \
    --max_regions 0
```

**Results (979 regions):**

| Template | Label mass | Style mass | CLIP | CV | Refusal% |
|---|---|---|---|---|---|
| A — baseline | 0.632% | 0.568% | 0.2104 | 0.668 | 0% |
| H — hybrid | 0.558% | 1.448% | 0.2122 | 0.594 | 0% |
| E — question | 0.554% | 1.420% | 0.2263 | 1.409 | 15.4% |
| F — chain-of-thought | 0.252% | 0.485% | 0.2103 | 1.051 | 0% |

**Key finding:** Template A wins label mass on full dataset. E wins on short prompts only.
Caption length dilutes attention — long prompts systematically reduce label mass.

---

## Step 2b/e — Visualisation + CLIP (DONE ✓)

```bash
python3 step2b_visualise_comparison.py \
    --results ../results/template_comparison_979.json \
    --output  ../attention_maps/figures/

python3 step2e_clip_scoring.py --output ../attention_maps/figures/
```

---

## Step 3 — LoRA fine-tuning (DONE ✓)

```bash
# LoRA-A: fine-tune on template A instructions
python3 step3_finetune_lora.py \
    --template A \
    --results_json ../results/template_comparison_979.json \
    --output ../models/lora_A/ --epochs 3

# LoRA-H: fine-tune on template H instructions (separate GPU)
python3 step3_finetune_lora.py \
    --template H \
    --results_json ../results/template_comparison_979.json \
    --output ../models/lora_H/ --epochs 3
```

**Results:** Both converged in ~25 min. LoRA-H lower final eval loss (0.215 vs 0.262).

---

## Step 4 — Attention extraction on fine-tuned models (DONE ✓)

```bash
python3 step4_attention_finetuned.py \
    --adapter   ../models/lora_A/adapter \
    --lora_name lora_A \
    --output    ../results/attention_lora_A.json

python3 step4_attention_finetuned.py \
    --adapter   ../models/lora_H/adapter \
    --lora_name lora_H \
    --output    ../results/attention_lora_H.json
```

**Four headline findings:**
1. LoRA-A improves 6/9 templates. LoRA-H degrades its own template H (-0.168%) — over-specialisation
2. Style mass improves on 7/9 templates under both runs — WikiArt grounding most improved
3. Template E refusals: 15.4% → 1.0% (LoRA-A), 0.1% (LoRA-H) — eliminated without explicit suppression
4. High-variance templates (E, F, C, I) show CV reduction — partial invariant grounding

---

## Step 4b — CLIP on fine-tuned outputs (DONE ✓)

```bash
python3 step4b_clip_finetuned.py
```

**Results:** LoRA-A improves CLIP on 6/9 templates. LoRA-H improves on 5/9.
Largest gains: G (+0.0054 LoRA-A, +0.0104 LoRA-H), I (+0.0037 LoRA-A, +0.0081 LoRA-H)

---

## Step 4c/d — Before/after visualisations (DONE ✓)

```bash
python3 step4c_visualise_finetuned.py   # fig8, fig9, fig10
python3 step4d_clip_visualise.py        # fig11
```

Copy all figures:
```powershell
scp -r yvs23@ogg.cs.bath.ac.uk:/mnt/vurm/homes/homes/yvs23/Benchmark_dataset/prompt_curation/attention_maps/figures/ "C:\Users\Yuvan Velkumar\Downloads\figures_final"
```

---

## Step 5 — Other models (PENDING)

```bash
# Download LLaMA-3-8B to fast1
export HF_HOME=/mnt/fast1/yvs23/hf_cache
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')"

# Run full pipeline on LLaMA-3-8B
python3 step1_generate_and_attend.py \
    --json ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output ../attention_maps/baseline_llama3.json \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --max_regions 0
```

---

## Troubleshooting

**Disk quota exceeded on model download:**
```bash
export HF_HOME=/mnt/fast1/yvs23/hf_cache
export HF_HUB_DISABLE_XET=1
export TMPDIR=/mnt/fast1/yvs23/tmp
```

**Label attention mass showing 0.000:**
```python
# Correct extraction — full causal attention matrix
step0   = output.attentions[0]
att     = torch.stack([l[0] for l in step0]).mean(dim=(0,1))
weights = att[-1, :input_len]
```

**LoRA load_best_model_at_end error:**
Set `load_best_model_at_end=False` in TrainingArguments — PEFT compatibility issue.

**Git push rejected:**
```bash
git pull origin main --no-rebase && git push
```

**tmux for long runs:**
```bash
tmux new -s jobname   # start
# Ctrl+B then D       # detach
tmux attach -t jobname  # reattach
tmux capture-pane -t jobname -p | tail -10  # quick check
```

**Panoptic PNGs not found:**
```bash
cd /mnt/fast1/yvs23/annotations && unzip -q panoptic_train2017.zip
```
