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

## Pipeline

### Step 1 — Generate instructions + extract attention
```bash
cd ~/Benchmark_dataset/prompt_curation/scripts

python3 step1_generate_and_attend.py \
    --json ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output ../attention_maps/baseline_mistral.json \
    --max_regions 0
```

### Step 1b — Visualise with COCONut masks
```bash
python3 step1b_visualise_attention.py \
    --attention_json ../attention_maps/baseline_mistral.json \
    --ann_json ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --pan_json /mnt/fast1/yvs23/annotations/panoptic_train2017.json \
    --pan_dir  /mnt/fast1/yvs23/annotations/panoptic_train2017 \
    --img_dir  ../../data/coconut_subset/images \
    --output   ../attention_maps/visualisations/ \
    --n 10
```

### Step 1c — Matplotlib heatmap figures
```bash
python3 step1c_heatmap_figure.py \
    --attention_json ../attention_maps/baseline_mistral.json \
    --output ../attention_maps/figures/
```

Copy figures to laptop:
```powershell
scp -r yvs23@ogg.cs.bath.ac.uk:/mnt/vurm/homes/homes/yvs23/Benchmark_dataset/prompt_curation/attention_maps/ "C:\Users\Yuvan Velkumar\Downloads\attention_maps"
```

### Step 2 — Prompt curation (NEXT)
```bash
python3 step2_curate_prompts.py \
    --json ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output ../results/template_comparison_mistral.json \
    --max_regions 0
```

### Step 3 — LoRA fine-tuning (PENDING)
```bash
python3 step3_finetune_lora.py \
    --base_model mistralai/Mistral-7B-Instruct-v0.2 \
    --template B \
    --output ../models/mistral_lora/
```

### Step 4 — Model comparison (PENDING)
```bash
python3 step4_compare_models.py \
    --models mistral phi3 gemma \
    --json ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output ../results/model_comparison.json
```

### Step 5 — Evaluation (PENDING)
```bash
python3 step5_evaluate.py \
    --results_dir ../results/ \
    --output ../results/final_evaluation.json
```

---

## Troubleshooting

**Disk quota exceeded on model download:**
HF_HOME must point to fast1, not home.
```bash
export HF_HOME=/mnt/fast1/yvs23/hf_cache
export HF_HUB_DISABLE_XET=1
export TMPDIR=/mnt/fast1/yvs23/tmp
```

**Label attention mass showing 0.000:**
Attention shape during generation is `(layers, heads, seq, seq)` — full causal matrix.
Correct extraction:
```python
att = torch.stack([l[0] for l in output.attentions[0]]).mean(dim=(0,1))
weights = att[-1, :input_len]
```

**Panoptic PNGs not found:**
Re-download to fast1:
```bash
cd /mnt/fast1/yvs23
curl -L -o pan.zip http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
unzip -q pan.zip && cd annotations && unzip -q panoptic_train2017.zip
```

**Git push rejected:**
```bash
git pull origin main --no-rebase && git push
```

**CUDA out of memory:**
Mistral-7B in 4-bit needs ~5 GB VRAM. Check no other process is using GPU:
```bash
nvidia-smi
```
