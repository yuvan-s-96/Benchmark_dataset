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

Add to bashrc permanently:
```bash
echo 'export HF_HOME=/mnt/fast1/yvs23/hf_cache' >> ~/.bashrc
echo 'export HF_HUB_DISABLE_XET=1' >> ~/.bashrc
echo 'export TMPDIR=/mnt/fast1/yvs23/tmp' >> ~/.bashrc
```

---

## Storage layout

```
~/Benchmark_dataset/
├── prompt_curation/
│   ├── scripts/         ← all pipeline scripts
│   ├── attention_maps/  ← baseline_mistral.json + visualisations + figures
│   ├── results/         ← template_comparison_mistral.json
│   ├── models/          ← LoRA adapters (not in git, large)
│   └── prompts/
├── data/
│   ├── coconut_subset/  ← 50 COCO images + annotations
│   ├── style_references/ ← WikiArt style images (458 MB)
│   └── annotations/
└── coconut_pipeline/    ← benchmark pipeline scripts (previous direction)

/mnt/fast1/yvs23/
├── hf_cache/            ← HuggingFace models (~14 GB Mistral-7B)
├── annotations/
│   ├── panoptic_train2017/  ← COCONut panoptic PNGs (1.1 GB)
│   └── panoptic_train2017.json
├── sam2.1_hiera_large.pt    ← SAM2 checkpoint (moved from home, 857 MB)
└── tmp/
```

---

## Step 1 — Generate instructions + extract attention (DONE ✓)

Single forward pass: generates instruction text AND extracts attention maps
from Mistral-7B-Instruct-v0.2 in 4-bit quantisation.

```bash
cd ~/Benchmark_dataset/prompt_curation/scripts

python3 step1_generate_and_attend.py \
    --json ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output ../attention_maps/baseline_mistral.json \
    --max_regions 0
```

Runtime: ~33 min for 229 regions.

**Results (template A baseline, 229 regions):**
- Label attention mass mean: 1.14% (COCO grounding)
- Style attention mass mean: 0.89% (WikiArt grounding)
- BOS token: ~55%, closing [/INST]: ~15%, task text: ~21%
- Region label: 0.21–3.60% (length artifact, not grounding quality)

---

## Step 1b — Visualise with COCONut masks (DONE ✓)

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

---

## Step 1c — Heatmap figures (DONE ✓)

```bash
python3 step1c_heatmap_figure.py \
    --attention_json ../attention_maps/baseline_mistral.json \
    --output ../attention_maps/figures/
```

Produces: fig1_attention_group_grid.png, fig2_label_mass_bars.png, fig3_*.png

---

## Step 2 — Prompt curation (DONE ✓)

Compare 8 templates (A–H) across all 229 regions.

```bash
python3 step2_curate_prompts.py \
    --json ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output ../results/template_comparison_mistral.json \
    --max_regions 0
```

Runtime: ~5.5 hours for 229 regions × 8 templates.

**Results — ranked by label attention mass:**

| Template | Label mass | Style mass | vs Baseline |
|---|---|---|---|
| E — question-style | 1.58% | 1.37% | +39% |
| A — baseline | 1.14% | 0.89% | — |
| C — caption-grounded | 0.90% | 1.17% | -21% |
| B — region first | 0.81% | 1.11% | -29% |
| D — contrastive | 0.76% | 1.10% | -33% |
| H — hybrid (E+G) | 0.74% | 1.56% | -35% |
| G — label repetition | 0.73% | 1.60% | -36% |
| F — chain-of-thought | 0.63% | 0.49% | -45% |

**Key findings:**
- Template E wins label grounding (+39%) and is the only template that beats
  baseline A on both label AND style simultaneously
- 217/229 regions improved with template E over baseline A
- Position alone does not improve grounding — grammatical framing matters more
- G/H win style mass but at the cost of label mass (trade-off)
- Chain-of-thought F is worst on both metrics

---

## Step 2b — Visualise template comparison (DONE ✓)

```bash
python3 step2b_visualise_comparison.py \
    --results ../results/template_comparison_mistral.json \
    --output  ../attention_maps/figures/
```

Produces: fig4_template_heatmap.png, fig5_label_style_scatter.png,
fig6_per_region_improvement.png

Copy all figures to laptop:
```powershell
scp -r yvs23@ogg.cs.bath.ac.uk:/mnt/vurm/homes/homes/yvs23/Benchmark_dataset/prompt_curation/attention_maps/ "C:\Users\Yuvan Velkumar\Downloads\attention_maps"
```

---

## Step 3 — Run other models (NEXT)

Run step1 + step2 (templates A and E only) on LLaMA-3-8B and Phi-3-mini
to verify Template E generalises across model architectures.

```bash
# Download models to fast1
export HF_HOME=/mnt/fast1/yvs23/hf_cache

# LLaMA-3-8B (~6 GB)
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
print('LLaMA tokenizer downloaded')
"

# Phi-3-mini (~2.3 GB)
python3 -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')
print('Phi-3 tokenizer downloaded')
"

# Run step1 for each model
python3 step1_generate_and_attend.py \
    --json ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output ../attention_maps/baseline_llama3.json \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --max_regions 0

python3 step1_generate_and_attend.py \
    --json ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output ../attention_maps/baseline_phi3.json \
    --model microsoft/Phi-3-mini-4k-instruct \
    --max_regions 0
```

Note: step1_generate_and_attend.py needs a --model argument added for this.

---

## Step 4 — LoRA fine-tuning (PENDING)

Fine-tune Mistral-7B (and other models) on Template E instructions.

```bash
pip install peft accelerate

python3 step3_finetune_lora.py \
    --base_model mistralai/Mistral-7B-Instruct-v0.2 \
    --template_results ../results/template_comparison_mistral.json \
    --best_template E \
    --output ../models/mistral_lora_E/ \
    --epochs 3
```

Then re-run step1 on fine-tuned model and compare attention mass before/after.

---

## Step 5 — Final evaluation (PENDING)

Score all model/template combinations:
- Attention mass (primary grounding metric)
- CLIP alignment (instruction vs style reference image)
- Label coverage (does instruction mention region label?)
- Visual specificity (visual descriptor word count)

```bash
python3 step5_evaluate.py \
    --results_dir ../results/ \
    --output ../results/final_evaluation.json
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
Attention shape during generation is `(layers, heads, seq, seq)` — full causal.
Correct extraction:
```python
step0   = output.attentions[0]
att     = torch.stack([l[0] for l in step0]).mean(dim=(0,1))
weights = att[-1, :input_len]
```

**Git push rejected:**
```bash
git pull origin main --no-rebase && git push
```

**CUDA out of memory:**
Check no other process is using GPU:
```bash
nvidia-smi
fuser /dev/nvidia1
```

**Panoptic PNGs not found:**
```bash
cd /mnt/fast1/yvs23/annotations
unzip -q panoptic_train2017.zip
ls panoptic_train2017/ | head -5
```

**tmux for long runs:**
```bash
tmux new -s step2
# run script
# detach: Ctrl+B then D
# reattach: tmux attach -t step2
```
