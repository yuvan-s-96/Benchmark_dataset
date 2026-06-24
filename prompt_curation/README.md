# Prompt Curation for Regional Style Transfer

## Dissertation contribution
Attention-guided prompt curation and fine-tuning for instruction-following LLMs
in regional style transfer. Comparative study across Mistral-7B, Phi-3-mini, Gemma-2B.

## Structure
scripts/          — all Python scripts (attention extraction, curation, fine-tuning)
attention_maps/   — extracted attention scores per region per template per model
prompts/          — 5 prompt templates (A-E) and generated instructions
results/          — evaluation scores, comparison tables
models/           — fine-tuned LoRA adapters (not in git — too large)

## Pipeline
Step 1  extract_attention.py     — baseline attention extraction on Mistral-7B
Step 2  visualise_attention.py   — heatmap visualisation per token
Step 3  curate_prompts.py        — compare 5 templates, score attention alignment
Step 4  finetune_lora.py         — LoRA fine-tune on best template
Step 5  compare_models.py        — cross-model comparison (Mistral, Phi-3, Gemma)

## Models
Mistral-7B-Instruct-v0.2    — 4-bit quantisation via bitsandbytes
Phi-3-mini-4k-instruct      — float16, fits in 8 GB comfortably
Gemma-2B-it                 — float16, smallest baseline

## Data inputs (from existing benchmark pipeline)
../data/coconut_subset/annotations/subset_auto_final_gguf.json
../data/style_references/
