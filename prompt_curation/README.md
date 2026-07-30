# Prompt Curation for Regional Style Transfer
**MSc Data Science Dissertation — University of Bath 2026**

---

## Current Status — Evaluation Pipeline Complete, Two Bugs Found and Fixed, Human Rating Pending

All four evaluation methods (causal ablation, sink-corrected attention mass, LLM-as-judge,
downstream InstructPix2Pix/ArtFID) are **complete**, on corrected instructions, across the
base model and all three LoRA variants (A, C, H), all 9 templates.

Two significant methodological issues were found and fixed during this process, both
through deliberate cross-checking rather than trusting a single pipeline's output at face
value:

1. **Sink-correction method inconsistency** — Template A's attention mass had always used
   the accurate weight-renormalisation method, while Templates B–I had only ever used a
   cruder approximation (raw weights were never saved for them). Fixed by extracting raw
   weights for all 9 templates, all 4 models, and recomputing consistently.
2. **Label-token index off-by-one bug** — found while building a token-level attention
   heatmap. The function locating the region label within the prompt tokenised without a
   BOS token, while the actual generation step tokenised with one, shifting every label
   index one position early. Affected Templates B–I (base model) and all 9 templates on
   all 3 LoRA models (35 template-runs); only base-model Template A was unaffected (it
   came from a separate, never-buggy script). Fixed and fully rerun; all 35 outputs
   verified 100% correct (0 mismatches) before use.

**Final, fully-corrected attention-grounding correlation** (both fixes applied):

| Model | Spearman r | p-value | Significant? |
|---|---|---|---|
| Base | 0.804 | 0.0089 | Yes (p<0.01) |
| LoRA-A | 0.621 | 0.0740 | No |
| LoRA-C | 0.842 | 0.0044 | Yes (p<0.01) |
| LoRA-H | 0.571 | 0.1080 | No |

This supersedes all previously reported correlation values, including an intermediate
figure (r=0.533, not significant) that was itself already a correction of an earlier,
inflated figure (r=0.857) — see `prompt_curation_progress_v12.docx` Section 14.2–14.3 for
the full correction history. **Template A's dominance in absolute attention mass and its
causal grounding result are unaffected by either bug** and remain the strongest, most
consistent finding in the project.

**Also complete since the last major update:** a full confound analysis into why LLM-judge
grounding scores diverge from causal grounding (three rounds of hypothesis testing plus
direct elicitation from both judges), a region-level correction of the ArtFID/LPIPS
finding (the original condition-level hypothesis did not replicate — see below), and
token-level attention heatmaps visualising raw vs sink-corrected attention across all 4
models.

**Outstanding:** human rating session (blind, 2 raters, 15 regions × 7 sources,
source-stripped) scheduled for 3rd August with Yudi. Supervisor meeting Friday to discuss
framing before the dissertation writeup begins.

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

### Step 1 — Baseline Attention, Template A only (DONE ✓ STANDS — never affected by either bug)
```bash
python3 step1_generate_and_attend.py \
    --json    ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output  ../attention_maps/baseline_mistral.json
```

### Step 1 (unified) — Raw Attention Weights, Any Template Subset, Optional LoRA (DONE ✓ — fixed, fully verified)
```bash
python3 step1_full_weights.py \
    --output      ../attention_maps/OUTPUT_NAME.json \
    --templates   B,C,D            # omit for all 9
    --adapter     ../models/lora_A_v3/adapter   # omit for base model
```
`get_label_indices` was fixed to tokenise with `add_special_tokens=True`, matching how
`generate_and_attend` actually tokenises the prompt (previously it tokenised without
special tokens, causing every label-token index to be off by one relative to the real
attention-weight array). Verify any new run against this before trusting it:
```bash
python3 -c "
import json, sys
sys.path.insert(0, '.')
from step1_full_weights import get_label_indices
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
with open('../attention_maps/OUTPUT_NAME.json') as f:
    d = json.load(f)
for t, regions in d['per_template'].items():
    mismatches = sum(1 for r in regions if get_label_indices(tokenizer, r['prompt'], r['region_label']) != r['label_token_indices'])
    print(f'{t}: {len(regions)} regions, {mismatches} mismatches')
"
```

### Step 2 — Prompt Curation, All 9 Templates (DONE ✓ — corrected token limits)
```bash
python3 step2_curate_prompts.py \
    --json        ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output      ../results/template_comparison_979_final.json \
    --max_regions 979
```
Token limits set per-template: 256 tokens for most templates, 512 for D/E/F/H/I, 768 for
A and G.

### Step 3 — LoRA Fine-Tuning (DONE ✓, one adapter per template A/C/H)
```bash
python3 step3_finetune_lora.py --template A \
    --results_json ../results/template_comparison_979_final.json \
    --output ../models/lora_A_v3/ --epochs 3
```
Repeat for C and H with the corresponding GPU/output paths.

### Step 4 — Post Fine-Tuning Instructions (DONE ✓ — corrected instructions; use step1_full_weights.py for raw attention weights)
```bash
python3 step4_attention_finetuned.py \
    --adapter    ../models/lora_A_v3/adapter \
    --lora_name  lora_A_v3 \
    --base_model mistralai/Mistral-7B-Instruct-v0.2 \
    --output     ../results/attention_lora_A_v3_fixed.json
```

### Step 5a — Sink-Corrected Attention Mass (DONE ✓ — base + 3 LoRA, all 9 templates, true method, label-index bug fixed)
```bash
python3 step5_sink_corrected_lora.py \
    --attention_json ../attention_maps/baseline_mistral_ALL9_v2_FINAL.json \
    --output         ../results/sink_corrected_base_v2_FINAL.json
```
Repeat for `lora_A_v3_ALL9_v2_FINAL.json`, `lora_C_v3_ALL9_v2_FINAL.json`,
`lora_H_v3_ALL9_v2_FINAL.json` — these `_v2_FINAL` files are the authoritative,
fully-corrected versions; do not use the earlier `_ALL9_full.json` or `_v3_TRUE.json`
files (superseded — see `DATA_FILES_REFERENCE.md`).

### Step 5b — Causal Ablation Test (DONE ✓ STANDS — never affected by either bug, base + 3 LoRA, full Wilcoxon testing)
```bash
python3 step5b_causal_test.py \
    --inputs_json ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output      ../results/causal_test.json \
    --n_sample    200
```
64 Wilcoxon signed-rank tests (Template A vs every other template, per-region BLEU-4 and
Jaccard, base + 3 LoRA), 60/64 significant at p<0.001.

### Step 5c — Caption-Masked Causal Test (DONE ✓ STANDS)
```bash
python3 step5c_caption_masked_causal.py \
    --inputs_json ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output      ../results/causal_test_caption_masked.json \
    --templates   A,C,H,G,E \
    --n_sample    200
```
Uses the `coconut_caption` field, populated for 150/200 images (75%); the remaining 50
have no caption, for which masking is a no-op by construction.

### Step 6 — LLM-as-Judge (DONE ✓ — Claude + Gemini, 30 regions × 7 sources, plus full confound analysis)
Judge input built from `template_comparison_979_final.json` and
`attention_lora_*_v3_fixed.json`, submitted manually with the rubric and ground-truth
region/style labels. Confound analysis (three rounds: confident-language hypothesis,
label-position/step-count discovery, direct judge elicitation) documented in the master
progress document Section 14.5.7.

### Step 7 — InstructPix2Pix Downstream, 9×4 (DONE ✓ — base + 3 LoRA, all 9 templates)
```bash
python3 step7_instruct_pix2pix.py \
    --results_json  ../results/template_comparison_979_final.json \
    --output        ../results/instruct_pix2pix_final.json \
    --n_sample      30

python3 step7b_instruct_pix2pix_lora.py \
    --adapter      ../models/lora_A_v3/adapter \
    --inputs_json  ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
    --output       ../results/pix2pix_lora_A_v3_all_templates.json \
    --templates    A,B,C,D,E,F,G,H,I \
    --n_sample     30
```

### Step 8 — ArtFID / Gram / LPIPS (DONE ✓ — base + 3 LoRA, all 9 templates, region-level corrected)
```bash
python3 step8_artfid_gram.py \
    --results_json  ../results/template_comparison_979_final.json \
    --output        ../results/artfid_gram_final.json \
    --n_sample      30
```
An initial hypothesis (ArtFID penalises successful stylisation) appeared confirmed at the
condition level (36 averaged points) but did not replicate when tested properly at the
region level (1,043 actual data points) — a textbook ecological fallacy. The real,
region-level-verified finding: more dramatic edits worsen background preservation
(r=−0.442, p<0.001), not regional style match.

---

## Check tmux/nohup Sessions

```bash
tmux ls
tmux capture-pane -t <session_name> -p | tail -10
# or, for nohup-launched background jobs:
ps aux | grep step1_full_weights
tail -20 ../logs/<jobname>.log
```

---

## Key File Locations

```
~/Benchmark_dataset/
├── prompt_curation/
│   ├── scripts/                    All pipeline scripts
│   ├── results/                    All JSON results
│   ├── attention_maps/             Raw per-token attention weights, base + LoRA
│   ├── data/                       Consolidated results, annotations, visualizations
│   │   ├── MASTER_RESULTS_CONSOLIDATED.json   Every key result, computed fresh, validated
│   │   ├── visualizations/         PNG charts — attention, causal, judge, downstream, confound
│   │   ├── annotations/            25 COCONut panoptic masks + segment_lookup.json
│   │   └── prompt_curation_inputs.json         Source annotation file (979 regions)
│   ├── DATA_FILES_REFERENCE.md     Every result file labelled CURRENT/SUPERSEDED/TEST
│   └── models/
│       └── lora_A_v3/, lora_C_v3/, lora_H_v3/   fine-tuned on corrected instructions
├── data/
│   ├── coconut_subset/images/      COCO images
│   ├── coconut_subset/annotations/ COCONut annotations
│   └── style_references/           WikiArt reference images

/mnt/fast1/yvs23/
├── coconut_panoptic/               Full COCONut panoptic data (611MB, not in git — see
│                                    DATA_FILES_REFERENCE.md for the 25-image subset that is)
└── hf_cache/                       Mistral-7B, InstructPix2Pix
```

---

## Headline Results Summary

### Causal Grounding (base + 3 LoRA, never affected by either bug — fully stable throughout)

| Template | Base | LoRA-A | LoRA-C | LoRA-H |
|---|---|---|---|---|
| A | **66.0pp** | **65.0pp** | **56.5pp** | **65.0pp** |
| E | 24.0pp | 25.0pp | 24.0pp | 25.0pp |
| B, C, D, F, G, H, I | -0.5–1.0pp | -0.5–1.0pp | -0.5–0.0pp | -0.5–2.0pp |

64/64 Wilcoxon tests significant, 60 at p<0.001.

### Sink-Corrected Attention Mass (final, both bugs fixed)

| Template | Base | LoRA-A | LoRA-C | LoRA-H |
|---|---|---|---|---|
| A | **6.17%** | **5.39%** | **6.83%** | **5.60%** |
| E | 2.77% | 2.73% | 3.04% | 2.87% |
| H | 2.47% | 2.20% | 2.50% | 2.01% |
| D | 1.99% | 1.89% | 2.22% | 1.67% |
| B | 1.92% | 1.96% | 2.12% | 1.79% |
| G | 1.75% | 1.62% | 1.92% | 1.45% |
| F | 1.31% | 1.35% | 1.52% | 1.51% |
| C | 1.23% | 1.34% | 1.46% | 1.35% |
| I | 1.19% | 1.26% | 1.33% | 1.28% |

Attention-grounding correlation (final): base r=0.804**, LoRA-A r=0.621 ns, LoRA-C
r=0.842**, LoRA-H r=0.571 ns.

### LLM-as-Judge (Claude + Gemini, 30 regions, 7 sources)

Template C and its LoRA variant win decisively (4.46–4.93/5), well ahead of every other
source (3.63–3.95/5). Inter-rater reliability: Pearson r=0.741, 96.9% agreement within ±1
point.

**Judge-grounding inversion:** judge grounding score is significantly negatively
correlated with causal grounding (Claude r=−0.993, p<0.0001; Gemini r=−0.794, p=0.033,
n=7). Confound analysis found two real, partial predictors — position of first region
mention (later = lower score) and instruction step-count (more steps = lower score),
together explaining 9–16% of variance. The originally hypothesised "confident language"
explanation was tested and not confirmed.

---

## Dataset

- 200 COCO images from COCONut-PanCap (`xdeng77/coconut_pancap` on HuggingFace)
- 979 segmented regions across 20 WikiArt styles
- COCONut panoptic masks from Kaggle (`xueqingdeng/coconut`)
- `region_label` is a natural-language referring expression from COCONut-PanCap's
  region-level annotations (e.g. "the white clock has black hands and numbers"), not a
  simple category label — constructed to identify a region even when multiple uniquely
  instances of the same object class appear in one image
- `coconut_caption` (image-level, populated for 150/200 images) is the separate,
  potentially confounding caption source tested in the caption-masked causal ablation

---


---

## Known Limitations

- Judge-grounding-vs-causal-grounding correlation (n=7 sources) — striking effect size
  but small sample, suggestive not definitive
- Confound model explains only 9–16% of variance in judge grounding scores — real,
  validated effects, far from a complete account; LLM-as-classifier flagged as future work
- InstructPix2Pix downstream evaluation uses only the first ~60 words of each instruction
  (CLIP's 77-token text encoder limit) — inherent architectural constraint
- LoRA fine-tuning covers only 3 of 9 templates — no claim of generalisation across all
  templates
- Findings specific to Mistral-7B-Instruct — no evidence of generalisation to other LLMs

---

*Last updated: ahead of supervisor meeting, following the label-index bug fix and full
rerun. See `prompt_curation_progress_v12.docx` for the complete, detailed account.*
