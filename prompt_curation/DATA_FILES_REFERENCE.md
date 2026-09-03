# Data Files Reference — prompt_curation/results/ and attention_maps/

This document labels every JSON file in the project as **CURRENT** (authoritative,
use this one), **SUPERSEDED** (kept for provenance/audit trail only, do not use),
or **TEST/DEBUG** (small artefact from a diagnostic run, safe to ignore or delete).

Note: several large files are tracked via Git LFS. Run `git lfs pull` after
cloning/pulling if any file appears as a small text pointer (~130 bytes)
instead of its real size.

---

## results/ — Core Pipeline Files

### Instructions (source of truth)

| File | Status | Notes |
|---|---|---|
| `template_comparison_979_final.json` | **CURRENT** | 979 regions × 9 templates, corrected per-template token limits (256/512/768). This is THE source of truth for all instruction text used downstream. |
| `template_comparison_979.json` | SUPERSEDED | Original 80-token version — 82% truncation. Kept for audit trail only. |
| `template_comparison_979_v2.json` | SUPERSEDED | 256-token intermediate version. |
| `template_comparison_979_v3.json` | SUPERSEDED | Further intermediate merge. |
| `template_comparison_A_v4.json`, `template_comparison_{D,E,F,G,H,I}_v3.json` | SUPERSEDED | Per-template intermediate outputs, later merged into `_final.json`. |
| `template_comparison_mistral.json`, `template_comparison_phi4.json`, `template_EI_comparison.json` | TEST/DEBUG | Early exploratory runs, not part of final pipeline. |
| `test_512_all_templates.json` | TEST/DEBUG | Sanity check during token-limit tuning. |

### Sink-Corrected Attention Mass

**Superseded twice over** — see the "Update: Standalone `/`-Token Bug Fix" section near the
end of this document for the final, authoritative files. The entries below are kept for
audit trail only; none of the files in this subsection should be used directly.

| File | Status | Notes |
|---|---|---|
| `sink_corrected_ALL9_TRUE.json` | SUPERSEDED | Base model, all 9 templates, true consistent weight-renormalisation method — but predates both the label-index fix and the standalone-`/`-token fix. |
| `sink_corrected_lora_A_v3_TRUE.json`, `sink_corrected_lora_C_v3_TRUE.json`, `sink_corrected_lora_H_v3_TRUE.json` | SUPERSEDED | Same as above, LoRA variants. |
| `sink_corrected_metrics.json`, `sink_corrected_metrics_v2.json` | SUPERSEDED | Original mixed-method file (Template A true, B–I approximate) — the inconsistency the first correction fixed. Kept because it's referenced in the dissertation as the "old" comparison point. |
| `sink_corrected_proper.json` | SUPERSEDED | Intermediate naming, predates the ALL9_TRUE fix. |
| `sink_corrected_all_models.json` | SUPERSEDED | Predates true LoRA correction. |
| `sink_corrected_lora_A_v3.json`, `sink_corrected_lora_C_v3.json`, `sink_corrected_lora_H_v3.json` | SUPERSEDED | Approximate-method versions. |
| `sink_corrected_base_v2_FINAL.json`, `sink_corrected_lora_A_v2_FINAL.json`, `sink_corrected_lora_C_v2_FINAL.json`, `sink_corrected_lora_H_v2_FINAL.json` | SUPERSEDED | Label-index bug fixed, but predates the standalone-`/`-token fix found during WACV paper figure verification. Use the `_v3_FIXED.json` files instead (see final update section). |

### Causal Ablation

| File | Status | Notes |
|---|---|---|
| `causal_test.json` | **CURRENT** | Base model, all 9 templates, n=200. Per Deblina's guidance, valid despite pre-dating the instruction-truncation fix, since this measures divergence between generations, not generation completeness. |
| `causal_test_lora_A_v3.json` | **CURRENT** | LoRA-A, all 9 templates, n=200. |
| `causal_test_lora_C_v3.json` | **CURRENT** | LoRA-C, all 9 templates, n=200. |
| `causal_test_lora_H_v3.json` | **CURRENT** | LoRA-H, all 9 templates, n=200. |
| `causal_test_lora_A_sanity512.json` | **CURRENT** | Confirmatory rerun at 512 tokens (n=30, Template A) — validates truncation-robustness of the causal test methodology. |
| `causal_test_caption_masked.json` | **CURRENT** | Caption-masking ablation variant. |
| `causal_test_lora_A.json`, `causal_test_lora_C.json`, `causal_test_lora_H.json` | SUPERSEDED | Pre-v3 LoRA model versions. |
| `causal_wilcoxon_A_vs_all.json` | **CURRENT** | Wilcoxon signed-rank results, base model, Template A vs each other. |
| `causal_wilcoxon_all_models.json` | **CURRENT** | Wilcoxon results, all 3 LoRA variants (Mistral). |
| `causal_wilcoxon_llama.json` | **CURRENT** | Wilcoxon results, Llama base model — completed alongside the cross-model check (see final update section). |
| `causal_wilcoxon_llama_lora.json` | **CURRENT** | Wilcoxon results, all 3 LoRA variants (Llama) — completed alongside the cross-model check. |
| `causal_drop_by_source.json` | **CURRENT** | Per-source causal drop lookup, used for judge-grounding correlation. |

Combined across both model families: 256 predefined paired Wilcoxon tests; 253/256
significant (98.8%). All three exceptions involve the Levenshtein metric on Llama (one on
base, two on LoRA-H); none involve Template E.

### Intervention Specificity Controls (New)

| File | Status | Notes |
|---|---|---|
| `intervention_controls.json` | **CURRENT** | Style-span control, caption-span control, and counterfactual-substitution results for Template A, both model families. Style/caption corruption produced label drops of at most 3.5pp (15 of 16 non-significant); counterfactual substitution reproduced the region-scrambling effect closely (61.5pp Mistral, 71.0pp Llama, 87.5%/90.0% adoption). Resolves the intervention-specificity question raised during dissertation review. |

### Raw Attention Weights (attention_maps/)

**Superseded twice over** — see the final update section for the authoritative files. Kept
here for audit trail only.

| File | Status | Notes |
|---|---|---|
| `baseline_mistral.json` | **CURRENT** | Template A, base model, original validated raw weights (per_region format) — this specific file was never affected by either attention bug. |
| `baseline_mistral_{B,C,D,E}.json`, `baseline_mistral_FG.json`, `baseline_mistral_HI.json` | SUPERSEDED (components) | Templates B–I, base model, raw weights, predating the label-index fix. |
| `baseline_mistral_ALL9_full.json` | SUPERSEDED | Predates the label-index fix. |
| `lora_A_v3_A.json`, `lora_A_v3_BCD.json`, `lora_A_v3_EFGHI.json` | SUPERSEDED (components) | Predates the label-index fix. |
| `lora_A_v3_ALL9_full.json` | SUPERSEDED | Predates the label-index fix. |
| `lora_C_v3_ABCD.json`, `lora_C_v3_EFGHI.json` | SUPERSEDED (components) | Predates the label-index fix. |
| `lora_C_v3_ALL9_full.json` | SUPERSEDED | Predates the label-index fix. |
| `lora_H_v3_ABCD.json`, `lora_H_v3_EFGHI.json` | SUPERSEDED (components) | Predates the label-index fix. |
| `lora_H_v3_ALL9_full.json` | SUPERSEDED | Predates the label-index fix. |
| `baseline_lora_A.json`, `baseline_lora_H.json`, `baseline_phi4.json` | TEST/DEBUG | Placeholder/near-empty files, not used. |
| `smoke_test.json` | TEST/DEBUG | Pipeline smoke test, safe to delete. |

### LLM-as-Judge

| File | Status | Notes |
|---|---|---|
| `judge_input_v3_final.json` | **CURRENT** | 30 regions × 7 sources, corrected instructions, ground truth included. Submitted to Claude and Gemini manually via chat interface (no API script for this step). |
| `judge_input.json`, `judge_input_full.json`, `judge_input_v3.json` | SUPERSEDED | Earlier versions predating the final source selection / instruction fix. |
| `llm_judge_scores.json` | SUPERSEDED | Early single-judge run. |
| `judge_scores_claude_v3.json`, `judge_scores_gemini_v3.json` | **CURRENT** | Full per-region, per-criterion judge scores, both judges. Used for the instruction-level correlation (n=210) reported in the dissertation. |
| `judge_grounding_vs_causal_correlation.json` | **CURRENT** | Formal correlation, judge grounding vs causal drop. |
| `full_metric_correlation_matrix.json` | **CURRENT** | Full metric correlation matrix; also the source of the 1,043-observation region-level LPIPS/NonRegion CLIP correlation used in the downstream analysis. |

### InstructPix2Pix Downstream

| File | Status | Notes |
|---|---|---|
| `instruct_pix2pix_final.json` | **CURRENT** | Base model, 9 templates, corrected COCONut masks. |
| `pix2pix_lora_A_v3_all_templates.json` | **CURRENT** | LoRA-A, 9 templates, corrected masks. |
| `pix2pix_lora_C_v3_all_templates.json` | **CURRENT** | LoRA-C, 9 templates, corrected masks. |
| `pix2pix_lora_H_v3_all_templates.json` | **CURRENT** | LoRA-H, 9 templates, corrected masks. |
| `instruct_pix2pix_eval_full_v2.json` | **CURRENT** | Full baseline_A/baseline_C evaluation set — note the `instruction` field in this file is truncated to 100 characters for storage; regenerate from the model directly if full instruction text is needed for a specific region. |
| `instruct_pix2pix_eval_full.json` | SUPERSEDED | Pre-mask-fix (`nonregion_clip`=1.0000 bug). |
| `instruct_pix2pix_eval_masked_test.json`, `instruct_pix2pix_v2_complete.json` | TEST/DEBUG | Intermediate diagnostic runs. |
| `instruct_pix2pix_eval.json` | TEST/DEBUG | 3-region smoke test only — do not use. |
| `pix2pix_lora_A_all_templates.json`, `pix2pix_lora_C_all_templates.json`, `pix2pix_lora_H_all_templates.json` | SUPERSEDED | Pre-mask-fix versions (`nonregion_clip`=1.0000 bug). |
| `pix2pix_lora_A_all_templates_v2.json`, `pix2pix_lora_C_all_templates_v2.json`, `pix2pix_lora_H_all_templates_v2.json` | SUPERSEDED | Mask fix applied, predates final instruction set / v3 naming. |

### ArtFID / Gram / LPIPS

| File | Status | Notes |
|---|---|---|
| `artfid_gram_final.json` | **CURRENT** | Base + 3 LoRA, all 9 templates, 30 regions each — full per-region data. |
| `artfid_region_clip_correlation.json` | **CURRENT** | Region-level LPIPS/Gram vs Region CLIP correlation, corrected version — confirms the condition-level ArtFID hypothesis was an ecological fallacy; the real region-level finding is LPIPS vs NonRegion CLIP (r=-0.442, p<0.001, n=1,043). |
| `artfid_gram_base_A_H.json`, `artfid_gram_complete.json`, `artfid_gram_eval.json`, `artfid_gram_eval_v2.json`, `artfid_gram_loraC.json` | SUPERSEDED | Intermediate/partial runs predating `_final.json`. |

### Other

| File | Status | Notes |
|---|---|---|
| `human_rating_indices.json` | **CURRENT** | The fixed 15-region sample indices used for the blind human rating study (2 raters). |
| `statistical_tests.json`, `complete_results_summary.json` | **CURRENT** | Summary/aggregate statistics. |
| `clip_scores.json`, `clip_finetuned.json`, `clip_scores_phi4.json` | TEST/DEBUG | Early exploratory CLIP scoring, not part of final pipeline. |
| `lora_A_v3_tmplA_instructions.json`, `lora_C_v3_tmplC_instructions.json`, `lora_H_v3_tmplH_instructions.json` | **CURRENT** | Instructions generated by each LoRA model on its own training template, used to build the judge input. |
| `attention_grounding_correlation_v3.json` | SUPERSEDED | Predates the true sink-correction fix; use the final Spearman correlation table below instead (values unchanged by the later `/`-token fix, since Spearman correlation is rank-based). |

---

## data/ — Consolidated Results and Visualizations

| File | Status | Notes |
|---|---|---|
| `MASTER_RESULTS_CONSOLIDATED.json` | **CURRENT** | Every key result computed fresh from source files. Regenerate after the standalone-`/`-token fix if not already done — attention mass values will change (correlation values will not). |
| `prompt_curation_inputs.json` | **CURRENT** | The 200-image / 979-region source annotation file used throughout the pipeline. |
| `visualizations/attention_mass_by_template.png` | SUPERSEDED | Pre-`/`-token-fix attention values. Regenerate from the final sink-corrected files. |
| `visualizations/causal_drop_by_template.png` | **CURRENT** | Causal ablation was never affected by any of the three attention bugs. |
| `visualizations/attention_vs_causal_scatter.png` | SUPERSEDED | Pre-`/`-token-fix. Use `fig_attention_causal_scatter_CORRECTED.svg` (dissertation/paper figures folder) instead. |
| `visualizations/sink_correction_old_vs_new.png` | **CURRENT** (as historical comparison) | Illustrates the *first* correction (inconsistent method fix) specifically; does not reflect the later `/`-token fix. |
| `visualizations/judge_scores_by_source.png` | **CURRENT** | Unaffected by attention bugs. |
| `visualizations/artfid_lpips_by_model.png` | **CURRENT** | Unaffected by attention bugs. |
| `visualizations/causal_full_metrics_panel.png` | **CURRENT** | Unaffected by attention bugs. |
| `visualizations/confound_coefficients_claude.png` | **CURRENT** | Unaffected by attention bugs. |

## data/annotations/ — COCONut Panoptic Masks (Evaluation Subset)

| File | Status | Notes |
|---|---|---|
| `*.png` (25 files) | **CURRENT** | COCONut panoptic segmentation masks for the 25 source images used in the InstructPix2Pix/ArtFID evaluation and LLM-judge sample. |
| `segment_lookup.json` | **CURRENT** | Region-label to COCONut panoptic segment-ID mapping. |

**Not included:** `coconut_b_panoptic.json` (611MB) — third-party source data, remains at
`/mnt/fast1/yvs23/coconut_panoptic/coconut_b_panoptic.json` on ogg.

---

## Update: Label-Index Bug Fix (Second Correction Pass)

The following files superseded the original attention-map and sink-correction files after
a second bug (label-token index off-by-one) was found and fixed. **These files were
themselves later superseded a third time — see the final update section below.**

### attention_maps/ — Second-Pass Corrected Raw Weights (now superseded)

| File | Status | Notes |
|---|---|---|
| `baseline_mistral_ALL9_v2_FINAL.json` | SUPERSEDED | Label-index bug fixed, but predates the standalone-`/`-token fix. The `/` token itself sits within this file's own token stream (as part of `[/INST]`) and was mislabelled as semantic rather than structural — this file's raw weights are correct, only the *downstream sink-correction computation* from it needs redoing, which the final update section covers. |
| `lora_A_v3_ALL9_v2_FINAL.json`, `lora_C_v3_ALL9_v2_FINAL.json`, `lora_H_v3_ALL9_v2_FINAL.json` | SUPERSEDED (as above) | Same situation — raw weights correct, sink-correction computation from them needs redoing. |

*(Component files predating this merge — `baseline_mistral_BDFH_v2.json`,
`lora_A_v3_ABCDE_v2.json`, `lora_H_v3_AB_v2.json`, etc. — are omitted from this table for
brevity; all share the same status as their corresponding `_ALL9_v2_FINAL.json` merge.)*

---

## Update: Standalone `/`-Token Sink-Correction Bug Fix (Third and Final Correction Pass)

Found during WACV 2027 paper Figure 1 verification, after dissertation writing was already
underway. The structural-token exclusion set in `step5_sink_corrected_lora.py` (**not**
the earlier-patched `step5_sink_corrected_metric.py` — the two files had independently
duplicated copies of the same function, and only one was fixed at first) failed to exclude
the standalone `/` token from Mistral's closing `[/INST]` delimiter. Unlike `[`, `INST`,
and `]`, which are caught as complete structural tokens, the lone `/` was treated as
semantic content and was absorbing a large, template-varying share of "corrected"
attention mass — as much as 47.9% of the entire corrected mass in one individual-region
test case.

**Llama was checked directly and confirmed clean**: Llama's special tokens
(`<|eot_id|>`, `<|start_header_id|>`, etc.) are atomic in its tokenizer vocabulary and
never fragment into a stray punctuation token the way `[/INST]` does. No Llama files in
this reference needed regenerating for this fix.

The raw attention weight files (`*_ALL9_v2_FINAL.json` / `llama_ALL9_merged.json`) did
**not** need regenerating — the bug was in the sink-correction computation applied to
those weights, not in the weights themselves.

### results/ — Final, Authoritative Sink-Corrected Attention Mass

| File | Status | Notes |
|---|---|---|
| `sink_corrected_base_v3_FIXED.json` | **CURRENT — authoritative** | Base model, true method, both the label-index and `/`-token fixes applied. Template A: 9.86% (up from the pre-fix 6.17%). |
| `sink_corrected_lora_A_v3_FIXED.json` | **CURRENT — authoritative** | LoRA-A. Template A: 9.36%. |
| `sink_corrected_lora_C_v3_FIXED.json` | **CURRENT — authoritative** | LoRA-C. Template A: 11.07%. |
| `sink_corrected_lora_H_v3_FIXED.json` | **CURRENT — authoritative** | LoRA-H. Template A: 9.36%. |
| `sink_corrected_llama.json` | **CURRENT — authoritative** | Llama base model, confirmed unaffected by this bug; no regeneration needed. Template A: 9.83%. |

### Final, corrected Template A attention mass (all models)

| Model | Base | LoRA-A | LoRA-C | LoRA-H |
|---|---|---|---|---|
| Mistral | **9.86%** | **9.36%** | **11.07%** | **9.36%** |
| Llama | **9.83%** | 7.70% | 9.83% | 7.77% |

### Final attention-grounding Spearman correlation (all models)

Unchanged by this fix — Spearman correlation is rank-based, and the fix did not reorder
any template within any model.

| Model | Mistral r | Mistral sig. | Llama r | Llama sig. |
|---|---|---|---|---|
| Base | 0.804 | Yes (p<0.01) | 0.460 | No |
| LoRA-A | 0.621 | No | 0.749 | Yes (p<0.05) |
| LoRA-C | 0.842 | Yes (p<0.01) | 0.723 | Yes (p<0.05) |
| LoRA-H | 0.571 | No | 0.428 | No |

### scripts/ — Modified for this fix

| File | Status | Notes |
|---|---|---|
| `step5_sink_corrected_lora.py` | **CURRENT** | The actual file used in production; contains its own independent copy of `get_structural_indices`, now with `/` added to the excluded-punctuation set. |
| `step5_sink_corrected_metric.py` | Patched but not authoritative | An earlier, separate copy of the same function was patched first, before discovering it wasn't the file actually used by the LoRA pipeline. Kept in sync for consistency but `step5_sink_corrected_lora.py` is the file that matters. |

---

## Cross-Model Comparison Status — Final

| Measure | Mistral | Llama |
|---|---|---|
| Attention extraction (all 9 templates) | ✓ Complete | ✓ Complete |
| Sink correction (true method, both bugs fixed) | ✓ Complete | ✓ Complete (confirmed unaffected by the `/`-token bug) |
| Causal ablation | ✓ Complete | ✓ Complete |
| Attention-grounding correlation | ✓ Complete | ✓ Complete |
| LoRA fine-tuning equivalent | ✓ Complete (3 adapters) | ✓ Complete (3 adapters) |
| Intervention specificity controls | ✓ Complete | ✓ Complete |

**Qualitative cross-model findings** (documented in the dissertation's "Cross-Model
Behavioural Observations" section):
1. Llama frequently generates markdown-formatted elaboration or illustrative code instead
   of a single concise instruction. Because attention mass is extracted from the first
   decoding step only, this does not affect the attention analysis, but can influence
   surface-level divergence measures in the causal ablation relative to Mistral.
2. Template D produced explicit refusals in 5 of 30 sampled regions for the Llama base
   model during downstream evaluation; no refusals occurred for any other template.
   Consistent with Template D's more literal, command-oriented framing.

---
Llama cross-model check (attention, causal ablation, and LoRA fine-tuning all complete),
the intervention-specificity controls, and submission of the accompanying WACV 2027 paper
(#1260).*
