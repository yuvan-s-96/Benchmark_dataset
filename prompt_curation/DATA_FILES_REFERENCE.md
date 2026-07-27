# Data Files Reference — prompt_curation/results/ and attention_maps/

This document labels every JSON file in the project as **CURRENT** (authoritative,
use this one), **SUPERSEDED** (kept for provenance/audit trail only, do not use),
or **TEST/DEBUG** (small artifact from a diagnostic run, safe to ignore or delete).

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

| File | Status | Notes |
|---|---|---|
| `sink_corrected_ALL9_TRUE.json` | **CURRENT** | Base model, all 9 templates, true consistent weight-renormalisation method (Section 14.2). |
| `sink_corrected_lora_A_v3_TRUE.json` | **CURRENT** | LoRA-A, all 9 templates, true method (Section 14.3). |
| `sink_corrected_lora_C_v3_TRUE.json` | **CURRENT** | LoRA-C, all 9 templates, true method. |
| `sink_corrected_lora_H_v3_TRUE.json` | **CURRENT** | LoRA-H, all 9 templates, true method. |
| `sink_corrected_metrics.json`, `sink_corrected_metrics_v2.json` | SUPERSEDED | Original mixed-method file (Template A true, B–I approximate) — the inconsistency this whole correction fixed. Kept because it's referenced in the document as the "old" comparison point. |
| `sink_corrected_proper.json` | SUPERSEDED | Intermediate naming, predates the ALL9_TRUE fix. |
| `sink_corrected_all_models.json` | SUPERSEDED | Predates true LoRA correction. |
| `sink_corrected_lora_A_v3.json`, `sink_corrected_lora_C_v3.json`, `sink_corrected_lora_H_v3.json` | SUPERSEDED | Approximate-method versions, superseded by the `_TRUE` files above. |

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
| `causal_wilcoxon_A_vs_all.json` | **CURRENT** | Wilcoxon signed-rank results, base model, Template A vs each other (Section 14.4.3). |
| `causal_wilcoxon_all_models.json` | **CURRENT** | Wilcoxon results, all 3 LoRA variants. |
| `causal_drop_by_source.json` | **CURRENT** | Per-source causal drop lookup, used for judge-grounding correlation. |

### Raw Attention Weights (attention_maps/)

| File | Status | Notes |
|---|---|---|
| `baseline_mistral.json` | **CURRENT** | Template A, base model, original validated raw weights (per_region format). |
| `baseline_mistral_{B,C,D,E}.json`, `baseline_mistral_FG.json`, `baseline_mistral_HI.json` | **CURRENT** (components) | Templates B–I, base model, raw weights. Merged into `baseline_mistral_ALL9_full.json`. |
| `baseline_mistral_ALL9_full.json` | **CURRENT** | All 9 templates merged, base model — feeds `sink_corrected_ALL9_TRUE.json`. |
| `lora_A_v3_A.json`, `lora_A_v3_BCD.json`, `lora_A_v3_EFGHI.json` | **CURRENT** (components) | LoRA-A raw weights per template group. Merged into `lora_A_v3_ALL9_full.json`. |
| `lora_A_v3_ALL9_full.json` | **CURRENT** | LoRA-A, all 9 templates merged — feeds `sink_corrected_lora_A_v3_TRUE.json`. |
| `lora_C_v3_ABCD.json`, `lora_C_v3_EFGHI.json` | **CURRENT** (components) | LoRA-C raw weights per template group. |
| `lora_C_v3_ALL9_full.json` | **CURRENT** | LoRA-C, all 9 templates merged. |
| `lora_H_v3_ABCD.json`, `lora_H_v3_EFGHI.json` | **CURRENT** (components) | LoRA-H raw weights per template group. |
| `lora_H_v3_ALL9_full.json` | **CURRENT** | LoRA-H, all 9 templates merged. |
| `baseline_lora_A.json`, `baseline_lora_H.json`, `baseline_phi4.json` | TEST/DEBUG | Placeholder/near-empty files, not used. |
| `smoke_test.json` | TEST/DEBUG | Pipeline smoke test, safe to delete. |

### LLM-as-Judge

| File | Status | Notes |
|---|---|---|
| `judge_input_v3_final.json` | **CURRENT** | 30 regions × 7 sources, corrected instructions, ground truth included. Submitted to Claude and Gemini. |
| `judge_input.json`, `judge_input_full.json`, `judge_input_v3.json` | SUPERSEDED | Earlier versions predating the final source selection / instruction fix. |
| `llm_judge_scores.json` | SUPERSEDED | Early single-judge run. |
| `judge_grounding_vs_causal_correlation.json` | **CURRENT** | Formal correlation, judge grounding vs causal drop (Section 14.5.6). |
| `full_metric_correlation_matrix.json` | **CURRENT** | Full 7×7 metric correlation matrix (Section 8.4). |

*Note: Claude and Gemini judge score responses (`judge_scores_claude_v3.json`, `judge_scores_gemini_v3.json`, `judge_comparison_v3.json`) were generated conversationally and stored separately — see project chat history / outputs, not present as ogg result files at time of writing.*

### InstructPix2Pix Downstream

| File | Status | Notes |
|---|---|---|
| `instruct_pix2pix_final.json` | **CURRENT** | Base model, 9 templates, corrected COCONut masks. |
| `pix2pix_lora_A_v3_all_templates.json` | **CURRENT** | LoRA-A, 9 templates, corrected masks. |
| `pix2pix_lora_C_v3_all_templates.json` | **CURRENT** | LoRA-C, 9 templates, corrected masks. |
| `pix2pix_lora_H_v3_all_templates.json` | **CURRENT** | LoRA-H, 9 templates, corrected masks. |
| `instruct_pix2pix_eval_full.json` | SUPERSEDED | Pre-mask-fix (`nonregion_clip`=1.0000 bug). |
| `instruct_pix2pix_eval_full_v2.json` | SUPERSEDED | Mask fix applied, but predates final instruction set. |
| `instruct_pix2pix_eval_masked_test.json`, `instruct_pix2pix_v2_complete.json` | TEST/DEBUG | Intermediate diagnostic runs. |
| `instruct_pix2pix_eval.json` | TEST/DEBUG | 3-region smoke test only — do not use. |
| `pix2pix_lora_A_all_templates.json`, `pix2pix_lora_C_all_templates.json`, `pix2pix_lora_H_all_templates.json` | SUPERSEDED | Pre-mask-fix versions (`nonregion_clip`=1.0000 bug). |
| `pix2pix_lora_A_all_templates_v2.json`, `pix2pix_lora_C_all_templates_v2.json`, `pix2pix_lora_H_all_templates_v2.json` | SUPERSEDED | Mask fix applied, predates final instruction set / v3 naming. |

### ArtFID / Gram / LPIPS

| File | Status | Notes |
|---|---|---|
| `artfid_gram_final.json` | **CURRENT** | Base + 3 LoRA, all 9 templates, 30 regions each — full per-region data (Section 8). |
| `artfid_region_clip_correlation.json` | **CURRENT** | Region-level LPIPS/Gram vs Region CLIP correlation (Section 8.4, corrected version). |
| `artfid_gram_base_A_H.json`, `artfid_gram_complete.json`, `artfid_gram_eval.json`, `artfid_gram_eval_v2.json`, `artfid_gram_loraC.json` | SUPERSEDED | Intermediate/partial runs predating `_final.json`. |

### Other

| File | Status | Notes |
|---|---|---|
| `human_rating_indices.json` | **CURRENT** | The fixed 30-region sample indices used throughout. |
| `statistical_tests.json`, `complete_results_summary.json` | **CURRENT** | Summary/aggregate statistics. |
| `clip_scores.json`, `clip_finetuned.json`, `clip_scores_phi4.json` | TEST/DEBUG | Early exploratory CLIP scoring, not part of final pipeline. |
| `lora_A_v3_tmplA_instructions.json`, `lora_C_v3_tmplC_instructions.json`, `lora_H_v3_tmplH_instructions.json` | **CURRENT** | Instructions generated by each LoRA model on its own training template, used to build the judge input. |
| `attention_grounding_correlation_v3.json` | SUPERSEDED | Predates the true sink-correction fix; use the tables in Section 14.2.2/14.3 instead. |

---

*Last updated: alongside master document v9.*

---

## data/ — Consolidated Results and Visualizations

| File | Status | Notes |
|---|---|---|
| `MASTER_RESULTS_CONSOLIDATED.json` | **CURRENT** | Every key result computed fresh from source files: attention mass, causal ablation (all 5 metrics), Wilcoxon tests, all correlations, judge scores, confound regression (all 9 features tested), downstream ArtFID/LPIPS/Gram/CLIP. Validated (16/16 keys, 0 errors, 0 warnings, cross-checked against known values). |
| `prompt_curation_inputs.json` | **CURRENT** | The 200-image / 979-region source annotation file used throughout the pipeline (region labels, styles, captions). |
| `visualizations/attention_mass_by_template.png` | **CURRENT** | Sink-corrected attention mass, grouped bar, all 9 templates x 4 models. |
| `visualizations/causal_drop_by_template.png` | **CURRENT** | Causal label drop, grouped bar, all 9 templates x 4 models. |
| `visualizations/attention_vs_causal_scatter.png` | **CURRENT** | Scatter of attention mass vs causal drop, shows Template A as the outlier driving the correlation. |
| `visualizations/sink_correction_old_vs_new.png` | **CURRENT** | Old (inconsistent method) vs new (true, consistent method) sink-corrected attention mass, base model. |
| `visualizations/judge_scores_by_source.png` | **CURRENT** | Claude vs Gemini overall scores, all 7 judge-evaluation sources. |
| `visualizations/artfid_lpips_by_model.png` | **CURRENT** | ArtFID and LPIPS means, all 4 models. |
| `visualizations/causal_full_metrics_panel.png` | **CURRENT** | BLEU-4, Jaccard, Levenshtein, BERTScore-F1 by template, base model, Template A highlighted. |
| `visualizations/confound_coefficients_claude.png` | **CURRENT** | Regression coefficients for all 9 tested confound features (Claude), significant features highlighted. |

## data/annotations/ — COCONut Panoptic Masks (Evaluation Subset)

| File | Status | Notes |
|---|---|---|
| `*.png` (25 files) | **CURRENT** | COCONut panoptic segmentation masks for the 25 source images used in the InstructPix2Pix/ArtFID evaluation and LLM-judge sample. Copied from `/mnt/fast1/yvs23/coconut_panoptic/` on ogg. |
| `segment_lookup.json` | **CURRENT** | Region-label to COCONut panoptic segment-ID mapping, built by keyword matching (see README dataset section). Used to derive Region CLIP / NonRegion CLIP masks. |

**Not included:** `coconut_b_panoptic.json` (611MB) — the full COCONut dataset's panoptic annotations. This is third-party source data, not generated by this project, and far too large for version control. It remains at `/mnt/fast1/yvs23/coconut_panoptic/coconut_b_panoptic.json` on ogg; only the 25-image subset actually used here is included above. If reproducing this pipeline elsewhere, download COCONut separately (see dataset citation in main README) and regenerate `segment_lookup.json` via the mask-building scripts if needed.
