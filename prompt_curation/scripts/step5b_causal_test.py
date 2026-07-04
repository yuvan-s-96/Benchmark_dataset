"""
Step 5b — Causal label ablation test
======================================
For each region, generate instruction with:
  1. Original label tokens
  2. Scrambled label tokens (random vocab tokens, same length)

Measure output divergence using:
  - BLEU-4 (n-gram overlap)
  - Token-level Jaccard similarity
  - Label word presence (does output mention the label?)

If output barely changes — model not causally grounding in label
If output changes substantially — model IS using the label

Templates tested: A (rank 1), E (rank 2), H (rank 3), F (rank 9)
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from Levenshtein import ratio as lev_ratio
from bert_score import score as bert_score_fn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm


def bleu4(ref_tokens, hyp_tokens):
    from collections import Counter
    if len(hyp_tokens) < 4 or len(ref_tokens) < 4:
        return 0.0
    score = 1.0
    for n in range(1, 5):
        ref_ngrams = Counter(tuple(ref_tokens[i:i+n])
                             for i in range(len(ref_tokens)-n+1))
        hyp_ngrams = Counter(tuple(hyp_tokens[i:i+n])
                             for i in range(len(hyp_tokens)-n+1))
        matches    = sum((hyp_ngrams & ref_ngrams).values())
        total      = sum(hyp_ngrams.values())
        if total == 0:
            return 0.0
        score *= matches / total
    bp = min(1.0, len(hyp_tokens) / max(len(ref_tokens), 1))
    return bp * (score ** 0.25)


def jaccard(set_a, set_b):
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


TEMPLATES = {
    "A": lambda l,s,c: (
        f"[INST] You are a style transfer assistant. "
        f"Region: {l}. Style: {s}. Scene caption: {c}. "
        f"Write one instruction to apply this style to this region only. [/INST]"),
    "B": lambda l,s,c: (
        f"[INST] TARGET REGION: {l}\nSTYLE TO APPLY: {s}\nSCENE: {c}\n"
        f"Write a single instruction that applies {s} ONLY to the {l}. [/INST]"),
    "C": lambda l,s,c: (
        f"[INST] Scene description: {c}\n"
        f"From this scene, focus exclusively on: {l}\n"
        f"Required artistic style: {s}\n"
        f"Write one instruction to stylise the {l} in {s} style. [/INST]"),
    "D": lambda l,s,c: (
        f"[INST] Apply {s} style to the {l} in this image.\n"
        f"Full scene: {c}\n"
        f"Important: apply the style to {l} only. "
        f"Preserve all other regions unchanged.\n"
        f"Write the style transfer instruction. [/INST]"),
    "E": lambda l,s,c: (
        f"[INST] Image scene: {c}\n"
        f"What single instruction would transfer {s} artistic style "
        f"specifically to the {l}, leaving everything else untouched? [/INST]"),
    "F": lambda l,s,c: (
        f"[INST] You are a style transfer assistant.\n"
        f"Scene: {c}\nTarget region: {l}\nTarget style: {s}\n"
        f"Think step by step: first identify the {l} in the scene, "
        f"then describe how to apply {s} specifically to it.\n"
        f"Write one instruction. [/INST]"),
    "G": lambda l,s,c: (
        f"[INST] You are stylising ONE specific region: {l}.\n"
        f"Apply {s} to the {l} only.\n"
        f"Scene context: {c}\n"
        f"Write a style transfer instruction for the {l}. [/INST]"),
    "H": lambda l,s,c: (
        f"[INST] {s} style transfer task.\n"
        f"What single instruction would apply {s} specifically to the {l}, "
        f"leaving everything else in the scene untouched?\n"
        f"The {l} is the only region to be stylised.\n"
        f"Scene context: {c} [/INST]"),
    "I": lambda l,s,c: (
        f"[INST] Image scene: {c}\n"
        f"Write a single image editing instruction that transfers {s} style "
        f"specifically to the {l} in the scene, "
        f"leaving all other regions completely unchanged.\n"
        f"The instruction must describe what to do to the {l} only, "
        f"using visual and stylistic language. [/INST]"),
}


def scramble_label(tokenizer, prompt, label, vocab_size=32000, seed=42):
    rng     = random.Random(seed)
    all_ids = tokenizer.encode(prompt, add_special_tokens=True)

    candidates = [
        tokenizer.encode(label, add_special_tokens=False),
        tokenizer.encode(" " + label, add_special_tokens=False),
    ]
    label_ids   = []
    label_start = -1
    for cand in candidates:
        if not cand:
            continue
        for i in range(len(all_ids) - len(cand) + 1):
            if all_ids[i:i+len(cand)] == cand:
                label_ids   = cand
                label_start = i
                break
        if label_start >= 0:
            break

    if label_start < 0:
        return None, []

    scrambled_ids = list(all_ids)
    replacement   = [rng.randint(4, vocab_size-1) for _ in range(len(label_ids))]
    for i, idx in enumerate(range(label_start, label_start+len(label_ids))):
        scrambled_ids[idx] = replacement[i]

    scrambled_prompt = tokenizer.decode(scrambled_ids, skip_special_tokens=False)
    return scrambled_prompt, replacement


@torch.no_grad()
def generate(model, tokenizer, prompt, device, max_new_tokens=80):
    inputs    = tokenizer(prompt, return_tensors="pt",
                          truncation=True, max_length=512).to(device)
    input_len = inputs["input_ids"].shape[1]
    output    = model.generate(**inputs, max_new_tokens=max_new_tokens,
                               do_sample=False,
                               pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(
        output[0][input_len:], skip_special_tokens=True).strip()


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb, device_map={"": device},
        attn_implementation="eager")

    if args.adapter:
        print(f"Loading LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(base, args.adapter)
        model.eval()
        print("Fine-tuned model loaded\n")
    else:
        model = base
        model.eval()
        print("Base model loaded\n")

    with open(args.inputs_json) as f:
        records = json.load(f)

    all_regions = []
    for rec in records:
        caption = rec.get("coconut_caption", "")
        for reg in rec["regions"]:
            all_regions.append({
                "image_id":   rec["image_id"],
                "mask_index": reg["mask_index"],
                "label":      reg["region_label"],
                "style":      reg["style_name"],
                "caption":    caption,
            })

    random.seed(42)
    sample = random.sample(all_regions, min(args.n_sample, len(all_regions)))
    print(f"Sample size: {len(sample)} regions × {len(TEMPLATES)} templates")
    print(f"Estimated runtime: ~{len(sample)*len(TEMPLATES)*2*40/3600:.1f} hours\n")

    results = {}
    summary = {}

    for tmpl_name, tmpl_fn in TEMPLATES.items():
        print(f"\n{'='*55}")
        print(f"Template {tmpl_name} — causal ablation")
        print(f"{'='*55}")

        tmpl_results = []
        bleu_scores, jacc_scores, lev_scores = [], [], []
        orig_instrs, scram_instrs = [], []
        label_in_orig, label_in_scram = [], []
        n_skipped = 0

        for reg in tqdm(sample, desc=f"T{tmpl_name}"):
            label   = reg["label"]
            style   = reg["style"]
            caption = reg["caption"]

            prompt_orig  = tmpl_fn(label, style, caption)
            prompt_scram, replacement = scramble_label(
                tokenizer, prompt_orig, label)

            if prompt_scram is None:
                n_skipped += 1
                continue

            instr_orig  = generate(model, tokenizer, prompt_orig,  device)
            instr_scram = generate(model, tokenizer, prompt_scram, device)

            orig_toks  = instr_orig.lower().split()
            scram_toks = instr_scram.lower().split()

            b4  = bleu4(orig_toks, scram_toks)
            jc  = jaccard(orig_toks, scram_toks)
            lev = lev_ratio(instr_orig.lower(), instr_scram.lower())

            label_words = [w for w in label.lower().split() if len(w) > 3]
            lo = any(w in instr_orig.lower()  for w in label_words)
            ls = any(w in instr_scram.lower() for w in label_words)

            bleu_scores.append(b4)
            jacc_scores.append(jc)
            lev_scores.append(lev)
            orig_instrs.append(instr_orig)
            scram_instrs.append(instr_scram)
            label_in_orig.append(lo)
            label_in_scram.append(ls)

            tmpl_results.append({
                "image_id":              reg["image_id"],
                "region_label":          label,
                "style_name":            style,
                "instruction_original":  instr_orig,
                "instruction_scrambled": instr_scram,
                "bleu4":                 round(b4, 4),
                "jaccard":               round(jc, 4),
                "levenshtein":           round(lev, 4),
                "label_in_orig":         lo,
                "label_in_scram":        ls,
            })

        mean_b   = np.mean(bleu_scores)
        mean_j   = np.mean(jacc_scores)
        mean_lev = np.mean(lev_scores)

        print(f"  Computing BERTScore for {len(orig_instrs)} pairs...")
        P, R, F1 = bert_score_fn(
            orig_instrs, scram_instrs,
            lang="en", model_type="distilbert-base-uncased",
            verbose=False, device=device,
        )
        mean_bert = float(F1.mean())
        print(f"  BERTScore F1 mean: {mean_bert:.3f}")
        lo_rate  = np.mean(label_in_orig)
        ls_rate  = np.mean(label_in_scram)

        print(f"  BLEU-4 similarity (orig vs scrambled): {mean_b:.3f}")
        print(f"  Jaccard similarity:                    {mean_j:.3f}")
        print(f"  Levenshtein similarity:                {mean_lev:.3f}")
        print(f"  BERTScore F1 (semantic similarity):    {mean_bert:.3f}")
        print(f"  Label in original instruction:         {lo_rate*100:.1f}%")
        print(f"  Label in scrambled instruction:        {ls_rate*100:.1f}%")
        print(f"  Label drop after scramble:             {(lo_rate-ls_rate)*100:.1f}pp")
        print(f"  Skipped (label not found):             {n_skipped}")

        results[tmpl_name] = tmpl_results
        summary[tmpl_name] = {
            "bleu4_mean":         round(float(mean_b), 4),
            "jaccard_mean":       round(float(mean_j), 4),
            "levenshtein_mean":   round(float(mean_lev), 4),
            "bertscore_f1_mean":  round(float(mean_bert), 4),
            "label_in_orig_pct":  round(float(lo_rate*100), 1),
            "label_in_scram_pct": round(float(ls_rate*100), 1),
            "label_drop_pp":      round(float((lo_rate-ls_rate)*100), 1),
            "n_scored":           len(tmpl_results),
            "n_skipped":          n_skipped,
        }

    print(f"\n{'='*70}")
    print("CAUSAL TEST SUMMARY")
    print("Lower BLEU-4/Jaccard = output changes more = stronger causal grounding")
    print(f"{'='*70}")
    print(f"\n  {'Tmpl':<6} {'BLEU-4':>8} {'Jaccard':>9} {'Lev':>7} {'BERT-F1':>9} {'LblDrop':>9}")
    print(f"  {'-'*60}")
    for t, s in sorted(summary.items(), key=lambda x: x[1]["bleu4_mean"]):
        print(f"  {t:<6} {s['bleu4_mean']:>8.3f} {s['jaccard_mean']:>9.3f} "
              f"{s['levenshtein_mean']:>7.3f} {s['bertscore_f1_mean']:>9.3f} "
              f"{s['label_drop_pp']:>+8.1f}pp")

    out = {
        "model":       "mistral-7b-instruct-v0.2",
        "experiment":  "causal label ablation",
        "n_sample":    len(sample),
        "description": (
            "Label tokens replaced with random vocabulary tokens of equal length "
            "(seed=42). BLEU-4 and Jaccard measure output similarity before/after "
            "scrambling. Low similarity = model causally grounded in label token."
        ),
        "summary":      summary,
        "per_template": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs_json",
        default="../../data/coconut_subset/annotations/prompt_curation_inputs.json")
    p.add_argument("--output",
        default="../results/causal_test.json")
    p.add_argument("--n_sample", type=int, default=200)
    p.add_argument("--adapter", default=None,
        help="Path to LoRA adapter directory (optional)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
