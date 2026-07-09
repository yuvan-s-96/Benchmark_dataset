"""
Step 5c — Caption-masked causal test
=====================================
Addresses Deblina's confound concern:
- Template A: short/empty captions, label token is main region signal
- Template C/H/G: rich captions that mention the object

Test: scramble BOTH label tokens AND caption object mentions.
If C's label drop increases after caption masking, it was grounded
through the caption (not the label token) — genuine confound.

Method:
1. Generate instruction with original prompt (baseline)
2. Generate with label tokens scrambled (standard causal test)
3. Generate with label tokens + caption mentions scrambled (new)
Compare divergence across all three conditions.
"""

import argparse
import json
import re
import random
import numpy as np
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
import Levenshtein


STOPWORDS = {
    "the","a","an","this","that","these","those","with","from","into",
    "over","under","and","or","but","for","of","in","on","at","to",
    "is","are","was","were","be","been","has","have","had","it","its",
    "which","who","what","how","there","their","they","them","some",
    "all","each","both","other","such","than","then","when","where"
}

TEMPLATES = {
    "A": lambda l,s,c: (f"[INST] You are a style transfer assistant. "
                        f"Region: {l}. Style: {s}. Scene caption: {c}. "
                        f"Write one instruction to apply this style to this region only. [/INST]"),
    "C": lambda l,s,c: (f"[INST] Scene description: {c}\n"
                        f"From this scene, focus exclusively on: {l}\n"
                        f"Required artistic style: {s}\n"
                        f"Write one instruction to stylise the {l} in {s} style. [/INST]"),
    "H": lambda l,s,c: (f"[INST] {s} style transfer task.\n"
                        f"What single instruction would apply {s} specifically to the {l}, "
                        f"leaving everything else in the scene untouched?\n"
                        f"The {l} is the only region to be stylised.\n"
                        f"Scene context: {c} [/INST]"),
    "G": lambda l,s,c: (f"[INST] You are stylising ONE specific region: {l}.\n"
                        f"Apply {s} to the {l} only.\nScene context: {c}\n"
                        f"Write a style transfer instruction for the {l}. [/INST]"),
    "E": lambda l,s,c: (f"[INST] Image scene: {c}\n"
                        f"What single instruction would transfer {s} artistic style "
                        f"specifically to the {l}, leaving everything else untouched? [/INST]"),
}

def get_label_words(label):
    """Get meaningful words from label for caption masking."""
    words = label.lower().replace(",","").replace(".","").split()
    return [w for w in words if w not in STOPWORDS and len(w) > 2]

def mask_caption(caption, label):
    """Replace label word mentions in caption with [REGION].
    Uses word boundaries to avoid partial matches (e.g. white != off-white).
    """
    label_words = get_label_words(label)
    masked = caption
    for word in label_words:
        # Strict boundary: no letter or hyphen on either side
        # Handles: off-white (hyphen left), twilight (letter right embedded)
        pattern = re.compile(
            r"(?<![a-zA-Z\-])" + re.escape(word) + r"(?![a-zA-Z\-])",
            re.IGNORECASE)
        masked = pattern.sub("[REGION]", masked)
    return masked

def scramble_label_tokens(tokenizer, prompt, label, device):
    """Return prompt with label tokens replaced by random tokens."""
    vocab_size = tokenizer.vocab_size
    all_ids = tokenizer.encode(prompt, add_special_tokens=True)
    label_ids_a = tokenizer.encode(label, add_special_tokens=False)
    label_ids_b = tokenizer.encode(" " + label, add_special_tokens=False)

    # Find label token span
    label_span = None
    for cand in [label_ids_a, label_ids_b]:
        for start in range(len(all_ids) - len(cand) + 1):
            if all_ids[start:start+len(cand)] == cand:
                label_span = (start, start+len(cand))
                break
        if label_span:
            break

    if label_span is None:
        return None

    # Replace with random tokens
    scrambled = all_ids.copy()
    for i in range(label_span[0], label_span[1]):
        scrambled[i] = random.randint(100, vocab_size-100)
    return scrambled

@torch.no_grad()
def generate(model, tokenizer, input_ids, device, max_new_tokens=80):
    ids = torch.tensor([input_ids]).to(device)
    input_len = ids.shape[1]
    out = model.generate(ids, max_new_tokens=max_new_tokens,
                         do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()

def compute_metrics(ref, hyp):
    ref_toks = ref.lower().split()
    hyp_toks = hyp.lower().split()
    if not hyp_toks:
        return {"bleu4": 1.0, "jaccard": 1.0,
                "levenshtein": 1.0, "bertscore": 1.0}
    bleu = sentence_bleu([ref_toks], hyp_toks,
                         smoothing_function=SmoothingFunction().method1)
    ref_set = set(ref_toks)
    hyp_set = set(hyp_toks)
    jacc = len(ref_set & hyp_set) / max(len(ref_set | hyp_set), 1)
    # Levenshtein similarity (normalised 0-1, higher=more similar)
    max_len = max(len(ref), len(hyp), 1)
    lev_dist = Levenshtein.distance(ref, hyp)
    lev_sim  = 1.0 - lev_dist / max_len
    return {"bleu4": bleu, "jaccard": jacc, "levenshtein": lev_sim}

def run(args):
    random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    base = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        quantization_config=bnb, device_map={"": device},
        attn_implementation="eager")

    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, args.adapter)
        print(f"Loaded adapter: {args.adapter}")
    else:
        model = base
        print("Base model loaded")
    model.eval()

    # Load regions
    with open(args.inputs_json) as f:
        inputs = json.load(f)

    all_regions = []
    for rec in inputs:
        caption = rec.get("coconut_caption", "")
        for reg in rec["regions"]:
            all_regions.append({
                "image_id":   rec["image_id"],
                "mask_index": reg["mask_index"],
                "label":      reg["region_label"],
                "style":      reg["style_name"],
                "caption":    caption,
            })

    random.shuffle(all_regions)
    sample = all_regions[:args.n_sample]

    results = {}
    summary = {}

    templates_to_test = args.templates.split(",")

    for tmpl_name in templates_to_test:
        tmpl_fn = TEMPLATES[tmpl_name]
        print(f"\n{'='*60}")
        print(f"Template {tmpl_name}")
        print(f"{'='*60}")

        tmpl_results = []
        metrics = {
            "standard": {"bleu4":[], "jaccard":[], "levenshtein":[], "label_drop":[]},
            "caption_masked": {"bleu4":[], "jaccard":[], "levenshtein":[], "label_drop":[]},
        }

        for reg in sample:
            label   = reg["label"]
            style   = reg["style"]
            caption = reg["caption"]

            # 1. Original instruction
            prompt_orig = tmpl_fn(label, style, caption)
            instr_orig  = generate(model, tokenizer,
                tokenizer.encode(prompt_orig, add_special_tokens=True),
                device)

            # 2. Standard causal: scramble label tokens only
            scrambled_ids = scramble_label_tokens(tokenizer, prompt_orig, label, device)
            if scrambled_ids is None:
                continue
            instr_label_scrambled = generate(model, tokenizer, scrambled_ids, device)

            # 3. Caption-masked: scramble label tokens + mask caption mentions
            masked_caption = mask_caption(caption, label)
            prompt_masked  = tmpl_fn(label, style, masked_caption)
            scrambled_masked_ids = scramble_label_tokens(
                tokenizer, prompt_masked, label, device)
            if scrambled_masked_ids is None:
                continue
            instr_both_scrambled = generate(model, tokenizer, scrambled_masked_ids, device)

            # Label presence
            label_words = get_label_words(label)
            orig_has_label    = any(w in instr_orig.lower() for w in label_words)
            scr_has_label     = any(w in instr_label_scrambled.lower() for w in label_words)
            masked_has_label  = any(w in instr_both_scrambled.lower() for w in label_words)

            # Standard causal metrics
            m_std = compute_metrics(instr_orig, instr_label_scrambled)
            m_std["label_drop"] = 1.0 if (orig_has_label and not scr_has_label) else 0.0

            # Caption-masked metrics
            m_cap = compute_metrics(instr_orig, instr_both_scrambled)
            m_cap["label_drop"] = 1.0 if (orig_has_label and not masked_has_label) else 0.0

            for k in ["bleu4","jaccard","levenshtein","label_drop"]:
                metrics["standard"][k].append(m_std[k])
                metrics["caption_masked"][k].append(m_cap[k])

            tmpl_results.append({
                "label": label,
                "style": style,
                "caption_mentions": get_label_words(label),
                "masked_caption": masked_caption != caption,
                "standard": m_std,
                "caption_masked": m_cap,
            })

        n = len(tmpl_results)
        std_drop  = np.mean(metrics["standard"]["label_drop"]) * 100
        cap_drop  = np.mean(metrics["caption_masked"]["label_drop"]) * 100
        std_bleu  = np.mean(metrics["standard"]["bleu4"])
        cap_bleu  = np.mean(metrics["caption_masked"]["bleu4"])
        std_lev   = np.mean(metrics["standard"]["levenshtein"])
        cap_lev   = np.mean(metrics["caption_masked"]["levenshtein"])
        std_jacc  = np.mean(metrics["standard"]["jaccard"])
        cap_jacc  = np.mean(metrics["caption_masked"]["jaccard"])

        print(f"  n={n}")
        print(f"  Standard (label scramble only):")
        print(f"    BLEU-4={std_bleu:.3f}  Jaccard={std_jacc:.3f}  Lev={std_lev:.3f}  Label drop={std_drop:.1f}pp")
        print(f"  Caption-masked (label + caption scramble):")
        print(f"    BLEU-4={cap_bleu:.3f}  Jaccard={cap_jacc:.3f}  Lev={cap_lev:.3f}  Label drop={cap_drop:.1f}pp")
        print(f"  Delta label drop: {cap_drop-std_drop:+.1f}pp")
        if cap_drop > std_drop + 5:
            print(f"  -> Caption was carrying region info for this template")
        elif cap_drop <= std_drop + 2:
            print(f"  -> No caption confound — grounding (or lack of it) was genuine")

        results[tmpl_name] = tmpl_results
        summary[tmpl_name] = {
            "n": n,
            "standard_bleu4":        round(float(std_bleu), 3),
            "standard_jaccard":      round(float(std_jacc), 3),
            "standard_levenshtein":  round(float(std_lev), 3),
            "standard_label_drop":   round(float(std_drop), 1),
            "caption_masked_bleu4":       round(float(cap_bleu), 3),
            "caption_masked_jaccard":     round(float(cap_jacc), 3),
            "caption_masked_levenshtein": round(float(cap_lev), 3),
            "caption_masked_label_drop":  round(float(cap_drop), 1),
            "delta_label_drop":     round(float(cap_drop - std_drop), 1),
            "delta_bleu4":          round(float(cap_bleu - std_bleu), 3),
            "delta_levenshtein":    round(float(cap_lev - std_lev), 3),
        }

    # Final table
    print(f"\n{'='*70}")
    print("CAPTION-MASKED CAUSAL TEST — SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Tmpl':<6} {'Std drop':>10} {'Cap drop':>10} {'Delta':>8} {'Confound?':>12}")
    print(f"  {'-'*50}")
    for t, s in sorted(summary.items(), key=lambda x: -x[1]["standard_label_drop"]):
        confound = "YES" if s["delta_label_drop"] > 5 else "no"
        print(f"  {t:<4} {s['standard_label_drop']:>9.1f}pp "
              f"{s['caption_masked_label_drop']:>9.1f}pp "
              f"{s['delta_label_drop']:>+7.1f}pp {confound:>12}")

    out = {"summary": summary, "results": results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.output}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs_json",
        default="../../data/coconut_subset/annotations/prompt_curation_inputs.json")
    p.add_argument("--output",
        default="../results/causal_test_caption_masked.json")
    p.add_argument("--templates", default="A,C,H,G,E")
    p.add_argument("--n_sample", type=int, default=200)
    p.add_argument("--adapter", default=None)
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())
