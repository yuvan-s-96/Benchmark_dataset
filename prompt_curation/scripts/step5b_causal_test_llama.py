"""
Step 5b — Causal label ablation test, LLAMA VERSION
======================================================
Adapted from step5b_causal_test.py. Same logic (scramble label tokens,
measure output divergence), but:
  - Templates stripped of [INST]/[/INST] wrapper (content only)
  - Prompt built via tokenizer.apply_chat_template() to match Llama's
    actual training format
  - scramble_label uses add_special_tokens=False (prompt already contains
    <|begin_of_text|> as literal text from apply_chat_template) and
    leading-space label matching (Llama's BPE tokenizes " the" differently
    from "the") -- both fixes validated against step1_full_weights_llama.py
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
        ref_ngrams = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1))
        hyp_ngrams = Counter(tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1))
        matches = sum((hyp_ngrams & ref_ngrams).values())
        total = sum(hyp_ngrams.values())
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

# Content-only templates -- identical wording to step5b_causal_test.py,
# [INST]/[/INST] wrapper stripped since apply_chat_template adds Llama's
# own formatting instead.
TEMPLATES = {
    "A": lambda l,s,c: (f"You are a style transfer assistant. "
                        f"Region: {l}. Style: {s}. Scene caption: {c}. "
                        f"Write one instruction to apply this style to this region only."),
    "B": lambda l,s,c: (f"TARGET REGION: {l}\nSTYLE TO APPLY: {s}\nSCENE: {c}\n"
                        f"Write a single instruction that applies {s} ONLY to the {l}."),
    "C": lambda l,s,c: (f"Scene description: {c}\n"
                        f"From this scene, focus exclusively on: {l}\n"
                        f"Required artistic style: {s}\n"
                        f"Write one instruction to stylise the {l} in {s} style."),
    "D": lambda l,s,c: (f"Apply {s} style to the {l} in this image.\n"
                        f"Full scene: {c}\n"
                        f"Important: apply the style to {l} only. "
                        f"Preserve all other regions unchanged.\n"
                        f"Write the style transfer instruction."),
    "E": lambda l,s,c: (f"Image scene: {c}\n"
                        f"What single instruction would transfer {s} artistic style "
                        f"specifically to the {l}, leaving everything else untouched?"),
    "F": lambda l,s,c: (f"You are a style transfer assistant.\n"
                        f"Scene: {c}\nTarget region: {l}\nTarget style: {s}\n"
                        f"Think step by step: first identify the {l} in the scene, "
                        f"then describe how to apply {s} specifically to it.\n"
                        f"Write one instruction."),
    "G": lambda l,s,c: (f"You are stylising ONE specific region: {l}.\n"
                        f"Apply {s} to the {l} only.\n"
                        f"Scene context: {c}\n"
                        f"Write a style transfer instruction for the {l}."),
    "H": lambda l,s,c: (f"{s} style transfer task.\n"
                        f"What single instruction would apply {s} specifically to the {l}, "
                        f"leaving everything else in the scene untouched?\n"
                        f"The {l} is the only region to be stylised.\n"
                        f"Scene context: {c}"),
    "I": lambda l,s,c: (f"Image scene: {c}\n"
                        f"Write a single image editing instruction that transfers {s} style "
                        f"specifically to the {l} in the scene, "
                        f"leaving all other regions completely unchanged.\n"
                        f"The instruction must describe what to do to the {l} only, "
                        f"using visual and stylistic language."),
}

def build_prompt(tokenizer, content):
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def find_label_ids(tokenizer, all_ids, label):
    """Try leading-space form first (matches mid-sentence occurrence), then no-space fallback."""
    candidates = [
        tokenizer.encode(" " + label, add_special_tokens=False),
        tokenizer.encode(label, add_special_tokens=False),
    ]
    for cand in candidates:
        if not cand:
            continue
        for i in range(len(all_ids) - len(cand) + 1):
            if all_ids[i:i+len(cand)] == cand:
                return cand, i
    return [], -1

def scramble_label(tokenizer, prompt, label, vocab_size=128000, seed=42):
    rng = random.Random(seed)
    # IMPORTANT: add_special_tokens=False -- prompt already contains
    # <|begin_of_text|> as literal text from apply_chat_template(), matching
    # exactly how generate() tokenizes the same prompt string.
    all_ids = tokenizer.encode(prompt, add_special_tokens=False)
    label_ids, label_start = find_label_ids(tokenizer, all_ids, label)
    if label_start < 0:
        return None, []
    scrambled_ids = list(all_ids)
    replacement = [rng.randint(4, vocab_size-1) for _ in range(len(label_ids))]
    for i, idx in enumerate(range(label_start, label_start+len(label_ids))):
        scrambled_ids[idx] = replacement[i]
    scrambled_prompt = tokenizer.decode(scrambled_ids, skip_special_tokens=False)
    return scrambled_prompt, replacement

@torch.no_grad()
def generate(model, tokenizer, prompt, device, max_new_tokens=80):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512,
                        add_special_tokens=False).to(device)  # BOS already in template text
    input_len = inputs["input_ids"].shape[1]
    output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()

def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    base_model_id = args.model
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=bnb, device_map={"": device},
        attn_implementation="eager")

    if args.adapter:
        print(f"Loading LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(base, args.adapter)
        model.eval()
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
                "image_id": rec["image_id"], "mask_index": reg["mask_index"],
                "label": reg["region_label"], "style": reg["style_name"], "caption": caption,
            })

    random.seed(42)
    sample = random.sample(all_regions, min(args.n_sample, len(all_regions)))
    print(f"Sample size: {len(sample)} regions x {len(TEMPLATES)} templates\n")

    templates_to_run = args.templates.split(",") if args.templates else list(TEMPLATES.keys())

    results = {}
    summary = {}
    for tmpl_name in templates_to_run:
        tmpl_fn = TEMPLATES[tmpl_name]
        print(f"\n{'='*55}\nTemplate {tmpl_name} — causal ablation\n{'='*55}")
        tmpl_results = []
        bleu_scores, jacc_scores, lev_scores = [], [], []
        orig_instrs, scram_instrs = [], []
        label_in_orig, label_in_scram = [], []
        n_skipped = 0

        for reg in tqdm(sample, desc=f"T{tmpl_name}"):
            label, style, caption = reg["label"], reg["style"], reg["caption"]
            content = tmpl_fn(label, style, caption)
            prompt_orig = build_prompt(tokenizer, content)
            prompt_scram, replacement = scramble_label(tokenizer, prompt_orig, label)
            if prompt_scram is None:
                n_skipped += 1
                continue

            instr_orig = generate(model, tokenizer, prompt_orig, device)
            instr_scram = generate(model, tokenizer, prompt_scram, device)

            orig_toks = instr_orig.lower().split()
            scram_toks = instr_scram.lower().split()
            b4 = bleu4(orig_toks, scram_toks)
            jc = jaccard(orig_toks, scram_toks)
            lev = lev_ratio(instr_orig.lower(), instr_scram.lower())

            label_words = [w for w in label.lower().split() if len(w) > 3]
            lo = any(w in instr_orig.lower() for w in label_words)
            ls = any(w in instr_scram.lower() for w in label_words)

            bleu_scores.append(b4); jacc_scores.append(jc); lev_scores.append(lev)
            orig_instrs.append(instr_orig); scram_instrs.append(instr_scram)
            label_in_orig.append(lo); label_in_scram.append(ls)

            tmpl_results.append({
                "image_id": reg["image_id"], "region_label": label, "style_name": style,
                "instruction_original": instr_orig, "instruction_scrambled": instr_scram,
                "bleu4": round(b4, 4), "jaccard": round(jc, 4), "levenshtein": round(lev, 4),
                "label_in_orig": lo, "label_in_scram": ls,
            })

        mean_b, mean_j, mean_lev = np.mean(bleu_scores), np.mean(jacc_scores), np.mean(lev_scores)
        print(f"  Computing BERTScore for {len(orig_instrs)} pairs...")
        P, R, F1 = bert_score_fn(orig_instrs, scram_instrs, lang="en",
                                   model_type="distilbert-base-uncased", verbose=False, device=device)
        mean_bert = float(F1.mean())
        lo_rate, ls_rate = np.mean(label_in_orig), np.mean(label_in_scram)

        print(f"  BLEU-4: {mean_b:.3f}  Jaccard: {mean_j:.3f}  Lev: {mean_lev:.3f}  "
              f"BERT-F1: {mean_bert:.3f}  Label drop: {(lo_rate-ls_rate)*100:.1f}pp  Skipped: {n_skipped}")

        results[tmpl_name] = tmpl_results
        summary[tmpl_name] = {
            "bleu4_mean": round(float(mean_b), 4), "jaccard_mean": round(float(mean_j), 4),
            "levenshtein_mean": round(float(mean_lev), 4), "bertscore_f1_mean": round(float(mean_bert), 4),
            "label_in_orig_pct": round(float(lo_rate*100), 1), "label_in_scram_pct": round(float(ls_rate*100), 1),
            "label_drop_pp": round(float((lo_rate-ls_rate)*100), 1),
            "n_scored": len(tmpl_results), "n_skipped": n_skipped,
        }

    out = {
        "model": args.model, "experiment": "causal label ablation (llama)",
        "n_sample": len(sample),
        "description": "Llama-3.1 version. Label tokens replaced with random vocab tokens (seed=42).",
        "summary": summary, "per_template": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.output}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs_json", default="../../data/coconut_subset/annotations/prompt_curation_inputs.json")
    p.add_argument("--output", default="../results/causal_test_llama.json")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--n_sample", type=int, default=200)
    p.add_argument("--templates", default=None)
    p.add_argument("--adapter", default=None)
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())
