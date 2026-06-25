"""
Step 2c — Test Template I vs Template E
========================================
Targeted run: only templates E and I on all 229 regions.
Checks whether I fixes E's refusal problem while maintaining
high attention mass.

Usage:
    export CUDA_VISIBLE_DEVICES=1
    python3 step2c_test_template_I.py \
        --json ../../data/coconut_subset/annotations/prompt_curation_inputs.json \
        --existing_results ../results/template_comparison_mistral.json \
        --output ../results/template_EI_comparison.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm


REFUSAL_PHRASES = [
    "i'm an ai", "i cannot", "i don't have",
    "as an ai", "language model", "i am an ai",
    "unfortunately, i", "i'm unable"
]

VISUAL_DESCRIPTORS = [
    "brushstroke","brush stroke","texture","colour","color","hue","tone",
    "light","shadow","contrast","palette","pigment","paint","stroke",
    "impressionist","expressionist","luminous","vivid","blend",
    "swirl","gestural","dynamic","loose","thick","impasto","dab"
]


def template_E(label, style, caption):
    return (
        f"[INST] Image scene: {caption}\n"
        f"What single instruction would transfer {style} artistic style "
        f"specifically to the {label}, leaving everything else untouched? [/INST]"
    )


def template_I(label, style, caption):
    return (
        f"[INST] Image scene: {caption}\n"
        f"Write a single image editing instruction that transfers {style} style "
        f"specifically to the {label} in the scene, "
        f"leaving all other regions completely unchanged.\n"
        f"The instruction must describe what to do to the {label} only, "
        f"using visual and stylistic language. [/INST]"
    )


TEMPLATES = {"E": template_E, "I": template_I}


def get_token_indices(tokenizer, prompt, text):
    all_ids  = tokenizer.encode(prompt, add_special_tokens=True)
    txt_ids  = tokenizer.encode(text, add_special_tokens=False)
    for i in range(len(all_ids) - len(txt_ids) + 1):
        if all_ids[i:i+len(txt_ids)] == txt_ids:
            return list(range(i, i + len(txt_ids)))
    return []


@torch.no_grad()
def generate_and_attend(model, tokenizer, prompt, label, style, device,
                        max_new_tokens=80):
    inputs   = tokenizer(prompt, return_tensors="pt",
                         truncation=True, max_length=512).to(device)
    input_len = inputs["input_ids"].shape[1]

    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        output_attentions=True, return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    instruction = tokenizer.decode(
        output.sequences[0][input_len:], skip_special_tokens=True
    ).strip()

    step0     = output.attentions[0]
    att_stack = torch.stack([a[0] for a in step0], dim=0)
    att_mean  = att_stack.mean(dim=(0, 1))
    last_row  = att_mean[-1, :input_len]
    last_row  = last_row / (last_row.sum() + 1e-8)
    weights   = last_row.cpu().float().tolist()

    label_idx = get_token_indices(tokenizer, prompt, label)
    style_idx = get_token_indices(tokenizer, prompt, style)

    return {
        "instruction":  instruction,
        "label_mass":   sum(weights[i] for i in label_idx if i < len(weights)),
        "style_mass":   sum(weights[i] for i in style_idx if i < len(weights)),
        "weights":      weights,
    }


def quality_metrics(instruction, label, style):
    instr = instruction.lower()
    label_words = [w for w in label.lower().split()[:3] if len(w) > 3]
    style_words = [w for w in style.lower().replace("-"," ").split() if len(w) > 3]
    return {
        "word_count":     len(instr.split()),
        "label_coverage": any(w in instr for w in label_words),
        "style_coverage": any(w in instr for w in style_words),
        "visual_descs":   sum(1 for d in VISUAL_DESCRIPTORS if d in instr),
        "has_only":       any(w in instr for w in ["only","exclusively","specifically"]),
        "is_refusal":     any(p in instr for p in REFUSAL_PHRASES),
    }


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Loading Mistral-7B-Instruct-v0.2 in 4-bit...")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        quantization_config=bnb,
        device_map={"": device},
        attn_implementation="eager",
    )
    model.eval()
    print("Model loaded.\n")

    with open(args.json) as f:
        records = json.load(f)

    # Load existing E results for comparison
    with open(args.existing_results) as f:
        existing = json.load(f)

    all_results = {}
    summary     = {}

    for tmpl_name, tmpl_fn in TEMPLATES.items():
        print(f"\n{'='*55}")
        print(f"Template {tmpl_name}")
        print(f"{'='*55}")

        results      = []
        label_masses = []
        style_masses = []
        refusals     = []
        visual_descs = []
        n = 0

        for record in tqdm(records, desc=f"T{tmpl_name}"):
            caption = record.get("coconut_caption", "")
            for region in record["regions"]:
                label = region.get("region_label", "")
                style = region.get("style_name",   "")

                prompt = tmpl_fn(label, style, caption)
                res    = generate_and_attend(
                    model, tokenizer, prompt, label, style, device
                )
                qm = quality_metrics(res["instruction"], label, style)

                results.append({
                    "image_id":             record["image_id"],
                    "mask_index":           region["mask_index"],
                    "region_label":         label,
                    "style_name":           style,
                    "template":             tmpl_name,
                    "instruction":          res["instruction"],
                    "label_attention_mass": res["label_mass"],
                    "style_attention_mass": res["style_mass"],
                    "is_refusal":           qm["is_refusal"],
                    "word_count":           qm["word_count"],
                    "visual_descs":         qm["visual_descs"],
                    "has_only":             qm["has_only"],
                    "prompt":               prompt,
                })
                label_masses.append(res["label_mass"])
                style_masses.append(res["style_mass"])
                refusals.append(qm["is_refusal"])
                visual_descs.append(qm["visual_descs"])
                n += 1

        lm = np.array(label_masses)
        sm = np.array(style_masses)
        ref_rate = np.mean(refusals) * 100

        print(f"  Label mass mean   : {lm.mean():.4f}")
        print(f"  Style mass mean   : {sm.mean():.4f}")
        print(f"  Refusal rate      : {ref_rate:.1f}%")
        print(f"  Visual descs mean : {np.mean(visual_descs):.2f}")

        all_results[tmpl_name] = results
        summary[tmpl_name] = {
            "label_mass":   {"mean": round(float(lm.mean()), 4),
                             "median": round(float(np.median(lm)), 4)},
            "style_mass":   {"mean": round(float(sm.mean()), 4)},
            "refusal_rate": round(ref_rate, 1),
            "visual_descs": round(float(np.mean(visual_descs)), 2),
            "n": n,
        }

    # Add existing A results for 3-way comparison
    existing_A = existing["summary"]["A"]
    summary["A_existing"] = {
        "label_mass":   existing_A["label_mass"],
        "style_mass":   existing_A["style_mass"],
        "refusal_rate": 0.0,
        "visual_descs": 2.18,
        "n": 229,
    }

    # Final comparison
    print(f"\n{'='*55}")
    print("Comparison: A (existing) vs E vs I")
    print(f"{'='*55}")
    print(f"  {'Tmpl':<6} {'LblMass':>9} {'StlMass':>9} {'Refusal':>9} {'VisDsc':>8}")
    print(f"  {'-'*45}")
    for t, s in [("A_existing", summary["A_existing"]),
                 ("E",          summary["E"]),
                 ("I",          summary["I"])]:
        print(f"  {t:<10} {s['label_mass']['mean']*100:>8.3f}% "
              f"{s['style_mass']['mean']*100:>8.3f}% "
              f"{s['refusal_rate']:>8.1f}% "
              f"{s['visual_descs']:>7.2f}")

    out = {
        "model":        "mistral-7b-instruct-v0.2-transformers",
        "summary":      summary,
        "per_template": all_results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nOutput: {args.output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json",
        default="../../data/coconut_subset/annotations/prompt_curation_inputs.json")
    p.add_argument("--existing_results",
        default="../results/template_comparison_mistral.json")
    p.add_argument("--output",
        default="../results/template_EI_comparison.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
