"""
Step 4 — Attention extraction on fine-tuned models
====================================================
Loads LoRA adapter on top of base Mistral-7B and re-runs
attention extraction on all 979 regions using all 9 templates.

Compares attention mass before vs after fine-tuning:
  - Does label attention mass increase?
  - Does variance (CV) decrease? — invariant grounding test
  - Do all templates improve, or just the training template?

Usage:
    export CUDA_VISIBLE_DEVICES=1
    python3 step4_attention_finetuned.py \
        --adapter ../models/lora_A/adapter \
        --lora_name lora_A \
        --output ../results/attention_lora_A.json

    python3 step4_attention_finetuned.py \
        --adapter ../models/lora_H/adapter \
        --lora_name lora_H \
        --output ../results/attention_lora_H.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm


TEMPLATES = {
    "A": lambda l,s,c: (f"[INST] You are a style transfer assistant. "
                        f"Region: {l}. Style: {s}. Scene caption: {c}. "
                        f"Write one instruction to apply this style to this region only. [/INST]"),
    "B": lambda l,s,c: (f"[INST] TARGET REGION: {l}\nSTYLE TO APPLY: {s}\nSCENE: {c}\n"
                        f"Write a single instruction that applies {s} ONLY to the {l}. "
                        f"Do not mention other regions. [/INST]"),
    "C": lambda l,s,c: (f"[INST] Scene description: {c}\n"
                        f"From this scene, focus exclusively on: {l}\n"
                        f"Required artistic style: {s}\n"
                        f"Write one instruction to stylise the {l} in {s} style. [/INST]"),
    "D": lambda l,s,c: (f"[INST] Apply {s} style to the {l} in this image.\n"
                        f"Full scene: {c}\n"
                        f"Important: apply the style to {l} only. "
                        f"Preserve all other regions unchanged.\n"
                        f"Write the style transfer instruction. [/INST]"),
    "E": lambda l,s,c: (f"[INST] Image scene: {c}\n"
                        f"What single instruction would transfer {s} artistic style "
                        f"specifically to the {l}, leaving everything else untouched? [/INST]"),
    "F": lambda l,s,c: (f"[INST] You are a style transfer assistant.\n"
                        f"Scene: {c}\nTarget region: {l}\nTarget style: {s}\n"
                        f"Think step by step: first identify the {l} in the scene, "
                        f"then describe how to apply {s} specifically to it.\n"
                        f"Write one instruction. [/INST]"),
    "G": lambda l,s,c: (f"[INST] You are stylising ONE specific region: {l}.\n"
                        f"Apply {s} to the {l} only.\n"
                        f"Scene context: {c}\n"
                        f"Write a style transfer instruction for the {l}. [/INST]"),
    "H": lambda l,s,c: (f"[INST] {s} style transfer task.\n"
                        f"What single instruction would apply {s} specifically to the {l}, "
                        f"leaving everything else in the scene untouched?\n"
                        f"The {l} is the only region to be stylised.\n"
                        f"Scene context: {c} [/INST]"),
    "I": lambda l,s,c: (f"[INST] Image scene: {c}\n"
                        f"Write a single image editing instruction that transfers {s} style "
                        f"specifically to the {l} in the scene, "
                        f"leaving all other regions completely unchanged.\n"
                        f"The instruction must describe what to do to the {l} only, "
                        f"using visual and stylistic language. [/INST]"),
}


def get_label_indices(tokenizer, prompt, text):
    all_ids = tokenizer.encode(prompt, add_special_tokens=True)
    txt_ids = tokenizer.encode(text, add_special_tokens=False)
    for i in range(len(all_ids) - len(txt_ids) + 1):
        if all_ids[i:i+len(txt_ids)] == txt_ids:
            return list(range(i, i+len(txt_ids)))
    return []


@torch.no_grad()
def generate_and_attend(model, tokenizer, prompt, label, style, device):
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=512).to(device)
    input_len = inputs["input_ids"].shape[1]

    output = model.generate(
        **inputs, max_new_tokens=512, do_sample=False,
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

    li = get_label_indices(tokenizer, prompt, label)
    si = get_label_indices(tokenizer, prompt, style)
    lm = sum(weights[i] for i in li if i < len(weights))
    sm = sum(weights[i] for i in si if i < len(weights))

    return instruction, lm, sm


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Adapter: {args.adapter}")
    print(f"LoRA name: {args.lora_name}")

    print("\nLoading base model + LoRA adapter...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        quantization_config=bnb,
        device_map={"": device},
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()
    print("Model loaded with LoRA adapter\n")

    # Load inputs
    with open(args.inputs_json) as f:
        records = json.load(f)

    all_results = {}
    summary     = {}

    for tmpl_name, tmpl_fn in TEMPLATES.items():
        print(f"\n{'='*50}")
        print(f"Template {tmpl_name}")
        print(f"{'='*50}")

        results      = []
        label_masses = []
        style_masses = []
        n = 0

        for record in tqdm(records, desc=f"T{tmpl_name}"):
            caption = record.get("coconut_caption", "")
            for region in record["regions"]:
                label = region["region_label"]
                style = region["style_name"]
                prompt = tmpl_fn(label, style, caption)

                instr, lm, sm = generate_and_attend(
                    model, tokenizer, prompt, label, style, device
                )

                results.append({
                    "image_id":             record["image_id"],
                    "mask_index":           region["mask_index"],
                    "region_label":         label,
                    "style_name":           style,
                    "template":             tmpl_name,
                    "lora":                 args.lora_name,
                    "prompt":               prompt,
                    "instruction":          instr,
                    "label_attention_mass": lm,
                    "style_attention_mass": sm,
                })
                label_masses.append(lm)
                style_masses.append(sm)
                n += 1

        lm_arr = np.array(label_masses)
        sm_arr = np.array(style_masses)
        cv     = lm_arr.std() / lm_arr.mean()

        print(f"  Label mass mean : {lm_arr.mean()*100:.3f}%")
        print(f"  Style mass mean : {sm_arr.mean()*100:.3f}%")
        print(f"  CV              : {cv:.3f}")

        all_results[tmpl_name] = results
        summary[tmpl_name] = {
            "label_mass": {
                "mean":   round(float(lm_arr.mean()), 4),
                "median": round(float(np.median(lm_arr)), 4),
                "std":    round(float(lm_arr.std()), 4),
                "cv":     round(float(cv), 4),
            },
            "style_mass": {"mean": round(float(sm_arr.mean()), 4)},
            "n": n,
        }

    # Print comparison vs baseline
    print(f"\n{'='*65}")
    print(f"COMPARISON — baseline vs {args.lora_name}")
    print(f"{'='*65}")
    baseline = {
        "A": {"mean": 0.00632, "cv": 0.668},
        "B": {"mean": 0.00489, "cv": 0.619},
        "C": {"mean": 0.00326, "cv": 1.169},
        "D": {"mean": 0.00502, "cv": 0.599},
        "E": {"mean": 0.00554, "cv": 1.409},
        "F": {"mean": 0.00252, "cv": 1.051},
        "G": {"mean": 0.00470, "cv": 0.565},
        "H": {"mean": 0.00558, "cv": 0.594},
        "I": {"mean": 0.00315, "cv": 1.129},
    }
    print(f"\n  {'Tmpl':<6} {'Base mean':>10} {'FT mean':>10} "
          f"{'Delta':>8} {'Base CV':>9} {'FT CV':>8}")
    print(f"  {'-'*55}")
    for t in ["A","B","C","D","E","F","G","H","I"]:
        if t not in summary:
            continue
        bm = baseline[t]["mean"]
        fm = summary[t]["label_mass"]["mean"]
        bc = baseline[t]["cv"]
        fc = summary[t]["label_mass"]["cv"]
        delta = (fm - bm) * 100
        print(f"  {t:<6} {bm*100:>9.3f}% {fm*100:>9.3f}% "
              f"{delta:>+7.3f}% {bc:>9.3f} {fc:>8.3f}")

    out = {
        "lora":       args.lora_name,
        "adapter":    args.adapter,
        "summary":    summary,
        "per_template": all_results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter",     required=True)
    p.add_argument("--lora_name",   required=True)
    p.add_argument("--base_model",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="HuggingFace base model id")
    p.add_argument("--inputs_json",
        default="../../data/coconut_subset/annotations/prompt_curation_inputs.json")
    p.add_argument("--output",      required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
