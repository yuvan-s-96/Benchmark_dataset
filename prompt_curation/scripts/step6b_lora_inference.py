"""
Step 6b — Fast LoRA instruction generation for LLM judge
===========================================================
Generates instructions from fine-tuned LoRA models on the 30
human-rating regions only. No attention extraction, no causal
test — just fast text generation for judge input preparation.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

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
}

@torch.no_grad()
def generate(model, tokenizer, prompt, device, max_new_tokens=768):
    ids = tokenizer.encode(prompt, add_special_tokens=True, truncation=True, max_length=512)
    ids_t = torch.tensor([ids]).to(device)
    out = model.generate(input_ids=ids_t, max_new_tokens=max_new_tokens,
                         do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][len(ids):], skip_special_tokens=True).strip()

def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Adapter: {args.adapter}")
    print(f"Template: {args.template}")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    base = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        quantization_config=bnb, device_map={"": device},
        attn_implementation="eager")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    print("Model loaded")

    with open(args.inputs_json) as f:
        inputs = json.load(f)
    with open(args.indices_json) as f:
        idx_data = json.load(f)

    with open(args.template_comparison_json) as f:
        tmpl = json.load(f)
    anchor = tmpl["per_template"]["A"]

    tmpl_fn = TEMPLATES[args.template]
    results = []

    for i, idx in enumerate(idx_data["indices"]):
        if idx >= len(anchor):
            continue
        r = anchor[idx]
        image_id   = r["image_id"]
        mask_index = r["mask_index"]
        label      = r["region_label"]
        style      = r["style_name"]

        rec = next((x for x in inputs if str(x["image_id"])==str(image_id)), None)
        caption = rec.get("coconut_caption","") if rec else ""

        prompt = tmpl_fn(label, style, caption)
        instr  = generate(model, tokenizer, prompt, device)

        results.append({
            "region_num": i+1,
            "image_id": image_id,
            "mask_index": mask_index,
            "region_label": label,
            "style_name": style,
            "instruction": instr,
        })
        print(f"  [{i+1:02d}/30] {label[:30]:<30} len={len(instr)}")

    out = {"adapter": args.adapter, "template": args.template, "results": results}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.output}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--template", required=True, choices=["A","C","H"])
    p.add_argument("--inputs_json",
        default="../../data/coconut_subset/annotations/prompt_curation_inputs.json")
    p.add_argument("--template_comparison_json",
        default="../results/template_comparison_979_final.json")
    p.add_argument("--indices_json",
        default="../results/human_rating_indices.json")
    p.add_argument("--output", required=True)
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())
