"""
Unified attention-weight extraction across all 9 templates, with
optional LoRA adapter. Reuses the proven generate_and_attend logic
from step1_generate_and_attend.py (Template A, base model — validated),
extended to loop over templates A-I and accept an optional adapter,
so raw att_weights are saved consistently everywhere.

Templates copied exactly from step4_attention_finetuned.py to ensure
consistency with the rest of the pipeline.
"""
import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

def get_label_indices(tokenizer, prompt, target_text):
    if not target_text:
        return []
    input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    if not target_ids:
        return []
    for i in range(len(input_ids) - len(target_ids) + 1):
        if input_ids[i:i+len(target_ids)] == target_ids:
            return list(range(i, i + len(target_ids)))
    return []

def generate_and_attend(model, tokenizer, prompt, region_label, style_name, device, max_new_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_len = inputs["input_ids"].shape[1]
    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0,
        output_attentions=True, return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_ids = output.sequences[0][input_len:]
    instruction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    step0_atts = output.attentions[0]
    att_stack = torch.stack([a[0] for a in step0_atts], dim=0)
    att_mean = att_stack.mean(dim=(0, 1))
    last_row = att_mean[-1, :input_len]
    last_row = last_row / (last_row.sum() + 1e-8)
    att_weights = last_row.cpu().float().tolist()

    input_ids = inputs["input_ids"][0].tolist()
    tokens_decoded = [tokenizer.decode([i]) for i in input_ids]

    label_indices = get_label_indices(tokenizer, prompt, region_label)
    style_indices = get_label_indices(tokenizer, prompt, style_name)

    label_mass = float(sum(att_weights[i] for i in label_indices if i < len(att_weights)))
    style_mass = float(sum(att_weights[i] for i in style_indices if i < len(att_weights)))

    return {
        "instruction": instruction, "att_weights": att_weights,
        "tokens_decoded": tokens_decoded,
        "label_token_indices": label_indices, "style_token_indices": style_indices,
        "label_attention_mass": label_mass, "style_attention_mass": style_mass,
        "n_input_tokens": input_len,
    }

def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {args.model} (adapter: {args.adapter or 'none — base model'})")

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb_config, device_map={"": device},
        attn_implementation="eager",
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    with open(args.json) as f:
        raw_data = json.load(f)

    data = []
    for image_item in raw_data:
        image_id = image_item["image_id"]
        caption = image_item.get("coconut_caption", "")
        for region in image_item["regions"]:
            data.append({
                "image_id": image_id,
                "mask_index": region["mask_index"],
                "region_label": region["region_label"],
                "style_name": region["style_name"],
                "caption": caption,
            })
    print(f"Flattened {len(raw_data)} images into {len(data)} regions")

    templates_to_run = args.templates.split(",") if args.templates else list(TEMPLATES.keys())

    per_template = {}
    for tmpl in templates_to_run:
        build_prompt = TEMPLATES[tmpl]
        print(f"\n=== Template {tmpl} ===")
        results = []
        n = 0
        for item in tqdm(data, desc=f"T{tmpl}"):
            if args.max_regions and n >= args.max_regions:
                break
            label = item["region_label"]
            style = item["style_name"]
            caption = item.get("caption", "")
            image_id = item["image_id"]
            mask_index = item["mask_index"]

            prompt = build_prompt(label, style, caption)
            with torch.no_grad():
                res = generate_and_attend(model, tokenizer, prompt, label, style, device)

            results.append({
                "image_id": image_id, "mask_index": mask_index,
                "region_label": label, "style_name": style,
                "template": tmpl, "lora": args.adapter or "base",
                "prompt": prompt,
                "instruction": res["instruction"],
                "att_weights": res["att_weights"],
                "tokens_decoded": res["tokens_decoded"],
                "label_token_indices": res["label_token_indices"],
                "style_token_indices": res["style_token_indices"],
                "label_attention_mass": res["label_attention_mass"],
                "style_attention_mass": res["style_attention_mass"],
                "n_input_tokens": res["n_input_tokens"],
            })
            n += 1
        per_template[tmpl] = results

    out = {"model": args.model, "adapter": args.adapter or "base", "per_template": per_template}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f)
    print(f"\nSaved: {args.output}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json", default="../../data/coconut_subset/annotations/prompt_curation_inputs.json")
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    p.add_argument("--adapter", default=None, help="Path to LoRA adapter, omit for base model")
    p.add_argument("--templates", default=None, help="Comma-separated e.g. B,C,D — omit for all 9")
    p.add_argument("--max_regions", type=int, default=0)
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())
