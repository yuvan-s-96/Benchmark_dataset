"""
Step 7b — InstructPix2Pix evaluation with Llama LoRA fine-tuned models
=========================================================================
Adapted from step7b_instruct_pix2pix_lora.py. Template content identical
to the Mistral version, wrapped via apply_chat_template instead of a
hardcoded [INST]...[/INST] string. Same tokenization fix as the other
Llama scripts: add_special_tokens=False in generation, since the prompt
already contains <|begin_of_text|> as literal text from the chat template.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from diffusers import StableDiffusionInstructPix2PixPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import clip

REFUSAL_PHRASES = ["i'm an ai","i cannot","language model","as an ai","i'm unable"]

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
                        f"Full scene: {c}\nPreserve all other regions unchanged.\n"
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
                        f"Apply {s} to the {l} only.\nScene context: {c}\n"
                        f"Write a style transfer instruction for the {l}."),
    "H": lambda l,s,c: (f"{s} style transfer task.\n"
                        f"What single instruction would apply {s} specifically to the {l}, "
                        f"leaving everything else in the scene untouched?\n"
                        f"The {l} is the only region to be stylised.\n"
                        f"Scene context: {c}"),
    "I": lambda l,s,c: (f"Image scene: {c}\n"
                        f"Write a single image editing instruction that transfers {s} style "
                        f"specifically to the {l} in the scene, "
                        f"leaving all other regions completely unchanged."),
}

def is_refusal(text):
    return any(p in text.lower() for p in REFUSAL_PHRASES)

def load_mask(pan_dir, pan_json, image_id, mask_index):
    import json as _json
    if not hasattr(load_mask, "_lookup"):
        lookup_path = Path(pan_dir) / "segment_lookup.json"
        with open(lookup_path) as f:
            load_mask._lookup = _json.load(f)
    key = f"{image_id}_{mask_index}"
    info = load_mask._lookup.get(key)
    if info is None:
        return None
    seg_id = info["segment_id"]
    pan_path = Path(pan_dir) / f"{str(image_id).zfill(12)}.png"
    if not pan_path.exists():
        return None
    pan_img = np.array(Image.open(pan_path).convert("RGB"))
    segment_map = (pan_img[:,:,0].astype(np.int32) +
                   pan_img[:,:,1].astype(np.int32) * 256 +
                   pan_img[:,:,2].astype(np.int32) * 65536)
    mask = (segment_map == seg_id).astype(np.uint8) * 255
    return mask

def masked_clip_score(clip_model, clip_preprocess, stylised_img, orig_img,
                      mask, style_ref_path, device):
    sty_arr  = np.array(stylised_img)
    orig_arr = np.array(orig_img)
    mask_3ch = np.stack([mask, mask, mask], axis=2) / 255.0
    inv_mask = 1.0 - mask_3ch
    sty_region = (sty_arr * mask_3ch).astype(np.uint8)
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any() or not cols.any():
        return None, None
    rmin, rmax = np.where(rows)[0][[0,-1]]
    cmin, cmax = np.where(cols)[0][[0,-1]]
    region_crop = Image.fromarray(sty_region[rmin:rmax+1, cmin:cmax+1])
    sty_bg   = (sty_arr  * inv_mask).astype(np.uint8)
    orig_bg  = (orig_arr * inv_mask).astype(np.uint8)
    style_ref = Image.open(style_ref_path).convert("RGB")
    with torch.no_grad():
        region_t   = clip_preprocess(region_crop).unsqueeze(0).to(device)
        style_t    = clip_preprocess(style_ref).unsqueeze(0).to(device)
        sty_bg_t   = clip_preprocess(Image.fromarray(sty_bg)).unsqueeze(0).to(device)
        orig_bg_t  = clip_preprocess(Image.fromarray(orig_bg)).unsqueeze(0).to(device)
        rf = clip_model.encode_image(region_t)
        sf = clip_model.encode_image(style_t)
        sbf = clip_model.encode_image(sty_bg_t)
        obf = clip_model.encode_image(orig_bg_t)
        rf  = rf  / rf.norm(dim=-1, keepdim=True)
        sf  = sf  / sf.norm(dim=-1, keepdim=True)
        sbf = sbf / sbf.norm(dim=-1, keepdim=True)
        obf = obf / obf.norm(dim=-1, keepdim=True)
        region_clip    = float((rf * sf).sum())
        nonregion_clip = float((sbf * obf).sum())
    return region_clip, nonregion_clip

@torch.no_grad()
def generate_instruction(model, tokenizer, content, device, max_new_tokens=512):
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512,
                        add_special_tokens=False).to(device)
    input_len = inputs["input_ids"].shape[1]
    out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                        do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()

def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading base model...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb, device_map={"": device},
        attn_implementation="eager")

    if args.adapter:
        print(f"Loading adapter: {args.adapter}")
        model = PeftModel.from_pretrained(base, args.adapter)
    else:
        model = base
    model.eval()
    print("LLM loaded")

    print("Loading InstructPix2Pix...")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    print("Loading CLIP...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    print("All models loaded\n")

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

    import random
    random.seed(42)
    sample = random.sample(all_regions, min(args.n_sample, len(all_regions)))
    templates_to_run = args.templates.split(",")
    print(f"Evaluating {len(sample)} regions x {len(templates_to_run)} templates")
    print(f"Adapter: {args.adapter}\n")

    results = {}
    summary = {}
    for tmpl_name in templates_to_run:
        tmpl_fn = TEMPLATES[tmpl_name]
        print(f"\n{'='*55}\nTemplate {tmpl_name}\n{'='*55}")
        tmpl_results = []
        region_clips, nonregion_clips = [], []
        n_skipped = 0

        for reg in sample:
            label, style, caption = reg["label"], reg["style"], reg["caption"]
            image_id, mask_index = reg["image_id"], reg["mask_index"]

            content = tmpl_fn(label, style, caption)
            instruction = generate_instruction(model, tokenizer, content, device)
            if is_refusal(instruction) or len(instruction.split()) < 5:
                n_skipped += 1
                continue
            instruction_trunc = " ".join(instruction.split()[:60])

            img_path = Path(args.img_dir) / f"{image_id}.jpg"
            if not img_path.exists():
                n_skipped += 1
                continue
            orig_img = Image.open(img_path).convert("RGB").resize((512,512))

            try:
                with torch.autocast("cuda"):
                    out = pipe(instruction_trunc, image=orig_img,
                               num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7.5)
                stylised = out.images[0]
            except Exception:
                n_skipped += 1
                continue

            mask = load_mask(args.pan_dir, args.pan_json, image_id, mask_index)
            style_refs = list(Path(args.style_ref_dir).glob(f"{style}/*.jpg"))
            if not style_refs:
                style_refs = list(Path(args.style_ref_dir).glob(f"{style.replace(' ','-')}/*.jpg"))
            if not style_refs:
                n_skipped += 1
                continue

            if mask is not None:
                mask_r = np.array(Image.fromarray(mask).resize((512,512), Image.NEAREST))
                rc, nc = masked_clip_score(clip_model, clip_preprocess, stylised, orig_img, mask_r, str(style_refs[0]), device)
            else:
                rc, nc = masked_clip_score(clip_model, clip_preprocess, stylised, orig_img,
                    np.ones((512,512), dtype=np.uint8)*255, str(style_refs[0]), device)

            if rc is not None:
                region_clips.append(rc)
                nonregion_clips.append(nc if nc else 0)
                tmpl_results.append({
                    "image_id": image_id, "region_label": label, "style": style,
                    "instruction": instruction[:100],
                    "region_clip": round(rc, 4), "nonregion_clip": round(nc, 4) if nc else None,
                })

        n = len(tmpl_results)
        mean_rc = np.mean(region_clips) if region_clips else 0
        mean_nc = np.mean(nonregion_clips) if nonregion_clips else 0
        print(f"  Scored: {n}  Skipped: {n_skipped}")
        print(f"  Region CLIP: {mean_rc:.4f}  NonRegion CLIP: {mean_nc:.4f}")
        results[tmpl_name] = tmpl_results
        summary[tmpl_name] = {
            "region_clip_mean": round(float(mean_rc), 4), "nonregion_clip_mean": round(float(mean_nc), 4),
            "n_scored": n, "n_skipped": n_skipped,
        }

    model_label = Path(args.adapter).parent.name if args.adapter else "base_model"
    print(f"\n{'='*65}\nFINAL — {model_label} on all templates\n{'='*65}")
    print(f"\n  {'Template':<12} {'Region CLIP':>12} {'NonReg CLIP':>13} {'n':>5}")
    for t in sorted(summary, key=lambda x: -summary[x]["region_clip_mean"]):
        s = summary[t]
        print(f"  {t:<12} {s['region_clip_mean']:>12.4f} {s['nonregion_clip_mean']:>13.4f} {s['n_scored']:>5}")

    out = {
        "adapter": args.adapter, "model": "timbrooks/instruct-pix2pix",
        "base_model": args.base_model, "n_sample": len(sample),
        "summary": summary, "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.output}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default=None, help="Path to LoRA adapter (omit for base model)")
    p.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--inputs_json", default="../../data/coconut_subset/annotations/prompt_curation_inputs.json")
    p.add_argument("--img_dir", default="../../data/coconut_subset/images")
    p.add_argument("--pan_dir", default="/mnt/fast1/yvs23/coconut_panoptic")
    p.add_argument("--pan_json", default="../../data/coconut_subset/annotations/prompt_curation_inputs.json")
    p.add_argument("--style_ref_dir", default="../../data/style_references")
    p.add_argument("--output", required=True)
    p.add_argument("--templates", default="A,B,C,D,E,F,G,H,I")
    p.add_argument("--n_sample", type=int, default=30)
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())
