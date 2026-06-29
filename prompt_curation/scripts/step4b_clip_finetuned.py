"""
Step 4b — CLIP scoring on fine-tuned model outputs
====================================================
Scores instructions from LoRA-A and LoRA-H against
WikiArt style reference images. Compares with baseline.
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm

STYLE_REF_DIR = Path("/homes/yvs23/Benchmark_dataset/data/style_references")
REFUSAL_PHRASES = ["i'm an ai","i cannot","language model","as an ai","i'm unable"]

def cache_styles(model, preprocess, device):
    cache = {}
    for sd in sorted(STYLE_REF_DIR.iterdir()):
        if not sd.is_dir() or sd.name == "style_references":
            continue
        imgs = sorted([f for f in sd.iterdir()
                      if f.suffix.lower() in (".jpg",".jpeg",".png")])
        embs = []
        for ip in imgs:
            try:
                img = preprocess(Image.open(ip).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    e = model.encode_image(img)
                    e = e / e.norm(dim=-1, keepdim=True)
                embs.append(e.cpu().float())
            except:
                continue
        if embs:
            all_e = torch.cat(embs, dim=0)
            cache[sd.name] = all_e
    return cache

def score_instructions(model, style_cache, regions, device):
    scores = []
    for r in regions:
        instr = r.get("instruction","")
        style = r.get("style_name","")
        if any(p in instr.lower() for p in REFUSAL_PHRASES):
            continue
        if style not in style_cache:
            continue
        text = clip.tokenize([instr[:77]], truncate=True).to(device)
        with torch.no_grad():
            te = model.encode_text(text)
            te = te / te.norm(dim=-1, keepdim=True)
        te = te.cpu().float()
        sims = (style_cache[style] @ te.T).squeeze()
        scores.append(float(sims.mean()))
    return scores

device = "cuda"
print("Loading CLIP...")
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

print("Caching style embeddings...")
style_cache = cache_styles(model, preprocess, device)
print(f"Cached {len(style_cache)} styles")

# Baseline CLIP from existing scores
with open("../results/clip_scores.json") as f:
    baseline_clip = json.load(f)

results = {}
for fname, name in [("../results/attention_lora_A.json","LoRA-A"),
                    ("../results/attention_lora_H.json","LoRA-H")]:
    with open(fname) as f:
        d = json.load(f)

    print(f"\n{name}:")
    tmpl_scores = {}
    for tmpl in ["A","B","C","D","E","F","G","H","I"]:
        if tmpl not in d["per_template"]:
            continue
        scores = score_instructions(model, style_cache,
                                    d["per_template"][tmpl], device)
        mean_score = float(np.mean(scores)) if scores else 0
        base_score = baseline_clip["summary"].get(tmpl,{}).get("clip_mean", 0)
        delta = mean_score - base_score
        tmpl_scores[tmpl] = {"clip_mean": round(mean_score,4),
                              "n_scored": len(scores)}
        print(f"  {tmpl}: CLIP={mean_score:.4f}  baseline={base_score:.4f}  "
              f"delta={delta:+.4f}  scored={len(scores)}/979")
    results[name] = tmpl_scores

# Save
out = {"baseline": baseline_clip["summary"], "results": results}
with open("../results/clip_finetuned.json","w") as f:
    json.dump(out, f, indent=2)
print("\nSaved: clip_finetuned.json")
