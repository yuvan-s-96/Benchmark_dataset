"""
Step 8 — ArtFID and Gram Matrix style evaluation, LLAMA VERSION
==================================================================
Adapted from step8_artfid_gram.py. No chat-template changes needed --
this script reads already-generated instruction text from the merged
attention-extraction JSON files (built by step1_full_weights_llama.py),
rather than generating instructions itself. Only the default file paths
differ: Llama's merged attention_maps files instead of Mistral's
template_comparison / attention_lora files, and pan_dir pointed at the
shared data/annotations location rather than ogg-local /mnt/fast1.

Verified before use: human_rating_indices.json's 30 sample indices were
checked to select identical (image_id, region_label) pairs in both
Mistral's and Llama's Template A region lists across all 30 indices,
confirming the two models' per-template region ordering is identical
(both flatten prompt_curation_inputs.json in the same deterministic
order), so the same index file is safe to reuse across models.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline
import torchvision.models as models
import torchvision.transforms as transforms
import lpips
import clip
from pytorch_fid.inception import InceptionV3

REFUSAL_PHRASES = ["i'm an ai","i cannot","language model","as an ai","i'm unable"]

def is_refusal(text):
    return any(p in text.lower() for p in REFUSAL_PHRASES)

def load_mask(pan_dir, image_id, mask_index):
    mask_path = Path(pan_dir) / f"{str(image_id).zfill(12)}.png"
    if not mask_path.exists():
        return None
    if not hasattr(load_mask, "_lookup"):
        with open(Path(pan_dir) / "segment_lookup.json") as f:
            load_mask._lookup = json.load(f)
    key = f"{image_id}_{mask_index}"
    info = load_mask._lookup.get(key)
    if info is None:
        return None
    seg_id = info["segment_id"]
    pan_img = np.array(Image.open(mask_path).convert("RGB"))
    segment_map = (pan_img[:,:,0].astype(np.int32) +
                   pan_img[:,:,1].astype(np.int32) * 256 +
                   pan_img[:,:,2].astype(np.int32) * 65536)
    mask = (segment_map == seg_id).astype(np.uint8) * 255
    return mask

class VGGFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slice1 = torch.nn.Sequential(*list(vgg.children())[:2])
        self.slice2 = torch.nn.Sequential(*list(vgg.children())[2:7])
        self.slice3 = torch.nn.Sequential(*list(vgg.children())[7:12])
        self.slice4 = torch.nn.Sequential(*list(vgg.children())[12:21])
        for p in self.parameters():
            p.requires_grad = False
    def forward(self, x):
        h1 = self.slice1(x); h2 = self.slice2(h1); h3 = self.slice3(h2); h4 = self.slice4(h3)
        return [h1, h2, h3, h4]

def gram_matrix(feat):
    b, c, h, w = feat.shape
    f = feat.view(b, c, h*w)
    return torch.bmm(f, f.transpose(1,2)) / (h*w)

def gram_distance(feats1, feats2):
    dists = []
    for f1, f2 in zip(feats1, feats2):
        g1, g2 = gram_matrix(f1), gram_matrix(f2)
        dists.append(F.mse_loss(g1, g2).item())
    return float(np.mean(dists))

def preprocess_for_vgg(img_pil, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return transform(img_pil).unsqueeze(0).to(device)

def masked_region(img_arr, mask):
    mask_3ch = np.stack([mask,mask,mask],axis=2)/255.0
    masked = (img_arr * mask_3ch).astype(np.uint8)
    rows = np.any(mask>0, axis=1); cols = np.any(mask>0, axis=0)
    if not rows.any() or not cols.any():
        return None
    rmin,rmax = np.where(rows)[0][[0,-1]]
    cmin,cmax = np.where(cols)[0][[0,-1]]
    return Image.fromarray(masked[rmin:rmax+1, cmin:cmax+1])

def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading InstructPix2Pix...")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    print("Loading VGG19..."); vgg_feat = VGGFeatures().to(device).eval()
    print("Loading LPIPS..."); lpips_fn = lpips.LPIPS(net='vgg').to(device)
    print("Loading CLIP..."); clip_model, clip_preprocess = clip.load("ViT-B/32", device=device); clip_model.eval()
    print("Loading InceptionV3 for FID...")
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()
    print("All models loaded\n")

    sources = {}
    if args.base_json and args.base_json != "SKIP":
        with open(args.base_json) as f:
            base = json.load(f)
        for t in ["A","B","C","D","E","F","G","H","I"]:
            sources[f"baseline_{t}"] = base["per_template"][t]
    if args.lora_a_json and args.lora_a_json != "SKIP":
        with open(args.lora_a_json) as f:
            lora_a = json.load(f)
        for t in ["A","B","C","D","E","F","G","H","I"]:
            sources[f"lora_A_tmpl{t}"] = lora_a["per_template"][t]
    if args.lora_c_json and args.lora_c_json != "SKIP":
        with open(args.lora_c_json) as f:
            lora_c = json.load(f)
        for t in ["A","B","C","D","E","F","G","H","I"]:
            sources[f"lora_C_tmpl{t}"] = lora_c["per_template"][t]
    if args.lora_h_json and args.lora_h_json != "SKIP":
        with open(args.lora_h_json) as f:
            lora_h = json.load(f)
        for t in ["A","B","C","D","E","F","G","H","I"]:
            sources[f"lora_H_tmpl{t}"] = lora_h["per_template"][t]

    with open(args.indices_json) as f:
        idx_data = json.load(f)
    sample_indices = idx_data["indices"][:args.n_sample]

    out_img_dir = Path(args.output).parent / "stylised_images_llama"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    summary = {}
    for src_name, regions in sources.items():
        print(f"\n{'='*60}\nSource: {src_name}\n{'='*60}")
        src_results = []
        gram_dists, lpips_scores, clip_scores, fid_dists, artfid_scores = [], [], [], [], []
        n_skipped = 0
        for idx in sample_indices:
            if idx >= len(regions):
                n_skipped += 1
                continue
            r = regions[idx]
            image_id, mask_index, style = r["image_id"], r["mask_index"], r["style_name"]
            instruction = r.get("instruction","")
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
                    out = pipe(instruction_trunc, image=orig_img, num_inference_steps=20,
                               image_guidance_scale=1.5, guidance_scale=7.5)
                stylised = out.images[0]
            except Exception:
                n_skipped += 1
                continue

            save_path = out_img_dir / f"{src_name}_{image_id}_{mask_index}.jpg"
            stylised.save(save_path)

            mask = load_mask(args.pan_dir, image_id, mask_index)
            if mask is None:
                n_skipped += 1
                continue
            mask_r = np.array(Image.fromarray(mask).resize((512,512), Image.NEAREST))

            orig_arr, sty_arr = np.array(orig_img), np.array(stylised)
            orig_region = masked_region(orig_arr, mask_r)
            sty_region = masked_region(sty_arr, mask_r)
            if orig_region is None or sty_region is None:
                n_skipped += 1
                continue

            style_refs = list(Path(args.style_ref_dir).glob(f"{style}/*.jpg"))
            if not style_refs:
                style_refs = list(Path(args.style_ref_dir).glob(f"{style.replace(' ','-')}/*.jpg"))
            if not style_refs:
                n_skipped += 1
                continue
            style_ref = Image.open(style_refs[0]).convert("RGB")

            with torch.no_grad():
                sty_t, ref_t = preprocess_for_vgg(sty_region, device), preprocess_for_vgg(style_ref, device)
                sty_f, ref_f = vgg_feat(sty_t), vgg_feat(ref_t)
                gram_d = gram_distance(sty_f, ref_f)
            gram_dists.append(gram_d)

            lpips_transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])
            with torch.no_grad():
                orig_t_lp = lpips_transform(orig_region).unsqueeze(0).to(device)
                sty_t_lp = lpips_transform(sty_region).unsqueeze(0).to(device)
                lp_score = float(lpips_fn(orig_t_lp, sty_t_lp).item())
            lpips_scores.append(lp_score)

            with torch.no_grad():
                sty_c, ref_c = clip_preprocess(sty_region).unsqueeze(0).to(device), clip_preprocess(style_ref).unsqueeze(0).to(device)
                sf, rf = clip_model.encode_image(sty_c), clip_model.encode_image(ref_c)
                sf, rf = sf/sf.norm(dim=-1,keepdim=True), rf/rf.norm(dim=-1,keepdim=True)
                clip_s = float((sf*rf).sum())
            clip_scores.append(clip_s)

            fid_transform = transforms.Compose([transforms.Resize((299,299)), transforms.ToTensor()])
            with torch.no_grad():
                sty_fid_t, ref_fid_t = fid_transform(sty_region).unsqueeze(0).to(device), fid_transform(style_ref).unsqueeze(0).to(device)
                sty_feat_fid = inception(sty_fid_t)[0].squeeze().cpu().numpy().flatten()
                ref_feat_fid = inception(ref_fid_t)[0].squeeze().cpu().numpy().flatten()
                fid_dist = float(np.linalg.norm(sty_feat_fid - ref_feat_fid))

            artfid_score = (1 + lp_score) * (1 + fid_dist/1000)
            fid_dists.append(fid_dist); artfid_scores.append(artfid_score)

            n = len(src_results)
            if n % 5 == 0:
                print(f"  [{n+1:02d}] gram={gram_d:.6f} lpips={lp_score:.4f} clip={clip_s:.4f} artfid={artfid_score:.4f}")
            src_results.append({
                "image_id": image_id, "mask_index": mask_index, "style": style,
                "gram_dist": round(gram_d,6), "lpips": round(lp_score,4),
                "clip": round(clip_s,4), "fid_dist": round(fid_dist,2), "artfid": round(artfid_score,4),
            })

        n_scored = len(src_results)
        mg, ml, mc = np.mean(gram_dists) if gram_dists else 0, np.mean(lpips_scores) if lpips_scores else 0, np.mean(clip_scores) if clip_scores else 0
        mf, ma = np.mean(fid_dists) if fid_dists else 0, np.mean(artfid_scores) if artfid_scores else 0
        print(f"\n  Scored: {n_scored}  Skipped: {n_skipped}")
        results[src_name] = src_results
        summary[src_name] = {"gram_dist_mean": round(float(mg),6), "lpips_mean": round(float(ml),4),
            "clip_mean": round(float(mc),4), "fid_dist_mean": round(float(mf),2),
            "artfid_mean": round(float(ma),4), "n_scored": n_scored}

    print(f"\n{'='*70}\nFINAL — Llama ArtFID/Gram/LPIPS comparison\n{'='*70}")
    for src in sorted(summary, key=lambda s: summary[s]["artfid_mean"]):
        s = summary[src]
        print(f"  {src:<22} gram={s['gram_dist_mean']:.6f}  lpips={s['lpips_mean']:.4f}  artfid={s['artfid_mean']:.4f}  n={s['n_scored']}")

    out = {"metrics": ["gram_dist (lower=closer to style)", "lpips (higher=more change)", "clip (higher=more style-aligned)"],
           "summary": summary, "results": results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.output}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_json", default="../attention_maps/llama_ALL9_merged.json")
    p.add_argument("--lora_a_json", default="../attention_maps/llama_lora_A_ALL9_merged.json")
    p.add_argument("--lora_c_json", default="../attention_maps/llama_lora_C_ALL9_merged.json")
    p.add_argument("--lora_h_json", default="SKIP")
    p.add_argument("--indices_json", default="../results/human_rating_indices.json")
    p.add_argument("--img_dir", default="../../data/coconut_subset/images")
    p.add_argument("--pan_dir", default="../data/annotations")
    p.add_argument("--style_ref_dir", default="../../data/style_references")
    p.add_argument("--output", default="../results/artfid_gram_llama.json")
    p.add_argument("--n_sample", type=int, default=30)
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())
