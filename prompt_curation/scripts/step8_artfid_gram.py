"""
Step 8 — ArtFID and Gram Matrix style evaluation
=================================================
Computes style quality metrics beyond CLIP:
1. Gram matrix distance — VGG19 texture similarity between
   stylised region and WikiArt style reference
2. LPIPS — perceptual similarity between stylised and original
   (lower = more change = stronger stylisation)

Run on base model all 9 templates + all 3 LoRA variants.
Saves stylised images for inspection.
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
from scipy import linalg

REFUSAL_PHRASES = ["i'm an ai","i cannot","language model","as an ai","i'm unable"]

def is_refusal(text):
    return any(p in text.lower() for p in REFUSAL_PHRASES)

def load_mask(pan_dir, image_id, mask_index):
    mask_path = Path(pan_dir) / f"{str(image_id).zfill(12)}.png"
    if not mask_path.exists():
        return None
    # Load segment lookup
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

# VGG19 feature extractor for Gram matrix
class VGGFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        # Use relu1_1, relu2_1, relu3_1, relu4_1 for style
        self.slice1 = torch.nn.Sequential(*list(vgg.children())[:2])
        self.slice2 = torch.nn.Sequential(*list(vgg.children())[2:7])
        self.slice3 = torch.nn.Sequential(*list(vgg.children())[7:12])
        self.slice4 = torch.nn.Sequential(*list(vgg.children())[12:21])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return [h1, h2, h3, h4]

def gram_matrix(feat):
    b, c, h, w = feat.shape
    f = feat.view(b, c, h*w)
    # Standard Gatys et al. normalisation: divide by h*w only
    gram = torch.bmm(f, f.transpose(1,2)) / (h*w)
    return gram

def gram_distance(feats1, feats2):
    """Mean Gram matrix distance across VGG layers."""
    dists = []
    for f1, f2 in zip(feats1, feats2):
        g1 = gram_matrix(f1)
        g2 = gram_matrix(f2)
        dists.append(F.mse_loss(g1, g2).item())
    return float(np.mean(dists))

def preprocess_for_vgg(img_pil, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    return transform(img_pil).unsqueeze(0).to(device)

def masked_region(img_arr, mask):
    """Extract masked region as PIL image, cropped to bounding box."""
    mask_3ch = np.stack([mask,mask,mask],axis=2)/255.0
    masked = (img_arr * mask_3ch).astype(np.uint8)
    rows = np.any(mask>0, axis=1)
    cols = np.any(mask>0, axis=0)
    if not rows.any() or not cols.any():
        return None
    rmin,rmax = np.where(rows)[0][[0,-1]]
    cmin,cmax = np.where(cols)[0][[0,-1]]
    region = masked[rmin:rmax+1, cmin:cmax+1]
    return Image.fromarray(region)

def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load models
    print("Loading InstructPix2Pix...")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=torch.float16, safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    print("Loading VGG19...")
    vgg_feat = VGGFeatures().to(device).eval()

    print("Loading LPIPS...")
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    print("Loading CLIP...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    print("Loading InceptionV3 for FID...")
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()
    print("All models loaded\n")

    # Load instruction sources
    sources = {}
    with open(args.results_json) as f:
        tmpl = json.load(f)
    # Load baseline only when not in LoRA-only mode
    lora_only = (args.lora_a_json=="SKIP" and args.lora_h_json=="SKIP"
                 and args.lora_c_json!="SKIP")
    if not lora_only:
        for t in ["A","B","C","D","E","F","G","H","I"]:
            sources[f"baseline_{t}"] = tmpl["per_template"][t]

    if args.lora_a_json and args.lora_a_json != 'SKIP':
        with open(args.lora_a_json) as f:
            lora_a = json.load(f)
        for t in ["A","B","C","D","E","F","G","H","I"]:
            sources[f"lora_A_tmpl{t}"] = lora_a["per_template"][t]

    if args.lora_h_json and args.lora_h_json != 'SKIP':
        with open(args.lora_h_json) as f:
            lora_h = json.load(f)
        for t in ["A","B","C","D","E","F","G","H","I"]:
            sources[f"lora_H_tmpl{t}"] = lora_h["per_template"][t]

    if args.lora_c_json and args.lora_c_json != 'SKIP':
        with open(args.lora_c_json) as f:
            lora_c = json.load(f)
        for t in ["A","B","C","D","E","F","G","H","I"]:
            sources[f"lora_C_tmpl{t}"] = lora_c["per_template"][t]

    # Load sample indices
    with open(args.indices_json) as f:
        idx_data = json.load(f)
    sample_indices = idx_data["indices"][:args.n_sample]
    # Use first available source as anchor for indices
    anchor_key = "baseline_A" if "baseline_A" in sources else list(sources.keys())[0]
    anchor = sources[anchor_key]

    # Output dir for stylised images
    out_img_dir = Path(args.output).parent / "stylised_images"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    summary = {}

    for src_name, regions in sources.items():
        print(f"\n{'='*60}")
        print(f"Source: {src_name}")
        print(f"{'='*60}")

        src_results = []
        gram_dists, lpips_scores, clip_scores, fid_dists, artfid_scores = [], [], [], [], []
        n_skipped = 0

        for idx in sample_indices:
            if idx >= len(regions):
                n_skipped += 1
                continue

            r = regions[idx]
            image_id   = r["image_id"]
            mask_index = r["mask_index"]
            style      = r["style_name"]
            instruction = r.get("instruction","")

            if is_refusal(instruction) or len(instruction.split()) < 5:
                n_skipped += 1
                continue

            instruction_trunc = " ".join(instruction.split()[:60])

            # Load image
            img_path = Path(args.img_dir) / f"{image_id}.jpg"
            if not img_path.exists():
                n_skipped += 1
                continue

            orig_img = Image.open(img_path).convert("RGB").resize((512,512))

            # Generate stylised image
            try:
                with torch.autocast("cuda"):
                    out = pipe(instruction_trunc, image=orig_img,
                               num_inference_steps=20,
                               image_guidance_scale=1.5,
                               guidance_scale=7.5)
                stylised = out.images[0]
            except Exception as e:
                n_skipped += 1
                continue

            # Save stylised image
            save_path = out_img_dir / f"{src_name}_{image_id}_{mask_index}.jpg"
            stylised.save(save_path)

            # Load mask
            mask = load_mask(args.pan_dir, image_id, mask_index)
            if mask is None:
                n_skipped += 1
                continue
            mask_r = np.array(Image.fromarray(mask).resize((512,512), Image.NEAREST))

            # Extract regions
            orig_arr = np.array(orig_img)
            sty_arr  = np.array(stylised)
            orig_region = masked_region(orig_arr, mask_r)
            sty_region  = masked_region(sty_arr,  mask_r)
            if orig_region is None or sty_region is None:
                n_skipped += 1
                continue

            # Style reference
            style_refs = list(Path(args.style_ref_dir).glob(f"{style}/*.jpg"))
            if not style_refs:
                style_refs = list(Path(args.style_ref_dir).glob(
                    f"{style.replace(' ','-')}/*.jpg"))
            if not style_refs:
                n_skipped += 1
                continue
            style_ref = Image.open(style_refs[0]).convert("RGB")

            # 1. Gram matrix distance (stylised region vs style reference)
            with torch.no_grad():
                sty_t   = preprocess_for_vgg(sty_region, device)
                ref_t   = preprocess_for_vgg(style_ref, device)
                sty_f   = vgg_feat(sty_t)
                ref_f   = vgg_feat(ref_t)
                gram_d  = gram_distance(sty_f, ref_f)
            gram_dists.append(gram_d)

            # 2. LPIPS (stylised region vs original region — measures change)
            lpips_transform = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3,[0.5]*3)
            ])
            with torch.no_grad():
                orig_t_lp = lpips_transform(orig_region).unsqueeze(0).to(device)
                sty_t_lp  = lpips_transform(sty_region).unsqueeze(0).to(device)
                lp_score  = float(lpips_fn(orig_t_lp, sty_t_lp).item())
            lpips_scores.append(lp_score)

            # 3. CLIP region vs style reference (as before)
            with torch.no_grad():
                sty_c  = clip_preprocess(sty_region).unsqueeze(0).to(device)
                ref_c  = clip_preprocess(style_ref).unsqueeze(0).to(device)
                sf = clip_model.encode_image(sty_c)
                rf = clip_model.encode_image(ref_c)
                sf = sf / sf.norm(dim=-1, keepdim=True)
                rf = rf / rf.norm(dim=-1, keepdim=True)
                clip_s = float((sf*rf).sum())
            clip_scores.append(clip_s)

            # 4. Inception features for FID-style distance (per-image proxy)
            fid_transform = transforms.Compose([
                transforms.Resize((299,299)),
                transforms.ToTensor(),
            ])
            with torch.no_grad():
                sty_fid_t = fid_transform(sty_region).unsqueeze(0).to(device)
                ref_fid_t = fid_transform(style_ref).unsqueeze(0).to(device)
                sty_feat_fid = inception(sty_fid_t)[0].squeeze().cpu().numpy().flatten()
                ref_feat_fid = inception(ref_fid_t)[0].squeeze().cpu().numpy().flatten()
                fid_dist = float(np.linalg.norm(sty_feat_fid - ref_feat_fid))

            # ArtFID-style combined score: (1+LPIPS) x (1+FID_dist)
            artfid_score = (1 + lp_score) * (1 + fid_dist/1000)

            fid_dists.append(fid_dist)
            artfid_scores.append(artfid_score)

            n = len(src_results)
            if n % 5 == 0:
                print(f"  [{n+1:02d}] gram={gram_d:.6f} lpips={lp_score:.4f} "
                      f"clip={clip_s:.4f} artfid={artfid_score:.4f}")

            src_results.append({
                "image_id":    image_id,
                "mask_index":  mask_index,
                "style":       style,
                "gram_dist":   round(gram_d, 6),
                "lpips":       round(lp_score, 4),
                "clip":        round(clip_s, 4),
                "fid_dist":    round(fid_dist, 2),
                "artfid":      round(artfid_score, 4),
            })

        n_scored = len(src_results)
        mg = np.mean(gram_dists) if gram_dists else 0
        ml = np.mean(lpips_scores) if lpips_scores else 0
        mc = np.mean(clip_scores) if clip_scores else 0
        mf = np.mean(fid_dists) if fid_dists else 0
        ma = np.mean(artfid_scores) if artfid_scores else 0
        print(f"\n  Scored: {n_scored}  Skipped: {n_skipped}")
        print(f"  Gram dist (lower=closer to style):  {mg:.6f}")
        print(f"  LPIPS (higher=more changed):        {ml:.4f}")
        print(f"  CLIP (higher=more style-aligned):   {mc:.4f}")
        print(f"  FID dist (lower=closer to style):   {mf:.2f}")
        print(f"  ArtFID (lower=better style quality): {ma:.4f}")

        results[src_name] = src_results
        summary[src_name] = {
            "gram_dist_mean": round(float(mg), 6),
            "lpips_mean":     round(float(ml), 4),
            "clip_mean":      round(float(mc), 4),
            "fid_dist_mean":  round(float(mf), 2),
            "artfid_mean":    round(float(ma), 4),
            "n_scored":       n_scored,
        }

    # Final table
    print(f"\n{'='*70}")
    print("FINAL — ArtFID/Gram/LPIPS comparison")
    print(f"{'='*70}")
    print(f"\n  {'Source':<22} {'Gram↓':>12} {'LPIPS↑':>8} {'CLIP↑':>8} {'ArtFID↓':>9} {'n':>5}")
    print(f"  {'-'*70}")
    for src in sorted(summary, key=lambda s: summary[s]["artfid_mean"]):
        s = summary[src]
        print(f"  {src:<22} {s['gram_dist_mean']:>12.6f} "
              f"{s['lpips_mean']:>8.4f} {s['clip_mean']:>8.4f} "
              f"{s['artfid_mean']:>9.4f} {s['n_scored']:>5}")

    out = {
        "metrics": ["gram_dist (lower=closer to style reference)",
                    "lpips (higher=more change from original)",
                    "clip (higher=more style-aligned)"],
        "summary": summary,
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.output}")
    print(f"Stylised images saved to: {out_img_dir}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_json",
        default="../results/template_comparison_979.json")
    p.add_argument("--lora_a_json",
        default="../results/attention_lora_A.json")
    p.add_argument("--lora_h_json",
        default="../results/attention_lora_H.json")
    p.add_argument("--lora_c_json",
        default="../results/attention_lora_C.json")
    p.add_argument("--indices_json",
        default="../results/human_rating_indices.json")
    p.add_argument("--img_dir",
        default="../../data/coconut_subset/images")
    p.add_argument("--pan_dir",
        default="/mnt/fast1/yvs23/coconut_panoptic")
    p.add_argument("--style_ref_dir",
        default="../../data/style_references")
    p.add_argument("--output",
        default="../results/artfid_gram_eval.json")
    p.add_argument("--n_sample", type=int, default=30)
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())
