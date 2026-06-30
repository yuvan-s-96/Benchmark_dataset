"""
Step 1b — Visualise attention maps with COCONut panoptic masks
==============================================================
For each region shows:
  1. COCO image with COCONut panoptic mask overlaid
  2. Attention bar chart (BOS excluded, rescaled to content tokens)
  3. Token heatmap

Usage:
    python3 step1b_visualise_attention.py \
        --attention_json ../attention_maps/baseline_mistral.json \
        --ann_json ../../data/coconut_subset/annotations/subset_auto_final_gguf.json \
        --pan_json /mnt/fast1/yvs23/annotations/panoptic_train2017.json \
        --pan_dir  /mnt/fast1/yvs23/annotations/panoptic_train2017 \
        --img_dir  ../../data/coconut_subset/images \
        --output   ../attention_maps/visualisations/ \
        --n 10
"""

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# COCONut panoptic mask overlay
# ─────────────────────────────────────────────────────────────────────────────

def load_panoptic_index(pan_json_path):
    """Build lookup: image_id -> list of segments with category info."""
    with open(pan_json_path) as f:
        pan = json.load(f)
    index = {}
    for ann in pan["annotations"]:
        index[ann["image_id"]] = ann
    return index


def get_mask_overlay(image_id, pan_index, pan_dir, img_dir, region_label,
                     coconut_caption):
    """
    Load COCO image and overlay the best-matching panoptic segment.
    Returns PIL image with overlay, or just the original if no match.
    """
    img_path = Path(img_dir) / f"{image_id}.jpg"
    if not img_path.exists():
        return None

    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    # Load panoptic PNG
    pan_path = Path(pan_dir) / f"{str(image_id).zfill(12)}.png"
    if not pan_path.exists():
        return img

    pan_img = Image.open(pan_path).convert("RGB")
    pan_arr = np.array(pan_img)

    # Decode segment IDs: R + G*256 + B*256^2
    seg_map = (pan_arr[:,:,0].astype(np.int32) +
               pan_arr[:,:,1].astype(np.int32) * 256 +
               pan_arr[:,:,2].astype(np.int32) * 256 * 256)

    ann = pan_index.get(int(image_id))
    if ann is None:
        return img

    # Find best matching segment by label word overlap
    label_words = set(w.lower().strip(".,") for w in region_label.split()
                      if len(w) > 2)

    best_seg_id = None
    best_overlap = 0
    for seg in ann["segments_info"]:
        seg_id = seg["id"]
        # Try matching via category name if available
        cat_name = str(seg.get("category_id", ""))
        overlap = sum(1 for w in label_words if w in cat_name.lower())
        if overlap > best_overlap:
            best_overlap = overlap
            best_seg_id = seg_id

    # Fall back to largest segment if no match
    if best_seg_id is None and ann["segments_info"]:
        sizes = [(np.sum(seg_map == s["id"]), s["id"])
                 for s in ann["segments_info"]]
        best_seg_id = max(sizes)[1]

    if best_seg_id is None:
        return img

    # Create overlay
    mask = (seg_map == best_seg_id)
    overlay = img.copy().convert("RGBA")
    highlight = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(highlight)

    # Fill mask pixels with semi-transparent teal
    mask_img = Image.fromarray((mask * 200).astype(np.uint8), mode="L")
    highlight.paste((29, 158, 117, 140), mask=mask_img)

    # Draw border around mask
    from PIL import ImageFilter
    border = mask_img.filter(ImageFilter.FIND_EDGES)
    highlight.paste((29, 158, 117, 255), mask=border)

    result = Image.alpha_composite(overlay, highlight).convert("RGB")
    return result


def img_to_b64(img, max_w=600):
    """Convert PIL image to base64 string for embedding in HTML."""
    ratio = max_w / img.width
    if ratio < 1:
        img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Attention extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_token_attentions(model, tokenizer, prompt, device):
    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=512
    ).to(device)

    outputs = model(**inputs, output_attentions=True)
    att_stack = torch.stack(outputs.attentions, dim=0)
    att_mean  = att_stack[:, 0, :, :, :].mean(dim=(0, 1))
    last_row  = att_mean[-1].cpu().numpy()
    last_row  = last_row / (last_row.sum() + 1e-8)

    ids    = inputs["input_ids"][0].tolist()
    tokens = [tokenizer.decode([i]) for i in ids]
    return tokens, last_row


def get_label_indices(tokenizer, prompt, region_label):
    all_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    candidates = [
        tokenizer.encode(region_label, add_special_tokens=False),
        tokenizer.encode(" " + region_label, add_special_tokens=False),
    ]
    for label_tokens in candidates:
        if not label_tokens:
            continue
        for i in range(len(all_tokens) - len(label_tokens) + 1):
            if all_tokens[i:i+len(label_tokens)] == label_tokens:
                return list(range(i, i + len(label_tokens)))
    return []


# ─────────────────────────────────────────────────────────────────────────────
# HTML generation
# ─────────────────────────────────────────────────────────────────────────────

def build_html(region_data, tokens, att_weights, label_indices, img_b64):
    label  = region_data["region_label"]
    style  = region_data["style_name"]
    img_id = region_data["image_id"]
    mass   = region_data["label_attention_mass"]
    prompt = region_data["prompt"]

    # Exclude BOS (<s>) from scale for content visualisation
    bos_idx   = 0
    content_w = att_weights.copy()
    content_w[bos_idx] = 0
    max_w = float(np.max(content_w)) if np.max(content_w) > 0 else 1.0

    # Token heatmap
    token_html = ""
    for i, (tok, w) in enumerate(zip(tokens, att_weights)):
        if i == bos_idx:
            bg = "rgb(220,220,220)"
            bold = ""
            border = ""
        else:
            norm = min(content_w[i] / max_w, 1.0)
            r = 255
            g = int(255 * (1 - norm * 0.85))
            b = int(255 * (1 - norm))
            bg = f"rgb({r},{g},{b})"
            bold   = "font-weight:700;" if i in label_indices else ""
            border = "border:2px solid #e53e3e;" if i in label_indices else ""

        tok_d = tok.replace("<","&lt;").replace(">","&gt;").replace(" ","·")
        pct   = f"{w*100:.3f}%"
        token_html += (
            f'<span title="{pct}" style="display:inline-block;'
            f'background:{bg};padding:3px 5px;margin:2px;border-radius:4px;'
            f'font-family:monospace;font-size:13px;{bold}{border}">'
            f'{tok_d}</span>'
        )

    # Bar chart — top 15 content tokens excluding BOS
    content_weights = [(w, t, i) for i,(t,w) in enumerate(zip(tokens, att_weights))
                       if i != bos_idx]
    top15 = sorted(content_weights, reverse=True)[:15]
    max_bar = top15[0][0] if top15 else 1.0

    bar_rows = ""
    for w, tok, idx in top15:
        bar_w  = int(w / max_bar * 300)
        is_lbl = idx in label_indices
        color  = "#1D9E75" if is_lbl else "#718096"
        tok_d  = tok.replace("<","&lt;").replace(">","&gt;")
        marker = " ← region label" if is_lbl else ""
        bar_rows += (
            f"<tr>"
            f"<td style='font-family:monospace;padding:3px 8px;font-size:12px;"
            f"{'font-weight:700;color:#1D9E75' if is_lbl else ''}'>{tok_d}{marker}</td>"
            f"<td style='padding:3px 8px;font-size:12px'>{w*100:.3f}%</td>"
            f"<td style='padding:3px 4px'>"
            f"<div style='width:{bar_w}px;height:14px;background:{color};"
            f"border-radius:3px'></div></td>"
            f"</tr>"
        )

    img_tag = (f"<img src='data:image/jpeg;base64,{img_b64}' "
               f"style='max-width:100%;border-radius:8px;border:1px solid #e2e8f0'>"
               if img_b64 else "<p style='color:#888'>Image not available</p>")

    bos_pct = att_weights[0] * 100

    return f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<title>Attention — {label}</title>
<style>
  body{{font-family:sans-serif;max-width:960px;margin:40px auto;padding:0 20px;background:#f9f9f9}}
  h2{{color:#2d3748}} h3{{color:#4a5568;margin-top:24px}}
  .meta{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:16px;margin-bottom:20px}}
  .meta p{{margin:4px 0;font-size:14px}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}}
  .card{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:16px}}
  .tokens{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:16px;line-height:2.4}}
  table{{border-collapse:collapse;width:100%}}
  .legend{{display:flex;gap:16px;align-items:center;font-size:13px;margin:10px 0}}
  .swatch{{width:20px;height:14px;border-radius:3px;display:inline-block}}
  .note{{font-size:12px;color:#888;margin:6px 0}}
</style>
</head><body>
<h2>Attention heatmap — template A (baseline)</h2>

<div class='meta'>
  <p><b>Image ID:</b> {img_id} &nbsp;|&nbsp;
     <b>Region:</b> {label} &nbsp;|&nbsp;
     <b>Style:</b> {style}</p>
  <p><b>Label attention mass:</b> {mass*100:.3f}%
     <span style='color:#888;font-size:12px'>
     &nbsp;(BOS token `&lt;s&gt;` takes {bos_pct:.1f}% — excluded from colour scale below)
     </span></p>
</div>

<div class='grid'>
  <div class='card'>
    <h3 style='margin-top:0'>Image + COCONut mask</h3>
    <p class='note'>Teal overlay = target region ({label})</p>
    {img_tag}
  </div>
  <div class='card'>
    <h3 style='margin-top:0'>Top attended tokens (excl. BOS)</h3>
    <p class='note'>Green bars = region label tokens</p>
    <table>{bar_rows}</table>
  </div>
</div>

<h3>Token-level attention heatmap</h3>
<p class='note'>BOS token shown in grey. Colour scale applied to content tokens only.
Red border = region label tokens.</p>
<div class='legend'>
  <span class='swatch' style='background:rgb(220,220,220)'></span> BOS (excluded from scale)
  <span class='swatch' style='background:rgb(255,255,255);border:1px solid #ccc'></span> low
  <span class='swatch' style='background:rgb(255,200,100)'></span> medium
  <span class='swatch' style='background:rgb(255,50,0)'></span> high
  &nbsp;|&nbsp;
  <span style='border:2px solid #e53e3e;padding:1px 4px;border-radius:3px;
  font-size:12px'>red border</span> = region label
</div>
<div class='tokens'>{token_html}</div>

<h3>Full prompt</h3>
<pre style='background:#fff;border:1px solid #e2e8f0;border-radius:8px;
padding:16px;font-size:12px;white-space:pre-wrap'>{prompt}</pre>
</body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with open(args.attention_json) as f:
        att_data = json.load(f)
    with open(args.ann_json) as f:
        records = json.load(f)

    print("Loading panoptic index...")
    pan_index = load_panoptic_index(args.pan_json)

    # Build region lookup
    region_lookup = {}
    for rec in records:
        caption = rec.get("coconut_caption", "")
        for reg in rec["regions"]:
            key = (rec["image_id"], reg["mask_index"])
            region_lookup[key] = {**reg, "coconut_caption": caption,
                                  "image_id": rec["image_id"]}

    print(f"Loading {args.model} in 4-bit...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map={"": device},
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.\n")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_region = att_data["per_region"]
    per_region_sorted = sorted(
        per_region, key=lambda x: x["label_attention_mass"] or 0
    )
    n = args.n
    samples = per_region_sorted[:n//2] + per_region_sorted[-(n//2):]

    index_rows = ""
    for i, region_att in enumerate(tqdm(samples, desc="Visualising")):
        img_id  = region_att["image_id"]
        mask_i  = region_att["mask_index"]
        label   = region_att["region_label"]
        mass    = region_att["label_attention_mass"]
        prompt  = region_att["prompt"]
        style   = region_att.get("style_name", "")

        rec = region_lookup.get((img_id, mask_i), {})
        caption = rec.get("coconut_caption", "")

        # Get attention
        tokens, att_weights = get_token_attentions(
            model, tokenizer, prompt, device
        )
        label_indices = get_label_indices(tokenizer, prompt, label)

        # Get image + mask overlay
        img_overlay = get_mask_overlay(
            img_id, pan_index, args.pan_dir,
            args.img_dir, label, caption
        )
        img_b64 = img_to_b64(img_overlay) if img_overlay else None

        region_data = {
            "image_id": img_id,
            "region_label": label,
            "style_name": style,
            "label_attention_mass": mass,
            "prompt": prompt,
        }

        html = build_html(region_data, tokens, att_weights,
                          label_indices, img_b64)

        tag   = "LOW" if i < n//2 else "HIGH"
        fname = f"region_{i:02d}_img{img_id}_mask{mask_i}.html"
        (out_dir / fname).write_text(html)

        index_rows += (
            f"<tr>"
            f"<td class='{tag}'>{tag}</td>"
            f"<td>{label[:45]}</td>"
            f"<td>{style}</td>"
            f"<td>{mass*100:.3f}%</td>"
            f"<td><a href='{fname}'>view</a></td>"
            f"</tr>\n"
        )
        print(f"  [{tag}] {label[:40]:<40} {mass*100:.3f}%")

    summary = att_data["summary"]["label_attention_mass"]
    index_html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>Attention visualisations — Mistral-7B baseline</title>
<style>
  body{{font-family:sans-serif;max-width:900px;margin:40px auto;padding:0 20px}}
  h1{{color:#2d3748}}
  table{{border-collapse:collapse;width:100%}}
  th{{background:#edf2f7;padding:8px;text-align:left}}
  td{{padding:8px;border-bottom:1px solid #e2e8f0}}
  .LOW{{color:#e53e3e;font-weight:700}}
  .HIGH{{color:#38a169;font-weight:700}}
  .stat{{display:inline-block;background:#edf2f7;border-radius:6px;
         padding:8px 16px;margin:4px;font-size:14px}}
</style></head><body>
<h1>Attention visualisations — Mistral-7B baseline (template A)</h1>
<p>
  <span class='stat'>Mean label mass: <b>{summary['mean']*100:.3f}%</b></span>
  <span class='stat'>Median: <b>{summary['median']*100:.3f}%</b></span>
  <span class='stat'>Min: <b>{summary['min']*100:.3f}%</b></span>
  <span class='stat'>Max: <b>{summary['max']*100:.3f}%</b></span>
</p>
<p style='font-size:14px;color:#555'>
  Showing {n//2} lowest and {n//2} highest attention regions.
  Each view shows the COCO image with COCONut mask overlay + token attention chart.
</p>
<table>
  <tr><th>Type</th><th>Region label</th><th>Style</th>
      <th>Label mass</th><th>View</th></tr>
  {index_rows}
</table></body></html>"""

    (out_dir / "index.html").write_text(index_html)
    print(f"\nDone. Copy to laptop:")
    print(f"  scp -r yvs23@ogg.cs.bath.ac.uk:~/Benchmark_dataset/prompt_curation/attention_maps/visualisations/ .")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2", help="HuggingFace model id")
    p.add_argument("--attention_json",
        default="../attention_maps/baseline_mistral.json")
    p.add_argument("--ann_json",
        default="../../data/coconut_subset/annotations/subset_auto_final_gguf.json")
    p.add_argument("--pan_json",
        default="/mnt/fast1/yvs23/annotations/panoptic_train2017.json")
    p.add_argument("--pan_dir",
        default="/mnt/fast1/yvs23/annotations/panoptic_train2017")
    p.add_argument("--img_dir",
        default="../../data/coconut_subset/images")
    p.add_argument("--output",
        default="../attention_maps/visualisations/")
    p.add_argument("--n", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
