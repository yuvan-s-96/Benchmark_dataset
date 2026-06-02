# Benchmark Dataset Curation — Complete Guide
## Multi-Region Stylisation | MSc Data Science, University of Bath, 2026
**Server:** `yvs23@ogg.cs.bath.ac.uk`

---

## Overview

This pipeline builds a benchmark for instruction-driven regional style transfer:
content images segmented into 2–N semantic regions, each paired with a WikiArt
style image and a natural-language instruction.

**Target:** ~500 samples · dynamic region count · 70/10/20 train/val/test split

---

## What is in this repo

```
benchmark_dataset/
├── scripts/
│   ├── 00_interactive_click.py   ← Gradio: click images to define regions
│   ├── 01_segment_regions.py     ← SAM segmentation (dynamic region count)
│   ├── 02_pair_styles.py         ← WikiArt pairing + instruction generation
│   ├── 03_quality_control.py     ← Filtering, corner-case tagging, splits
│   ├── 04_annotate.py            ← Gradio: label and verify samples
│   ├── 05_export.py              ← Package final benchmark into a zip
│   └── download_wikiart.py       ← Download WikiArt via HuggingFace datasets
├── configs/
│   └── benchmark_schema.json     ← JSON schema + example record
├── .gitignore
└── README_HEX.md                 ← this file
```

`data/` is created at runtime and is NOT tracked by git.

---

## Key design decisions

| Decision | Detail |
|---|---|
| Region count | Fully dynamic — one region per click, or auto-detected via grid prompts |
| SAM approach | Grid of 16 point prompts (4×4) with IoU deduplication — no SamAutomaticMaskGenerator needed |
| WikiArt download | Uses `huggan/wikiart` on HuggingFace datasets — no API key, no rate limits |
| GPU model | `sam-vit-large` — fits on ogg RTX 2080 (8 GB) comfortably |
| Local dev | Use `sam-vit-base` on RTX 3050 (4 GB) for testing |
| Crash safety | Step 1 writes stub JSON after every image — safe to interrupt |
| Resume | Step 1 skips already-processed images on rerun |
| Annotation resume | Step 4 starts from first unannotated sample automatically |
| Export | Step 5 zips only what is needed (masks + styles + JSON) |

---

## One-time setup on ogg

### 1. Connect (VPN required off-campus)

```bash
ssh yvs23@ogg.cs.bath.ac.uk
```

### 2. Check GPU availability

```bash
nvidia-smi
```

GPU 1 and GPU 3 are usually the quietest on ogg (RTX 2080, 8 GB each).

```bash
export CUDA_VISIBLE_DEVICES=1
```

### 3. Clone repo

```bash
cd ~
git clone https://github.com/yuvan-s-96/Benchmark_dataset.git
cd Benchmark_dataset
```

> GitHub no longer accepts passwords. Use a Personal Access Token as the password.
> Create one at: https://github.com/settings/tokens → Generate new token (classic) → tick `repo`

### 4. Set git identity (first time only)

```bash
git config --global user.email "your-email@bath.ac.uk"
git config --global user.name "Yuvan Velkumar"
```

### 5. Create virtual environment

```bash
python3 -m venv ~/benchmark_env
source ~/benchmark_env/bin/activate
```

### 6. Install dependencies

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers pillow tqdm numpy gradio requests openai scikit-image datasets
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 7. Pre-download SAM model (once, on login node)

```bash
python3 -c "
from transformers import SamModel, SamProcessor
print('Downloading sam-vit-large...')
SamModel.from_pretrained('facebook/sam-vit-large')
SamProcessor.from_pretrained('facebook/sam-vit-large')
print('Done.')
"
```

---

## Step-by-step pipeline

---

### Step 0 — Collect COCO content images

```bash
mkdir -p ~/Benchmark_dataset/data/content_images
cd ~/Benchmark_dataset/data/content_images
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
mv val2017/* .
rm -rf val2017 val2017.zip
cd ~/Benchmark_dataset
```

Gives 5000 images (~1 GB).

---

### Step 0b — Download WikiArt style references

Uses HuggingFace `huggan/wikiart` dataset — no API key needed.

```bash
cd ~/Benchmark_dataset/scripts
python3 download_wikiart.py \
    --output_dir ../data/style_references \
    --max_per_style 50
```

Output: 20 styles × 50 images = 1000 JPEGs.
Runtime: ~5 min.

---

### Step 0c — Collect click points (optional, improves quality)

Click on images in your browser to mark region seeds.
Each click = one SAM region. No limit per image.

```bash
# On ogg
source ~/benchmark_env/bin/activate
cd ~/Benchmark_dataset/scripts
python3 00_interactive_click.py --port 7861
```

On your laptop (new terminal):
```bash
ssh -N -L 7861:localhost:7861 yvs23@ogg.cs.bath.ac.uk
```

Open **http://localhost:7861**

- Click on the image to add region markers
- Click **Save & Next ▶** when done
- Use **⏩ Skip to next unannotated** to resume after a break
- Saves to `data/annotations/clicks.json` after every navigation

---

### Step 1 — Segment regions with SAM

Run inside tmux so it survives disconnections.

```bash
tmux new -s benchmark
source ~/benchmark_env/bin/activate
export CUDA_VISIBLE_DEVICES=1
cd ~/Benchmark_dataset/scripts

python3 01_segment_regions.py \
    --image_dir   ../data/content_images \
    --output_dir  ../data/masks \
    --sam_model_id facebook/sam-vit-large \
    --mode mixed \
    --min_area 0.02 \
    --max_masks 6
```

Detach tmux: **Ctrl+B then D**
Reattach: `tmux attach -t benchmark`

**How it segments:**
- Uses a 4×4 grid of point prompts across each image
- Deduplicates masks with IoU > 0.5
- Click mode: one mask per click point (uses clicks.json if available)
- Auto mode: grid prompts for all images
- Mixed (default): clicks if available, grid otherwise

**Progress check (from outside tmux):**
```bash
python3 -c "
import json
with open('/mnt/vurm/homes/homes/yvs23/Benchmark_dataset/data/annotations/masks_stub.json') as f:
    d = json.load(f)
print(f'Processed: {len(d)}/5000')
print(f'Regions: min={min(r[\"num_regions\"] for r in d)}  max={max(r[\"num_regions\"] for r in d)}  mean={sum(r[\"num_regions\"] for r in d)/len(d):.1f}')
"
```

Expected runtime: ~7–8 s/image → ~10–11 h for 5000 images on RTX 2080.
Output: `data/masks/<image_id>/mask_NN.png` + `data/annotations/masks_stub.json`

---

### Step 1b — Visual preview of masks (recommended before full run)

Run on a test batch first to verify regions look sensible:

```bash
# Create 20-image test set
mkdir -p ~/Benchmark_dataset/data/test_images
ls ~/Benchmark_dataset/data/content_images/*.jpg | head -20 | xargs -I{} cp {} ~/Benchmark_dataset/data/test_images/

# Run SAM on test set
python3 01_segment_regions.py \
    --image_dir  ../data/test_images \
    --output_dir ../data/masks \
    --mode auto \
    --min_area 0.02 \
    --max_masks 6 \
    --no_resume
```

Generate colour-coded overlays:

```bash
python3 - << 'EOF'
import json, numpy as np, os
from pathlib import Path
from PIL import Image, ImageDraw

with open("../data/annotations/masks_stub.json") as f:
    records = json.load(f)

COLOURS = [
    (255, 80,  80,  140), (80,  180, 255, 140), (80,  255, 130, 140),
    (255, 200, 50,  140), (200, 80,  255, 140), (255, 140, 0,   140),
]

out_dir = Path("../data/mask_previews")
out_dir.mkdir(parents=True, exist_ok=True)

for record in records[:5]:
    candidates = list(Path("../data/test_images").glob(f"{record['image_id']}.*"))
    if not candidates:
        continue
    base    = Image.open(candidates[0]).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    for i, region in enumerate(record["regions"]):
        mask_path = Path("../data") / region["mask_file"]
        if not mask_path.exists():
            continue
        mask   = np.array(Image.open(mask_path).convert("L")) > 127
        colour = COLOURS[i % len(COLOURS)]
        layer  = np.zeros((*base.size[::-1], 4), dtype=np.uint8)
        layer[mask, :3] = colour[:3]
        layer[mask, 3]  = colour[3]
        overlay = Image.alpha_composite(overlay, Image.fromarray(layer))
        ys, xs = np.where(mask)
        if len(xs):
            draw.text((int(xs.mean())-8, int(ys.mean())-8),
                      str(i+1), fill=(255, 255, 255, 255))
    result = Image.alpha_composite(base, overlay).convert("RGB")
    out_path = out_dir / f"{record['image_id']}_preview.jpg"
    result.save(out_path, quality=90)
    print(f"Saved: {out_path}  ({record['num_regions']} regions)")
EOF
```

Copy previews to your laptop (new terminal on laptop):
```bash
scp "yvs23@ogg.cs.bath.ac.uk:~/Benchmark_dataset/data/mask_previews/*.jpg" "C:\Users\Yuvan Velkumar\Downloads\"
```

---

### Step 2 — Pair style references and generate instructions

CPU only, runs quickly.

```bash
python3 02_pair_styles.py \
    --masks_stub   ../data/annotations/masks_stub.json \
    --style_dir    ../data/style_references \
    --output_json  ../data/annotations/benchmark_draft.json \
    --instruction_model stub
```

To use GPT-4o for richer instructions (optional):
```bash
export OPENAI_API_KEY="sk-..."
python3 02_pair_styles.py \
    --masks_stub   ../data/annotations/masks_stub.json \
    --style_dir    ../data/style_references \
    --output_json  ../data/annotations/benchmark_draft.json \
    --instruction_model gpt4v
```

Output: `data/annotations/benchmark_draft.json`

---

### Step 3 — Quality control and split

```bash
python3 03_quality_control.py \
    --draft_json  ../data/annotations/benchmark_draft.json \
    --output_json ../data/annotations/benchmark_final.json \
    --min_regions 2
```

Tags corner cases:

| Tag | Meaning | Target |
|---|---|---|
| `similar_entities` | ≥2 regions same semantic class | ~30% |
| `encompassed` | one region ≥85% inside another | ~20% |
| `background_heavy` | largest region > 50% of image | ~25% |

Stratified 70/10/20 train/val/test split.

Output: `data/annotations/benchmark_final.json` + `dropped_records.json`

---

### Step 4 — Human annotation

```bash
python3 04_annotate.py \
    --benchmark_json ../data/annotations/benchmark_final.json \
    --output_json    ../data/annotations/benchmark_annotated.json \
    --port 7860
```

On your laptop (new terminal):
```bash
ssh -N -L 7860:localhost:7860 yvs23@ogg.cs.bath.ac.uk
```

Open **http://localhost:7860**

For each sample:
1. Review colour-coded region overlay
2. Fill in region labels: `Region 1: sky`, `Region 2: left giraffe`
3. Edit instructions to be fluent and specific
4. Set status: `approved` or `rejected`
5. Click **Next ▶** — saves automatically

Annotation priorities:
- All 100 **test** samples: full labels + instructions + accept/reject
- Train/val: verify labels at minimum

Output: `data/annotations/benchmark_annotated.json`

---

### Step 5 — Export final benchmark

```bash
python3 05_export.py \
    --annotated_json ../data/annotations/benchmark_annotated.json \
    --output_zip     ../benchmark_final_export.zip
```

Copy zip to laptop:
```bash
# On your laptop
scp "yvs23@ogg.cs.bath.ac.uk:~/Benchmark_dataset/benchmark_final_export.zip" "C:\Users\Yuvan Velkumar\Downloads\"

# Or with rsync (resumable)
rsync -avz --progress yvs23@ogg.cs.bath.ac.uk:~/Benchmark_dataset/benchmark_final_export.zip .
```

---

## Saving results to GitHub

`.gitignore` already excludes large binary data. Only push code and JSONs:

```bash
cd ~/Benchmark_dataset
git add scripts/
git add data/annotations/benchmark_annotated.json
git add data/annotations/benchmark_final.json
git commit -m "updated pipeline + final benchmark"
git push
```

Use your Personal Access Token as the password when prompted.

---

## Checking results at any point

```bash
python3 - << 'EOF'
import json
with open("data/annotations/benchmark_annotated.json") as f:
    data = json.load(f)
splits   = {}
tags     = {}
counts   = []
for r in data:
    splits[r.get("split","?")] = splits.get(r.get("split","?"),0) + 1
    counts.append(r.get("num_regions", 0))
    for t in r.get("corner_case_tags", []):
        tags[t] = tags.get(t, 0) + 1
approved = sum(1 for r in data if r.get("annotation_status") == "approved")
print(f"Total    : {len(data)}")
print(f"Splits   : {splits}")
print(f"Approved : {approved}")
print(f"Tags     : {tags}")
print(f"Regions  : min={min(counts)} max={max(counts)} mean={sum(counts)/len(counts):.1f}")
EOF
```

---

## GPU selection on ogg

```bash
nvidia-smi          # see all 6 GPUs and usage
export CUDA_VISIBLE_DEVICES=1   # pick the quietest one
```

6× RTX 2080, 8 GB VRAM each. `sam-vit-large` uses ~2.5 GB, leaving 5.5 GB headroom.

---

## Local machine (RTX 3050, 4 GB VRAM)

Use locally for: click collection, WikiArt download, annotation UI, steps 2–5.
Use ogg for: Step 1 SAM segmentation only.

```bash
# Local testing with smaller model
python3 01_segment_regions.py \
    --sam_model_id facebook/sam-vit-base \
    --mode auto \
    --image_dir ../data/test_images
```

---

## Troubleshooting

**CUDA out of memory:**
Switch GPU: re-check `nvidia-smi`, pick one with < 1 GB used.
Or use smaller model: `sam-vit-base`.

**Gradio not loading:**
Make sure SSH tunnel terminal is still open.
Port must match: `ssh -N -L 7860:localhost:7860 ...`

**Step 1 stopped halfway:**
Just rerun — `--resume` is on by default, already-done images are skipped.

**tmux session gone:**
```bash
tmux ls
tmux attach -t benchmark
```

**GitHub push rejected:**
Use Personal Access Token not password.
Create at: https://github.com/settings/tokens

**WikiArt download gets 0 images:**
Do NOT use `download_wikiart.py` with the old WikiArt API — it is broken.
The current script uses HuggingFace `huggan/wikiart` which works reliably.

**SAM gives only 1 region per image:**
This was a bug in the original code (wrong post_process_masks usage).
The current `01_segment_regions.py` uses grid point prompts and is fixed.

---

## Timeline

| Script | Project week |
|---|---|
| `download_wikiart.py` + `00_interactive_click.py` | Wk 3 |
| `01_segment_regions.py` | Wk 3 |
| `02_pair_styles.py` | Wk 7 |
| `03_quality_control.py` | Wk 7–8 |
| `04_annotate.py` | Wk 7–8 |
| `05_export.py` + push to GitHub | Wk 12 |

---

## Target statistics

| Property | Target |
|---|---|
| Total samples | ~500 |
| Regions per sample | dynamic, mean ~4.7 (from test run) |
| Distinct WikiArt styles | 20 |
| `similar_entities` | ~30% |
| `encompassed` | ~20% |
| Human-annotated test split | 100% |
