# Benchmark Dataset Curation — Complete Guide
## Multi-Region Stylisation | MSc Data Science, University of Bath, 2026
**Server:** `yvs23@ogg.cs.bath.ac.uk`

---

## Overview

This pipeline builds a benchmark for instruction-driven regional style transfer:
content images segmented into 2–N semantic regions, each paired with a WikiArt
style image and a natural-language instruction.

**Target:** ~500 samples · dynamic region count · 70/10/20 train/val/test split

**Segmentation model: SAM2** (facebook/sam2-hiera-large) — better boundary quality
than SAM1, confirmed by side-by-side comparison on 20 test images.

---

## ⚠️ Important: Save to GitHub frequently

**ogg does not backup your data. If files are deleted they are gone.**

After every step run:
```bash
cd ~/Benchmark_dataset
git add data/annotations/
git commit -m "step X complete"
git push
```

The JSONs in `data/annotations/` are your progress checkpoints.
Large binary files (masks, images, checkpoints) are excluded from git.

---

## What is in this repo

```
benchmark_dataset/
├── scripts/
│   ├── 00_interactive_click.py        ← Gradio: click images to define regions
│   ├── 01_segment_regions.py          ← SAM1 segmentation (kept for reference)
│   ├── 01_segment_regions_sam2.py     ← SAM2 segmentation ✅ USE THIS
│   ├── 02_pair_styles.py              ← WikiArt pairing + instruction generation
│   ├── 03_quality_control.py          ← Filtering, corner-case tagging, splits
│   ├── 04_annotate.py                 ← Gradio: label and verify samples
│   ├── 05_export.py                   ← Package final benchmark into a zip
│   └── download_wikiart.py            ← Download WikiArt via HuggingFace datasets
├── checkpoints/
│   └── sam2.1_hiera_large.pt          ← SAM2 weights (not in git, redownload if lost)
├── configs/
│   └── benchmark_schema.json
├── .gitignore
└── README_HEX.md
```

`data/` is created at runtime and is NOT tracked by git (except `data/annotations/`).

---

## SAM1 vs SAM2 — why we use SAM2

| Property | SAM1 (vit-large) | SAM2 (hiera-large) |
|---|---|---|
| Boundary quality | Good | Better — tighter edges |
| Small region detail | Misses fine details | Handles small regions well |
| Speed | ~7 s/image | ~5 s/image |
| VRAM | ~2.5 GB | ~3.0 GB |
| Mask format | HuggingFace transformers | Official Meta SAM2 repo |

SAM2 tested and confirmed working on ogg RTX 2080 (8 GB).

---

## Directory structure at runtime

```
data/
├── content_images/           ← COCO 2017 val (5000 images) — redownload if lost
├── content_images_filtered/  ← filtered COCO subset (500 images)
├── style_references/         ← WikiArt via HuggingFace (1000 images)
├── test_images/              ← 20-image test subset
├── masks_sam2/               ← SAM2 auto mask PNGs
├── masks_sam2_click/         ← SAM2 click-mode mask PNGs
├── annotations/              ← ✅ TRACKED IN GIT
│   ├── clicks.json               ← click coords (main run)
│   ├── clicks_sam2_test.json     ← click coords (SAM2 test)
│   ├── masks_stub.json           ← SAM1 results (reference)
│   ├── masks_stub_sam2.json      ← SAM2 results ✅ USE THIS
│   ├── benchmark_draft.json
│   ├── benchmark_final.json
│   └── benchmark_annotated.json
└── previews_sam2/            ← visual previews — regenerate anytime
```

---

## One-time setup on ogg

### 1. Connect (VPN required off-campus)

```bash
ssh yvs23@ogg.cs.bath.ac.uk
```

### 2. Check GPU availability

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=1   # pick quietest GPU
```

### 3. Clone repo

```bash
cd ~
git clone https://github.com/yuvan-s-96/Benchmark_dataset.git
cd Benchmark_dataset
```

> Use Personal Access Token as password: https://github.com/settings/tokens → Generate new token (classic) → tick `repo`

### 4. Set git identity (once only)

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
pip install transformers==4.47.1 pillow tqdm numpy gradio requests openai scikit-image datasets
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/facebookresearch/sam2.git
```

### 7. Download SAM2 weights (once only)

```bash
mkdir -p ~/Benchmark_dataset/checkpoints
cd ~/Benchmark_dataset/checkpoints
wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
echo "Downloaded: $(ls -lh sam2.1_hiera_large.pt)"
cd ~/Benchmark_dataset
```

---

## Recovering lost data

```bash
# 1. Content images
mkdir -p ~/Benchmark_dataset/data/content_images
cd ~/Benchmark_dataset/data/content_images
wget -q --show-progress http://images.cocodataset.org/zips/val2017.zip
unzip -q val2017.zip && mv val2017/* . && rm -rf val2017 val2017.zip
cd ~/Benchmark_dataset

# 2. Style references
cd scripts && python3 download_wikiart.py --output_dir ../data/style_references --max_per_style 50

# 3. Test images
mkdir -p ../data/test_images
ls ../data/content_images/*.jpg | head -20 | xargs -I{} cp {} ../data/test_images/

# 4. SAM2 weights
mkdir -p ~/Benchmark_dataset/checkpoints && cd ~/Benchmark_dataset/checkpoints
wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# 5. Annotations — already in GitHub
cd ~/Benchmark_dataset && git pull
```

---

## Step-by-step pipeline

---

### Step 0 — Collect COCO content images

```bash
mkdir -p ~/Benchmark_dataset/data/content_images
cd ~/Benchmark_dataset/data/content_images
wget -q --show-progress http://images.cocodataset.org/zips/val2017.zip
unzip -q val2017.zip && mv val2017/* . && rm -rf val2017 val2017.zip
echo "Images: $(ls | wc -l)"
cd ~/Benchmark_dataset
```

---

### Step 0b — Filter COCO to cleaner images

Filters to 500 images with 2–5 large distinct objects.
Avoids edge cases like small eyes, fine details SAM struggles with.

```bash
cd ~/Benchmark_dataset/data/content_images
wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q annotations_trainval2017.zip
cd ~/Benchmark_dataset/scripts

python3 - << 'EOF'
import json, shutil
from pathlib import Path

with open("../data/content_images/annotations/instances_val2017.json") as f:
    coco = json.load(f)

img_anns = {}
for ann in coco["annotations"]:
    img_anns.setdefault(ann["image_id"], []).append(ann)

good = []
for img in coco["images"]:
    W, H  = img["width"], img["height"]
    cats  = {a["category_id"] for a in img_anns.get(img["id"], [])
             if a["area"] / (W * H) > 0.03}
    if 2 <= len(cats) <= 5:
        good.append(img["file_name"])

out = Path("../data/content_images_filtered")
out.mkdir(exist_ok=True)
for fname in good[:500]:
    src = Path("../data/content_images") / fname
    if src.exists():
        shutil.copy(src, out / fname)

print(f"Filtered: {len(list(out.glob('*.jpg')))} images")
EOF
```

**Save to GitHub after:**
```bash
cd ~/Benchmark_dataset
git add data/annotations/
git commit -m "step 0 complete - COCO downloaded and filtered"
git push
```

---

### Step 0c — Download WikiArt style references

```bash
cd ~/Benchmark_dataset/scripts
python3 download_wikiart.py \
    --output_dir ../data/style_references \
    --max_per_style 50
```

---

### Step 0d — Collect click points (for test split)

```bash
python3 00_interactive_click.py \
    --image_dir   ../data/content_images_filtered \
    --output_json ../data/annotations/clicks.json \
    --port 7861
```

On your laptop (new terminal):
```bash
ssh -N -L 7861:localhost:7861 yvs23@ogg.cs.bath.ac.uk
```

Open **http://localhost:7861** — click 2–4 regions per image only.

**Save to GitHub after:**
```bash
cd ~/Benchmark_dataset
git add data/annotations/clicks.json
git commit -m "clicks collected"
git push
```

---

### Step 1 — Segment regions with SAM2 ✅

Run inside tmux:

```bash
tmux new -s benchmark
source ~/benchmark_env/bin/activate
export CUDA_VISIBLE_DEVICES=1
cd ~/Benchmark_dataset/scripts

python3 01_segment_regions_sam2.py \
    --image_dir   ../data/content_images_filtered \
    --output_dir  ../data/masks_sam2 \
    --checkpoint  ../checkpoints/sam2.1_hiera_large.pt \
    --config      configs/sam2.1/sam2.1_hiera_l.yaml \
    --mode mixed \
    --min_area 0.02 \
    --max_masks 6
```

Detach: **Ctrl+B then D** | Reattach: `tmux attach -t benchmark`

**Progress check:**
```bash
python3 -c "
import json
with open('$HOME/Benchmark_dataset/data/annotations/masks_stub_sam2.json') as f:
    d = json.load(f)
print(f'Processed: {len(d)} images')
print(f'Regions: min={min(r[\"num_regions\"] for r in d)}  max={max(r[\"num_regions\"] for r in d)}  mean={sum(r[\"num_regions\"] for r in d)/len(d):.1f}')
"
```

**Save to GitHub after:**
```bash
cd ~/Benchmark_dataset
git add data/annotations/masks_stub_sam2.json
git commit -m "step 1 complete - SAM2 segmentation done"
git push
```

---

### Step 1b — Visual preview + pure mask saving

```bash
cd ~/Benchmark_dataset/scripts

python3 - << 'EOF'
import json, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

with open("../data/annotations/masks_stub_sam2.json") as f:
    records = json.load(f)

COLOURS = [
    (255, 80,  80,  140), (80,  180, 255, 140), (80,  255, 130, 140),
    (255, 200, 50,  140), (200, 80,  255, 140), (255, 140, 0,   140),
    (0,   210, 210, 140), (255, 100, 160, 140),
]

overlay_dir = Path("../data/previews_sam2")
pure_dir    = Path("../data/pure_masks_sam2")
overlay_dir.mkdir(parents=True, exist_ok=True)
pure_dir.mkdir(parents=True, exist_ok=True)

for record in records[:5]:   # change to [:] for all images
    candidates = list(Path("../data/content_images_filtered").glob(f"{record['image_id']}.*"))
    if not candidates:
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
        # Pure binary mask
        pure = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        pure.save(pure_dir / f"{record['image_id']}_mask_{i+1:02d}.png")

    result = Image.alpha_composite(base, overlay).convert("RGB")
    result.save(overlay_dir / f"{record['image_id']}_overlay.jpg", quality=90)
    print(f"Saved: {record['image_id']}  ({record['num_regions']} regions)")
EOF
```

Copy to laptop (run on laptop):
```bash
scp "yvs23@ogg.cs.bath.ac.uk:~/Benchmark_dataset/data/previews_sam2/*.jpg" "C:/Users/Yuvan Velkumar/Downloads/"
scp "yvs23@ogg.cs.bath.ac.uk:~/Benchmark_dataset/data/pure_masks_sam2/*.png" "C:/Users/Yuvan Velkumar/Downloads/"
```

---

### Step 2 — Pair style references and generate instructions

```bash
cd ~/Benchmark_dataset/scripts

python3 02_pair_styles.py \
    --masks_stub   ../data/annotations/masks_stub_sam2.json \
    --style_dir    ../data/style_references \
    --output_json  ../data/annotations/benchmark_draft.json \
    --instruction_model stub
```

**Save to GitHub after:**
```bash
cd ~/Benchmark_dataset
git add data/annotations/benchmark_draft.json
git commit -m "step 2 complete - style pairing done"
git push
```

---

### Step 3 — Quality control and split

```bash
python3 03_quality_control.py \
    --draft_json  ../data/annotations/benchmark_draft.json \
    --output_json ../data/annotations/benchmark_final.json \
    --min_regions 2
```

**Save to GitHub after:**
```bash
cd ~/Benchmark_dataset
git add data/annotations/benchmark_final.json
git add data/annotations/dropped_records.json
git commit -m "step 3 complete - QC and split done"
git push
```

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

- Fill in region labels: `Region 1: sky`, `Region 2: person`
- Edit instructions to be fluent and specific
- Set status: `approved` or `rejected`
- Click **Next ▶** — saves automatically

**Save to GitHub frequently:**
```bash
cd ~/Benchmark_dataset
git add data/annotations/benchmark_annotated.json
git commit -m "annotation progress - N samples done"
git push
```

---

### Step 5 — Export final benchmark

```bash
python3 05_export.py \
    --annotated_json ../data/annotations/benchmark_annotated.json \
    --output_zip     ../benchmark_final_export.zip
```

Copy to laptop:
```bash
scp "yvs23@ogg.cs.bath.ac.uk:~/Benchmark_dataset/benchmark_final_export.zip" "C:/Users/Yuvan Velkumar/Downloads/"
```

**Final GitHub push:**
```bash
cd ~/Benchmark_dataset
git add data/annotations/
git commit -m "final benchmark complete"
git push
```

---

## Cleaning up disk space

Check usage:
```bash
du -sh ~/Benchmark_dataset/data/*/
quota -s
```

Safe to delete:
```bash
rm -rf ~/Benchmark_dataset/data/content_images      # ~779 MB, redownloadable
rm -rf ~/Benchmark_dataset/data/masks_sam2           # regeneratable with Step 1
rm -rf ~/Benchmark_dataset/data/masks_sam2_click     # regeneratable
rm -rf ~/Benchmark_dataset/data/previews_sam2        # regeneratable
rm -rf ~/Benchmark_dataset/data/pure_masks_sam2      # regeneratable
rm -rf ~/Benchmark_dataset/data/test_images          # regeneratable
rm -f  ~/Benchmark_dataset/benchmark_final_export.zip
rm -rf ~/.cache/huggingface/hub/datasets--huggan--wikiart/   # already downloaded
rm -rf ~/data/                                        # duplicate style refs
```

Never delete:
```bash
~/Benchmark_dataset/data/annotations/    # push to GitHub first
~/Benchmark_dataset/data/style_references/
~/Benchmark_dataset/checkpoints/         # SAM2 weights, slow to redownload
~/benchmark_env/                         # takes 10 min to recreate
```

**Always push before clearing:**
```bash
cd ~/Benchmark_dataset
git add data/annotations/
git commit -m "backup before cleanup"
git push
```

---

## GPU selection on ogg

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=1   # pick GPU with least memory used
```

6× RTX 2080, 8 GB each. SAM2 uses ~3 GB VRAM.

---

## Local machine (RTX 3050, 4 GB VRAM)

SAM2 needs ~3 GB — tight on 4 GB. Use smaller config:

```bash
# Download smaller SAM2 weights locally
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

python3 01_segment_regions_sam2.py \
    --checkpoint ../checkpoints/sam2.1_hiera_small.pt \
    --config     configs/sam2.1/sam2.1_hiera_s.yaml \
    --image_dir  ../data/test_images \
    --mode auto
```

Use ogg for the full 500-image run.

---

## Troubleshooting

**Files disappeared on ogg:**
ogg does not backup. Always push annotations to GitHub after each step.
See Recovery section above.

**CUDA out of memory:**
Pick less busy GPU: `nvidia-smi` then `export CUDA_VISIBLE_DEVICES=N`

**SAM2 import error (float8 attribute):**
Transformers version conflict. Fix: `pip install transformers==4.47.1`

**Gradio not loading:**
SSH tunnel must be open: `ssh -N -L 7860:localhost:7860 yvs23@ogg.cs.bath.ac.uk`

**Step 1 stopped halfway:**
Just rerun — `--resume` is on by default.

**tmux session gone:**
`tmux ls` then `tmux attach -t benchmark`

**GitHub push rejected:**
`git pull --rebase origin main` then `git push`

**GitHub asks for password:**
Use Personal Access Token: https://github.com/settings/tokens

**WikiArt gets 0 images:**
Use HuggingFace only — old WikiArt API is broken.

---

## Timeline

| Script | Project week |
|---|---|
| `download_wikiart.py` + `00_interactive_click.py` | Wk 3 |
| `01_segment_regions_sam2.py` | Wk 3 |
| `02_pair_styles.py` | Wk 7 |
| `03_quality_control.py` | Wk 7–8 |
| `04_annotate.py` | Wk 7–8 |
| `05_export.py` + final GitHub push | Wk 12 |

---

## Target statistics

| Property | Target |
|---|---|
| Total samples | ~500 |
| Content images | 500 filtered COCO images |
| Style references | 50 images × 20 WikiArt styles |
| Regions per sample | dynamic, mean ~4–6 |
| Distinct WikiArt styles | 20 |
| `similar_entities` | ~30% |
| `encompassed` | ~20% |
| Human-annotated test split | 100% |
