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
Large binary files (masks, images) are excluded from git — keep those in ogg
or export them with Step 5.

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
│   └── benchmark_schema.json
├── .gitignore
└── README_HEX.md
```

`data/` is created at runtime and is NOT tracked by git (except `data/annotations/`).

---

## Directory structure at runtime

```
data/
├── content_images/          ← COCO 2017 val (5000 images, ~1 GB) — redownload if lost
├── content_images_filtered/ ← filtered COCO subset (500 images) — redownload if lost
├── style_references/        ← WikiArt via HuggingFace (1000 images) — redownload if lost
├── test_images/             ← 20-image test subset — recreate from content_images
├── masks/                   ← SAM mask PNGs — regenerate with Step 1
├── annotations/             ← ✅ TRACKED IN GIT — save these always
│   ├── clicks.json
│   ├── masks_stub.json
│   ├── benchmark_draft.json
│   ├── benchmark_final.json
│   └── benchmark_annotated.json
└── mask_previews/           ← visual previews — regenerate anytime
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
```

GPU 1 and GPU 3 are usually quietest (RTX 2080, 8 GB each).

```bash
export CUDA_VISIBLE_DEVICES=1
```

### 3. Clone repo

```bash
cd ~
git clone https://github.com/yuvan-s-96/Benchmark_dataset.git
cd Benchmark_dataset
```

> Use a Personal Access Token as the password (not your GitHub password).
> Create at: https://github.com/settings/tokens → Generate new token (classic) → tick `repo`

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
pip install transformers pillow tqdm numpy gradio requests openai scikit-image datasets
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 7. Pre-download SAM model (once only)

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

## Recovering lost data

If ogg loses your files (it has happened), here is what to recover:

```bash
# 1. Content images — redownload COCO
mkdir -p ~/Benchmark_dataset/data/content_images
cd ~/Benchmark_dataset/data/content_images
wget -q --show-progress http://images.cocodataset.org/zips/val2017.zip
unzip -q val2017.zip
mv val2017/* .
rm -rf val2017 val2017.zip
cd ~/Benchmark_dataset

# 2. Style references — redownload from HuggingFace
# If ~/data/style_references still exists, copy it:
cp -r ~/data/style_references ~/Benchmark_dataset/data/style_references
# Otherwise rerun:
cd scripts && python3 download_wikiart.py --output_dir ../data/style_references --max_per_style 50

# 3. Test images — recreate from content_images
mkdir -p ~/Benchmark_dataset/data/test_images
ls ~/Benchmark_dataset/data/content_images/*.jpg | head -20 | xargs -I{} cp {} ~/Benchmark_dataset/data/test_images/

# 4. Annotations — already in GitHub, pull them
cd ~/Benchmark_dataset && git pull

# 5. Masks — regenerate with Step 1 (use --resume to skip done images)
cd scripts
python3 01_segment_regions.py \
    --image_dir  ../data/content_images \
    --output_dir ../data/masks \
    --mode auto --min_area 0.02 --max_masks 6
```

---

## Step-by-step pipeline

---

### Step 0 — Collect COCO content images

```bash
mkdir -p ~/Benchmark_dataset/data/content_images
cd ~/Benchmark_dataset/data/content_images
wget -q --show-progress http://images.cocodataset.org/zips/val2017.zip
unzip -q val2017.zip
mv val2017/* .
rm -rf val2017 val2017.zip
echo "Images: $(ls | wc -l)"
```

### Step 0b — Filter COCO to easier images (recommended)

Filters to images with 2–5 large distinct objects — avoids tiny hard-to-segment
regions like eyes.

```bash
# Download COCO annotations
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

### Step 0c — Download WikiArt style references

```bash
cd ~/Benchmark_dataset/scripts
python3 download_wikiart.py \
    --output_dir ../data/style_references \
    --max_per_style 50
```

Output: 20 styles × 50 images = 1000 JPEGs. Runtime ~5 min.

### Step 0d — Collect click points (optional, for test split)

```bash
python3 00_interactive_click.py --port 7861
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
git commit -m "clicks collected for test split"
git push
```

---

### Step 1 — Segment regions with SAM

Run inside tmux:

```bash
tmux new -s benchmark
source ~/benchmark_env/bin/activate
export CUDA_VISIBLE_DEVICES=1
cd ~/Benchmark_dataset/scripts

python3 01_segment_regions.py \
    --image_dir   ../data/content_images_filtered \
    --output_dir  ../data/masks \
    --sam_model_id facebook/sam-vit-large \
    --mode mixed \
    --min_area 0.02 \
    --max_masks 6
```

Detach: **Ctrl+B then D** | Reattach: `tmux attach -t benchmark`

**Progress check:**
```bash
python3 -c "
import json
with open('$HOME/Benchmark_dataset/data/annotations/masks_stub.json') as f:
    d = json.load(f)
print(f'Processed: {len(d)} images')
print(f'Regions: min={min(r[\"num_regions\"] for r in d)}  max={max(r[\"num_regions\"] for r in d)}  mean={sum(r[\"num_regions\"] for r in d)/len(d):.1f}')
"
```

**Save to GitHub after:**
```bash
cd ~/Benchmark_dataset
git add data/annotations/masks_stub.json
git commit -m "step 1 complete - SAM segmentation done"
git push
```

---

### Step 1b — Visual preview + pure mask saving

This generates two outputs per image as requested by supervisor:
- **Colour overlay** — original image with coloured region overlays (for review)
- **Pure binary masks** — black and white PNG per region (white = region, black = background)

```bash
cd ~/Benchmark_dataset/scripts

python3 - << 'EOF'
import json, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

with open("../data/annotations/masks_stub.json") as f:
    records = json.load(f)

COLOURS = [
    (255, 80,  80,  140), (80,  180, 255, 140), (80,  255, 130, 140),
    (255, 200, 50,  140), (200, 80,  255, 140), (255, 140, 0,   140),
    (0,   210, 210, 140), (255, 100, 160, 140), (160, 255, 80,  140),
    (80,  80,  255, 140),
]

# Output folders
overlay_dir = Path("../data/mask_previews")
pure_dir    = Path("../data/pure_mask_previews")
overlay_dir.mkdir(parents=True, exist_ok=True)
pure_dir.mkdir(parents=True, exist_ok=True)

for record in records[:5]:   # change [:5] to [:] for all images
    # Find source image
    candidates = list(Path("../data/content_images_filtered").glob(f"{record['image_id']}.*"))
    if not candidates:
        candidates = list(Path("../data/test_images").glob(f"{record['image_id']}.*"))
    if not candidates:
        print(f"  [skip] {record['image_id']}: source image not found")
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

        # ── Colour overlay ────────────────────────────────────────────────
        layer = np.zeros((*base.size[::-1], 4), dtype=np.uint8)
        layer[mask, :3] = colour[:3]
        layer[mask, 3]  = colour[3]
        overlay = Image.alpha_composite(overlay, Image.fromarray(layer))
        ys, xs = np.where(mask)
        if len(xs):
            draw.text((int(xs.mean())-8, int(ys.mean())-8),
                      str(i+1), fill=(255, 255, 255, 255))

        # ── Pure binary mask (white region on black background) ───────────
        pure = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        pure.save(pure_dir / f"{record['image_id']}_mask_{i+1:02d}.png")

    # Save colour overlay
    result = Image.alpha_composite(base, overlay).convert("RGB")
    result.save(overlay_dir / f"{record['image_id']}_overlay.jpg", quality=90)
    print(f"Saved: {record['image_id']}  ({record['num_regions']} regions)")

print(f"\nColour overlays : {overlay_dir}")
print(f"Pure masks      : {pure_dir}")
EOF
```

**What you get in `pure_mask_previews/`:**
```
000000000139_mask_01.png   ← region 1 pure mask (white=region, black=background)
000000000139_mask_02.png   ← region 2 pure mask
000000000139_mask_03.png   ← region 3 pure mask
...
```

**Copy both to your laptop** (run on your laptop, not ogg):
```bash
# Colour overlays
scp "yvs23@ogg.cs.bath.ac.uk:~/Benchmark_dataset/data/mask_previews/*.jpg" "C:/Users/Yuvan Velkumar/Downloads/"

# Pure binary masks
scp "yvs23@ogg.cs.bath.ac.uk:~/Benchmark_dataset/data/pure_mask_previews/*.png" "C:/Users/Yuvan Velkumar/Downloads/"
```

---

### Step 2 — Pair style references and generate instructions

```bash
cd ~/Benchmark_dataset/scripts

python3 02_pair_styles.py \
    --masks_stub   ../data/annotations/masks_stub.json \
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

**Save to GitHub frequently during annotation:**
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
git commit -m "final benchmark complete - all steps done"
git push
```

---

## Cleaning up disk space

Check usage first:
```bash
du -sh ~/Benchmark_dataset/data/*/
du -sh ~/data/
df -h ~
```

Safe to delete (regeneratable):
```bash
# Remove raw COCO images (redownloadable, ~1 GB)
rm -rf ~/Benchmark_dataset/data/content_images

# Remove masks (regeneratable with Step 1, can be large)
rm -rf ~/Benchmark_dataset/data/masks
rm -rf ~/Benchmark_dataset/data/masks_click

# Remove previews
rm -rf ~/Benchmark_dataset/data/mask_previews
rm -rf ~/Benchmark_dataset/data/mask_previews_click
rm -rf ~/Benchmark_dataset/data/pure_mask_previews

# Remove test images
rm -rf ~/Benchmark_dataset/data/test_images

# Remove zip after copying to laptop
rm -f ~/Benchmark_dataset/benchmark_final_export.zip
```

**Never delete:**
```bash
# These are your work — keep them
~/Benchmark_dataset/data/annotations/     # push to GitHub first
~/Benchmark_dataset/data/style_references/ # takes 5 min to redownload
~/benchmark_env/                           # takes 10 min to recreate
```

**Before clearing anything, always push annotations to GitHub:**
```bash
cd ~/Benchmark_dataset
git add data/annotations/
git commit -m "backup before cleanup"
git push
```

---

## GPU selection on ogg

```bash
nvidia-smi          # see all 6 GPUs
export CUDA_VISIBLE_DEVICES=1   # pick quietest one
```

6× RTX 2080, 8 GB VRAM. `sam-vit-large` uses ~2.5 GB.

---

## Local machine (RTX 3050, 4 GB VRAM)

Use locally for: click collection, annotation UI, steps 2–5.
Use ogg for: Step 1 SAM segmentation only.

```bash
python3 01_segment_regions.py \
    --sam_model_id facebook/sam-vit-base \
    --mode auto \
    --image_dir ../data/test_images
```

---

## Troubleshooting

**Files disappeared on ogg:**
ogg does not backup. Always push annotations to GitHub after each step.
Regenerate data using the Recovery section above.

**CUDA out of memory:**
Pick a less busy GPU: `nvidia-smi` then `export CUDA_VISIBLE_DEVICES=N`.

**Gradio not loading:**
SSH tunnel must be open in a separate terminal:
`ssh -N -L 7860:localhost:7860 yvs23@ogg.cs.bath.ac.uk`

**Step 1 stopped halfway:**
Just rerun — `--resume` is on by default.

**tmux session gone:**
`tmux ls` then `tmux attach -t benchmark`

**GitHub push rejected:**
`git pull --rebase origin main` then `git push`

**GitHub asks for password:**
Use Personal Access Token not your GitHub password.
Create at: https://github.com/settings/tokens

**WikiArt gets 0 images:**
Use `download_wikiart.py` with HuggingFace only — the old WikiArt API is broken.

**SAM gives 1 region per image:**
Fixed in current `01_segment_regions.py` — uses grid point prompts.

---

## Timeline

| Script | Project week |
|---|---|
| `download_wikiart.py` + `00_interactive_click.py` | Wk 3 |
| `01_segment_regions.py` | Wk 3 |
| `02_pair_styles.py` | Wk 7 |
| `03_quality_control.py` | Wk 7–8 |
| `04_annotate.py` | Wk 7–8 |
| `05_export.py` + final GitHub push | Wk 12 |

---

## Target statistics

| Property | Target |
|---|---|
| Total samples | ~500 |
| Regions per sample | dynamic, mean ~4.7 |
| Distinct WikiArt styles | 20 |
| `similar_entities` | ~30% |
| `encompassed` | ~20% |
| Human-annotated test split | 100% |
