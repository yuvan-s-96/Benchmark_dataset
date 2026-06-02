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
│   └── download_wikiart.py       ← Download WikiArt style reference images
├── slurm_jobs/                   ← (not needed for ogg — use tmux instead)
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
| Region count | Fully dynamic — one region per click, or auto-detected. No fixed cap. |
| GPU model choice | Default `sam-vit-large` — fits on ogg's RTX 2080 (8 GB) comfortably |
| Local dev | Use `sam-vit-base` on your RTX 3050 (4 GB) for testing |
| Crash safety | Step 1 writes the stub JSON after every image — safe to interrupt |
| Resume | Step 1 skips already-processed images on rerun |
| Annotation resume | Step 4 starts from first unannotated sample automatically |
| Export | Step 5 zips only what is needed (masks + styles + JSON), excludes raw COCO |

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

Pick a GPU with the least memory used. GPU 1 or GPU 3 are usually quietest.
Set your chosen GPU for this session:

```bash
export CUDA_VISIBLE_DEVICES=1    # change to your chosen GPU number
```

### 3. Clone repo and set up environment

```bash
cd ~
git clone https://github.com/<your-username>/benchmark_dataset.git
cd benchmark_dataset

conda create -n benchmark python=3.11 -y
conda activate benchmark

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers pillow tqdm numpy gradio requests openai scikit-image
```

### 4. Pre-download the SAM model (do this on the login node, once)

This prevents the compute step from spending time downloading on a GPU node.

```bash
conda activate benchmark
export HF_HOME=~/.cache/huggingface

# For ogg RTX 2080 (8 GB VRAM) — recommended
python -c "
from transformers import SamModel, SamProcessor
SamModel.from_pretrained('facebook/sam-vit-large')
SamProcessor.from_pretrained('facebook/sam-vit-large')
print('Done.')
"
```

> **Local RTX 3050 (4 GB)?** Use `facebook/sam-vit-base` instead everywhere below.

---

## Step-by-step pipeline

---

### Step 0 — Collect COCO content images

Run on the ogg login node (network I/O only, no GPU needed).

```bash
mkdir -p data/content_images
cd data/content_images
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
mv val2017/* .
rm -rf val2017 val2017.zip
cd ../..
```

~5 000 images, ~1 GB.

---

### Step 0b — Download WikiArt style references

Run on the ogg login node. Respects the WikiArt rate limit (~20–30 min).

```bash
conda activate benchmark
cd scripts
python download_wikiart.py \
    --output_dir ../data/style_references \
    --max_per_style 50
cd ..
```

Output: 20 styles × 50 images = ~1 000 JPEG files.

---

### Step 0c — Collect click points (Gradio UI)

Click on images in your browser to mark region seeds.
**Each click = one SAM region. No limit per image.**

#### Option A — Run locally (simplest)

```bash
# On your local machine
conda activate benchmark
cd scripts
python 00_interactive_click.py \
    --image_dir ../data/content_images \
    --output_json ../data/annotations/clicks.json \
    --port 7861
```

Open **http://localhost:7861** directly.

#### Option B — Run on ogg, view locally

```bash
# Terminal 1 — on ogg
conda activate benchmark
cd ~/benchmark_dataset/scripts
python 00_interactive_click.py --port 7861
```

```bash
# Terminal 2 — on your laptop (new window)
ssh -N -L 7861:localhost:7861 yvs23@ogg.cs.bath.ac.uk
```

Open **http://localhost:7861**

**How to use:**
- Click on the image to add numbered region markers.
- Add as many clicks as the image needs (2–5 typical).
- Click **Save & Next ▶** to save and move on.
- Use **⏩ Save & skip to next unannotated** to resume after a break.
- **↩ Undo last** and **🗑 Clear all** fix mis-clicks.
- Closes safely at any time — resumes from where you left off.

Output: `data/annotations/clicks.json`

---

### Step 1 — Segment regions with SAM

Run in a **tmux session** so it survives disconnections.

```bash
# On ogg
export CUDA_VISIBLE_DEVICES=1         # pick your free GPU
conda activate benchmark
tmux new -s segment

cd ~/benchmark_dataset/scripts
python 01_segment_regions.py \
    --image_dir   ../data/content_images \
    --output_dir  ../data/masks \
    --sam_model_id facebook/sam-vit-large \
    --mode        mixed \
    --clicks_json ../data/annotations/clicks.json

# Detach tmux: Ctrl+B then D
# Reattach later: tmux attach -t segment
```

**Mode options:**
- `mixed` (default) — uses clicks when available, auto otherwise
- `click` — only processes images that have clicks
- `auto`  — ignores clicks, uses SAM automatic mode for everything

**What it does:**
- Click mode: one mask per click, exactly your intended regions.
- Auto mode: filters masks smaller than 2% area, keeps up to 8 per image.
- Writes `data/annotations/masks_stub.json` after every image (crash-safe).
- Skips already-processed images on rerun (`--resume` is on by default).

Expected runtime: ~25 s/image on RTX 2080 → ~3.5 h for 500 images.

Output: `data/masks/<image_id>/mask_NN.png` + `data/annotations/masks_stub.json`

---

### Step 2 — Pair style references and generate instructions

CPU only — runs quickly, no GPU needed.

```bash
conda activate benchmark
cd ~/benchmark_dataset/scripts

python 02_pair_styles.py \
    --masks_stub   ../data/annotations/masks_stub.json \
    --style_dir    ../data/style_references \
    --output_json  ../data/annotations/benchmark_draft.json \
    --instruction_model stub
```

**To use GPT-4o for richer instructions (optional):**

```bash
export OPENAI_API_KEY="sk-..."
python 02_pair_styles.py \
    --masks_stub   ../data/annotations/masks_stub.json \
    --style_dir    ../data/style_references \
    --output_json  ../data/annotations/benchmark_draft.json \
    --instruction_model gpt4v
```

Output: `data/annotations/benchmark_draft.json`

---

### Step 3 — Quality control and split

CPU only.

```bash
python 03_quality_control.py \
    --draft_json  ../data/annotations/benchmark_draft.json \
    --output_json ../data/annotations/benchmark_final.json \
    --min_regions 2
```

**What it does:**
- Drops samples with fewer than 2 regions or missing mask files.
- Drops samples where two non-contained masks overlap (IoU > 0.5).
- Tags corner cases:

  | Tag | Meaning | Target |
  |---|---|---|
  | `similar_entities` | ≥2 regions same semantic class | ~30% |
  | `encompassed` | one region ≥85% inside another | ~20% |
  | `background_heavy` | largest region > 50% of image | ~25% |

- Stratified 70/10/20 train/val/test split by corner-case tag.

Output: `data/annotations/benchmark_final.json` + `dropped_records.json`

---

### Step 4 — Human annotation

Same SSH tunnel approach as Step 0c, on port 7860.

```bash
# Terminal 1 — on ogg (or locally)
conda activate benchmark
cd ~/benchmark_dataset/scripts
python 04_annotate.py \
    --benchmark_json ../data/annotations/benchmark_final.json \
    --output_json    ../data/annotations/benchmark_annotated.json \
    --port 7860
```

```bash
# Terminal 2 — on your laptop (if running on ogg)
ssh -N -L 7860:localhost:7860 yvs23@ogg.cs.bath.ac.uk
```

Open **http://localhost:7860**

**For each sample:**
1. Review the colour-coded region overlay.
2. Fill in **Region labels** — one line per region: `Region 1: sky`, `Region 2: left giraffe`
3. Edit **Region instructions** to be fluent and specific.
4. Set status: `approved` or `rejected`.
5. Click **Next ▶** — saves automatically.

**Annotation priorities:**
- All 100 **test** samples: full labels + instructions + accept/reject.
- Train/val: verify labels at minimum; auto instructions are acceptable.

Output: `data/annotations/benchmark_annotated.json`

---

### Step 5 — Export final benchmark

```bash
python 05_export.py \
    --annotated_json ../data/annotations/benchmark_annotated.json \
    --output_zip     ../benchmark_final_export.zip
```

This creates a self-contained zip with:
- `benchmark_annotated.json`
- All mask PNGs referenced in the JSON
- All style reference images referenced in the JSON
- `export_summary.txt` with dataset statistics

**Copy the zip to your laptop:**

```bash
# Run in a new terminal ON YOUR LAPTOP
scp yvs23@ogg.cs.bath.ac.uk:~/benchmark_dataset/benchmark_final_export.zip .

# Or with rsync (resumable if connection drops):
rsync -avz --progress yvs23@ogg.cs.bath.ac.uk:~/benchmark_dataset/benchmark_final_export.zip .
```

---

## Saving results to GitHub

Keep the JSON in git (small), exclude the binary data files.

`.gitignore`:
```
data/masks/
data/content_images/
data/style_references/
*.zip
logs/
venv/
__pycache__/
*.pyc
```

After annotation is complete:

```bash
cd ~/benchmark_dataset
git add data/annotations/benchmark_annotated.json
git add data/annotations/benchmark_final.json
git commit -m "final annotated benchmark - 500 samples"
git push
```

You now have two complementary backups:
- **GitHub** — JSON + code, version-controlled
- **zip file** — everything including binary images, portable archive

---

## Checking results at any point

```bash
python - << 'EOF'
import json
with open("data/annotations/benchmark_annotated.json") as f:
    data = json.load(f)

splits = {}
tags   = {}
counts = []
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

## GPU selection on ogg (quick reference)

```bash
# See all GPUs and who is using them
nvidia-smi

# GPU 1 and GPU 3 are usually the quietest on ogg
# Set before running anything:
export CUDA_VISIBLE_DEVICES=1
```

RTX 2080 has 8 GB VRAM. `sam-vit-large` uses ~2.5 GB, leaving 5.5 GB headroom.
Close other processes if you see OOM errors.

---

## Local machine (RTX 3050, 4 GB VRAM)

Use for: click collection, WikiArt download, style pairing, QC, annotation UI.
Use ogg for: SAM segmentation (Step 1).

When running Step 1 locally for testing:

```bash
python 01_segment_regions.py \
    --sam_model_id facebook/sam-vit-base \   # fits in 4 GB
    --mode auto \
    --image_dir ../data/content_images
```

Close Chrome, VS Code, and other GPU-heavy apps before running.

---

## Troubleshooting

**CUDA out of memory:**
Switch to a smaller model: `sam-vit-large` → `sam-vit-base`.
Or pick a less-busy GPU: re-check `nvidia-smi`.

**Gradio not loading in browser:**
Make sure the SSH tunnel terminal is still open.
Check the port matches: `ssh -N -L 7860:localhost:7860 ...`

**Step 1 stopped halfway:**
Just rerun the same command. `--resume` is on by default — already-done images are skipped.

**tmux session disappeared:**
```bash
tmux ls            # list sessions
tmux attach -t segment   # reattach
```

**Not enough `similar_entities` samples:**
Pre-filter COCO to images with ≥2 annotations of the same category using
`instances_val2017.json` before running the pipeline.

**HuggingFace download fails inside a job:**
Pre-download on the login node (see One-time setup Step 4 above).

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
| Regions per sample | dynamic, mean ~2.8 |
| Distinct WikiArt styles in test | ≥ 20 |
| `similar_entities` | ~30% |
| `encompassed` | ~20% |
| Human-annotated test split | 100% |
