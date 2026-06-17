# COCONut Hybrid Pipeline
## Multi-Region Stylisation Benchmark — Hybrid Track
**Server:** `yvs23@ogg.cs.bath.ac.uk`

---

## What this is

A **separate pipeline** from the original SAM2 track in `scripts/`.
Builds the benchmark using COCONut-PanCap as the foundation for rich
region descriptions, paired with WikiArt style references and dual
instruction formats.

> **The original `scripts/` directory is not modified.**
> All new code lives in `coconut_pipeline/`.

---

## Why COCONut instead of SAM2 from scratch

| What we need | SAM2 from scratch | COCONut hybrid |
|---|---|---|
| Mask quality | Good (SAM2) | SAM2 + guided by captions |
| Region descriptions | Auto-generated stubs | Human-edited, 203 words avg |
| Scale | ~10h GPU for 500 images | 143K to sample from |
| Instruction generation | Template only | Template or API |
| Corner case tagging | 3 tags | 5 tags (adds small_object, cluttered_scene) |

---

## Dual instruction format (per Yudi's suggestion)

Each region gets two instructions supporting two evaluation paradigms:

```json
{
  "region_label": "clear blue sky",
  "style_name": "impressionism",
  "style_reference": "data/style_references/impressionism/image_0016.jpg",
  "instruction_text": "Render clear blue sky in impressionist brushstrokes with soft colours and light.",
  "instruction_ref":  "Render clear blue sky using the style of the reference image."
}
```

- `instruction_text` — for **text-based** style transfer models
- `instruction_ref`  — for **reference image-based** models (MAST, InST, etc.)

---

## Two segmentation tracks

Both tracks use the same 50 COCONut-PanCap images and captions.
They differ only in how SAM2 receives its prompts:

### Track 1 — Auto Grid
- SAM2 prompted with a uniform 4×4 grid across each image
- No prior knowledge of region locations
- Output: `data/coconut_subset/masks_auto/`
- JSON: `subset_auto_final.json`

### Track 2 — Label-Guided Clicks
- SAM2 prompted with click coordinates from COCO instance annotation centroids
- 63/256 regions matched to COCO categories; remainder use grid fallback
- Better alignment between captions and masks
- Output: `data/coconut_subset/masks_click/`
- JSON: `subset_click_final.json`

---

## 50-sample subset results

| Property | Track 1 — Auto | Track 2 — Click |
|---|---|---|
| Samples | 50 | 50 |
| Train/Val/Test | 33/2/15 | 33/2/15 |
| Total regions | 229 | 256 |
| Mean regions/image | 4.6 | 5.1 |
| `similar_entities` | 0 | 0 |
| `encompassed` | 12 | 37 |
| `background_heavy` | 10 | 5 |
| `small_object` | 21 | 38 |
| `cluttered_scene` | 31 | 36 |
| Missing files | 0 | 0 |
| Instruction coverage | 229/229 (100%) | 256/256 (100%) |

**Recommendation:** Use Track 1 as primary benchmark. Use Track 2 to
source `encompassed` and `small_object` corner cases specifically.

---

## Directory structure

```
coconut_pipeline/
├── A_download_coconut.py     ← download images + parse captions from narrative
├── A2_merge_masks.py         ← merge SAM2 mask paths into COCONut stub
├── A3_extract_clicks.py      ← extract COCO annotation centroids as click prompts
├── B_build_subset.py         ← style pairing + dual instruction generation
├── C_quality_control.py      ← QC, 5-tag corner case tagging, 70/10/20 split
├── D_export_subset.py        ← zip archive for supervisor baseline testing
└── README_COCONUT.md         ← this file

data/coconut_subset/          ← created at runtime (not in git except annotations/)
├── images/                   ← COCO train2017 content images
├── masks_auto/               ← Track 1 SAM2 masks
├── masks_click/              ← Track 2 SAM2 masks
└── annotations/              ← ✅ tracked in git
    ├── coconut_stub.json
    ├── coconut_stub_merged_auto.json
    ├── coconut_stub_merged_click.json
    ├── clicks_coconut.json
    ├── subset_auto_draft.json
    ├── subset_auto_final.json    ← Track 1 main deliverable
    ├── subset_click_draft.json
    ├── subset_click_final.json   ← Track 2 main deliverable
    └── dropped_records.json
```

---

## Corner case tags (5 total)

| Tag | Definition | New? |
|---|---|---|
| `similar_entities` | ≥2 regions share same semantic class | original |
| `encompassed` | one region ≥85% inside another | original |
| `background_heavy` | largest region >50% of image | original |
| `small_object` | any region <3% of image area | **new** |
| `cluttered_scene` | ≥5 regions in one image | **new** |

---

## Setup (one time)

```bash
source ~/benchmark_env/bin/activate
pip install pycocotools google-generativeai
```

Ensure WikiArt styles are downloaded:
```bash
ls ~/Benchmark_dataset/data/style_references/ | wc -l   # should show 20
```

---

## Step-by-step pipeline

### Step A — Download COCONut subset

```bash
cd ~/Benchmark_dataset/coconut_pipeline

python3 A_download_coconut.py \
    --output_dir  ../data/coconut_subset \
    --num_samples 50
```

**Save to GitHub after:**
```bash
cd ~/Benchmark_dataset
git add data/coconut_subset/annotations/coconut_stub.json
git commit -m "step A - coconut subset downloaded"
git push
```

---

### Step A3 — Extract click coordinates for Track 2

Requires COCO train annotations:
```bash
cd ~/Benchmark_dataset/data/content_images
wget -q --show-progress http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q annotations_trainval2017.zip
cd ~/Benchmark_dataset/coconut_pipeline
```

```bash
python3 A3_extract_clicks.py \
    --stub          ../data/coconut_subset/annotations/coconut_stub.json \
    --coco_ann      ../data/content_images/annotations/instances_train2017.json \
    --output_clicks ../data/coconut_subset/annotations/clicks_coconut.json
```

---

### SAM2 segmentation — Track 1 (auto)

```bash
export CUDA_VISIBLE_DEVICES=1

python3 ../scripts/01_segment_regions_sam2.py \
    --image_dir   ../data/coconut_subset/images \
    --output_dir  ../data/coconut_subset/masks_auto \
    --checkpoint  ../checkpoints/sam2.1_hiera_large.pt \
    --config      configs/sam2.1/sam2.1_hiera_l.yaml \
    --mode auto \
    --min_area 0.02 \
    --max_masks 6 \
    --no_resume
```

---

### SAM2 segmentation — Track 2 (click)

```bash
python3 ../scripts/01_segment_regions_sam2.py \
    --image_dir   ../data/coconut_subset/images \
    --output_dir  ../data/coconut_subset/masks_click \
    --checkpoint  ../checkpoints/sam2.1_hiera_large.pt \
    --config      configs/sam2.1/sam2.1_hiera_l.yaml \
    --clicks_json ../data/coconut_subset/annotations/clicks_coconut.json \
    --mode click \
    --no_resume
```

---

### Step A2 — Merge masks into stub

```bash
# Track 1
python3 A2_merge_masks.py \
    --stub      ../data/coconut_subset/annotations/coconut_stub.json \
    --sam2_stub ../data/coconut_subset/annotations/masks_stub_sam2.json \
    --output    ../data/coconut_subset/annotations/coconut_stub_merged_auto.json

# Track 2 (rerun SAM2 click first, then merge)
python3 A2_merge_masks.py \
    --stub      ../data/coconut_subset/annotations/coconut_stub.json \
    --sam2_stub ../data/coconut_subset/annotations/masks_stub_sam2.json \
    --output    ../data/coconut_subset/annotations/coconut_stub_merged_click.json
```

---

### Step B — Style pairing + dual instructions

```bash
# Track 1
python3 B_build_subset.py \
    --stub        ../data/coconut_subset/annotations/coconut_stub_merged_auto.json \
    --style_dir   ../data/style_references \
    --output_json ../data/coconut_subset/annotations/subset_auto_draft.json \
    --model stub   # change to gemini for richer instructions

# Track 2
python3 B_build_subset.py \
    --stub        ../data/coconut_subset/annotations/coconut_stub_merged_click.json \
    --style_dir   ../data/style_references \
    --output_json ../data/coconut_subset/annotations/subset_click_draft.json \
    --model stub
```

For Gemini (free, recommended):
```bash
export GEMINI_API_KEY="your-key"   # https://aistudio.google.com/app/apikey
python3 B_build_subset.py ... --model gemini
```

**Save to GitHub after:**
```bash
cd ~/Benchmark_dataset
git add data/coconut_subset/annotations/
git commit -m "step B - style pairing done"
git push
```

---

### Step C — Quality control and split

```bash
# Track 1
python3 C_quality_control.py \
    --draft_json  ../data/coconut_subset/annotations/subset_auto_draft.json \
    --output_json ../data/coconut_subset/annotations/subset_auto_final.json

# Track 2
python3 C_quality_control.py \
    --draft_json  ../data/coconut_subset/annotations/subset_click_draft.json \
    --output_json ../data/coconut_subset/annotations/subset_click_final.json
```

**Save to GitHub after:**
```bash
cd ~/Benchmark_dataset
git add data/coconut_subset/annotations/
git commit -m "step C - QC and split done"
git push
```

---

### Step D — Export zips

```bash
python3 D_export_subset.py \
    --annotated_json ../data/coconut_subset/annotations/subset_auto_final.json \
    --output_zip     ../coconut_auto_export.zip

python3 D_export_subset.py \
    --annotated_json ../data/coconut_subset/annotations/subset_click_final.json \
    --output_zip     ../coconut_click_export.zip
```

Copy to laptop:
```powershell
scp "yvs23@ogg.cs.bath.ac.uk:/mnt/vurm/homes/homes/yvs23/Benchmark_dataset/coconut_auto_export.zip" "C:/Users/Yuvan Velkumar/Downloads/"
scp "yvs23@ogg.cs.bath.ac.uk:/mnt/vurm/homes/homes/yvs23/Benchmark_dataset/coconut_click_export.zip" "C:/Users/Yuvan Velkumar/Downloads/"
```

Delete zips after copying to save disk space:
```bash
rm ~/Benchmark_dataset/coconut_auto_export.zip
rm ~/Benchmark_dataset/coconut_click_export.zip
```

---

### Verification

```bash
cd ~/Benchmark_dataset/coconut_pipeline

python3 - << 'EOF'
import json
from pathlib import Path

for track, json_path in [
    ("AUTO ", "../data/coconut_subset/annotations/subset_auto_final.json"),
    ("CLICK", "../data/coconut_subset/annotations/subset_click_final.json"),
]:
    with open(json_path) as f:
        records = json.load(f)
    root = Path(json_path).parent.parent
    missing = sum(1 for r in records for reg in r["regions"]
                  if not (root / reg["mask_file"]).exists())
    total   = sum(r["num_regions"] for r in records)
    instr   = sum(1 for r in records for reg in r["regions"]
                  if reg.get("instruction_text","").strip())
    print(f"{track}: {len(records)} samples, {total} regions, "
          f"missing={missing}, instr={instr}/{total}")
EOF
```

---

## Troubleshooting

**Step A downloads 0 samples:**
The dataset has only `txt`, `__key__`, `__url__` fields.
The current script parses the narrative text format correctly.
Check field names with: `print(next(iter(ds)).keys())`

**0 regions matched in A3:**
Make sure you're using `instances_train2017.json` not `instances_val2017.json`
— COCONut images come from train2017.

**SAM2 config not found:**
Run from `coconut_pipeline/` directory, not from `scripts/`.
The config path `configs/sam2.1/sam2.1_hiera_l.yaml` is relative to
`~/Benchmark_dataset/`.

**GitHub push rejected:**
```bash
git pull --rebase origin main && git push
```

---

## Scaling up to 500 samples



```bash
python3 A_download_coconut.py \
    --output_dir  ../data/coconut_subset \
    --num_samples 500
```

Then rerun SAM2, A2, B, C, D with the same commands above.
Expected time: ~8 min for SAM2 on 500 images.

---

## Step E — Mask quality IoU evaluation

Evaluates SAM2 mask quality against COCONut panoptic ground truth.
Run this before scaling to 500 samples.

### Setup

```bash
cd ~/Benchmark_dataset/data/content_images/annotations
wget -q --show-progress http://images.cocodataset.org/annotations/panoptic_train2017.zip
unzip -q panoptic_train2017.zip
```

### Run

```bash
cd ~/Benchmark_dataset/coconut_pipeline

python3 E_mask_quality_iou.py \
    --stub        ../data/coconut_subset/annotations/coconut_stub.json \
    --pan_json    ../data/content_images/annotations/panoptic_train2017.json \
    --pan_dir     ../data/content_images/annotations/panoptic_train2017 \
    --masks_auto  ../data/coconut_subset/masks_auto \
    --masks_click ../data/coconut_subset/masks_click \
    --output      ../data/coconut_subset/annotations/mask_quality_iou.json
```

### Output fields per region

| Field | Description |
|---|---|
| `iou` | Intersection over Union vs COCONut ground truth |
| `precision` | Fraction of predicted pixels that are correct |
| `recall` | Fraction of ground truth pixels captured |
| `gt_area` | Ground truth region area as fraction of image |

### Interpreting results

| IoU | Quality |
|---|---|
| >= 0.75 | Good — mask boundary is accurate |
| 0.50–0.75 | Acceptable — some boundary error |
| < 0.50 | Poor — mask misaligned with ground truth |

Track 2 (click-guided) is expected to score higher than Track 1 (auto grid)
because click prompts are derived from COCO annotation centroids.
