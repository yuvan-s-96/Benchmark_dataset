# COCONut Hybrid Pipeline
## Multi-Region Stylisation Benchmark — Hybrid Track
**Server:** `yvs23@ogg.cs.bath.ac.uk`

---

## What this is

A separate pipeline from the original SAM2 track in `scripts/`.
Builds the benchmark using COCONut-PanCap as the foundation for rich
region descriptions, paired with WikiArt style references and three
instruction formats per region.

The original `scripts/` directory is not modified.
All new code lives in `coconut_pipeline/`.

---

## Three instruction formats per region

```json
{
  "region_label": "clear blue sky",
  "style_name": "impressionism",
  "style_reference": "data/style_references/impressionism/image_0016.jpg",
  "instruction_text":      "Apply impressionist brushstrokes to the clear blue sky, using soft pastel tones to capture the shifting afternoon light.",
  "instruction_ref":       "Render clear blue sky using the style of the reference image.",
  "instruction_ref_named": "Render clear blue sky using the impressionism style of the reference image."
}
```

| Field | Use for | Available in |
|---|---|---|
| `instruction_text` | Text-guided style transfer models | All 4 archives |
| `instruction_ref` | Pure image-guided models (MAST) — no text bias | All 4 archives |
| `instruction_ref_named` | Multimodal models (GPT-4V, LLaVA) — image + style name hint | GGUF archives only |

---

## Four export archives

| Archive | Segmentation | Instructions | ref_named | Regions | Best for |
|---|---|---|---|---|---|
| `coconut_auto_export.zip` | SAM2 auto grid | Template (stub) | No | 229 | Quick baseline test |
| `coconut_click_export.zip` | SAM2 label-guided | Template (stub) | No | 256 | Corner case diversity |
| `coconut_auto_gguf_export.zip` | SAM2 auto grid | Mistral-7B GGUF | Yes | 229 | **Primary benchmark ✅** |
| `coconut_click_gguf_export.zip` | SAM2 label-guided | Mistral-7B GGUF | Yes | 256 | Corner cases + rich |

**Recommended:** `coconut_auto_gguf_export.zip` as primary. `coconut_click_gguf_export.zip` for encompassed/small_object corner cases.

---

## Instruction generation modes

| Mode | Script | Quality | Notes |
|---|---|---|---|
| stub | `B_build_subset.py --model stub` | Basic templates | No model needed |
| GGUF/local ✅ | `B_build_subset_gguf.py` | Rich, specific | Mistral-7B on ogg GPU |
| OpenRouter | `B_build_subset_openrouter.py` | Good | Blocked on ogg (no outbound HTTP) |

**Use GGUF mode** — runs on ogg GPU, no internet needed, best quality.
Model: `mistral-7b-instruct-v0.2.Q4_K_M.gguf` at `~/Benchmark_dataset/models/`

---

## Two segmentation tracks

| Property | Track 1 — Auto | Track 2 — Click |
|---|---|---|
| SAM2 prompts | 4×4 uniform grid | COCO annotation centroids |
| Samples | 50 | 50 |
| Total regions | 229 | 256 |
| Mean regions/img | 4.6 | 5.1 |
| `encompassed` | 12 | **37** |
| `background_heavy` | 10 | 5 |
| `small_object` | 21 | **38** |
| `cluttered_scene` | 31 | 36 |
| Instruction coverage | 229/229 ✅ | 256/256 ✅ |

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

## Scripts

```
coconut_pipeline/
├── A_download_coconut.py        ← download images + parse captions from narrative
├── A2_merge_masks.py            ← merge SAM2 mask paths into COCONut stub
├── A3_extract_clicks.py         ← extract COCO annotation centroids as click prompts
├── B_build_subset.py            ← style pairing + stub instructions
├── B_build_subset_gguf.py       ← style pairing + Mistral-7B GGUF instructions ✅
├── B_build_subset_openrouter.py ← style pairing + OpenRouter API (blocked on ogg)
├── C_quality_control.py         ← QC, 5-tag corner case tagging, 70/10/20 split
├── D_export_subset.py           ← zip archive for supervisor baseline testing
├── E_mask_quality_iou.py        ← mask quality evaluation vs COCONut ground truth
└── README_COCONUT.md
```

---

## Directory structure

```
data/coconut_subset/
├── images/                              ← COCO train2017 content images
├── masks_auto/                          ← Track 1 SAM2 masks
├── masks_click/                         ← Track 2 SAM2 masks
└── annotations/                         ← ✅ tracked in git
    ├── coconut_stub.json
    ├── coconut_stub_merged_auto.json
    ├── coconut_stub_merged_click.json
    ├── clicks_coconut.json
    ├── subset_auto_final_gguf.json      ← Track 1 GGUF ✅ primary deliverable
    ├── subset_click_final_gguf.json     ← Track 2 GGUF ✅
    └── mask_quality_iou.json            ← IoU evaluation results (Step E)

data/content_images/annotations/
    ├── instances_train2017.json         ← COCO instance annotations (for A3)
    └── panoptic_train2017.json          ← COCONut panoptic GT (for Step E)

models/
└── mistral-7b-instruct-v0.2.Q4_K_M.gguf  ← local LLM (~4 GB, not in git)
```

---

## Setup (one time)

```bash
source ~/benchmark_env/bin/activate

# Install dependencies
pip install pycocotools
pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Download Mistral-7B GGUF (~4 GB)
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='TheBloke/Mistral-7B-Instruct-v0.2-GGUF',
    filename='mistral-7b-instruct-v0.2.Q4_K_M.gguf',
    local_dir='/homes/yvs23/Benchmark_dataset/models'
)
print('Done.')
"
```

---

## Full pipeline commands

### Step A — Download COCONut subset
```bash
cd ~/Benchmark_dataset/coconut_pipeline

python3 A_download_coconut.py \
    --output_dir  ../data/coconut_subset \
    --num_samples 50
```

**Save to GitHub:**
```bash
cd ~/Benchmark_dataset
git add data/coconut_subset/annotations/coconut_stub.json
git commit -m "step A - coconut downloaded" && git push
```

---

### Step A3 — Extract click coordinates (Track 2 only)

Requires COCO train annotations:
```bash
cd ~/Benchmark_dataset/data/content_images/annotations
curl -L -o annotations_trainval2017.zip \
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q annotations_trainval2017.zip
rm annotations_trainval2017.zip
cd ~/Benchmark_dataset/coconut_pipeline
```

Note: use `curl -L` not `wget` — wget silently fails on ogg for this URL.

```bash
python3 A3_extract_clicks.py \
    --stub          ../data/coconut_subset/annotations/coconut_stub.json \
    --coco_ann      ../data/content_images/annotations/instances_train2017.json \
    --output_clicks ../data/coconut_subset/annotations/clicks_coconut.json
```

---

### SAM2 — Track 1 (auto grid)
```bash
export CUDA_VISIBLE_DEVICES=1

python3 ../scripts/01_segment_regions_sam2.py \
    --image_dir   ../data/coconut_subset/images \
    --output_dir  ../data/coconut_subset/masks_auto \
    --checkpoint  ../checkpoints/sam2.1_hiera_large.pt \
    --config      configs/sam2.1/sam2.1_hiera_l.yaml \
    --mode auto --min_area 0.02 --max_masks 6 --no_resume
```

---

### SAM2 — Track 2 (label-guided clicks)
```bash
python3 ../scripts/01_segment_regions_sam2.py \
    --image_dir   ../data/coconut_subset/images \
    --output_dir  ../data/coconut_subset/masks_click \
    --checkpoint  ../checkpoints/sam2.1_hiera_large.pt \
    --config      configs/sam2.1/sam2.1_hiera_l.yaml \
    --clicks_json ../data/coconut_subset/annotations/clicks_coconut.json \
    --mode click --no_resume
```

---

### Step A2 — Merge masks into stub

Run once per track immediately after each SAM2 run:
```bash
# Track 1 — run after masks_auto SAM2
python3 A2_merge_masks.py \
    --stub      ../data/coconut_subset/annotations/coconut_stub.json \
    --sam2_stub ../data/coconut_subset/annotations/masks_stub_sam2.json \
    --output    ../data/coconut_subset/annotations/coconut_stub_merged_auto.json

# Track 2 — run after masks_click SAM2
python3 A2_merge_masks.py \
    --stub      ../data/coconut_subset/annotations/coconut_stub.json \
    --sam2_stub ../data/coconut_subset/annotations/masks_stub_sam2.json \
    --output    ../data/coconut_subset/annotations/coconut_stub_merged_click.json
```

---

### Step B — Style pairing + GGUF instructions ✅
```bash
export CUDA_VISIBLE_DEVICES=1

# Track 1
python3 B_build_subset_gguf.py \
    --stub        ../data/coconut_subset/annotations/coconut_stub_merged_auto.json \
    --style_dir   ../data/style_references \
    --output_json ../data/coconut_subset/annotations/subset_auto_draft_gguf.json

# Track 2
python3 B_build_subset_gguf.py \
    --stub        ../data/coconut_subset/annotations/coconut_stub_merged_click.json \
    --style_dir   ../data/style_references \
    --output_json ../data/coconut_subset/annotations/subset_click_draft_gguf.json
```

**Save to GitHub:**
```bash
cd ~/Benchmark_dataset
git add data/coconut_subset/annotations/
git commit -m "step B - GGUF instructions generated" && git push
```

---

### Step C — Quality control and split
```bash
# Track 1
python3 C_quality_control.py \
    --draft_json  ../data/coconut_subset/annotations/subset_auto_draft_gguf.json \
    --output_json ../data/coconut_subset/annotations/subset_auto_final_gguf.json

# Track 2
python3 C_quality_control.py \
    --draft_json  ../data/coconut_subset/annotations/subset_click_draft_gguf.json \
    --output_json ../data/coconut_subset/annotations/subset_click_final_gguf.json
```

**Save to GitHub:**
```bash
cd ~/Benchmark_dataset
git add data/coconut_subset/annotations/
git commit -m "step C - QC and split done" && git push
```

---

### Step D — Export zips
```bash
python3 D_export_subset.py \
    --annotated_json ../data/coconut_subset/annotations/subset_auto_final_gguf.json \
    --output_zip     ../coconut_auto_gguf_export.zip

python3 D_export_subset.py \
    --annotated_json ../data/coconut_subset/annotations/subset_click_final_gguf.json \
    --output_zip     ../coconut_click_gguf_export.zip
```

Copy to laptop:
```powershell
scp "yvs23@ogg.cs.bath.ac.uk:/mnt/vurm/homes/homes/yvs23/Benchmark_dataset/coconut_auto_gguf_export.zip" "C:/Users/Yuvan Velkumar/Downloads/"
scp "yvs23@ogg.cs.bath.ac.uk:/mnt/vurm/homes/homes/yvs23/Benchmark_dataset/coconut_click_gguf_export.zip" "C:/Users/Yuvan Velkumar/Downloads/"
```

Delete after copying:
```bash
rm ~/Benchmark_dataset/coconut_auto_gguf_export.zip
rm ~/Benchmark_dataset/coconut_click_gguf_export.zip
```

---

### Step E — Mask quality IoU evaluation

Evaluates SAM2 mask quality against COCONut panoptic ground truth.
Run before scaling to 500 samples to confirm mask accuracy.

#### Setup — panoptic annotations

Note: panoptic annotations are in a separate zip from the standard annotations:
```bash
cd ~/Benchmark_dataset/data/content_images/annotations
curl -L -o panoptic_annotations_trainval2017.zip \
    http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
unzip -q panoptic_annotations_trainval2017.zip
rm panoptic_annotations_trainval2017.zip
ls panoptic_train2017.json
```

#### Run
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

#### Output fields per region

| Field | Description |
|---|---|
| `iou` | Intersection over Union vs COCONut ground truth (0.0–1.0) |
| `precision` | Fraction of predicted pixels that are correct |
| `recall` | Fraction of ground truth pixels captured by SAM2 |
| `gt_area` | Ground truth region area as fraction of image |

#### Interpreting results

| IoU range | Quality | Action |
|---|---|---|
| >= 0.75 | Good — boundary is accurate | Safe to scale up |
| 0.50–0.75 | Acceptable — some boundary error | Review samples visually |
| < 0.50 | Poor — mask misaligned | Investigate before scaling |

Track 2 (click-guided) expected to score higher than Track 1 (auto grid)
because click prompts are derived from COCO annotation centroids.

**Save to GitHub after running:**
```bash
cd ~/Benchmark_dataset
git add data/coconut_subset/annotations/mask_quality_iou.json
git commit -m "step E - mask quality IoU results" && git push
```

---

### Verify before sending

```bash
python3 - << 'EOF'
import json
for track, jfile in [
    ("AUTO ", "../data/coconut_subset/annotations/subset_auto_final_gguf.json"),
    ("CLICK", "../data/coconut_subset/annotations/subset_click_final_gguf.json"),
]:
    with open(jfile) as f:
        records = json.load(f)
    total = sum(r["num_regions"] for r in records)
    t_ok  = sum(1 for r in records for reg in r["regions"] if reg.get("instruction_text","").strip())
    r_ok  = sum(1 for r in records for reg in r["regions"] if reg.get("instruction_ref","").strip())
    rn_ok = sum(1 for r in records for reg in r["regions"] if reg.get("instruction_ref_named","").strip())
    reg   = records[0]["regions"][0]
    print(f"\n{track}: {len(records)} samples, {total} regions")
    print(f"  instruction_text      : {t_ok}/{total}")
    print(f"  instruction_ref       : {r_ok}/{total}")
    print(f"  instruction_ref_named : {rn_ok}/{total}")
    print(f"  Example ref_named     : {reg['instruction_ref_named']}")
EOF
```

---

## Disk space management

Check usage:
```bash
quota -s
du -sh ~/Benchmark_dataset/data/*/
du -sh ~/Benchmark_dataset/models/
du -sh ~/benchmark_env/
```

Safe to delete (regeneratable):
```bash
rm -rf ~/Benchmark_dataset/data/content_images/      # 1.8 GB, redownloadable
rm -rf ~/Benchmark_dataset/data/mask_previews*/       # preview images
rm -rf ~/Benchmark_dataset/data/previews*/            # preview images
rm -rf ~/Benchmark_dataset/data/masks_sam1/           # test run masks
rm -rf ~/Benchmark_dataset/data/masks_sam2/           # test run masks
rm -rf ~/Benchmark_dataset/data/masks_sam2_click/     # test run masks
rm -rf ~/Benchmark_dataset/data/test_images/          # 20-image test set
rm -f  ~/Benchmark_dataset/*.zip                      # after copying to laptop
rm -rf ~/.cache/huggingface/hub/datasets--*/          # dataset cache
# Delete annotation zips after unzipping
rm -f ~/Benchmark_dataset/data/content_images/annotations/*.zip
```

Never delete:
```bash
~/Benchmark_dataset/models/                    # Mistral GGUF, slow to redownload
~/Benchmark_dataset/data/coconut_subset/       # push to GitHub first
~/Benchmark_dataset/data/style_references/     # 5 min to redownload
~/benchmark_env/                               # 10 min to recreate
~/Benchmark_dataset/checkpoints/               # SAM2 weights
```

Always push before clearing:
```bash
cd ~/Benchmark_dataset
git add data/coconut_subset/annotations/
git commit -m "backup before cleanup" && git push
```

---

## Scaling to 500 samples

Target: minimum 50 samples per corner case category before stopping.
Categories: similar_entities, encompassed, background_heavy, small_object, cluttered_scene.
Expected total: ~250–400 samples with natural overlap.

```bash
python3 A_download_coconut.py \
    --output_dir  ../data/coconut_subset \
    --num_samples 500
```

Then rerun SAM2, A2, B (GGUF), C, D with same commands.
Expected times for 500 samples:
- SAM2 auto: ~8 min
- Mistral-7B GGUF: ~25 min (500 images × ~3s/region × 4.6 regions)
- Total: ~35 min

Run E (mask quality IoU) again after scaling to confirm quality holds.

---

## GitHub workflow

```bash
cd ~/Benchmark_dataset
git pull origin main --no-rebase
git add coconut_pipeline/
git add data/coconut_subset/annotations/
git commit -m "description"
git push
```

Note: if merge conflict in README_COCONUT.md:
```bash
git checkout coconut_pipeline/README_COCONUT.md
git pull origin main --no-rebase
git push
```

---

## Troubleshooting

**Git push rejected:**
```bash
git pull origin main --no-rebase && git push
```

**Git merge conflict in README:**
```bash
git checkout coconut_pipeline/README_COCONUT.md
git pull origin main --no-rebase
git push
```

**Disk quota exceeded:**
Delete content images, zips, preview folders. See Disk space management above.

**SAM2 config not found:**
Run from `coconut_pipeline/` directory.
Config path is relative to `~/Benchmark_dataset/`.

**0 regions matched in A3:**
Use `instances_train2017.json` not `instances_val2017.json`.
COCONut uses COCO train2017 images.

**panoptic_train2017.json not found (Step E):**
It is NOT in `annotations_trainval2017.zip`. Download separately:
```bash
curl -L -o panoptic_annotations_trainval2017.zip \
    http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
unzip -q panoptic_annotations_trainval2017.zip
```

**wget silently downloads 325-byte error file:**
Use `curl -L` instead of `wget` for COCO downloads on ogg.

**llama-cpp-python install fails:**
```bash
pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

**instruction_ref_named missing in older exports:**
Only GGUF archives have this field. Rerun B_build_subset_gguf.py to regenerate.
