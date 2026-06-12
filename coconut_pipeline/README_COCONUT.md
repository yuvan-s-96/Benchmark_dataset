# COCONut Hybrid Pipeline
## Multi-Region Stylisation Benchmark — Hybrid Track
**Server:** `yvs23@ogg.cs.bath.ac.uk`

## What this is

A separate pipeline from the original SAM2 track in `scripts/`.
Builds the benchmark using COCONut-PanCap as the foundation for rich
region descriptions, paired with WikiArt style references and dual
instruction formats.

The original `scripts/` directory is not modified.
All new code lives in `coconut_pipeline/`.

## Dual instruction format

Each region gets two instructions:

  instruction_text — "Render clear blue sky in impressionist brushstrokes."
  instruction_ref  — "Render clear blue sky using the style of the reference image."

## Two segmentation tracks

Track 1 — Auto Grid:
  SAM2 prompted with 4x4 grid across each image.
  Output: data/coconut_subset/masks_auto/
  JSON: subset_auto_final.json

Track 2 — Label-Guided Clicks:
  SAM2 prompted with COCO instance annotation centroids.
  63/256 regions matched to COCO categories.
  Output: data/coconut_subset/masks_click/
  JSON: subset_click_final.json

## 50-sample subset results (verified)

Property            | Track 1 Auto | Track 2 Click
--------------------|--------------|---------------
Samples             | 50           | 50
Train/Val/Test      | 33/2/15      | 33/2/15
Total regions       | 229          | 256
Mean regions/image  | 4.6          | 5.1
encompassed         | 12           | 37
background_heavy    | 10           | 5
small_object        | 21           | 38
cluttered_scene     | 31           | 36
Missing files       | 0            | 0
Instruction cover   | 229/229      | 256/256

## Corner case tags (5 total)

similar_entities  — >=2 regions share same semantic class
encompassed       — one region >=85% inside another
background_heavy  — largest region >50% of image
small_object      — any region <3% of image area (NEW)
cluttered_scene   — >=5 regions in one image (NEW)

## Scripts

A_download_coconut.py   — download images + parse captions from narrative
A2_merge_masks.py       — merge SAM2 mask paths into COCONut stub
A3_extract_clicks.py    — extract COCO annotation centroids as click prompts
B_build_subset.py       — style pairing + dual instruction generation
C_quality_control.py    — QC, 5-tag corner case tagging, 70/10/20 split
D_export_subset.py      — zip archive for supervisor baseline testing

## Directory structure

data/coconut_subset/
├── images/                        <- COCO train2017 content images
├── masks_auto/                    <- Track 1 SAM2 masks
├── masks_click/                   <- Track 2 SAM2 masks
└── annotations/                   <- tracked in git
    ├── coconut_stub.json
    ├── coconut_stub_merged_auto.json
    ├── coconut_stub_merged_click.json
    ├── clicks_coconut.json
    ├── subset_auto_final.json     <- Track 1 main deliverable
    └── subset_click_final.json    <- Track 2 main deliverable

## Running the pipeline

Step A - Download:
    python3 A_download_coconut.py --output_dir ../data/coconut_subset --num_samples 50

Step A3 - Extract clicks (Track 2):
    python3 A3_extract_clicks.py \
        --stub ../data/coconut_subset/annotations/coconut_stub.json \
        --coco_ann ../data/content_images/annotations/instances_train2017.json \
        --output_clicks ../data/coconut_subset/annotations/clicks_coconut.json

SAM2 Track 1 (auto):
    export CUDA_VISIBLE_DEVICES=1
    python3 ../scripts/01_segment_regions_sam2.py \
        --image_dir ../data/coconut_subset/images \
        --output_dir ../data/coconut_subset/masks_auto \
        --checkpoint ../checkpoints/sam2.1_hiera_large.pt \
        --config configs/sam2.1/sam2.1_hiera_l.yaml \
        --mode auto --min_area 0.02 --max_masks 6 --no_resume

SAM2 Track 2 (click):
    python3 ../scripts/01_segment_regions_sam2.py \
        --image_dir ../data/coconut_subset/images \
        --output_dir ../data/coconut_subset/masks_click \
        --checkpoint ../checkpoints/sam2.1_hiera_large.pt \
        --config configs/sam2.1/sam2.1_hiera_l.yaml \
        --clicks_json ../data/coconut_subset/annotations/clicks_coconut.json \
        --mode click --no_resume

Step A2 - Merge (run for each track):
    python3 A2_merge_masks.py \
        --stub ../data/coconut_subset/annotations/coconut_stub.json \
        --sam2_stub ../data/coconut_subset/annotations/masks_stub_sam2.json \
        --output ../data/coconut_subset/annotations/coconut_stub_merged_auto.json

Step B - Style pairing + instructions:
    python3 B_build_subset.py \
        --stub ../data/coconut_subset/annotations/coconut_stub_merged_auto.json \
        --style_dir ../data/style_references \
        --output_json ../data/coconut_subset/annotations/subset_auto_draft.json \
        --model stub

    For richer instructions (free Gemini API):
    export GEMINI_API_KEY="your-key"  # https://aistudio.google.com/app/apikey
    python3 B_build_subset.py ... --model gemini

Step C - QC and split:
    python3 C_quality_control.py \
        --draft_json ../data/coconut_subset/annotations/subset_auto_draft.json \
        --output_json ../data/coconut_subset/annotations/subset_auto_final.json

Step D - Export:
    python3 D_export_subset.py \
        --annotated_json ../data/coconut_subset/annotations/subset_auto_final.json \
        --output_zip ../coconut_auto_export.zip

Copy to laptop:
    scp "yvs23@ogg.cs.bath.ac.uk:/mnt/vurm/homes/homes/yvs23/Benchmark_dataset/coconut_auto_export.zip" "C:/Users/Yuvan Velkumar/Downloads/"

Delete zips after copying:
    rm ~/Benchmark_dataset/coconut_auto_export.zip
    rm ~/Benchmark_dataset/coconut_click_export.zip

## Scaling to 500 samples

python3 A_download_coconut.py --output_dir ../data/coconut_subset --num_samples 500
Then rerun SAM2, A2, B, C, D with same commands above.
Expected SAM2 time: ~8 min for 500 images.

## Troubleshooting

Step A downloads 0 samples:
    Dataset has only txt/__key__/__url__ fields.
    Script parses narrative text format correctly.

0 regions matched in A3:
    Use instances_train2017.json not instances_val2017.json.
    COCONut images come from train2017.

SAM2 config not found:
    Run from coconut_pipeline/ directory.
    Config path is relative to ~/Benchmark_dataset/.

GitHub push rejected:
    git pull --rebase origin main && git push
