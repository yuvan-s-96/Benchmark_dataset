import argparse
import io
import json
import time
from pathlib import Path
from PIL import Image
from datasets import load_dataset


STYLES_WANTED = {
    0:  "abstract-expressionism",
    1:  "art-nouveau",
    2:  "baroque",
    3:  "expressionism",
    4:  "impressionism",
    5:  "post-impressionism",
    6:  "realism",
    7:  "renaissance",
    8:  "romanticism",
    9:  "symbolism",
    10: "cubism",
    11: "fauvism",
    12: "minimalism",
    13: "pointillism",
    14: "pop-art",
    15: "ukiyo-e",
    16: "watercolor",
    17: "art-deco",
    18: "color-field-painting",
    19: "neoclassicism",
}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir",    default="../data/style_references")
    p.add_argument("--max_per_style", type=int, default=50)
    p.add_argument("--styles",        nargs="*", default=None,
                   help="Specific style names to download (default: all 20)")
    args = p.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    wanted = {}
    if args.styles:
        for k, v in STYLES_WANTED.items():
            if v in args.styles:
                wanted[k] = v
    else:
        wanted = STYLES_WANTED

    print(f"Loading WikiArt from HuggingFace...")
    print(f"Downloading {len(wanted)} styles x {args.max_per_style} images each")
    print(f"Output root: {out_root}")

    ds = load_dataset("huggan/wikiart", split="train", streaming=True)

    counts = {v: 0 for v in wanted.values()}

    for item in ds:
        label = item.get("style", item.get("label", -1))
        if label not in wanted:
            continue
        style_name = wanted[label]
        if counts[style_name] >= args.max_per_style:
            continue

        style_dir = out_root / style_name
        style_dir.mkdir(exist_ok=True)

        try:
            img = item["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(io.BytesIO(img)).convert("RGB")
            img = img.convert("RGB")
            img.thumbnail((1024, 1024))
            n = counts[style_name]
            img.save(style_dir / f"image_{n:04d}.jpg", quality=90)
            counts[style_name] += 1
            done  = sum(counts.values())
            total = args.max_per_style * len(wanted)
            print(f"  [{done}/{total}] {style_name}: {counts[style_name]}", end="\r")
        except Exception:
            continue

        if all(v >= args.max_per_style for v in counts.values()):
            break

    print("\n\nFinal counts:")
    for style, count in counts.items():
        print(f"  {style}: {count}")

    # Write manifest
    manifest = {}
    for style_dir in sorted(out_root.iterdir()):
        if not style_dir.is_dir():
            continue
        imgs = sorted(str(p) for p in style_dir.glob("*.jpg"))
        manifest[style_dir.name] = {
            "display_name": style_dir.name,
            "images": imgs,
            "count": len(imgs),
        }
    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    total = sum(v["count"] for v in manifest.values())
    print(f"\nTotal: {total} images downloaded")
    print(f"Manifest: {out_root}/manifest.json")


if __name__ == "__main__":
    main()
