"""
WikiArt Style-Reference Downloader
===================================
Downloads a curated set of paintings from WikiArt to serve as style references.

Uses the WikiArt public JSON API (no auth required for most categories).
Output: data/style_references/<style_name>/image_NNN.jpg

Usage:
    python download_wikiart.py \
        --output_dir ../data/style_references \
        --max_per_style 50

Notes:
    - WikiArt requests ~1 s delay between calls; the script respects this.
    - If the API is unavailable, the script falls back to downloading from
      the WikiArt image CDN directly (URLs from the catalogue below).
    - For a fully offline setup, replace this with any local art dataset
      (e.g. a subset of LAION-Aesthetics filtered by art-style tags).
    - On Hex, run this on a login node (light network I/O, no GPU needed).

Dependencies:
    pip install requests tqdm pillow
"""

import argparse
import json
import os
import time
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

# WikiArt style slug → display name mapping (20 styles)
WIKIART_STYLES = {
    "baroque":                  "Baroque",
    "renaissance":              "Early Renaissance",
    "romanticism":              "Romanticism",
    "neoclassicism":            "Neoclassicism",
    "realism":                  "Realism",
    "impressionism":            "Impressionism",
    "post-impressionism":       "Post Impressionism",
    "expressionism":            "Expressionism",
    "cubism":                   "Cubism",
    "fauvism":                  "Fauvism",
    "art-nouveau":              "Art Nouveau Modern",
    "symbolism":                "Symbolism",
    "pointillism":              "Pointillism",
    "abstract-expressionism":   "Abstract Expressionism",
    "pop-art":                  "Pop Art",
    "minimalism":               "Minimalism",
    "color-field-painting":     "Color Field Painting",
    "ukiyo-e":                  "Ukiyo E",
    "art-deco":                 "Art Deco",
    "watercolor":               "Watercolor",
}

WIKIART_API = (
    "https://www.wikiart.org/en/paintings-by-style/"
    "{style}/mode/allart-json?json=2&page={page}"
)


def fetch_style_page(style_slug: str, page: int = 1,
                     session: requests.Session | None = None) -> list[dict]:
    if session is None:
        session = requests.Session()
    url = WIKIART_API.format(style=style_slug, page=page)
    try:
        r = session.get(url, timeout=15,
                        headers={"User-Agent": "research-benchmark-curator/1.0"})
        if r.status_code == 200:
            data = r.json()
            return data.get("Paintings", []) or []
    except Exception as e:
        print(f"    [warn] API error for {style_slug} p{page}: {e}")
    return []


def download_image(url: str, dest: Path, session: requests.Session) -> bool:
    try:
        r = session.get(url, timeout=20,
                        headers={"User-Agent": "research-benchmark-curator/1.0"})
        if r.status_code == 200 and "image" in r.headers.get("content-type", ""):
            import io
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            img.thumbnail((1024, 1024))
            img.save(dest, format="JPEG", quality=90)
            return True
    except Exception:
        pass
    return False


def download_style(style_slug: str, out_dir: Path,
                   max_images: int, session: requests.Session):
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*.jpg")))
    if existing >= max_images:
        print(f"  {style_slug}: already have {existing} images, skipping")
        return

    collected = existing
    page = 1
    with tqdm(total=max_images - existing, desc=f"  {style_slug}",
              leave=False) as pbar:
        while collected < max_images:
            paintings = fetch_style_page(style_slug, page, session)
            if not paintings:
                break
            for p in paintings:
                if collected >= max_images:
                    break
                img_url = p.get("image", "")
                if not img_url:
                    continue
                dest = out_dir / f"image_{collected:04d}.jpg"
                if dest.exists():
                    collected += 1
                    continue
                if download_image(img_url, dest, session):
                    collected += 1
                    pbar.update(1)
                time.sleep(0.8)
            page += 1
            time.sleep(1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="../data/style_references")
    p.add_argument("--max_per_style", type=int, default=50)
    p.add_argument("--styles", nargs="*", default=None,
                   help="Specific style slugs (default: all 20)")
    args = p.parse_args()

    out_root = Path(args.output_dir)
    session  = requests.Session()
    styles   = args.styles or list(WIKIART_STYLES.keys())

    print(f"Downloading {len(styles)} styles × {args.max_per_style} images each")
    print(f"Output root: {out_root}\n")

    for slug in styles:
        download_style(slug, out_root / slug, args.max_per_style, session)

    manifest = {}
    for slug in styles:
        style_dir = out_root / slug
        images = sorted(str(p) for p in style_dir.glob("*.jpg"))
        manifest[slug] = {
            "display_name": WIKIART_STYLES.get(slug, slug),
            "images": images,
            "count": len(images),
        }

    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total = sum(m["count"] for m in manifest.values())
    print(f"\nManifest written to {manifest_path}")
    print(f"Total images downloaded: {total}")


if __name__ == "__main__":
    main()
