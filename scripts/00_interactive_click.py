"""
Step 0c: Interactive Click Collection — Dynamic Region Selection
================================================================
Gradio app: click on images in your browser to define region seeds.
Each click → one SAM region in Step 1. No fixed limit per image.

Usage — LOCAL (no tunnel needed):
    python 00_interactive_click.py \
        --image_dir ../data/content_images \
        --output_json ../data/annotations/clicks.json \
        --port 7861
    Open: http://localhost:7861

Usage — on ogg (SSH tunnel required):
    # On ogg:
    python 00_interactive_click.py --port 7861
    # On your laptop (new terminal):
    ssh -N -L 7861:localhost:7861 yvs23@ogg.cs.bath.ac.uk
    Open: http://localhost:7861

Saves progress after every navigation click — safe to close and resume.

Dependencies:
    pip install gradio pillow numpy
"""

import argparse
import json
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_clicks(image_path: str, clicks: list[list[int]]) -> Image.Image:
    """Overlay numbered click markers on the image."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    r = max(8, min(img.width, img.height) // 80)
    for i, (x, y) in enumerate(clicks):
        draw.ellipse([x - r, y - r, x + r, y + r],
                     fill=(255, 60, 60), outline=(255, 255, 255), width=2)
        draw.text((x + r + 3, y - r), str(i + 1), fill=(255, 255, 255))
    return img


def load_existing(clicks_path: Path, image_id: str) -> list[list[int]]:
    if clicks_path.exists():
        with open(clicks_path) as f:
            return json.load(f).get(image_id, [])
    return []


def save_clicks(clicks_path: Path, image_id: str,
                clicks: list[list[int]]) -> None:
    data: dict = {}
    if clicks_path.exists():
        with open(clicks_path) as f:
            data = json.load(f)
    data[image_id] = clicks
    clicks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(clicks_path, "w") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# App state
# ─────────────────────────────────────────────────────────────────────────────

class ClickState:
    def __init__(self, image_paths: list[Path], clicks_path: Path):
        self.image_paths = image_paths
        self.clicks_path = clicks_path
        self.index = 0
        self.clicks: list[list[int]] = []

    def current_path(self) -> Path:
        return self.image_paths[self.index]

    def image_id(self) -> str:
        return self.current_path().stem

    def load(self):
        self.clicks = load_existing(self.clicks_path, self.image_id())

    def save(self):
        save_clicks(self.clicks_path, self.image_id(), self.clicks)

    def progress(self) -> str:
        done = 0
        if self.clicks_path.exists():
            with open(self.clicks_path) as f:
                done = len(json.load(f))
        return (f"**Image {self.index + 1} / {len(self.image_paths)}**"
                f" — `{self.image_id()}`"
                f" | Saved: {done}/{len(self.image_paths)}")

    def next(self):
        self.index = min(self.index + 1, len(self.image_paths) - 1)
        self.load()

    def prev(self):
        self.index = max(self.index - 1, 0)
        self.load()

    def skip_to_unannotated(self):
        done: set[str] = set()
        if self.clicks_path.exists():
            with open(self.clicks_path) as f:
                done = set(json.load(f).keys())
        for i, p in enumerate(self.image_paths):
            if p.stem not in done:
                self.index = i
                self.load()
                return


# ─────────────────────────────────────────────────────────────────────────────
# Gradio app
# ─────────────────────────────────────────────────────────────────────────────

def build_app(state: ClickState):

    def render(msg: str = ""):
        img = draw_clicks(str(state.current_path()), state.clicks)
        clicks_str = (
            "\n".join(f"Click {i+1}: ({x}, {y})"
                      for i, (x, y) in enumerate(state.clicks))
            or "(no clicks yet — click on the image)"
        )
        header = state.progress() + (f"\n\n_{msg}_" if msg else "")
        return img, header, clicks_str

    def on_click(evt: gr.SelectData):
        state.clicks.append([int(evt.index[0]), int(evt.index[1])])
        return render()

    def undo():
        if state.clicks:
            state.clicks.pop()
        return render("Undid last click.")

    def clear():
        state.clicks = []
        return render("Cleared all clicks.")

    def save_next():
        state.save()
        state.next()
        return render("Saved ✓")

    def save_prev():
        state.save()
        state.prev()
        return render("Saved ✓")

    def skip():
        state.save()
        state.skip_to_unannotated()
        return render("Jumped to next unannotated image.")

    with gr.Blocks(title="Click Region Collector") as demo:
        gr.Markdown(
            "## Click Region Collector\n"
            "Click on the image to add region seed points. "
            "**Add as many clicks as you need** — each click = one SAM region. "
            "Click **Save & Next** when done with an image."
        )
        with gr.Row():
            with gr.Column(scale=3):
                img_out = gr.Image(label="Click to add region points",
                                   type="pil", interactive=True, height=550)
            with gr.Column(scale=1):
                info    = gr.Markdown()
                clicks  = gr.Textbox(label="Current clicks",
                                     lines=10, interactive=False)
        with gr.Row():
            prev_btn  = gr.Button("◀ Save & Prev")
            undo_btn  = gr.Button("↩ Undo last")
            clear_btn = gr.Button("🗑 Clear all")
            next_btn  = gr.Button("Save & Next ▶", variant="primary")
        skip_btn = gr.Button("⏩ Save & skip to next unannotated")

        outs = [img_out, info, clicks]
        demo.load(fn=lambda: render(), outputs=outs)
        img_out.select(fn=on_click, outputs=outs)
        undo_btn.click(fn=undo, outputs=outs)
        clear_btn.click(fn=clear, outputs=outs)
        next_btn.click(fn=save_next, outputs=outs)
        prev_btn.click(fn=save_prev, outputs=outs)
        skip_btn.click(fn=skip, outputs=outs)

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",   default="../data/content_images")
    p.add_argument("--output_json", default="../data/annotations/clicks.json")
    p.add_argument("--port",        type=int, default=7861)
    args = p.parse_args()

    image_dir = Path(args.image_dir)
    paths: list[Path] = []
    for ext in ["jpg", "jpeg", "png", "JPG", "PNG"]:
        paths.extend(image_dir.glob(f"*.{ext}"))
    paths = sorted(set(paths))

    if not paths:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(paths)} images.")
    print(f"Clicks will be saved to: {args.output_json}")

    clicks_path = Path(args.output_json)
    state = ClickState(paths, clicks_path)
    state.skip_to_unannotated()

    app = build_app(state)
    app.launch(server_port=args.port, server_name="0.0.0.0", share=False)


if __name__ == "__main__":
    main()
