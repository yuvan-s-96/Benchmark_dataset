"""
Step 4: Human Annotation Interface (Gradio)
============================================
Browser UI to review, label, and accept/reject each benchmark sample.
Adapts dynamically to the actual number of regions per image.

Usage — LOCAL:
    python 04_annotate.py \
        --benchmark_json ../data/annotations/benchmark_final.json \
        --output_json    ../data/annotations/benchmark_annotated.json \
        --port 7860
    Open: http://localhost:7860

Usage — on ogg (SSH tunnel required):
    # On ogg:
    python 04_annotate.py --port 7860
    # On your laptop (new terminal):
    ssh -N -L 7860:localhost:7860 yvs23@ogg.cs.bath.ac.uk
    Open: http://localhost:7860

Saves after every navigation click — safe to interrupt and resume.
Resumes from first unannotated sample automatically.

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
# Mask overlay
# ─────────────────────────────────────────────────────────────────────────────

COLOURS = [
    (255, 80,  80,  120), (80,  180, 255, 120), (80,  255, 130, 120),
    (255, 200, 50,  120), (200, 80,  255, 120), (255, 140, 0,   120),
    (0,   210, 210, 120), (255, 100, 160, 120), (160, 255, 80,  120),
    (80,  80,  255, 120),
]


def overlay_masks(content_path: str, regions: list[dict],
                  root: Path) -> Image.Image:
    base    = Image.open(content_path).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    for i, region in enumerate(regions):
        p = root / region["mask_file"]
        if not p.exists():
            continue
        mask   = np.array(Image.open(p).convert("L")) > 127
        colour = COLOURS[i % len(COLOURS)]

        layer        = np.zeros((*base.size[::-1], 4), dtype=np.uint8)
        layer[mask, :3] = colour[:3]
        layer[mask, 3]  = colour[3]
        overlay = Image.alpha_composite(overlay, Image.fromarray(layer))

        ys, xs = np.where(mask)
        if len(xs):
            draw.text((int(xs.mean()) - 8, int(ys.mean()) - 8),
                      str(i + 1), fill=(255, 255, 255, 255))

    return Image.alpha_composite(base, overlay).convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

class AnnotationState:
    def __init__(self, records: list[dict], root: Path, out_path: Path):
        self.records  = records
        self.root     = root
        self.out_path = out_path
        self.index    = 0
        # Resume from first unannotated
        for i, r in enumerate(records):
            if r.get("annotation_status", "pending") == "pending":
                self.index = i
                break

    def save(self):
        with open(self.out_path, "w") as f:
            json.dump(self.records, f, indent=2)

    def next(self): self.index = min(self.index + 1, len(self.records) - 1)
    def prev(self): self.index = max(self.index - 1, 0)

    def progress(self) -> str:
        done = sum(1 for r in self.records
                   if r.get("annotation_status", "pending") != "pending")
        return f"{done}/{len(self.records)} annotated"


# ─────────────────────────────────────────────────────────────────────────────
# Gradio app
# ─────────────────────────────────────────────────────────────────────────────

def build_app(state: AnnotationState):

    def load_sample(idx: int):
        r = state.records[idx]
        vis = overlay_masks(str(state.root / r["image_file"]),
                            r["regions"], state.root)
        n = len(r["regions"])
        header = (f"**Sample {idx+1}/{len(state.records)}** "
                  f"— `{r['image_id']}` | {n} regions | {state.progress()}")
        labels = "\n".join(
            f"Region {i+1}: {reg.get('region_label', '')}"
            for i, reg in enumerate(r["regions"])
        )
        instrs = "\n".join(
            f"Region {i+1}: {reg.get('instruction', '')}"
            for i, reg in enumerate(r["regions"])
        )
        composite = r.get("composite_instruction", "")
        status    = r.get("annotation_status", "pending")
        tags      = ", ".join(r.get("corner_case_tags", [])) or "none"
        return vis, header, labels, instrs, composite, status, tags

    def save_navigate(direction, labels_txt, instrs_txt, composite, status):
        r = state.records[state.index]

        label_lines = [l.split(":", 1)[-1].strip()
                       for l in labels_txt.splitlines() if l.strip()]
        instr_lines = [l.split(":", 1)[-1].strip()
                       for l in instrs_txt.splitlines() if l.strip()]

        for i, reg in enumerate(r["regions"]):
            if i < len(label_lines): reg["region_label"] = label_lines[i]
            if i < len(instr_lines): reg["instruction"]  = instr_lines[i]

        r["composite_instruction"] = composite
        r["annotation_status"]     = status
        state.save()

        if direction == "next": state.next()
        else:                   state.prev()
        return load_sample(state.index)

    with gr.Blocks(title="Regional Stylisation Annotator") as demo:
        gr.Markdown(
            "## Regional Stylisation Benchmark Annotator\n"
            "Each image has a **variable number of regions** — "
            "one line per region in the text boxes below."
        )
        with gr.Row():
            vis_img = gr.Image(label="Content + masks", type="pil", height=520)
            with gr.Column():
                info_md      = gr.Markdown()
                tags_md      = gr.Markdown(label="Corner-case tags")
                labels_box   = gr.Textbox(
                    label="Region labels  (Region N: <label>)", lines=8)
                instrs_box   = gr.Textbox(
                    label="Region instructions  (Region N: <instruction>)", lines=8)
                composite_box = gr.Textbox(
                    label="Composite instruction", lines=3)
                status_dd    = gr.Dropdown(
                    choices=["pending", "approved", "rejected"],
                    value="pending", label="Status")

        with gr.Row():
            prev_btn = gr.Button("◀ Previous")
            next_btn = gr.Button("Next ▶", variant="primary")

        outs = [vis_img, info_md, labels_box, instrs_box,
                composite_box, status_dd, tags_md]

        demo.load(fn=lambda: load_sample(state.index), outputs=outs)
        prev_btn.click(
            fn=lambda lb, ib, cb, st: save_navigate("prev", lb, ib, cb, st),
            inputs=[labels_box, instrs_box, composite_box, status_dd],
            outputs=outs)
        next_btn.click(
            fn=lambda lb, ib, cb, st: save_navigate("next", lb, ib, cb, st),
            inputs=[labels_box, instrs_box, composite_box, status_dd],
            outputs=outs)

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark_json",
                   default="../data/annotations/benchmark_final.json")
    p.add_argument("--output_json",
                   default="../data/annotations/benchmark_annotated.json")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    with open(args.benchmark_json) as f:
        records = json.load(f)

    root     = Path(args.benchmark_json).parent.parent
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(records)} samples.")
    print(f"\nSSH tunnel (if on ogg):")
    print(f"  ssh -N -L {args.port}:localhost:{args.port} yvs23@ogg.cs.bath.ac.uk")
    print(f"Then open: http://localhost:{args.port}\n")

    state = AnnotationState(records, root, out_path)
    build_app(state).launch(server_port=args.port,
                            server_name="0.0.0.0", share=False)


if __name__ == "__main__":
    main()
