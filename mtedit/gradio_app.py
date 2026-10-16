"""Gradio editing workspace for multi-turn Qwen-Image 2.1 edits.

  CUDA_HOME=/usr/local/cuda-13.0 python -m mtedit.gradio_app --int8 --share

Generate a base image (or upload one), draw boxes on the canvas and label each
with what should be there, add an optional global instruction, apply. Every
edit becomes a version; click any version to continue or branch from it.
Box colors are assigned automatically; the model sees each box outlined in its
own color and the prompt names it ("inside the blue box: ...").
"""

from __future__ import annotations

import argparse
import os
import threading
import uuid

import gradio as gr
from gradio_image_annotation import image_annotator

from mtedit.session import MARK_PALETTE, EditSession, Engine

ASPECTS = {
    "1:1": (1024, 1024), "4:3": (1184, 896), "3:4": (896, 1184), "3:2": (1248, 832),
    "2:3": (832, 1248), "16:9": (1344, 768), "9:16": (768, 1344),
}
_engine: Engine | None = None
_sessions: dict[str, EditSession] = {}
_lock = threading.Lock()

CSS = """
#canvas { min-height: 520px; }
.mt-status { font-size: 13px; opacity: .85; }
"""


def _session(state) -> EditSession:
    session = _sessions.get((state or {}).get("sid"))
    if session is None:
        raise gr.Error("Generate or upload an image first.")
    return session


def _new_session(image, settings, root_latent_id=None):
    layout, paste_back, prewarm = settings
    session = EditSession(
        _engine, image, layout=layout, paste_back=paste_back, prewarm=prewarm,
        root_latent_id=root_latent_id, size=image.size if root_latent_id else None,
        session_id=uuid.uuid4().hex[:8],
    )
    _sessions[session.id] = session
    return session


def _view(session: EditSession, note: str = ""):
    """Canvas value, history gallery, status line, download path.

    Files, not PIL images: Gradio encodes a PIL image on every event, while a
    path is copied into its cache once, so a long history stays cheap.
    """
    canvas = {"image": session.path(), "boxes": []}
    history = []
    for vid, version in session.versions.items():
        label = version.instruction or ("; ".join(t for _, t in version.regions) if version.regions else "")
        caption = f"{'▶ ' if vid == session.head else ''}{vid}  {label[:48] or 'base'}"
        history.append((session.path(vid), caption))
    head = session.current
    timing = f" · {head.seconds:.2f} s" if head.seconds else ""
    status = f"**{session.head}**{timing} · {session.size[0]}×{session.size[1]} · {len(session.versions)} versions"
    return canvas, history, f"{status}{' · ' + note if note else ''}", session.path()


def generate(prompt, aspect, seed, layout, paste_back, prewarm, state):
    if not prompt.strip():
        raise gr.Error("Write a prompt to generate an image.")
    width, height = ASPECTS[aspect]
    with _lock:
        image, latent_id = _engine.text_to_image(prompt, width=width, height=height,
                                                 seed=None if seed < 0 else int(seed))
        session = _new_session(image, (layout, paste_back, prewarm), root_latent_id=latent_id)
    return (*_view(session, "generated"), {"sid": session.id})


def upload(image, layout, paste_back, prewarm, state):
    if image is None:
        raise gr.Error("Choose an image to upload.")
    with _lock:
        session = _new_session(image, (layout, paste_back, prewarm))
    return (*_view(session, "uploaded"), {"sid": session.id})


def apply_edit(canvas, instruction, seed, feather, state):
    session = _session(state)
    boxes = (canvas or {}).get("boxes") or []
    instruction = (instruction or "").strip()
    regions = []
    for box in boxes:
        text = (box.get("label") or "").strip() or instruction
        if not text:
            raise gr.Error("Label each box with what should be there, or write a global instruction.")
        regions.append(((box["xmin"], box["ymin"], box["xmax"], box["ymax"]), text))
    if not regions and not instruction:
        raise gr.Error("Describe the edit, or draw and label boxes.")
    if len(regions) > len(MARK_PALETTE):
        raise gr.Error(f"At most {len(MARK_PALETTE)} boxes per edit.")
    labeled = [bool((b.get("label") or "").strip()) for b in boxes]
    seed = None if seed < 0 else int(seed)
    with _lock:
        if not regions:  # a global edit
            session.edit(instruction, seed=seed, feather_px=float(feather))
        elif len(regions) == 1 and not labeled[0]:  # one unlabeled box: a plain boxed edit
            session.edit(instruction, boxes=[regions[0][0]], seed=seed, feather_px=float(feather))
        else:  # per-box instructions; a global instruction is extra only when every box has its own label
            session.edit(instruction if all(labeled) else "", regions=regions, seed=seed, feather_px=float(feather))
    return _view(session)


def reroll(seed, state):
    session = _session(state)
    if session.current.parent is None:
        raise gr.Error("Nothing to re-roll yet.")
    with _lock:
        session.reroll(seed=None if seed < 0 else int(seed))
    return _view(session, "re-rolled")


def undo(state):
    session = _session(state)
    session.undo()
    return _view(session, "undo")


def select_version(evt: gr.SelectData, state):
    session = _session(state)
    vid = list(session.versions)[evt.index]
    session.checkout(vid)
    return _view(session, f"continuing from {vid}")


def clear_boxes(state):
    session = _session(state)
    return {"image": session.path(), "boxes": []}


def build() -> gr.Blocks:
    with gr.Blocks(title="mtedit", css=CSS, theme=gr.themes.Soft()) as demo:
        state = gr.State({})
        gr.Markdown("### mtedit · multi-turn editing on Qwen-Image 2.1")
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=280):
                with gr.Tab("Generate"):
                    t2i_prompt = gr.Textbox(label="Prompt", lines=4, placeholder="A sunlit reading nook with a green armchair…")
                    aspect = gr.Dropdown(list(ASPECTS), value="1:1", label="Aspect")
                    t2i_seed = gr.Number(value=-1, precision=0, label="Seed (-1 random)")
                    gen_btn = gr.Button("Generate", variant="primary")
                with gr.Tab("Upload"):
                    upload_img = gr.Image(type="pil", label="Image", sources=["upload", "clipboard"])
                    upload_btn = gr.Button("Start editing", variant="primary")
                with gr.Accordion("Settings", open=False):
                    layout = gr.Radio(["chain", "anchor"], value="chain", label="Reference layout",
                                      info="anchor also shows the model the original image")
                    paste_back = gr.Checkbox(value=True, label="Keep pixels outside boxes exact")
                    prewarm = gr.Checkbox(value=True, label="Prewarm next turn in the background")
                    feather = gr.Slider(0, 48, value=16, step=4, label="Box edge feather (px)")
                    edit_seed = gr.Number(value=-1, precision=0, label="Edit seed (-1 random)")
            with gr.Column(scale=3, min_width=420):
                canvas = image_annotator(
                    label="Canvas: drag to draw a box, then type what should be there",
                    elem_id="canvas", image_type="pil", sources=[], use_default_label=False,
                    label_list=None, box_min_size=12, handle_size=6, box_thickness=2,
                    show_clear_button=False, show_remove_button=False,
                )
                instruction = gr.Textbox(label="Instruction", lines=2,
                                         placeholder="Global edit, or what to do in unlabeled boxes…")
                with gr.Row():
                    apply_btn = gr.Button("Apply edit", variant="primary", scale=2)
                    reroll_btn = gr.Button("Re-roll")
                    undo_btn = gr.Button("Undo")
                    clear_btn = gr.Button("Clear boxes")
                status = gr.Markdown(elem_classes=["mt-status"])
            with gr.Column(scale=1, min_width=240):
                history = gr.Gallery(label="Versions (click to continue from one)", columns=2, height=640,
                                     allow_preview=False, object_fit="contain")
                download = gr.File(label="Current image")

        view = [canvas, history, status, download]
        settings = [layout, paste_back, prewarm]
        gen_btn.click(generate, [t2i_prompt, aspect, t2i_seed, *settings, state], [*view, state])
        upload_btn.click(upload, [upload_img, *settings, state], [*view, state])
        apply_btn.click(apply_edit, [canvas, instruction, edit_seed, feather, state], view)
        instruction.submit(apply_edit, [canvas, instruction, edit_seed, feather, state], view)
        reroll_btn.click(reroll, [edit_seed, state], view)
        undo_btn.click(undo, [state], view)
        clear_btn.click(clear_boxes, [state], [canvas])
        history.select(select_version, [state], view)
    return demo


def main():
    global _engine
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen-Image-2.1-Turbo")
    ap.add_argument("--int8", action="store_true", help="online INT8 ConvRot DiT (needs CUDA_HOME at CUDA 13)")
    ap.add_argument("--out", default="mt_out")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="public gradio.live link")
    args = ap.parse_args()

    extra = {"component_quantizations": {"transformer": "convrot_int8"}} if args.int8 else {}
    _engine = Engine(args.model, out_dir=args.out, **extra)
    # Gradio's file checks call os.getcwd(); run from the output folder so code syncs can't remove it
    os.chdir(_engine.out_dir)
    try:
        build().queue(default_concurrency_limit=1).launch(
            server_name="0.0.0.0", server_port=args.port, share=args.share, show_error=True,
            allowed_paths=[_engine.out_dir],
        )
    finally:
        for session in _sessions.values():
            session.flush()
        _engine.shutdown()


if __name__ == "__main__":
    main()
