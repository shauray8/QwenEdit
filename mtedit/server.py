"""Editor server: the web editor (``web/editor.html``) and its JSON API.

  CUDA_HOME=/usr/local/cuda-13.0 python -m mtedit.server --int8 --share

``--share`` opens a Gradio share tunnel (``*.gradio.live``) to the local port, for
pods with no open ports; nothing else of Gradio is used.

API (boxes are in the session image's pixels):
  GET  /api/config
  POST /api/generate                      {prompt, aspect, seed?}
  POST /api/sessions                      multipart image
  GET  /api/sessions/{id}
  POST /api/sessions/{id}/edit            {instruction?, regions?: [{box, label}], raw_prompt?, seed?,
                                           speed?, crop?, attention?, feather?, from_version?}
  POST /api/sessions/{id}/reroll          {seed?, speed?, attention?}
  POST /api/sessions/{id}/undo | /checkout {version}
  POST /api/sessions/{id}/enhance         {instruction?, regions?}
  GET  /api/sessions/{id}/image/{version}?w=
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import secrets
import threading
import uuid
from dataclasses import asdict

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from PIL import Image

from mtedit.session import MARK_PALETTE, SPEED_PRESETS, EditSession, Engine

logger = logging.getLogger(__name__)

ASPECTS = {
    "1:1": (1024, 1024), "4:3": (1184, 896), "3:4": (896, 1184), "3:2": (1248, 832),
    "2:3": (832, 1248), "16:9": (1344, 768), "9:16": (768, 1344),
}
ATTENTIONS = {"sdpa": None, "sage2": "sage_attn", "sage3": "sage_attn_3"}
# working resolution: the Turbo card edits at 2048x2048 (~4 MP); 1K is fastest
RESOLUTIONS = {"1k": 1.0, "1.5k": 2.25, "2k": 4.0}


def _scaled(size, megapixels):
    w, h = size
    s = (megapixels / ((w * h) / (1024 * 1024))) ** 0.5
    return max(32, round(w * s / 32) * 32), max(32, round(h * s / 32) * 32)

_engine: Engine | None = None
_enhancer = None
_resolver = None
_graphs = False
_sessions: dict[str, EditSession] = {}
_thumbs: dict[tuple, bytes] = {}
_lock = threading.Lock()


def _preview(session: EditSession, version: str, w: int = 1600) -> str:
    """The version as a ~1600 px JPEG data URL, sent with the response so the canvas updates at once."""
    import base64

    image = session.image(version).convert("RGB")
    image.thumbnail((w, w))
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _describe(session: EditSession, preview: bool = False) -> dict:
    versions = {vid: {k: val for k, val in asdict(v).items() if k != "image_path"} for vid, v in session.versions.items()}
    out = {"id": session.id, "head": session.head, "size": list(session.size), "layout": session.layout,
           "versions": versions, "order": list(session.versions)}
    if preview:
        out["preview"] = _preview(session, session.head)
    return out


def _session(session_id: str) -> EditSession:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "no such session")
    return session


def _configure(session: EditSession, body: dict) -> None:
    """Per-request settings (the client is stateless; one generation runs at a time)."""
    speed = body.get("speed", "quality")
    if speed not in SPEED_PRESETS:
        raise HTTPException(400, f"speed must be one of {list(SPEED_PRESETS)}")
    session.sigmas = SPEED_PRESETS[speed]
    attention = body.get("attention", "sage2")
    if attention not in ATTENTIONS:
        raise HTTPException(400, f"attention must be one of {list(ATTENTIONS)}")
    # with CUDA graphs the DiT's attention is fixed server-side (Sage2); per-request overrides are refused
    session.attention = None if _graphs else ATTENTIONS[attention]
    # progressive resolution for unboxed edits at >= 1.5K: the first ~3/8 of the steps at half size
    steps = len(SPEED_PRESETS[speed] or [0] * 8)
    session.progressive = round(3 * steps / 8) if body.get("progressive", True) and \
        session.size[0] * session.size[1] >= 2.0 * 1024 * 1024 else 0
    if "crop" in body:
        session.crop = bool(body["crop"])
    # follow-ups ("make it red") are rewritten against the history unless the client turns it off
    session.resolver = _resolver if body.get("context", True) else None
    if "passthrough" in body:
        session.passthrough = bool(body["passthrough"])
    # unboxed edits read a smaller reference (cuts the K/V the target attends to; crops stay sharp)
    cap = {"full": None, "1.5k": 2.25, "1k": 1.0}.get(body.get("ref", "1.5k"))
    mp = session.size[0] * session.size[1] / (1024 * 1024)
    session.ref_scale = 1.0 if cap is None else min(1.0, cap / mp)
    if "cumulative" in body:
        session.cumulative = bool(body["cumulative"])


def _regions(body: dict):
    """(instruction, boxes, regions) from the request, with the editor's box semantics.

    No boxes: a global edit. One unlabeled box: a plain boxed edit. Otherwise each box is a
    region with its label, or the instruction when unlabeled; the instruction is then extra
    only when every box has its own label.
    """
    instruction = (body.get("instruction") or "").strip()
    raw = body.get("regions") or []
    if len(raw) > len(MARK_PALETTE):
        raise HTTPException(400, f"at most {len(MARK_PALETTE)} boxes per edit")
    if not raw:
        return instruction, None, None
    labeled = [bool((r.get("label") or "").strip()) for r in raw]
    boxes = [tuple(float(x) for x in r["box"]) for r in raw]
    if len(raw) == 1 and not labeled[0]:
        return instruction, boxes, None
    regions = [(b, (r.get("label") or "").strip() or instruction) for b, r in zip(boxes, raw)]
    if any(not text for _, text in regions):
        raise HTTPException(400, "label each box, or write an instruction for the unlabeled ones")
    return (instruction if all(labeled) else ""), None, regions


def _new_session(image: Image.Image, root_latent_id=None, megapixels: float = 1.0) -> EditSession:
    session = EditSession(_engine, image, prewarm=True, root_latent_id=root_latent_id, megapixels=megapixels,
                          size=image.size if root_latent_id else None, session_id=uuid.uuid4().hex[:8])
    _sessions[session.id] = session
    if root_latent_id is None:
        session.prewarm()  # stores the upload's latents (crops, passthrough) and warms its prefix
    return session


def create_app() -> FastAPI:
    app = FastAPI(title="mtedit")

    @app.get("/", response_class=HTMLResponse)
    def editor():
        with open(os.path.join(os.path.dirname(__file__), "web", "editor.html")) as f:
            return f.read()

    @app.get("/api/config")
    def config():
        return {"speeds": list(SPEED_PRESETS), "attentions": list(ATTENTIONS), "aspects": ASPECTS,
                "palette": [{"name": n, "rgb": c} for n, c in MARK_PALETTE], "enhancer": _enhancer is not None, "context": _resolver is not None, "resolutions": list(RESOLUTIONS),
                "model": _engine.model_path}

    @app.post("/api/generate")
    def generate(body: dict = Body(...)):
        prompt = (body.get("prompt") or "").strip()
        if not prompt:
            raise HTTPException(400, "write a prompt")
        mp = RESOLUTIONS.get(body.get("resolution", "1k"), 1.0)
        width, height = _scaled(ASPECTS.get(body.get("aspect", "1:1"), (1024, 1024)), mp)
        seed = body.get("seed")
        with _lock:
            image, latent_id = _engine.text_to_image(prompt, width=width, height=height,
                                                     seed=None if seed in (None, "", -1) else int(seed))
            session = _new_session(image, root_latent_id=latent_id, megapixels=mp)
        return _describe(session, preview=True)

    @app.post("/api/sessions")
    async def upload(image: UploadFile = File(...), resolution: str = "1k"):
        pil = Image.open(io.BytesIO(await image.read()))
        pil.load()
        with _lock:
            session = _new_session(pil, megapixels=RESOLUTIONS.get(resolution, 1.0))
        return _describe(session)

    @app.get("/api/sessions/{session_id}")
    def show(session_id: str):
        return _describe(_session(session_id))

    @app.post("/api/sessions/{session_id}/edit")
    def edit(session_id: str, body: dict = Body(...)):
        session = _session(session_id)
        instruction, boxes, regions = _regions(body)
        raw_prompt = (body.get("raw_prompt") or "").strip() or None
        if not (instruction or boxes or regions or raw_prompt):
            raise HTTPException(400, "describe the edit, or draw and label boxes")
        seed = body.get("seed")
        with _lock:
            _configure(session, body)
            version = session.edit(
                instruction, boxes=boxes, regions=regions, raw_prompt=raw_prompt,
                seed=None if seed in (None, "", -1) else int(seed), feather_px=float(body.get("feather", 16)),
                from_version=body.get("from_version"),
            )
        return {"version": version.id, **_describe(session, preview=True)}

    @app.post("/api/sessions/{session_id}/reroll")
    def reroll(session_id: str, body: dict = Body(default={})):
        session = _session(session_id)
        if session.current.parent is None:
            raise HTTPException(400, "nothing to re-roll yet")
        seed = body.get("seed")
        with _lock:
            _configure(session, body)
            version = session.reroll(seed=None if seed in (None, "", -1) else int(seed))
        return {"version": version.id, **_describe(session, preview=True)}

    @app.post("/api/sessions/{session_id}/undo")
    def undo(session_id: str):
        session = _session(session_id)
        session.undo()
        return _describe(session, preview=True)

    @app.post("/api/sessions/{session_id}/checkout")
    def checkout(session_id: str, body: dict = Body(...)):
        session = _session(session_id)
        try:
            session.checkout(body["version"])
        except KeyError:
            raise HTTPException(404, "no such version")
        return _describe(session, preview=True)

    @app.post("/api/sessions/{session_id}/enhance")
    def enhance(session_id: str, body: dict = Body(default={})):
        if _enhancer is None:
            raise HTTPException(503, "no prompt enhancer configured (start the server with --enhancer-url)")
        session = _session(session_id)
        instruction = (body.get("instruction") or "").strip()
        _, boxes, regions = _regions(body)
        regions = regions or ([(boxes[0], instruction)] if boxes else [])
        result = _enhancer.enhance(session.image(), instruction if not regions else "", regions,
                                   thinking=bool(body.get("thinking", False)))
        if not result["prompt"]:
            raise HTTPException(502, "the enhancer's answer did not parse")
        return {"prompt": result["prompt"], "seconds": result["seconds"]}

    @app.get("/api/sessions/{session_id}/image/{version}")
    def image(session_id: str, version: str, w: int | None = None):
        session = _session(session_id)
        if version not in session.versions:
            raise HTTPException(404, "no such version")
        headers = {"Cache-Control": "max-age=31536000"}
        if not w:
            with open(session.path(version), "rb") as f:
                return Response(f.read(), media_type="image/png", headers=headers)
        key = (session_id, version, w)
        if key not in _thumbs:
            thumb = session.image(version).convert("RGB")
            thumb.thumbnail((w, w))
            buf = io.BytesIO()
            thumb.save(buf, "JPEG", quality=85)
            _thumbs[key] = buf.getvalue()
        return Response(_thumbs[key], media_type="image/jpeg", headers=headers)

    return app


def main():
    global _engine, _enhancer, _resolver, _graphs
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen-Image-2.1-Turbo")
    ap.add_argument("--int8", action="store_true", help="online INT8 ConvRot DiT (needs CUDA_HOME at CUDA 13)")
    ap.add_argument("--out", default="mt_out")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--share", action="store_true", help="public *.gradio.live link via Gradio's share tunnel")
    ap.add_argument("--enhancer-url", default=None, help="SGLang server running Qwen/Qwen-Image-2.1-PE-I2I")
    ap.add_argument("--resolver-url", default=None, help="SGLang server running a small instruct LLM for follow-ups")
    ap.add_argument("--no-graphs", action="store_true", help="disable breakable CUDA graphs for the DiT")
    args = ap.parse_args()

    import uvicorn

    extra = {"component_quantizations": {"transformer": "convrot_int8"}} if args.int8 else {}
    if not args.no_graphs:  # -17% per 1K edit; graphs bake attention in, so Sage2 is set for the DiT only
        _graphs = True
        extra.update(enable_breakable_cuda_graph=True, component_attention_backends={"transformer": "sage_attn"})
    _engine = Engine(args.model, out_dir=args.out, **extra)
    os.chdir(_engine.out_dir)  # never inside the code tree: syncs must not remove the working directory
    if args.enhancer_url:
        from mtedit.enhance import Enhancer

        _enhancer = Enhancer(args.enhancer_url)
    if args.resolver_url:
        from mtedit.context import Resolver

        _resolver = Resolver(args.resolver_url)
    if args.share:
        from gradio.networking import setup_tunnel

        url = setup_tunnel("127.0.0.1", args.port, secrets.token_urlsafe(32), None, None)
        print(f"* Public URL: {url}", flush=True)
    try:
        uvicorn.run(create_app(), host=args.host, port=args.port)
    finally:
        for session in _sessions.values():
            session.flush()
        _engine.shutdown()


if __name__ == "__main__":
    main()
