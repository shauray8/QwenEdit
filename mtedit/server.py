"""HTTP API and a minimal playground for multi-turn edit sessions.

  python -m mtedit.server --model Qwen/Qwen-Image-2.1-Turbo --int8 --port 8000

POST /sessions                      multipart image (+ layout, size) -> session
POST /sessions/{id}/edit            json {instruction, boxes?, seed?, feather_px?, grow_px?, from_version?}
POST /sessions/{id}/reroll          json {seed?}
POST /sessions/{id}/undo
POST /sessions/{id}/checkout        json {version}
GET  /sessions/{id}                 version tree + head
GET  /sessions/{id}/image/{version} PNG
GET  /                              playground (draw boxes, type an instruction, edit / re-roll / undo)

Boxes are in the session image's pixel coordinates. One GPU runs one edit at a
time; requests are serialized by a lock.
"""

from __future__ import annotations

import argparse
import io
import os
import threading
import uuid
from dataclasses import asdict

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from PIL import Image

from mtedit.session import EditSession, Engine

_lock = threading.Lock()
_sessions: dict[str, EditSession] = {}
_engine: Engine | None = None


def _describe(session: EditSession) -> dict:
    return {
        "id": session.id,
        "head": session.head,
        "size": list(session.size),
        "layout": session.layout,
        "versions": {vid: {k: v for k, v in asdict(ver).items() if k != "image_path"} for vid, ver in session.versions.items()},
    }


def create_app():
    app = FastAPI(title="mtedit")

    def get(session_id: str) -> EditSession:
        session = _sessions.get(session_id)
        if session is None:
            raise HTTPException(404, "no such session")
        return session

    @app.post("/sessions")
    async def create(image: UploadFile = File(...), layout: str = Form("chain"), width: int | None = Form(None),
                     height: int | None = Form(None)):
        pil = Image.open(io.BytesIO(await image.read()))
        size = (width, height) if width and height else None
        with _lock:
            session = EditSession(_engine, pil, layout=layout, size=size, session_id=uuid.uuid4().hex[:8])
            _sessions[session.id] = session
        return _describe(session)

    @app.get("/sessions/{session_id}")
    def show(session_id: str):
        return _describe(get(session_id))

    @app.post("/sessions/{session_id}/edit")
    def edit(session_id: str, body: dict = Body(...)):
        session = get(session_id)
        boxes = [tuple(map(float, b)) for b in body.get("boxes") or []] or None
        with _lock:
            version = session.edit(
                body["instruction"], boxes=boxes, seed=body.get("seed"),
                feather_px=float(body.get("feather_px", 16)), grow_px=float(body.get("grow_px", 0)),
                from_version=body.get("from_version"),
            )
        return {"version": version.id, "seconds": version.seconds, **_describe(session)}

    @app.post("/sessions/{session_id}/reroll")
    def reroll(session_id: str, body: dict = Body(default={})):
        session = get(session_id)
        with _lock:
            version = session.reroll(seed=body.get("seed"))
        return {"version": version.id, "seconds": version.seconds, **_describe(session)}

    @app.post("/sessions/{session_id}/undo")
    def undo(session_id: str):
        session = get(session_id)
        session.undo()
        return _describe(session)

    @app.post("/sessions/{session_id}/checkout")
    def checkout(session_id: str, body: dict = Body(...)):
        session = get(session_id)
        try:
            session.checkout(body["version"])
        except KeyError:
            raise HTTPException(404, "no such version")
        return _describe(session)

    @app.get("/sessions/{session_id}/image/{version}")
    def image(session_id: str, version: str):
        session = get(session_id)
        if version not in session.versions:
            raise HTTPException(404, "no such version")
        with open(session.path(version), "rb") as f:
            return Response(f.read(), media_type="image/png")

    @app.get("/", response_class=HTMLResponse)
    def playground():
        with open(os.path.join(os.path.dirname(__file__), "playground.html")) as f:
            return f.read()

    return app


def main():
    global _engine
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen-Image-2.1-Turbo")
    ap.add_argument("--int8", action="store_true", help="online INT8 ConvRot DiT (needs CUDA_HOME at CUDA 13)")
    ap.add_argument("--out", default="mt_out")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    import uvicorn

    extra = {"component_quantizations": {"transformer": "convrot_int8"}} if args.int8 else {}
    _engine = Engine(args.model, out_dir=args.out, **extra)
    try:
        uvicorn.run(create_app(), host=args.host, port=args.port)
    finally:
        for session in _sessions.values():
            session.flush()
        _engine.shutdown()


if __name__ == "__main__":
    main()
