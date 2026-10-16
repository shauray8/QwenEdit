"""DiT-only low-bit attention: SDPA vs SageAttention2 vs SageAttention3 on edits (same seeds).

  python scripts/attn_check.py --int8 [--sigmas 5]

The text encoder keeps SDPA (Sage3 has no GQA); only the DiT switches, per request,
through SGLang's attention_backend_override.
"""
import argparse, os, sys, time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from general_check import psnr, sheet  # noqa: E402
from steps_check import EDITS, SCHEDULES  # noqa: E402

BACKENDS = [None, "sage_attn", "sage_attn_3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--sigmas", default="8", choices=list(SCHEDULES))
    ap.add_argument("--out", default="/workspace/mt/attn")
    args = ap.parse_args()
    from mtedit.session import EditSession, Engine

    extra = {"component_quantizations": {"transformer": "convrot_int8"}} if args.int8 else {}
    engine = Engine("Qwen/Qwen-Image-2.1-Turbo", out_dir=args.out, **extra)
    sig = SCHEDULES[args.sigmas]
    for backend in BACKENDS:  # warm each backend's kernels
        engine.attention = backend
        w = EditSession(engine, "/workspace/mt/photos/972a_cat.png", session_id=f"w{backend}", crop=False, sigmas=sig)
        w.edit("Make it warmer.", seed=1)
    rows, labels, boxes = [], [], []
    for photo, text, frac in EDITS:
        row, lab, bx, ref = [], [], [], None
        for backend in BACKENDS:
            engine.attention = backend
            s = EditSession(engine, os.path.join("/workspace/mt/photos", photo), crop=False, sigmas=sig,
                            session_id=f"{backend}_{photo[:4]}")
            W, H = s.size
            box = [(frac[0] * W, frac[1] * H, frac[2] * W, frac[3] * H)] if frac else None
            times = []
            for rep in range(2):  # rep 1 is a re-roll of the same edit: steady-state timing
                t = time.perf_counter()
                v = s.edit(text, boxes=box, seed=500, from_version="v000")
                times.append(time.perf_counter() - t)
            s.flush()
            img = Image.open(s.path(v.id))
            ref = ref or img
            q = psnr(img, ref)
            row.append(s.path(v.id)); bx.append(box)
            lab.append(f"{backend or 'sdpa'} {min(times):.2f}s psnr {q:.1f}")
            print(f"[attn] {photo[:4]} {backend or 'sdpa':12s} {min(times):.3f}s psnr_vs_sdpa={q:.1f}", flush=True)
        rows.append(row); labels.append(lab); boxes.append(bx)
    sheet(os.path.join(args.out, f"sheet_attn_{args.sigmas}.png"), rows, boxes, labels)
    engine.shutdown()


if __name__ == "__main__":
    main()
