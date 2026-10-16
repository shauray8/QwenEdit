"""Session latency benchmark: run the photo-suite scenarios in a fresh engine process.

Run once per configuration so no cache survives between them, e.g.

  python scripts/bench_sessions.py --tag cached --int8
  MTEDIT_PREFIX_CACHE=0 MTEDIT_ENCODER_CACHE=0 python scripts/bench_sessions.py --tag uncached --int8

Each op's wall time is written to <out>/bench_<tag>.json; the first edit of the
first scene is a warmup and excluded from the totals.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from general_check import SCENES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--photos", default="/workspace/mt/photos")
    ap.add_argument("--out", default="/workspace/mt/bench")
    ap.add_argument("--model", default="Qwen/Qwen-Image-2.1-Turbo")
    ap.add_argument("--int8", action="store_true")
    args = ap.parse_args()

    from mtedit.session import EditSession, Engine

    extra = {"component_quantizations": {"transformer": "convrot_int8"}} if args.int8 else {}
    engine = Engine(args.model, out_dir=args.out, **extra)
    # warm kernels, allocator and pools on a throwaway session
    warm = EditSession(engine, os.path.join(args.photos, "972a_cat.png"), session_id="warm")
    warm.edit("Make the image slightly warmer.", seed=1)
    warm.edit("Add a red ball.", boxes=[(10, 10, 200, 200)], seed=2)

    rows = []
    for name, (photo, steps) in SCENES.items():
        session = EditSession(engine, os.path.join(args.photos, photo), session_id=f"{name}_{args.tag}")
        w, h = session.size
        for index, (op, text, frac) in enumerate(steps):
            boxes = [(a * w, b * h, c * w, d * h) for a, b, c, d in frac] if frac else None
            t = time.perf_counter()
            if op == "edit":
                session.edit(text, boxes=boxes, seed=100 + index)
            elif op == "reroll":
                session.reroll(seed=100 + index)
            else:
                session.undo()
                continue
            timings = session.current.timings
            rows.append(dict(scene=name, op=op, boxed=bool(boxes), seconds=time.perf_counter() - t, timings=timings))
            print(f"[{args.tag}] {name} {op} {rows[-1]['seconds']:.3f}s " +
                  " ".join(f"{k}={v:.3f}" for k, v in timings.items()), flush=True)
    summary = dict(
        tag=args.tag,
        total=sum(r["seconds"] for r in rows),
        mean_edit=sum(r["seconds"] for r in rows if r["op"] == "edit") / max(1, sum(r["op"] == "edit" for r in rows)),
        rerolls=[r["seconds"] for r in rows if r["op"] == "reroll"],
        rows=rows,
    )
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"bench_{args.tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)
    engine.shutdown()


if __name__ == "__main__":
    main()
