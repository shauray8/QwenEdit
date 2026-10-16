"""Next-turn latency with and without background prefix warming (unboxed chain edits).

  python scripts/bench_warm.py --int8

Each turn: edit, then 1.5 s of "user think time" (during which the warm-up runs),
then the next edit is timed. Turn 1 is excluded (its root image may be cached).
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

EDITS = [
    "Make it nighttime with soft moonlight.",
    "Add light snowfall.",
    "Turn the scene into a watercolor painting.",
    "Give the cat small round sunglasses.",
    "Make the colors warmer and more saturated.",
]


def run(engine, photo, prewarm, seed0, think):
    from mtedit.session import EditSession

    s = EditSession(engine, photo, prewarm=prewarm, session_id=f"warm{int(prewarm)}")
    times = []
    for i, text in enumerate(EDITS):
        t = time.perf_counter()
        s.edit(text, seed=seed0 + i)
        times.append(time.perf_counter() - t)
        time.sleep(think)
    if s._warm is not None:
        s._warm.result()
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", default="/workspace/mt/photos/972a_cat.png")
    ap.add_argument("--out", default="/workspace/mt/bench_warm")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--think", type=float, default=1.5)
    args = ap.parse_args()

    from mtedit.session import Engine

    extra = {"component_quantizations": {"transformer": "convrot_int8"}} if args.int8 else {}
    engine = Engine("Qwen/Qwen-Image-2.1-Turbo", out_dir=args.out, **extra)
    run(engine, args.photo, False, 1, 0.0)  # warm kernels and pools
    cold = run(engine, args.photo, False, 100, args.think)
    warm = run(engine, args.photo, True, 200, args.think)
    out = dict(no_prewarm=cold, prewarm=warm,
               mean_no_prewarm=sum(cold[1:]) / (len(cold) - 1), mean_prewarm=sum(warm[1:]) / (len(warm) - 1))
    print("WARM_RESULT " + json.dumps(out), flush=True)
    engine.shutdown()


if __name__ == "__main__":
    main()
