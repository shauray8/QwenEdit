"""Fibo 2.0 Edit latency and GPU memory through EditSession, configured as the editor runs Fibo: NVFP4 (or the given
quant), Sage2 self-attention, no CUDA graphs, the trained ~1 MP size, 2K box edits as a ~1 MP crop.

  python scripts/fibo_bench.py --image photo.jpg [--out out/fibo_bench] [--resolutions 1k,2k] [--repeats 3]

Every edit starts from the session's root, so each one sees the same input; outputs land in --out.
"""

import argparse
import json
import os
import statistics
import subprocess
import threading
import time

from PIL import Image

RESOLUTIONS = {"1k": 1.0, "2k": 4.0}
# (name, instruction, box as fractions of the image or None); the box kinds follow session.fibo_box_kind
EDITS = [
    ("full_night", "Make it night time with warm lamp light.", None),
    ("full_watercolor", "Turn it into a watercolor painting.", None),
    ("box_add", "Add a small potted cactus.", (0.08, 0.55, 0.35, 0.9)),
    ("box_put", "Put a red coffee mug here.", (0.6, 0.6, 0.85, 0.9)),
    ("box_modify", "Make it blue.", (0.6, 0.6, 0.85, 0.9)),
]


class GpuPeak:
    """Peak used GPU memory over a span, sampled with nvidia-smi: the DiT runs in SGLang's worker process."""

    def __init__(self, interval=0.05):
        self.interval, self.peak, self._stop = interval, 0, threading.Event()

    @staticmethod
    def used_mib():
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        return int(out.split()[0])

    def _poll(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.used_mib())
            time.sleep(self.interval)

    def __enter__(self):
        self.peak = self.used_mib()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="out/fibo_bench")
    ap.add_argument("--resolutions", default="1k,2k")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--quant", default="nvfp4", choices=["nvfp4", "bf16"])
    ap.add_argument("--attention", default="sage_attn")
    args = ap.parse_args()
    if args.quant == "nvfp4":
        os.environ["MULTI_TURN_NVFP4"] = "1"  # read at import, as the server sets it

    from multi_turn.app.session import Engine, EditSession

    os.makedirs(args.out, exist_ok=True)
    loaded = GpuPeak.used_mib()
    with GpuPeak() as load_peak:
        engine = Engine("briaai/Fibo-2.0-edit-turbo", out_dir=args.out,
                        component_attention_backends={"transformer": args.attention})
    report = {"quant": args.quant, "attention": args.attention, "gpu_mib_before": loaded,
              "gpu_mib_load_peak": load_peak.peak, "gpu_mib_loaded": GpuPeak.used_mib(), "edits": []}
    print({k: v for k, v in report.items() if k != "edits"}, flush=True)
    src = Image.open(args.image).convert("RGB")
    for res in args.resolutions.split(","):
        session = EditSession(engine, src, megapixels=RESOLUTIONS[res], session_id=f"bench_{res}")
        W, H = session.size
        for name, instruction, frac in EDITS:
            boxes = [(frac[0] * W, frac[1] * H, frac[2] * W, frac[3] * H)] if frac else None
            session.edit(instruction, boxes=boxes, from_version="v000", seed=0)  # warm-up for this shape
            times, generate = [], []
            with GpuPeak() as peak:
                for _ in range(args.repeats):
                    version = session.edit(instruction, boxes=boxes, from_version="v000", seed=0)
                    times.append(version.seconds)
                    generate.append(version.timings["generate"])
            session.image(version.id).save(os.path.join(args.out, f"{res}_{name}.png"))
            row = dict(res=res, size=[W, H], edit=name, crop=version.timings["crop"],
                       seconds_min=round(min(times), 3), seconds_median=round(statistics.median(times), 3),
                       generate_min=round(min(generate), 3), gpu_mib_peak=peak.peak)
            print(row, flush=True)
            report["edits"].append(row)
        session.flush()
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    engine.shutdown()


if __name__ == "__main__":
    main()
