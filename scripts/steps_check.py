"""Turbo sigma subsets: quality vs time on global and boxed edits (full frame, same seeds).

  python scripts/steps_check.py --int8
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from general_check import sheet  # noqa: E402

TURBO = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
SCHEDULES = {
    "8": TURBO,
    "6": [1.0, 0.95418, 0.89508, 0.845148, 0.704534, 0.414568],
    "5": [1.0, 0.926626, 0.845148, 0.704534, 0.414568],
    "4": [1.0, 0.89508, 0.704534, 0.414568],
}
EDITS = [
    ("972a_cat.png", "Replace the green background with a cozy living room with a wooden bookshelf.", None),
    ("a6d2_TI2I_Qwen_Image_Edit_Input.jpg", "Make her hair black, keep the hairstyle.", None),
    ("c354_coco_sample.png", "Add a white ceramic mug of coffee on the couch.", (0.04, 0.62, 0.30, 0.96)),
    ("2b60_astronaut.jpg", "Remove the astronaut and fill the area with matching moon surface and rocks.", (0.38, 0.22, 0.72, 0.86)),
]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--int8", action="store_true"); ap.add_argument("--out", default="/workspace/mt/steps")
    args = ap.parse_args()
    from mtedit.session import EditSession, Engine
    extra = {"component_quantizations": {"transformer": "convrot_int8"}} if args.int8 else {}
    engine = Engine("Qwen/Qwen-Image-2.1-Turbo", out_dir=args.out, **extra)
    w = EditSession(engine, "/workspace/mt/photos/972a_cat.png", session_id="w"); w.edit("Make it warmer.", seed=1)
    rows, labels, boxes = [], [], []
    for photo, text, frac in EDITS:
        row, lab, bx = [], [], []
        for name, sig in SCHEDULES.items():
            s = EditSession(engine, os.path.join("/workspace/mt/photos", photo), crop=False, sigmas=sig, session_id=f"s{name}_{photo[:4]}")
            W, H = s.size
            box = [(frac[0] * W, frac[1] * H, frac[2] * W, frac[3] * H)] if frac else None
            t = time.perf_counter(); v = s.edit(text, boxes=box, seed=500); dt = time.perf_counter() - t
            s.flush(); row.append(s.path(v.id)); lab.append(f"{name} steps {dt:.2f}s"); bx.append(box)
            print(f"[steps] {photo[:4]} {name} {dt:.3f}s", flush=True)
        rows.append(row); labels.append(lab); boxes.append(bx)
    sheet(os.path.join(args.out, "sheet_steps.png"), rows, boxes, labels)
    engine.shutdown()

if __name__ == "__main__":
    main()
