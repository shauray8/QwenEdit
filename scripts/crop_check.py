"""Crop edits vs full-frame edits on the photo suite's boxed edits (same seeds).

  python scripts/crop_check.py --int8

Each scene runs as two sessions (crop off / on), both warmed first so uploaded
roots have stored latents. Writes sheet_crop_<scene>.png (top: full frame,
bottom: crop) and crop_report.json with per-edit seconds and crop windows.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from general_check import outside, sheet  # noqa: E402

SCENES = {
    "cat": ("972a_cat.png", [("Add a small red knitted beanie hat on the cat's head.", (0.28, 0.0, 0.72, 0.32)),
                              ("Make the cat's eyes bright blue.", (0.33, 0.28, 0.63, 0.46))]),
    "couch_cats": ("c354_coco_sample.png", [("Add a white ceramic mug of coffee on the couch.", (0.04, 0.62, 0.30, 0.96))]),
    "dog_bench": ("0315_init_image.png", [("Replace the white poodle with a golden retriever sitting on the bench.", (0.30, 0.16, 0.68, 0.98)),
                                          ("Add a red balloon on a string tied to the left end of the bench.", (0.05, 0.02, 0.38, 0.55))]),
    "astronaut": ("2b60_astronaut.jpg", [("Add a small planet Earth in the black sky.", (0.08, 0.0, 0.42, 0.28))]),
    "anime_text": ("a6d2_TI2I_Qwen_Image_Edit_Input.jpg", [('Change the large orange text "SGL" to read "EDIT" in the same style.', (0.58, 0.08, 0.99, 0.46))]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", default="/workspace/mt/photos")
    ap.add_argument("--out", default="/workspace/mt/crop")
    ap.add_argument("--int8", action="store_true")
    args = ap.parse_args()

    from PIL import Image

    from mtedit.session import EditSession, Engine

    extra = {"component_quantizations": {"transformer": "convrot_int8"}} if args.int8 else {}
    engine = Engine("Qwen/Qwen-Image-2.1-Turbo", out_dir=args.out, **extra)
    warm = EditSession(engine, os.path.join(args.photos, "972a_cat.png"), session_id="warmup")
    warm.prewarm(background=False)
    warm.edit("Add a red ball.", boxes=[(20, 20, 220, 220)], seed=1)
    warm.edit("Add a red ball.", boxes=[(20, 20, 220, 220)], seed=1)

    report = {}
    for name, (photo, steps) in SCENES.items():
        runs = {}
        for crop in (False, True):
            s = EditSession(engine, os.path.join(args.photos, photo), crop=crop, session_id=f"{name}_{'crop' if crop else 'full'}")
            s.prewarm(background=False)  # stores the root's latents (and warms its prefix)
            w, h = s.size
            log = []
            for index, (text, frac) in enumerate(steps):
                box = (frac[0] * w, frac[1] * h, frac[2] * w, frac[3] * h)
                parent = s.current
                t = time.perf_counter()
                v = s.edit(text, boxes=[box], seed=300 + index)
                log.append(dict(text=text, seconds=time.perf_counter() - t, crop=v.timings.get("crop"),
                                has_latents=v.latent_id is not None,
                                outside=outside(s.image(), Image.open(s.path(parent.id)), [box])))
            s.flush()
            runs[crop] = (s, log)
        report[name] = {"full": runs[False][1], "crop": runs[True][1]}
        rows = [[runs[c][0].path("v000")] + [runs[c][0].path(f"v{i + 1:03d}") for i in range(len(steps))] for c in (False, True)]
        boxes = [[None] + [runs[c][0].versions[f"v{i + 1:03d}"].boxes for i in range(len(steps))] for c in (False, True)]
        labels = [["full frame"] + [f"{e['seconds']:.2f}s {e['text']}" for e in runs[False][1]],
                  ["crop"] + [f"{e['seconds']:.2f}s crop {e['crop']}" for e in runs[True][1]]]
        sheet(os.path.join(args.out, f"sheet_crop_{name}.png"), rows, boxes, labels)
        print(f"[{name}] full " + " ".join(f"{e['seconds']:.2f}" for e in runs[False][1]) +
              " | crop " + " ".join(f"{e['seconds']:.2f}{'' if e['has_latents'] else '(no-lat)'}" for e in runs[True][1]), flush=True)
    with open(os.path.join(args.out, "crop_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    engine.shutdown()


if __name__ == "__main__":
    main()
