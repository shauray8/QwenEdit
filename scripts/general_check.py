"""General multi-turn edit suite on real photos: add / remove / replace / global / text edits,
re-roll and undo, each version checked against an uncached reference session (same seeds).

  python scripts/general_check.py --photos /workspace/mt/photos --out /workspace/mt/general [--int8]

Boxes are fractions of the image (x0, y0, x1, y1). Writes one contact sheet per scene
(top row: cached session, bottom row: uncached reference) and report.json.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SCENES = {
    "cat": ("972a_cat.png", [
        ("edit", "Add a small red knitted beanie hat on the cat's head.", [(0.28, 0.0, 0.72, 0.32)]),
        ("edit", "Replace the green background with a cozy living room with a wooden bookshelf.", None),
        ("reroll", None, None),
        ("undo", None, None),
        ("edit", "Make the cat's eyes bright blue.", [(0.33, 0.28, 0.63, 0.46)]),
    ]),
    "couch_cats": ("c354_coco_sample.png", [
        ("edit", "Remove the two remote controls; show the pink couch fabric where they were.",
         [(0.0, 0.04, 0.32, 0.27), (0.49, 0.08, 0.60, 0.48)]),
        ("edit", "Add a white ceramic mug of coffee on the couch.", [(0.04, 0.62, 0.30, 0.96)]),
    ]),
    "dog_bench": ("0315_init_image.png", [
        ("edit", "Replace the white poodle with a golden retriever sitting on the bench.", [(0.30, 0.16, 0.68, 0.98)]),
        ("edit", "Add a red balloon on a string tied to the left end of the bench.", [(0.05, 0.02, 0.38, 0.55)]),
    ]),
    "astronaut": ("2b60_astronaut.jpg", [
        ("edit", "Remove the astronaut and fill the area with matching moon surface and rocks.", [(0.38, 0.22, 0.72, 0.86)]),
        ("edit", "Add a small planet Earth in the black sky.", [(0.08, 0.0, 0.42, 0.28)]),
    ]),
    "anime_text": ("a6d2_TI2I_Qwen_Image_Edit_Input.jpg", [
        ("edit", 'Change the large orange text "SGL" to read "EDIT" in the same style.', [(0.58, 0.08, 0.99, 0.46)]),
        ("edit", "Make her hair black, keep the hairstyle.", None),
    ]),
}


def psnr(a, b):
    x = np.asarray(a.convert("RGB"), np.float64)
    y = np.asarray(b.convert("RGB"), np.float64)
    mse = ((x - y) ** 2).mean()
    return float("inf") if mse == 0 else float(10 * np.log10(255**2 / mse))


def outside(out, src, boxes, margin=24):
    x = np.asarray(out.convert("RGB"), np.float64)
    y = np.asarray(src.convert("RGB"), np.float64)
    keep = np.ones(x.shape[:2], bool)
    for x0, y0, x1, y1 in boxes:
        keep[max(0, int(y0) - margin) : int(y1) + margin, max(0, int(x0) - margin) : int(x1) + margin] = False
    d = np.abs(x - y)[keep]
    return float(d.mean()), float(d.max())


def run_scene(engine, name, photo, steps, seed0, cache):
    from mtedit.session import EditSession

    session = EditSession(engine, photo, cache=cache, session_id=f"{name}_{'c' if cache else 'r'}")
    w, h = session.size
    log = []
    for index, (op, text, frac) in enumerate(steps):
        boxes = [(a * w, b * h, c * w, d * h) for a, b, c, d in frac] if frac else None
        seed = seed0 + index
        parent = session.current
        t = time.perf_counter()
        if op == "edit":
            version = session.edit(text, boxes=boxes, seed=seed)
        elif op == "reroll":
            version = session.reroll(seed=seed)
        else:
            version = session.undo()
        entry = dict(op=op, text=text, version=version.id, seconds=time.perf_counter() - t, path=session.path(version.id))
        if op != "undo" and version.boxes:
            entry["outside_mean_max"] = outside(session.image(), Image.open(session.path(parent.id)), version.boxes)
        log.append(entry)
    return session, log


def sheet(path, rows, boxes_rows, labels):
    thumbs = []
    for row, boxes in zip(rows, boxes_rows):
        line = []
        for image_path, bx in zip(row, boxes):
            im = Image.open(image_path).convert("RGB")
            scale = 320 / im.width
            t = im.resize((320, int(im.height * scale)))
            d = ImageDraw.Draw(t)
            for x0, y0, x1, y1 in bx or []:
                d.rectangle((x0 * scale, y0 * scale, x1 * scale, y1 * scale), outline=(255, 0, 255), width=2)
            line.append(t)
        thumbs.append(line)
    cols = max(len(r) for r in thumbs)
    th = max(t.height for r in thumbs for t in r)
    out = Image.new("RGB", (cols * 330, len(thumbs) * (th + 22)), "white")
    d = ImageDraw.Draw(out)
    for r, line in enumerate(thumbs):
        for c, t in enumerate(line):
            out.paste(t, (c * 330, r * (th + 22) + 20))
            d.text((c * 330 + 2, r * (th + 22) + 4), labels[r][c][:52], fill=(0, 0, 0))
    out.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", default="/workspace/mt/photos")
    ap.add_argument("--out", default="/workspace/mt/general")
    ap.add_argument("--model", default="Qwen/Qwen-Image-2.1-Turbo")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--scenes", default=",".join(SCENES))
    args = ap.parse_args()

    from mtedit.session import Engine

    extra = {"component_quantizations": {"transformer": "convrot_int8"}} if args.int8 else {}
    engine = Engine(args.model, out_dir=args.out, **extra)
    report = {}
    for name in args.scenes.split(","):
        photo, steps = SCENES[name]
        photo = os.path.join(args.photos, photo)
        cached, clog = run_scene(engine, name, photo, steps, 100, cache=True)
        ref, rlog = run_scene(engine, name, photo, steps, 100, cache=False)
        for c, r in zip(clog, rlog):
            c["psnr_vs_uncached"] = psnr(Image.open(c["path"]), Image.open(r["path"]))
            c["identical_to_uncached"] = c["psnr_vs_uncached"] == float("inf")
            c["uncached_seconds"] = r["seconds"]
        report[name] = clog
        cached.flush(); ref.flush()
        rows = [[cached.versions["v000"].image_path] + [e["path"] for e in clog],
                [ref.versions["v000"].image_path] + [e["path"] for e in rlog]]
        boxes = [[None] + [cached.versions[e["version"]].boxes for e in clog]] * 2
        labels = [["original"] + [f"{e['op']}: {e['text'] or ''}" for e in clog],
                  ["uncached ref"] + [f"psnr {e['psnr_vs_uncached']:.1f}" for e in clog]]
        sheet(os.path.join(args.out, f"sheet_{name}.png"), rows, boxes, labels)
        print(f"[{name}] " + json.dumps([{k: v for k, v in e.items() if k != 'path'} for e in clog]), flush=True)
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    engine.shutdown()


if __name__ == "__main__":
    main()
