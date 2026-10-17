"""Re-run a saved session's edit (same image, boxes, prompt, seed) on the current code; side-by-side sheet.

  python scripts/demo/rerun_last.py <old session json> <old session dir> [version] [--out /tmp/rv.jpg]
"""

import io
import json
import sys
import urllib.request
import uuid

from PIL import Image

API = "http://127.0.0.1:8000"


def main():
    meta, folder = sys.argv[1], sys.argv[2]
    vid = sys.argv[3] if len(sys.argv) > 3 else None
    old = json.load(open(meta))
    vid = vid or old["head"]
    v = old["versions"][vid]
    parent = v["parent"]
    img = open(f"{folder}/{parent}.png", "rb").read()
    b = "----" + uuid.uuid4().hex
    body = (f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="a.png"\r\n'
            f"Content-Type: image/png\r\n\r\n").encode() + img + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(f"{API}/api/sessions?resolution=2k", data=body,
                                 headers={"Content-Type": "multipart/form-data; boundary=" + b})
    sid = json.load(urllib.request.urlopen(req, timeout=600))["id"]
    if v.get("regions"):
        regions, instruction = [{"box": r[0], "label": r[1]} for r in v["regions"]], ""
    else:
        regions, instruction = [{"box": bx, "label": ""} for bx in (v.get("boxes") or [])], v.get("request") or v.get("instruction") or ""
    edit = json.dumps({"instruction": instruction, "regions": regions, "seed": v["seed"]}).encode()
    out = json.load(urllib.request.urlopen(urllib.request.Request(f"{API}/api/sessions/{sid}/edit", data=edit,
                                                                  headers={"Content-Type": "application/json"}), timeout=600))
    nv = out["versions"][out["head"]]
    print("seconds", round(nv["seconds"], 2), "boxes", nv["boxes"])
    res = Image.open(io.BytesIO(urllib.request.urlopen(f"{API}/api/sessions/{sid}/image/{out['head']}").read())).convert("RGB")
    a = Image.open(f"{folder}/{parent}.png").convert("RGB")
    prev = Image.open(f"{folder}/{vid}.png").convert("RGB")
    sheet = Image.new("RGB", (a.width * 3 + 20, a.height), "white")
    for i, im in enumerate((a, prev, res)):
        sheet.paste(im, (i * (a.width + 10), 0))
    sheet.thumbnail((2000, 2000))
    sheet.save("/tmp/rv.jpg", quality=90)


if __name__ == "__main__":
    main()
