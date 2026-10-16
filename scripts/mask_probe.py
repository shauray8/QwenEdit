"""Masked vs unmasked step time on the same prompt (store warm), for MTEDIT_DEBUG timing logs."""

import os
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    from mtedit.session import Engine

    out = "/workspace/mt/probe"
    engine = Engine("Qwen/Qwen-Image-2.1-Turbo", out_dir=out)
    src = os.path.join(out, "src.png")
    Image.open("chess_test/board_clean.png").convert("RGB").resize((1024, 1024)).save(src)
    prompt = "Move the white pawn from e2 to e4. Keep everything else unchanged."
    mask = {"boxes": [[512, 768, 640, 896], [512, 512, 640, 640]], "canvas_ref": 0}
    for i, mt in enumerate([{}, {}, {"mask": mask}, {}, {"mask": mask}, {}, {"mask": mask}]):
        t = time.perf_counter()
        engine.run(prompt=prompt, images=[src], width=1024, height=1024, seed=3, mt=mt, name=f"probe{i}")
        print(f"[probe{i}] mask={'mask' in mt} {time.perf_counter() - t:.3f}s", flush=True)
    engine.shutdown()


if __name__ == "__main__":
    main()
