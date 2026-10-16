"""In-memory output == SGLang's saved PNG for the same request (pixel-exact)."""
import os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mtedit.session import Engine

if __name__ == "__main__":
    e = Engine("Qwen/Qwen-Image-2.1-Turbo", out_dir="/workspace/mt/inmem",
               component_quantizations={"transformer": "convrot_int8"})
    kw = dict(prompt="Add a red scarf.", images=["/workspace/mt/photos/972a_cat.png"], width=1216, height=832, seed=5, mt={})
    a, ra = e.run(name="saved", save=True, **kw)
    b, rb = e.run(name="mem", save=False, **kw)
    print("FRAMES", type(rb.frames), None if not rb.frames else (type(rb.frames[0]), getattr(rb.frames[0], "shape", None), getattr(rb.frames[0], "dtype", None)))
    x, y = np.asarray(a.convert(b.mode)), np.asarray(b)
    print("INMEM_EQUAL", a.mode, b.mode, a.size, b.size, np.array_equal(x, y), float(np.abs(x.astype(int) - y.astype(int)).max()))
    e.shutdown()
