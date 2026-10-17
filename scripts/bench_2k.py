"""2K edit latency: attention backend x steps (same edit, same seed), one engine, synced wall times.

  python scripts/bench_2k.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def main():
    from mtedit.session import SPEED_PRESETS, EditSession, Engine
    engine = Engine("Qwen/Qwen-Image-2.1-Turbo", out_dir="/workspace/mt/b2k", component_quantizations={"transformer": "convrot_int8"})
    photo = "/workspace/mt/photos/972a_cat.png"
    text = "Replace the green background with a cozy living room with a wooden bookshelf."
    for attention in (None, "sage_attn", "sage_attn_3"):
        engine.attention = attention
        for speed in ("quality", "fast"):
            s = EditSession(engine, photo, megapixels=4.0, crop=False, sigmas=SPEED_PRESETS[speed], session_id=f"{attention}_{speed}")
            s.edit(text, seed=1)  # warm this config
            t = time.perf_counter(); s.edit(text, seed=9, from_version="v000"); dt = time.perf_counter() - t
            s.flush()
            print(f"[2k] {str(attention or 'sdpa'):12s} {speed:8s} {dt:.2f}s  size={s.size}", flush=True)
    engine.shutdown()

if __name__ == "__main__":
    main()
