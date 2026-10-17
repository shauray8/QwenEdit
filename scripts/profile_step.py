"""Kernel profile of one steady-state DiT step of a full-frame edit.

  MTEDIT_PROFILE_CALLS=12 MTEDIT_PROFILE_ROWS=30 python scripts/profile_step.py --int8 [--attention sage_attn]
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--int8", action="store_true"); ap.add_argument("--attention")
    args = ap.parse_args()
    from mtedit.session import EditSession, Engine
    extra = {"component_quantizations": {"transformer": "convrot_int8"}} if args.int8 else {}
    engine = Engine("Qwen/Qwen-Image-2.1-Turbo", out_dir="/workspace/mt/prof", attention=args.attention, **extra)
    s = EditSession(engine, "/workspace/mt/photos/972a_cat.png", session_id="prof", crop=False, megapixels=float(os.environ.get("MP", "1")))
    s.edit("Make it warmer.", seed=1)                      # calls 0-7
    s.edit("Add a red scarf.", seed=2, from_version="v000")  # calls 8-15; 12 = a mid step
    engine.shutdown()

if __name__ == "__main__":
    main()
