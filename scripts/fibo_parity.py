"""Fibo DiT parity: the SGLang port against diffusers on one forward with the same inputs.

  python scripts/fibo_parity.py dump --out out/fibo_parity.pt      # in an env with Bria's diffusers branch
  python scripts/fibo_parity.py check --ref out/fibo_parity.pt     # in the multi_turn env
"""

import argparse

import torch

MODEL = "briaai/Fibo-2.0-edit-turbo"
H, W, CH, CW, T = 48, 64, 48, 64, 300  # latent tokens of target and context, text length


def patchify(x):  # [B, 32, 2h, 2w] -> [B, h*w, 128], features (c, dy, dx)
    b, c, hh, ww = x.shape
    x = x.view(b, c, hh // 2, 2, ww // 2, 2).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, (hh // 2) * (ww // 2), c * 4)


def dump(out):
    from diffusers import BriaFibo2Transformer2DModel

    model = BriaFibo2Transformer2DModel.from_pretrained(MODEL, subfolder="transformer", torch_dtype=torch.bfloat16)
    model = model.cuda().eval()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 32, 2 * H, 2 * W, generator=g).bfloat16()
    ctx = torch.randn(1, 32, 2 * CH, 2 * CW, generator=g).bfloat16()
    text = torch.randn(1, 6, T, 7680, generator=g).bfloat16()
    t = torch.tensor([0.75])
    with torch.no_grad():
        y = model(hidden_states=x.cuda(), timestep=t.cuda(), encoder_hidden_states=text.cuda(),
                  context_latents=[ctx.cuda()], return_dict=False)[0]
    torch.save(dict(x=patchify(x), ctx=patchify(ctx), text=text, t=t, y=patchify(y.float().cpu())), out)
    print("saved", out)


def check(ref):
    from sglang.multimodal_gen.runtime.distributed import maybe_init_distributed_environment_and_model_parallel
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    import glob
    import json
    import os

    from multi_turn.fibo.dit import FiboDitConfig, FiboTransformer2DModel

    from sglang.multimodal_gen.runtime.server_args.server_args import ServerArgs, set_global_server_args

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29511", RANK="0", WORLD_SIZE="1", LOCAL_RANK="0")
    set_global_server_args(ServerArgs.from_kwargs(model_path=MODEL))
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    path = snapshot_download(MODEL, allow_patterns=["transformer/*"])
    hf_config = json.load(open(os.path.join(path, "transformer/config.json")))
    with torch.device("cuda"):
        model = FiboTransformer2DModel(FiboDitConfig(), hf_config).to(torch.bfloat16)
    state = {}
    for f in glob.glob(os.path.join(path, "transformer/*.safetensors")):
        state.update(load_file(f))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("missing", missing[:5], len(missing), "unexpected", unexpected[:5], len(unexpected))
    r = torch.load(ref)
    from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context

    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        y = model(r["x"].cuda(), r["text"].cuda(), r["t"].cuda() * 1000, latent_hw=(H, W),
                  context_tokens=r["ctx"].cuda(), context_hw=[(CH, CW)])
    ref_y = r["y"].cuda()
    err = (y - ref_y).abs()
    cos = torch.nn.functional.cosine_similarity(y.flatten(), ref_y.flatten(), dim=0)
    print(f"max {err.max():.4f} mean {err.mean():.5f} ref_mean_abs {ref_y.abs().mean():.4f} cos {cos:.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dump", "check"])
    ap.add_argument("--out", default="out/fibo_parity.pt")
    ap.add_argument("--ref", default="out/fibo_parity.pt")
    a = ap.parse_args()
    dump(a.out) if a.mode == "dump" else check(a.ref)
