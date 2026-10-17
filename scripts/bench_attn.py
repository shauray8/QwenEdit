"""Attention microbenchmark at Qwen-Image 2.1 edit shapes (one layer's target attention).

  python scripts/bench_attn.py [--lq 4096 --lp 4130]

Q = target tokens, K/V = [prefix ; target]. Measures:
  * SDPA (flash) vs SageAttention2 (sageattn) vs SageAttention3
  * how much of Sage2 is spent quantizing K/V (the prefix part is fixed for a whole
    request, so it could be quantized once instead of every step x layer)
  * that splitting K/V into prefix + target and merging by log-sum-exp is exact,
    the building block for attention over a cached, pre-quantized prefix
"""

import argparse
import inspect

import torch
import torch.nn.functional as F


def timed(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def sdpa_lse(q, k, v):
    """(out, lse) for [B, H, L, D] via the math definition (reference only)."""
    s = (q.float() @ k.float().transpose(-1, -2)) * q.shape[-1] ** -0.5
    lse = torch.logsumexp(s, -1)
    return (torch.softmax(s, -1) @ v.float()), lse


def merge(o1, l1, o2, l2):
    m = torch.maximum(l1, l2)
    w1, w2 = torch.exp(l1 - m), torch.exp(l2 - m)
    return (o1 * w1[..., None] + o2 * w2[..., None]) / (w1 + w2)[..., None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lq", type=int, default=4096)
    ap.add_argument("--lp", type=int, default=4130)
    ap.add_argument("--heads", type=int, default=32)
    args = ap.parse_args()
    dev, dt = "cuda", torch.bfloat16
    H, D, Lq, Lk = args.heads, 128, args.lq, args.lp + args.lq
    q = torch.randn(1, H, Lq, D, device=dev, dtype=dt)
    k = torch.randn(1, H, Lk, D, device=dev, dtype=dt)
    v = torch.randn(1, H, Lk, D, device=dev, dtype=dt)
    flops = 4 * H * Lq * Lk * D
    report = lambda name, ms: print(f"{name:34s} {ms:7.3f} ms  {flops / ms / 1e9:6.0f} TFLOPS")

    report("sdpa (flash, bf16)", timed(lambda: F.scaled_dot_product_attention(q, k, v)))

    try:
        import sageattention as sa

        print("sageattention exports:", [n for n in dir(sa) if not n.startswith("_")])
        report("sage2 sageattn (HND)", timed(lambda: sa.sageattn(q, k, v, tensor_layout="HND")))
        for name in ("sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp16_cuda"):
            fn = getattr(sa, name, None)
            if fn is not None:
                report(f"sage2 {name}", timed(lambda fn=fn: fn(q, k, v, tensor_layout="HND")))
        # quantization cost alone (what a cached quantized prefix would skip per step x layer)
        try:
            from sageattention import quant

            print("sageattention.quant:", [n for n in dir(quant) if not n.startswith("_")])
            kp = k[:, :, : args.lp]
            if hasattr(quant, "per_warp_int8"):
                sig = inspect.signature(quant.per_warp_int8)
                print("per_warp_int8", sig)
                report("  quant per_warp_int8(q, prefix k)", timed(lambda: quant.per_warp_int8(q, kp, tensor_layout="HND")))
            if hasattr(quant, "per_channel_fp8"):
                print("per_channel_fp8", inspect.signature(quant.per_channel_fp8))
                report("  quant per_channel_fp8(prefix v)", timed(lambda: quant.per_channel_fp8(v[:, :, : args.lp], tensor_layout="HND")))
        except Exception as err:
            print("quant probe failed:", err)
        # LSE support (needed to merge a cached-prefix attention with the target part)
        if "return_lse" in inspect.signature(sa.sageattn).parameters:
            o, lse = sa.sageattn(q, k, v, tensor_layout="HND", return_lse=True)
            print("sage2 return_lse supported, lse", tuple(lse.shape))
        else:
            print("sage2 sageattn has no return_lse")
    except ImportError:
        print("sageattention not installed")

    try:
        import sageattn3

        print("sageattn3 exports:", [n for n in dir(sageattn3) if not n.startswith("_")])
        fn = getattr(sageattn3, "sageattn3_blackwell", None)
        if fn is not None:
            report("sage3 sageattn3_blackwell", timed(lambda: fn(q, k, v)))
    except ImportError:
        print("sageattn3 not installed")

    # exactness of the split: attention over [prefix; target] == LSE-merge of the two parts
    qs, ks, vs = q[:, :2, :256], k[:, :2], v[:, :2]
    full, _ = sdpa_lse(qs, ks, vs)
    o1, l1 = sdpa_lse(qs, ks[:, :, : args.lp], vs[:, :, : args.lp])
    o2, l2 = sdpa_lse(qs, ks[:, :, args.lp :], vs[:, :, args.lp :])
    print(f"split+LSE merge max |diff| = {(merge(o1, l1, o2, l2) - full).abs().max().item():.2e}")


if __name__ == "__main__":
    main()
