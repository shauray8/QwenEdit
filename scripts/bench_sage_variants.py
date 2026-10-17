"""SageAttention2 variants at Q21 2K/1K edit shapes: time and error vs exact attention."""
import inspect, torch, sageattention as sa
import torch.nn.functional as F
from bench_attn import timed
fn = sa.sageattn_qk_int8_pv_fp8_cuda
print("params:", list(inspect.signature(fn).parameters))
for Lq, Lp in ((16384, 9216), (4096, 4130)):
    H, D, Lk = 32, 128, Lq + Lp
    q = torch.randn(1, H, Lq, D, device="cuda", dtype=torch.bfloat16); k = torch.randn(1, H, Lk, D, device="cuda", dtype=torch.bfloat16); v = torch.randn(1, H, Lk, D, device="cuda", dtype=torch.bfloat16)
    ref = F.scaled_dot_product_attention(q[:, :, :1024], k, v).float()
    err = lambda o: ((o[:, :, :1024].float() - ref).norm() / ref.norm()).item()
    print(f"Lq={Lq} Lk={Lk}: default sageattn {timed(lambda: sa.sageattn(q, k, v, tensor_layout='HND')):.3f}ms err {err(sa.sageattn(q, k, v, tensor_layout='HND')):.4f}", flush=True)
    for kw in (dict(pv_accum_dtype="fp32+fp32"), dict(pv_accum_dtype="fp32+fp16"), dict(pv_accum_dtype="fp32+fp16", qk_quant_gran="per_warp"),
               dict(pv_accum_dtype="fp32+fp16", qk_quant_gran="per_thread"), dict(pv_accum_dtype="fp32+fp16", smooth_v=False)):
        try:
            f = lambda: fn(q, k, v, tensor_layout="HND", **kw)
            print(f"   {kw}: {timed(f):.3f}ms err {err(f()):.4f}", flush=True)
        except Exception as e:
            print(f"   {kw}: x {str(e)[:80]}", flush=True)
