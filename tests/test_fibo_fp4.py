"""Fibo's fused NVFP4 producers against flashinfer.fp4_quantize on the same BF16 values (needs a Blackwell GPU)."""

from types import SimpleNamespace

import pytest
import torch

flashinfer = pytest.importorskip("flashinfer")
pytest.importorskip("sglang.multimodal_gen")
if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
    pytest.skip("NVFP4 needs a Blackwell GPU", allow_module_level=True)

from multi_turn.fibo.dit import _norm_scale, silu_mul  # noqa: E402
from multi_turn.fibo.fp4 import norm_scale_fp4, silu_mul_fp4  # noqa: E402
from multi_turn.pipeline.fp4 import NVFP4Weight  # noqa: E402


def global_scale(y):
    return (448.0 * 6.0 / (y.float().abs().amax() * 4)).to(torch.float32)


def assert_matches_flashinfer(y, fused, gs):
    """Same encoding up to rounding ties, and a GEMM on it as accurate as on flashinfer's."""
    k = y.shape[-1]
    q_ref, sf_ref = flashinfer.fp4_quantize(y.reshape(-1, k), gs)
    assert fused.q.shape == q_ref.shape and fused.sf.numel() == sf_ref.numel()
    assert (fused.q == q_ref).float().mean() > 0.95
    assert (fused.sf.reshape(-1) == sf_ref.reshape(-1)).float().mean() > 0.95  # includes the zero padding rows
    weight = torch.randn(256, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    exact = y.reshape(-1, k).float() @ weight.float().T
    w4 = NVFP4Weight(weight)

    def err(q, sf):
        return ((w4.mm(q, sf.view(sf_ref.shape), gs).float() - exact).norm() / exact.norm()).item()

    assert err(fused.q, fused.sf) < 1.02 * err(q_ref, sf_ref)


def test_norm_scale_fp4_rows_batches_and_token_mask():
    torch.manual_seed(0)
    b, s, k = 2, 150, 3584  # 300 rows: the last 128-row scale tile is partial
    x = torch.randn(b, s, k, device="cuda", dtype=torch.bfloat16) * 3
    norm = SimpleNamespace(weight=torch.rand(k, device="cuda", dtype=torch.bfloat16) + 0.5, eps=1e-5)
    mods = torch.randn(b, 4 * k, device="cuda", dtype=torch.bfloat16) * 0.3
    noisy, clean = mods[:, :k], mods[:, k:2 * k]  # column slices of one [B, 4K] tensor, as adaLN chunks are
    mask = torch.zeros(s, dtype=torch.bool, device="cuda")
    mask[20:90] = True
    y = _norm_scale(x, norm.weight, norm.eps, noisy, clean, mask)
    gs = global_scale(y)
    assert_matches_flashinfer(y, norm_scale_fp4(x, norm, noisy, clean, mask, gs), gs)


def test_silu_mul_fp4_partial_column_block():
    torch.manual_seed(0)
    gate, value = (torch.randn(1, 300, 9536, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    y = silu_mul(gate, value)
    gs = global_scale(y)
    assert_matches_flashinfer(y, silu_mul_fp4(gate, value, gs), gs)
