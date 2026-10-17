"""AROUND hooks on the Q21 denoising stage and pipeline config."""

from __future__ import annotations

import logging
import os
import time

import torch

from mtedit import prefill, state
from mtedit.encoding import MASK_EXTRA_KEY, PREFILL_ONLY_KEY, PROGRESSIVE_KEY, MTCaches
from mtedit.request import MTRequest

logger = logging.getLogger(__name__)


# _predict_noise: prefix K/V from the store ---------------------------------------


_calls = 0


def _profiled(fn, *args, **kwargs):
    """MTEDIT_PROFILE_CALLS=12,20 logs the top CUDA kernels of those _predict_noise calls."""
    global _calls
    import os

    index, _calls = _calls, _calls + 1
    wanted = {int(c) for c in os.environ.get("MTEDIT_PROFILE_CALLS", "").split(",") if c}
    if os.environ.get("MTEDIT_TIME") == "1":
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn(*args, **kwargs)
        end.record()
        end.synchronize()
        logger.info("mtedit: call %d gpu %.1f ms", index, start.elapsed_time(end))
        return out
    if index not in wanted:
        return fn(*args, **kwargs)
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = fn(*args, **kwargs)
        torch.cuda.synchronize()
    logger.info("mtedit: profile of _predict_noise call %d\n%s", index,
                prof.key_averages().table(sort_by="cuda_time_total", row_limit=int(os.environ.get("MTEDIT_PROFILE_ROWS", 15)), max_name_column_width=90))
    return out


def predict_noise_around(original, self, current_model, latent_model_input, timestep, target_dtype, guidance, **kwargs):
    return _profiled(_predict_noise_impl, original, self, current_model, latent_model_input, timestep,
                     target_dtype, guidance, **kwargs)


def _predict_noise_impl(original, self, current_model, latent_model_input, timestep, target_dtype, guidance, **kwargs):
    caches = kwargs.get("prefix_caches")
    if not isinstance(caches, MTCaches) or caches.prepared:
        return original(self, current_model, latent_model_input, timestep, target_dtype, guidance, **kwargs)
    caches.prepared = True  # once per branch per request: the first step's call
    model = prefill.unwrap(current_model)
    from mtedit import fp4

    fp4.maybe_convert(model)  # MTEDIT_NVFP4=1: swap the DiT linears to NVFP4 once
    try:
        pending = _fill_from_pool(model, caches, kwargs)
    except Exception:
        logger.exception("mtedit: prefix pool lookup failed; falling back to a full prefill")
        caches.release()
        for sample in range(len(caches)):
            for layer in caches[sample]:
                layer.clear()
        pending = [i for i, v in enumerate(caches.vids) if v is not None]
    out = original(self, current_model, latent_model_input, timestep, target_dtype, guidance, **kwargs)
    try:
        _store_computed(model, caches, pending)
    except Exception:
        logger.exception("mtedit: prefix pool insert failed")
    return out


def _encoder_hidden_states(kwargs):
    ehs = kwargs.get("encoder_hidden_states")
    return ehs[0] if isinstance(ehs, list) else ehs


def _set_caches(caches, sample, kv) -> None:
    for layer in range(kv.shape[0]):
        caches[sample][layer].update(key=kv[layer, 0][None], value=kv[layer, 1][None])


def _fill_from_pool(model, caches: MTCaches, kwargs) -> list[int]:
    """Fill each sample's prefix K/V from the pool; return the samples the DiT must prefill itself."""
    pool = state.dit_pool(model)
    layouts = kwargs["layouts"]
    pending = []
    for sample, vids in enumerate(caches.vids):
        if vids is None:
            continue
        t0 = time.perf_counter()
        total = caches.prefix_lengths[sample]
        match = pool.match(vids)
        m = match.length
        if m == total:
            # full hit: the pool slots themselves (a view when contiguous); held until the request ends
            _set_caches(caches, sample, pool.gather(match.indices))
            caches.matches.append(match)
        elif m >= state.MIN_REUSE:
            kv = torch.empty(pool.layers, 2, total, pool.heads, pool.head_dim, device=pool.kv.device, dtype=pool.kv.dtype)
            kv[:, :, :m] = pool.gather(match.indices)
            prefix = prefill.prefix_inputs(
                model, _encoder_hidden_states(kwargs), layouts[sample], kwargs.get("condition_latents"), sample
            )
            prefill.prefill_from(model, prefix, layouts[sample], kv, m)
            _set_caches(caches, sample, kv)
            slots = pool.alloc(total - m)
            if slots is not None:
                pool.write(slots, kv[:, :, m:])
                pool.insert(vids, torch.cat([match.indices, slots]), owned_from=m)
            match.release()
        else:
            match.release()
            pending.append(sample)  # the DiT's own fused prefill; stored after its first call
            m = 0
        if state.DEBUG:
            logger.info(
                "mtedit: dit sample %d reused %d/%d prefix tokens in %.1f ms (pool %s)",
                sample, m, total, (time.perf_counter() - t0) * 1e3,
                {k: v for k, v in pool.usage().items() if k in ("free", "evictable", "contiguous_views", "gathers")},
            )
    return pending


def _store_computed(model, caches: MTCaches, pending: list[int]) -> None:
    pool = state.dit_pool(model)
    for sample in pending:
        layers = caches[sample]
        if not layers[0]:
            continue
        total = caches.prefix_lengths[sample]
        slots = pool.alloc(total)
        if slots is None:
            continue
        for index, layer in enumerate(layers):
            pool.kv[index, 0].index_copy_(0, slots, layer["key"][0])
            pool.kv[index, 1].index_copy_(0, slots, layer["value"][0])
        pool.insert(caches.vids[sample], slots, owned_from=0)


# input validation: per-request sigma schedule -----------------------------------------


def input_validation_around(original, self, batch, server_args):
    mt = MTRequest.from_batch(batch)
    if mt.sigmas:
        batch.sigmas = list(mt.sigmas)  # upstream derives num_inference_steps from len(sigmas)
    return original(self, batch, server_args)


# _run_denoising_step: latent mask --------------------------------------------------


def _next_sigma(scheduler, step_index: int):
    """Sigma of the sample the scheduler just produced (its step index already advanced).

    Returned as a 0-d tensor on the sigmas' device: ``float()`` on a GPU sigma
    would sync the host every step and stall the launch queue.
    """
    sigmas = scheduler.sigmas
    index = getattr(scheduler, "step_index", None)
    index = step_index + 1 if index is None else index
    return sigmas[index] if index < len(sigmas) else sigmas.new_zeros(())


def _swap_layouts(batch, layouts_by_branch):
    for branch, layouts in layouts_by_branch.items():
        cond = batch.extra.get(branch)
        if cond is not None:
            cond["layouts"][:] = layouts  # in place: the kwargs dict holds this list


def _progressive_step(original, self, ctx, step, batch, server_args, prog):
    """Steps < k at half resolution; at the switch, upsample x0 and re-noise with the full-res noise."""
    import torch.nn.functional as F

    H, W = prog["hw"]
    if step.step_index == 0:
        prog["eps"] = ctx.latents.detach().clone()  # [B, H*W, C] full-res initial noise
        b, _, c = ctx.latents.shape
        ctx.latents = ctx.latents.view(b, H, W, c)[:, ::2, ::2].reshape(b, (H // 2) * (W // 2), c).contiguous()
        prog["full"] = {k: list(batch.extra[k]["layouts"]) for k in prog["low"] if k in batch.extra}
        _swap_layouts(batch, prog["low"])
    before = ctx.latents
    result = original(self, ctx, step, batch, server_args)
    if step.step_index == prog["steps"] - 1:
        sigmas = ctx.scheduler.sigmas
        s0, s1 = sigmas[step.step_index], sigmas[step.step_index + 1]
        v = (ctx.latents - before) / (s1 - s0)
        x0 = ctx.latents - s1 * v
        b, _, c = x0.shape
        x0 = x0.view(b, H // 2, W // 2, c).permute(0, 3, 1, 2)
        x0 = F.interpolate(x0.float(), size=(H, W), mode="bilinear", align_corners=False).to(ctx.latents.dtype)
        x0 = x0.permute(0, 2, 3, 1).reshape(b, H * W, c)
        ctx.latents = (1 - s1) * x0 + s1 * prog["eps"].to(x0)
        _swap_layouts(batch, prog["full"])
    return result


def run_denoising_step_around(original, self, ctx, step, batch, server_args):
    if step.step_index > 0 and batch.extra.get(PREFILL_ONLY_KEY):
        return None  # a warm-up request: the prefix is in the pools after step 0
    prog = batch.extra.get(PROGRESSIVE_KEY)
    if prog and step.step_index < prog["steps"]:
        return _progressive_step(original, self, ctx, step, batch, server_args, prog)
    mask = batch.extra.get(MASK_EXTRA_KEY)
    if mask is None:
        return original(self, ctx, step, batch, server_args)
    if mask.eps is None:
        mask.eps = ctx.latents.detach().clone()  # the request's own initial noise
        if mask.init_from_source:  # SDEdit start: the source noised to the schedule's first sigma
            sigma = ctx.scheduler.sigmas[0]
            ctx.latents = (1 - sigma) * mask.x0.to(ctx.latents) + sigma * mask.eps.to(ctx.latents)
    if state.DEBUG:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    before = ctx.latents.clone() if (mask.lock_sigma > 0 and not mask.locked) else None
    result = original(self, ctx, step, batch, server_args)
    if state.DEBUG:
        torch.cuda.synchronize()
        t1 = time.perf_counter()
    s1 = _next_sigma(ctx.scheduler, step.step_index)
    dump = os.environ.get("MTEDIT_LOCK_DUMP")
    if dump and before is not None:  # calibration: |x0_pred - source| per cell at every step
        import numpy as np
        sh = [float(x) for x in ctx.scheduler.sigmas.tolist()] + [0.0]
        i = step.step_index
        if sh[i + 1] > 0:
            xp = ctx.latents - sh[i + 1] * (ctx.latents - before) / (sh[i + 1] - sh[i])
            h, w = mask.hw
            dd = (xp[:1].float() - mask.x0.to(xp).float()).view(h, w, -1)
            os.makedirs(dump, exist_ok=True)
            np.savez(f"{dump}/{int(time.time() * 1e3) % 10**9:09d}_s{i}.npz", d=dd.cpu().numpy(),
                     inside=mask.inside.view(h, w).cpu().numpy(), editable=mask.editable.view(h, w).cpu().numpy(), sigma=sh[i + 1])
    if before is not None and mask.inside is not None:
        if mask.sigmas_host is None:  # one host sync per request, not per step
            mask.sigmas_host = [float(x) for x in ctx.scheduler.sigmas.tolist()] + [0.0]
        sh, i = mask.sigmas_host, step.step_index
        s0f, s1f = sh[i], sh[i + 1]
        half = 2 * (i + 1) >= len(sh) - 2  # the structure is set; >= 1 step left
        if 0 < s1f <= mask.lock_sigma and half:
            x0_pred = ctx.latents - s1f * (ctx.latents - before) / (s1f - s0f)
            info = mask.lock(x0_pred)
            logger.info("mtedit: change lock at step %d (sigma %.2f): %s", step.step_index, s1f, info)
    ctx.latents = mask.blend(ctx.latents, s1)
    if state.DEBUG:
        torch.cuda.synchronize()
        logger.info(
            "mtedit: mask step %d: step %.1f ms, blend %.2f ms, latents %s %s",
            step.step_index, (t1 - t0) * 1e3, (time.perf_counter() - t1) * 1e3,
            tuple(ctx.latents.shape), ctx.latents.dtype,
        )
    return result


# post_denoising_loop: keep final latents for passthrough ------------------------------


def _compose(spec, crop_latents, crop_grid) -> None:
    """Write a crop edit's latents into a copy of the base's full latents; store under ``spec["as"]``."""
    base = state.OUTPUT_LATENTS.get(spec.get("base"))
    base_grid = state.OUTPUT_GRID.get(spec.get("base"))
    if base is None or base_grid is None:
        logger.warning("mtedit: compose base %s missing; the new version has no stored latents", spec.get("base"))
        return
    x0, y0, x1, y1 = (int(v) for v in spec["box"])
    if (y1 - y0, x1 - x0) != crop_grid:
        logger.warning("mtedit: compose box %s does not match the crop grid %s", spec["box"], crop_grid)
        return
    h, w = base_grid
    full = base.view(1, h, w, -1).clone()
    full[:, y0:y1, x0:x1] = crop_latents.detach().to(full.dtype).view(1, y1 - y0, x1 - x0, -1)
    state.OUTPUT_LATENTS.put(spec["as"], full.view(1, h * w, -1))
    state.OUTPUT_GRID[spec["as"]] = base_grid


def post_denoising_loop_around(original, self, latents, batch):
    mt = MTRequest.from_batch(batch)
    grid = (batch.height // 16, batch.width // 16)
    if mt.store_output_as:
        for index in range(latents.shape[0]):
            name = mt.store_output_as if index == 0 else f"{mt.store_output_as}#{index}"
            state.OUTPUT_LATENTS.put(name, latents[index : index + 1].detach().to(torch.bfloat16).clone())
            state.OUTPUT_GRID[name] = grid
    if mt.compose:
        _compose(mt.compose, latents[:1], grid)
    batch.extra.pop(MASK_EXTRA_KEY, None)
    for branch in ("qwen21_positive", "qwen21_negative"):
        caches = (batch.extra.get(branch) or {}).get("prefix_caches")
        if isinstance(caches, MTCaches):
            caches.release()
    return original(self, latents, batch)
