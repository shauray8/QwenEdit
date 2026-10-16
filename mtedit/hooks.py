"""AROUND hooks on the Q21 denoising stage and pipeline config."""

from __future__ import annotations

import logging
import time

import torch

from mtedit import prefill, state
from mtedit.encoding import MASK_EXTRA_KEY, MTCaches
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
                prof.key_averages().table(sort_by="cuda_time_total", row_limit=15, max_name_column_width=70))
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


def run_denoising_step_around(original, self, ctx, step, batch, server_args):
    mask = batch.extra.get(MASK_EXTRA_KEY)
    if mask is None:
        return original(self, ctx, step, batch, server_args)
    if mask.eps is None:
        mask.eps = ctx.latents.detach().clone()  # the request's own initial noise
    if state.DEBUG:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    result = original(self, ctx, step, batch, server_args)
    if state.DEBUG:
        torch.cuda.synchronize()
        t1 = time.perf_counter()
    ctx.latents = mask.blend(ctx.latents, _next_sigma(ctx.scheduler, step.step_index))
    if state.DEBUG:
        torch.cuda.synchronize()
        logger.info(
            "mtedit: mask step %d: step %.1f ms, blend %.2f ms, latents %s %s",
            step.step_index, (t1 - t0) * 1e3, (time.perf_counter() - t1) * 1e3,
            tuple(ctx.latents.shape), ctx.latents.dtype,
        )
    return result


# post_denoising_loop: keep final latents for passthrough ------------------------------


def post_denoising_loop_around(original, self, latents, batch):
    mt = MTRequest.from_batch(batch)
    if mt.store_output_as:
        for index in range(latents.shape[0]):
            name = mt.store_output_as if index == 0 else f"{mt.store_output_as}#{index}"
            state.OUTPUT_LATENTS.put(name, latents[index : index + 1].detach().to(torch.bfloat16).clone())
    batch.extra.pop(MASK_EXTRA_KEY, None)
    for branch in ("qwen21_positive", "qwen21_negative"):
        caches = (batch.extra.get(branch) or {}).get("prefix_caches")
        if isinstance(caches, MTCaches):
            caches.release()
    return original(self, latents, batch)
