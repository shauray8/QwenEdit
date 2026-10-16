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
    try:
        _fill_from_store(prefill.unwrap(current_model), caches, kwargs)
    except Exception:
        logger.exception("mtedit: prefix store lookup failed; falling back to a full prefill")
        for sample in range(len(caches)):
            caches.reused_from[sample] = 0
            for layer in caches[sample]:
                layer.clear()
    out = original(self, current_model, latent_model_input, timestep, target_dtype, guidance, **kwargs)
    try:
        _insert_into_store(caches)
    except Exception:
        logger.exception("mtedit: prefix store insert failed")
    return out


def _encoder_hidden_states(kwargs):
    ehs = kwargs.get("encoder_hidden_states")
    return ehs[0] if isinstance(ehs, list) else ehs


def _fill_from_store(model, caches: MTCaches, kwargs) -> None:
    layouts = kwargs["layouts"]
    layers, heads, head_dim = prefill.kv_geometry(model)
    device = model.img_in.weight.device
    t0 = time.perf_counter()
    for sample, chain in enumerate(caches.chains):
        if chain is None:
            continue
        nodes = state.STORE.match([key for _, _, key in chain], device)
        total = caches.prefix_lengths[sample]
        if not nodes:
            state.STORE.record(0, total)
            continue  # a full miss: the DiT's own fused prefill is faster
        start = chain[len(nodes) - 1][1]
        kv = torch.empty(layers, 2, total, heads, head_dim, device=device, dtype=nodes[0].kv.dtype)
        for node, (s, e, _) in zip(nodes, chain):
            kv[:, :, s:e].copy_(node.kv, non_blocking=True)
        if start < total:
            prefix = prefill.prefix_inputs(
                model, _encoder_hidden_states(kwargs), layouts[sample], kwargs.get("condition_latents"), sample
            )
            prefill.prefill_from(model, prefix, layouts[sample], kv, start)
        for layer in range(layers):
            caches[sample][layer].update(key=kv[layer, 0][None], value=kv[layer, 1][None])
        caches.reused_from[sample] = len(nodes)
        state.STORE.record(start, total - start)
        if state.DEBUG:
            logger.info(
                "mtedit: sample %d reused %d/%d prefix tokens (%d/%d segments) in %.1f ms",
                sample, start, total, len(nodes), len(chain), (time.perf_counter() - t0) * 1e3,
            )


def _insert_into_store(caches: MTCaches) -> None:
    protect = set()
    for sample, chain in enumerate(caches.chains):
        if chain is None or not caches[sample][0]:
            continue
        keys_ = [layer["key"][0] for layer in caches[sample]]
        values = [layer["value"][0] for layer in caches[sample]]
        parent = None
        for index, (s, e, key) in enumerate(chain):
            node = state.STORE.nodes.get(key)
            if node is None:
                kv = torch.stack(
                    [torch.stack([k[s:e], v[s:e]]) for k, v in zip(keys_, values)]
                )  # a fresh [layers, 2, n, H, D] copy, independent of the request buffers
                node = state.STORE.insert(parent, key, kv)
            parent = node
            protect.add(key)
    state.STORE.enforce_budgets(protect)


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
    return original(self, latents, batch)
