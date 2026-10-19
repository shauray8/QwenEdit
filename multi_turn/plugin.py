from __future__ import annotations

import ast
import hashlib
import inspect
import logging
import textwrap

logger = logging.getLogger(__name__)

STAGES = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.qwen_image21"
CONFIG = "sglang.multimodal_gen.configs.pipeline_configs.qwen_image21"

# upstream QwenImage21EncodingStage.forward that pipeline.encoding copies (sglang e6072935b)
ENCODING_FORWARD_FINGERPRINT = "a1daf90471ae9c5d"
_fingerprint_checked = False


def _fingerprint(fn) -> str | None:
    try:
        source = textwrap.dedent(inspect.getsource(fn))
        return hashlib.sha256(ast.dump(ast.parse(source)).encode()).hexdigest()[:16]
    except (OSError, TypeError, SyntaxError):
        return None


def _encoding_forward(original, self, batch, server_args):
    global _fingerprint_checked
    if not _fingerprint_checked:
        _fingerprint_checked = True
        found = _fingerprint(original)
        if found != ENCODING_FORWARD_FINGERPRINT:
            logger.warning(
                "multi_turn: upstream QwenImage21EncodingStage.forward changed (fingerprint %s, expected %s); "
                "multi_turn's copy may be missing upstream fixes",
                found,
                ENCODING_FORWARD_FINGERPRINT,
            )
    from multi_turn.pipeline.encoding import encoding_forward

    return encoding_forward(self, batch, server_args)


def _predict_noise(original, *args, **kwargs):
    from multi_turn.pipeline.hooks import predict_noise_around

    return predict_noise_around(original, *args, **kwargs)


def _input_validation(original, *args, **kwargs):
    from multi_turn.pipeline.hooks import input_validation_around

    return input_validation_around(original, *args, **kwargs)


def _run_denoising_step(original, *args, **kwargs):
    from multi_turn.pipeline.hooks import run_denoising_step_around

    return run_denoising_step_around(original, *args, **kwargs)


def _post_denoising_loop(original, *args, **kwargs):
    from multi_turn.pipeline.hooks import post_denoising_loop_around

    return post_denoising_loop_around(original, *args, **kwargs)


def register() -> None:
    from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

    HookRegistry.register(f"{STAGES}.QwenImage21EncodingStage.forward", _encoding_forward, HookType.AROUND)
    HookRegistry.register(f"{STAGES}.QwenImage21DenoisingStage._predict_noise", _predict_noise, HookType.AROUND)
    HookRegistry.register(f"{STAGES}.QwenImage21InputValidationStage.forward", _input_validation, HookType.AROUND)
    HookRegistry.register(
        f"{STAGES}.QwenImage21DenoisingStage._run_denoising_step", _run_denoising_step, HookType.AROUND
    )
    HookRegistry.register(
        f"{CONFIG}.QwenImage21PipelineConfig.post_denoising_loop", _post_denoising_loop, HookType.AROUND
    )
    logger.info("multi_turn: registered Qwen-Image 2.1 multi-turn hooks")
