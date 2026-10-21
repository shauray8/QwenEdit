"""Bria's prompt step for Fibo edits: ``bria_vgl.modify`` turns an instruction and its images into a VGL caption of the
result plus a rewritten instruction, the prompt form Fibo 2.0 was trained on.

The LLM is any litellm model string; Bria's default is Gemini (``GEMINI_API_KEY``). An OpenAI-compatible server works
as ``openai/<model>`` with ``OPENAI_API_BASE``.
"""

from __future__ import annotations

import logging
import os
import time

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


class VGL:
    def __init__(self, llm: str | None = None):
        from bria_vgl import DEFAULT_LLM

        self.llm = llm or DEFAULT_LLM

    @staticmethod
    def available(llm: str | None) -> bool:
        import importlib.util

        if importlib.util.find_spec("bria_vgl") is None:
            return False
        model = llm or "gemini/"
        return not model.startswith("gemini/") or bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

    def rewrite(self, instruction: str, images: list[Image.Image], boxes=None, seed: int | None = None):
        """``(prompt, seconds)``: the VGL JSON for the pipeline, or ``(None, seconds)`` when the LLM call fails."""
        from bria_vgl import modify
        from bria_vgl.dialects.thales.payloads import EditImages, EditImageWithMask
        from bria_vgl.payload import vgl_json

        start = time.perf_counter()
        try:
            if boxes:
                mask = Image.new("L", images[0].size, 0)
                draw = ImageDraw.Draw(mask)
                for box in boxes:
                    draw.rectangle([round(v) for v in box], fill=255)
                payload = EditImageWithMask(image=images[0].convert("RGB"), mask=mask, prompt=instruction)
            else:
                payload = EditImages(images=[im.convert("RGB") for im in images], prompt=instruction)
            prompt = vgl_json(modify(payload, llm=self.llm, seed=seed))
        except Exception as err:
            logger.warning("multi_turn: VGL rewrite failed (%s); sending the plain instruction", err)
            return None, time.perf_counter() - start
        return prompt, time.perf_counter() - start
