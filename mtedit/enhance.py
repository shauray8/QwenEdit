"""Prompt enhancement with Qwen's official edit enhancer (Qwen/Qwen-Image-2.1-PE-I2I).

The enhancer is a fine-tuned Qwen3.5-VL 9B trained on its bundled
``system_prompt.txt``; it sees the canvas with each box outlined in its color and
the user's per-box notes, and answers with JSON whose ``positive_prompt`` is a
precise, model-friendly edit instruction. It runs on SGLang's LLM server:

  python -m sglang.launch_server --model-path <PE-I2I snapshot> --served-model-name pe-i2i \
      --port 30000 --mem-fraction-static 0.26 --context-length 32768

The system prompt is identical on every call, so the server's radix cache keeps
its prefill; ``thinking`` follows the model card (on = best quality, slower).
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
import urllib.request

from PIL import Image, ImageDraw

from mtedit.session import MARK_PALETTE


def _balanced_objects(text: str) -> list[str]:
    """Top-level ``{...}`` spans, last first (the answer's JSON follows any prose)."""
    spans, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                spans.append(text[start : i + 1])
    return spans[::-1]


def parse_answer(text: str) -> str | None:
    """``positive_prompt`` from the enhancer's answer (after any ``</think>``), or None."""
    answer = text.split("</think>")[-1]
    for span in _balanced_objects(answer):
        try:
            obj = json.loads(span)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        # the model names the field differently across answers
        for key in ("positive_prompt", "rewritten_prompt", "rewrite", "prompt", "edit_prompt", "instruction"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        strings = [v for k, v in obj.items() if isinstance(v, str) and k not in ("wh_ratio", "ratio_follow")]
        if strings and max(map(len, strings)) > 40:
            return max(strings, key=len).strip()
    return None


def annotated(image: Image.Image, regions, width=6) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for i, (box, _) in enumerate(regions):
        draw.rectangle(tuple(box), outline=MARK_PALETTE[i][1], width=width)
    return out


def user_text(instruction: str, regions) -> str:
    parts = [f"Inside the {MARK_PALETTE[i][0]} box: {text.strip().rstrip('.')}." for i, (_, text) in enumerate(regions)]
    if parts:
        parts.append("Remove all colored box outlines and keep everything outside the boxes unchanged.")
    if instruction.strip():
        parts.append(instruction.strip())
    return " ".join(parts)


class Enhancer:
    def __init__(self, url: str = "http://127.0.0.1:30000", system_prompt_path: str | None = None,
                 model: str = "pe-i2i", thinking: bool = True, max_tokens: int = 8000, timeout: float = 120.0):
        if system_prompt_path is None:
            import glob
            import os

            hits = glob.glob(os.path.expanduser(
                os.environ.get("HF_HOME", "~/.cache/huggingface")
                + "/hub/models--Qwen--Qwen-Image-2.1-PE-I2I/snapshots/*/system_prompt.txt"))
            if not hits:
                raise FileNotFoundError("PE-I2I system_prompt.txt not found; pass system_prompt_path")
            system_prompt_path = hits[0]
        with open(system_prompt_path) as f:
            self.system = f.read()
        self.url, self.model, self.thinking = url.rstrip("/"), model, thinking
        self.max_tokens, self.timeout = max_tokens, timeout

    def enhance(self, image: Image.Image, instruction: str = "", regions=None, thinking: bool | None = None) -> dict:
        """Returns {"prompt", "seconds", "completion_tokens", "raw"}; "prompt" is None on a parse failure."""
        regions = regions or []
        buf = io.BytesIO()
        annotated(image, regions).save(buf, "JPEG", quality=90)
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}},
                    {"type": "text", "text": user_text(instruction, regions)},
                ]},
            ],
            # the model card's edit sampling settings
            "temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.thinking if thinking is None else thinking},
        }
        start = time.perf_counter()
        request = urllib.request.Request(f"{self.url}/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        message = result["choices"][0]["message"]
        raw = (message.get("content") or "")
        return {
            "prompt": parse_answer(raw),
            "seconds": time.perf_counter() - start,
            "completion_tokens": result.get("usage", {}).get("completion_tokens"),
            "raw": raw,
        }
