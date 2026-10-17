"""Conversation context for multi-turn edits.

The image model is single-turn: each request sees one image and one instruction,
so "make it red" after "add a sun" means nothing to it. The prefix K/V cache is
attention state, not memory of earlier instructions. A small text LLM
(Qwen3-4B-Instruct on SGLang, ~0.3 s for a short answer) rewrites the new
instruction against the session history into a self-contained one, classifies
the edit, and says which earlier edit's region it refers to, so the session can
reuse that box.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request

SYSTEM = """You turn follow-up image-edit requests into self-contained instructions for a single-turn image editing model that cannot see the conversation.

You get the earlier edits of this image (oldest first, each with an id) and the new request. Answer with one JSON object and nothing else:
{"instruction": "...", "mode": "add|remove|modify|global", "target": <id of the earlier edit the request refers to, or null>}

Rules:
- Resolve references like "it", "that", "them", "the thing", "make it bigger", "move it left" to the concrete object from the history ("make the sun red").
- Keep the user's intent and wording; do not add details they did not ask for. Keep it one sentence.
- mode: "add" puts a new object in; "remove" deletes something; "modify" changes an existing object (color, size, material, pose, expression, style of that object); "global" changes the whole image (lighting, weather, style, background).
- target: the id of the earlier edit whose object the request is about; null if it is about something else or the whole image.
- If the request is already self-contained, return it unchanged with the right mode and target.

Examples:
history: [1] add a sun in the sky
request: make it red
{"instruction": "Make the sun red.", "mode": "modify", "target": 1}

history: [1] add a red mug on the table  [2] make it nighttime
request: make the mug bigger
{"instruction": "Make the red mug bigger.", "mode": "modify", "target": 1}

history: [1] add a cat on the sofa
request: replace the background with a beach
{"instruction": "Replace the background with a beach.", "mode": "global", "target": null}

history: [1] add a hat on the man  [2] add a dog next to him
request: remove it
{"instruction": "Remove the dog.", "mode": "remove", "target": 2}"""


class Resolver:
    def __init__(self, url: str = "http://127.0.0.1:30001", model: str = "resolver", timeout: float = 20.0):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout

    def resolve(self, history: list[tuple[int, str]], request: str) -> dict:
        """history: [(id, instruction)]; returns {"instruction", "mode", "target", "seconds"}."""
        lines = "  ".join(f"[{i}] {text}" for i, text in history) or "(none)"
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"history: {lines}\nrequest: {request}"}],
            "temperature": 0.0, "max_tokens": 120,
        }
        start = time.perf_counter()
        req = urllib.request.Request(f"{self.url}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            text = json.load(response)["choices"][0]["message"]["content"] or ""
        out = {"instruction": request, "mode": "global", "target": None}
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed.get("instruction"), str) and parsed["instruction"].strip():
                    out["instruction"] = parsed["instruction"].strip()
                if parsed.get("mode") in ("add", "remove", "modify", "global"):
                    out["mode"] = parsed["mode"]
                if isinstance(parsed.get("target"), int) and any(i == parsed["target"] for i, _ in history):
                    out["target"] = parsed["target"]
            except json.JSONDecodeError:
                pass
        out["seconds"] = time.perf_counter() - start
        return out
