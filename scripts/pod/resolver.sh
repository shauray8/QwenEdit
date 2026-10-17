#!/bin/bash
# Qwen3-4B-Instruct on SGLang: rewrites follow-up instructions against the edit history (~10 GB).
source /workspace/multi_turn/scripts/pod/env.sh
exec python -m sglang.launch_server --model-path Qwen/Qwen3-4B-Instruct-2507 --served-model-name resolver \
  --port 30001 --host 127.0.0.1 --mem-fraction-static 0.11 --context-length 8192
