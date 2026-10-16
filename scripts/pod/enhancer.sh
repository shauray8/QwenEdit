#!/bin/bash
# Qwen-Image-2.1-PE-I2I on SGLang's LLM server (~29 GB).
source /workspace/multi_turn/scripts/pod/env.sh
exec python -m sglang.launch_server --model-path "$PE_CKPT" --served-model-name pe-i2i --port 30000 \
  --host 127.0.0.1 --mem-fraction-static 0.26 --context-length 32768 --trust-remote-code
