#!/bin/bash
# The editor with a public *.gradio.live link; uses the enhancer if it is up.
source /workspace/multi_turn/scripts/pod/env.sh
export MTEDIT_KV_GPU_GIB=${MTEDIT_KV_GPU_GIB:-16} MTEDIT_ENC_GPU_GIB=${MTEDIT_ENC_GPU_GIB:-3}
ENH=""; curl -sf localhost:30000/health >/dev/null && ENH="--enhancer-url http://127.0.0.1:30000"
curl -sf localhost:30001/health >/dev/null && ENH="$ENH --resolver-url http://127.0.0.1:30001"
export MTEDIT_SAVE_RAW=${MTEDIT_SAVE_RAW:-}; cd /workspace/mt && exec python -m mtedit.server --quant ${MTEDIT_QUANT:-nvfp4} --share --port 8000 --out /workspace/mt/editor ${MTEDIT_MODEL:+--model $MTEDIT_MODEL} ${MTEDIT_STEPS:+--steps $MTEDIT_STEPS} ${MTEDIT_GUIDANCE:+--guidance $MTEDIT_GUIDANCE} $ENH
