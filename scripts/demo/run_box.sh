#!/bin/bash
# Box-edit A/B on the pod: Turbo with/without the change lock, Lightning 4 and 8 steps (with lock).
# Usage: scripts/demo/run_box.sh > /workspace/mt/logs/boxcmp.log 2>&1
P=/workspace/qi/bin/python; D=/workspace/multi_turn/scripts/demo; E=/workspace/multi_turn/scripts/pod/editor.sh
LORA=/workspace/loras/lightning/pytorch_lora_weights.safetensors
up() {  # start the editor with the given env, wait until it serves
  pkill -f mtedit.server; sleep 5
  env "$@" setsid nohup $E >> /workspace/mt/logs/editor_cmp.log 2>&1 < /dev/null &
  for i in $(seq 1 200); do curl -sf localhost:8000/api/config >/dev/null && return 0; sleep 3; done; echo "editor failed: $*"; tail -20 /workspace/mt/logs/editor_cmp.log; return 1
}
up MTEDIT_LOCK=0.75 && { [ -f /workspace/mt/boxcmp/src/lamp.png ] || $P $D/box_compare.py --sources; $P $D/box_compare.py --tag a2_turbo_lock; }

up MTEDIT_LOCK=0.75 MTEDIT_MODEL=Qwen/Qwen-Image-2.1 MTEDIT_LORA=$LORA MTEDIT_SIGMAS=1,0.75,0.5,0.25 MTEDIT_STEPS=4 && $P $D/box_compare.py --tag c_light4_lock
up MTEDIT_LOCK=0.75 MTEDIT_MODEL=Qwen/Qwen-Image-2.1 MTEDIT_LORA=$LORA MTEDIT_SIGMAS=1,0.875,0.75,0.625,0.5,0.375,0.25,0.125 MTEDIT_STEPS=8 && $P $D/box_compare.py --tag d_light8_lock
$P $D/box_compare.py --sheet; grep -h "change lock" /workspace/mt/logs/editor_cmp.log | tail -25
echo BOX_DONE
