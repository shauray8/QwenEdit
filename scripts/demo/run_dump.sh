#!/bin/bash
P=/workspace/qi/bin/python; D=/workspace/multi_turn/scripts/demo
pkill -f mtedit.server; sleep 5; rm -rf /workspace/mt/dump
MTEDIT_LOCK=0.0001 MTEDIT_LOCK_DUMP=/workspace/mt/dump setsid nohup /workspace/multi_turn/scripts/pod/editor.sh >> /workspace/mt/logs/editor_cmp.log 2>&1 < /dev/null &
for i in $(seq 1 200); do curl -sf localhost:8000/api/config >/dev/null && break; sleep 3; done
$P $D/box_compare.py --tag z_dump; echo DUMP_DONE
