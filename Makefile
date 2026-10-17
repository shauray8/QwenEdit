# Pod workflow (RunPod PTY-only SSH; override with RP_HOST=user@ssh.runpod.io).
#   make sync          ship the working tree to the pod and reinstall mtedit
#   make setup         full pod setup (background; `make logs` to follow)
#   make enhancer      start the PE-I2I enhancer        make editor   start the editor (prints the public URL)
#   make url           current public URL               make stop     stop editor + enhancer
#   make test          unit tests locally and on the pod
#   make check | general | bench   GPU validation suites
#   make pull          copy pod outputs (reports, sheets, sessions) to ./pod_out
MAKEFLAGS += --silent
RP := scripts/rp
POD := /workspace/multi_turn/scripts/pod

.PHONY: resolver sync setup logs enhancer editor url stop restart test check general bench pull status

sync:
	scripts/sync.sh

setup: sync
	echo 'setsid nohup $(POD)/setup.sh > /workspace/mt/logs/setup.log 2>&1 < /dev/null & disown; echo started' | $(RP)

logs:
	echo 'tail -n 20 /workspace/mt/logs/setup.log; tail -n 3 /workspace/mt/logs/editor.log 2>/dev/null' | $(RP)

enhancer: sync
	echo 'pkill -f sglang.launch_server; sleep 3; setsid nohup $(POD)/enhancer.sh > /workspace/mt/logs/pe.log 2>&1 < /dev/null & disown; for i in $$(seq 1 100); do curl -sf localhost:30000/health >/dev/null && echo enhancer up && break; sleep 3; done' | $(RP)

resolver: sync
	echo 'pkill -f "served-model-name resolver"; sleep 3; setsid nohup $(POD)/resolver.sh > /workspace/mt/logs/resolver.log 2>&1 < /dev/null & disown; for i in $$(seq 1 100); do curl -sf localhost:30001/health >/dev/null && echo resolver up && break; sleep 3; done' | $(RP)

editor: sync
	echo 'pkill -f mtedit.server; sleep 4; setsid nohup $(POD)/editor.sh > /workspace/mt/logs/editor.log 2>&1 < /dev/null & disown; for i in $$(seq 1 80); do grep -q "Public URL" /workspace/mt/logs/editor.log && break; sleep 3; done; grep "Public URL" /workspace/mt/logs/editor.log' | $(RP)

restart: editor

url:
	echo 'grep "Public URL" /workspace/mt/logs/editor.log' | $(RP)

stop:
	echo 'pkill -f mtedit.server; pkill -f sglang.launch_server; echo stopped' | $(RP)

status:
	echo 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; pgrep -fa "mtedit.server|sglang.launch_server" | cut -c1-90' | $(RP)

test: sync
	python3 -m pytest -q tests || true
	echo 'cd /workspace/multi_turn && /workspace/qi/bin/python -m pytest -q tests 2>&1 | tail -1' | $(RP)

check general bench: sync
	echo 'source $(POD)/env.sh; cd /workspace/multi_turn/scripts && case $@ in check) python gpu_check.py --int8 --image ../chess_test/board_clean.png --out /workspace/mt/check;; general) python general_check.py --int8 --out /workspace/mt/general;; bench) python bench_sessions.py --tag cached --int8;; esac 2>&1 | grep -E "^\[|^\{|Traceback" | tail -30' | $(RP)

pull:
	mkdir -p pod_out
	echo 'cd /workspace/mt && tar czf - --exclude="*.safetensors" check* general* crop steps attn bench* logs editor 2>/dev/null | base64 -w 76' | $(RP) | grep -E '^[A-Za-z0-9+/=]+$$' | base64 -d | tar xzf - -C pod_out && du -sh pod_out

# speed experiments: fresh process per config
speed: sync
	echo 'source $(POD)/env.sh; cd /workspace/multi_turn/scripts && python bench_attn.py 2>&1 | grep -v Warning; for cfg in "base:" "bcg:--bcg"; do t=$${cfg%%:*}; f=$${cfg#*:}; python bench_sessions.py --tag $$t --int8 $$f > /workspace/mt/logs/speed_$$t.log 2>&1; tail -n1 /workspace/mt/logs/speed_$$t.log; done' | $(RP)
