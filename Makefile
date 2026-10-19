MAKEFLAGS += --silent
PY ?= python
ARGS ?=

.PHONY: setup editor enhancer resolver test check

setup:
	scripts/setup.sh

editor:
	$(PY) -m multi_turn.app.server $(ARGS)

enhancer:
	$(PY) -m sglang.launch_server --model-path Qwen/Qwen-Image-2.1-PE-I2I --served-model-name pe-i2i \
		--port 30000 --mem-fraction-static 0.26 --context-length 32768 --trust-remote-code

resolver:
	$(PY) -m sglang.launch_server --model-path Qwen/Qwen3-4B-Instruct-2507 --served-model-name resolver \
		--port 30001 --mem-fraction-static 0.11 --context-length 8192

test:
	$(PY) -m pytest -q tests

check:
	$(PY) scripts/gpu_check.py $(ARGS)
