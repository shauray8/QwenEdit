#!/bin/bash
# Fresh-pod setup for multi-turn edit work: venv, sglang@e6072935b (matches pro6000 report), Q21 + Turbo weights.
set -u
W=/workspace; L=$W/mt/logs; mkdir -p $L $W/mt/refs
export HF_HOME=$W/.cache/huggingface HF_HUB_ENABLE_HF_TRANSFER=1
(
  uv venv -q --python 3.12 $W/qi && $W/qi/bin/python -m ensurepip >/dev/null 2>&1
  uv pip install -q --python $W/qi/bin/python "huggingface_hub[cli]" hf_transfer
  $W/qi/bin/hf download Qwen/Qwen-Image-2.1-Turbo && echo TURBO_DONE
  $W/qi/bin/hf download Qwen/Qwen-Image-2.1 && echo BASE_DONE
  echo DL_DONE
) >$L/dl.log 2>&1 &
(
  cd $W && git clone -q https://github.com/sgl-project/sglang.git && cd sglang && git checkout -q e6072935b
  while [ ! -x $W/qi/bin/python ]; do sleep 2; done
  uv pip install --python $W/qi/bin/python -e "python[diffusion]"
  echo "EXIT=$?"
) >$L/install.log 2>&1 &
wait
echo SETUP_DONE >>$L/install.log
