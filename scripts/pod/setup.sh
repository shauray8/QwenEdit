#!/bin/bash
# Full pod setup: venv, sglang@e6072935b + ConvRot fusion patch, CUDA 13, SageAttention 2/3,
# Turbo + PE-I2I weights, mtedit. Idempotent; logs to /workspace/mt/logs/setup_*.log.
set -eu
W=/workspace; L=$W/mt/logs; mkdir -p $L
export HF_HOME=$W/.cache/huggingface HF_HUB_ENABLE_HF_TRANSFER=1
step() { echo "== $1 $(date +%T)"; }

step venv
[ -x $W/qi/bin/python ] || { uv venv -q --python 3.12 $W/qi && $W/qi/bin/python -m ensurepip >/dev/null; }
uv pip install -q --python $W/qi/bin/python "huggingface_hub[cli]" hf_transfer pytest fastapi uvicorn python-multipart gradio

step weights
( $W/qi/bin/hf download Qwen/Qwen-Image-2.1-Turbo >$L/setup_dl.log 2>&1 && $W/qi/bin/hf download Qwen/Qwen-Image-2.1-PE-I2I >>$L/setup_dl.log 2>&1; echo "DL_EXIT=$?" >>$L/setup_dl.log ) &

step cuda13
[ -d /usr/local/cuda-13.0 ] || DEBIAN_FRONTEND=noninteractive apt-get install -y -q cuda-toolkit-13-0 >$L/setup_cuda.log 2>&1
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH

step rust
command -v cargo >/dev/null || { curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null; }
source $HOME/.cargo/env

step sglang
[ -d $W/sglang ] || git clone -q https://github.com/sgl-project/sglang.git $W/sglang
cd $W/sglang && git checkout -q e6072935b
git apply --check $W/multi_turn/patches/convrot_fusions.patch 2>/dev/null && git apply $W/multi_turn/patches/convrot_fusions.patch || echo "fusion patch already applied (or conflicts)"
uv pip install --python $W/qi/bin/python -e "python[diffusion]" >$L/setup_sglang.log 2>&1

step sageattention
if ! $W/qi/bin/python -c "import sageattention, sageattn3" 2>/dev/null; then
  export TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=48 EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8"
  rm -rf $W/SageAttention && git clone -q https://github.com/thu-ml/SageAttention.git $W/SageAttention && cd $W/SageAttention
  grep -rl "c++17" --include=setup.py . | xargs sed -i 's/c++17/c++20/g'
  uv pip install -q --python $W/qi/bin/python setuptools wheel ninja packaging
  $W/qi/bin/python -m pip install --no-build-isolation . >$L/setup_sage2.log 2>&1
  cd sageattention3_blackwell && $W/qi/bin/python -m pip install --no-build-isolation . >$L/setup_sage3.log 2>&1
fi

step mtedit
uv pip install -q --python $W/qi/bin/python -e $W/multi_turn --no-deps
wait
step done
