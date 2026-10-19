#!/bin/bash
# Installs SGLang (pinned, with the ConvRot fusion patch), SageAttention 2 and multi_turn into the
# active Python environment. Needs a CUDA toolkit (CUDA_HOME) for the SageAttention build.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC=${SRC:-$ROOT/.deps}
mkdir -p "$SRC"

pip install "huggingface_hub[cli]" fastapi uvicorn python-multipart gradio pytest

[ -d "$SRC/sglang" ] || git clone -q https://github.com/sgl-project/sglang.git "$SRC/sglang"
cd "$SRC/sglang" && git checkout -q e6072935b
git apply --check "$ROOT/patches/convrot_fusions.patch" 2>/dev/null && git apply "$ROOT/patches/convrot_fusions.patch" || true
pip install -e "python[diffusion]"

if ! python -c "import sageattention" 2>/dev/null; then
  [ -d "$SRC/SageAttention" ] || git clone -q https://github.com/thu-ml/SageAttention.git "$SRC/SageAttention"
  cd "$SRC/SageAttention"
  grep -rl "c++17" --include=setup.py . | xargs sed -i.bak 's/c++17/c++20/g'
  pip install setuptools wheel ninja packaging
  pip install --no-build-isolation .
fi

pip install -e "$ROOT" --no-deps
