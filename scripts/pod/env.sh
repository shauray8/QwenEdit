# sourced by the run scripts
source /workspace/qi/bin/activate
export HF_HOME=/workspace/.cache/huggingface FLASHINFER_DISABLE_VERSION_CHECK=1
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
PE_CKPT=$(ls -d $HF_HOME/hub/models--Qwen--Qwen-Image-2.1-PE-I2I/snapshots/* 2>/dev/null | head -1)
