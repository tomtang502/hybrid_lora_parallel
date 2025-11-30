# Clean any wheel that pulled a too-new glibc
# pip uninstall -y flash-attn flash_attn

# Tooling
pip install -U pip setuptools wheel ninja cmake packaging

export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"

#   A100=80, 3090/3080=86, 4090=89, H100=90
export TORCH_CUDA_ARCH_LIST="80;86;89;90"
# For newer blackwell gpu arch
export TORCH_CUDA_ARCH_LIST="12.0;12.0+ptx"
export MAX_JOBS="$(nproc)"

# Build from source (links against your cluster's glibc)
pip install --no-binary=flash-attn flash-attn --no-build-isolation
# python test/test_fa2.py --B 4 --H 32 --N 2048 --D 64 --dtype bf16 --causal