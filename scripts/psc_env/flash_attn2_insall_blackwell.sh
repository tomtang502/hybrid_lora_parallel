git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
export TORCH_CUDA_ARCH_LIST="12.0;12.0+ptx"
# export TORCH_CUDA_ARCH_LIST="9.0;9.0+PTX" for h100
export MAX_JOBS=16
python setup.py bdist_wheel
pip install dist/flash_attn-*.whl