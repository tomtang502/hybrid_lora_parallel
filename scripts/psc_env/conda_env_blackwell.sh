module load anaconda3
conda create -n dlora python=3.12 -y
conda activate dlora

conda config --set channel_priority strict

# Install CUDA 12.6.1 toolkit from the exact label, with strict channel priority
conda install cuda -c nvidia/label/cuda-12.8.1

# install pacakges, use --override-channels flag if you see InvalidSpec flag
conda install -c conda-forge ipykernel

