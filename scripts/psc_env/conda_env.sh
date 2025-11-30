module load anaconda3
conda create -n dlora python=3.12 -y
conda activate dlora

conda config --add channels conda-forge
conda config --add channels defaults
conda config --set channel_priority strict
conda config --add channels nvidia/label/cuda-12.6.1
# Install CUDA 12.6.1 toolkit from the exact label, with strict channel priority
conda install cuda -c nvidia/label/cuda-12.6.1

# install pacakges, use --override-channels flag if you see InvalidSpec flag
conda install -c conda-forge ipykernel --override-channels 

