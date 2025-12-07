#!/bin/bash
#SBATCH -A CHANGETHIS                      #account
#SBATCH -p batch_large,batch                #partition
#SBATCH -t 02:00:00                         #wall time limit, hr:min:sec
#SBATCH -N 1                                #number of nodes
#SBATCH -J cmu15418:hybrid_par              #job name
#SBATCH --array=1-6%1
#SBATCH --gpus-per-node 8
#SBATCH --exclusive

if command -v srun &> /dev/null; then
    # on nv cluster
    nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
    nodes_array=($nodes)
    head_node=${nodes_array[0]}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

    export LOGLEVEL=INFO

    export PATH="/ocean/projects/cis250196p/ltang2/.conda/envs/dlora/bin:$PATH"
    CODEDIR="$HOME/codedir"
    cd $CODEDIR
else
    SLURM_NNODES=1
    SLURM_NODEID=0
    SLURM_JOB_ID=0
    head_node_ip=localhost
    : "${CODEDIR:=$(pwd)}"
fi


export OMP_NUM_THREADS=32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_ALLOW_CODE_EVAL=1

export WANDB_ENTITY="spanningtree"
export WANDB_PROJECT="hybrid_lora_parallel"
export WANDB_RESUME=allow




if [[ -z $CUDA_VISIBLE_DEVICES ]]; then
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    gpu_count=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
# gpu_count=1
echo "Available GPU count: $gpu_count"

if command -v srun &> /dev/null; then
    # on nv cluster
    BATCH_SIZE=1
    NUM_STEPS=100
    CHUNK_SIZE=4096
    GLOBAL_BATCH_SIZE=64
else
    # on local machine
    BATCH_SIZE=1
    NUM_STEPS=100
    CHUNK_SIZE=4096
    GLOBAL_BATCH_SIZE=16
fi

PAR_STRETAGY="fsdp_dtensor"
export WANDB_RUNTYPE=$PAR_STRETAGY

cmd="torchrun --nnodes $SLURM_NNODES --nproc_per_node=$gpu_count --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 \
src/script/train.py \
    --per_device_batch_size $BATCH_SIZE \
    --num_devices $((SLURM_NNODES * gpu_count)) \
    --chunk_size $CHUNK_SIZE \
    --num_steps $NUM_STEPS \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --parallel_stretagy $PAR_STRETAGY \
"

if command -v srun &> /dev/null; then
    # on psc cluster
    echo $cmd
    srun bash -c "${cmd}"
else
    # on local machine
    echo $cmd
    bash -c "${cmd}"
fi