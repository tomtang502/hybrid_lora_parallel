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
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

    export LOGLEVEL=INFO

    export PATH="/ocean/projects/cis250196p/ltang2/.conda/envs/dlora/bin:$PATH"
    : "${CODEDIR:=/jet/home/ltang2/hybrid_lora_parallel}"
    cd "$CODEDIR"
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

: "${DATASET_DIR:=/ocean/projects/cis250196p/ltang2/nemotron_data_sample}"

if [[ -z $CUDA_VISIBLE_DEVICES ]]; then
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    gpu_count=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
echo "Available GPU count: $gpu_count"

if command -v srun &> /dev/null; then
    # on nv cluster
    BATCH_SIZE=2
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

PAR_STRETAGY="ddp"
export WANDB_RUNTYPE=$PAR_STRETAGY

: "${MASTER_ADDR:=$head_node_ip}"
if [ -z "${MASTER_PORT:-}" ]; then
    if command -v python3 >/dev/null 2>&1; then
        MASTER_PORT=$(python3 - <<'PY'
import socket
s=socket.socket()
s.bind(('',0))
print(s.getsockname()[1])
s.close()
PY
)
    else
        MASTER_PORT=$((29500 + RANDOM % 1000))
    fi
fi
export MASTER_ADDR MASTER_PORT

cmd="torchrun --nnodes $SLURM_NNODES --nproc_per_node=$gpu_count --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
src/script/train.py \
    --per_device_batch_size $BATCH_SIZE \
    --num_devices $((SLURM_NNODES * gpu_count)) \
    --chunk_size $CHUNK_SIZE \
    --num_steps $NUM_STEPS \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --parallel_stretagy $PAR_STRETAGY \
    --dataset_dir \"$DATASET_DIR\" \
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