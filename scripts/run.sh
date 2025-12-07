#!/bin/bash
#SBATCH -p GPU-shared                       #partition
#SBATCH -t 08:00:00                         #wall time limit, hr:min:sec (increased for multiple experiments)
#SBATCH -N 1                                #number of nodes
#SBATCH -J cmu15418:hybrid_par              #job name
#SBATCH -o logs/slurm/slurm-%j.out          #SLURM stdout
#SBATCH -e logs/slurm/slurm-%j.err          #SLURM stderr

# Create SLURM log directory
mkdir -p logs/slurm

# Parse command line arguments - now only takes num_gpus
NUM_GPUS_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --num-gpus|-g)
            NUM_GPUS_ARG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            echo "Usage: $0 --num-gpus <count>"
            echo "       $0 -g <count>"
            echo "Available GPU counts: 1, 2, 4"
            exit 1
            ;;
    esac
done

# Check if num_gpus is provided
if [ -z "$NUM_GPUS_ARG" ]; then
    echo "Usage: $0 --num-gpus <count>"
    echo "       $0 -g <count>"
    echo "Available GPU counts: 1, 2, 4"
    exit 1
fi

# Define experiment parameters
STRATEGIES=("ddp" "fsdp" "fsdp_dtensor")
CHUNK_SIZES=(1024 2048 4096 8192 16384)

echo "=========================================="
echo "Starting batch experiments for $NUM_GPUS_ARG GPU(s)"
echo "Total experiments: ${#STRATEGIES[@]} strategies × ${#CHUNK_SIZES[@]} chunk sizes = $((${#STRATEGIES[@]} * ${#CHUNK_SIZES[@]})) experiments"
echo "=========================================="

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

# Determine GPU count (should match NUM_GPUS_ARG)
if [[ -z $CUDA_VISIBLE_DEVICES ]]; then
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    gpu_count=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
echo "Available GPU count: $gpu_count (requested: $NUM_GPUS_ARG)"

# Set training parameters based on environment
if command -v srun &> /dev/null; then
    # on cluster
    NUM_STEPS=100
    GLOBAL_BATCH_SIZE=64
else
    # on local machine
    NUM_STEPS=100
    GLOBAL_BATCH_SIZE=16
fi

: "${MASTER_ADDR:=$head_node_ip}"
export MASTER_ADDR

# Counter for experiments
experiment_count=0
total_experiments=$((${#STRATEGIES[@]} * ${#CHUNK_SIZES[@]}))

# Loop through all strategy and chunk size combinations
for STRAT in "${STRATEGIES[@]}"; do
    for CHUNK_SIZE in "${CHUNK_SIZES[@]}"; do
        experiment_count=$((experiment_count + 1))

        echo ""
        echo "=========================================="
        echo "Experiment $experiment_count/$total_experiments"
        echo "Strategy: $STRAT, Chunk Size: $CHUNK_SIZE, GPUs: $gpu_count"
        echo "=========================================="

        # Set batch size based on strategy (fsdp_dtensor needs smaller batch)
        if command -v srun &> /dev/null; then
            if [[ "$STRAT" == "fsdp_dtensor" ]]; then
                BATCH_SIZE=1
            else
                BATCH_SIZE=2
            fi
        else
            BATCH_SIZE=1
        fi

        # Generate a unique master port for each experiment
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
        export MASTER_PORT

        # Update wandb run type
        export WANDB_RUNTYPE=$STRAT

        # Build command
        cmd="torchrun --nnodes $SLURM_NNODES --nproc_per_node=$gpu_count --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
src/script/train.py \
    --per_device_batch_size $BATCH_SIZE \
    --num_devices $((SLURM_NNODES * gpu_count)) \
    --chunk_size $CHUNK_SIZE \
    --num_steps $NUM_STEPS \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --parallel_stretagy $STRAT \
"

        # Execute command
        if command -v srun &> /dev/null; then
            # on psc cluster
            echo "Running: $cmd"
            srun bash -c "${cmd}"
            exit_code=$?
        else
            # on local machine
            echo "Running: $cmd"
            bash -c "${cmd}"
            exit_code=$?
        fi

        # Check exit status
        if [ $exit_code -eq 0 ]; then
            echo "✓ Experiment $experiment_count completed successfully"
        else
            echo "✗ Experiment $experiment_count failed with exit code $exit_code"
            echo "Continuing with remaining experiments..."
        fi

        # Unset MASTER_PORT for next iteration
        unset MASTER_PORT

        # Small delay between experiments
        sleep 2
    done
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Total: $total_experiments experiments"
echo "Check logs/ directory for results"
echo "=========================================="