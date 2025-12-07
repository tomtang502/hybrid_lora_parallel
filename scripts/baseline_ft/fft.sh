#!/bin/bash
# Minimal conda initialization for non-interactive shells (SLURM-friendly)
__conda_setup="$('/opt/packages/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)" || __conda_setup=""
if [ -n "$__conda_setup" ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/packages/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/opt/packages/anaconda3/etc/profile.d/conda.sh"
    elif command -v conda >/dev/null 2>&1; then
        # fallback: try to initialize using the conda binary found in PATH
        CONDA_BIN=$(command -v conda)
        eval "$("$CONDA_BIN" 'shell.bash' 'hook' 2>/dev/null || true)"
    else
        # last resort: add a common conda location to PATH
        export PATH="/opt/packages/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup

echo "Activating conda environment"
# Allow caller to override the env name or provide a full env path via CONDA_ENV
: "${CONDA_ENV:=dlora}"
conda activate "$CONDA_ENV" || { echo "conda activate failed for $CONDA_ENV"; exit 1; }

export WANDB_ENTITY="spanningtree"
export WANDB_PROJECT="dlora"
export WANDB_RESUME=allow
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM=false

BASE_MODEL_NAME="Qwen/Qwen2.5-1.5B"
: "${DATASET_PATH:=/jet/home/ltang2/hybrid_lora_parallel/part_000000.parquet}"

if [[ -z $CUDA_VISIBLE_DEVICES ]]; then
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    gpu_count=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

echo "Available GPU count: $gpu_count"

# Local defaults (override via env vars before invoking)
BATCH_SIZE=${BATCH_SIZE:-2}
NUM_STEPS=${NUM_STEPS:-100}
CHUNK_SIZE=${CHUNK_SIZE:-4096}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
PAR_STRETAGY=${PAR_STRETAGY:-ddp}
NUM_DEVICES=${NUM_DEVICES:-$gpu_count}
export WANDB_RUNTYPE=${WANDB_RUNTYPE:-$PAR_STRETAGY}

# Ensure torch.distributed rendezvous doesn't fail due to port 29500 already in use.
# Allow caller to override MASTER_ADDR/MASTER_PORT. If not set, pick localhost and a free port.
: "${MASTER_ADDR:=127.0.0.1}"
if [ -z "${MASTER_PORT:-}" ]; then
    if command -v python3 >/dev/null 2>&1; then
        MASTER_PORT=$(python3 - <<'PY'
import socket
s=socket.socket()
s.bind(('127.0.0.1',0))
port=s.getsockname()[1]
s.close()
print(port)
PY
)
    elif command -v python >/dev/null 2>&1; then
        MASTER_PORT=$(python - <<'PY'
import socket
s=socket.socket()
s.bind(('127.0.0.1',0))
port=s.getsockname()[1]
s.close()
print(port)
PY
)
    else
        MASTER_PORT=29501
    fi
fi
export MASTER_ADDR MASTER_PORT
echo "Using MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

# Run with explicit rendezvous endpoint so torchrun uses the chosen port/address.
torchrun --nnodes 1 --nproc_per_node=$gpu_count --rdzv_backend=c10d --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} src/script/train.py \
    --model_name_or_path "$BASE_MODEL_NAME" \
    --per_device_batch_size "$BATCH_SIZE" \
    --num_devices "$NUM_DEVICES" \
    --chunk_size "$CHUNK_SIZE" \
    --num_steps "$NUM_STEPS" \
    --global_batch_size "$GLOBAL_BATCH_SIZE" \
    --parallel_stretagy "$PAR_STRETAGY" \
    --dataset_path "$DATASET_PATH" \
    "$@"
