export WANDB_ENTITY="spanningtree"
export WANDB_PROJECT="dlora"
export WANDB_RESUME=allow

BASE_MODEL_NAME="Qwen/Qwen2.5-1.5B"
DATASET_TYPE="math" # code, bbq
export TOKENIZERS_PARALLELISM=false

if [[ -z $CUDA_VISIBLE_DEVICES ]]; then
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    gpu_count=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
echo "Available GPU count: $gpu_count"


torchrun --nnodes 1 --nproc_per_node=$gpu_count src/script/train.py \
    --model_name_or_path $BASE_MODEL_NAME \
    --per_device_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --dataset_type $DATASET_TYPE \



