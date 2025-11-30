CKPT_DIR_PATH="ckpts/baseline"
CKPT_PATH="baseline_lftr=32_bbq_2e-05_cosine0d02"

FT_MODEL_PATH="$CKPT_PATH/$MODEL_PATH"
BASE_MODEL_PATH="Qwen/Qwen2.5-1.5B"

BATCH_SIZE=512

echo "Load model from: $FT_MODEL_PATH"
python src/evaluation/eval_ubtox.py \
    --model_path "$FT_MODEL_PATH" \
    --batch_size $BATCH_SIZE \
    --num_samples 50000 \
    --output_file "result/alignment/results_lora32_model.txt" \
    --max_new_tokens 30