BASE_MODEL_PATH="Qwen/Qwen2.5-1.5B"
FT_MODEL_PATH="ckpts/baseline/baseline_lftr=32_bbq_2e-05_cosine0d02"
BATCH_SIZE=512
BENCHMARK_DATA_DIR="data/bold"

python src/evaluation/eval_BOLD.py \
    --data_dir "$BENCHMARK_DATA_DIR" \
    --model_path "$FT_MODEL_PATH" \
    --batch_size $BATCH_SIZE \
    --limit_per_group 1000 \
    --output_file "result/alignment/results_base_model_bold.txt" \
    --max_new_tokens 30