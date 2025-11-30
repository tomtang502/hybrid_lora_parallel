#!/bin/bash
#SBATCH --job-name=crux-full
#SBATCH --output=logs/%x-%A.out
#SBATCH --error=logs/%x-%A.err
#SBATCH -p GPU-shared
#SBATCH -A cis250214p
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --mem=22G
#SBATCH --time=18:00:00

MODEL="/ocean/projects/cis250214p/xzhang50/dlora/ckpts/baseline/baseline_fft_math_2e-05_cosine0d02/checkpoint-3000"
SIZE=800   
TEMP=0.2
SAVE_DIR="crux_model_generations"
RESULT_DIR="crux_evaluation_results"

module load anaconda3
source activate /ocean/projects/cis250214p/xzhang50/conda_envs/cruxeval310

echo "===================="
echo " Running CRUXEval Full Pipeline"
echo " Model = $MODEL"
echo "===================="

INPUT_DIR="${SAVE_DIR}/$(basename $MODEL)+cot_temp${TEMP}_input"
mkdir -p $INPUT_DIR

echo ">> Stage 1 input_generation"
python crux_py/infe_crux.py \
    --model $MODEL \
    --tasks input_prediction \
    --use_auth_token \
    --trust_remote_code \
    --save_generations \
    --cot \
    --batch_size 10 \
    --n_samples 10 \
    --precision fp16 \
    --max_length_generation 2048 \
    --save_generations_path ${INPUT_DIR}/generations.json \
    --shuffle \
    --limit 10

OUTPUT_DIR="${SAVE_DIR}/$(basename $MODEL)+cot_temp${TEMP}_output"
mkdir -p $OUTPUT_DIR

echo ">> Stage 2 output_generation"
python crux_py/infe_crux.py \
    --model $MODEL \
    --tasks output_prediction \
    --use_auth_token \
    --trust_remote_code \
    --save_generations \
    --cot \
    --batch_size 10 \
    --n_samples 10 \
    --precision fp16 \
    --max_length_generation 2048 \
    --save_generations_path ${OUTPUT_DIR}/generations.json \
    --shuffle \
    --limit 10

mkdir -p ${RESULT_DIR}

echo ">> Stage 3 evaluating INPUT"
python crux_py/eval_crux.py \
    --generations_path ${INPUT_DIR}/generations.json \
    --scored_results_path ${RESULT_DIR}/$(basename $MODEL)_cot_temp${TEMP}_input.json \
    --mode input

echo ">> Stage 4 evaluating OUTPUT"
python crux_py/eval_crux.py \
    --generations_path ${OUTPUT_DIR}/generations.json \
    --scored_results_path ${RESULT_DIR}/$(basename $MODEL)_cot_temp${TEMP}_output.json \
    --mode output

echo "===================="
echo " ALL STAGES COMPLETE"
echo " Results saved in:"
echo "   $INPUT_DIR / $OUTPUT_DIR"
echo "   ${RESULT_DIR}/"
echo "===================="
