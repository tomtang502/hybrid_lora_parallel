#!/bin/bash

MODELS=(
# "/ocean/projects/cis250214p/xzhang50/dlora/ckpts/baseline/baseline_fft_math_2e-05_cosine0d02/checkpoint-3000"
"Qwen/Qwen2.5-1.5B"
)

TASKS=("mbpp")
DTYPE="float16"
SBATCH_TIME="12:00:00"
SBATCH_GPU="1"
SBATCH_ACC="cis250214p"
SBATCH_PART="GPU-shared"
TEMPS=("0.2" "0.8")
SAVE_DIR="crux_model_generations"
RESULT_DIR="crux_evaluation_results"
CRUX_SIZE=800

for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    if [[ "$task" == "math" ]]; then
      JOB_NAME=$(basename "$model")"-$task"
cat <<EOF > temp_eval_${JOB_NAME}.sh
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/%x-%A.out
#SBATCH --error=logs/%x-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:${SBATCH_GPU}
#SBATCH --mem=22G
#SBATCH --time=${SBATCH_TIME}
#SBATCH -p ${SBATCH_PART}
#SBATCH -A ${SBATCH_ACC}

module load anaconda3
source activate /ocean/projects/cis250214p/xzhang50/conda_envs/dlora
echo "======================================="
echo "Model   : ${model}"
echo "Task    : ${task}"
echo "Node    : \$(hostname)"
echo "GPU     : \$CUDA_VISIBLE_DEVICES"
echo "Env     : \$(which python)"
echo "======================================="

python eval_math.py \
  --model "${model}" 

EOF
      echo "Submitted: $JOB_NAME"
      sbatch temp_eval_${JOB_NAME}.sh
      rm temp_eval_${JOB_NAME}.sh
      continue
    fi

    if [[ "$task" == "cruxeval" ]]; then
      for TEMP in "${TEMPS[@]}"; do
        JOB_NAME=$(basename "$model")"-$task-temp${TEMP}"
cat <<EOF > temp_eval_${JOB_NAME}.sh
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/%x-%A.out
#SBATCH --error=logs/%x-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:${SBATCH_GPU}
#SBATCH --mem=22G
#SBATCH --time=${SBATCH_TIME}
#SBATCH -p ${SBATCH_PART}
#SBATCH -A ${SBATCH_ACC}

module load anaconda3
source activate /ocean/projects/cis250214p/xzhang50/conda_envs/cruxeval310

echo "===================="
echo " Running CRUXEval Full Pipeline"
echo " Model = $model"
echo "===================="

INPUT_DIR="${SAVE_DIR}/$(basename $model)+cot_temp${TEMP}_input"
mkdir -p \$INPUT_DIR

echo ">> Stage 1 input_generation"
python crux_py/infe_crux.py \
    --model $model \
    --tasks input_prediction \
    --use_auth_token \
    --trust_remote_code \
    --save_generations \
    --cot \
    --batch_size 10 \
    --n_samples 10 \
    --precision fp16 \
    --temperature $TEMP \
    --max_length_generation 2048 \
    --save_generations_path \${INPUT_DIR}/generations.json \
    --shuffle \
    --limit ${CRUX_SIZE}

OUTPUT_DIR="${SAVE_DIR}/$(basename $model)+cot_temp${TEMP}_output"
mkdir -p \$OUTPUT_DIR

echo ">> Stage 2 output_generation"
python crux_py/infe_crux.py \
    --model $model \
    --tasks output_prediction \
    --use_auth_token \
    --trust_remote_code \
    --save_generations \
    --cot \
    --batch_size 10 \
    --n_samples 10 \
    --precision fp16 \
    --temperature $TEMP \
    --max_length_generation 2048 \
    --save_generations_path \${OUTPUT_DIR}/generations.json \
    --shuffle \
    --limit ${CRUX_SIZE}

mkdir -p ${RESULT_DIR}

echo ">> Stage 3 evaluating INPUT"
python crux_py/eval_crux.py \
    --generations_path \${INPUT_DIR}/generations.json \
    --scored_results_path ${RESULT_DIR}/$(basename $model)_cot_temp${TEMP}_input.json \
    --mode input

echo ">> Stage 4 evaluating OUTPUT"
python crux_py/eval_crux.py \
    --generations_path \${OUTPUT_DIR}/generations.json \
    --scored_results_path ${RESULT_DIR}/$(basename $model)_cot_temp${TEMP}_output.json \
    --mode output

echo "===================="
echo " ALL STAGES COMPLETE"
echo " Results saved in:"
echo "   \$INPUT_DIR / \$OUTPUT_DIR"
echo "   ${RESULT_DIR}/"
echo "===================="

EOF

        echo "Submitted: $JOB_NAME"
        sbatch temp_eval_${JOB_NAME}.sh
        rm temp_eval_${JOB_NAME}.sh
      done
      continue
    fi

    JOB_NAME=$(basename "$model")"-$task"

cat <<EOF > temp_eval_${JOB_NAME}.sh
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/%x-%A.out
#SBATCH --error=logs/%x-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:${SBATCH_GPU}
#SBATCH --mem=22G
#SBATCH --time=${SBATCH_TIME}
#SBATCH -p ${SBATCH_PART}
#SBATCH -A ${SBATCH_ACC}

module load anaconda3
source activate /ocean/projects/cis250214p/xzhang50/conda_envs/dlora

echo "======================================="
echo "Model   : ${model}"
echo "Task    : ${task}"
echo "Node    : \$(hostname)"
echo "GPU     : \$CUDA_VISIBLE_DEVICES"
echo "Env     : \$(which python)"
echo "======================================="

python evaluator.py \
  --model "${model}" \
  --dtype "${DTYPE}" \
  --task "${task}" \
  --batch_size auto

EOF

    echo "Submitted: $JOB_NAME"
    sbatch temp_eval_${JOB_NAME}.sh
    rm temp_eval_${JOB_NAME}.sh

  done
done