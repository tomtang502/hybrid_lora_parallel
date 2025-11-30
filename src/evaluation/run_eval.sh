#!/bin/bash
#SBATCH --job-name=llm-eval-array
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=22G
#SBATCH --time=12:00:00
#SBATCH -p GPU-shared
#SBATCH -A cis250214p
#SBATCH --array=0-0

MODEL_NAME="/ocean/projects/cis250214p/xzhang50/dlora/ckpts/baseline/baseline_fft_math_2e-05_cosine0d02/checkpoint-3000"
DTYPE="float16"

TASKS=("mmlu")
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

echo "======================================="
echo "Running job ${SLURM_ARRAY_TASK_ID} for task ${TASK}"
echo "Node: $(hostname)"
echo "GPU : $CUDA_VISIBLE_DEVICES"
echo "======================================="

module load anaconda3
source activate dlora

echo $CONDA_PREFIX

mkdir -p logs

python evaluator.py \
  --model "${MODEL_NAME}" \
  --task "${TASK}" \
  --dtype "${DTYPE}" \
  --batch_size auto 
