#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/two_stage_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpue05
#SBATCH --mem=30G
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00

set -e

echo "=== Job started on $(hostname) at $(date) ==="

PROJECT_ROOT="/info/etu/m2/s2506967/Big Data/github-issues-prediction-pipeline"
cd "$PROJECT_ROOT"

mkdir -p logs ml/models ml/artifacts mlruns

eval "$(/info/etu/m2/s2506967/anaconda3/bin/conda shell.bash hook)"
conda activate bigdata-envir

echo "Python: $(which python)"
python -V
nvidia-smi || true

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export ARTIFACTS_DIR="$PROJECT_ROOT/ml/artifacts"

export MLFLOW_TRACKING_URI="file:$PROJECT_ROOT/mlruns"
export MLFLOW_EXPERIMENT_NAME_STAGE1="github_stage1_close30_xgb_cv"
export MLFLOW_EXPERIMENT_NAME_STAGE2="github_stage2_within30_reg_xgb_cv"

export HORIZON_DAYS="30"
export TREE_METHOD="hist"

# run
python -m ml.train.train_stage1_classifier_30d_xgb
python -m ml.train.train_stage2_regression_within30_xgb

echo "=== Job finished at $(date) ==="
