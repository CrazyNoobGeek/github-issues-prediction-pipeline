#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/train_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpue06
#SBATCH --mem=30G
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00


set -e

echo "=== Job started on $(hostname) at $(date) ==="
mkdir -p logs ml/models ml/artifacts ml/artifacts/datasets ml/artifacts/embeddings


eval "$(/info/etu/m2/s2506967/anaconda3/bin/conda shell.bash hook)"
conda activate bigdata-envir
echo "Conda env: $CONDA_DEFAULT_ENV"
which python
python -V

PROJECT_ROOT="/info/etu/m2/s2506967/Big Data/github-issues-prediction-pipeline/"
cd "$PROJECT_ROOT"


export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# --- GPU info
nvidia-smi || true

export MLFLOW_TRACKING_URI="file:$PROJECT_ROOT/mlruns"
export ARTIFACTS_DIR="$PROJECT_ROOT/ml/artifacts"

# --- Training controls
export TIME_SPLIT="1"   
export CALIBRATE="1"      


python -m ml.train.train_classification_xgb

echo "=== Job finished at $(date) ==="