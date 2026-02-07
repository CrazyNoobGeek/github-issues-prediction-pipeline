#!/bin/bash
#SBATCH --job-name=gh_emb
#SBATCH --output=logs/gh_emb_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpue07
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

# --- Config
export ARTIFACTS_DIR="$PROJECT_ROOT/ml/artifacts"
export EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
export EMBED_BATCH="64"


python -m ml.features.generate_embeddings

echo "=== Job finished at $(date) ==="
