#!/bin/bash
#SBATCH --job-name=build_dataset
#SBATCH --output=ml/logs/build_dataset_%j.log
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -e

# Always run from repo root so `python -m ml...` works,
# even if the job is submitted from inside the `ml/` directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

echo "Working directory: $PWD"
echo "=== Job started on $(hostname) at $(date) ==="
mkdir -p "ml/logs"

CONDA_BASE="${CONDA_BASE:-/info/etu/m2/s2506967/anaconda3}"

# In SLURM batch jobs, `conda activate` often fails unless conda is initialized.
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate bigdata-envir
    python -m ml.data.build_dataset
else
    "$CONDA_BASE/bin/conda" run -n bigdata-envir python -m ml.data.build_dataset
fi

echo "==== Job finished at: $(date) ===="
