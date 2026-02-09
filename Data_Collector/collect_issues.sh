#!/bin/bash
#SBATCH --job-name=collector
#SBATCH --output=Data_Collector/logs/issues_collector_%j.log
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -e

# In SLURM, start from the submission directory (most predictable for relative paths).
WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$WORKDIR"

# Run from Data_Collector so relative paths (config.json, data/, logs/) match the project layout.
cd "Data_Collector"
echo "Working directory: $PWD"

echo "=== Job started on $(hostname) at $(date) ==="
mkdir -p "logs"

# GitHub auth (required by the collector).
# Recommended: export GITHUB_TOKEN in your shell and submit with `sbatch --export=ALL,GITHUB_TOKEN collect_issues.sh`.
if [[ -z "${GITHUB_TOKEN:-}" ]]; then
	if [[ -n "${GITHUB_TOKEN_FILE:-}" && -f "$GITHUB_TOKEN_FILE" ]]; then
		export GITHUB_TOKEN
		GITHUB_TOKEN="$(<"$GITHUB_TOKEN_FILE")"
	fi
fi

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
	echo "ERROR: Missing GITHUB_TOKEN environment variable (GitHub Personal Access Token)." >&2
	echo "Set it before submitting, e.g.: export GITHUB_TOKEN=...; sbatch --export=ALL,GITHUB_TOKEN collect_issues.sh" >&2
	exit 2
fi
CONDA_BASE="${CONDA_BASE:-/info/etu/m2/s2506967/anaconda3}"

# In SLURM batch jobs, `conda activate` often fails unless conda is initialized.
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
	source "$CONDA_BASE/etc/profile.d/conda.sh"
	conda activate bigdata-envir
	python run_collector.py
else
	"$CONDA_BASE/bin/conda" run -n bigdata-envir python run_collector.py
fi

echo "==== Job finished at: $(date) ===="
