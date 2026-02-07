#!/usr/bin/env bash
set -e

# Activate your env
# Example:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env

export ARTIFACTS_DIR="ml/artifacts"
export JSONL_ROOT="Data_Collector/data/raw/issues"
export DATA_SOURCE="jsonl"

export EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
export EMBED_BATCH="64"

export TIME_SPLIT="1"
export CALIBRATE="1"