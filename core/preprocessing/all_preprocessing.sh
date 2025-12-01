#!/bin/bash

set -e

# -------- Get script directory and cd to repo root --------
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/.." || { echo "Failed to cd to repo root"; exit 1; }

# -------- Check for conda --------
if ! command -v conda &> /dev/null; then
    echo "Conda could not be found. Please install Anaconda/Miniconda first."
    exit 1
fi

# -------- Activate environment --------
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -q bikesafeai; then
    echo "Conda environment 'bikesafeai' does not exist. Please create it first."
    exit 1
fi
conda activate bikesafeai

# -------- Sample size (default -1) --------
SAMPLES=${1:--1}
echo "Running with sample size: $SAMPLES"

# -------- Run preprocessing steps --------
echo "Running bbox extraction..."
python preprocessing/bboxes_extraction.py --max-image-count "$SAMPLES"

echo "Running flow extraction..."
python preprocessing/flow_extraction.py --max-image-count "$SAMPLES"

echo "Running feature extraction..."
python preprocessing/feature_extraction.py --max-image-count "$SAMPLES"

echo "Running score calibration..."
python preprocessing/score_calibration.py --max-image-count "$SAMPLES"

echo "All steps completed successfully!"
