#!/bin/bash
set -e

ENV_NAME=${1:-bikesafeai}
PYTHON_VERSION=3.12

echo "Setting up conda environment: $ENV_NAME with Python $PYTHON_VERSION"

# -------- Check for conda --------
if ! command -v conda &> /dev/null; then
    echo "Conda could not be found. Please install Anaconda/Miniconda first."
    exit 1
fi

# -------- Initialize conda in this shell --------
eval "$(conda shell.bash hook)"

# -------- Create environment if it doesn't exist --------
if conda env list | grep -q "^$ENV_NAME\s"; then
    echo "Environment '$ENV_NAME' already exists. Skipping creation."
else
    echo "Creating environment '$ENV_NAME'..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

# -------- Activate environment --------
conda activate "$ENV_NAME"

# -------- Install requirements --------
if [[ -f "requirements.txt" ]]; then
    echo "Installing Python dependencies from requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Skipping pip install."
fi

echo "Setup complete! Environment '$ENV_NAME' is ready."
