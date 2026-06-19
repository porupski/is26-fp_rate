#!/usr/bin/env bash
# 0_setup_env.sh — create the `is26-fp-rate` mamba environment.
# Run from the repo root:   bash scripts/0_setup_env.sh
#
# CPU-only — no GPU dependencies in this pipeline.

set -euo pipefail

ENV_NAME="is26-fp-rate"
PY_VERSION="3.11"

echo "→ Checking mamba is available..."
if ! command -v mamba &> /dev/null; then
    echo "mamba not found. Install miniforge first: https://github.com/conda-forge/miniforge"
    exit 1
fi
mamba --version

if mamba env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Env '${ENV_NAME}' already exists."
    echo "  To recreate: mamba env remove -n ${ENV_NAME} && bash scripts/0_setup_env.sh"
    exit 1
fi

echo
echo "→ Creating env '${ENV_NAME}' with Python ${PY_VERSION}..."
mamba create -n "${ENV_NAME}" python="${PY_VERSION}" -y

echo
echo "→ Installing dependencies from conda-forge..."
mamba install -n "${ENV_NAME}" -c conda-forge -y \
    numpy pandas scipy \
    statsmodels patsy \
    matplotlib seaborn \
    pillow \
    requests tqdm

echo
echo "Done."
echo
echo "To activate:"
echo "   mamba activate ${ENV_NAME}"
echo
echo "To verify:"
echo "   python -c 'import numpy, pandas, scipy, statsmodels, patsy, matplotlib, seaborn, PIL, requests, tqdm; print(\"ok\")'"
