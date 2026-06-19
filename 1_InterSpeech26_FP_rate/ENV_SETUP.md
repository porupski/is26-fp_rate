# Environment setup

CPU-only environment — no GPU needed for any stage of this pipeline.

## Prerequisites

Miniforge or micromamba. If `mamba` is not on your PATH, initialise it first:

```bash
eval "$(micromamba shell hook --shell bash)"
alias mamba=micromamba   # if needed
```

## Quick setup

From the repo root:

```bash
bash scripts/0_setup_env.sh
mamba activate is26-fp-rate
```

That creates a fresh env named `is26-fp-rate` with Python 3.11 and installs everything from `conda-forge`.

## Verify

```bash
python -c "import numpy, pandas, scipy, statsmodels, patsy, matplotlib, seaborn, PIL, requests, tqdm; print('ok')"
```

## Manual setup

```bash
mamba create -n is26-fp-rate python=3.11 -y
mamba activate is26-fp-rate
mamba install -c conda-forge -y \
    numpy pandas scipy \
    statsmodels patsy \
    matplotlib seaborn pillow \
    requests tqdm
```

Or with pip (`requirements.txt`):

```bash
pip install -r requirements.txt
```

## Reset

```bash
mamba env remove -n is26-fp-rate
```

## What each library is for

| Library | Used by |
|---|---|
| numpy, pandas, scipy | every stage |
| statsmodels, patsy | step 4 — GEE/GLM Negative Binomial fitting |
| matplotlib, seaborn, pillow | steps 4 & 5 — distribution / forest / partial-residual plots |
| requests, tqdm | step 1 — downloading ParlaSpeech JSONLs from CLARIN |
