# FP rate modelling — Slavic parliaments (InterSpeech 2026)

[![DOI](https://zenodo.org/badge/1274287623.svg)](https://doi.org/10.5281/zenodo.20766877)

Code accompanying:

> Porupski, I., Dropuljić, B., Ljubešić, N. (2026). *Umm… With Transformers? Insights from Filled Pause Use across Four Slavic Parliaments.* InterSpeech 2026.

Fits the **filled-pause rate** models reported in the paper: utterance-level Negative Binomial GEE/GLM with a `log(audio_length)` offset, Mundlak between/within decomposition, global + per-parliament scopes — over ~4,000 h of Croatian, Czech, Polish, and Serbian parliamentary speech (ParlaSpeech v3).

CPU-only. See [ENV_SETUP.md](./ENV_SETUP.md) for the env (`is26-fp-rate`).

## Pipeline

Six sequential steps. Run from the repo root.

| Step | Script | What it does |
|---|---|---|
| 0 | `scripts/0_setup_env.sh` | Creates the `is26-fp-rate` mamba env with all dependencies. |
| 1 | `scripts/1_download_ParlaSpeech_jsonls.py` | Pulls the four v3.0 JSONL files from CLARIN into `ParlaSpeech_v3_jsonls/` and decompresses them. |
| 2 | `scripts/2_extract_FP_counts.py` | One row per utterance: FP count, silent-pause aggregates, syllable / word speech rates, speaker metadata. Writes per-language JSONLs into `fp_count_jsonls/`. |
| 3 | `scripts/3_build_TSV_with_Mundlak.py` | Merges the per-language JSONLs into one TSV; maps `Party_status` → {0, 1}; adds Mundlak between/within columns. |
| 4 | `scripts/4_run_GEE_suite.py` | Fits the NB GEE/GLM suite — baseline, Mundlak, and seven interaction tracks — globally and per parliament. One `.txt` log per fitted model + diagnostic / forest plots in `results/run_<timestamp>_<specs>/`. |
| 5 | `scripts/5_collect_results_and_plot.py` | Walks one results run, parses every per-model log, writes a combined CSV, and renders the per-track + master forest plots used in the paper. |
| 6 | `scripts/6_plot_paper_figure.py` | Renders the paper's Figure 1 layout (5-panel: GLOBAL + CZ/HR/PL/RS, baseline + Mundlak dots, GEE_NB_Indep). |

Shared helpers live in `scripts/utils_GEE.py` (step 4) and `scripts/utils_results.py` (steps 5 & 6).

## End-to-end

```bash
bash scripts/0_setup_env.sh
mamba activate is26-fp-rate

python scripts/1_download_ParlaSpeech_jsonls.py
python scripts/2_extract_FP_counts.py
python scripts/3_build_TSV_with_Mundlak.py
python scripts/4_run_GEE_suite.py
# then point steps 5 & 6 at the run folder just produced:
python scripts/5_collect_results_and_plot.py
python scripts/6_plot_paper_figure.py --results-dir results/run_<...>
```

## Layout

```
is26-fp_rate/                                  ← repo root
├── README.md
├── ENV_SETUP.md
├── requirements.txt
├── scripts/
│   ├── 0_setup_env.sh
│   ├── 1_download_ParlaSpeech_jsonls.py
│   ├── 2_extract_FP_counts.py
│   ├── 3_build_TSV_with_Mundlak.py
│   ├── 4_run_GEE_suite.py
│   ├── 5_collect_results_and_plot.py
│   ├── 6_plot_paper_figure.py
│   ├── utils_GEE.py
│   └── utils_results.py
├── ParlaSpeech_v3_jsonls/                     (populated by step 1)
├── fp_count_jsonls/                           (populated by steps 2-3)
└── results/                                   (populated by step 4)
    └── run_YYYY-MM-DD_HH-MM-SS_<specs>/
```

Scripts derive their paths from their own location (`__file__`), so they work regardless of cwd.

## Inputs

- **ParlaSpeech v3** annotation JSONLs — one per language (CZ, HR, PL, RS). Source: https://clarinsi.github.io/parlaspeech/ (CLARIN.SI handle `11356/1833`).
- Speaker metadata (gender, age, party, party orientation, party power status) and ParlaSent sentiment labels are embedded in each utterance record.

Sub-selection at step 1:

```bash
python scripts/1_download_ParlaSpeech_jsonls.py HR PL   # just two languages
```

## Outputs

- **Step 2** → per-language `fp_ParlaSpeech-*.jsonl` in `fp_count_jsonls/`
- **Step 3** → `ParlaSpeech-ALL.v3.0-fp-counts.tsv` (+ a stratified subset for development)
- **Step 4** → one `.txt` log per fitted model under `results/run_<timestamp>_<specs>/{scope}/`, plus count-distribution, partial-residual, and per-model forest PNGs
- **Step 5** → `collected_results.csv`, per-model-type sub-CSVs, and 5-panel forest PNGs (per-track and master)
- **Step 6** → `global_vs_local_comparison_GEE_NB_Indep.png` — the paper's Figure 1

## Model

For each utterance *i, t*:

```
FP_count_it ~ Gender_i + Age_it + Rate*_it + Sent*_it + Orient*_it + Status*_it
              + C(Parliament)_i + offset(log audio_length_it)
```

where `x*` is either the raw covariate (baseline) or the Mundlak pair `(x̄_i, x_it − x̄_i)`. Clustered on `Speaker_ID`. GEE with independent working correlation is the primary specification reported in the paper; GEE-exchangeable, GLM-naive, and GLM-clustered are written alongside for sensitivity.

## Filters (step 4)

Speakers with `role == "Regular"`, utterances ≥ 10 words and ≥ 3 s audio, ≥ 10 utterances per speaker. After filtering: 1,001,787 utterances from 1,561 speakers (3,889 h).

## Citation

If you use this code, please cite both the paper and the software:

```bibtex
@inproceedings{porupski2026umm,
  title     = {Umm... With Transformers? Insights from Filled Pause Use across Four Slavic Parliaments},
  author    = {Porupski, Ivan and Dropulji\'{c}, Branimir and Ljube\v{s}i\'{c}, Nikola},
  booktitle = {Proc. Interspeech 2026},
  year      = {2026}
}

@software{porupski2026is26fprate,
  author    = {Porupski, Ivan and Dropulji\'{c}, Branimir and Ljube\v{s}i\'{c}, Nikola},
  title     = {is26-fp\_rate: Filled-pause rate modelling pipeline (InterSpeech 2026)},
  year      = {2026},
  version   = {v1.0.0},
  doi       = {10.5281/zenodo.20766877},
  url       = {https://doi.org/10.5281/zenodo.20766877}
}
```
