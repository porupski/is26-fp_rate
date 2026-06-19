"""
4_run_GEE_suite.py — FP rate modelling
======================================
Negative Binomial GEE/GLM (log link + offset), one row = one utterance.

Three specification tracks:
  BASELINE     — no Mundlak, no interactions
  MUNDLAK      — rate + sentiment decomposed (between/within)
  INTERACTION  — baseline + one interaction term (×7 separate models)

Each track fits 4 models per scope:
  GLM_NB         — naive SEs
  GLM_NB_Cluster — clustered sandwich SEs
  GEE_NB_Indep   — independence working correlation  ← PRIMARY (reported in paper)
  GEE_NB_Exch    — exchangeable working correlation

Scopes: global + per-parliament.

Results folder:
  RESULTS_BASE/{TIMESTAMP}_nb_{rate}_cluster-{X}/
    global/
    CZ/ HR/ PL/ RS/

Edit only the CONFIG block.
"""

import os, datetime, warnings
os.nice(19)

import numpy as np
import pandas as pd
import statsmodels.api as sm

from utils_GEE import (
    filter_data, cluster_size_check, party_switch_diagnostic,
    build_formula, build_formula_interaction, get_interaction_specs,
    get_reference_levels, format_reference_levels,
    estimate_nb_alpha, get_sd_context, format_sd_context,
    run_model_suite,
    plain_english_block,
    plot_count_distribution, plot_irr_forest, plot_partial_residuals,
    save_single_model,
)

# =================================================================
# CONFIG — only edit here
# =================================================================

ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_TSV    = os.path.join(ROOT, "fp_count_jsonls", "ParlaSpeech-ALL.v3.0-fp-counts.tsv")
RESULTS_BASE = os.path.join(ROOT, "results")

# ── Rate type ─────────────────────────────────────────────────
USE_SYLLABLE_RATE = True   # True -> speech_rate_syll | False -> speech_rate_word

# ── Age scaling ───────────────────────────────────────────────
ZSCORE_AGE = False          # True -> scale(age) per 1 SD | False -> per 10 years

# ── Orientation coding ────────────────────────────────────────
USE_ABS_ORIENTATION = False  # True -> |orientation| | False -> left vs right

# ── Cluster size filter ───────────────────────────────────────
MIN_CLUSTER_OBS      = 10    # min utterances per speaker
APPLY_CLUSTER_FILTER = True

# ── Scaling flags ─────────────────────────────────────────────
SCALE_SENTIMENT   = False
SCALE_ORIENTATION = False

# ── Mundlak: per-variable (only for MUNDLAK track) ───────────
MUNDLAK_RATE         = True
MUNDLAK_SENTIMENT    = True
MUNDLAK_PARTY_STATUS = True
MUNDLAK_ORIENTATION  = True

# ── Track toggles ────────────────────────────────────────────
RUN_BASELINE     = True
RUN_MUNDLAK      = True
RUN_INTERACTIONS = True

# ── Scope toggles ────────────────────────────────────────────
RUN_GLOBAL         = True
RUN_PER_PARLIAMENT = True
PARL_MIN_ROWS      = 200
PARL_MIN_SPEAKERS  = 15

# ── GEE clustering ───────────────────────────────────────────
GEE_CLUSTER_BY     = "Speaker_ID"   # "Speaker_ID" | "Speech_ID"
GLM_CLUSTER_SE     = True

# =================================================================
# DERIVED — do not edit
# =================================================================

RATE_COL    = "speech_rate_syll" if USE_SYLLABLE_RATE else "speech_rate_word"
RATE_SUFFIX = "syll" if USE_SYLLABLE_RATE else "word"

VARS = {
    "target":            "fp_count",
    "gender":            "Speaker_gender",
    "age":               "Speaker_age_at_session",
    "rate":              RATE_COL,
    "sentiment":         "Sentiment_label_float",
    # Mundlak decomposed (precomputed in TSV)
    "rate_mean":         f"{RATE_COL}_mean",
    "rate_dev":          f"{RATE_COL}_dev",
    "sentiment_mean":    "Sentiment_label_float_mean",
    "sentiment_dev":     "Sentiment_label_float_dev",
    "party_status":      "Party_status",
    "party_status_mean": "Party_status_mean",
    "party_status_dev":  "Party_status_dev",
    "orientation":       "Party_orientation_float",
    "orientation_mean":  "Party_orientation_float_mean",
    "orientation_dev":   "Party_orientation_float_dev",
    # Structure
    "grouping":          "Lang",
    "cluster":           "Speaker_ID",
    "speech":            "Speech_ID",
    "exposure":          "audio_length",
}

TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

_cluster_col_map = {
    "Speaker_ID": VARS["cluster"],
    "Speech_ID":  VARS["speech"],
}
_gee_clust_col = _cluster_col_map.get(GEE_CLUSTER_BY, VARS["cluster"])

_flags = (f"nb"
          f"_{RATE_SUFFIX}"
          f"_cluster-{GEE_CLUSTER_BY}")
RESULTS_DIR = os.path.join(RESULTS_BASE, f"run_{TIMESTAMP}_{_flags}")

# Shared formula kwargs (baseline = no Mundlak)
FORMULA_KWARGS_BASE = dict(
    use_abs_orientation=USE_ABS_ORIENTATION,
    zscore_age=ZSCORE_AGE,
    age_mean=None,  # filled per-scope
    mundlak_rate=False,
    mundlak_sentiment=False,
    mundlak_party_status=False,
    mundlak_orientation=False,
    scale_sentiment=SCALE_SENTIMENT,
    scale_orientation=SCALE_ORIENTATION,
)

FORMULA_KWARGS_MUNDLAK = dict(
    use_abs_orientation=USE_ABS_ORIENTATION,
    zscore_age=ZSCORE_AGE,
    age_mean=None,  # filled per-scope
    mundlak_rate=MUNDLAK_RATE,
    mundlak_sentiment=MUNDLAK_SENTIMENT,
    mundlak_party_status=MUNDLAK_PARTY_STATUS,
    mundlak_orientation=MUNDLAK_ORIENTATION,
    scale_sentiment=SCALE_SENTIMENT,
    scale_orientation=SCALE_ORIENTATION,
)

# Shared interpretation kwargs
INTERP_KWARGS_BASE = dict(
    rate_suffix=RATE_SUFFIX,
    use_abs_orientation=USE_ABS_ORIENTATION,
    zscore_age=ZSCORE_AGE,
    scale_sentiment=SCALE_SENTIMENT,
    scale_orientation=SCALE_ORIENTATION,
    mundlak_rate=False,
    mundlak_sentiment=False,
    mundlak_party_status=False,
    mundlak_orientation=False,
)

INTERP_KWARGS_MUNDLAK = dict(
    rate_suffix=RATE_SUFFIX,
    use_abs_orientation=USE_ABS_ORIENTATION,
    zscore_age=ZSCORE_AGE,
    scale_sentiment=SCALE_SENTIMENT,
    scale_orientation=SCALE_ORIENTATION,
    mundlak_rate=MUNDLAK_RATE,
    mundlak_sentiment=MUNDLAK_SENTIMENT,
    mundlak_party_status=MUNDLAK_PARTY_STATUS,
    mundlak_orientation=MUNDLAK_ORIENTATION,
)

# Shared save kwargs
SAVE_KWARGS_BASE = dict(
    use_abs_orientation=USE_ABS_ORIENTATION,
    zscore_age=ZSCORE_AGE,
    scale_sentiment=SCALE_SENTIMENT,
    scale_orientation=SCALE_ORIENTATION,
    mundlak_rate=False,
    mundlak_sentiment=False,
    mundlak_party_status=False,
    mundlak_orientation=False,
    use_mundlak=False,
    gee_cluster_by=GEE_CLUSTER_BY,
)

SAVE_KWARGS_MUNDLAK = dict(
    use_abs_orientation=USE_ABS_ORIENTATION,
    zscore_age=ZSCORE_AGE,
    scale_sentiment=SCALE_SENTIMENT,
    scale_orientation=SCALE_ORIENTATION,
    mundlak_rate=MUNDLAK_RATE,
    mundlak_sentiment=MUNDLAK_SENTIMENT,
    mundlak_party_status=MUNDLAK_PARTY_STATUS,
    mundlak_orientation=MUNDLAK_ORIENTATION,
    use_mundlak=True,
    gee_cluster_by=GEE_CLUSTER_BY,
)


# =================================================================
# PRINT CONFIG
# =================================================================

print(f"\n{'='*60}")
print(f"  FP rate pipeline  |  {TIMESTAMP}")
print(f"  Unit of analysis : utterance (one row = one utterance)")
print(f"  Model            : NB GEE/GLM (log link + offset)")
print(f"  Target           : fp_count")
print(f"  Rate             : {RATE_SUFFIX}")
print(f"  Age scaling      : {'z-score' if ZSCORE_AGE else 'per decade'}")
print(f"  Orientation      : {'|abs| extremism' if USE_ABS_ORIENTATION else 'left vs right'}")
print(f"  GEE cluster by   : {GEE_CLUSTER_BY}  ({_gee_clust_col})")
print(f"  Mundlak (track 2):")
print(f"    rate           : {'ON' if MUNDLAK_RATE else 'OFF'}")
print(f"    sentiment      : {'ON' if MUNDLAK_SENTIMENT else 'OFF'}")
print(f"    party status   : {'ON' if MUNDLAK_PARTY_STATUS else 'OFF'}")
print(f"    orientation    : {'ON' if MUNDLAK_ORIENTATION else 'OFF'}")
print(f"  Tracks:")
print(f"    Baseline       : {'ON' if RUN_BASELINE else 'OFF'}")
print(f"    Mundlak        : {'ON' if RUN_MUNDLAK else 'OFF'}")
print(f"    Interactions   : {'ON' if RUN_INTERACTIONS else 'OFF'}")
print(f"  Scopes:")
print(f"    Global         : {'ON' if RUN_GLOBAL else 'OFF'}")
print(f"    Per-parl       : {'ON' if RUN_PER_PARLIAMENT else 'OFF'}")
print(f"  Results          : {RESULTS_DIR}")
print(f"{'='*60}\n")


# =================================================================
# HELPERS
# =================================================================

def make_dirs(scope_label):
    base = os.path.join(RESULTS_DIR, scope_label)
    dirs = {
        "base":   base,
        "dist":   os.path.join(base, "plots", "distributions"),
        "forest": os.path.join(base, "plots", "forest_plots"),
        "resid":  os.path.join(base, "plots", "residuals"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def process_and_save(label, fit, df_scope, dirs, refs, sd_ctx,
                     formula, alpha, scope_str, interp_kwargs,
                     save_kwargs, interaction_label=None):
    """Post-process a single fitted model: print, interpret, plot, save."""
    print(f"\n{'='*60}\n  {label}  |  {scope_str}\n{'='*60}")

    try:
        for line in format_reference_levels(refs, VARS):
            print(line)
    except Exception as e:
        print(f"  [WARN] reference levels: {e}")

    try:
        print(fit.summary())
    except Exception as e:
        print(f"  [WARN] summary print failed: {e}")

    try:
        if "Exch" in label and hasattr(fit, "cov_struct"):
            ac = fit.cov_struct.dep_params
            interp = ("very weak" if ac < 0.1
                      else "moderate" if ac < 0.3
                      else "strong -- dominant fingerprint")
            print(f"\n  Exchangeable alpha = {ac:.4f}  ({interp})")
            print(f"  Cluster unit       : {_gee_clust_col}")
    except Exception as e:
        print(f"  [WARN] GEE alpha print failed: {e}")

    pe_lines = []
    try:
        pe_lines = plain_english_block(
            fit, label, VARS, refs, sd_ctx=sd_ctx, **interp_kwargs)
        for line in pe_lines:
            print(line)
    except Exception as e:
        pe_lines = [f"[plain_english_block failed: {e}]"]
        print(f"  [WARN] plain_english_block failed for {label}: {e}")

    try:
        plot_irr_forest(fit, f"{label}_{scope_str}",
                        dirs["forest"], RATE_SUFFIX, TIMESTAMP)
    except Exception as e:
        print(f"  [WARN] plot_irr_forest failed: {e}")

    try:
        save_single_model(
            label=label, fit=fit, pe_lines=pe_lines,
            sd_ctx=sd_ctx, refs=refs,
            out_dir=dirs["base"], rate_suffix=RATE_SUFFIX,
            scope_label=scope_str, timestamp=TIMESTAMP,
            formula=formula, alpha=alpha, v=VARS,
            n_rows=len(df_scope),
            n_speakers=df_scope[VARS["cluster"]].nunique(),
            interaction_label=interaction_label,
            **save_kwargs,
        )
    except Exception as e:
        print(f"  [WARN] save_single_model failed for {label}: {e}")


# =================================================================
# SCOPE RUNNER
# =================================================================

def run_scope(df_scope, log_exposure, parliament):
    """Run all enabled tracks for a single scope."""
    scope_str = parliament if parliament != "global" else "global"

    print(f"\n{'#'*60}")
    print(f"  SCOPE       : {scope_str}")
    print(f"  Utterances  : {len(df_scope):,}")
    print(f"  Speakers    : {df_scope[VARS['cluster']].nunique():,}")
    if VARS["speech"] in df_scope.columns:
        print(f"  Speeches    : {df_scope[VARS['speech']].nunique():,}")
    print(f"  GEE cluster : {_gee_clust_col}")
    print(f"{'#'*60}\n")

    refs   = get_reference_levels(df_scope, VARS)
    sd_ctx = get_sd_context(df_scope, VARS)
    age_mean = df_scope[VARS["age"]].mean()

    for line in format_reference_levels(refs, VARS):
        print(line)

    # Count distribution plot
    dirs_base = make_dirs(f"{scope_str}_baseline")
    try:
        plot_count_distribution(
            df_scope, VARS, f"nb_{scope_str}",
            dirs_base["dist"], TIMESTAMP)
    except Exception as e:
        print(f"  [WARN] plot_count_distribution failed: {e}")

    # FP count summary
    fp = df_scope[VARS["target"]]
    print(f"  FP count summary:")
    print(f"    mean ± SD : {fp.mean():.2f} ± {fp.std():.2f}")
    print(f"    median    : {fp.median():.1f}")
    print(f"    zero-FP   : {(fp == 0).sum():,} ({100*(fp==0).mean():.1f}%)\n")

    is_global = (parliament == "global")

    # ── Track 1: Baseline ─────────────────────────────────────
    if RUN_BASELINE:
        print(f"\n  === TRACK: BASELINE (no Mundlak) ===")
        fkw = {**FORMULA_KWARGS_BASE, "age_mean": age_mean,
               "include_lang": is_global}
        formula = build_formula(VARS, **fkw)
        print(f"  Formula: {formula}\n")

        alpha = estimate_nb_alpha(formula, df_scope, log_exposure)
        dirs = make_dirs(f"{scope_str}_baseline")

        run_model_suite(
            formula=formula, df=df_scope,
            log_exposure=log_exposure, alpha=alpha,
            cluster_col=_gee_clust_col,
            glm_cluster_se=GLM_CLUSTER_SE,
            track_label="BASE",
            on_model_ready=lambda lbl, fit: process_and_save(
                lbl, fit, df_scope, dirs, refs, sd_ctx,
                formula, alpha, f"{scope_str}_baseline",
                INTERP_KWARGS_BASE, SAVE_KWARGS_BASE,
            ),
        )

    # ── Track 2: Mundlak ─────────────────────────────────────
    if RUN_MUNDLAK:
        print(f"\n  === TRACK: MUNDLAK ===")
        fkw = {**FORMULA_KWARGS_MUNDLAK, "age_mean": age_mean,
               "include_lang": is_global}
        formula = build_formula(VARS, **fkw)
        print(f"  Formula: {formula}\n")

        alpha = estimate_nb_alpha(formula, df_scope, log_exposure)
        dirs = make_dirs(f"{scope_str}_mundlak")

        run_model_suite(
            formula=formula, df=df_scope,
            log_exposure=log_exposure, alpha=alpha,
            cluster_col=_gee_clust_col,
            glm_cluster_se=GLM_CLUSTER_SE,
            track_label="MUNDLAK",
            on_model_ready=lambda lbl, fit: process_and_save(
                lbl, fit, df_scope, dirs, refs, sd_ctx,
                formula, alpha, f"{scope_str}_mundlak",
                INTERP_KWARGS_MUNDLAK, SAVE_KWARGS_MUNDLAK,
            ),
        )

    # ── Track 3: Interactions (baseline + one interaction) ────
    if RUN_INTERACTIONS:
        print(f"\n  === TRACK: INTERACTIONS ===")
        interaction_specs = get_interaction_specs(VARS, ZSCORE_AGE, age_mean)

        for int_label, int_term in interaction_specs:
            print(f"\n  --- Interaction: {int_label} ---")
            fkw = {**FORMULA_KWARGS_BASE, "age_mean": age_mean,
                   "include_lang": is_global}
            formula = build_formula_interaction(
                VARS, interaction_term=int_term, **fkw)
            print(f"  Formula: {formula}\n")

            alpha = estimate_nb_alpha(formula, df_scope, log_exposure)
            dirs = make_dirs(f"{scope_str}_int_{int_label}")

            # Use baseline interp/save kwargs (interactions are non-Mundlak)
            run_model_suite(
                formula=formula, df=df_scope,
                log_exposure=log_exposure, alpha=alpha,
                cluster_col=_gee_clust_col,
                glm_cluster_se=GLM_CLUSTER_SE,
                track_label=f"INT_{int_label.upper()}",
                on_model_ready=lambda lbl, fit, _il=int_label, _f=formula, _a=alpha, _d=dirs: process_and_save(
                    lbl, fit, df_scope, _d, refs, sd_ctx,
                    _f, _a, f"{scope_str}_int_{_il}",
                    INTERP_KWARGS_BASE, SAVE_KWARGS_BASE,
                    interaction_label=_il,
                ),
            )


# =================================================================
# LOAD & FILTER
# =================================================================

print("Loading data ...")
df_raw = pd.read_csv(INPUT_TSV, sep="\t", low_memory=False)
print(f"  Loaded {len(df_raw):,} utterances\n")

if VARS["grouping"] in df_raw.columns:
    print("  Utterances per language:")
    for lang, cnt in df_raw[VARS["grouping"]].value_counts().items():
        print(f"    {lang}: {cnt:,}")
    print()

# Filter
df_clean = filter_data(df_raw, VARS, word_min=10, audio_min=3.0, role="Regular")

# Cluster size filter
df_clean = cluster_size_check(df_clean, VARS,
                               min_obs=MIN_CLUSTER_OBS,
                               apply_filter=APPLY_CLUSTER_FILTER)
party_switch_diagnostic(df_clean, VARS)

# Final summary
print(f"\n  Final dataset : {len(df_clean):,} utterances")
print(f"  Speakers      : {df_clean[VARS['cluster']].nunique():,}")
if VARS["speech"] in df_clean.columns:
    print(f"  Speeches      : {df_clean[VARS['speech']].nunique():,}")
fp = df_clean[VARS["target"]]
print(f"  FP count      : mean={fp.mean():.2f}  "
      f"median={fp.median():.1f}  "
      f"zero={100*(fp==0).mean():.1f}%\n")

# Pre-modelling distribution
_pre_dirs = make_dirs("_diagnostics")
try:
    plot_count_distribution(
        df_clean, VARS, "post_filter", _pre_dirs["dist"], TIMESTAMP)
except Exception as e:
    print(f"  [WARN] post-filter distribution plot failed: {e}")


# =================================================================
# RUN SCOPES
# =================================================================

log_exposure_full = np.log(df_clean[VARS["exposure"]])

# Global
if RUN_GLOBAL:
    run_scope(df_clean, log_exposure_full, "global")

# Per parliament
if RUN_PER_PARLIAMENT:
    for parl in sorted(df_clean[VARS["grouping"]].unique()):
        df_p = df_clean[df_clean[VARS["grouping"]] == parl]
        if len(df_p) < PARL_MIN_ROWS:
            print(f"  [SKIP] {parl} -- {len(df_p)} utts < {PARL_MIN_ROWS}")
            continue
        if df_p[VARS["cluster"]].nunique() < PARL_MIN_SPEAKERS:
            print(f"  [SKIP] {parl} -- "
                  f"{df_p[VARS['cluster']].nunique()} speakers "
                  f"< {PARL_MIN_SPEAKERS}")
            continue
        log_exp_p = np.log(df_p[VARS["exposure"]])
        run_scope(df_p, log_exp_p, parl)


# =================================================================
# DONE
# =================================================================

print(f"\n{'='*60}")
print(f"  Done.  Results in: {RESULTS_DIR}")
print(f"  Timestamp: {TIMESTAMP}")
print(f"{'='*60}\n")