"""
utils_GEE.py
============
Helpers for the FP rate modelling pipeline (used by 4_run_GEE_suite.py).

Model: Negative Binomial GEE/GLM (log link), with log(audio_length) offset.
Unit of analysis: utterance (one row = one utterance).

Three specification tracks:
  1. Baseline    — no Mundlak correction
  2. Mundlak     — each time-varying predictor decomposed into between/within
                   (rate, sentiment, party status, orientation)
  3. Interactions — baseline + one interaction term per model (×7)

Mundlak: per-variable sub-toggles (rate, sentiment, party_status, orientation).
"""

import warnings, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import nbinom
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy


# ─────────────────────────────────────────────────────────────
# FILTERING
# ─────────────────────────────────────────────────────────────

def _step_report(label, n_before, n_after, target_col, df_before, df_after):
    dropped = n_before - n_after
    z_b = (df_before[target_col] == 0).sum()
    z_a = (df_after[target_col] == 0).sum()
    pz_b = 100 * z_b / max(n_before, 1)
    pz_a = 100 * z_a / max(n_after, 1)
    print(f"  [{label}]")
    print(f"    Rows : {n_before:>9,} -> {n_after:>9,}  (-{dropped:,})")
    print(f"    Zeros: {z_b:>9,} ({pz_b:.1f}%) -> {z_a:>9,} ({pz_a:.1f}%)")


def filter_data(df, v, word_min=10, audio_min=3.0, role="Regular"):
    """
    Filtering pipeline for utterance-level FP count data.
      1. Speaker role filter
      2. word_count > word_min
      3. audio_length > audio_min
      4. Numeric/categorical coercion
      5. Drop rows missing any core model variable
      6. Drop zero-length clips
    Returns cleaned DataFrame.
    """
    target = v["target"]
    exp    = v["exposure"]

    print("\n" + "=" * 60)
    print("FILTERING PIPELINE")
    print("=" * 60)

    df = df.copy()
    df[target] = pd.to_numeric(df[target], errors="coerce")

    n0 = len(df)
    z0 = (df[target] == 0).sum()
    print(f"  [Raw] rows={n0:,}  zero-FP={z0:,} ({100*z0/n0:.1f}%)\n")

    # Speaker role
    if "Speaker_role" in df.columns and role:
        nb = len(df); prev = df.copy()
        df = df[df["Speaker_role"] == role]
        _step_report(f"Speaker_role == '{role}'", nb, len(df), target, prev, df)

    # Word count
    wc = "word_count"
    if wc in df.columns:
        df[wc] = pd.to_numeric(df[wc], errors="coerce")
        nb = len(df); prev = df.copy()
        df = df[df[wc] > word_min]
        _step_report(f"word_count > {word_min}", nb, len(df), target, prev, df)

    # Audio length
    df[exp] = pd.to_numeric(df[exp], errors="coerce")
    nb = len(df); prev = df.copy()
    df = df[df[exp] > audio_min]
    _step_report(f"audio_length > {audio_min}s", nb, len(df), target, prev, df)

    # Coerce numeric columns
    numeric_cols = [
        v["age"], v["rate"], v["orientation"], v["sentiment"], v["exposure"],
    ]
    for col in numeric_cols:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Coerce categorical columns
    for col in [v["gender"], v["grouping"]]:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(
                ["nan", "None", "-", ""], np.nan)

    # Drop rows missing any core model variable
    core = [
        v["target"], v["gender"], v["age"], v["rate"],
        v["sentiment"], v["party_status"], v["orientation"],
        v["grouping"], v["cluster"], v["exposure"],
    ]
    core = [c for c in core if c in df.columns]
    nb = len(df); prev = df.copy()
    df = df.dropna(subset=core)
    _step_report("Drop missing core model vars", nb, len(df), target, prev, df)

    # Drop zero-length clips
    nb = len(df); prev = df.copy()
    df = df[df[exp] > 0]
    _step_report("Drop zero-length clips", nb, len(df), target, prev, df)

    df = df.copy()  # defrag

    nf = len(df)
    zf = (df[target] == 0).sum()
    total_audio = df[exp].sum()
    print(f"\n  [Final] rows={nf:,}  zero-FP={zf:,} ({100*zf/nf:.1f}%)")
    print(f"  [Speakers] {df[v['cluster']].nunique():,}")
    print(f"  [Total Audio] {total_audio:,.2f}s ({total_audio/3600:,.2f}h)")
    print("=" * 60 + "\n")

    return df


def cluster_size_check(df, v, min_obs=10, apply_filter=False):
    col = v["cluster"]
    counts = df[col].value_counts()
    small = counts[counts < min_obs]
    print(f"\n  Cluster size check (min={min_obs} utterances/speaker):")
    print(f"    Speakers < {min_obs} utts : {len(small):,} / {len(counts):,}")
    print(f"    Rows affected            : {df[col].isin(small.index).sum():,} / {len(df):,}")
    if apply_filter:
        df = df[~df[col].isin(small.index)].copy()
        print(f"    -> APPLIED. Remaining: {len(df):,} rows, "
              f"{df[col].nunique():,} speakers.")
    else:
        print(f"    -> NOT applied.")
    return df


def party_switch_diagnostic(df, v):
    cid = v["cluster"]
    print("\n  Party consistency diagnostic:")
    for col, lbl in [(v["party_status"], "Party_status"),
                     (v["orientation"],  "Party_orientation_float")]:
        if col not in df.columns:
            continue
        n_vals = df.groupby(cid)[col].nunique()
        switchers = n_vals[n_vals > 1]
        pct = 100 * len(switchers) / max(len(n_vals), 1)
        n_rows = df[df[cid].isin(switchers.index)].shape[0]
        print(f"    {lbl}: {len(switchers):,} speakers switch "
              f"({pct:.1f}% of speakers, {n_rows:,} rows)")


# ─────────────────────────────────────────────────────────────
# FORMULA BUILDERS
# ─────────────────────────────────────────────────────────────

def _age_term(v, zscore_age, age_mean):
    if zscore_age:
        return f"scale({v['age']})"
    if age_mean is None:
        raise ValueError("age_mean required when zscore_age=False")
    return f"I(({v['age']} - {age_mean:.6f}) / 10)"


def _scale_or_raw(col, do_scale):
    return f"scale({col})" if do_scale else col


def _mundlak_terms_rate(v, mundlak_rate):
    if mundlak_rate:
        return [f"scale({v['rate_mean']})", f"scale({v['rate_dev']})"]
    return [f"scale({v['rate']})"]


def _mundlak_terms_sentiment(v, mundlak_sentiment, scale_sentiment):
    if mundlak_sentiment:
        return [
            _scale_or_raw(v["sentiment_mean"], scale_sentiment),
            _scale_or_raw(v["sentiment_dev"],  scale_sentiment),
        ]
    return [_scale_or_raw(v["sentiment"], scale_sentiment)]


def _mundlak_terms_party_status(v, mundlak_party_status):
    if mundlak_party_status:
        return [v["party_status_mean"], v["party_status_dev"]]
    return [v["party_status"]]


def _mundlak_terms_orientation(v, mundlak_orientation,
                               scale_orientation, use_abs_orientation):
    """Decomposed orientation; abs-transform is incompatible with Mundlak."""
    if mundlak_orientation:
        if use_abs_orientation:
            raise ValueError(
                "use_abs_orientation=True is incompatible with "
                "mundlak_orientation=True (no abs decomposition in TSV)."
            )
        return [
            _scale_or_raw(v["orientation_mean"], scale_orientation),
            _scale_or_raw(v["orientation_dev"],  scale_orientation),
        ]
    if use_abs_orientation:
        raw = f"np.abs({v['orientation']})"
        return [f"scale({raw})" if scale_orientation else raw]
    return [_scale_or_raw(v["orientation"], scale_orientation)]


def _core_rhs_terms(v, use_abs_orientation, zscore_age, age_mean,
                    mundlak_rate, mundlak_sentiment,
                    mundlak_party_status, mundlak_orientation,
                    scale_sentiment, scale_orientation,
                    include_lang=True):
    """Build the shared RHS predictor list (no interactions)."""
    at = _age_term(v, zscore_age, age_mean)
    rate_terms   = _mundlak_terms_rate(v, mundlak_rate)
    sent_terms   = _mundlak_terms_sentiment(v, mundlak_sentiment, scale_sentiment)
    status_terms = _mundlak_terms_party_status(v, mundlak_party_status)
    orient_terms = _mundlak_terms_orientation(
        v, mundlak_orientation, scale_orientation, use_abs_orientation)

    terms = [
        v["gender"],
        at,
        *rate_terms,
        *sent_terms,
        *status_terms,
        *orient_terms,
    ]

    if include_lang:
        terms.append(f"C({v['grouping']})")
    return terms


def build_formula(v, **kwargs):
    """NB count formula. Target: fp_count. Offset passed separately."""
    terms = _core_rhs_terms(v, **kwargs)
    return f"{v['target']} ~ " + " + ".join(terms)


def build_formula_interaction(v, interaction_term, **kwargs):
    """Baseline formula + one interaction term."""
    terms = _core_rhs_terms(v, **kwargs)
    terms.append(interaction_term)
    return f"{v['target']} ~ " + " + ".join(terms)


def get_interaction_specs(v, zscore_age, age_mean):
    """
    Returns list of (label, interaction_term) for each interaction model.
    All use non-Mundlak predictors.
    """
    at = _age_term(v, zscore_age, age_mean)
    return [
        ("gender_x_rate",
         f"{v['gender']} : scale({v['rate']})"),
        ("gender_x_power",
         f"{v['gender']} : {v['party_status']}"),
        ("age_x_power",
         f"{at} : {v['party_status']}"),
        ("sentiment_x_power",
         f"{v['sentiment']} : {v['party_status']}"),
        ("gender_x_orientation",
         f"{v['gender']} : {v['orientation']}"),
        ("age_x_orientation",
         f"{at} : {v['orientation']}"),
        ("sentiment_x_orientation",
         f"{v['sentiment']} : {v['orientation']}"),
    ]


# ─────────────────────────────────────────────────────────────
# REFERENCE LEVELS & SD CONTEXT
# ─────────────────────────────────────────────────────────────

def get_reference_levels(df, v):
    refs = {}
    for key in ("gender", "party_status", "grouping"):
        col = v[key]
        vals = df[col].dropna().unique()
        refs[col] = sorted(str(x) for x in vals)[0]
    return refs


def format_reference_levels(refs, v):
    lines = ["", "Reference levels:"]
    for key in ("gender", "party_status", "grouping"):
        col = v[key]
        lines.append(f"  {col:<35} ref = {refs.get(col, 'N/A')}")
    lines += [
        "  Sentiment_label_float           -2.5=negative  +2.5=positive",
        "  Party_orientation_float         -3=left  0=center  +3=right", ""
    ]
    return lines


def get_sd_context(df, v):
    cols = [v["age"], v["rate"], v["sentiment"], v["orientation"]]
    return {col: float(df[col].std()) for col in cols if col in df.columns}


def format_sd_context(sd_ctx, v, zscore_age):
    units = {
        v.get("age"):         "years",
        v.get("rate"):        "sylls/s or words/s",
        v.get("sentiment"):   "units (-2.5 to +2.5)",
        v.get("orientation"): "units (-3 to +3)",
    }
    age_note = ("z-scored -> coeff = per 1 SD" if zscore_age
                else "mean-centred -> coeff = per 10 years")
    lines = ["", f"Predictor SDs  [age: {age_note}]:"]
    for col, sd in sd_ctx.items():
        u = units.get(col, "units")
        lines.append(f"  {col:<35} 1 SD = {sd:.3f} {u}")
    lines.append("")
    return lines


# ─────────────────────────────────────────────────────────────
# NB ALPHA ESTIMATION
# ─────────────────────────────────────────────────────────────

def estimate_nb_alpha(formula, df, log_exposure, maxiter=300):
    """Estimate NB dispersion parameter via MLE."""
    # Drop constant columns (can happen in per-parliament scopes)
    all_constants = [c for c in df.columns if df[c].nunique() <= 1]
    to_drop = [c for c in all_constants if c != "Lang"]

    if to_drop:
        print(f"    [WARNING] Dropping constant columns: {to_drop}")
        df = df.drop(columns=to_drop, errors="ignore")
        for c in to_drop:
            formula = re.sub(rf'\b{c}\b\s*\+?', '', formula)
        formula = re.sub(r'\+\s*\+', '+', formula)
        formula = re.sub(r'~\s*\+', '~', formula)
        formula = formula.strip().rstrip('+').strip()
        print(f"    [INFO] Adjusted formula: {formula}")

    print("\n" + "=" * 60)
    print("Estimating NB alpha ...")

    try:
        exposure_arr = np.asarray(log_exposure).flatten()
        y, X = patsy.dmatrices(formula, data=df, return_type="dataframe")
        nb_model = sm.NegativeBinomial(
            endog=y.values.ravel(), exog=X,
            offset=exposure_arr, loglike_method="nb2",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nb_res = nb_model.fit(maxiter=maxiter, disp=False, method="lbfgs")

        if "alpha" not in nb_res.params:
            raise ValueError("Alpha not found in model parameters.")

        alpha = float(nb_res.params["alpha"])
        converged = getattr(nb_res, "mle_retvals", {}).get("converged", True)
        print(f"  Converged : {converged}  |  Alpha : {alpha:.6f}")

        if not converged:
            print("  [WARN] Did not converge — using alpha=1.0")
            return 1.0
        if alpha <= 0:
            print("  [WARN] alpha <= 0 — using alpha=0.01")
            return 0.01

        interp = ("near-Poisson" if alpha < 0.1
                  else "moderate overdispersion" if alpha < 1.0
                  else "heavy overdispersion")
        print(f"  Interpretation: {interp}")
        return alpha

    except Exception as e:
        print(f"  [ERROR] alpha estimation failed: {e} — using alpha=1.0")
        return 1.0


# ─────────────────────────────────────────────────────────────
# MODEL FITTING — GENERIC SUITE
# ─────────────────────────────────────────────────────────────

def run_model_suite(formula, df, log_exposure, alpha, cluster_col,
                    glm_cluster_se=True, on_model_ready=None,
                    track_label=""):
    """
    Fits GLM_NB, GLM_NB_Cluster, GEE_NB_Indep, GEE_NB_Exch.
    Calls on_model_ready(label, fit) immediately after each model.
    """
    results = {}
    family = sm.families.NegativeBinomial(alpha=alpha)
    prefix = f"{track_label}_" if track_label else ""

    def _emit(label, fit):
        full_label = f"{prefix}{label}"
        results[full_label] = fit
        if on_model_ready is not None:
            try:
                on_model_ready(full_label, fit)
            except Exception as e:
                print(f"  [WARN] on_model_ready failed for {full_label}: {e}")

    # GLM
    print(f"  Fitting {prefix}GLM_NB ...")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            glm_fit = smf.glm(
                formula, data=df, offset=log_exposure, family=family,
            ).fit(disp=False)
        print(f"    Converged: {glm_fit.converged}")
        _emit("GLM_NB", glm_fit)
    except Exception as e:
        print(f"    [ERROR] {prefix}GLM_NB failed: {e}")

    # GLM with clustered SEs
    if glm_cluster_se:
        print(f"  Fitting {prefix}GLM_NB_Cluster (on {cluster_col}) ...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                glm_c = smf.glm(
                    formula, data=df, offset=log_exposure, family=family,
                ).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": df[cluster_col].values},
                    disp=False,
                )
            print(f"    Done. N clusters = {df[cluster_col].nunique():,}")
            _emit("GLM_NB_Cluster", glm_c)
        except np.linalg.LinAlgError as e:
            print(f"    [WARNING] Singular matrix, falling back to HC3: {e}")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    glm_c = smf.glm(
                        formula, data=df, offset=log_exposure, family=family,
                    ).fit(cov_type="HC3", disp=False)
                _emit("GLM_NB_Cluster", glm_c)
            except Exception as e2:
                print(f"    [ERROR] HC3 fallback also failed: {e2}")
        except Exception as e:
            print(f"    [ERROR] {prefix}GLM_NB_Cluster failed: {e}")

    # GEE models
    for gee_label, cov_struct in [
        ("GEE_NB_Indep", sm.cov_struct.Independence()),
        ("GEE_NB_Exch",  sm.cov_struct.Exchangeable()),
    ]:
        print(f"  Fitting {prefix}{gee_label} ...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.gee(
                    formula,
                    groups=df[cluster_col],
                    data=df,
                    offset=log_exposure,
                    family=family,
                    cov_struct=cov_struct,
                ).fit()
            print(f"    Done.")
            _emit(gee_label, fit)
        except Exception as e:
            print(f"    [ERROR] {prefix}{gee_label} failed: {e}")

    return results


# ─────────────────────────────────────────────────────────────
# PLAIN-ENGLISH INTERPRETATION
# ─────────────────────────────────────────────────────────────

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "[n.s.]"


def _safe_get(params, pvalues, key):
    try:
        if key in params.index:
            return float(params[key]), float(pvalues[key])
    except Exception:
        pass
    return None, None


def plain_english_block(fit, model_label, v, refs, rate_suffix,
                        sd_ctx, use_abs_orientation, zscore_age,
                        scale_sentiment=False, scale_orientation=False,
                        mundlak_rate=False, mundlak_sentiment=False,
                        mundlak_party_status=False, mundlak_orientation=False):
    """Plain-english interpretation for NB count model."""
    try:
        lines = [
            "", "=" * 60,
            f"INTERPRETATION -- {model_label}  [{rate_suffix}]",
            f"Model: NB (log link + offset)  |  exp(coef) = IRR on FP count",
            f"Target: fp_count  (one row = one utterance)",
            "=" * 60,
            "*** p<.001  ** p<.01  * p<.05  [n.s.] p>=.05", "",
        ]
        lines += _interpret_predictor_block(
            fit.params, fit.pvalues,
            v=v, refs=refs, rate_suffix=rate_suffix, sd_ctx=sd_ctx,
            use_abs_orientation=use_abs_orientation, zscore_age=zscore_age,
            scale_sentiment=scale_sentiment, scale_orientation=scale_orientation,
            mundlak_rate=mundlak_rate, mundlak_sentiment=mundlak_sentiment,
            mundlak_party_status=mundlak_party_status,
            mundlak_orientation=mundlak_orientation,
            exp_func_name="IRR",
            direction_phrase=lambda r: f"{abs((r-1)*100):.1f}% "
                                       f"{'more' if r >= 1 else 'fewer'} FPs",
        )
        return lines
    except Exception as e:
        return [f"[plain_english_block failed: {e}]"]


def _interpret_predictor_block(params, pvalues, v, refs, sd_ctx,
                               rate_suffix, use_abs_orientation, zscore_age,
                               scale_sentiment, scale_orientation,
                               mundlak_rate, mundlak_sentiment,
                               mundlak_party_status, mundlak_orientation,
                               exp_func_name="IRR",
                               direction_phrase=None):
    if direction_phrase is None:
        direction_phrase = lambda r: f"{abs((r-1)*100):.1f}% {'more' if r >= 1 else 'fewer'}"

    lines = []

    # Gender
    try:
        coef, p = _safe_get(params, pvalues, f"{v['gender']}[T.M]")
        if coef is not None:
            ev = np.exp(coef)
            lines += [f"GENDER (ref={refs.get(v['gender'], '?')}):",
                      f"  Men -> {direction_phrase(ev)}  "
                      f"({exp_func_name}={ev:.3f}, {sig_stars(p)})", ""]
    except Exception as e:
        lines.append(f"  [WARN] Gender block failed: {e}")

    # Speech rate
    try:
        sd_rate = sd_ctx.get(v["rate"])
        if mundlak_rate:
            for col_key, lbl in [
                (f"scale({v['rate_mean']})", "SPEECH RATE (between-speaker mean)"),
                (f"scale({v['rate_dev']})",  "SPEECH RATE (within-speaker dev)"),
            ]:
                coef, p = _safe_get(params, pvalues, col_key)
                if coef is not None:
                    ev = np.exp(coef)
                    lines += [f"{lbl}:",
                              f"  +1 SD -> {direction_phrase(ev)}  "
                              f"({exp_func_name}={ev:.3f}, {sig_stars(p)})", ""]
        else:
            coef, p = _safe_get(params, pvalues, f"scale({v['rate']})")
            if coef is not None:
                sd_str = f"1 SD = {sd_rate:.3f}" if sd_rate else ""
                ev = np.exp(coef)
                lines += [f"SPEECH RATE [{rate_suffix}]  ({sd_str}):",
                          f"  +1 SD -> {direction_phrase(ev)}  "
                          f"({exp_func_name}={ev:.3f}, {sig_stars(p)})", ""]
    except Exception as e:
        lines.append(f"  [WARN] Speech rate block failed: {e}")

    # Sentiment
    try:
        if mundlak_sentiment:
            for col_key, lbl in [
                (v["sentiment_mean"] if not scale_sentiment
                 else f"scale({v['sentiment_mean']})",
                 "SENTIMENT (between-speaker mean)"),
                (v["sentiment_dev"] if not scale_sentiment
                 else f"scale({v['sentiment_dev']})",
                 "SENTIMENT (within-speaker dev)"),
            ]:
                coef, p = _safe_get(params, pvalues, col_key)
                if coef is not None:
                    ev = np.exp(coef)
                    lines += [f"{lbl}:",
                              f"  +1 unit -> {direction_phrase(ev)}  "
                              f"({exp_func_name}={ev:.3f}, {sig_stars(p)})", ""]
        else:
            sent_key = (f"scale({v['sentiment']})" if scale_sentiment
                        else v["sentiment"])
            coef, p = _safe_get(params, pvalues, sent_key)
            if coef is not None:
                ev = np.exp(coef)
                unit_str = "1 SD" if scale_sentiment else "1 unit"
                lines += [f"SENTIMENT:",
                          f"  +{unit_str} -> {direction_phrase(ev)}  "
                          f"({exp_func_name}={ev:.3f}, {sig_stars(p)})", ""]
    except Exception as e:
        lines.append(f"  [WARN] Sentiment block failed: {e}")

    # Party status
    try:
        if mundlak_party_status:
            for col_key, lbl in [
                (v["party_status_mean"], "PARTY STATUS (between-speaker mean)"),
                (v["party_status_dev"],  "PARTY STATUS (within-speaker dev)"),
            ]:
                coef, p = _safe_get(params, pvalues, col_key)
                if coef is not None:
                    ev = np.exp(coef)
                    lines += [f"{lbl}:",
                              f"  +1 unit -> {direction_phrase(ev)}  "
                              f"({exp_func_name}={ev:.3f}, {sig_stars(p)})", ""]
        else:
            coef, p = _safe_get(params, pvalues, v["party_status"])
            if coef is not None:
                ev = np.exp(coef)
                lines += [f"PARTY STATUS (Coalition=0, Opposition=1):",
                          f"  Opposition -> {direction_phrase(ev)}  "
                          f"({exp_func_name}={ev:.3f}, {sig_stars(p)})", ""]
    except Exception as e:
        lines.append(f"  [WARN] Party status block failed: {e}")

    # Orientation
    try:
        if mundlak_orientation:
            unit_str = "1 SD" if scale_orientation else "1 unit"
            for col_base, lbl in [
                (v["orientation_mean"], "PARTY ORIENTATION (between-speaker mean)"),
                (v["orientation_dev"],  "PARTY ORIENTATION (within-speaker dev)"),
            ]:
                col_key = _scale_or_raw(col_base, scale_orientation)
                coef, p = _safe_get(params, pvalues, col_key)
                if coef is not None:
                    ev = np.exp(coef)
                    lines += [f"{lbl}:",
                              f"  +{unit_str} -> {direction_phrase(ev)}  "
                              f"({exp_func_name}={ev:.3f}, {sig_stars(p)})", ""]
        else:
            if use_abs_orientation:
                orient_key = (f"scale(np.abs({v['orientation']}))"
                              if scale_orientation
                              else f"np.abs({v['orientation']})")
            else:
                orient_key = (f"scale({v['orientation']})"
                              if scale_orientation else v["orientation"])
            coef, p = _safe_get(params, pvalues, orient_key)
            if coef is not None:
                ev = np.exp(coef)
                unit_str = "1 SD" if scale_orientation else "1 unit"
                lines += [f"PARTY ORIENTATION:",
                          f"  +{unit_str} -> {direction_phrase(ev)}  "
                          f"({exp_func_name}={ev:.3f}, {sig_stars(p)})", ""]
    except Exception as e:
        lines.append(f"  [WARN] Orientation block failed: {e}")

    # Age
    try:
        sd_age = sd_ctx.get(v["age"])
        age_key = f"scale({v['age']})" if zscore_age else None
        if age_key is None or age_key not in params.index:
            age_key = next((k for k in params.index if v["age"] in k), None)
        coef, p = (_safe_get(params, pvalues, age_key)
                   if age_key else (None, None))
        if coef is not None:
            ev = np.exp(coef)
            if zscore_age and sd_age:
                per_yr = abs((ev - 1) * 100 / sd_age)
                lines += [f"AGE  (1 SD = {sd_age:.1f} yrs):",
                          f"  +1 SD   -> {direction_phrase(ev)}  "
                          f"({exp_func_name}={ev:.3f}, {sig_stars(p)})",
                          f"  ~ {per_yr:.2f}% per year  |  "
                          f"10 yrs ~ {per_yr*10:.1f}%", ""]
            else:
                lines += [f"AGE (per decade):",
                          f"  +10 yrs -> {direction_phrase(ev)}  "
                          f"({exp_func_name}={ev:.3f}, {sig_stars(p)})", ""]
    except Exception as e:
        lines.append(f"  [WARN] Age block failed: {e}")

    # Language
    try:
        lang_keys = [k for k in params.index
                     if k.startswith(f"C({v['grouping']})[T.")]
        if lang_keys:
            lines.append(f"LANGUAGE (ref={refs.get(v['grouping'], '?')}):")
            for k in lang_keys:
                lang = k.split("[T.")[-1].rstrip("]")
                ev = np.exp(float(params[k]))
                p = float(pvalues[k])
                lines.append(f"  {lang:<6} -> {direction_phrase(ev)}  "
                             f"({exp_func_name}={ev:.3f}, {sig_stars(p)})")
            lines.append("")
    except Exception as e:
        lines.append(f"  [WARN] Language block failed: {e}")

    # Interaction terms (catch-all for any ":" terms not yet handled)
    try:
        interaction_keys = [k for k in params.index if ":" in k]
        if interaction_keys:
            lines.append("INTERACTION TERMS:")
            for k in interaction_keys:
                coef, p = _safe_get(params, pvalues, k)
                if coef is not None:
                    ev = np.exp(coef)
                    lines.append(f"  {k}:")
                    lines.append(f"    {exp_func_name}={ev:.4f}  "
                                 f"({sig_stars(p)}, p={p:.4f})")
            lines.append("")
    except Exception as e:
        lines.append(f"  [WARN] Interaction block failed: {e}")

    return lines


# ─────────────────────────────────────────────────────────────
# SAVE — one file per model, written immediately
# ─────────────────────────────────────────────────────────────

def save_single_model(label, fit, pe_lines, sd_ctx, refs,
                      out_dir, rate_suffix, scope_label, timestamp,
                      formula, alpha, v,
                      use_abs_orientation, zscore_age,
                      scale_sentiment, scale_orientation,
                      mundlak_rate, mundlak_sentiment,
                      mundlak_party_status, mundlak_orientation,
                      use_mundlak,
                      gee_cluster_by,
                      n_rows=None, n_speakers=None,
                      interaction_label=None):
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{label}_{rate_suffix}_{timestamp}.txt"
    fpath = os.path.join(out_dir, fname)

    with open(fpath, "w", encoding="utf-8") as f:
        def w(x):
            lns = [x] if isinstance(x, str) else list(x)
            for line in lns:
                f.write(str(line) + "\n")

        w(f"Model              : {label}")
        w(f"Scope              : {scope_label}")
        w(f"Timestamp          : {timestamp}")
        w(f"Rate type          : {rate_suffix}")
        w(f"GEE cluster by     : {gee_cluster_by}")
        if alpha is not None:
            w(f"NB alpha           : {alpha:.6f}")
        w(f"Abs orient.        : {use_abs_orientation}")
        w(f"Z-score age        : {zscore_age}  (False -> per decade)")
        w(f"Scale sentiment    : {scale_sentiment}")
        w(f"Scale orientation  : {scale_orientation}")
        w(f"USE_MUNDLAK        : {use_mundlak}")
        w(f"  Mundlak rate         : {mundlak_rate}")
        w(f"  Mundlak sentiment    : {mundlak_sentiment}")
        w(f"  Mundlak party status : {mundlak_party_status}")
        w(f"  Mundlak orientation  : {mundlak_orientation}")
        if interaction_label:
            w(f"Interaction        : {interaction_label}")
        if n_rows is not None:
            w(f"N rows (this model): {n_rows:,}")
        if n_speakers is not None:
            w(f"N speakers         : {n_speakers:,}")
        if formula:
            w(f"Formula            : {formula}")
        w(format_reference_levels(refs, v))
        w(format_sd_context(sd_ctx, v, zscore_age))

        w(f"\n{'='*60}\n{label}\n{'='*60}")

        try:
            w(fit.summary().as_text())
        except Exception as e:
            w(f"  [ERROR] summary failed: {e}")

        # IRR table
        try:
            irr_tbl = pd.DataFrame({
                "IRR":   np.exp(fit.params),
                "CI_lo": np.exp(fit.conf_int()[0]),
                "CI_hi": np.exp(fit.conf_int()[1]),
                "p":     fit.pvalues,
            })
            w("\nIRR TABLE:")
            w(irr_tbl.to_string(float_format="{:.4f}".format))
        except Exception as e:
            w(f"  [ERROR] IRR table failed: {e}")

        # GEE exchangeable alpha
        try:
            if "Exch" in label and hasattr(fit, "cov_struct"):
                ac = fit.cov_struct.dep_params
                w(f"\nExchangeable alpha = {ac:.4f}")
                w("  -> " + ("very weak" if ac < 0.1
                             else "moderate" if ac < 0.3
                             else "strong -- dominant fingerprint"))
        except Exception as e:
            w(f"  [ERROR] GEE alpha write failed: {e}")

        w("\n" + "=" * 60 + "\nPLAIN-ENGLISH INTERPRETATION\n" + "=" * 60)
        w(pe_lines if pe_lines else ["[no interpretation available]"])

    print(f"  Saved: {fpath}")
    return fpath


# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────

def plot_count_distribution(df, v, label, out_dir, timestamp):
    try:
        target = v["target"]
        counts = df[target].dropna().astype(int)
        max_k  = min(int(counts.max()), 50)
        bins   = np.arange(-0.5, max_k + 1.5)
        zero_pct = 100 * (counts == 0).mean()
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"FP Count Distribution -- {label}", fontsize=12)
        for ax, yscale in zip(axes, ["linear", "log"]):
            ax.hist(counts.clip(upper=max_k), bins=bins,
                    color="#2980b9", edgecolor="white", linewidth=0.3)
            ax.axvline(0, color="#c0392b", linestyle="--", linewidth=1.5,
                       label=f"Zero: {zero_pct:.1f}%")
            ax.set_yscale(yscale)
            ax.set_xlabel("Filled pause count (clipped at 50)")
            ax.set_ylabel("Frequency" + (" (log)" if yscale == "log" else ""))
            ax.legend(fontsize=9)
        plt.tight_layout()
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, f"dist_{label}_{timestamp}.png")
        plt.savefig(fname, dpi=150); plt.close()
        print(f"  Saved: {fname}")
    except Exception as e:
        print(f"  [WARN] plot_count_distribution failed: {e}")
        plt.close("all")


def plot_irr_forest(fit, title, out_dir, rate_suffix, timestamp,
                    p_thresh=0.05):
    try:
        if not hasattr(fit, "params"):
            return
        params = fit.params.drop(["Intercept"], errors="ignore")
        ci = fit.conf_int().loc[params.index]
        pvals = fit.pvalues.loc[params.index]
        irr = np.exp(params)
        colors = ["#c0392b" if float(p) < p_thresh else "#7f8c8d"
                  for p in pvals]
        fig, ax = plt.subplots(
            figsize=(10, max(6, len(params) * 0.55)))
        for i, (name, val, lo, hi, col) in enumerate(
                zip(params.index, irr, np.exp(ci[0]), np.exp(ci[1]),
                    colors)):
            ax.plot([lo, hi], [i, i], color=col, linewidth=2, zorder=1)
            ax.scatter(val, i, color=col, s=55, zorder=2)
        ax.axvline(1.0, color="#2c3e50", linestyle="--", linewidth=1.5)
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels(params.index, fontsize=9)
        ax.set_xscale("log"); ax.set_xlabel("IRR (log scale)")
        ax.set_title(f"{title}  [{rate_suffix}]  red=p<{p_thresh}",
                     fontsize=11, pad=10)
        ax.grid(True, axis="x", linestyle=":", alpha=0.5)
        plt.tight_layout()
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(
            out_dir,
            f"{title.replace(' ', '_')}_{rate_suffix}_{timestamp}.png")
        plt.savefig(fname, dpi=160); plt.close()
        print(f"  Saved: {fname}")
    except Exception as e:
        print(f"  [WARN] plot_irr_forest failed: {e}")
        plt.close("all")


def plot_partial_residuals(fit, df, v, out_dir, label, timestamp,
                           rate_suffix):
    try:
        if not hasattr(fit, "params"):
            return
        continuous = {
            f"scale({v['age']})":         (v["age"],       "Age"),
            f"scale({v['rate']})":        (v["rate"],
                                           f"Speech Rate [{rate_suffix}]"),
            v["sentiment"]:               (v["sentiment"], "Sentiment"),
            f"scale({v['sentiment']})":   (v["sentiment"],
                                           "Sentiment (scaled)"),
            v["orientation"]:             (v["orientation"], "Orientation"),
            f"scale({v['orientation']})": (v["orientation"],
                                           "Orientation (scaled)"),
        }
        valid = [(k, col, lbl) for k, (col, lbl) in continuous.items()
                 if k in fit.params.index and col in df.columns]
        if not valid:
            return

        fig, axes = plt.subplots(1, len(valid),
                                 figsize=(5 * len(valid), 5))
        if len(valid) == 1:
            axes = [axes]
        fig.suptitle(f"Partial Residuals -- {label}\n"
                     f"Flat LOESS = log-linear OK", fontsize=10)

        raw_resid = (fit.resid_response if hasattr(fit, "resid_response")
                     else fit.resid_deviance)

        for ax, (pname, col, lbl) in zip(axes, valid):
            try:
                x = df[col].values
                xs = (x - x.mean()) / x.std()
                pr = raw_resid + fit.params[pname] * xs
                n = len(x)
                if n > 20_000:
                    idx = np.random.choice(n, 20_000, replace=False)
                    xp, yp = x[idx], pr[idx]
                else:
                    xp, yp = x, pr
                ax.scatter(xp, yp, alpha=0.04, s=3, color="#7f8c8d",
                           rasterized=True)
                try:
                    sm_ = lowess(yp, xp, frac=0.3, return_sorted=True)
                    ax.plot(sm_[:, 0], sm_[:, 1], color="#c0392b",
                            linewidth=2, label="LOESS")
                except Exception:
                    pass
                ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
                ax.set_xlabel(lbl)
                ax.set_ylabel("Partial residual")
                ax.legend(fontsize=8)
            except Exception as e:
                ax.set_title(f"[failed: {e}]")

        plt.tight_layout()
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(
            out_dir, f"partial_resid_{label}_{timestamp}.png")
        plt.savefig(fname, dpi=130); plt.close()
        print(f"  Saved: {fname}")
    except Exception as e:
        print(f"  [WARN] plot_partial_residuals failed: {e}")
        plt.close("all")