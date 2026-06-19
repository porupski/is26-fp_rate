"""
utils_results.py
================
Helpers for collecting and plotting model results
(used by 5_collect_results_and_plot.py and 6_plot_paper_figure.py).

Contains:
  - Label parsing (track / model_type decomposition)
  - Per-log .txt parser
  - Predictor display-name mapping & ordering
  - Forest-plot subplot rendering (single panel, legend, group bands)
"""

import os
import re
import io

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


# =================================================================
# LABEL PARSING
# =================================================================

_MODEL_SUFFIXES = [
    "GLM_NB_Cluster",
    "GEE_NB_Indep",
    "GEE_NB_Exch",
    "GLM_NB",
    "GLM_Cluster",
    "GEE_Indep",
    "GEE_Exch",
    "GLM",
]


def parse_label(full_label):
    """
    Split a full model label into (track, model_type).

    Examples:
      'BASE_GLM_NB'                      -> ('BASE', 'GLM_NB')
      'MUNDLAK_GEE_NB_Exch'             -> ('MUNDLAK', 'GEE_NB_Exch')
      'INT_GENDER_X_RATE_GEE_NB_Indep'  -> ('INT_GENDER_X_RATE', 'GEE_NB_Indep')
      'GAMMA_GLM'                        -> ('GAMMA', 'GLM')
      'GAMMA_GEE_Exch'                   -> ('GAMMA', 'GEE_Exch')
    """
    for suffix in _MODEL_SUFFIXES:
        if full_label.endswith(suffix):
            prefix = full_label[:-(len(suffix))].rstrip("_")
            return (prefix or full_label, suffix)
    return (full_label, full_label)


# =================================================================
# LOG-FILE PARSER
# =================================================================

def _extract_header_value(lines, key):
    pattern = re.compile(r"^" + re.escape(key) + r"\s*:\s*(.+)$")
    for line in lines:
        m = pattern.match(line.rstrip())
        if m:
            return m.group(1).strip()
    return None


def parse_log_file(filepath):
    """Parse a single per-model log .txt file. Returns list of dicts."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
        lines = text.split("\n")

    # ── Metadata ──────────────────────────────────────────────
    full_label = _extract_header_value(lines, "Model") or "Unknown"
    parliament = _extract_header_value(lines, "Scope") or "unknown"
    timestamp  = _extract_header_value(lines, "Timestamp") or ""
    rate_type  = _extract_header_value(lines, "Rate type") or ""
    formula    = _extract_header_value(lines, "Formula") or ""

    nb_alpha_raw = _extract_header_value(lines, "NB alpha") or ""
    nb_alpha = ""
    if nb_alpha_raw:
        m = re.match(r"([\d.]+)", nb_alpha_raw)
        nb_alpha = m.group(1) if m else nb_alpha_raw

    use_mundlak          = _extract_header_value(lines, "USE_MUNDLAK") or ""
    mundlak_rate         = _extract_header_value(lines, "Mundlak rate") or ""
    mundlak_sentiment    = _extract_header_value(lines, "Mundlak sentiment") or ""
    mundlak_party_status = _extract_header_value(lines, "Mundlak party status") or ""
    mundlak_orientation  = _extract_header_value(lines, "Mundlak orientation") or ""
    scale_sentiment      = _extract_header_value(lines, "Scale sentiment") or ""
    scale_orientation    = _extract_header_value(lines, "Scale orientation") or ""
    interaction_label    = _extract_header_value(lines, "Interaction") or ""
    gee_cluster          = _extract_header_value(lines, "GEE cluster by") or ""

    n_rows    = (_extract_header_value(lines, "N rows (this model)") or "").replace(",", "")
    n_speakers = (_extract_header_value(lines, "N speakers") or "").replace(",", "")

    # ── Parse track / model type ──────────────────────────────
    track, model_type = parse_label(full_label)

    # ── Strip parliament scope to just the parliament name ────
    parl_clean = parliament
    for p in ["global", "CZ", "HR", "PL", "RS"]:
        if parliament.startswith(p):
            parl_clean = p
            break

    # ── Find the EXP(COEF) / IRR TABLE ────────────────────────
    rows = []
    in_table = False

    _table_row_re = re.compile(
        r"^(.+?)\s+"           # predictor name (non-greedy)
        r"(-?[\d.]+)\s+"       # exp_coef / IRR
        r"(-?[\d.]+)\s+"       # CI_lo
        r"(-?[\d.]+)\s+"       # CI_hi
        r"(-?[\d.]+)\s*$"      # p_value
    )

    for line in lines:
        stripped = line.strip()

        if stripped in ("EXP(COEF) TABLE:", "IRR TABLE:"):
            in_table = True
            continue

        if in_table:
            if stripped == "" or stripped.startswith("===="):
                if rows:
                    break
                continue

            if "CI_lo" in stripped and "CI_hi" in stripped:
                continue

            m = _table_row_re.match(stripped)
            if m:
                try:
                    pred_name = m.group(1).strip()
                    exp_coef  = float(m.group(2))
                    ci_lo     = float(m.group(3))
                    ci_hi     = float(m.group(4))
                    p_val     = float(m.group(5))
                except ValueError:
                    continue

                rows.append({
                    "Track":             track,
                    "Model_Type":        model_type,
                    "Full_Label":        full_label,
                    "Parliament":        parl_clean,
                    "Scope_Raw":         parliament,
                    "Mundlak":           use_mundlak,
                    "Interaction":       interaction_label,
                    "Predictor":         pred_name,
                    "exp_coef":          exp_coef,
                    "CI_lo":             ci_lo,
                    "CI_hi":             ci_hi,
                    "p_value":           p_val,
                    "N_rows":            n_rows,
                    "N_speakers":        n_speakers,
                    "NB_alpha":          nb_alpha,
                    "Formula":           formula,
                    "Rate_type":         rate_type,
                    "Timestamp":         timestamp,
                    "GEE_cluster":       gee_cluster,
                    "Mundlak_rate":         mundlak_rate,
                    "Mundlak_sentiment":    mundlak_sentiment,
                    "Mundlak_party_status": mundlak_party_status,
                    "Mundlak_orientation":  mundlak_orientation,
                    "Scale_sentiment":      scale_sentiment,
                    "Scale_orientation":    scale_orientation,
                })

    # ── Exchangeable alpha ────────────────────────────────────
    exch_alpha = ""
    m_exch = re.search(r"Exchangeable alpha\s*=\s*([\d.]+)", text)
    if m_exch:
        exch_alpha = m_exch.group(1)
    for row in rows:
        row["Exch_alpha"] = exch_alpha

    return rows


def collect_all(results_dir):
    """Walk results_dir, parse all .txt logs, return combined DataFrame."""
    all_rows = []
    for dirpath, _, filenames in os.walk(results_dir):
        if "_diagnostics" in dirpath:
            continue
        for fname in sorted(filenames):
            if not fname.endswith(".txt"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                rows = parse_log_file(fpath)
                if rows:
                    all_rows.extend(rows)
                    rel = os.path.relpath(fpath, results_dir)
                    print(f"  {len(rows):>3} rows  <-  {rel}")
                else:
                    rel = os.path.relpath(fpath, results_dir)
                    print(f"  [SKIP] No table in {rel}")
            except Exception as e:
                print(f"  [ERROR] {fname}: {e}")

    if not all_rows:
        print("No data found!")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    col_order = [
        "Track", "Model_Type", "Full_Label", "Parliament", "Scope_Raw",
        "Mundlak", "Interaction",
        "Predictor", "exp_coef", "CI_lo", "CI_hi", "p_value",
        "N_rows", "N_speakers", "NB_alpha", "Exch_alpha",
        "Formula", "Rate_type", "GEE_cluster", "Timestamp",
        "Mundlak_rate", "Mundlak_sentiment",
        "Mundlak_party_status", "Mundlak_orientation",
        "Scale_sentiment", "Scale_orientation",
    ]
    col_order = [c for c in col_order if c in df.columns]
    return df[col_order]


# =================================================================
# PREDICTOR DISPLAY NAMES & ORDERING
# =================================================================

_RAW_TO_DISPLAY = [
    (r"^Intercept$",                   "Intercept"),
    (r"Speaker_gender|Gender",         "Speaker gender"),
    (r"Speaker_age",                   "Speaker age"),
    (r"speech_rate.*_dev",             "Speech rate (within)"),
    (r"speech_rate.*_mean",            "Speech rate (between)"),
    (r"speech_rate",                   "Speech rate"),
    (r"Sentiment.*_dev",               "Sentiment (within)"),
    (r"Sentiment.*_mean",              "Sentiment (between)"),
    (r"Sentiment",                     "Sentiment"),
    (r"Party_orientation.*_dev",       "Political orientation (within)"),
    (r"Party_orientation.*_mean",      "Political orientation (between)"),
    (r"Party_orientation",             "Political orientation"),
    (r"Party_status_dev",              "Power status (within)"),
    (r"Party_status_mean",             "Power status (between)"),
    (r"Party_status(?!_)",             "Power status"),
    (r"C\(Lang\)\[T\.HR\]|Lang.*HR",  "HR"),
    (r"C\(Lang\)\[T\.PL\]|Lang.*PL",  "PL"),
    (r"C\(Lang\)\[T\.RS\]|Lang.*RS",  "RS"),
]

PREDICTOR_ORDER = [
    "Speaker gender",
    "Speaker age",
    "Speech rate",
    "Speech rate (between)",
    "Speech rate (within)",
    "Sentiment",
    "Sentiment (between)",
    "Sentiment (within)",
    "Political orientation",
    "Political orientation (between)",
    "Political orientation (within)",
    "Power status",
    "Power status (between)",
    "Power status (within)",
]

LANG_LABELS = ["HR", "PL", "RS"]

GROUP_BANDS = [
    (["Speaker gender", "Speaker gender (between)", "Speaker gender (within)"],
     "#f9c6c6"),   # warm pink
    (["Speaker age", "Speaker age (between)", "Speaker age (within)"],
     "#a8d8ea"),   # cold blue
    (["Speech rate", "Speech rate (between)", "Speech rate (within)"],
     "#fde9c6"),   # warm peach
    (["Sentiment", "Sentiment (between)", "Sentiment (within)"],
     "#c9daf8"),   # cold periwinkle
    (["Political orientation", "Political orientation (between)",
      "Political orientation (within)"],
     "#ffd6a5"),   # warm apricot
    (["Power status", "Power status (between)", "Power status (within)"],
     "#c9f0c8"),   # cold mint
    (LANG_LABELS,
     "#e8d5f5"),   # warm lavender
]

REF_ANNOTATIONS = {
    "Speaker gender": "(T.M)",
    "Power status":   "(T.Opposition)",
}

PARL_ORDER = ["global", "CZ", "HR", "PL", "RS"]


def _strip_scale(name):
    m = re.match(r"^scale\((.+)\)$", name.strip())
    return m.group(1) if m else name.strip()


def clean_predictor_name(raw):
    stripped = _strip_scale(raw)
    if ":" in stripped:
        parts = stripped.split(":")
        cleaned = [clean_predictor_name(p.strip()) for p in parts]
        return " × ".join(cleaned)
    for pattern, display in _RAW_TO_DISPLAY:
        if re.search(pattern, stripped, re.IGNORECASE):
            return display
    return stripped


def get_ordered_predictors(sub_df, is_global=False, drop_intercept=True,
                           drop_lang=True):
    """Return ordered predictor list for a single-panel forest plot."""
    present = set(sub_df["Predictor_Clean"].unique())

    if drop_intercept:
        present.discard("Intercept")
    if drop_lang and is_global:
        for lbl in LANG_LABELS:
            present.discard(lbl)

    ordered = [p for p in PREDICTOR_ORDER if p in present]

    if is_global and not drop_lang:
        ordered += [p for p in LANG_LABELS if p in present]

    known = set(PREDICTOR_ORDER) | set(LANG_LABELS) | {"Intercept"}
    interactions = sorted(p for p in present if p not in known and "×" in p)
    extras       = sorted(p for p in present if p not in known and "×" not in p)
    ordered += extras + interactions

    return ordered


# =================================================================
# FOREST PLOT RENDERING
# =================================================================

ROW_HEIGHT_IN  = 0.45
TITLE_PAD_IN   = 0.5
BOTTOM_PAD_IN  = 0.5
DPI            = 150
PLOT_WIDTH_IN  = 3.4
LABEL_WIDTH_IN = 3.0

NONSIG_COLOR = "#ff5c5c"
MAIN_COLOR   = "#1f77b4"

sns.set_theme(style="whitegrid")


def is_significant(row):
    return not (row["CI_lo"] <= 1.0 <= row["CI_hi"])


def draw_group_bands(ax, predictor_order, extra_bands=None):
    """Draw coloured horizontal bands for predictor groups."""
    bands = list(GROUP_BANDS)
    if extra_bands:
        bands.extend(extra_bands)

    # Auto-add interaction band if present
    int_indices = [i for i, p in enumerate(predictor_order) if "×" in p]
    if int_indices:
        int_labels = [predictor_order[i] for i in int_indices]
        bands.append((int_labels, "#ffe0b2"))

    for group_labels, color in bands:
        indices = [i for i, p in enumerate(predictor_order) if p in group_labels]
        if not indices:
            continue
        lo = min(indices) - 0.5
        hi = max(indices) + 0.5
        ax.axhspan(lo, hi, color=color, alpha=0.25, zorder=0, linewidth=0)


def make_subplot(parl, sub_df, predictor_order, show_ylabel,
                 n_rows_global, xlabel="IRR", divider_y=None):
    """
    Render a single forest-plot panel.

    Parameters
    ----------
    divider_y : float or None
        If set, draw a horizontal separator at this y position.
    """
    n_local = len(predictor_order)
    if n_local == 0:
        return None

    fig_h = n_rows_global * ROW_HEIGHT_IN + TITLE_PAD_IN + BOTTOM_PAD_IN
    fig_w = PLOT_WIDTH_IN + (LABEL_WIDTH_IN if show_ylabel else 0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    top_margin = 1.0 - TITLE_PAD_IN / fig_h
    ax_height = (n_local * ROW_HEIGHT_IN) / fig_h

    fig.subplots_adjust(
        left   = (LABEL_WIDTH_IN / fig_w) if show_ylabel else 0.08,
        right  = 0.95,
        top    = top_margin,
        bottom = top_margin - ax_height,
    )

    draw_group_bands(ax, predictor_order)

    y_positions = {p: i for i, p in enumerate(predictor_order)}

    # ── Detect duplicate predictors (base + Mundlak on same label) ──
    # Group rows by Predictor_Clean; if >1 row per label, jitter them.
    JITTER = 0.12
    pred_groups = {}
    for idx, row in sub_df.iterrows():
        pred = row["Predictor_Clean"]
        if pred not in y_positions:
            continue
        pred_groups.setdefault(pred, []).append(idx)

    has_track = "Track" in sub_df.columns

    for _, row in sub_df.iterrows():
        pred = row["Predictor_Clean"]
        if pred not in y_positions:
            continue
        color = MAIN_COLOR if is_significant(row) else NONSIG_COLOR
        y = float(y_positions[pred])

        # Apply jitter when multiple rows share the same label
        group = pred_groups.get(pred, [])
        if len(group) == 2:
            # Use Track to determine direction: baseline up, Mundlak down
            if has_track and "MUNDLAK" in str(row.get("Track", "")).upper():
                y += JITTER
            elif has_track:
                y -= JITTER
            else:
                # Fallback: first up, second down
                my_pos = group.index(row.name)
                if my_pos == 0:
                    y -= JITTER
                else:
                    y += JITTER
        elif len(group) > 2:
            # Shouldn't happen after _collect_master_rows, but be safe:
            # spread evenly
            my_pos = group.index(row.name)
            n = len(group)
            y += JITTER * (2 * my_pos / (n - 1) - 1)

        ax.errorbar(
            row["exp_coef"], y,
            xerr=[[row["exp_coef"] - row["CI_lo"]],
                  [row["CI_hi"] - row["exp_coef"]]],
            fmt="o", capsize=3, alpha=0.9, markersize=7,
            color=color, linewidth=2, capthick=2,
        )

    ax.axvline(1, color="black", linestyle="-", alpha=0.3, lw=1)

    # ── Horizontal divider ────────────────────────────────────
    if divider_y is not None:
        ax.axhline(divider_y, color="black", linestyle="--", alpha=0.5, lw=1.2)

    ax.set_yticks(range(n_local))

    if show_ylabel:
        tick_labels = []
        for p in predictor_order:
            if p in REF_ANNOTATIONS:
                tick_labels.append(f"{p} {REF_ANNOTATIONS[p]}")
            else:
                tick_labels.append(p)
        ax.set_yticklabels(tick_labels, fontsize=10, fontweight="bold")
    else:
        ax.set_yticklabels([])

    ax.set_ylim(n_local - 0.5, -0.5)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(parl.upper(), fontsize=20, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches=None)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def make_legend(total_width, effect_name="IRR"):
    legend_fig_h = 0.65
    legend_fig_w = total_width / DPI
    fig, ax = plt.subplots(figsize=(legend_fig_w, legend_fig_h))
    ax.axis("off")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                    markerfacecolor=MAIN_COLOR, markersize=9,
                    label=f"Significant ({effect_name})"),
        plt.Line2D([0], [0], marker="o", color="w",
                    markerfacecolor=NONSIG_COLOR, markersize=9,
                    label="Non-significant (CI crosses 1)"),
    ]

    ax.legend(
        handles=handles, loc="center", bbox_to_anchor=(0.5, 0.5),
        ncol=2, fontsize=12, frameon=True, markerscale=1.5,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def make_supertitle(title_text, total_width):
    """Render a supertitle banner image matching total_width."""
    title_h = int(DPI * 0.55)
    title_fig, title_ax = plt.subplots(
        figsize=(total_width / DPI, title_h / DPI))
    title_ax.axis("off")
    title_ax.text(0.5, 0.45, title_text,
                  ha="center", va="center",
                  fontsize=18, fontweight="bold",
                  transform=title_ax.transAxes)
    title_buf = io.BytesIO()
    title_fig.savefig(title_buf, format="png", dpi=DPI,
                      bbox_inches="tight", pad_inches=0.1)
    plt.close(title_fig)
    title_buf.seek(0)
    title_img = Image.open(title_buf).copy()
    if title_img.width != total_width:
        canvas = Image.new("RGB", (total_width, title_img.height), "white")
        x_off = (total_width - title_img.width) // 2
        canvas.paste(title_img, (x_off, 0))
        title_img = canvas
    return title_img


def _centre_image(img, target_width):
    """Centre an image on a white canvas of target_width."""
    if img.width == target_width:
        return img
    canvas = Image.new("RGB", (target_width, img.height), "white")
    x_off = (target_width - img.width) // 2
    canvas.paste(img, (x_off, 0))
    return canvas


def make_footnote(text, total_width):
    """Render a footnote text bar matching total_width."""
    fig_h = 0.45
    fig_w = total_width / DPI
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.text(0.5, 0.5, text,
            ha="center", va="center", fontsize=9, fontstyle="italic",
            color="#555555", transform=ax.transAxes, wrap=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).copy()
    return _centre_image(img, total_width)


def assemble_final_png(title_text, panel_images, xlabel, out_path,
                       divider_y=None, footnote_text=None):
    """
    Combine panel images horizontally, add supertitle + legend, save.

    Parameters
    ----------
    panel_images : dict
        {parl_name: PIL.Image} in PARL_ORDER order.
    """
    total_width = sum(img.width for img in panel_images.values())
    max_height  = max(img.height for img in panel_images.values())

    combined = Image.new("RGB", (total_width, max_height), "white")
    x_offset = 0
    for parl in PARL_ORDER:
        if parl in panel_images:
            combined.paste(panel_images[parl], (x_offset, 0))
            x_offset += panel_images[parl].width

    effect_name = {"IRR": "Incidence Rate Ratio",
                   "exp(b)": "Multiplicative Effect",
                   "OR": "Odds Ratio"}.get(xlabel, xlabel)
    legend_img = _centre_image(make_legend(total_width, effect_name),
                               total_width)
    title_img = make_supertitle(title_text, total_width)

    # Optional footnote
    footnote_img = None
    if footnote_text:
        footnote_img = make_footnote(footnote_text, total_width)

    total_height = title_img.height + max_height + legend_img.height
    if footnote_img:
        total_height += footnote_img.height

    final = Image.new("RGB", (total_width, total_height), "white")
    final.paste(title_img, (0, 0))
    final.paste(combined, (0, title_img.height))
    final.paste(legend_img, (0, title_img.height + max_height))
    if footnote_img:
        final.paste(footnote_img,
                    (0, title_img.height + max_height + legend_img.height))

    final.save(out_path, dpi=(DPI, DPI))
    print(f"    Saved: {out_path}")


def get_track_xlabel(track):
    """Infer xlabel from track name."""
    t_lower = track.lower()
    if "gamma" in t_lower:
        return "exp(b)"
    elif "logistic" in t_lower:
        return "OR"
    return "IRR"