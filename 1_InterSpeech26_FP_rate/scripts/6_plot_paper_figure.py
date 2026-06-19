"""
6_plot_paper_figure.py
======================
Reproduces the Figure 1 layout from the InterSpeech 2026 paper.

Reads collected_results.csv (produced by 5_collect_results_and_plot.py) and
renders a 5-panel forest plot (global + CZ, HR, PL, RS) for the primary
GEE_NB_Indep specification, showing baseline (BASE) and Mundlak tracks
side-by-side per predictor row (jittered ±0.15 vertically).

Legend:
  nomundlak (blue)   — baseline track, significant
  mundlak   (orange) — Mundlak track, significant
  red                — non-significant (CI crosses 1) for either track

Usage:
    python scripts/6_plot_paper_figure.py --results-dir results/run_<...>
    python scripts/6_plot_paper_figure.py --results-dir <dir> --model GEE_NB_Exch
"""

import os
import io
import argparse

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from utils_results import clean_predictor_name


# =================================================================
# CONFIG
# =================================================================

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results", "<run_FOLDER_NAME>")

MODEL_TYPE     = "GEE_NB_Indep"   # the primary spec reported in the paper
DROP_INTERCEPT = True
DROP_LANG      = True             # hide HR/PL/RS dummies from the GLOBAL panel

# ── Visual constants ─────────────────────────────────────────
DPI            = 150
ROW_HEIGHT_IN  = 0.45
TITLE_PAD_IN   = 0.5
BOTTOM_PAD_IN  = 0.5
PLOT_WIDTH_IN  = 3.4
LABEL_WIDTH_IN = 3.0
JITTER         = 0.15

PALETTE = {
    "nomundlak": "#1f77b4",
    "mundlak":   "#ffa500",
}
NONSIG_COLOR = "#ff5c5c"
HUE_ORDER    = ["nomundlak", "mundlak"]

PARL_ORDER  = ["global", "CZ", "HR", "PL", "RS"]
LANG_LABELS = ["HR", "PL", "RS"]

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

GROUP_BANDS = [
    (["Speaker gender"],                                                   "#f9c6c6"),
    (["Speaker age"],                                                      "#fde9c6"),
    (["Speech rate", "Speech rate (between)", "Speech rate (within)"],     "#a8d8ea"),
    (["Sentiment", "Sentiment (between)", "Sentiment (within)"],           "#ffd6a5"),
    (["Political orientation",
      "Political orientation (between)",
      "Political orientation (within)"],                                   "#cca5ff"),
    (["Power status", "Power status (between)", "Power status (within)"],  "#c9f0c8"),
    (LANG_LABELS,                                                          "#e8d5f5"),
]

REF_ANNOTATIONS = {
    "Speaker gender": "T.M",
    "Power status":   "T.Opposition",
}

TRACK_TO_STATUS = {"BASE": "nomundlak", "MUNDLAK": "mundlak"}

sns.set_theme(style="whitegrid")


# =================================================================
# HELPERS
# =================================================================

def is_significant(row):
    return not (row["CI_lo"] <= 1.0 <= row["CI_hi"])


def get_ordered_predictors(sub_df, is_global=False):
    present = set(sub_df["Predictor_Clean"].unique())
    if DROP_INTERCEPT:
        present.discard("Intercept")
    if DROP_LANG:
        for lbl in LANG_LABELS:
            present.discard(lbl)
    ordered = [p for p in PREDICTOR_ORDER if p in present]
    if is_global and not DROP_LANG:
        ordered += [p for p in LANG_LABELS if p in present]
    known  = set(PREDICTOR_ORDER) | set(LANG_LABELS)
    extras = sorted(p for p in present if p not in known)
    return ordered + extras


def draw_group_bands(ax, predictor_order):
    for group_labels, color in GROUP_BANDS:
        indices = [i for i, p in enumerate(predictor_order) if p in group_labels]
        if not indices:
            continue
        ax.axhspan(min(indices) - 0.5, max(indices) + 0.5,
                   color=color, alpha=0.25, zorder=0, linewidth=0)


def make_subplot(parl, sub_df, predictor_order, show_ylabel, n_rows_global):
    n_local = len(predictor_order)
    if n_local == 0:
        return None

    fig_h = n_rows_global * ROW_HEIGHT_IN + TITLE_PAD_IN + BOTTOM_PAD_IN
    fig_w = PLOT_WIDTH_IN + (LABEL_WIDTH_IN if show_ylabel else 0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    top_margin = 1.0 - TITLE_PAD_IN / fig_h
    ax_height  = (n_local * ROW_HEIGHT_IN) / fig_h
    fig.subplots_adjust(
        left   = (LABEL_WIDTH_IN / fig_w) if show_ylabel else 0.08,
        right  = 0.95,
        top    = top_margin,
        bottom = top_margin - ax_height,
    )

    draw_group_bands(ax, predictor_order)
    y_positions = {p: i for i, p in enumerate(predictor_order)}

    for status in HUE_ORDER:
        group      = sub_df[sub_df["Mundlak_Status"] == status]
        base_color = PALETTE[status]
        offset     = -JITTER if status == "nomundlak" else +JITTER
        for _, row in group.iterrows():
            pred = row["Predictor_Clean"]
            if pred not in y_positions:
                continue
            color = base_color if is_significant(row) else NONSIG_COLOR
            y = y_positions[pred] + offset
            ax.errorbar(
                row["exp_coef"], y,
                xerr=[[row["exp_coef"] - row["CI_lo"]],
                      [row["CI_hi"] - row["exp_coef"]]],
                fmt="o", capsize=3, alpha=0.9, markersize=7,
                color=color, linewidth=2, capthick=2,
            )

    ax.axvline(1, color="black", linestyle="-", alpha=0.3, lw=1)
    ax.set_yticks(range(n_local))

    if show_ylabel:
        tick_labels = []
        for p in predictor_order:
            if p in REF_ANNOTATIONS:
                tick_labels.append(f"{p} ({REF_ANNOTATIONS[p]})")
            else:
                tick_labels.append(p)
        ax.set_yticklabels(tick_labels, fontsize=10, fontweight="bold")
    else:
        ax.set_yticklabels([])

    ax.set_ylim(n_local - 0.5, -0.5)
    ax.set_xlabel("IRR", fontsize=9)
    ax.set_title(parl.upper(), fontsize=20, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches=None)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def make_legend(total_width):
    fig, ax = plt.subplots(figsize=(total_width / DPI, 0.65))
    ax.axis("off")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=PALETTE["nomundlak"], markersize=9, label="nomundlak"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=PALETTE["mundlak"],   markersize=9, label="mundlak"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=NONSIG_COLOR,         markersize=9, label="non-significant"),
    ]
    ax.legend(
        handles=handles, title="Model / Significance",
        loc="center", bbox_to_anchor=(0.5, 0.5),
        ncol=3, fontsize=13, frameon=True,
        title_fontsize=12, markerscale=1.5,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).copy()
    if img.width != total_width:
        canvas = Image.new("RGB", (total_width, img.height), "white")
        canvas.paste(img, ((total_width - img.width) // 2, 0))
        img = canvas
    return img


# =================================================================
# MAIN
# =================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=RESULTS_DIR,
                    help="Path to the run folder containing collected_results.csv.")
    ap.add_argument("--model", default=MODEL_TYPE,
                    help=f"Which Model_Type to plot (default: {MODEL_TYPE}).")
    args = ap.parse_args()

    csv_path = os.path.join(args.results_dir, "collected_results.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(f"Not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df["Model_Type"] == args.model].copy()
    df = df[df["Track"].isin(TRACK_TO_STATUS)]
    df["Mundlak_Status"]  = df["Track"].map(TRACK_TO_STATUS)
    df["Predictor_Clean"] = df["Predictor"].apply(clean_predictor_name)

    if df.empty:
        raise SystemExit(f"No BASE/MUNDLAK rows for Model_Type={args.model}.")

    parl_pred_orders = {
        parl: get_ordered_predictors(
            df[df["Parliament"] == parl], is_global=(parl == "global"))
        for parl in PARL_ORDER
    }
    n_rows_global = max(len(v) for v in parl_pred_orders.values())

    images = {}
    for parl in PARL_ORDER:
        sub = df[df["Parliament"] == parl]
        if sub.empty:
            continue
        show_ylabel = parl in ("global", "CZ")
        img = make_subplot(parl, sub, parl_pred_orders[parl],
                           show_ylabel, n_rows_global)
        if img is not None:
            images[parl] = img

    if not images:
        raise SystemExit(f"No panels rendered for {args.model}.")

    total_width = sum(img.width for img in images.values())
    max_height  = max(img.height for img in images.values())

    combined = Image.new("RGB", (total_width, max_height), "white")
    x = 0
    for parl in PARL_ORDER:
        if parl in images:
            combined.paste(images[parl], (x, 0))
            x += images[parl].width

    legend_img = make_legend(total_width)

    final = Image.new("RGB",
                      (total_width, max_height + legend_img.height), "white")
    final.paste(combined,   (0, 0))
    final.paste(legend_img, (0, max_height))

    out_path = os.path.join(
        args.results_dir, f"global_vs_local_comparison_{args.model}.png")
    final.save(out_path, dpi=(DPI, DPI))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
