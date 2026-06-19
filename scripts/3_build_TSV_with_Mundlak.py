"""
3_build_TSV_with_Mundlak.py
===========================
Reads per-language FP-count JSONL files produced by step 2,
combines into a single TSV with:
  - Lang column
  - Party_status numeric coding (Coalition=0, Opposition=1)
  - Mundlak decompositions (speaker-level mean + within-speaker dev)
  - Verification printouts and sanity checks
  - Stratified subset for development/testing
"""

import json
import os
import pandas as pd
import numpy as np

# ── Config ────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "fp_count_jsonls")
OUTPUT_DIR = DATA_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = [
    "fp_ParlaSpeech-CZ.v3.0.jsonl",
    "fp_ParlaSpeech-HR.v3.0.jsonl",
    "fp_ParlaSpeech-PL.v3.0.jsonl",
    "fp_ParlaSpeech-RS.v3.0.jsonl",
]

# ── Mundlak flags ─────────────────────────────────────────────
MUNDLAK = {
    "Party_status":            True,
    "Party_orientation_float": True,
    "speech_rate_syll":        True,
    "speech_rate_word":        True,
    "speech_rate_syll_np":     True,
    "speech_rate_word_np":     True,
    "Sentiment_label_float":   True,
    "sp_count":                True,
    "mean_word_duration_ms":   True,
}

# ── Subset config ─────────────────────────────────────────────
SUBSET_N_PER_GENDER = 5
SUBSET_UTTS         = 200
STRATIFY_ON         = "fp_count"


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def get_lang(filename):
    return filename.split("-")[1].split(".")[0]


def add_mundlak(df, col, speaker_col="Speaker_ID"):
    if col not in df.columns:
        print(f"  [SKIP] '{col}' not in dataframe.")
        return
    df[col] = pd.to_numeric(df[col], errors="coerce")
    means = df.groupby(speaker_col)[col].transform("mean")
    df[f"{col}_mean"] = means
    df[f"{col}_dev"]  = df[col] - means
    print(f"  Mundlak: {col}")


# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────

all_rows = []

for filename in FILES:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"Skipping {filename}: not found.")
        continue
    lang = get_lang(filename)
    print(f"Loading {filename} ({lang})...")
    n = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["Lang"] = lang
            all_rows.append(row)
            n += 1

    print(f"  {n:,} utterances")

df = pd.DataFrame(all_rows)
print(f"\nTotal: {len(df):,} utterances | {df['Speaker_ID'].nunique():,} speakers\n")

# Put Lang first
lead = ["Lang"]
df = df[lead + [c for c in df.columns if c not in lead]]

# ─────────────────────────────────────────────────────────────
# TYPE COERCION
# ─────────────────────────────────────────────────────────────

numeric_cols = [
    "Party_orientation_float", "Sentiment_label_float",
    "speech_rate_syll", "speech_rate_word",
    "speech_rate_syll_np", "speech_rate_word_np",
    "mean_word_duration_ms", "sd_word_duration_ms",
    "words_total_duration_ms", "audio_length",
    "fp_count", "fp_total_duration_ms", "fp_average_dur_ms", "fp_rate",
    "sp_count", "sp_total_duration_ms", "sp_average_dur_ms",
    "Speaker_age_at_session", "word_count", "syllable_count",
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "Party_status" in df.columns:
    df["Party_status"] = df["Party_status"].map({"Coalition": 0, "Opposition": 1})
    print("Party_status mapped: Coalition→0, Opposition→1")
    print(df["Party_status"].value_counts(dropna=False).to_string())

# ─────────────────────────────────────────────────────────────
# MUNDLAK DECOMPOSITIONS
# ─────────────────────────────────────────────────────────────

print("\n--- Mundlak Decompositions ---")
for col, do_it in MUNDLAK.items():
    if do_it:
        add_mundlak(df, col)

# ─────────────────────────────────────────────────────────────
# VERIFICATION
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("DATASET VERIFICATION")
print("=" * 60)
print(f"Utterances   : {len(df):,}")
print(f"Speakers     : {df['Speaker_ID'].nunique():,}")
print(f"Languages    : {df['Lang'].unique().tolist()}")

print("\n--- FP Count Summary ---")
print(f"  mean ± SD : {df['fp_count'].mean():.2f} ± {df['fp_count'].std():.2f}")
print(f"  median    : {df['fp_count'].median():.1f}")
print(f"  min / max : {df['fp_count'].min():.0f} / {df['fp_count'].max():.0f}")
print(f"  zero-FP   : {(df['fp_count'] == 0).sum():,} "
      f"({100 * (df['fp_count'] == 0).mean():.1f}%)")

print("\n--- Average by language ---")
agg_cols = {c: "mean" for c in [
    "fp_count", "fp_rate", "speech_rate_syll", "speech_rate_word",
] if c in df.columns}
agg_cols["id"] = "count"
summary = (df.groupby("Lang").agg(agg_cols)
             .rename(columns={"id": "n_utterances"})
             .round(3))
print(summary.to_string())

print("\n--- Mundlak Sanity Check ---")
for dev_col in [c for c in df.columns if c.endswith("_dev")]:
    val = df.groupby("Speaker_ID")[dev_col].mean().abs().mean()
    status = "OK" if val < 1e-6 else "!! CHECK"
    print(f"  {dev_col:<45} {val:.8f}  [{status}]")

# ─────────────────────────────────────────────────────────────
# SAVE FULL TSV
# ─────────────────────────────────────────────────────────────

out_path = os.path.join(OUTPUT_DIR, "ParlaSpeech-ALL.v3.0-fp-counts.tsv")
df.to_csv(out_path, sep="\t", index=False, encoding="utf-8")
print(f"\nSaved: {out_path}  ({len(df):,} rows × {len(df.columns)} cols)")

# ─────────────────────────────────────────────────────────────
# SUBSET GENERATION
# 5M + 5F per language, stratified by fp_count,
# min SUBSET_UTTS utterances per speaker.
# ─────────────────────────────────────────────────────────────

print("\n--- Subset Generation ---")
subset_frames = []

for lang, lang_df in df.groupby("Lang"):
    for gender in ["M", "F"]:
        g_df = lang_df[lang_df["Speaker_gender"] == gender]
        counts   = g_df.groupby("Speaker_ID").size()
        eligible = counts[counts >= SUBSET_UTTS].index
        g_df     = g_df[g_df["Speaker_ID"].isin(eligible)]

        if len(eligible) == 0:
            print(f"  WARNING [{lang}/{gender}]: no eligible speakers. Skipping.")
            continue

        spk_summary = (
            g_df.groupby("Speaker_ID")[STRATIFY_ON].mean()
            .reset_index().rename(columns={STRATIFY_ON: "strat_val"})
            .dropna(subset=["strat_val"])
        )

        n_bins = min(4, len(spk_summary))
        spk_summary["bin"] = pd.qcut(
            spk_summary["strat_val"], q=n_bins, labels=False, duplicates="drop")

        selected_ids = (
            spk_summary.groupby("bin", group_keys=False)
            .apply(lambda x: x.sample(
                min(len(x), max(1, round(
                    SUBSET_N_PER_GENDER * len(x) / len(spk_summary)))),
                random_state=42))["Speaker_ID"].tolist()
        )

        if len(selected_ids) > SUBSET_N_PER_GENDER:
            selected_ids = selected_ids[:SUBSET_N_PER_GENDER]
        elif len(selected_ids) < SUBSET_N_PER_GENDER:
            pool = [s for s in spk_summary["Speaker_ID"] if s not in selected_ids]
            selected_ids += pool[:SUBSET_N_PER_GENDER - len(selected_ids)]

        print(f"  [{lang}/{gender}] {len(selected_ids)} speakers: {selected_ids}")

        for spk_id in selected_ids:
            utts = g_df[g_df["Speaker_ID"] == spk_id]
            if len(utts) > SUBSET_UTTS:
                utts = utts.sample(SUBSET_UTTS, random_state=42)
            subset_frames.append(utts)

if subset_frames:
    df_sub = pd.concat(subset_frames).reset_index(drop=True)
    sub_path = os.path.join(OUTPUT_DIR, "ParlaSpeech-ALL.v3.0-fp-counts-subset.tsv")
    df_sub.to_csv(sub_path, sep="\t", index=False, encoding="utf-8")
    print(f"\nSubset saved: {sub_path}  "
          f"({df_sub.shape[0]:,} utterances | {df_sub['Speaker_ID'].nunique()} speakers)")

print("\nDone.")