"""
2_extract_FP_counts.py
======================
Reads raw ParlaSpeech JSONL files and produces a cleaned JSONL
with one row per utterance, enriched with:
  - syllable counts and speech rates
  - filled pause count and rate
  - silent pause aggregates
  - speaker metadata (age, sentiment, orientation as floats)

Output is consumed by 3_build_TSV_with_Mundlak.py.
"""

import json
import os
import re

# ── Config ────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR  = os.path.join(ROOT, "ParlaSpeech_v3_jsonls")
OUTPUT_DIR = os.path.join(ROOT, "fp_count_jsonls")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_SP_MS = 100
MIN_FP_MS = 80

FILES = [
    "ParlaSpeech-CZ.v3.0.jsonl",
    "ParlaSpeech-HR.v3.0.jsonl",
    "ParlaSpeech-PL.v3.0.jsonl",
    "ParlaSpeech-RS.v3.0.jsonl",
]

ORIENTATION_MAP = {
    "Far-left": -3.0, "Left to far-left": -2.5, "Left": -2.0,
    "Centre-left to left": -1.5, "Centre-left": -1.0,
    "Centre to centre-left": -0.5, "Centre": 0.0, "Big tent": 0.0,
    "Centre to centre-right": 0.5, "Centre-right": 1.0,
    "Centre-right to right": 1.5, "Right": 2.0,
    "Right to far-right": 2.5, "Far-right": 3.0,
}

SENTIMENT_MAP = {
    "Negative": -2.5, "Mixed Negative": -1.5, "Neutral Negative": -0.5,
    "Neutral Positive": 0.5, "Mixed Positive": 1.5, "Positive": 2.5,
}

VOWELS = {
    "HR": set("aeiou"),
    "RS": set("aeiou"),
    "CZ": set("aeiouyáéíóúýěů"),
    "PL": set("aeiouyóąę"),
}


# ── Syllable counting ─────────────────────────────────────────

def count_syllables(text, lang):
    vowels = VOWELS.get(lang, set("aeiou"))
    if lang == "CZ":
        cleaned = re.sub(r"[^a-zčďěľňřšťůžáéíóúý]", " ", text.lower())
    elif lang == "PL":
        cleaned = re.sub(r"[^a-ząćęłńśźżó]", " ", text.lower())
    else:
        cleaned = re.sub(r"[^a-zćčđšž]", " ", text.lower())
    total = 0
    for word in cleaned.split():
        count = 0
        for i, ch in enumerate(word):
            if ch in vowels:
                count += 1
            elif ch == "r" and lang in ("HR", "RS", "CZ"):
                pv = i > 0 and word[i - 1] in vowels
                nv = i < len(word) - 1 and word[i + 1] in vowels
                if not pv and not nv:
                    count += 1
            elif ch == "l" and lang == "CZ":
                pv = i > 0 and word[i - 1] in vowels
                nv = i < len(word) - 1 and word[i + 1] in vowels
                if not pv and not nv:
                    count += 1
        total += count
    return total


# ── Silent pause extraction ───────────────────────────────────

def extract_silent_pauses(word_list, fp_intervals):
    """Extract inter-word silent pauses that don't overlap with FPs."""

    def overlaps_fp(ts, te):
        for fs, fe in fp_intervals:
            if ts < fe and te > fs:
                return True
        return False

    pauses = []
    for i in range(len(word_list) - 1):
        cur, nxt = word_list[i], word_list[i + 1]
        if "time_e" not in cur or "time_s" not in nxt:
            continue
        gap_ms = round((nxt["time_s"] - cur["time_e"]) * 1000)
        if gap_ms < MIN_SP_MS:
            continue
        ts = round(cur["time_e"], 3)
        te = round(nxt["time_s"], 3)
        if overlaps_fp(ts, te):
            continue
        pauses.append({"time_s": ts, "time_e": te, "duration_ms": gap_ms})
    return pauses


# ── Main processor ────────────────────────────────────────────

def process_line(data, lang):
    audio_len  = data.get("audio_length", 0)
    text       = data.get("text", "")
    word_list  = data.get("words", [])
    word_count = len(word_list)
    syll_count = count_syllables(text, lang)

    # ── Speech rates ──────────────────────────────────────────
    speech_rate_word = round(word_count / audio_len, 3) if audio_len > 0 else 0
    speech_rate_syll = round(syll_count / audio_len, 3) if audio_len > 0 else 0

    # ── Word durations ────────────────────────────────────────
    word_durations_ms = []
    for w in word_list:
        if "time_s" in w and "time_e" in w:
            word_durations_ms.append(round((w["time_e"] - w["time_s"]) * 1000))

    words_total_ms = sum(word_durations_ms)
    words_total_s  = words_total_ms / 1000

    if word_durations_ms:
        mean_wd = round(sum(word_durations_ms) / len(word_durations_ms), 3)
        sd_wd   = round(
            (sum((d - mean_wd) ** 2 for d in word_durations_ms)
             / len(word_durations_ms)) ** 0.5, 3
        ) if len(word_durations_ms) > 1 else 0.0
    else:
        mean_wd = sd_wd = 0.0

    # ── Net speech rates ──────────────────────────────────────
    speech_rate_word_np = round(word_count / words_total_s, 3) if words_total_s > 0 else 0
    speech_rate_syll_np = round(syll_count / words_total_s, 3) if words_total_s > 0 else 0

    # ── Filled pauses ─────────────────────────────────────────
    raw_fps     = data.get("filled_pauses", [])
    fp_list_raw = raw_fps if isinstance(raw_fps, list) else []

    # Build FP intervals for SP overlap check, filter by min duration
    fp_intervals = []
    valid_fps    = []
    for fp in fp_list_raw:
        if "time_s" in fp and "time_e" in fp:
            dur = round((fp["time_e"] - fp["time_s"]) * 1000)
            if dur >= MIN_FP_MS:
                fp_intervals.append((fp["time_s"], fp["time_e"]))
                valid_fps.append({"time_s": fp["time_s"], "time_e": fp["time_e"],
                                  "duration_ms": dur})

    fp_count       = len(valid_fps)
    fp_total_ms    = sum(fp["duration_ms"] for fp in valid_fps)
    fp_avg_ms      = round(fp_total_ms / fp_count, 3) if fp_count > 0 else 0.0
    fp_rate        = round(fp_count / audio_len, 3) if audio_len > 0 else 0

    # ── Silent pauses ─────────────────────────────────────────
    silent_pauses  = extract_silent_pauses(word_list, fp_intervals)
    sp_count       = len(silent_pauses)
    sp_total_ms    = sum(sp["duration_ms"] for sp in silent_pauses)
    sp_avg_ms      = round(sp_total_ms / sp_count, 3) if sp_count > 0 else 0.0

    # ── Speaker metadata ──────────────────────────────────────
    si        = data.get("speaker_info", {})
    date_str  = si.get("Date", "")
    birth_str = si.get("Speaker_birth", "")
    age       = None
    try:
        if date_str and birth_str and birth_str != "-":
            age = int(date_str.split("-")[0]) - int(birth_str)
    except (ValueError, IndexError):
        pass

    sent_label   = data.get("sentiment", {}).get("ParlaSent_6")
    orientation  = si.get("Party_orientation")
    sent_float   = SENTIMENT_MAP.get(sent_label.strip() if sent_label else None, float("nan"))
    orient_float = ORIENTATION_MAP.get(orientation.strip() if orientation else None, float("nan"))

    return {
        "id":                      data.get("id"),
        "audio_length":            audio_len,
        "word_count":              word_count,
        "syllable_count":          syll_count,
        "speech_rate_word":        speech_rate_word,
        "speech_rate_syll":        speech_rate_syll,
        "words_total_duration_ms": words_total_ms,
        "mean_word_duration_ms":   mean_wd,
        "sd_word_duration_ms":     sd_wd,
        "speech_rate_word_np":     speech_rate_word_np,
        "speech_rate_syll_np":     speech_rate_syll_np,
        # filled pauses
        "fp_count":                fp_count,
        "fp_total_duration_ms":    fp_total_ms,
        "fp_average_dur_ms":       fp_avg_ms,
        "fp_rate":                 fp_rate,
        # silent pauses
        "sp_count":                sp_count,
        "sp_total_duration_ms":    sp_total_ms,
        "sp_average_dur_ms":       sp_avg_ms,
        # speaker metadata
        "Speech_ID":               si.get("ID"),
        "Speaker_ID":              si.get("Speaker_ID"),
        "Speaker_name":            si.get("Speaker_name"),
        "Speaker_gender":          si.get("Speaker_gender"),
        "Speaker_birth":           birth_str,
        "Speaker_age_at_session":  age,
        "Sentiment_label":         sent_label,
        "Sentiment_label_float":   sent_float,
        "Speaker_role":            si.get("Speaker_role"),
        "Speaker_MP":              si.get("Speaker_MP"),
        "Speaker_minister":        si.get("Speaker_minister"),
        "Speaker_party":           si.get("Speaker_party"),
        "Party_status":            si.get("Party_status"),
        "Party_orientation":       orientation,
        "Party_orientation_float": orient_float,
        "Date":                    date_str,
        "audio":                   data.get("audio"),
        "text":                    text,
    }


# ── Main loop ─────────────────────────────────────────────────

for filename in FILES:
    lang     = filename.split("-")[1].split(".")[0]
    in_path  = os.path.join(INPUT_DIR, filename)
    out_path = os.path.join(OUTPUT_DIR, f"fp_{filename}")

    if not os.path.exists(in_path):
        print(f"Skipping {filename}: not found.")
        continue

    print(f"Processing {filename} ({lang})...")
    n_utts = n_fps = n_sps = 0

    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = process_line(json.loads(line), lang)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_utts += 1
            n_fps  += row["fp_count"]
            n_sps  += row["sp_count"]

    print(f"  Done: {n_utts:,} utterances | {n_fps:,} FPs | {n_sps:,} SPs → {out_path}")

print("\nAll done.")