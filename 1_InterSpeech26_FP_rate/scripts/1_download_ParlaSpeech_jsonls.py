"""
1_download_ParlaSpeech_jsonls.py
================================
Downloads the four ParlaSpeech v3.0 JSONL annotation files from CLARIN.SI
into <ROOT>/ParlaSpeech_v3_jsonls/ and decompresses each .gz into .jsonl.

Default: all four languages. Pass language codes to pick a subset:

    python scripts/1_download_ParlaSpeech_jsonls.py
    python scripts/1_download_ParlaSpeech_jsonls.py HR PL
    python scripts/1_download_ParlaSpeech_jsonls.py --force CZ
"""

import argparse
import gzip
import os
import shutil
import sys

import requests
from tqdm.auto import tqdm


# ── Config ────────────────────────────────────────────────────
VERSION  = "v3.0"
HANDLE   = "11356/1833"
BASE_URL = f"https://www.clarin.si/repository/xmlui/bitstream/handle/{HANDLE}"

# (lang, approx_size_mb)
FILES = [
    ("CZ",  565),
    ("HR", 2580),
    ("PL",  393),
    ("RS",  586),
]

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, f"ParlaSpeech_{VERSION.split('.')[0]}_jsonls")


# ── Download helper ───────────────────────────────────────────
def download(url, dest, chunk=1 << 20):
    """Stream url to dest via a .part file (atomic on success)."""
    part = dest + ".part"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(part, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024,
            desc=os.path.basename(dest), leave=True,
        ) as bar:
            for c in r.iter_content(chunk_size=chunk):
                if c:
                    f.write(c)
                    bar.update(len(c))
    os.replace(part, dest)


def gunzip(src, dest):
    with gzip.open(src, "rb") as fi, open(dest, "wb") as fo:
        shutil.copyfileobj(fi, fo)


# ── Main ──────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("langs", nargs="*", default=[],
                    help="Subset of languages to fetch (CZ HR PL RS). Default = all.")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if the .jsonl already exists.")
    ap.add_argument("--keep-gz", action="store_true",
                    help="Keep the intermediate .jsonl.gz file after decompress.")
    args = ap.parse_args()

    valid = {l for l, _ in FILES}
    if args.langs:
        wanted = [l.upper() for l in args.langs]
        bad = [l for l in wanted if l not in valid]
        if bad:
            sys.exit(f"Unknown language(s): {bad}. Valid: {sorted(valid)}")
        targets = [(l, sz) for l, sz in FILES if l in wanted]
    else:
        targets = FILES

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}\n")

    total_mb = sum(sz for _, sz in targets)
    print(f"Planned: {len(targets)} file(s), ~{total_mb:,} MB compressed total")
    for lang, sz in targets:
        print(f"  ParlaSpeech-{lang}.{VERSION}.jsonl.gz  (~{sz:,} MB)")
    print()

    for lang, sz in targets:
        gz_name   = f"ParlaSpeech-{lang}.{VERSION}.jsonl.gz"
        jsonl     = os.path.join(OUTPUT_DIR, f"ParlaSpeech-{lang}.{VERSION}.jsonl")
        gz_path   = os.path.join(OUTPUT_DIR, gz_name)
        url       = f"{BASE_URL}/{gz_name}"

        if os.path.exists(jsonl) and not args.force:
            print(f"  [skip] {os.path.basename(jsonl)} already present")
            continue

        print(f"  [get ] {gz_name}  (~{sz:,} MB)")
        try:
            download(url, gz_path)
        except Exception as e:
            print(f"    download failed: {e}")
            continue

        print(f"  [gunzip] {gz_name} -> {os.path.basename(jsonl)}")
        try:
            gunzip(gz_path, jsonl)
        except Exception as e:
            print(f"    gunzip failed: {e}")
            continue

        if not args.keep_gz:
            os.remove(gz_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
