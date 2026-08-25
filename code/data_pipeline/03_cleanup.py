#!/usr/bin/env python
"""Stage 3: sequence cleanup & QC (spec Section 4 rules 4-6).

Reads enriched.csv.gz (stage 2), writes cleaned.csv.gz + manifest_stage3.json.
Per-file chunked processing (grouping never crosses files).
    .venv/bin/python data_pipeline/03_cleanup.py [--limit N]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

OUT_DIR = Path("data_pipeline/output")
VOCAB = Path("model_data/iglm_weights/vocab.txt")

SIGNAL_PEPTIDE = "MKYLLPTAAAGLLLLAAQPAMA"
RE_ALPHABET = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
RE_HIS_N = re.compile(r"^H{6,}")
RE_HIS_C = re.compile(r"H{6,}$")

HEAVY_MIN, HEAVY_MAX = 90, 160
LIGHT_MIN, LIGHT_MAX = 90, 130

DROP_RULES = ["missing_heavy", "alphabet_heavy", "length_heavy",
              "missing_light", "alphabet_light", "length_light", "dedup"]


def verify_alphabet():
    """The 20 standard AAs must be covered by the IgLM vocab (runtime check)."""
    aa = set("ACDEFGHIKLMNPQRSTVWY")
    vocab = {ln.strip() for ln in open(VOCAB)}
    missing = aa - vocab
    assert not missing, f"IgLM vocab missing amino acids: {missing}"


def trim_flanks(seqs):
    """Strip known signal peptide + His-tags. Returns (trimmed, n_trimmed)."""
    out = seqs.copy()
    trimmed = pd.Series(False, index=seqs.index)
    m = out.str.startswith(SIGNAL_PEPTIDE, na=False)
    out.loc[m] = out.loc[m].str[len(SIGNAL_PEPTIDE):]
    trimmed |= m
    for rx in (RE_HIS_N, RE_HIS_C):
        m = out.str.contains(rx, na=False)
        out.loc[m] = out.loc[m].str.replace(rx, "", n=1, regex=True)
        trimmed |= m
    return out, trimmed


def clean_file(df):
    """Apply rules 4-6 to one source file's rows. Returns (kept, stats)."""
    stats = {"n_in": int(len(df)), "drops": {r: 0 for r in DROP_RULES},
             "trimmed_heavy": 0, "trimmed_light": 0,
             "dedup_groups": 0, "dedup_conflicts": 0}

    df = df[df["seq_heavy"].notna() & (df["seq_heavy"] != "")]
    stats["drops"]["missing_heavy"] = stats["n_in"] - len(df)

    heavy, th = trim_flanks(df["seq_heavy"])
    stats["trimmed_heavy"] = int(th.sum())
    df = df.assign(seq_heavy=heavy)
    has_light = df["seq_light"].notna() & (df["seq_light"] != "")
    light, tl = trim_flanks(df["seq_light"].fillna(""))
    stats["trimmed_light"] = int((tl & has_light).sum())
    df = df.assign(seq_light=light.where(has_light, pd.NA))

    m = df["seq_heavy"].str.fullmatch(RE_ALPHABET)
    stats["drops"]["alphabet_heavy"] = int((~m).sum())
    df = df[m]
    hl = df["seq_heavy"].str.len()
    m = hl.between(HEAVY_MIN, HEAVY_MAX)
    stats["drops"]["length_heavy"] = int((~m).sum())
    df = df[m]

    light_missing = df["seq_light"].isna()
    m = ~light_missing | (df["format"] == "VHH")
    stats["drops"]["missing_light"] = int((~m).sum())
    df = df[m]

    has_light = df["seq_light"].notna()
    bad_alpha = has_light & ~df["seq_light"].str.fullmatch(RE_ALPHABET, na=False)
    stats["drops"]["alphabet_light"] = int(bad_alpha.sum())
    df = df[~bad_alpha]
    has_light = df["seq_light"].notna()
    ll = df["seq_light"].str.len()
    bad_len = has_light & ~ll.between(LIGHT_MIN, LIGHT_MAX)
    stats["drops"]["length_light"] = int(bad_len.sum())
    df = df[~bad_len]

    # rule 6: dedup within (antigen) on (heavy, light); conflicts -> median
    n_before = len(df)
    keys = ["antigen", "seq_heavy", "seq_light"]
    grp = df.groupby(keys, dropna=False, sort=False)
    sizes = grp.size()
    stats["dedup_groups"] = int((sizes > 1).sum())
    stats["dedup_conflicts"] = int((grp["label_raw"].nunique(dropna=True) > 1).sum())
    df = df.assign(_label_med=grp["label_raw"].transform("median"))
    keep = df.drop_duplicates(subset=keys).copy()
    keep["label_raw"] = keep["_label_med"]
    keep = keep.drop(columns=["_label_med"])
    stats["drops"]["dedup"] = n_before - len(keep)
    stats["n_out"] = int(len(keep))
    return keep, stats


def iter_file_groups(path, chunksize=500_000):
    """Yield (source_file, dataframe) streaming; rows are contiguous per file."""
    buf, cur = [], None
    for chunk in pd.read_csv(path, chunksize=chunksize):
        for src, sub in chunk.groupby("source_file", sort=False):
            if src != cur and cur is not None:
                yield cur, pd.concat(buf, ignore_index=True)
                buf = []
            cur = src
            buf.append(sub)
    if cur is not None:
        yield cur, pd.concat(buf, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(OUT_DIR / "enriched.csv.gz"))
    ap.add_argument("--output", default=str(OUT_DIR / "cleaned.csv.gz"))
    ap.add_argument("--manifest", default=str(OUT_DIR / "manifest_stage3.json"))
    ap.add_argument("--limit", type=int, default=None,
                    help="max rows per source file (smoke test)")
    args = ap.parse_args()

    verify_alphabet()

    manifest, first, total = {"files": {}}, True, 0
    for src, sub in iter_file_groups(args.input):
        if args.limit:
            sub = sub.head(args.limit)
        corpus = sub["corpus"].iloc[0]
        if corpus not in ("main", "aux_pairwise"):
            manifest["files"][src] = {"excluded": corpus, "n_in": int(len(sub))}
            continue
        kept, stats = clean_file(sub)
        stats["corpus"] = corpus
        manifest["files"][src] = stats
        kept.to_csv(args.output, mode="w" if first else "a", header=first,
                    index=False, compression="gzip")
        first = False
        total += len(kept)
        print(f"  {src}: {stats['n_in']:>8,} -> {stats['n_out']:>8,} "
              f"(trimmed H={stats['trimmed_heavy']:,} L={stats['trimmed_light']:,}, "
              f"drops={ {k: v for k, v in stats['drops'].items() if v} })")

    manifest["n_rows_total"] = total
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[stage3] wrote {total:,} rows -> {args.output}")
    print(f"[stage3] manifest -> {args.manifest}")


if __name__ == "__main__":
    sys.exit(main())
