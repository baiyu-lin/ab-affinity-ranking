#!/usr/bin/env python
"""Stage 1: walk the raw CSV tree, skip preambles, normalize schemas.

Emits one long table (normalized.csv.gz) + manifest_stage1.json.
See data_pipeline/README.md. Run from repo root:
    .venv/bin/python data_pipeline/01_schema_normalization.py [--limit N]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path("data/raw/sequences")
OUT_DIR = Path("data_pipeline/output")

HEAVY_COLS = ["heavy", "HC", "Ab_heavy_chain_seq"]
LIGHT_COLS = ["light", "LC", "Ab_light_chain_seq"]
ANTIGEN_COLS = ["Antigen", "Ag_name", "Target"]

# tokens that mark the true header line when skipping preambles
HEADER_TOKENS = set(HEAVY_COLS + LIGHT_COLS + ANTIGEN_COLS +
                    ["fitness", "Pred_affinity", "genotype", "id", "POI",
                     "Ab_name", "Ab", "chain", "Chain Mutated", "sequence"])

RE_NEGLOG = re.compile(r"neg.?log|^\s*-\s*log", re.I)
RE_NM = re.compile(r"\bnm\b|\[nm\]|\(nm\)", re.I)
RE_M_UNIT = re.compile(r"\[m\]|\(m\)|\(m\s*\)|\bkd \(m", re.I)
RE_IC50 = re.compile(r"ic50|ec50|adcc", re.I)


def find_header_offset(path, max_lines=200):
    """Return (offset, encoding). Header = first line containing a known column."""
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(path, encoding=enc, newline="") as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    cells = {c.strip().strip('"') for c in line.rstrip("\n").split(",")}
                    if cells & HEADER_TOKENS:
                        return i, enc
        except UnicodeDecodeError:
            continue
    return None, None


def is_kd_col(name):
    lo = name.lower()
    return "kd" in lo or "affinity" in lo


def choose_label(cols, sample):
    """Pick (label_col, branch, note) from header columns + a data sample."""
    cols = list(cols)
    # 1. already-harmonized -log / neg_log columns
    for c in cols:
        if RE_NEGLOG.search(c):
            return c, "neglog_asis", ""
    # 2. Pred_affinity screening score (rank-only)
    if "Pred_affinity" in cols:
        return "Pred_affinity", "pred_affinity_rankonly", ""
    # 3. fitness, with pairwise cross-checks against unit-Kd columns
    if "fitness" in cols:
        fit = pd.to_numeric(sample["fitness"], errors="coerce")
        unit_cols = [c for c in cols if c != "fitness" and is_kd_col(c)
                     and (RE_NM.search(c) or RE_M_UNIT.search(c))]
        for c in unit_cols:
            kd = pd.to_numeric(sample[c], errors="coerce")
            both = pd.DataFrame({"f": fit, "k": kd}).dropna()
            both = both[both["k"] > 0]
            if len(both) < 3:
                continue
            is_nm = bool(RE_NM.search(c))
            lg = np.log10(both["k"])
            harmonized = -lg + (9.0 if is_nm else 0.0)
            if (both["f"] - lg).abs().median() < 0.1:
                # fitness = +log10(raw Kd) — wrong direction (AbRank);
                # the unit column is the trustworthy source
                branch = "kd_nm" if is_nm else "kd_m"
                return c, branch, "fitness = +log10(raw Kd); using unit column"
            if (both["f"] - harmonized).abs().median() < 0.1:
                return "fitness", "fitness_range", "fitness verified = -log10(Kd M)"
            if (both["f"] - both["k"]).abs().median() <= 1e-9 + 0.05 * both["k"].abs().median():
                if is_nm:  # raw nM duplicate: fitness_range cannot detect nM-ness
                    return c, "kd_nm", "fitness duplicates raw nM Kd; using unit column"
                # raw M duplicate: keep fitness so the stage-3 range check fires
                return "fitness", "fitness_range", \
                    "fitness duplicates raw Kd (M); range check will log-transform"
        if not unit_cols:
            # ELISA-style "* Binding" readout duplicated into fitness (makowski)
            for c in cols:
                if c != "fitness" and re.search(r"binding$", c.strip(), re.I):
                    b = pd.to_numeric(sample[c], errors="coerce").dropna()
                    med = fit.median()
                    if len(b) and abs(b.median() - med) <= 2 * abs(med) + 1e-12:
                        return c, "rankonly_excluded", "fitness duplicates ELISA binding readout"
        return "fitness", "fitness_range", ""
    # 4. Kd in nM
    for c in cols:
        if is_kd_col(c) and RE_NM.search(c):
            return c, "kd_nm", ""
    # 5. Kd in M
    for c in cols:
        if is_kd_col(c) and RE_M_UNIT.search(c):
            return c, "kd_m", ""
    # 6. IC50/EC50/ADCC/ELISA-style binding readouts (excluded, rank-only)
    for c in cols:
        if RE_IC50.search(c) or re.search(r"binding$", c.strip(), re.I):
            return c, "rankonly_excluded", ""
    return None, "none", "no label column detected"


def classify_corpus(relpath, label_col, branch, sample):
    name = relpath.name.lower()
    if "binary" in name:
        return "aux_pairwise"
    if branch == "rankonly_excluded":
        if label_col and re.search(r"binding$", label_col.strip(), re.I) and not RE_IC50.search(label_col):
            return "excluded_rank_only"
        return "excluded_ic50_ec50"
    if RE_IC50.search(name):
        return "excluded_ic50_ec50"
    # binary content check (fitness in {0,1})
    if label_col and label_col in sample:
        vals = pd.to_numeric(sample[label_col], errors="coerce").dropna().unique()
        if len(vals) and set(vals.tolist()).issubset({0.0, 1.0}):
            return "aux_pairwise"
    return "main"


def process_file(path, relpath, limit):
    """Normalize one CSV. Returns (dataframe_or_None, manifest_entry)."""
    entry = {"file": relpath.as_posix(), "skipped": None}

    offset, enc = find_header_offset(path)
    if offset is None:
        entry["skipped"] = "no header found"
        return None, entry
    entry["header_offset"] = offset
    entry["encoding"] = enc

    sample_n = min(limit, 50000) if limit else 50000
    sample = pd.read_csv(path, encoding=enc, skiprows=offset, dtype=str,
                         nrows=sample_n, engine="python", on_bad_lines="skip")
    sample.columns = [c.strip() for c in sample.columns]
    cols = list(sample.columns)
    entry["columns"] = cols

    heavy_col = next((c for c in HEAVY_COLS if c in cols), None)
    if heavy_col is None:
        entry["skipped"] = "no antibody schema"
        return None, entry
    light_col = next((c for c in LIGHT_COLS if c in cols), None)
    antigen_col = next((c for c in ANTIGEN_COLS if c in cols), None)

    label_col, branch, note = choose_label(cols, sample)
    corpus = classify_corpus(relpath, label_col, branch, sample)
    if corpus == "aux_pairwise" and branch not in ("fitness_range",):
        # binary files: label is the {0,1} column
        if "fitness" in cols:
            label_col, branch = "fitness", "binary"
        elif label_col is None:
            for c in cols:
                vals = pd.to_numeric(sample[c], errors="coerce").dropna().unique()
                if len(vals) and set(vals.tolist()).issubset({0.0, 1.0}):
                    label_col, branch = c, "binary"
                    break
    if corpus == "aux_pairwise" and branch == "fitness_range":
        branch = "binary"

    entry.update({"heavy_col": heavy_col, "light_col": light_col,
                  "antigen_col": antigen_col, "label_col": label_col,
                  "label_branch": branch, "corpus": corpus})
    if note:
        entry["label_note"] = note

    ag_seq_col = "antigen_seq" if "antigen_seq" in cols else None
    usecols = [heavy_col] + ([light_col] if light_col else []) + \
              ([antigen_col] if antigen_col else []) + \
              ([label_col] if label_col else []) + (["format"] if "format" in cols else []) + \
              ([ag_seq_col] if ag_seq_col else [])
    nrows = limit if limit else None
    try:
        df = pd.read_csv(path, encoding=enc, skiprows=offset, dtype=str,
                         usecols=usecols, nrows=nrows)
    except Exception:  # fall back to the tolerant python engine
        df = pd.read_csv(path, encoding=enc, skiprows=offset, dtype=str,
                         usecols=usecols, nrows=nrows, engine="python",
                         on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]

    out = pd.DataFrame({
        "source_file": relpath.as_posix(),
        "row_idx": np.arange(len(df), dtype=np.int64),
        "seq_heavy": df[heavy_col],
        "seq_light": df[light_col] if light_col else pd.Series(pd.NA, index=df.index, dtype="object"),
        "label_raw": pd.to_numeric(df[label_col], errors="coerce") if label_col else np.nan,
        "label_col": label_col if label_col else "",
        "antigen": df[antigen_col].fillna("single") if antigen_col else "single",
        "format": df["format"].fillna("scFv") if "format" in df.columns
                  else ("VHH" if light_col is None else "scFv"),
        "corpus": corpus,
        "ag_seq_raw": df[ag_seq_col] if ag_seq_col else pd.Series(pd.NA, index=df.index, dtype="object"),
    })
    out = out[out["seq_heavy"].notna()]
    entry["n_rows"] = int(len(out))
    entry["n_label_na"] = int(out["label_raw"].isna().sum())
    entry["n_antigens"] = int(out["antigen"].nunique())
    return out, entry


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(DATA_ROOT), help="raw data root")
    ap.add_argument("--output", default=str(OUT_DIR / "normalized.csv.gz"))
    ap.add_argument("--manifest", default=str(OUT_DIR / "manifest_stage1.json"))
    ap.add_argument("--limit", type=int, default=None, help="max rows per file (smoke test)")
    args = ap.parse_args()

    root = Path(args.input)
    files = sorted(p for p in root.rglob("*.csv"))
    print(f"[stage1] {len(files)} CSVs under {root}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    manifest, first = [], True
    total = 0
    for path in files:
        rel = path.relative_to(root)
        df, entry = process_file(path, rel, args.limit)
        manifest.append(entry)
        if df is None:
            print(f"  SKIP {rel}: {entry['skipped']}")
            continue
        df.to_csv(args.output, mode="w" if first else "a", header=first,
                  index=False, compression="gzip")
        first = False
        total += len(df)
        print(f"  {rel}: {len(df):>8,} rows  corpus={entry['corpus']:<18} "
              f"label={entry['label_col']} ({entry['label_branch']})")

    with open(args.manifest, "w") as f:
        json.dump({"n_files": len(files),
                   "n_rows_total": total,
                   "files": manifest}, f, indent=2, ensure_ascii=False)
    print(f"[stage1] wrote {total:,} rows -> {args.output}")
    print(f"[stage1] manifest -> {args.manifest}")


if __name__ == "__main__":
    sys.exit(main())
