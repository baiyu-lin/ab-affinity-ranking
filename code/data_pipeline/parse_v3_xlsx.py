#!/usr/bin/env python
"""Parse the v3 benchmark xlsx into the canonical benchmark test CSV.

Source : predictions/benchmark_v3.xlsx  (v3 benchmark xlsx, mAbs sheet;
         not tracked — place it manually before running)
Output : code/data/benchmark_mAbs_v3.csv  (40 rows: group, seq_id, antigen_name, antigen, VH, VL)

The v3 CSV is generated ONLY by this script (traceability). If the output file
already exists, its content is diffed against the freshly parsed table and any
mismatch aborts the write unless --force is given.

Run from anywhere:
    .venv/bin/python data_pipeline/parse_v3_xlsx.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
PRELIM_DIR = CODE_DIR.parent
DEFAULT_XLSX = PRELIM_DIR / "predictions" / "benchmark_v3.xlsx"
DEFAULT_OUT = CODE_DIR / "data" / "benchmark_mAbs_v3.csv"

# v3 template header uses the typo "Sequene"; match case-insensitively after strip
COL_CANDIDATES = {
    "group": ["group"],
    "antigen_name": ["antigen"],
    "antigen": ["antigen sequene", "antigen sequence"],
    "seq_id": ["sequene id", "sequence id", "seq id", "seq_id"],
    "VH": ["vh"],
    "VL": ["vl"],
}
OUT_COLS = ["group", "seq_id", "antigen_name", "antigen", "VH", "VL"]


def resolve_columns(header):
    """Map output column names to actual xlsx header labels."""
    normed = {str(h).strip().lower(): h for h in header}
    mapping = {}
    for out_col, cands in COL_CANDIDATES.items():
        for c in cands:
            if c in normed:
                mapping[out_col] = normed[c]
                break
        else:
            sys.exit(f"ERROR: no xlsx column found for '{out_col}' "
                     f"(tried {cands}); header={list(header)}")
    return mapping


def load_table(xlsx_path):
    book = pd.read_excel(xlsx_path, sheet_name=None, dtype=str)
    mab_sheets = [n for n in book if "mab" in n.lower()]
    sheet = mab_sheets[0] if mab_sheets else next(iter(book))
    if len(book) > 1:
        print(f"note: workbook has sheets {list(book)}; using '{sheet}'")
    raw = book[sheet]
    mapping = resolve_columns(raw.columns)
    df = pd.DataFrame({out: raw[src].astype(str).str.strip()
                       for out, src in mapping.items()})
    # official sheet merges Group/Antigen/Antigen-Sequene cells per group block:
    # only the first row of each block is filled, the rest read back as NaN
    df[["group", "antigen_name", "antigen"]] = (
        df[["group", "antigen_name", "antigen"]].ffill())
    df["group"] = df["group"].astype(int)
    return df[OUT_COLS]


def validate(df):
    errors = []
    if len(df) != 40:
        errors.append(f"expected 40 rows, got {len(df)}")
    counts = df.groupby("group").size()
    if not all(n == 20 for n in counts):
        errors.append(f"expected 20 rows per group, got {counts.to_dict()}")
    if df.duplicated(subset=["group", "seq_id"]).any():
        errors.append("duplicated (group, seq_id) pairs")
    g2_vl = df.loc[df["group"] == 2, "VL"].unique()
    if len(g2_vl) != 1:
        errors.append(f"group 2 VL should be identical across 20 rows, got {len(g2_vl)}")
    bad_chars = df[~df[["antigen", "VH", "VL"]].apply(
        lambda s: s.str.fullmatch(r"[A-Z]+")).any(axis=1)]
    if len(bad_chars):
        errors.append(f"non-AA characters in rows: {bad_chars['seq_id'].tolist()}")
    if df[OUT_COLS].isna().any().any() or (df[OUT_COLS] == "").any().any():
        errors.append("empty cells present")
    if errors:
        for e in errors:
            print(f"VALIDATION FAIL: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"validation ok: 40 rows, groups={counts.to_dict()}, "
          f"group2 VL unique={len(g2_vl)}")


def diff_existing(df_new, out_path):
    """Return True if an existing CSV at out_path equals df_new cell-by-cell."""
    if not out_path.exists():
        return None
    df_old = pd.read_csv(out_path, dtype=str)
    df_cmp = df_new.astype(str)
    if list(df_old.columns) != OUT_COLS or df_old.shape != df_cmp.shape:
        same = False
    else:
        same = df_old.reset_index(drop=True).equals(df_cmp.reset_index(drop=True))
    if not same:
        merged = df_old.merge(df_cmp, on=["group", "seq_id"], how="outer",
                              suffixes=("_old", "_new"), indicator=True)
        n_diff = (merged["_merge"] != "both").sum()
        for c in ["antigen_name", "antigen", "VH", "VL"]:
            n_diff += (merged[f"{c}_old"] != merged[f"{c}_new"]).sum()
        print(f"existing CSV DIFFERS ({n_diff} cell/row mismatches)")
    return same


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true",
                    help="overwrite even if the existing CSV differs")
    args = ap.parse_args()

    if not args.xlsx.exists():
        sys.exit(f"ERROR: xlsx not found: {args.xlsx}")
    df = load_table(args.xlsx)
    validate(df)

    same = diff_existing(df, args.out)
    if same is True:
        print(f"existing {args.out} is identical (40/40 match) -> left untouched")
        return
    if same is False and not args.force:
        sys.exit("ERROR: existing CSV differs; re-run with --force to overwrite")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
