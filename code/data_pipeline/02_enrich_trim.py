#!/usr/bin/env python
"""Stage 2: enrich the normalized table + trim constant regions.

Implements the P0 gaps from preparation/data-pipeline-research.md §6:

1. Constant-region trimming. AbRank (SAbDab-derived) heavy/light chains carry
   CH1/CL constant regions (median heavy length 217 aa), which the stage-3
   length filter (heavy 90-160, light 90-130) would delete wholesale. We cut
   each over-long chain right after its framework-4 anchor (heavy: ...WGxG..
   VTVSS; light: ...FGxG..[LV]EIK), before length QC runs in stage 3. Only
   over-long sequences are touched; V-domains are never this long.
2. AbRank enrichment. The raw AbRank CSV carries columns that stage 1
   dropped: Aff_op (measurement operator '=', '>', '<' — 10% of rows are
   right-censored), Ag_seq, escape fraction, IC50 [ug/mL], and the paper's
   own 75%-identity cluster ids. Rows are re-attached by row_idx and the join
   is validated by comparing heavy sequences.
3. Label recovery. Stage 1 kept only Affinity_Kd [nM] for AbRank, so the 247k
   escape-only / IC50-only rows were NaN-labelled and would die in stage 4.
   This stage assigns a per-row label_kind (kd_nm > escape > ic50_ugml) so
   those rows survive as rank-only group members (research §4.5: only
   within-target order of DMS/IC50 data is meaningful).

Input : data_pipeline/output/normalized.csv.gz (stage 1)
        data/raw/sequences/4/AbRank_dataset.csv (raw)
Output: data_pipeline/output/enriched.csv.gz
        data_pipeline/output/manifest_stage2.json

Run from repo root:
    .venv/bin/python data_pipeline/02_enrich_trim.py [--limit N]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V1_NORMALIZED = Path("data_pipeline/output/normalized.csv.gz")
ABRANK_RAW = Path("data/raw/sequences/4/AbRank_dataset.csv")
ABRANK_SRC = "4/AbRank_dataset.csv"
OUT_DIR = Path("data_pipeline/output")

# FW4 anchors. The V-domain ends at IMGT 128: heavy FW4 starts at the
# conserved WGxG (IMGT 118, 11 residues before the end), light FW4 at the
# conserved FGxG (10 residues before the end). Sequence species vary too much
# downstream (VTVSS / LTVSS / VTVSA / LVSVTA …) to match the full motif, so we
# anchor on WGxG / FGxG alone and cut at the fixed IMGT offset.
HEAVY_MAX, LIGHT_MAX = 160, 130  # stage-3 limits; trim only above these
RE_FW4_H = re.compile(r"WG\wG")
RE_FW4_L = re.compile(r"FG(?:\wG|G\w)")  # FGxG (kappa) or FGGx (lambda FGGATR…)
RE_FW4_LOOSE = re.compile(r"W\w{2,5}VTVSS")  # divergent FW4, e.g. WSPGTLVTVSS
FW4_H_MINPOS, FW4_H_CUT = 95, 11   # WGxG at ~103+, V-domain ends +11
FW4_L_MINPOS, FW4_L_CUT = 80, 10   # FGxG at ~88+, V-domain ends +10

ABRANK_USECOLS = ["Ab_heavy_chain_seq", "Aff_op", "Ag_seq", "escape",
                  "IC50 [ug/mL]", "Ab_Lev3_cluster", "Ag_Lev3_cluster"]


def trim_vdomain(seq, chain):
    """Cut constant-region tail after the FW4 anchor. Returns (seq, flag).

    flag: 'unchanged' | 'trimmed' | 'anchor_missing'. Only sequences longer
    than the stage-3 length cap are candidates. Some AbRank rows have the
    two chains swapped between columns, so both anchors are tried on either
    column (column's own anchor first); the loose VTVSS core catches
    divergent heavy FW4 variants (e.g. WSPGTLVTVSS).
    """
    if not isinstance(seq, str):
        return seq, "unchanged"
    limit = HEAVY_MAX if chain == "heavy" else LIGHT_MAX
    if len(seq) <= limit:
        return seq, "unchanged"
    h, l = (RE_FW4_H, FW4_H_MINPOS, FW4_H_CUT), (RE_FW4_L, FW4_L_MINPOS, FW4_L_CUT)
    anchors = [l, h] if chain == "light" else [h, l]
    for rx, min_pos, cut in anchors:
        m = next((mm for mm in rx.finditer(seq) if mm.start() >= min_pos), None)
        if m is not None:
            return seq[:m.start() + cut], "trimmed"
    m = next((mm for mm in RE_FW4_LOOSE.finditer(seq) if mm.start() >= 95), None)
    if m is not None:
        return seq[:m.end()], "trimmed"
    return seq, "anchor_missing"


def enrich_abrank(sub, extra, entry):
    """Attach aff_op/ag_seq/escape/ic50/clusters + per-row label kind."""
    idx = sub["row_idx"].to_numpy()
    ex = extra.iloc[idx].reset_index(drop=True)
    sub = sub.reset_index(drop=True)

    # join validation: heavy sequences must match (row_idx = raw row order)
    probe = sub["seq_heavy"].notna()
    n_probe = int(probe.sum())
    if n_probe:
        same = (sub.loc[probe, "seq_heavy"].to_numpy()
                == ex.loc[probe, "Ab_heavy_chain_seq"].to_numpy()).mean()
        entry["join_heavy_match_rate"] = round(float(same), 6)
        if same < 0.99:
            raise RuntimeError(f"AbRank join misaligned: heavy match {same:.4f}")

    sub["aff_op"] = ex["Aff_op"].fillna("=").where(ex["Aff_op"].notna(), "=")
    sub["ag_seq"] = ex["Ag_seq"].where(ex["Ag_seq"].notna(), pd.NA)
    sub["ab_cluster_raw"] = ex["Ab_Lev3_cluster"]
    sub["ag_cluster_raw"] = ex["Ag_Lev3_cluster"]

    kd = pd.to_numeric(sub["label_raw"], errors="coerce")  # Affinity_Kd [nM]
    esc = pd.to_numeric(ex["escape"], errors="coerce")
    ic50 = pd.to_numeric(ex["IC50 [ug/mL]"], errors="coerce")
    kind = np.select([kd.notna(), esc.notna(), ic50.notna()],
                     ["kd_nm", "escape", "ic50_ugml"], default="none")
    sub["label_kind"] = kind
    sub["label_raw_v2"] = np.select(
        [kd.notna(), esc.notna(), ic50.notna()], [kd, esc, ic50],
        default=np.nan)
    entry["label_kind_counts"] = {k: int(v) for k, v in
                                  pd.Series(kind).value_counts().items()}
    entry["aff_op_counts"] = {k: int(v) for k, v in
                              sub["aff_op"].value_counts().items()}
    return sub


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
    ap.add_argument("--input", default=str(V1_NORMALIZED))
    ap.add_argument("--abrank-raw", default=str(ABRANK_RAW))
    ap.add_argument("--output", default=str(OUT_DIR / "enriched.csv.gz"))
    ap.add_argument("--manifest", default=str(OUT_DIR / "manifest_stage2.json"))
    ap.add_argument("--limit", type=int, default=None,
                    help="max rows per source file (smoke test)")
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    extra = None
    manifest, first, total = {"files": {}}, True, 0
    for src, sub in iter_file_groups(args.input):
        if args.limit:
            sub = sub.head(args.limit)
        entry = {"n_in": int(len(sub))}

        if src == ABRANK_SRC:
            if extra is None:
                extra = pd.read_csv(args.abrank_raw, dtype=str,
                                    usecols=ABRANK_USECOLS)
                extra.columns = [c.strip() for c in extra.columns]
            sub = enrich_abrank(sub, extra, entry)
        else:
            ag = sub["ag_seq_raw"] if "ag_seq_raw" in sub.columns else pd.NA
            sub = sub.assign(aff_op="=", ag_seq=ag, ab_cluster_raw=pd.NA,
                             ag_cluster_raw=pd.NA, label_kind="",
                             label_raw_v2=sub["label_raw"])
        if "ag_seq_raw" in sub.columns:
            sub = sub.drop(columns=["ag_seq_raw"])

        # constant-region trim (before stage-3 length QC); only over-long
        # chains are candidates, so map over those rows only
        for col, chain, lim in (("seq_heavy", "heavy", HEAVY_MAX),
                                ("seq_light", "light", LIGHT_MAX)):
            s = sub[col]
            long_mask = s.str.len().gt(lim).fillna(False)
            trimmed = s[long_mask].map(lambda x: trim_vdomain(x, chain))
            if len(trimmed):
                sub.loc[long_mask, col] = [t[0] for t in trimmed]
            flags = pd.Series([t[1] for t in trimmed])
            entry[f"trimmed_{chain}"] = int((flags == "trimmed").sum())
            entry[f"anchor_missing_{chain}"] = int(
                (flags == "anchor_missing").sum())
        entry["n_out"] = int(len(sub))

        manifest["files"][src] = entry
        sub.to_csv(args.output, mode="w" if first else "a", header=first,
                   index=False, compression="gzip")
        first = False
        total += len(sub)
        print(f"  {src}: {entry['n_in']:>8,} rows "
              f"trimH={entry['trimmed_heavy']:,} trimL={entry['trimmed_light']:,} "
              f"noAnchorH={entry['anchor_missing_heavy']:,}"
              + (f" kinds={entry['label_kind_counts']}"
                 if "label_kind_counts" in entry else ""))

    manifest["n_rows_total"] = total
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[stage2] wrote {total:,} rows -> {args.output}")
    print(f"[stage2] manifest -> {args.manifest}")


if __name__ == "__main__":
    sys.exit(main())
