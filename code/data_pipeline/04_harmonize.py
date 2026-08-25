#!/usr/bin/env python
"""Stage 4: label harmonization + operator-aware censoring + sign audit.

Supersedes the retired v1 `03_label_harmonization.py` (now in
`retired/data_pipeline_v1/`); implements research §6 gaps P0/P1:

1. Per-row label kinds (AbRank). The v1 stage used one label column per file,
   which NaN-dropped the 247k AbRank rows that only have an escape fraction
   or an IC50. This stage harmonizes per row: kd_nm -> fitness =
   9 - log10(Kd[nM]); escape -> fitness = -escape (higher escape = weaker
   binder); ic50_ugml -> fitness = -log10(IC50). Each (file x antigen) group
   keeps a single dominant label kind (mixed-kind rows are dropped and
   counted); non-Kd groups are marked rank_only — their order is meaningful,
   their scale is not, and they are never pooled with Kd groups
   (research §4.1).
2. Operator-aware censoring. aff_op '>' rows are right-censored
   ("weaker than"), '<' left-censored ("stronger than") — these are marked
   censored with an explicit censor_dir, on top of the modal-value
   detection inherited from v1 (SKEMPI 2.0: threshold values are
   inequalities, not points).
3. Label sign audit. For every main-corpus file, Spearman(fitness,
   label_raw_v2) must have the expected sign for the file's branch
   (negative for Kd-like raw labels, positive for already-harmonized ones).
   A mismatch means the ranker would train inverted (the AbRank
   fitness = +log10(Kd) trap from research §2.2).

Input : data_pipeline/output/cleaned.csv.gz (stage 3)
        data_pipeline/output/manifest_stage1.json (stage-1 label branches)
Output: data_pipeline/output/harmonized.csv.gz (main corpus)
        data_pipeline/output/harmonized_aux.csv.gz (aux pairwise)
        data_pipeline/output/group_stats.csv
        data_pipeline/output/manifest_stage4.json

Run from repo root:
    .venv/bin/python data_pipeline/04_harmonize.py [--limit N]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

V1_MANIFEST = Path("data_pipeline/output/manifest_stage1.json")
OUT_DIR = Path("data_pipeline/output")
ABRANK_SRC = "4/AbRank_dataset.csv"
CENSOR_FRAC = 0.30  # modal fitness covering >30% of group rows -> censored

# expected Spearman sign between fitness and the raw label, per branch
BRANCH_SIGN = {"neglog_asis": +1, "fitness_range": +1,
               "pred_affinity_rankonly": +1, "kd_nm": -1, "kd_m": -1}

# canonical main-corpus column order (chunks are appended header-less)
MAIN_COLUMNS = ["source_file", "row_idx", "seq_heavy", "seq_light",
                "label_raw", "label_col", "antigen", "format", "corpus",
                "aff_op", "ag_seq", "ab_cluster_raw", "ag_cluster_raw",
                "label_kind", "label_raw_v2", "group_id", "fitness",
                "dominant_kind", "rank_only", "censored", "censor_dir",
                "fitness_z", "fitness_rank_pct"]


def harmonize_rows(sub, branch):
    """v1 logic: return (fitness, info) for one file's main-corpus rows."""
    x = sub["label_raw_v2"].astype(float)
    info = {"branch_in": branch, "unit_corrected": False,
            "branch_taken": branch}
    med = x.median()
    if branch == "neglog_asis":
        if med < 0.01:
            info["branch_taken"] = "neglog_raw_kd_log_transformed"
            return -np.log10(x.where(x > 0)), info
        return x, info
    if branch == "pred_affinity_rankonly":
        return x, info
    if branch == "fitness_range":
        if med < 0.01:
            info["branch_taken"] = "fitness_raw_kd_log_transformed"
            return -np.log10(x.where(x > 0)), info
        return x, info
    if branch == "kd_nm":
        return 9.0 - np.log10(x.where(x > 0)), info
    if branch == "kd_m":
        if med > 0.01:
            info["unit_corrected"] = True
            info["branch_taken"] = "kd_m_unit_corrected_to_nm"
            return 9.0 - np.log10(x.where(x > 0)), info
        return -np.log10(x.where(x > 0)), info
    raise ValueError(f"unexpected branch for main corpus: {branch}")


def harmonize_abrank(sub, info):
    """Per-row label kinds; per-group dominant kind; rank_only for non-Kd."""
    x = sub["label_raw_v2"].astype(float)
    kind = sub["label_kind"]
    fitness = pd.Series(np.nan, index=sub.index)
    fitness[kind == "kd_nm"] = 9.0 - np.log10(x[kind == "kd_nm"].where(lambda s: s > 0))
    fitness[kind == "escape"] = -x[kind == "escape"]
    fitness[kind == "ic50_ugml"] = -np.log10(x[kind == "ic50_ugml"].where(lambda s: s > 0))
    sub = sub.assign(fitness=fitness)
    n_before = len(sub)
    sub = sub[sub["fitness"].notna()].copy()
    info["n_dropped_no_label"] = n_before - len(sub)
    # per-group dominant label kind; drop rows of other kinds
    g = sub.groupby("group_id", sort=False)["label_kind"]
    dominant = g.agg(lambda s: s.value_counts().idxmax())
    sub = sub.merge(dominant.rename("dominant_kind"), left_on="group_id",
                    right_index=True, how="left")
    mixed = sub["label_kind"] != sub["dominant_kind"]
    info["n_dropped_mixed_kind"] = int(mixed.sum())
    sub = sub[~mixed].copy()
    info["dominant_kind_counts"] = {k: int(v) for k, v in
                                    sub["dominant_kind"].value_counts().items()}
    sub["rank_only"] = sub["dominant_kind"] != "kd_nm"
    info["branch_taken"] = "abrank_per_row_kind"
    return sub, info


def sign_audit(sub, branch, info):
    """Spearman(fitness, label_raw_v2) must match the branch's expected sign."""
    kinds = sub.get("label_kind")
    s = sub
    if kinds is not None:  # audit AbRank per label kind instead
        out = {}
        for k, grp in sub.groupby("label_kind"):
            if len(grp) < 20:
                continue
            rho = spearmanr(grp["fitness"], grp["label_raw_v2"].astype(float)).statistic
            out[k] = round(float(rho), 4)
        # kd_nm / escape / ic50 are all "higher raw = weaker" -> expect -1
        bad = {k: r for k, r in out.items() if r > -0.9}
        info["sign_audit"] = out
        info["sign_audit_ok"] = not bad
        if bad:
            print(f"    WARNING sign audit failed: {bad}")
        return
    expect = BRANCH_SIGN.get(branch)
    if expect is None or len(s) < 20:
        info["sign_audit"] = "skipped"
        return
    rho = spearmanr(s["fitness"], s["label_raw_v2"].astype(float)).statistic
    ok = bool(np.sign(rho) == expect) if np.isfinite(rho) else False
    info["sign_audit"] = round(float(rho), 4)
    info["sign_audit_ok"] = ok
    if not ok:
        print(f"    WARNING sign audit failed: rho={rho:.3f} expected {'+' if expect > 0 else '-'}")


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
    ap.add_argument("--input", default=str(OUT_DIR / "cleaned.csv.gz"))
    ap.add_argument("--manifest-in", default=str(V1_MANIFEST))
    ap.add_argument("--output", default=str(OUT_DIR / "harmonized.csv.gz"))
    ap.add_argument("--output-aux", default=str(OUT_DIR / "harmonized_aux.csv.gz"))
    ap.add_argument("--group-stats", default=str(OUT_DIR / "group_stats.csv"))
    ap.add_argument("--manifest", default=str(OUT_DIR / "manifest_stage4.json"))
    ap.add_argument("--limit", type=int, default=None,
                    help="max rows per source file (smoke test)")
    args = ap.parse_args()

    branches = {e["file"]: e.get("label_branch")
                for e in json.load(open(args.manifest_in))["files"]}

    manifest, group_rows = {"files": {}}, []
    first_main = first_aux = True
    n_main = n_aux = 0
    for src, sub in iter_file_groups(args.input):
        if args.limit:
            sub = sub.head(args.limit)
        corpus = sub["corpus"].iloc[0]
        if corpus == "aux_pairwise":
            aux = sub.copy()
            aux["group_id"] = src + "||" + aux["antigen"].fillna("single")
            aux.to_csv(args.output_aux, mode="w" if first_aux else "a",
                       header=first_aux, index=False, compression="gzip")
            first_aux = False
            n_aux += len(aux)
            pos = int((aux["label_raw"] == 1).sum())
            manifest["files"][src] = {"corpus": corpus, "n": int(len(aux)),
                                      "n_positive": pos}
            print(f"  {src}: AUX {len(aux):>8,} rows ({pos:,} positive)")
            continue
        if corpus != "main":
            continue

        branch = branches.get(src)
        info = {"branch_in": branch, "n_in": int(len(sub))}
        sub["group_id"] = src + "||" + sub["antigen"].fillna("single")
        if src == ABRANK_SRC:
            sub, info = harmonize_abrank(sub, info)
            info["unit_corrected"] = False
        else:
            fitness, info = harmonize_rows(sub, branch)
            info["n_in"] = int(len(sub))
            sub = sub.assign(fitness=fitness)
            bad = sub["fitness"].isna()
            info["n_dropped_invalid_label"] = int(bad.sum())
            sub = sub[~bad].copy()
            sub["rank_only"] = branch == "pred_affinity_rankonly"
            sub["dominant_kind"] = branch

        sign_audit(sub, branch, info)

        # censoring: modal-value detection + aff_op operators
        g = sub.groupby("group_id", sort=False)
        modal = g["fitness"].agg(lambda s: s.value_counts().idxmax())
        modal_n = g["fitness"].agg(lambda s: s.value_counts().max())
        size = g.size()
        stats = pd.DataFrame({"modal": modal, "modal_n": modal_n, "n": size})
        stats["censored_n"] = np.where(stats["modal_n"] / stats["n"] > CENSOR_FRAC,
                                       stats["modal_n"], 0)
        sub = sub.merge(stats[["modal", "censored_n"]], left_on="group_id",
                        right_index=True, how="left")
        modal_cens = (sub["censored_n"] > 0) & (sub["fitness"] == sub["modal"])
        sub = sub.drop(columns=["modal", "censored_n"])
        sub["censored"] = modal_cens
        sub["censor_dir"] = np.where(modal_cens, "modal", "")
        if "aff_op" in sub.columns:
            op_weak = sub["aff_op"] == ">"
            op_strong = sub["aff_op"] == "<"
            sub.loc[op_weak, "censored"] = True
            sub.loc[op_weak, "censor_dir"] = "weaker_than"
            sub.loc[op_strong, "censored"] = True
            sub.loc[op_strong, "censor_dir"] = "stronger_than"
            info["n_op_censored"] = int((op_weak | op_strong).sum())

        # per-group normalization (never pooled across groups)
        gg = sub.groupby("group_id", sort=False)["fitness"]
        mu, sd = gg.transform("mean"), gg.transform("std")
        sub["fitness_z"] = ((sub["fitness"] - mu) / sd.replace(0, np.nan)).fillna(0.0)
        sub["fitness_rank_pct"] = gg.rank(pct=True)

        censed = sub.groupby("group_id", sort=False)["censored"].sum()
        rankonly = sub.groupby("group_id", sort=False)["rank_only"].first()
        for gid, nrow in size.items():
            group_rows.append({"group_id": gid, "source_file": src,
                               "antigen": gid.split("||", 1)[1],
                               "n_rows": int(nrow),
                               "n_censored": int(censed.get(gid, 0)),
                               "censored_frac": float(censed.get(gid, 0) / nrow),
                               "modal_fitness": float(stats.loc[gid, "modal"]),
                               "rank_only": bool(rankonly.get(gid, False))})

        info.update({"corpus": "main", "n_out": int(len(sub)),
                     "n_groups": int(len(size)),
                     "n_censored_rows": int(sub["censored"].sum()),
                     "censored_frac": float(sub["censored"].mean()) if len(sub) else 0.0})
        manifest["files"][src] = info
        # canonical column order: chunks are appended with header=False, so
        # every file must write identical columns in identical order
        sub = sub.reindex(columns=MAIN_COLUMNS)
        sub.to_csv(args.output, mode="w" if first_main else "a", header=first_main,
                   index=False, compression="gzip")
        first_main = False
        n_main += len(sub)
        print(f"  {src}: {info['n_in']:>8,} -> {info['n_out']:>8,} "
              f"groups={info['n_groups']:,} censored={info['censored_frac']:.3f} "
              f"branch={info['branch_taken']}"
              + (" UNIT-CORRECTED" if info.get("unit_corrected") else ""))

    pd.DataFrame(group_rows).to_csv(args.group_stats, index=False)
    manifest["n_main_rows"] = n_main
    manifest["n_aux_rows"] = n_aux
    manifest["n_groups"] = len(group_rows)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[stage4] main={n_main:,} -> {args.output}")
    print(f"[stage4] aux={n_aux:,} -> {args.output_aux}")
    print(f"[stage4] {len(group_rows):,} groups -> {args.group_stats}")


if __name__ == "__main__":
    sys.exit(main())
