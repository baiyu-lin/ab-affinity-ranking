#!/usr/bin/env python
"""Independent self-eval of v3 predictions against the PUBLIC ground truth.

Joins an infer_benchmark.py ranking CSV with
flab_reference/benchmark_v3_public_groundtruth.csv on (group, seq_id) and
reports per-group Spearman of prediction vs public Kd-derived ranking.

This is a validation-only diagnostic (model selection / report material).
It is NEVER used to fill the final prediction file (see plan §2 T3.3).

Run from the repo root:
  .venv/bin/python training/eval_v3_public.py \
      --pred predictions/mAbs_ranking.csv \
      --truth data/benchmark_v3_public_groundtruth.csv
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--truth", required=True)
    args = ap.parse_args()

    pred = pd.read_csv(args.pred)
    truth = pd.read_csv(args.truth)
    df = truth.merge(pred, on=["group", "seq_id"], how="left",
                     suffixes=("", "_p"))
    n_missing = int(df["pred_rank"].isna().sum())
    if n_missing:
        raise SystemExit(f"ERROR: {n_missing}/40 benchmark rows missing "
                         f"from {args.pred}")

    score_cols = [c for c in df.columns if c.startswith("score_")]
    rank_cols = [c for c in df.columns if c.startswith("rank_")]
    df["neg_log_kd"] = -np.log10(df["Kd_nM"])

    report = {"rows": len(df), "groups": {}}
    for gid, sub in df.groupby("group"):
        g = {"ensemble_spearman": float(spearmanr(sub["pred_rank"],
                                                  sub["implied_rank"]).statistic)}
        for c in sorted(score_cols):
            g[c] = float(spearmanr(sub[c], sub["neg_log_kd"]).statistic)
        report["groups"][str(gid)] = g
    report["mean_ensemble_spearman"] = float(np.mean(
        [g["ensemble_spearman"] for g in report["groups"].values()]))
    for c in sorted(score_cols):
        report[f"mean_{c}"] = float(np.mean(
            [g[c] for g in report["groups"].values()]))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
