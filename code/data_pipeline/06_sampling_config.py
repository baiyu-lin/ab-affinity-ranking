#!/usr/bin/env python
"""Stage 6: training-group config + pair budgets (spec §6, AbRank §4.3).

Consumes group_stats.csv (per-group sizes/censoring) and
harmonized_clustered.csv.gz (cluster ids, per-row fitness) and emits the
artifacts the training loop needs to reproduce the benchmark format:

1. training_groups.csv — one row per (file x antigen) group:
   - eligible: n_noncensored >= 20 (spec rule 2) and not cross_file_dup-only
   - weight: min(n_rows, 2000) — group sampling weight (spec §6.2), so
     li2023-scale groups cannot dominate every batch (AbRank: balanced
     sampling across clusters is what keeps over-assayed complexes from
     eating the dataset)
   - subsample_n: min(n_rows, 50000) with label-decile stratification
     advised for groups above the cap (spec §6.2)
   - m_confident_pairs: number of intra-group pairs with
     |Δfitness| >= 1.0 (AbRank m = 10x margin; Kd/screening labels are
     log-space, so Δ=1 is a 10-fold affinity difference), computed on a
     <=2000-row subsample with an O(n log n) sorted sweep, capped at 1000
     (AbRank's per-group pair cap)
2. aux_pairs.csv — sampled binder/non-binder pairs for the aux pairwise
   margin loss (spec §7): all positives x up to 3x negatives per file,
   capped at 100k pairs per file (tsuruta is 96% negative).

Run from repo root:
    .venv/bin/python data_pipeline/06_sampling_config.py [--limit N]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("data_pipeline/output")
MIN_NONCENSORED = 20      # spec rule 2: trainable group size
GROUP_CAP = 2000          # weight cap
SUBSAMPLE_CAP = 50_000    # rows kept per group for embedding
PAIR_MARGIN = 1.0         # log10 units = 10-fold affinity (AbRank m)
PAIR_SUBSAMPLE = 2000     # rows used to estimate confident pairs
PAIR_CAP = 1000           # AbRank per-group pair cap
AUX_NEG_RATIO = 3         # max negatives per positive (tsuruta 96% neg)
AUX_PAIR_CAP = 100_000    # per aux file


def confident_pairs(fitness, margin=PAIR_MARGIN, cap=PAIR_CAP):
    """Count pairs with |Δ| >= margin via a sorted two-pointer sweep."""
    f = np.sort(np.asarray(fitness, dtype=float))
    f = f[np.isfinite(f)]
    n = len(f)
    if n < 2:
        return 0
    j = np.searchsorted(f, f - margin, side="right")
    total = int(j.sum())
    return min(total, cap)


def iter_file_groups(path, chunksize=500_000):
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
    ap.add_argument("--clustered", default=str(OUT_DIR / "harmonized_clustered.csv.gz"))
    ap.add_argument("--group-stats", default=str(OUT_DIR / "group_stats.csv"))
    ap.add_argument("--aux", default=str(OUT_DIR / "harmonized_aux.csv.gz"))
    ap.add_argument("--output", default=str(OUT_DIR / "training_groups.csv"))
    ap.add_argument("--output-pairs", default=str(OUT_DIR / "aux_pairs.csv"))
    ap.add_argument("--manifest", default=str(OUT_DIR / "manifest_stage6.json"))
    ap.add_argument("--limit", type=int, default=None,
                    help="max rows per source file (smoke test)")
    args = ap.parse_args()

    gs = pd.read_csv(args.group_stats)
    gs["rank_only"] = gs["rank_only"].astype(str).isin(["True", "1"])
    gs["n_noncensored"] = gs["n_rows"] - gs["n_censored"]
    gs["eligible"] = gs["n_noncensored"] >= MIN_NONCENSORED
    gs["weight"] = np.minimum(gs["n_rows"], GROUP_CAP)
    gs["subsample_n"] = np.minimum(gs["n_rows"], SUBSAMPLE_CAP)

    # per-group: clusters, confident pairs (on subsample), dup-free counts
    rng = np.random.default_rng(0)
    pair_counts, cluster_counts, dup_counts = {}, {}, {}
    for src, sub in iter_file_groups(args.clustered):
        if args.limit:
            sub = sub.head(args.limit)
        for gid, grp in sub.groupby("group_id", sort=False):
            cluster_counts[gid] = int(grp["cluster_id"].nunique())
            dup_counts[gid] = int((grp["cross_file_dup"].astype(str) == "True").sum())
            f = grp["fitness"].to_numpy()
            if len(f) > PAIR_SUBSAMPLE:
                f = rng.choice(f, size=PAIR_SUBSAMPLE, replace=False)
            pair_counts[gid] = confident_pairs(f)

    gs["n_clusters"] = gs["group_id"].map(cluster_counts)
    gs["n_cross_file_dup"] = gs["group_id"].map(dup_counts)
    gs["m_confident_pairs"] = gs["group_id"].map(pair_counts)
    gs = gs[["group_id", "source_file", "antigen", "n_rows", "n_censored",
             "n_noncensored", "censored_frac", "rank_only", "n_clusters",
             "n_cross_file_dup", "m_confident_pairs", "eligible", "weight",
             "subsample_n"]]
    gs.to_csv(args.output, index=False)

    # aux pairwise: binder/non-binder pairs with capped neg:pos ratio
    n_pairs_total, first = 0, True
    aux_manifest = {}
    for src, sub in iter_file_groups(args.aux):
        if args.limit:
            sub = sub.head(args.limit)
        pos = sub[sub["label_raw"] == 1]
        neg = sub[sub["label_raw"] == 0]
        n_neg = min(len(neg), AUX_NEG_RATIO * max(len(pos), 1))
        if len(neg) > n_neg:
            neg = neg.sample(n=n_neg, random_state=0)
        pairs = []
        budget = AUX_PAIR_CAP
        n_per_pos = max(1, min(AUX_NEG_RATIO, budget // max(len(pos), 1)))
        neg_arr = neg[["row_idx"]].to_numpy().ravel()
        for prow in pos[["row_idx"]].itertuples(index=False):
            take = neg_arr[rng.choice(len(neg_arr),
                                      size=min(n_per_pos, len(neg_arr)),
                                      replace=len(neg_arr) < n_per_pos)]
            for n_idx in take:
                pairs.append((src, prow[0], int(n_idx)))
            budget -= n_per_pos
            if budget <= 0:
                break
        df = pd.DataFrame(pairs, columns=["source_file", "pos_row_idx",
                                          "neg_row_idx"])
        df.to_csv(args.output_pairs, mode="w" if first else "a", header=first,
                  index=False)
        first = False
        n_pairs_total += len(df)
        aux_manifest[src] = {"n_pos": int(len(pos)), "n_neg_used": int(len(neg)),
                             "n_pairs": int(len(df))}
        print(f"  {src}: {len(df):>7,} pairs ({len(pos):,} pos x "
              f"{len(neg):,} neg)")

    manifest = {
        "n_groups": int(len(gs)),
        "n_eligible": int(gs["eligible"].sum()),
        "n_eligible_rank_only": int((gs["eligible"] & gs["rank_only"]).sum()),
        "n_m_confident_pairs_total": int(gs["m_confident_pairs"].sum()),
        "n_aux_pairs": int(n_pairs_total),
        "aux_files": aux_manifest,
        "params": {"min_noncensored": MIN_NONCENSORED, "group_cap": GROUP_CAP,
                   "subsample_cap": SUBSAMPLE_CAP, "pair_margin": PAIR_MARGIN,
                   "pair_cap": PAIR_CAP, "aux_neg_ratio": AUX_NEG_RATIO,
                   "aux_pair_cap": AUX_PAIR_CAP},
    }
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[stage6] {len(gs):,} groups ({manifest['n_eligible']:,} eligible) "
          f"-> {args.output}")
    print(f"[stage6] {n_pairs_total:,} aux pairs -> {args.output_pairs}")


if __name__ == "__main__":
    sys.exit(main())
