#!/usr/bin/env python
"""Stage 8 (v3): benchmark de-leakage of the harmonized corpus.

Exact-match the 40 v3 benchmark antibodies (VH+VL pairs) against
harmonized_clustered.csv.gz, then expand the removal set to the full
antibody cluster (ab_cluster_raw) of every hit. Outputs:

  data_pipeline/output/deleak_v3_hits.csv     matched rows x benchmark seq
  data_pipeline/output/deleak_v3_exclude.json {"ab_clusters": [...],
                                               "rows": [[source_file,row_idx],...],
                                               "stats": {...}}

The JSON is consumed by training/train.py via data.exclude_json — the corpus
itself is left untouched (exclusion happens at load time).

Run from the repo root:
    .venv/bin/python data_pipeline/08_deleak_v3.py [--benchmark ...] [--no-cluster-expansion]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

PIPE_OUT = Path("data_pipeline/output")
DEFAULT_BENCH = Path("data/benchmark_mAbs_v3.csv")
DEFAULT_HARM = PIPE_OUT / "harmonized_clustered.csv.gz"

COLS = ["source_file", "row_idx", "seq_heavy", "seq_light", "antigen",
        "group_id", "ab_cluster_raw", "fitness", "cross_file_dup"]


def norm(s):
    return s.fillna("").str.strip().str.upper()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCH)
    ap.add_argument("--harmonized", type=Path, default=DEFAULT_HARM)
    ap.add_argument("--no-cluster-expansion", action="store_true",
                    help="remove only exact-match rows, not their clusters "
                         "(risk fallback, see plan §4)")
    args = ap.parse_args()

    bench = pd.read_csv(args.benchmark, dtype=str)
    bench["vh_n"] = norm(bench["VH"])
    bench["vl_n"] = norm(bench["VL"])
    keys = set(zip(bench["vh_n"], bench["vl_n"]))
    print(f"benchmark: {len(bench)} antibodies "
          f"({len(keys)} unique VH+VL pairs)")

    df = pd.read_csv(args.harmonized, low_memory=False, usecols=COLS)
    n_total = len(df)
    df["vh_n"] = norm(df["seq_heavy"])
    df["vl_n"] = norm(df["seq_light"])
    hit_mask = pd.Series(
        [k in keys for k in zip(df["vh_n"], df["vl_n"])], index=df.index)
    hits = df[hit_mask].copy()
    print(f"exact (VH,VL) hits in corpus: {len(hits)} rows")

    # per-benchmark-seq coverage + antigen context sanity
    bench_hit = bench.set_index(["vh_n", "vl_n"]).join(
        hits.set_index(["vh_n", "vl_n"])[["source_file"]],
        how="left", rsuffix="_hit")
    coverage = bench_hit.groupby(["group", "seq_id"])["source_file"].apply(
        lambda s: s.notna().sum())
    n_cov = int((coverage > 0).sum())
    print(f"benchmark seqs with >=1 corpus hit: {n_cov}/{len(bench)}")
    if n_cov < len(bench):
        missing = coverage[coverage == 0].index.tolist()
        print(f"  no-hit benchmark seqs: {missing}")

    # removal set: hit clusters + (for cluster-less hits) the rows themselves
    cl = hits["ab_cluster_raw"].dropna().unique().tolist()
    orphan = hits[hits["ab_cluster_raw"].isna()]
    if args.no_cluster_expansion:
        cl = []
        orphan = hits
    exclude_rows = orphan[["source_file", "row_idx"]].values.tolist()

    rm_mask = pd.Series(False, index=df.index)
    if cl:
        rm_mask |= df["ab_cluster_raw"].isin(cl)
    if exclude_rows:
        rowset = {(sf, int(ri)) for sf, ri in exclude_rows}
        rm_mask |= pd.Series(
            [(sf, int(ri)) in rowset
             for sf, ri in zip(df["source_file"], df["row_idx"])],
            index=df.index)
    removed = df[rm_mask]
    n_fit = int(removed["fitness"].notna().sum())

    hits_out = hits.drop(columns=["vh_n", "vl_n"])
    hits_out.to_csv(PIPE_OUT / "deleak_v3_hits.csv", index=False)
    payload = {
        "ab_clusters": sorted(int(c) for c in cl),
        "rows": [[sf, int(ri)] for sf, ri in exclude_rows],
        "stats": {
            "benchmark_antibodies": len(bench),
            "benchmark_with_hit": n_cov,
            "exact_hit_rows": int(len(hits)),
            "clusters_removed": len(cl),
            "rows_removed_total": int(len(removed)),
            "rows_removed_with_fitness": n_fit,
            "corpus_rows_total": int(n_total),
            "cluster_expansion": not args.no_cluster_expansion,
            "per_source": removed["source_file"].value_counts().to_dict(),
        },
    }
    with open(PIPE_OUT / "deleak_v3_exclude.json", "w") as f:
        json.dump(payload, f, indent=1)

    s = payload["stats"]
    print(f"removal: {s['clusters_removed']} clusters + "
          f"{len(exclude_rows)} orphan rows -> {s['rows_removed_total']} rows "
          f"({s['rows_removed_with_fitness']} with fitness label) "
          f"of {s['corpus_rows_total']} total")
    print("per-source:", json.dumps(s["per_source"], indent=1))
    print(f"wrote {PIPE_OUT}/deleak_v3_hits.csv and deleak_v3_exclude.json")


if __name__ == "__main__":
    main()
