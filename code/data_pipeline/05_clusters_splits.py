#!/usr/bin/env python
"""Stage 5: identity clustering + leakage flags + CV split definitions.

Implements spec rule 7 following the literature (research §4.4): random
splits inflate affinity-prediction performance ~3-4x; the standard antibody
cutoff is 90% length-matched CDR identity (Graphinity), AbRank uses 75%
global identity on concatenated chains.

Cluster assignment, in priority order:
1. AbRank rows carry the paper's own 75%-identity Levenshtein cluster ids
   (ab_cluster_raw) — used directly.
2. All other main-corpus files: CDR-H3 is extracted with an FW3/FW4 anchor
   regex (last Cys before WGxG; no numbering tool in the venv, documented
   heuristic), then clustered per file at 90% length-matched identity
   (Hamming within CDR-H3-length bins, greedy, vectorized). Files with
   > MAX_UNIQUE unique CDR-H3s are clustered on a subsample and the rest
   become singletons (recorded in the manifest).
3. li2023/engelhart-scale files are single-antigen screening sets handled
   by leave-one-file-out CV; they get a file-level pseudo cluster.
   The li2023 affinity1/affinity2 overlap is flagged by exact (heavy,
   light) cross-file match -> cross_file_dup=True on the affinity2 rows.

Outputs:
  harmonized_clustered.csv.gz  = harmonized + cluster_id + cross_file_dup
  splits.json                  = leave-one-file-out folds + AbRank
                                 cluster-stratified 5-fold + leakage flags
  manifest_stage5.json

Run from repo root:
    .venv/bin/python data_pipeline/05_clusters_splits.py [--limit N]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("data_pipeline/output")
ABRANK_SRC = "4/AbRank_dataset.csv"
LI2023_A1 = "3/li2023machine_scFv-SARS-CoV-2_affinity1.csv"
LI2023_A2 = "3/li2023machine_scFv-SARS-CoV-2_affinity2.csv"
CLUSTER_MAX_ROWS = 60_000    # larger files -> file-level pseudo cluster
MAX_UNIQUE = 300_000         # global cap on CDR-H3 clustering input
IDENTITY = 0.90

# CDR-H3: residues between the last conserved Cys (IMGT 104) and the WGxG
# of FW4 (IMGT 118). Heuristic, no numbering tool available.
RE_H3 = re.compile(r"C\w\w(\w{3,30})WG\wG")


def extract_h3(seq):
    if not isinstance(seq, str):
        return None
    m = None
    for m in RE_H3.finditer(seq):  # last match = Cys closest to WGxG
        pass
    return m.group(1) if m else None


def cluster_bin(seqs, identity=IDENTITY):
    """Greedy Hamming clustering of equal-length sequences. Returns labels."""
    n = len(seqs)
    labels = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return labels
    arr = np.frombuffer("".join(seqs).encode(), dtype=np.uint8).reshape(n, -1)
    reps = []
    for i in range(n):
        if labels[i] != -1:
            continue
        labels[i] = len(reps)
        reps.append(i)
        unl = labels == -1
        if unl.any():
            idx = np.where(unl)[0]
            match = ((arr[idx] != arr[i]).mean(axis=1) <= 1 - identity)
            labels[idx[match]] = labels[i]
    return labels


def cluster_file(h3_series, entry):
    """90% length-matched clustering of CDR-H3 sequences (global call)."""
    h3 = h3_series.fillna("")
    uniq = h3.unique().tolist()
    entry["n_unique_h3"] = int(len(uniq) - ("" in uniq))
    entry["h3_extract_fail"] = int((h3 == "").sum())
    if len(uniq) > MAX_UNIQUE:
        rng = np.random.default_rng(0)
        keep = set(rng.choice(uniq, size=MAX_UNIQUE, replace=False).tolist())
        keep.add("")
        entry["h3_cluster_subsampled"] = True
    else:
        keep = set(uniq)
        entry["h3_cluster_subsampled"] = False
    mapping = {"": "noh3"}
    n_clusters = 0
    by_len = {}
    for s in uniq:
        if s in keep:
            by_len.setdefault(len(s), []).append(s)
    for L, seqs in sorted(by_len.items()):
        if L == 0:
            continue
        labs = cluster_bin(seqs)
        for s, lab in zip(seqs, labs):
            mapping[s] = f"L{L}c{lab + n_clusters}"
        n_clusters += int(labs.max()) + 1
    entry["n_clusters"] = int(n_clusters)
    out = h3.map(lambda s: mapping.get(s, "singleton"))
    n_sg = int((out == "singleton").sum())
    if n_sg:  # unclustered overflow must not collapse into one fake cluster
        out = out.copy()
        out[out == "singleton"] = [f"sg{i}" for i in range(n_sg)]
        entry["n_singletons"] = n_sg
    return out


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
    ap.add_argument("--input", default=str(OUT_DIR / "harmonized.csv.gz"))
    ap.add_argument("--output", default=str(OUT_DIR / "harmonized_clustered.csv.gz"))
    ap.add_argument("--splits", default=str(OUT_DIR / "splits.json"))
    ap.add_argument("--manifest", default=str(OUT_DIR / "manifest_stage5.json"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    # Large single-antigen screening files are clustered trivially (leave-
    # one-file-out CV handles them); AbRank brings its own 75% clusters.
    # All remaining (small) main-corpus files are buffered and clustered
    # GLOBALLY, so parents shared across files/antigens (hie2023 families,
    # phillips cr9114 vs cr6261, …) land in one cluster and become visible
    # as cross-file leakage flags.
    li_a1_keys = None
    manifest, first, total = {"files": {}}, True, 0
    small = []  # buffered (src, df) for global clustering
    for src, sub in iter_file_groups(args.input):
        if args.limit:
            sub = sub.head(args.limit)
        entry = {"n_in": int(len(sub))}

        if src == ABRANK_SRC:
            sub["cluster_id"] = "abr_" + sub["ab_cluster_raw"].fillna("none").astype(str)
            entry["n_clusters"] = int(sub["cluster_id"].nunique())
            write_now = True
        elif len(sub) > CLUSTER_MAX_ROWS:
            sub["cluster_id"] = "file_" + src.split("/")[0]
            entry["n_clusters"] = 1
            entry["note"] = "large single-antigen screening file; file-level pseudo cluster"
            write_now = True
        else:
            small.append((src, sub, entry))
            write_now = False

        sub["cross_file_dup"] = False
        if src == LI2023_A1:
            li_a1_keys = set(zip(sub["seq_heavy"], sub["seq_light"].fillna("")))
        elif src == LI2023_A2 and li_a1_keys is not None:
            keys = list(zip(sub["seq_heavy"], sub["seq_light"].fillna("")))
            dup = pd.Series([k in li_a1_keys for k in keys], index=sub.index)
            sub["cross_file_dup"] = dup
            entry["n_cross_file_dup"] = int(dup.sum())

        if write_now:
            manifest["files"][src] = entry
            sub.to_csv(args.output, mode="w" if first else "a", header=first,
                       index=False, compression="gzip")
            first = False
            total += len(sub)
            print(f"  {src}: {entry['n_in']:>8,} rows clusters={entry.get('n_clusters')}"
                  + (f" cross_dup={entry['n_cross_file_dup']:,}"
                     if "n_cross_file_dup" in entry else ""))

    # global clustering of the buffered small files
    if small:
        pooled_h3 = pd.concat([sub["seq_heavy"].map(extract_h3)
                               for _, sub, _ in small])
        glob_entry = {}
        mapping = cluster_file(pooled_h3, glob_entry)
        manifest["global_small_file_clustering"] = glob_entry
        print(f"  [global] {len(small)} small files: "
              f"{glob_entry['n_unique_h3']:,} unique CDR-H3 -> "
              f"{glob_entry['n_clusters']:,} clusters "
              f"(extract_fail={glob_entry['h3_extract_fail']:,})"
              + (" SUBSAMPLED" if glob_entry["h3_cluster_subsampled"] else ""))
        pos = 0
        for src, sub, entry in small:
            n = len(sub)
            sub["cluster_id"] = "g_" + mapping.iloc[pos:pos + n].to_numpy()
            pos += n
            entry["n_clusters"] = int(sub["cluster_id"].nunique())
            sub["cross_file_dup"] = False
            manifest["files"][src] = entry
            sub.to_csv(args.output, mode="w" if first else "a", header=first,
                       index=False, compression="gzip")
            first = False
            total += len(sub)
            print(f"  {src}: {entry['n_in']:>8,} rows clusters={entry['n_clusters']}")

    # leakage flags: file pairs sharing at least one global cluster id
    flags = []
    file_clusters = {}
    for src, sub, _ in small:
        file_clusters[src] = set(sub["cluster_id"].unique())
    srcs = sorted(file_clusters)
    for i in range(len(srcs)):
        for j in range(i + 1, len(srcs)):
            shared = file_clusters[srcs[i]] & file_clusters[srcs[j]]
            shared.discard("g_noh3")
            if shared:
                flags.append({"file_a": srcs[i], "file_b": srcs[j],
                              "n_shared_clusters": len(shared)})
    if flags:
        print(f"  [leakage] {len(flags)} small-file pairs share CDR-H3 clusters")

    splits = {
        "cv_strategy": "leave-one-file-out (all antigen groups of a file "
                       "together); AbRank additionally gets cluster-stratified "
                       "5-fold by ab_cluster_raw hash; small files sharing a "
                       "global CDR-H3 cluster (leakage_flags) must be held out "
                       "together",
        "folds_lofo": [{"holdout": s} for s in
                       sorted(manifest["files"].keys())],
        "abrank_kfold": {"n_folds": 5,
                         "assign": "int(ab_cluster_raw) % 5 where numeric else 0"},
        "leakage_flags_cross_file": flags,
        "notes": [
            "li2023 affinity2 rows with cross_file_dup=True overlap affinity1; "
            "exclude them from training when affinity1 is in the train fold.",
            "AbRank test-time decontamination (paper): exclude Abs >75% "
            "identical to SAbDab; our Lev3 cluster ids encode the same "
            "clustering and can be reused for cluster-held-out CV.",
        ],
    }
    with open(args.splits, "w") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)
    manifest["n_rows_total"] = total
    manifest["n_cross_file_leak_flags"] = len(flags)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[stage5] wrote {total:,} rows -> {args.output}")
    print(f"[stage5] splits -> {args.splits}")


if __name__ == "__main__":
    sys.exit(main())
