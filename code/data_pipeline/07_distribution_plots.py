#!/usr/bin/env python
"""Stage 7: distribution plots over the final harmonized corpus.

Reads harmonized.csv.gz + harmonized_aux.csv.gz + group_stats.csv,
writes PNGs to data_pipeline/output/plots/ (matplotlib Agg, no seaborn).
    .venv/bin/python data_pipeline/07_distribution_plots.py [--limit N]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path("data_pipeline/output")
PLOTS = OUT_DIR / "plots"
TRAINABLE_N = 20  # spec rule 2 threshold (annotated only, not enforced)

# files shown individually in the fitness panel (main assay files)
MAIN_FILES = [
    "3/li2023machine_scFv-SARS-CoV-2_affinity1.csv",
    "3/li2023machine_scFv-SARS-CoV-2_affinity2.csv",
    "6/engelhart2022dataset_scFv-SARS-CoV-2_affinity.csv",
    "4/AbRank_dataset.csv",
    "13/phillips2021binding_cr9114_h3_kd.csv",
    "5/adams2017measuring_4420-fluorescein_kd-titeseq.csv",
    "10/koenig2017mutational_kd_g6.csv",
    "9/kirby2024retrospective_ab-SARSCoV2_kd.csv",
]


def short(src):
    return Path(src).stem[:38]


def plot_group_sizes(groups, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    sizes = groups["n_rows"]
    bins = np.logspace(0, np.log10(sizes.max() + 1), 60)
    ax.hist(sizes, bins=bins, color="#4878cf")
    ax.set_xscale("log")
    ax.axvline(TRAINABLE_N, color="red", ls="--",
               label=f"trainable threshold N={TRAINABLE_N}")
    n_pass = int((sizes >= TRAINABLE_N).sum())
    ax.set_title(f"Group sizes (file × antigen): {len(sizes):,} groups, "
                 f"{n_pass:,} with N>={TRAINABLE_N}")
    ax.set_xlabel("rows per group (log)")
    ax.set_ylabel("groups")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "group_sizes.png", dpi=130)
    plt.close(fig)


def plot_fitness(df, out):
    files = [f for f in MAIN_FILES if f in set(df["source_file"])]
    n = len(files)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.1 * n), squeeze=False)
    for i, src in enumerate(files):
        sub = df[df["source_file"] == src]
        for j, col in enumerate(("fitness", "fitness_z")):
            ax = axes[i][j]
            vals = sub[col].dropna()
            ax.hist(vals, bins=60, color="#4878cf" if j == 0 else "#6acc65")
            if j == 0 and sub["censored"].any():
                ax.axvline(vals[sub["censored"]].median(), color="red", ls="--",
                           lw=1, label="censored mode")
                ax.legend(fontsize=7)
            ax.set_yticks([])
            if i == 0:
                ax.set_title("raw fitness" if j == 0 else "z-scored within group")
            if j == 0:
                ax.set_ylabel(short(src), fontsize=8)
    fig.suptitle("Fitness per file: raw vs per-group z-score", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "fitness_distributions.png", dpi=130)
    plt.close(fig)


def plot_lengths(df, aux, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"scFv": "#4878cf", "VHH": "#d65f5f"}
    both = pd.concat([df, aux], ignore_index=True)
    for ax, col, lo, hi in ((axes[0], "len_heavy", 90, 160),
                            (axes[1], "len_light", 90, 130)):
        for fmt, sub in both.groupby("format"):
            vals = sub[col].dropna()
            if not len(vals):
                continue
            ax.hist(vals, bins=80, histtype="step", lw=1.4, density=True,
                    color=colors.get(fmt, "#888888"),
                    label=f"{fmt} (n={len(vals):,})")
        ax.axvspan(lo, hi, color="green", alpha=0.06)
        ax.set_yscale("log")
        ax.set_title(f"{col} (accepted band {lo}-{hi})")
        ax.legend(fontsize=8)
    fig.suptitle("Sequence length distributions by format")
    fig.tight_layout()
    fig.savefig(out / "length_distributions.png", dpi=130)
    plt.close(fig)


def plot_censoring(groups, out):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    per_file = groups.groupby("source_file").agg(
        censored=("n_censored", "sum"), total=("n_rows", "sum"))
    per_file = per_file[per_file["censored"] > 0]
    if len(per_file):
        per_file["frac"] = per_file["censored"] / per_file["total"]
        per_file = per_file.sort_values("frac", ascending=False)
        ax.bar(range(len(per_file)), per_file["frac"], color="#d65f5f")
        ax.set_xticks(range(len(per_file)))
        ax.set_xticklabels([short(s) for s in per_file.index], rotation=45,
                           ha="right", fontsize=8)
        for i, (fr, tot) in enumerate(zip(per_file["frac"], per_file["total"])):
            ax.text(i, fr + 0.01, f"{fr:.3f}", ha="center", fontsize=8)
    ax.set_ylabel("censored fraction")
    ax.set_title("Censored rows per file (share of all rows in the file)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out / "censored_fraction.png", dpi=130)
    plt.close(fig)


def plot_composition(df, aux, norm_counts, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    counts = {"main (harmonized)": len(df), "aux_pairwise": len(aux)}
    counts.update(norm_counts)
    ax = axes[0]
    ax.bar(counts.keys(), counts.values(),
           color=["#4878cf", "#6acc65", "#b47cc7", "#c4c4c4"])
    ax.set_ylabel("rows")
    ax.set_title("Rows per corpus category")
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(counts.values()):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)

    per_file = pd.concat([df, aux]).groupby("source_file").size().sort_values()
    ax = axes[1]
    ax.barh(range(len(per_file)), per_file.values, color="#4878cf")
    ax.set_yticks(range(len(per_file)))
    ax.set_yticklabels([short(s) for s in per_file.index], fontsize=6)
    ax.set_xscale("log")
    ax.set_xlabel("rows (log)")
    ax.set_title("Rows per source file (main + aux, after cleanup)")
    fig.tight_layout()
    fig.savefig(out / "corpus_composition.png", dpi=130)
    plt.close(fig)


def plot_summary(df, aux, groups, norm_counts, out):
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3)
    # group sizes
    ax = fig.add_subplot(gs[0, 0])
    sizes = groups["n_rows"]
    ax.hist(sizes, bins=np.logspace(0, np.log10(sizes.max() + 1), 50),
            color="#4878cf")
    ax.set_xscale("log")
    ax.axvline(TRAINABLE_N, color="red", ls="--", lw=1)
    ax.set_title(f"group sizes ({len(sizes):,} groups)", fontsize=10)
    # corpus composition
    ax = fig.add_subplot(gs[0, 1])
    counts = {"main": len(df), "aux": len(aux), **norm_counts}
    ax.bar(counts.keys(), counts.values(),
           color=["#4878cf", "#6acc65", "#b47cc7", "#c4c4c4"])
    ax.set_title("rows per corpus", fontsize=10)
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    # censoring
    ax = fig.add_subplot(gs[0, 2])
    pf = groups.groupby("source_file").agg(
        censored=("n_censored", "sum"), total=("n_rows", "sum"))
    pf = pf[pf["censored"] > 0]
    if len(pf):
        frac = (pf["censored"] / pf["total"]).sort_values(ascending=False)
        ax.bar(range(len(frac)), frac.values, color="#d65f5f")
        ax.set_xticks(range(len(frac)))
        ax.set_xticklabels([short(s)[:18] for s in frac.index], rotation=45,
                           ha="right", fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_title("censored fraction", fontsize=10)
    # fitness raw vs z (main Kd files only, pooled per file median curves)
    big = ["4/AbRank_dataset.csv", "13/phillips2021binding_cr9114_h3_kd.csv",
           "5/adams2017measuring_4420-fluorescein_kd-titeseq.csv",
           "10/koenig2017mutational_kd_g6.csv"]
    for j, col in enumerate(("fitness", "fitness_z")):
        ax = fig.add_subplot(gs[1, j])
        for src in big:
            vals = df.loc[df["source_file"] == src, col].dropna()
            if len(vals):
                ax.hist(vals, bins=60, histtype="step", density=True, lw=1.2,
                        label=short(src)[:25])
        ax.set_title(f"{col} (Kd files)", fontsize=10)
        ax.legend(fontsize=6)
        ax.set_yticks([])
    # lengths
    ax = fig.add_subplot(gs[1, 2])
    both = pd.concat([df, aux], ignore_index=True)
    for fmt, sub in both.groupby("format"):
        ax.hist(sub["len_heavy"].dropna(), bins=80, histtype="step",
                density=True, lw=1.2, label=fmt)
    ax.set_yscale("log")
    ax.set_title("heavy length by format", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle("Data pipeline summary — harmonized corpus")
    fig.tight_layout()
    fig.savefig(out / "summary.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(OUT_DIR / "harmonized.csv.gz"))
    ap.add_argument("--input-aux", default=str(OUT_DIR / "harmonized_aux.csv.gz"))
    ap.add_argument("--group-stats", default=str(OUT_DIR / "group_stats.csv"))
    ap.add_argument("--normalized", default=str(OUT_DIR / "normalized.csv.gz"),
                    help="stage-1 output, used only for excluded-corpus counts")
    ap.add_argument("--outdir", default=str(PLOTS))
    ap.add_argument("--limit", type=int, default=None, help="rows to read (smoke)")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, nrows=args.limit)
    df["len_heavy"] = df["seq_heavy"].str.len()
    df["len_light"] = df["seq_light"].str.len()
    aux = pd.read_csv(args.input_aux, nrows=args.limit)
    aux["len_heavy"] = aux["seq_heavy"].str.len()
    aux["len_light"] = aux["seq_light"].str.len()
    groups = pd.read_csv(args.group_stats)
    print(f"[stage7] main={len(df):,} aux={len(aux):,} groups={len(groups):,}")

    # excluded corpus sizes from the stage-1 table (corpus column only)
    norm = pd.read_csv(args.normalized, usecols=["corpus"])
    vc = norm["corpus"].value_counts()
    norm_counts = {}
    if vc.get("excluded_ic50_ec50"):
        norm_counts["excluded_ic50_ec50"] = int(vc["excluded_ic50_ec50"])
    if vc.get("excluded_rank_only"):
        norm_counts["excluded_rank_only"] = int(vc["excluded_rank_only"])

    plot_group_sizes(groups, out)
    plot_fitness(df, out)
    plot_lengths(df, aux, out)
    plot_censoring(groups, out)
    plot_composition(df, aux, norm_counts, out)
    plot_summary(df, aux, groups, norm_counts, out)
    for p in sorted(out.glob("*.png")):
        print(f"  wrote {p}")


if __name__ == "__main__":
    sys.exit(main())
