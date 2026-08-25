#!/usr/bin/env python
"""Chain-degradation probe (T3.6 trigger criterion).

Scores the v3 test set with the deleak_final ensemble under three input
variants — full antibody, VH-only, VL-only — by zeroing the attention mask of
the other chain (no re-extraction; h_ab is concat(VH, VL) with the boundary
at len(VH)). Reports per-group Spearman vs the public ground truth.

Interpretation for the mixed-encoder decision (plan T3.6):
  - full ~= VH-only and VL-only ~ 0   -> VH carries the signal, no encoder gap
  - both partial, full >> each        -> chains complementary (healthy)
  - VL informative but underused      -> AntiBERTy VL deficiency suspected;
                                        consider the B1/B2 mixed-tower arms

Run from the repo root:
  .venv/bin/python training/probe_chains.py \
      [--ckpt-dirs deleak_final_s0 deleak_final_s1 deleak_final_s2]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

from data import collate_antibodies  # noqa: E402
from infer_benchmark import (CACHE_DIR, clean_ab, clean_ag,  # noqa: E402
                             collate_antibodies_ag, extract_antibody,
                             extract_antigen)
from model import AffinityRanker  # noqa: E402

log = logging.getLogger("probe_chains")

TEST_CSV = ROOT / "data" / "benchmark_mAbs_v3.csv"
TRUTH = ROOT / "data" / "benchmark_v3_public_groundtruth.csv"


@torch.no_grad()
def score_variant(model, sub, device, variant):
    """Score one group with ab_mask restricted per variant."""
    h_abs = [torch.from_numpy(h) for h in sub.h_ab]
    lens = list(sub.len_h)
    h_ab, ab_mask = collate_antibodies(h_abs)
    for i, lh in enumerate(lens):
        L = h_abs[i].shape[0]
        if variant == "vh_only":
            ab_mask[i, lh:L] = False
        elif variant == "vl_only":
            ab_mask[i, :lh] = False
    h_ag, ag_mask = collate_antibodies_ag(list(sub.h_ag))
    sc = model(h_ab.to(device), ab_mask.to(device),
               h_ag.to(device), ag_mask.to(device))
    return sc.float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dirs", nargs="+",
                    default=[f"deleak_final_s{s}" for s in (0, 1, 2)])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s: %(message)s")
    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(TEST_CSV)
    df["VH_c"] = df.VH.map(clean_ab)
    df["VL_c"] = df.VL.map(clean_ab)
    df["ag_c"] = df.antigen.map(clean_ag)
    df["len_h"] = df.VH_c.str.len()
    # reuse the disk cache filled by infer_benchmark.py (encoders not needed
    # unless the cache is cold: then these raise a clear error)
    try:
        df["h_ab"] = [extract_antibody(None, vh, vl, CACHE_DIR)
                      for vh, vl in zip(df.VH_c, df.VL_c)]
        df["h_ag"] = df.ag_c.map(
            {s: extract_antigen(None, None, s, CACHE_DIR) for s in df.ag_c})
    except Exception as e:
        raise SystemExit("benchmark embedding cache cold; run "
                         "training/infer_benchmark.py first") from e

    truth = pd.read_csv(TRUTH)
    truth["neg_log_kd"] = -np.log10(truth["Kd_nM"])
    df = df.merge(truth[["group", "seq_id", "neg_log_kd"]],
                  on=["group", "seq_id"], how="left")
    if df["neg_log_kd"].isna().any():
        raise SystemExit("benchmark rows missing from the ground truth CSV")

    variants = ["full", "vh_only", "vl_only"]
    scores = {v: np.zeros((len(df), len(args.ckpt_dirs))) for v in variants}
    for j, d in enumerate(args.ckpt_dirs):
        ck = torch.load(ROOT / "training" / "runs" / d / "best.pt",
                        map_location="cpu", weights_only=False)
        model = AffinityRanker(ck["config"]["model"]).to(device)
        model.load_state_dict(ck["model_state"])
        model.eval()
        for gid, sub in df.groupby("group"):
            for v in variants:
                scores[v][sub.index, j] = score_variant(model, sub, device, v)
        log.info("scored %s", d)

    report = {"ckpts": args.ckpt_dirs, "variants": {}}
    for v in variants:
        S = scores[v]
        entry = {}
        for gid, sub in df.groupby("group"):
            idx = sub.index.to_numpy()
            y = sub["neg_log_kd"].to_numpy()
            per_seed = [float(spearmanr(S[idx, j], y).statistic)
                        for j in range(S.shape[1])]
            # Borda ensemble of the variant
            ranks = np.apply_along_axis(
                lambda c: np.argsort(np.argsort(-c)) + 1, 0, S[idx])
            ens = float(spearmanr(ranks.mean(axis=1), y).statistic)
            entry[str(gid)] = {"per_seed": per_seed, "ensemble": ens}
        entry["mean_ensemble"] = float(np.mean(
            [entry[g]["ensemble"] for g in entry]))
        report["variants"][v] = entry
    print(json.dumps(report, indent=1))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
