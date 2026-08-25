"""A0 baselines (no main-model training).

(a) linear probe: mean-pooled [h_ab || h_ag] (512+1280) -> single Linear,
    trained with the same ListMLE protocol as the main model.
(b) zero-shot IgLM / ProGen2 NLL baselines from extraction/cache/index.csv:
    score = -nll, evaluated as mean per-group Spearman on the same folds.

Usage:
    .venv/bin/python training/probe.py --config cfg.json --mode linear
    .venv/bin/python training/probe.py --config cfg.json --mode zeroshot \
        --cv lofo --max-folds 10
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import AB_DIM, AG_DIM, AntibodyCache, AntigenCache, attach_seq_ids  # noqa: E402
from evaluate import mean_group_spearman  # noqa: E402
from train import (DEFAULT_CFG, deep_merge, load_main_corpus, prepare_data,  # noqa: E402
                   resolve_fold, set_seed, train_model)

log = logging.getLogger("probe")


class LinearProbe(nn.Module):
    """mean-pool(h_ab) || mean-pool(h_ag) -> Linear(1792, 1)."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(AB_DIM + AG_DIM, 1)
        n = sum(p.numel() for p in self.parameters())
        print(f"[LinearProbe] trainable params: {n:,}")

    @staticmethod
    def _mean_pool(h, mask):
        m = mask.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, h_ab, ab_mask, h_ag=None, ag_mask=None):
        fa = self._mean_pool(h_ab, ab_mask)
        if h_ag is None:
            fg = fa.new_zeros(fa.shape[0], AG_DIM)
        else:
            fg = self._mean_pool(h_ag, ag_mask)
            if fg.shape[0] == 1 and fa.shape[0] > 1:
                fg = fg.expand(fa.shape[0], -1)
        return self.lin(torch.cat([fa, fg], dim=-1)).squeeze(-1)


def run_linear(cfg):
    torch.set_num_threads(cfg["num_threads"])
    set_seed(cfg["seed"])
    device = torch.device(cfg.get("device", "cpu"))
    sampler, aux_pairs, val_groups, ab_cache, ag_cache, fold_name = \
        prepare_data(cfg)
    model = LinearProbe().to(device)
    summary = train_model(model, cfg, sampler, aux_pairs, val_groups,
                          ab_cache, ag_cache, fold_name, cfg["output_dir"],
                          device)
    log.info("done: %s", summary)


# ---------------------------------------------------------------------------
# Zero-shot NLL baselines
# ---------------------------------------------------------------------------

def zeroshot_scores(df, nll):
    """Return df with score_progen / score_iglm columns (-nll; higher=better)."""
    df = df.merge(nll[["seq_id", "nll_iglm_heavy", "nll_iglm_light",
                       "nll_progen"]], on="seq_id", how="left")
    df["score_progen"] = -df["nll_progen"]
    df["score_iglm"] = -df[["nll_iglm_heavy", "nll_iglm_light"]].mean(axis=1)
    return df


def eval_zeroshot_fold(val_df, fold_name):
    out = {"fold": fold_name}
    for col in ("score_iglm", "score_progen"):
        fit, sc = {}, {}
        for gid, sub in val_df.groupby("group_id", observed=True):
            s = sub.dropna(subset=[col])
            if len(s) >= 3:
                fit[gid] = s["fitness"].to_numpy()
                sc[gid] = s[col].to_numpy()
        sp, n = mean_group_spearman(fit, sc)
        out[col.replace("score_", "spearman_")] = sp
        out["n_groups"] = n
    return out


def run_zeroshot(cfg, cv_name, max_folds):
    torch.set_num_threads(cfg["num_threads"])
    data_cfg = cfg["data"]
    splits = json.load(open(data_cfg["splits"]))
    ab_cache = AntibodyCache(data_cfg["ab_cache"])
    nll = pd.read_csv(data_cfg["nll_index"])

    df = load_main_corpus(data_cfg)
    df = attach_seq_ids(df, ab_cache, "main")
    df = df[df["seq_id"].notna()].reset_index(drop=True)
    df = zeroshot_scores(df, nll)

    if cv_name == "lofo":
        folds = [{"strategy": "lofo", "holdout": [f["holdout"]]}
                 for f in splits["folds_lofo"]]
    elif cv_name == "abrank":
        n = splits["abrank_kfold"]["n_folds"]
        folds = [{"strategy": "abrank", "fold": k} for k in range(n)]
    elif cv_name == "agcluster":
        n = int(cfg["cv"].get("n_folds", 5))
        folds = [{"strategy": "agcluster", "fold": k, "n_folds": n}
                 for k in range(n)]
    else:
        raise ValueError(cv_name)
    if max_folds:
        folds = folds[:max_folds]

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"zeroshot_{cv_name}.jsonl"
    with open(path, "a") as fh:
        for cv in folds:
            _, val_mask, fold_name = resolve_fold(cv, df, splits)
            rec = eval_zeroshot_fold(df[val_mask], fold_name)
            fh.write(json.dumps(rec) + "\n")
            log.info("%s: iglm=%.4f progen=%.4f (%d groups)", rec["fold"],
                     rec["spearman_iglm"], rec["spearman_progen"],
                     rec["n_groups"])
    log.info("wrote %s", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["linear", "zeroshot"], required=True)
    ap.add_argument("--cv", choices=["lofo", "abrank", "agcluster"],
                    default="lofo", help="zeroshot only: which folds")
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--override", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s: %(message)s")
    cfg = deep_merge(DEFAULT_CFG, json.load(open(args.config)))
    if args.override:
        cfg = deep_merge(cfg, json.loads(args.override))

    if args.mode == "linear":
        run_linear(cfg)
    else:
        run_zeroshot(cfg, args.cv, args.max_folds)


if __name__ == "__main__":
    main()
