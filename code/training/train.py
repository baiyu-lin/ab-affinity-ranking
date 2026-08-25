"""Train one CV fold of the antigen-conditioned AffinityRanker.

Usage:
    .venv/bin/python training/train.py --config training/configs/lofo_example.json

Writes <output_dir>/results.jsonl (one line per epoch) and best.pt
(checkpoint at the best validation Spearman; early stopping on Spearman).
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (AntibodyCache, AntigenCache, ListSampler, StructCache,  # noqa: E402
                  StructTokenCache, attach_seq_ids, build_groups,
                  collate_antibodies, collate_struct_tokens, filter_cached,
                  load_group_tensors, load_main_corpus)
from evaluate import evaluate  # noqa: E402
from losses import aux_margin_loss, listmle_loss, make_target_order  # noqa: E402
from model import AffinityRanker  # noqa: E402

log = logging.getLogger("train")

# Leakage note from splits.json: li2023 affinity2 rows with cross_file_dup=True
# overlap affinity1; exclude them from training when affinity1 is in the train fold.
AFF1 = "3/li2023machine_scFv-SARS-CoV-2_affinity1.csv"
AFF2 = "3/li2023machine_scFv-SARS-CoV-2_affinity2.csv"

DEFAULT_CFG = {
    "model": {},
    "lr": 3e-4,
    "fusion_lr": 1e-3,
    "weight_decay": 0.01,
    "betas": [0.9, 0.999],
    "grad_clip": 1.0,
    "warmup_frac": 0.05,
    "lambda_aux": 0.0,
    "margin": 1.0,
    "aux_pairs_per_step": 8,
    "list_size": 20,
    "max_censored_frac": 0.25,
    "epochs": 10,
    "steps_per_epoch": 1500,
    "early_stop_patience": 3,
    "batch_tokens": 65536,
    "seed": 0,
    "num_threads": 8,
    "device": "cpu",
    "cv": {"strategy": "lofo", "holdout": ["11/kothiwal2025htp_DCC_spr.csv"]},
    "output_dir": "training/runs/run",
    "data": {
        "harmonized": "data_pipeline/output/harmonized_clustered.csv.gz",
        "training_groups": "data_pipeline/output/training_groups.csv",
        "aux_pairs": "data_pipeline/output/aux_pairs.csv",
        "aux_harmonized": "data_pipeline/output/harmonized_aux.csv.gz",
        "splits": "data_pipeline/output/splits.json",
        "ab_cache": "extraction/cache_antiberty",
        "ag_cache": "extraction/cache_esm2",
        "nll_index": "extraction/cache/index.csv",
        "struct_cache": None,  # e.g. "extraction/cache_struct_prostt5"
        "struct_token_cache": None,  # gated_pre: "extraction/cache_struct_saprot_tok"
    },
}


def deep_merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Fold resolution
# ---------------------------------------------------------------------------

def resolve_fold(cv, df, splits):
    """Return (train_mask, val_mask, fold_name) boolean Series aligned to df."""
    strat = cv["strategy"]
    if strat == "lofo":
        holdout = set(cv["holdout"])
        # small files sharing a global CDR-H3 cluster must be held out together
        flags = splits.get("leakage_flags_cross_file", [])
        changed = True
        while changed:
            changed = False
            for f in flags:
                a, b = f["file_a"], f["file_b"]
                if (a in holdout) != (b in holdout):
                    holdout.update((a, b))
                    changed = True
        val = df["source_file"].isin(holdout)
        extra = sorted(holdout - set(cv["holdout"]))
        if extra:
            log.info("lofo leakage union: +%d files held out together: %s",
                     len(extra), extra)
        name = "lofo__" + "+".join(sorted(cv["holdout"])) + \
            (f"[+{len(extra)}union]" if extra else "")
    elif strat == "abrank":
        k, n = int(cv["fold"]), int(cv.get("n_folds", 5))
        # splits.json recipe: int(ab_cluster_raw) % n where numeric else 0
        assign = (df["ab_cluster_raw"] % n).fillna(0)
        val = (df["source_file"] == "4/AbRank_dataset.csv") & (assign == k)
        name = f"abrank__fold{k}"
    elif strat == "agcluster":
        k, n = int(cv["fold"]), int(cv.get("n_folds", 5))
        cl = df["ag_cluster_raw"]
        val = cl.notna() & ((cl % n) == k)
        name = f"agcluster__fold{k}"
    elif strat == "all":
        # Final-model mode: train on ALL eligible rows. `val` is only a
        # monitor/early-stop signal — a deterministic sample of training
        # groups that REMAIN in the training set (no holdout).
        n_mon = int(cv.get("monitor_groups", 100))
        rng = np.random.default_rng(int(cv.get("seed", 0)) + 777)
        gids = np.array(df["group_id"].unique())
        mon = set(rng.choice(gids, size=min(n_mon, len(gids)),
                             replace=False).tolist())
        val = df["group_id"].isin(mon)
        name = f"all__mon{n_mon}"
    else:
        raise ValueError(f"unknown cv strategy: {strat}")
    if strat == "all":
        train = pd.Series(True, index=df.index)
    else:
        train = ~val
    if (df.loc[train, "source_file"] == AFF1).any():
        drop = (df["source_file"] == AFF2) & df["cross_file_dup"].fillna(False)
        n_drop = int((train & drop).sum())
        if n_drop:
            log.info("leakage guard: dropped %d affinity2 cross_file_dup rows", n_drop)
        train = train & ~drop
    return train, val, name


# ---------------------------------------------------------------------------
# Aux pairs
# ---------------------------------------------------------------------------

def apply_exclusions(df, data_cfg):
    """De-leakage hook (v3): drop rows listed in data_cfg['exclude_json'].

    The JSON is produced by data_pipeline/08_deleak_v3.py:
    {"ab_clusters": [...], "rows": [[source_file, row_idx], ...]}.
    The corpus file itself is never modified.
    """
    path = data_cfg.get("exclude_json")
    if not path:
        return df
    exc = json.load(open(path))
    mask = pd.Series(False, index=df.index)
    if exc.get("ab_clusters"):
        mask |= df["ab_cluster_raw"].isin(exc["ab_clusters"])
    if exc.get("rows"):
        rowset = {(sf, int(ri)) for sf, ri in exc["rows"]}
        mask |= pd.Series(
            [(sf, int(ri)) in rowset
             for sf, ri in zip(df["source_file"], df["row_idx"])],
            index=df.index)
    n = int(mask.sum())
    log.info("exclude_json %s: dropping %d/%d rows "
             "(%d clusters, %d orphan rows)", path, n, len(df),
             len(exc.get("ab_clusters", [])), len(exc.get("rows", [])))
    return df[~mask].reset_index(drop=True)

def load_aux_pairs(data_cfg, ab_cache, val_source_files):
    pairs = pd.read_csv(data_cfg["aux_pairs"])
    pairs = pairs[~pairs["source_file"].isin(val_source_files)]
    aux = pd.read_csv(data_cfg["aux_harmonized"], low_memory=False,
                      usecols=["source_file", "row_idx"])
    aux = attach_seq_ids(aux, ab_cache, "aux")
    key = aux.set_index(["source_file", "row_idx"])["seq_id"]
    pairs = pairs.join(key.rename("pos_seq_id"),
                       on=["source_file", "pos_row_idx"])
    pairs = pairs.join(key.rename("neg_seq_id"),
                       on=["source_file", "neg_row_idx"])
    ok = (pairs["pos_seq_id"].map(ab_cache.available)
          & pairs["neg_seq_id"].map(ab_cache.available))
    pairs = pairs[ok].reset_index(drop=True)
    log.info("aux pairs usable: %d", len(pairs))
    return pairs


def aux_step(model, pairs, ab_cache, ag_cache, n_pairs, margin, rng, device):
    """One aux margin-loss step: n_pairs from one random aux source file."""
    sf = pairs["source_file"].iloc[rng.integers(len(pairs))]
    sub = pairs[pairs["source_file"] == sf]
    rows = sub.iloc[rng.choice(len(sub), size=min(n_pairs, len(sub)),
                               replace=False)]
    seq_ids = list(rows["pos_seq_id"]) + list(rows["neg_seq_id"])
    h_list = [ab_cache.get(s) for s in seq_ids]
    keep = [i for i, h in enumerate(h_list) if h is not None]
    if len(keep) < 4:
        return None
    h_ab, ab_mask = collate_antibodies([h_list[i] for i in keep])
    h_ag, ag_mask = ag_cache.get(f"AUX::{sf}")
    scores = model(h_ab.to(device), ab_mask.to(device),
                   h_ag.unsqueeze(0).to(device), ag_mask.unsqueeze(0).to(device))
    n = len(rows)
    pos = [scores[j] for j, i in enumerate(keep) if i < n]
    neg = [scores[j] for j, i in enumerate(keep) if i >= n]
    m = min(len(pos), len(neg))
    if m == 0:
        return None
    return aux_margin_loss(torch.stack(pos[:m]), torch.stack(neg[:m]), margin)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_optimizer(model, cfg):
    if hasattr(model, "build_param_groups"):
        groups = model.build_param_groups(cfg["lr"], cfg["fusion_lr"],
                                          cfg["weight_decay"])
    else:
        decay = [p for p in model.parameters() if p.ndim >= 2]
        no_decay = [p for p in model.parameters() if p.ndim < 2]
        groups = [{"params": decay, "weight_decay": cfg["weight_decay"]},
                  {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg["lr"], betas=tuple(cfg["betas"]))


def make_scheduler(opt, total_steps, warmup_frac):
    warm = max(1, int(total_steps * warmup_frac))

    def fn(step):
        if step < warm:
            return step / warm
        t = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1.0 + np.cos(np.pi * min(t, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def train_model(model, cfg, sampler, aux_pairs, val_groups, ab_cache, ag_cache,
                fold_name, out_dir, device, struct_cache=None,
                struct_token_cache=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = make_optimizer(model, cfg)
    total_steps = cfg["epochs"] * cfg["steps_per_epoch"]
    sched = make_scheduler(opt, total_steps, cfg["warmup_frac"])
    rng = np.random.default_rng(cfg["seed"] + 1)
    results_path = out_dir / "results.jsonl"
    chain_aware = bool(cfg["model"].get("chain_aware", False))
    if chain_aware and cfg["lambda_aux"] > 0:
        raise ValueError("chain_aware arm does not support aux pairs "
                         "(no chain_ids path in aux_step); keep lambda_aux=0")
    use_struct = struct_cache is not None
    use_struct_tok = struct_token_cache is not None
    if (use_struct or use_struct_tok) and cfg["lambda_aux"] > 0:
        raise ValueError("struct arm does not support aux pairs "
                         "(no struct path in aux_step); keep lambda_aux=0")
    struct_dims = (getattr(model, "struct_ab_dim", 0),
                   getattr(model, "struct_ag_dim", 0))

    best_spearman, best_epoch, bad_epochs = -np.inf, -1, 0
    step = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        t0, losses = time.time(), []
        for _ in range(cfg["steps_per_epoch"]):
            g, idx = sampler.sample()
            struct_kw = {}
            if use_struct_tok:
                h_list, (h_ag, ag_mask), kept, s_tok = load_group_tensors(
                    g, ab_cache, ag_cache, idx,
                    struct_token_cache=struct_token_cache)
            elif use_struct:
                h_list, (h_ag, ag_mask), kept, (s_ab, s_ag) = \
                    load_group_tensors(g, ab_cache, ag_cache, idx,
                                       struct_cache=struct_cache,
                                       struct_dims=struct_dims)
                struct_kw = {"s_ab": s_ab.to(device), "s_ag": s_ag.to(device)}
            else:
                h_list, (h_ag, ag_mask), kept = load_group_tensors(
                    g, ab_cache, ag_cache, idx)
            if len(h_list) < 2:
                continue
            if use_struct_tok:
                struct_kw = {"s_ab": collate_struct_tokens(
                    s_tok, h_list, dim=struct_dims[0]).to(device)}
            if chain_aware:
                lens = [ab_cache.len_of(g.seq_ids[i]) for i in kept]
                if any(n is None for n in lens):
                    raise RuntimeError("chain_aware needs len_heavy in the "
                                       "antibody cache npz files")
                h_ab, ab_mask, chain_ids = collate_antibodies(h_list, lens)
                scores = model(h_ab.to(device), ab_mask.to(device),
                               h_ag.to(device), ag_mask.to(device),
                               chain_ids=chain_ids.to(device), **struct_kw)
            else:
                h_ab, ab_mask = collate_antibodies(h_list)
                scores = model(h_ab.to(device), ab_mask.to(device),
                               h_ag.to(device), ag_mask.to(device),
                               **struct_kw)
            order = make_target_order(g.fitness[kept], g.censored[kept], rng)
            loss = listmle_loss(scores, torch.from_numpy(order).to(device))
            if aux_pairs is not None and cfg["lambda_aux"] > 0:
                al = aux_step(model, aux_pairs, ab_cache, ag_cache,
                              cfg["aux_pairs_per_step"], cfg["margin"], rng,
                              device)
                if al is not None:
                    loss = loss + cfg["lambda_aux"] * al
            if not torch.isfinite(loss):
                log.warning("non-finite loss at step %d, skipping", step)
                opt.zero_grad(set_to_none=True)
                step += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            step += 1

        val_sp, n_val = evaluate(model, val_groups, ab_cache, ag_cache,
                                 cfg["batch_tokens"], device,
                                 struct_cache=struct_cache,
                                 struct_token_cache=struct_token_cache)
        rec = {"fold": fold_name, "epoch": epoch,
               "train_loss": float(np.mean(losses)) if losses else None,
               "val_spearman": val_sp, "n_val_groups": n_val,
               "lr": sched.get_last_lr()[0], "secs": round(time.time() - t0, 1)}
        with open(results_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        log.info("epoch %d: loss=%.4f val_spearman=%.4f (%d groups)",
                 epoch, rec["train_loss"] or float("nan"), val_sp, n_val)

        torch.save({"model_state": model.state_dict(), "config": cfg,
                    "epoch": epoch, "val_spearman": val_sp},
                   out_dir / "last.pt")
        if val_sp > best_spearman:
            best_spearman, best_epoch, bad_epochs = val_sp, epoch, 0
            torch.save({"model_state": model.state_dict(), "config": cfg,
                        "epoch": epoch, "val_spearman": val_sp},
                       out_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["early_stop_patience"]:
                log.info("early stop at epoch %d (best %.4f @ %d)",
                         epoch, best_spearman, best_epoch)
                break
    return {"fold": fold_name, "best_val_spearman": best_spearman,
            "best_epoch": best_epoch}


def prepare_data(cfg):
    """Load corpus, resolve the fold, build samplers and caches."""
    data_cfg = cfg["data"]
    splits = json.load(open(data_cfg["splits"]))
    ab_cache = AntibodyCache(data_cfg["ab_cache"])
    ag_cache = AntigenCache(data_cfg["ag_cache"])

    df = load_main_corpus(data_cfg)
    df = apply_exclusions(df, data_cfg)
    df = attach_seq_ids(df, ab_cache, "main")
    cv = {**cfg["cv"], "seed": cfg["seed"]}
    train_mask, val_mask, fold_name = resolve_fold(cv, df, splits)
    log.info("fold %s: %d train rows, %d val rows",
             fold_name, int(train_mask.sum()), int(val_mask.sum()))

    train_df = filter_cached(df[train_mask & df["eligible"]], ab_cache,
                             "train rows")
    val_df = filter_cached(df[val_mask], ab_cache, "val rows")

    train_groups = build_groups(train_df)
    val_groups = build_groups(val_df)
    sampler = ListSampler(train_groups, list_size=cfg["list_size"],
                          max_censored_frac=cfg["max_censored_frac"],
                          rank_only_factor=cfg.get("rank_only_factor", 0.5),
                          seed=cfg["seed"])
    aux_pairs = None
    if cfg["lambda_aux"] > 0:
        val_files = set(df.loc[val_mask, "source_file"].unique())
        aux_pairs = load_aux_pairs(data_cfg, ab_cache, val_files)
    struct_cache = None
    if data_cfg.get("struct_cache"):
        struct_cache = StructCache(data_cfg["struct_cache"])
        log.info("struct cache: %d ab ids, %d ag ids",
                 len(struct_cache.ab_path), len(struct_cache.ag_path))
    struct_token_cache = None
    if data_cfg.get("struct_token_cache"):
        struct_token_cache = StructTokenCache(data_cfg["struct_token_cache"])
        log.info("struct token cache: %d ab ids",
                 len(struct_token_cache.ab_path))
    return sampler, aux_pairs, val_groups, ab_cache, ag_cache, fold_name, \
        struct_cache, struct_token_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", default=None,
                    help="JSON object deep-merged over the config file")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s: %(message)s")
    cfg = deep_merge(DEFAULT_CFG, json.load(open(args.config)))
    if args.override:
        cfg = deep_merge(cfg, json.loads(args.override))

    torch.set_num_threads(cfg["num_threads"])
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config device is cuda but CUDA is unavailable")

    sampler, aux_pairs, val_groups, ab_cache, ag_cache, fold_name, \
        struct_cache, struct_token_cache = prepare_data(cfg)
    model = AffinityRanker(cfg["model"]).to(device)
    summary = train_model(model, cfg, sampler, aux_pairs, val_groups,
                          ab_cache, ag_cache, fold_name, cfg["output_dir"],
                          device, struct_cache=struct_cache,
                          struct_token_cache=struct_token_cache)
    if ab_cache.n_missing_files:
        log.warning("antibody npz files missing at read time: %d",
                    ab_cache.n_missing_files)
    if struct_cache is not None and struct_cache.n_missing:
        log.warning("struct npy files missing at read time: %d",
                    struct_cache.n_missing)
    if struct_token_cache is not None and struct_token_cache.n_missing:
        log.warning("struct token npy files missing at read time: %d",
                    struct_token_cache.n_missing)
    log.info("done: %s", summary)


if __name__ == "__main__":
    main()
