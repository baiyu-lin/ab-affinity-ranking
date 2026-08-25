"""Evaluation helpers: the benchmark metric is mean per-group Spearman."""

import numpy as np
import torch
from scipy.stats import spearmanr

from data import (collate_antibodies, collate_struct_tokens,
                  load_group_tensors, token_budget_slices)


def mean_group_spearman(fitness_by_group, scores_by_group):
    """Average Spearman over groups; groups with <3 rows or constant
    fitness are skipped; constant predictions score 0."""
    vals = []
    for gid, fit in fitness_by_group.items():
        sc = scores_by_group.get(gid)
        if sc is None or len(fit) < 3 or np.ptp(fit) == 0:
            continue
        r = spearmanr(fit, sc).statistic
        vals.append(0.0 if np.isnan(r) else float(r))
    return float(np.mean(vals)) if vals else float("nan"), len(vals)


@torch.no_grad()
def score_groups(model, groups, ab_cache, ag_cache, batch_tokens=65536,
                 device=None, struct_cache=None, struct_token_cache=None):
    """Score every row of every group. Returns {group_id: scores np.ndarray}.

    The antigen embedding is loaded once per group; rows are processed in
    token-budget batches. When the model has struct arms (struct_ab_dim /
    struct_ag_dim > 0) a struct_cache (pooled modes) or struct_token_cache
    (gated_pre) must be given; missing ids zero-fill.
    """
    model.eval()
    chain_aware = getattr(model, "chain_aware", False)
    use_struct = getattr(model, "struct_ab_dim", 0) > 0 or \
        getattr(model, "struct_ag_dim", 0) > 0
    if use_struct and struct_cache is None and struct_token_cache is None:
        raise ValueError("model has struct arms but no struct cache given")
    struct_dims = (getattr(model, "struct_ab_dim", 0),
                   getattr(model, "struct_ag_dim", 0))
    out = {}
    for g in groups:
        if struct_token_cache is not None:
            h_list, (h_ag, ag_mask), kept, s_tok = load_group_tensors(
                g, ab_cache, ag_cache, struct_token_cache=struct_token_cache)
        elif use_struct:
            h_list, (h_ag, ag_mask), kept, (s_ab, s_ag) = load_group_tensors(
                g, ab_cache, ag_cache, struct_cache=struct_cache,
                struct_dims=struct_dims)
        else:
            h_list, (h_ag, ag_mask), kept = load_group_tensors(
                g, ab_cache, ag_cache)
        if len(h_list) < 3:
            continue
        lengths = [h.shape[0] for h in h_list]
        lens = None
        if chain_aware:
            lens = [ab_cache.len_of(g.seq_ids[i]) for i in kept]
            if any(n is None for n in lens):
                raise RuntimeError("chain_aware needs len_heavy in the "
                                   "antibody cache npz files")
        scores = np.full(len(h_list), np.nan)
        for s, e in token_budget_slices(lengths, batch_tokens):
            kw = {}
            if struct_token_cache is not None:
                kw = {"s_ab": collate_struct_tokens(
                    s_tok[s:e], h_list[s:e], dim=struct_dims[0])}
                if device is not None:
                    kw = {"s_ab": kw["s_ab"].to(device)}
            elif use_struct:
                kw = {"s_ab": s_ab[s:e], "s_ag": s_ag}
                if device is not None:
                    kw = {"s_ab": kw["s_ab"].to(device),
                          "s_ag": kw["s_ag"].to(device)}
            if chain_aware:
                h_ab, ab_mask, chain_ids = collate_antibodies(h_list[s:e],
                                                              lens[s:e])
                if device is not None:
                    h_ab, ab_mask = h_ab.to(device), ab_mask.to(device)
                    h_ag, ag_mask = h_ag.to(device), ag_mask.to(device)
                    chain_ids = chain_ids.to(device)
                sc = model(h_ab, ab_mask, h_ag, ag_mask, chain_ids=chain_ids,
                           **kw)
            else:
                h_ab, ab_mask = collate_antibodies(h_list[s:e])
                if device is not None:
                    h_ab, ab_mask = h_ab.to(device), ab_mask.to(device)
                    h_ag, ag_mask = h_ag.to(device), ag_mask.to(device)
                sc = model(h_ab, ab_mask, h_ag, ag_mask, **kw)
            scores[s:e] = sc.float().cpu().numpy()
        out[g.group_id] = scores
    return out


def evaluate(model, groups, ab_cache, ag_cache, batch_tokens=65536,
             device=None, struct_cache=None, struct_token_cache=None):
    """Mean per-group Spearman of predicted scores vs fitness."""
    scores = score_groups(model, groups, ab_cache, ag_cache, batch_tokens,
                          device, struct_cache=struct_cache,
                          struct_token_cache=struct_token_cache)
    fitness = {g.group_id: g.fitness for g in groups if g.group_id in scores}
    return mean_group_spearman(fitness, scores)
