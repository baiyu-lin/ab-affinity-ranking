"""Losses for the affinity-ranking trainer.

- ListMLE over per-group lists (Plackett-Luce NLL of the target permutation),
  with censored rows pinned to the bottom of the target order and tie-jitter
  (ties within equal-fitness blocks re-broken at random on every call).
- Pairwise margin (hinge) loss for the aux binder/non-binder pairs.
"""

import numpy as np
import torch


def make_target_order(fitness, censored, rng):
    """Target permutation: best -> worst.

    Non-censored rows sorted by fitness descending with random tie-breaks
    (tie-jitter); censored rows pinned to the bottom in random order.
    Returns an int64 np.ndarray of row indices.
    """
    fitness = np.asarray(fitness, dtype=np.float64)
    censored = np.asarray(censored, dtype=bool)
    n = len(fitness)
    order = np.lexsort((rng.random(n), -fitness))
    cens = censored[order]
    return np.concatenate([order[~cens], order[cens]]).astype(np.int64)


def listmle_loss(scores, target_order):
    """ListMLE = NLL of the target permutation under Plackett-Luce.

    scores: [N] predicted scores; target_order: indices best -> worst.
    Normalized per list position to keep the scale list-size independent.
    """
    s = scores[target_order]
    if s.numel() < 2:
        return scores.sum() * 0.0
    # cumulative logsumexp from the bottom up: lse_remaining[i] = lse(s[i:])
    lse_remaining = torch.flip(
        torch.logcumsumexp(torch.flip(s, [0]), dim=0), [0])
    return (lse_remaining - s).sum() / s.numel()


def aux_margin_loss(pos_scores, neg_scores, margin=1.0):
    """Hinge: positive binder must outscore the negative by `margin`."""
    return torch.clamp(margin - pos_scores + neg_scores, min=0.0).mean()
