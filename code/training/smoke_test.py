"""Smoke test for the training harness.

Builds a tiny synthetic cache (50 seqs, 3 antigens, 2 groups) under
.scratch/train_smoke/, runs 20 training steps + one eval for each ablation
arm (cross/none x bilinear/concat, antigen_blind), and asserts losses are
finite and scores vary.

If the real antibody cache has enough sequences, additionally runs 50 steps
on one small real group (kothiwal2025htp_DCC) and prints the loss trajectory.

Usage: .venv/bin/python training/smoke_test.py
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (AntibodyCache, AntigenCache, ListSampler, StructCache,  # noqa: E402
                  StructTokenCache, Group, collate_antibodies,
                  collate_struct_tokens, load_group_tensors,
                  filter_cached, attach_seq_ids)
from evaluate import evaluate  # noqa: E402
from losses import listmle_loss, make_target_order  # noqa: E402
from model import AffinityRanker  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = ROOT / ".scratch" / "train_smoke"


def make_synthetic_cache(n_seqs=50, n_antigens=3, seed=0):
    rng = np.random.default_rng(seed)
    ab_dir = SCRATCH / "cache_ab"
    ag_dir = SCRATCH / "cache_ag"
    idx_rows, map_rows = [], []
    groups = ["g1.csv||A", "g1.csv||B"]
    for i in range(n_seqs):
        sid = hashlib.md5(f"seq{i}".encode()).hexdigest()[:16]
        Lh = int(rng.integers(60, 130))
        Ll = int(rng.integers(0, 110))
        h = rng.normal(size=(Lh + Ll, 512)).astype(np.float16)
        p = ab_dir / "seqs" / sid[:2]
        p.mkdir(parents=True, exist_ok=True)
        np.savez(p / f"{sid}.npz", h_ab=h, len_heavy=np.int16(Lh))
        idx_rows.append({"seq_id": sid, "path": f"seqs/{sid[:2]}/{sid}.npz",
                         "len_heavy": Lh, "len_light": Ll})
        src = groups[i % 2]
        map_rows.append({"corpus": "main", "source_file": src,
                         "row_idx": i // 2, "seq_id": sid})
    pd.DataFrame(idx_rows).to_csv(ab_dir / "index.csv", index=False)
    pd.DataFrame(map_rows).to_csv(ab_dir / "row_map.csv", index=False)

    ag_keys = groups + ["g1.csv||C"]  # group C gets the null antigen
    ag_rows = []
    for j, key in enumerate(ag_keys[:n_antigens]):
        aid = hashlib.md5(f"ag{j}".encode()).hexdigest()[:16]
        L = int(rng.integers(80, 200))
        h = rng.normal(size=(L, 1280)).astype(np.float16)
        p = ag_dir / "ags" / aid[:2]
        p.mkdir(parents=True, exist_ok=True)
        np.savez(p / f"{aid}.npz", h_ag=h)
        ag_rows.append({"key": key, "ag_id": aid})
    pd.DataFrame(ag_rows).to_csv(ag_dir / "antigen_index.csv", index=False)

    # struct arm: pooled [32] vectors per seq_id / ag_id
    st_dir = SCRATCH / "cache_struct"
    sab_rows, sag_rows = [], []
    for i in range(n_seqs):
        sid = hashlib.md5(f"seq{i}".encode()).hexdigest()[:16]
        p = st_dir / "ab" / sid[:2]
        p.mkdir(parents=True, exist_ok=True)
        np.save(p / f"{sid}.npy", rng.normal(size=32).astype(np.float32))
        sab_rows.append({"seq_id": sid, "path": f"{sid[:2]}/{sid}.npy"})
    for j in range(n_antigens):
        aid = hashlib.md5(f"ag{j}".encode()).hexdigest()[:16]
        p = st_dir / "ag" / aid[:2]
        p.mkdir(parents=True, exist_ok=True)
        np.save(p / f"{aid}.npy", rng.normal(size=32).astype(np.float32))
        sag_rows.append({"ag_id": aid, "path": f"{aid[:2]}/{aid}.npy"})
    pd.DataFrame(sab_rows).to_csv(st_dir / "ab" / "index.csv", index=False)
    pd.DataFrame(sag_rows).to_csv(st_dir / "ag" / "index.csv", index=False)

    # gated_pre arm: per-residue [L,32] fp16 tokens per seq_id (L = Lh+Ll)
    tok_dir = SCRATCH / "cache_struct_tok"
    tok_rows = []
    for i in range(n_seqs):
        sid = hashlib.md5(f"seq{i}".encode()).hexdigest()[:16]
        L = int(idx_rows[i]["len_heavy"] + idx_rows[i]["len_light"])
        p = tok_dir / "ab" / sid[:2]
        p.mkdir(parents=True, exist_ok=True)
        np.save(p / f"{sid}.npy", rng.normal(size=(L, 32)).astype(np.float16))
        tok_rows.append({"seq_id": sid, "path": f"{sid[:2]}/{sid}.npy"})
    pd.DataFrame(tok_rows).to_csv(tok_dir / "ab" / "index.csv", index=False)
    return ab_dir, ag_dir, st_dir, tok_dir


def make_groups(n_seqs=50, seed=1):
    rng = np.random.default_rng(seed)
    groups = []
    for gi, gid in enumerate(["g1.csv||A", "g1.csv||C"]):
        sid = [hashlib.md5(f"seq{gi + 2 * i}".encode()).hexdigest()[:16]
               for i in range(n_seqs // 2)]
        fit = rng.normal(size=len(sid))
        # force some ties and censored rows
        fit[:6] = fit[0]
        cens = np.zeros(len(sid), bool)
        cens[-3:] = True
        groups.append(Group(gid, gid, weight=10.0, rank_only=False,
                            seq_ids=sid, fitness=fit, censored=cens))
    return groups


def run_arm(name, model_cfg, sampler, groups, ab_cache, ag_cache, steps=20,
            struct_cache=None, struct_token_cache=None):
    torch.manual_seed(0)
    model = AffinityRanker(model_cfg)
    opt = torch.optim.AdamW(model.build_param_groups(), lr=3e-4,
                            betas=(0.9, 0.999))
    rng = np.random.default_rng(0)
    chain_aware = model_cfg.get("chain_aware", False)
    use_struct = struct_cache is not None
    use_struct_tok = struct_token_cache is not None

    def _load(g, idx):
        if use_struct_tok:
            h_list, (h_ag, ag_mask), kept, s_tok = load_group_tensors(
                g, ab_cache, ag_cache, idx,
                struct_token_cache=struct_token_cache)
            s_ab = collate_struct_tokens(
                s_tok, h_list, dim=model_cfg.get("struct_ab_dim", 0))
            return h_list, (h_ag, ag_mask), kept, {"s_ab": s_ab}
        if use_struct:
            h_list, (h_ag, ag_mask), kept, (s_ab, s_ag) = load_group_tensors(
                g, ab_cache, ag_cache, idx, struct_cache=struct_cache,
                struct_dims=(model_cfg.get("struct_ab_dim", 0),
                             model_cfg.get("struct_ag_dim", 0)))
            return h_list, (h_ag, ag_mask), kept, {"s_ab": s_ab, "s_ag": s_ag}
        h_list, (h_ag, ag_mask), kept = load_group_tensors(
            g, ab_cache, ag_cache, idx)
        return h_list, (h_ag, ag_mask), kept, {}

    def _collate(g, h_list, kept):
        if chain_aware:
            lens = [ab_cache.len_of(g.seq_ids[i]) for i in kept]
            return collate_antibodies(h_list, lens)
        return (*collate_antibodies(h_list), None)

    losses, first_scores = [], None
    for step in range(steps):
        g, idx = sampler.sample()
        h_list, (h_ag, ag_mask), kept, skw = _load(g, idx)
        h_ab, ab_mask, chain_ids = _collate(g, h_list, kept)
        scores = model(h_ab, ab_mask,
                       None if model_cfg.get("antigen_blind") else h_ag,
                       None if model_cfg.get("antigen_blind") else ag_mask,
                       chain_ids=chain_ids, **skw)
        assert torch.isfinite(scores).all(), f"{name}: non-finite scores"
        if first_scores is None:
            first_scores = scores.detach().clone()
        order = make_target_order(g.fitness[kept], g.censored[kept], rng)
        loss = listmle_loss(scores, torch.from_numpy(order))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
    assert np.isfinite(losses).all(), f"{name}: non-finite loss"
    assert np.std(losses) > 0 or abs(losses[0] - losses[-1]) > 1e-8, \
        f"{name}: loss did not move"
    sp, n_groups = evaluate(model, groups, ab_cache, ag_cache,
                            batch_tokens=8192, struct_cache=struct_cache,
                            struct_token_cache=struct_token_cache)
    assert np.isfinite(sp), f"{name}: eval spearman not finite"
    # scores must vary across rows after training
    g, idx = sampler.sample()
    h_list, (h_ag, ag_mask), kept, skw = _load(g, idx)
    h_ab, ab_mask, chain_ids = _collate(g, h_list, kept)
    model.eval()
    with torch.no_grad():
        sc = model(h_ab, ab_mask,
                   None if model_cfg.get("antigen_blind") else h_ag,
                   None if model_cfg.get("antigen_blind") else ag_mask,
                   chain_ids=chain_ids, **skw)
    assert sc.std() > 1e-6, f"{name}: constant scores"
    print(f"  [{name}] loss {losses[0]:.3f} -> {losses[-1]:.3f} "
          f"(min {min(losses):.3f}), eval spearman {sp:.3f} on {n_groups} groups")
    return losses


def test_target_order():
    rng = np.random.default_rng(0)
    fit = np.array([1.0, 2.0, 2.0, 3.0, 1.0])
    cens = np.array([False, False, False, False, True])
    order = make_target_order(fit, cens, rng)
    assert order[0] == 3, "best row first"
    assert order[-1] == 4, "censored pinned to bottom"
    orders = {tuple(make_target_order(fit, cens, rng)[:3]) for _ in range(50)}
    assert len(orders) > 1, "tie-jitter should re-break ties"
    s = torch.tensor([0.1, 0.5, 0.4, 0.9, 0.0])
    l = listmle_loss(s, torch.from_numpy(order))
    assert torch.isfinite(l)
    print("  [unit] target order: censored pinned, tie-jitter ok, "
          f"listmle={float(l):.3f}")


def _real_steps(g, ab_cache, ag_cache, steps, tag):
    torch.manual_seed(0)
    model = AffinityRanker({"interaction": "cross", "fusion": "bilinear"})
    opt = torch.optim.AdamW(model.build_param_groups(), lr=3e-4,
                            betas=(0.9, 0.999))
    rng = np.random.default_rng(0)
    sampler = ListSampler([g], list_size=20, seed=0)
    losses = []
    for step in range(steps):
        _, idx = sampler.sample()
        h_list, (h_ag, ag_mask), kept = load_group_tensors(
            g, ab_cache, ag_cache, idx)
        h_ab, ab_mask = collate_antibodies(h_list)
        scores = model(h_ab, ab_mask, h_ag, ag_mask)
        order = make_target_order(g.fitness[kept], g.censored[kept], rng)
        loss = listmle_loss(scores, torch.from_numpy(order))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
    assert np.isfinite(losses).all()
    sp, _ = evaluate(model, [g], ab_cache, ag_cache)
    print(f"  [{tag}] loss trajectory "
          f"(every 5 steps): {[round(l, 3) for l in losses[::5]]}")
    print(f"  [{tag}] group spearman after {steps} steps: {sp:.3f}")


def run_real(steps=50):
    ab_idx = ROOT / "extraction" / "cache_antiberty" / "index.csv"
    n_cached = sum(1 for _ in open(ab_idx)) - 1
    print(f"real antibody cache: {n_cached} seqs")
    if n_cached < 1000:
        print("  not enough real sequences cached; skipping real smoke run")
        return
    ab_cache = AntibodyCache(ROOT / "extraction" / "cache_antiberty")
    ag_cache = AntigenCache(ROOT / "extraction" / "cache_esm2")
    df = pd.read_csv(ROOT / "data_pipeline" / "output" /
                     "harmonized_clustered.csv.gz", low_memory=False)
    sub = df[df["source_file"] == "11/kothiwal2025htp_DCC_spr.csv"].copy()
    sub = attach_seq_ids(sub, ab_cache, "main")
    sub = filter_cached(sub, ab_cache, "kothiwal rows")
    print(f"  kothiwal2025htp_DCC: {len(sub)} cached rows")
    if len(sub) >= 10:
        g = Group(sub["group_id"].iloc[0], sub["group_id"].iloc[0], weight=1.0,
                  rank_only=False, seq_ids=list(sub["seq_id"]),
                  fitness=sub["fitness"].to_numpy(),
                  censored=sub["censored"].fillna(False).to_numpy(bool))
        _real_steps(g, ab_cache, ag_cache, steps, "real kothiwal_DCC")
        return
    # fallback: main-corpus extraction has not landed yet; exercise the real
    # npz path with cached aux VHH sequences (binary binder labels).
    print("  main corpus not cached yet; falling back to cached aux sequences")
    aux = pd.read_csv(ROOT / "data_pipeline" / "output" /
                      "harmonized_aux.csv.gz", low_memory=False)
    aux = aux[aux["source_file"] == "18/tsuruta2024avida-hIL6_binary.csv"]
    aux = attach_seq_ids(aux, ab_cache, "aux")
    aux = filter_cached(aux, ab_cache, "aux rows").head(60)
    if len(aux) < 10:
        print("  not enough cached aux rows either; skipping")
        return
    g = Group("aux::avida-hIL6", "AUX::18/tsuruta2024avida-hIL6_binary.csv",
              weight=1.0, rank_only=False, seq_ids=list(aux["seq_id"]),
              fitness=aux["label_raw"].to_numpy(),
              censored=np.zeros(len(aux), bool))
    _real_steps(g, ab_cache, ag_cache, steps, "real aux avida-hIL6")


def test_gated_pre_zero_init():
    """gated_pre starts exactly at the sequence-only baseline: with the gate
    zero-initialized, arbitrary s_ab must not change the scores."""
    torch.manual_seed(0)
    cfg = {"struct_ab_dim": 32, "struct_fusion": "gated_pre"}
    model = AffinityRanker(cfg)
    model.eval()
    h_ab = torch.randn(3, 40, 512)
    ab_mask = torch.ones(3, 40, dtype=torch.bool)
    h_ag = torch.randn(1, 50, 1280)
    ag_mask = torch.ones(1, 50, dtype=torch.bool)
    s_ab = torch.randn(3, 40, 32)
    with torch.no_grad():
        sc1 = model(h_ab, ab_mask, h_ag, ag_mask, s_ab=s_ab)
        sc2 = model(h_ab, ab_mask, h_ag, ag_mask,
                    s_ab=torch.zeros_like(s_ab))
    assert torch.equal(sc1, sc2), "gated_pre: zero-init gate leaks struct info"
    print("  [gated_pre_zero_init] scores identical for real/zero s_ab")


def main():
    torch.set_num_threads(4)
    print("building synthetic cache under", SCRATCH)
    ab_dir, ag_dir, st_dir, tok_dir = make_synthetic_cache()
    ab_cache = AntibodyCache(ab_dir)
    ag_cache = AntigenCache(ag_dir)
    struct_cache = StructCache(st_dir)
    struct_tok_cache = StructTokenCache(tok_dir)
    groups = make_groups()
    sampler = ListSampler(groups, list_size=20, seed=0)

    test_target_order()
    arms = [
        ("cross+bilinear", {"interaction": "cross", "fusion": "bilinear"}),
        ("none+bilinear", {"interaction": "none", "fusion": "bilinear"}),
        ("cross+concat", {"interaction": "cross", "fusion": "concat"}),
        ("none+concat", {"interaction": "none", "fusion": "concat"}),
        ("antigen_blind", {"antigen_blind": True}),
        ("chain_aware", {"chain_aware": True}),
    ]
    for name, cfg in arms:
        run_arm(name, cfg, sampler, groups, ab_cache, ag_cache, steps=20)
    # struct arm: pooled embeddings residual-fused onto both pools
    run_arm("struct", {"struct_ab_dim": 32, "struct_ag_dim": 32},
            sampler, groups, ab_cache, ag_cache, steps=20,
            struct_cache=struct_cache)
    # struct arm, cross-attn fusion: pool attends over k struct tokens
    run_arm("struct_xattn", {"struct_ab_dim": 32, "struct_ag_dim": 32,
                             "struct_fusion": "cross_attn", "struct_tokens": 4},
            sampler, groups, ab_cache, ag_cache, steps=20,
            struct_cache=struct_cache)
    # struct arm degrades to baseline when every id is missing
    empty_st = SCRATCH / "cache_struct_empty"
    (empty_st / "ab").mkdir(parents=True, exist_ok=True)
    (empty_st / "ag").mkdir(parents=True, exist_ok=True)
    run_arm("struct_missing", {"struct_ab_dim": 32, "struct_ag_dim": 32},
            sampler, groups, ab_cache, ag_cache, steps=20,
            struct_cache=StructCache(empty_st))
    # gated_pre: per-residue struct tokens fused before the antibody adapter
    test_gated_pre_zero_init()
    run_arm("struct_gated_pre", {"struct_ab_dim": 32,
                                 "struct_fusion": "gated_pre"},
            sampler, groups, ab_cache, ag_cache, steps=20,
            struct_token_cache=struct_tok_cache)
    empty_tok = SCRATCH / "cache_struct_tok_empty"
    (empty_tok / "ab").mkdir(parents=True, exist_ok=True)
    run_arm("struct_gated_pre_missing", {"struct_ab_dim": 32,
                                         "struct_fusion": "gated_pre"},
            sampler, groups, ab_cache, ag_cache, steps=20,
            struct_token_cache=StructTokenCache(empty_tok))
    print("synthetic smoke test: ALL ARMS PASSED")

    run_real(steps=50)


if __name__ == "__main__":
    main()
