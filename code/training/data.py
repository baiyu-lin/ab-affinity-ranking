"""Data access for the antigen-conditioned affinity-ranking trainer.

Cache readers (lazy per-sequence npz loading with an in-memory LRU), the
ListMLE group/list sampler, and padding/collate helpers.

All caches are read-only and may still be extracting: missing index rows or
missing npz files are tolerated (rows are skipped with a count warning).
"""

import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

log = logging.getLogger(__name__)

AB_DIM = 512   # AntiBERTy
AG_DIM = 1280  # ESM-2 650M


# ---------------------------------------------------------------------------
# Cache readers
# ---------------------------------------------------------------------------

class _Lru:
    """Tiny OrderedDict LRU mapping key -> torch.float32 tensor."""

    def __init__(self, capacity):
        self.capacity = capacity
        self._d = OrderedDict()

    def get(self, key):
        v = self._d.get(key)
        if v is not None:
            self._d.move_to_end(key)
        return v

    def put(self, key, value):
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.capacity:
            self._d.popitem(last=False)


class AntibodyCache:
    """Reader for extraction/cache_antiberty (h_ab fp16 [L,512] per seq_id)."""

    def __init__(self, root="extraction/cache_antiberty", lru_size=4096):
        self.root = Path(root)
        self._lru = _Lru(lru_size)
        self._lens = {}  # seq_id -> len_heavy (chain boundary, T3.5 chain-aware arm)
        idx_path = self.root / "index.csv"
        if idx_path.exists():
            idx = pd.read_csv(idx_path)
            self.seq_path = dict(zip(idx["seq_id"], idx["path"]))
        else:
            log.warning("antibody cache index missing: %s", idx_path)
            self.seq_path = {}
        rm_path = self.root / "row_map.csv"
        rm = pd.read_csv(rm_path, dtype={"corpus": "category",
                                         "source_file": "category"})
        self.row_map = rm  # merged directly by attach_seq_ids (no dict round-trip)
        self.n_missing_files = 0

    def available(self, seq_id):
        return seq_id in self.seq_path

    def get(self, seq_id):
        """Return h_ab as float32 [L, 512] tensor, or None if not yet cached."""
        t = self._lru.get(seq_id)
        if t is not None:
            return t
        rel = self.seq_path.get(seq_id)
        if rel is None:
            return None
        path = self.root / rel
        if not path.exists():
            self.n_missing_files += 1
            return None
        with np.load(path) as z:
            h = torch.from_numpy(z["h_ab"].astype(np.float32))
            if "len_heavy" in z:
                self._lens[seq_id] = int(z["len_heavy"])
        self._lru.put(seq_id, h)
        return h

    def len_of(self, seq_id):
        """Return len_heavy (VH residue count) for seq_id; None if unknown."""
        n = self._lens.get(seq_id)
        if n is None:
            rel = self.seq_path.get(seq_id)
            if rel is None:
                return None
            path = self.root / rel
            if not path.exists():
                self.n_missing_files += 1
                return None
            with np.load(path) as z:
                if "len_heavy" not in z:
                    return None
                n = int(z["len_heavy"])
            self._lens[seq_id] = n
        return n


class AntigenCache:
    """Reader for extraction/cache_esm2 (h_ag fp16 [L,1280] per ag_id).

    Keys: group_id for the main corpus, ``AUX::<source_file>`` for aux.
    Groups missing from the index are "null antigen": a fixed zero
    [1, 1280] pseudo-antigen with an all-true mask of length 1.
    """

    def __init__(self, root="extraction/cache_esm2", lru_size=1024):
        self.root = Path(root)
        self._lru = _Lru(lru_size)
        idx_path = self.root / "antigen_index.csv"
        if idx_path.exists():
            idx = pd.read_csv(idx_path)
            self.key2ag = dict(zip(idx["key"], idx["ag_id"]))
        else:
            log.warning("antigen cache index missing: %s "
                        "(all groups will use the null antigen)", idx_path)
            self.key2ag = {}
        self.n_null = 0

    def _ag_path(self, ag_id):
        return self.root / "ags" / ag_id[:2] / f"{ag_id}.npz"

    def get(self, key):
        """Return (h_ag float32 [L,1280], mask bool [L]); null antigen if absent."""
        t = self._lru.get(key)
        if t is not None:
            return t
        ag_id = self.key2ag.get(key)
        h = None
        if ag_id is not None:
            path = self._ag_path(ag_id)
            if path.exists():
                with np.load(path) as z:
                    h = torch.from_numpy(z["h_ag"].astype(np.float32))
        if h is None:
            self.n_null += 1
            h = torch.zeros(1, AG_DIM)
        mask = torch.ones(h.shape[0], dtype=torch.bool)
        out = (h, mask)
        self._lru.put(key, out)
        return out


class StructCache:
    """Reader for pooled structure-aware embeddings (fp32 [d] .npy per id).

    Layout (mirrors the other caches, sharded by 2-char prefix):
        root/ab/<pp>/<seq_id>.npy   + root/ab/index.csv  (seq_id,path)
        root/ag/<pp>/<ag_id>.npy    + root/ag/index.csv  (ag_id,path)

    Missing ids/files return None; callers zero-fill (degrades to baseline).
    """

    def __init__(self, root, lru_size=8192):
        self.root = Path(root)
        self._lru = _Lru(lru_size)
        self.n_missing = 0
        self.ab_path, self.ag_path = {}, {}
        idx = self.root / "ab" / "index.csv"
        if idx.exists():
            d = pd.read_csv(idx)
            self.ab_path = dict(zip(d["seq_id"], d["path"]))
        idx = self.root / "ag" / "index.csv"
        if idx.exists():
            d = pd.read_csv(idx)
            self.ag_path = dict(zip(d["ag_id"], d["path"]))

    def _get(self, table, base, key):
        ck = (base, key)
        t = self._lru.get(ck)
        if t is not None:
            return t
        rel = table.get(key)
        if rel is None:
            return None
        path = self.root / base / rel
        if not path.exists():
            self.n_missing += 1
            return None
        t = torch.from_numpy(np.load(path).astype(np.float32))
        self._lru.put(ck, t)
        return t

    def get_ab(self, seq_id):
        return self._get(self.ab_path, "ab", seq_id)

    def get_ag(self, ag_id):
        return self._get(self.ag_path, "ag", ag_id)


class StructTokenCache:
    """Reader for per-residue structure-aware embeddings (fp16 [L,d] .npy).

    Layout (antibody side only):
        root/ab/<pp>/<seq_id>.npy   + root/ab/index.csv  (seq_id,path)

    L must match the antibody h_ab length (same VH+VL residue order);
    callers treat length mismatches as missing and zero-fill (baseline
    contribution). fp16 tensors are kept in the LRU and upcast by the
    caller to keep resident memory down.
    """

    def __init__(self, root, lru_size=512):
        self.root = Path(root)
        self._lru = _Lru(lru_size)
        self.n_missing = 0
        self.ab_path = {}
        idx = self.root / "ab" / "index.csv"
        if idx.exists():
            d = pd.read_csv(idx)
            self.ab_path = dict(zip(d["seq_id"], d["path"]))
        else:
            log.warning("struct token cache index missing: %s", idx)

    def get_ab(self, seq_id):
        """Return fp16 [L,d] tensor, or None if not cached."""
        t = self._lru.get(seq_id)
        if t is not None:
            return t
        rel = self.ab_path.get(seq_id)
        if rel is None:
            return None
        path = self.root / "ab" / rel
        if not path.exists():
            self.n_missing += 1
            return None
        t = torch.from_numpy(np.load(path))  # fp16 [L,d]
        self._lru.put(seq_id, t)
        return t


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

CORPUS_COLS = ["source_file", "row_idx", "group_id", "fitness", "censored",
               "rank_only", "ab_cluster_raw", "ag_cluster_raw", "cross_file_dup"]


def load_main_corpus(data_cfg):
    """Load harmonized main corpus joined to cache seq_ids and group weights.

    Returns a DataFrame with the columns in CORPUS_COLS plus:
    seq_id (None if not yet cached), eligible, weight (from training_groups).
    Only the columns the harness actually uses are read (memory: the full
    table with sequence/antigen strings is several GB).
    """
    df = pd.read_csv(data_cfg["harmonized"], low_memory=False,
                     usecols=CORPUS_COLS,
                     dtype={"source_file": "category", "group_id": "category"})
    df = df[df["fitness"].notna()].reset_index(drop=True)
    groups = pd.read_csv(data_cfg["training_groups"])
    df = df.merge(
        groups[["group_id", "eligible", "weight", "rank_only"]],
        on="group_id", how="left", suffixes=("", "_g"))
    if "rank_only_g" in df.columns:
        df["rank_only"] = df["rank_only_g"].fillna(df["rank_only"])
        df = df.drop(columns=["rank_only_g"])
    df["eligible"] = df["eligible"].fillna(False).astype(bool)
    df["weight"] = df["weight"].fillna(0.0)
    return df


def attach_seq_ids(df, ab_cache, corpus):
    """Vectorized seq_id lookup; returns df with a 'seq_id' column."""
    rm = ab_cache.row_map
    rm = rm[rm["corpus"] == corpus][["source_file", "row_idx", "seq_id"]]
    return df.merge(rm, on=["source_file", "row_idx"], how="left")


def filter_cached(df, ab_cache, what="rows"):
    """Keep rows whose seq_id is in the index and whose npz exists on disk."""
    has_id = df["seq_id"].notna() & df["seq_id"].map(ab_cache.available)
    out = df[has_id]
    exists = out["seq_id"].map(
        lambda sid: (ab_cache.root / ab_cache.seq_path[sid]).exists())
    out = out[exists]
    n_drop = len(df) - len(out)
    if n_drop:
        log.warning("skipped %d/%d %s not yet in the antibody cache",
                    n_drop, len(df), what)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Groups and the ListMLE sampler
# ---------------------------------------------------------------------------

class Group:
    __slots__ = ("group_id", "key", "weight", "rank_only", "seq_ids",
                 "fitness", "censored")

    def __init__(self, group_id, key, weight, rank_only, seq_ids, fitness, censored):
        self.group_id = group_id
        self.key = key  # antigen cache key
        self.weight = float(weight)
        self.rank_only = bool(rank_only)
        self.seq_ids = seq_ids      # list[str]
        self.fitness = fitness      # np.float64 [n]
        self.censored = censored    # np.bool [n]


def build_groups(df, antigen_key_col="group_id", min_uncensored=2):
    """Build Group objects from a corpus slice (one Group per group_id)."""
    groups = []
    for gid, sub in df.groupby("group_id", sort=False, observed=True):
        fitness = sub["fitness"].to_numpy(np.float64)
        censored = sub["censored"].fillna(False).to_numpy(bool)
        if (~censored).sum() < min_uncensored:
            continue
        key = str(sub[antigen_key_col].iloc[0])
        groups.append(Group(
            group_id=gid, key=key,
            weight=float(sub["weight"].iloc[0]) if "weight" in sub else 1.0,
            rank_only=bool(sub["rank_only"].iloc[0]) if "rank_only" in sub else False,
            seq_ids=list(sub["seq_id"]), fitness=fitness, censored=censored))
    return groups


class ListSampler:
    """Implements the ListMLE list-sampling protocol.

    Per step: sample a group with probability ∝ weight (rank_only groups at
    0.5x), then sample up to ``list_size`` rows with at most
    ``max_censored_frac`` censored rows per sampled list.
    """

    def __init__(self, groups, list_size=20, max_censored_frac=0.25,
                 rank_only_factor=0.5, seed=0):
        self.groups = [g for g in groups if g.weight > 0]
        if not self.groups:
            raise ValueError("no samplable groups (all weights zero)")
        w = np.array([g.weight * (rank_only_factor if g.rank_only else 1.0)
                      for g in self.groups], dtype=np.float64)
        self.probs = w / w.sum()
        self.list_size = list_size
        self.max_censored_frac = max_censored_frac
        self.rng = np.random.default_rng(seed)

    def sample(self):
        """Return (Group, row_indices np.ndarray)."""
        g = self.groups[self.rng.choice(len(self.groups), p=self.probs)]
        n = len(g.seq_ids)
        if n <= self.list_size:
            idx = np.arange(n)
            # still enforce the censored cap by dropping excess censored rows
            cap = max(1, int(self.list_size * self.max_censored_frac))
            cens_idx = idx[g.censored]
            if len(cens_idx) > cap:
                keep_cens = self.rng.choice(cens_idx, size=cap, replace=False)
                idx = np.concatenate([idx[~g.censored], keep_cens])
        else:
            cens_pool = np.flatnonzero(g.censored)
            unc_pool = np.flatnonzero(~g.censored)
            n_cens = min(len(cens_pool),
                         int(self.list_size * self.max_censored_frac))
            n_unc = min(len(unc_pool), self.list_size - n_cens)
            parts = []
            if n_unc:
                parts.append(self.rng.choice(unc_pool, size=n_unc, replace=False))
            if n_cens:
                parts.append(self.rng.choice(cens_pool, size=n_cens, replace=False))
            idx = np.concatenate(parts)
        return g, idx


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_antibodies(h_list, chain_lens=None):
    """Pad a list of [L_i, 512] tensors -> (h_ab [B,Lmax,512], mask [B,Lmax]).

    If chain_lens (len_heavy per tensor) is given, also return chain_ids
    [B,Lmax] (0 = VH position, 1 = VL position; padded slots stay 0 and are
    masked out anyway). VHH rows have len_heavy == L -> all-zero chain_ids.
    """
    b = len(h_list)
    lmax = max(h.shape[0] for h in h_list)
    h_ab = torch.zeros(b, lmax, AB_DIM)
    mask = torch.zeros(b, lmax, dtype=torch.bool)
    chain_ids = torch.zeros(b, lmax, dtype=torch.long) if chain_lens is not None else None
    for i, h in enumerate(h_list):
        h_ab[i, : h.shape[0]] = h
        mask[i, : h.shape[0]] = True
        if chain_ids is not None:
            n_h = min(int(chain_lens[i]), h.shape[0])
            chain_ids[i, n_h: h.shape[0]] = 1
    if chain_ids is not None:
        return h_ab, mask, chain_ids
    return h_ab, mask


def collate_struct_tokens(t_list, h_list, dim=0):
    """Pad per-residue struct tensors to float32 [B,Lmax,dim].

    t_list entries are fp16 [L_i,dim] tensors or None (missing); h_list
    carries the matching h_ab tensors and defines L_i/Lmax. Missing and
    padded slots stay zero (baseline contribution through the bias-free
    gated_pre projection).
    """
    if not dim:
        dim = next((t.shape[1] for t in t_list if t is not None), 1)
    b = len(h_list)
    lmax = max(h.shape[0] for h in h_list)
    out = torch.zeros(b, lmax, dim)
    for i, t in enumerate(t_list):
        if t is not None:
            out[i, : t.shape[0]] = t.float()
    return out


def token_budget_slices(lengths, batch_tokens):
    """Greedy contiguous slices such that (slice_len * max_L_in_slice) <= budget."""
    slices, start, lmax = [], 0, 0
    for i, L in enumerate(lengths):
        new_max = max(lmax, L)
        if i > start and (i - start + 1) * new_max > batch_tokens:
            slices.append((start, i))
            start, lmax = i, L
        else:
            lmax = new_max
    slices.append((start, len(lengths)))
    return slices


def load_group_tensors(group, ab_cache, ag_cache, row_idx=None,
                       struct_cache=None, struct_dims=None,
                       struct_token_cache=None):
    """Load embeddings for (a subset of) a group.

    Returns (h_list, (h_ag [1,L,1280], ag_mask [1,L]), kept) where h_list is
    the list of antibody tensors and kept the surviving row indices. Rows
    whose npz vanished are skipped.

    If struct_cache is given, a fourth element (s_ab, s_ag) is returned:
    s_ab float32 [len(kept), d_ab] pooled struct embeddings and s_ag float32
    [1, d_ag] for the group's antigen, zero-filled where the id is missing.
    struct_dims=(d_ab, d_ag) gives the expected widths (from the model
    config); when omitted they are probed from the first cached vector.

    If struct_token_cache is given (gated_pre arm), the fourth element is
    instead a list of fp16 [L_i, d] per-residue tensors aligned with h_list,
    with None where the id is missing or its length disagrees with h_ab
    (collate_struct_tokens zero-fills those slots).
    """
    idx = np.arange(len(group.seq_ids)) if row_idx is None else np.asarray(row_idx)
    h_list, kept = [], []
    for i in idx:
        h = ab_cache.get(group.seq_ids[i])
        if h is None:
            continue
        h_list.append(h)
        kept.append(i)
    h_ag, ag_mask = ag_cache.get(group.key)
    kept = np.array(kept, dtype=int)
    if struct_token_cache is not None:
        s_tok = []
        for j, i in enumerate(kept):
            t = struct_token_cache.get_ab(group.seq_ids[i])
            if t is None or t.shape[0] != h_list[j].shape[0]:
                s_tok.append(None)
            else:
                s_tok.append(t)
        return h_list, (h_ag.unsqueeze(0), ag_mask.unsqueeze(0)), kept, s_tok
    if struct_cache is None:
        return h_list, (h_ag.unsqueeze(0), ag_mask.unsqueeze(0)), kept
    d_ab = struct_dims[0] if struct_dims else 0
    d_ag = struct_dims[1] if struct_dims else d_ab
    s_list = [struct_cache.get_ab(group.seq_ids[i]) for i in kept]
    if not d_ab:
        d_ab = next((t.shape[0] for t in s_list if t is not None), 1)
    s_ab = torch.zeros(len(kept), d_ab)
    for j, t in enumerate(s_list):
        if t is not None and t.shape[0] == d_ab:
            s_ab[j] = t
    ag_id = ag_cache.key2ag.get(group.key)
    s_ag = struct_cache.get_ag(ag_id) if ag_id is not None else None
    if s_ag is not None and (not d_ag or s_ag.shape[0] != d_ag):
        if not d_ag:
            d_ag = s_ag.shape[0]
        else:
            s_ag = None
    if not d_ag:
        d_ag = d_ab
    s_ag = s_ag.unsqueeze(0) if s_ag is not None else torch.zeros(1, d_ag)
    return h_list, (h_ag.unsqueeze(0), ag_mask.unsqueeze(0)), kept, (s_ab, s_ag)
