#!/usr/bin/env python
"""Milestone 4/5: frozen-encoder embedding extraction for the ranking model.

Extracts, for every unique (heavy, light) sequence pair in the eligible
training groups + aux pairs:

  - ProGen2-small last-layer hidden states  [L, 1024]  (fp16, linker/sentinels stripped)
  - IgLM last-layer hidden states           [L,  512]  (fp16, per-chain conditioned, specials stripped)
  - mean NLL per stream (the log-ppl bypass features)

Cache layout (resumable — existing .npz files are skipped):

  <cache-dir>/manifest.csv      seq_id, seq_heavy, seq_light, format, in_main, in_aux
  <cache-dir>/row_map.csv       corpus, source_file, row_idx, seq_id
  <cache-dir>/seqs/<xx>/<seq_id>.npz   h_progen, h_iglm, len_heavy, nll_* scalars

Conventions (verified against the reference implementations):
  - ProGen2: input '1' + VH + (GGGGS)3 + VL + '2' (salesforce sentinel tokens);
    NLL over AA labels only, logits restricted to vocab ids 5..29 as in
    progen2/likelihood.py. Linker/sentinel positions dropped from hidden states
    and excluded from the NLL.
  - IgLM: per chain '[HEAVY]'/'[LIGHT]' + '[HUMAN]' + seq + '[SEP]' via
    BertTokenizerFast.convert_tokens_to_ids (iglm==0.1.0 IgLM.log_likelihood);
    NLL = mean CE of labels[2:] (includes the [SEP] target, as upstream).
Run from the repo root with the project venv.
"""

import argparse
import hashlib
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, GPT2LMHeadModel

LINKER = "GGGGS" * 3
PROGEN_DIR = "model_data/progen2-small"
IGLM_DIR = "model_data/iglm_weights/IgLM"
IGLM_VOCAB = "model_data/iglm_weights/vocab.txt"
PIPE_OUT = "data_pipeline/output"

# ProGen2 sentinel token ids ('1' -> 3, '2' -> 4); AA vocab ids 5..29 (salesforce convention)
PROGEN_BOS, PROGEN_EOS = 3, 4
PROGEN_AA_LO, PROGEN_AA_HI = 5, 29


# ---------------------------------------------------------------- manifest

def seq_id_of(heavy: str, light: str) -> str:
    return hashlib.sha1((heavy + "|" + light).encode()).hexdigest()[:16]


def build_manifest(cache_dir: str) -> pd.DataFrame:
    tg = pd.read_csv(os.path.join(PIPE_OUT, "training_groups.csv"))
    eligible = set(tg.loc[tg.eligible == True, "group_id"])  # noqa: E712

    main = pd.read_csv(
        os.path.join(PIPE_OUT, "harmonized_clustered.csv.gz"),
        usecols=["source_file", "row_idx", "group_id", "seq_heavy", "seq_light", "format"],
    )
    main = main[main.group_id.isin(eligible)].copy()
    main["corpus"] = "main"

    aux_pairs = pd.read_csv(os.path.join(PIPE_OUT, "aux_pairs.csv"))
    aux_all = pd.read_csv(os.path.join(PIPE_OUT, "harmonized_aux.csv.gz"), low_memory=False)
    aux_all = aux_all[["source_file", "row_idx", "seq_heavy", "seq_light", "format"]]
    keys = pd.concat([
        aux_pairs[["source_file", "pos_row_idx"]].rename(columns={"pos_row_idx": "row_idx"}),
        aux_pairs[["source_file", "neg_row_idx"]].rename(columns={"neg_row_idx": "row_idx"}),
    ]).drop_duplicates()
    aux = keys.merge(aux_all, on=["source_file", "row_idx"], how="inner")
    aux["group_id"] = pd.NA
    aux["corpus"] = "aux"

    rows = pd.concat([main, aux], ignore_index=True)
    rows["seq_light"] = rows.seq_light.fillna("")
    rows["seq_id"] = [seq_id_of(h, l) for h, l in zip(rows.seq_heavy, rows.seq_light)]

    # row map: every corpus row -> its dedup id
    row_map = rows[["corpus", "source_file", "row_idx", "seq_id"]].drop_duplicates()
    row_map.to_csv(os.path.join(cache_dir, "row_map.csv"), index=False)

    manifest = (
        rows.groupby("seq_id")
        .agg(
            seq_heavy=("seq_heavy", "first"),
            seq_light=("seq_light", "first"),
            format=("format", "first"),
            in_main=("corpus", lambda c: bool((c == "main").any())),
            in_aux=("corpus", lambda c: bool((c == "aux").any())),
        )
        .reset_index()
    )
    manifest["len_heavy"] = manifest.seq_heavy.str.len()
    manifest["len_light"] = manifest.seq_light.str.len()
    manifest = manifest.sort_values("seq_id").reset_index(drop=True)
    manifest.to_csv(os.path.join(cache_dir, "manifest.csv"), index=False)
    print(f"manifest: {len(manifest)} unique sequences "
          f"({manifest.in_main.sum()} main, {manifest.in_aux.sum()} aux); "
          f"row_map: {len(row_map)} rows")
    return manifest


# ---------------------------------------------------------------- encoders

class ProGenEncoder:
    def __init__(self):
        self.tok = Tokenizer.from_file(os.path.join(PROGEN_DIR, "tokenizer.json"))
        self.model = AutoModelForCausalLM.from_pretrained(PROGEN_DIR, trust_remote_code=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def encode_ids(self, heavy: str, light: str) -> torch.Tensor:
        s = "1" + heavy + (LINKER + light if light else "") + "2"
        return torch.tensor(self.tok.encode(s).ids, dtype=torch.long)

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, mask: torch.Tensor):
        out = self.model(ids, attention_mask=mask, output_hidden_states=True)
        return out.logits, out.hidden_states[-1]


class IgLMEncoder:
    def __init__(self):
        # char-level vocab; line number == token id (matches iglm trained_models/vocab.txt).
        # BertTokenizerFast under transformers>=5 maps these tokens to UNK, hence the manual map.
        with open(IGLM_VOCAB) as f:
            self.vocab = {tok.strip(): i for i, tok in enumerate(f) if tok.strip()}
        self.model = GPT2LMHeadModel.from_pretrained(IGLM_DIR)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def encode_ids(self, chain: str, chain_token: str) -> torch.Tensor:
        tokens = [chain_token, "[HUMAN]", *chain, "[SEP]"]
        return torch.tensor([self.vocab[t] for t in tokens], dtype=torch.long)

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, mask: torch.Tensor):
        out = self.model(ids, attention_mask=mask, output_hidden_states=True)
        return out.logits, out.hidden_states[-1]


# ---------------------------------------------------------------- batching

def pad_batch(list_of_ids, pad_id=0):
    maxlen = max(len(t) for t in list_of_ids)
    ids = torch.full((len(list_of_ids), maxlen), pad_id, dtype=torch.long)
    mask = torch.zeros((len(list_of_ids), maxlen), dtype=torch.long)
    for i, t in enumerate(list_of_ids):
        ids[i, : len(t)] = t
        mask[i, : len(t)] = 1
    return ids, mask


def progen_nll(logits_row: torch.Tensor, ids_row: torch.Tensor, n_aa_heavy: int, n_aa_light: int) -> float:
    """Mean CE over AA labels only (linker/sentinels excluded), vocab restricted to ids 5..29."""
    shift_logits = logits_row[:-1]  # predicts token t+1
    shift_labels = ids_row[1:]
    keep = (shift_labels >= PROGEN_AA_LO) & (shift_labels <= PROGEN_AA_HI)
    # exclude linker positions: label index of linker start = 1 + n_aa_heavy (0-based in ids_row)
    if n_aa_light:
        linker_lo = 1 + n_aa_heavy
        linker_hi = linker_lo + len(LINKER)
        label_pos = torch.arange(1, len(ids_row))
        keep &= ~((label_pos >= linker_lo) & (label_pos < linker_hi))
    lg = shift_logits[keep][:, PROGEN_AA_LO : PROGEN_AA_HI + 1]
    lb = shift_labels[keep] - PROGEN_AA_LO
    return torch.nn.functional.cross_entropy(lg, lb).item()


def iglm_nll(logits_row: torch.Tensor, ids_row: torch.Tensor) -> float:
    """iglm==0.1.0 IgLM.log_likelihood: mean CE of labels[2:] (full vocab)."""
    shift_logits = logits_row[1:-1]
    shift_labels = ids_row[2:].long()
    return torch.nn.functional.cross_entropy(shift_logits, shift_labels).item()


# ---------------------------------------------------------------- extraction

def extract(cache_dir: str, limit: int | None, batch_tokens: int, threads: int):
    torch.set_num_threads(threads)
    manifest = pd.read_csv(os.path.join(cache_dir, "manifest.csv"))
    manifest.seq_light = manifest.seq_light.fillna("")

    seqs_dir = os.path.join(cache_dir, "seqs")
    index_path = os.path.join(cache_dir, "index.csv")

    done = set()
    if os.path.exists(index_path):
        done = set(pd.read_csv(index_path, usecols=["seq_id"]).seq_id)
    todo = manifest[~manifest.seq_id.isin(done)].copy()
    if limit:
        todo = todo.head(limit)
    if len(todo) == 0:
        print("nothing to do")
        return
    # length-sorted for efficient batching; dedup chain work is small, keep pairs
    todo["n_aa"] = todo.len_heavy + todo.len_light
    todo = todo.sort_values("n_aa").reset_index(drop=True)
    print(f"{len(todo)} sequences to extract ({len(done)} already cached)")

    progen = ProGenEncoder()
    iglm = IgLMEncoder()

    write_header = not os.path.exists(index_path)
    index_file = open(index_path, "a")
    if write_header:
        index_file.write("seq_id,path,len_heavy,len_light,nll_progen,nll_iglm_heavy,nll_iglm_light\n")

    n_done, t0 = 0, time.time()
    i = 0
    while i < len(todo):
        # greedy pack by token budget (progen length = n_aa + linker + 2)
        batch, budget = [], batch_tokens
        while i < len(todo):
            row = todo.iloc[i]
            cost = int(row.n_aa) + (len(LINKER) if row.len_light > 0 else 0) + 2
            if batch and cost > budget:
                break
            batch.append(row)
            budget -= cost
            i += 1

        # ---- ProGen2 batch
        p_ids = [progen.encode_ids(r.seq_heavy, r.seq_light) for r in batch]
        ids, mask = pad_batch(p_ids)
        logits, hidden = progen.forward(ids, mask)
        progen_out = []
        for b, r in enumerate(batch):
            L = len(p_ids[b])
            lh, ll = int(r.len_heavy), int(r.len_light)
            keep = list(range(1, 1 + lh)) + (list(range(1 + lh + len(LINKER), L - 1)) if ll else [])
            h = hidden[b, keep].to(torch.float16).numpy()
            nll = progen_nll(logits[b, :L], ids[b, :L], lh, ll)
            progen_out.append((h, nll))

        # ---- IgLM batches (one forward per chain)
        chains = []  # (batch_idx, is_light, ids)
        for b, r in enumerate(batch):
            chains.append((b, False, iglm.encode_ids(r.seq_heavy, "[HEAVY]")))
            if r.seq_light:
                chains.append((b, True, iglm.encode_ids(r.seq_light, "[LIGHT]")))
        iglm_hidden = {}
        iglm_nlls = {}
        for j in range(0, len(chains), max(1, batch_tokens // 200)):
            sub = chains[j : j + max(1, batch_tokens // 200)]
            c_ids, c_mask = pad_batch([c[2] for c in sub])
            c_logits, c_hidden = iglm.forward(c_ids, c_mask)
            for k, (b, is_light, cids) in enumerate(sub):
                L = len(cids)
                iglm_hidden[(b, is_light)] = c_hidden[k, 2 : L - 1].to(torch.float16).numpy()
                iglm_nlls[(b, is_light)] = iglm_nll(c_logits[k, :L], c_ids[k, :L])

        # ---- write cache
        for b, r in enumerate(batch):
            h_p, nll_p = progen_out[b]
            h_h = iglm_hidden[(b, False)]
            has_light = bool(r.seq_light)
            h_l = iglm_hidden.get((b, True))
            h_i = np.concatenate([h_h, h_l], axis=0) if has_light else h_h
            assert h_i.shape[0] == h_p.shape[0] == int(r.n_aa), (r.seq_id, h_i.shape, h_p.shape)
            sub = os.path.join(seqs_dir, r.seq_id[:2])
            os.makedirs(sub, exist_ok=True)
            path = os.path.join(sub, r.seq_id + ".npz")
            tmp = os.path.join(sub, r.seq_id + ".tmp.npz")
            np.savez(tmp,
                     h_progen=h_p, h_iglm=h_i,
                     len_heavy=np.int16(r.len_heavy),
                     nll_progen=np.float32(nll_p),
                     nll_iglm_heavy=np.float32(iglm_nlls[(b, False)]),
                     nll_iglm_light=np.float32(iglm_nlls[(b, True)]) if has_light else np.float32(np.nan))
            os.rename(tmp, path)
            rel = os.path.relpath(path, cache_dir)
            index_file.write(f"{r.seq_id},{rel},{r.len_heavy},{r.len_light},"
                             f"{nll_p:.6f},{iglm_nlls[(b, False)]:.6f},"
                             f"{iglm_nlls[(b, True)]:.6f}\n" if has_light else
                             f"{r.seq_id},{rel},{r.len_heavy},{r.len_light},"
                             f"{nll_p:.6f},{iglm_nlls[(b, False)]:.6f},\n")
        index_file.flush()

        n_done += len(batch)
        dt = time.time() - t0
        rate = n_done / dt
        eta = (len(todo) - n_done) / rate if rate > 0 else math.inf
        print(f"{n_done}/{len(todo)} seqs  {rate:.2f} seq/s  eta {eta/3600:.1f} h", flush=True)

    index_file.close()
    print("done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="extraction/cache")
    ap.add_argument("--build-manifest", action="store_true", help="build manifest + row_map and exit")
    ap.add_argument("--limit", type=int, default=None, help="extract only the first N pending sequences")
    ap.add_argument("--batch-tokens", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    if args.build_manifest:
        build_manifest(args.cache_dir)
        return
    if not os.path.exists(os.path.join(args.cache_dir, "manifest.csv")):
        build_manifest(args.cache_dir)
    extract(args.cache_dir, args.limit, args.batch_tokens, args.threads)


if __name__ == "__main__":
    main()
