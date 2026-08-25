#!/usr/bin/env python
"""ProstT5 structure-aware embedding extraction (struct arm, no structures needed).

ProstT5 (Rostlab) is a ProtT5-XL encoder-decoder finetuned to translate
between amino-acid and Foldseek 3Di sequences. Encoding an AA sequence with
the "<AA2fold>" prefix yields residue embeddings informed by the model's
internal structure translation — a cheap proxy for structure-aware features
without running structure prediction.

Conventions mirror extract_antiberty.py:
  per chain: "<AA2fold> " + spaced AA  ->  encoder last hidden state,
             prefix/EOS positions stripped
  scFv/Fab:  concat(heavy, light) residue embeddings -> mean-pooled [1024]
  VHH:       heavy only
  output:    fp32 [1024] .npy per seq_id/ag_id + index.csv (resumable)

Inputs:
  antibodies: extraction/cache_antiberty/manifest.csv (seq_id, seq_heavy, seq_light)
  antigens:   extraction/cache_esm2/antigen_manifest.csv (ag_id, ag_seq)

Run from the repo root:
  .venv/bin/python extraction/extract_prostt5.py [--limit N] [--skip-ab] [--skip-ag]
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_embeddings import pad_batch  # noqa: E402

PROSTT5_DIR = "model_data/prostt5"
STRUCT_DIM = 1024


class ProstT5Encoder:
    """Frozen ProstT5 encoder producing per-residue AA embeddings.

    Tokenization uses sentencepiece directly on spiece.model (transformers v5
    misdetects the slow T5 tokenizer as tiktoken); the "<AA2fold>" prefix id
    comes from added_tokens.json, EOS is sp.eos_id(), pad is 0 (== pad_batch).
    """

    def __init__(self, model_dir=PROSTT5_DIR):
        import json as _json

        import sentencepiece as spm
        from transformers import T5EncoderModel
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sp = spm.SentencePieceProcessor(
            model_file=os.path.join(model_dir, "spiece.model"))
        with open(os.path.join(model_dir, "added_tokens.json")) as f:
            self.prefix_id = _json.load(f)["<AA2fold>"]
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = T5EncoderModel.from_pretrained(
            model_dir, torch_dtype=dtype).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def encode_ids(self, chain: str) -> torch.Tensor:
        ids = [self.prefix_id, *self.sp.encode(" ".join(chain)),
               self.sp.eos_id()]
        return torch.tensor(ids, dtype=torch.long)

    @torch.no_grad()
    def hidden_chains(self, chains):
        """chains: list of AA strings -> list of fp32 [L,1024] (prefix/EOS
        stripped)."""
        ids_list = [self.encode_ids(c) for c in chains]
        ids, mask = pad_batch(ids_list)
        out = self.model(input_ids=ids.to(self.device),
                         attention_mask=mask.to(self.device))
        h = out.last_hidden_state  # [B, Lmax+2, 1024]
        res = []
        for k, cids in enumerate(ids_list):
            L = len(cids)  # prefix + residues + EOS
            res.append(h[k, 1: L - 1].to(torch.float32).cpu().numpy())
        return res


def pooled_antibody(enc, seq_heavy, seq_light):
    """Mean-pooled [1024] over heavy+light residues (concat convention)."""
    chains = [seq_heavy] + ([seq_light] if seq_light else [])
    hs = enc.hidden_chains(chains)
    h = np.concatenate(hs, axis=0)
    assert h.shape[0] == len(seq_heavy) + len(seq_light), (h.shape, len(chains))
    return h.mean(axis=0).astype(np.float32)


def _save_npy(root, sub, key, vec):
    sub_dir = os.path.join(root, sub, key[:2])
    os.makedirs(sub_dir, exist_ok=True)
    path = os.path.join(sub_dir, key + ".npy")
    tmp = os.path.join(sub_dir, key + ".tmp.npy")
    np.save(tmp, vec)
    os.rename(tmp, path)
    return os.path.join(key[:2], key + ".npy")


def extract(cache_dir, model_dir, limit, batch_tokens, threads,
            skip_ab, skip_ag):
    torch.set_num_threads(threads)
    enc = ProstT5Encoder(model_dir)

    jobs = []
    if not skip_ab:
        man = pd.read_csv("extraction/cache_antiberty/manifest.csv")
        man.seq_light = man.seq_light.fillna("")
        jobs.append(("ab", man[["seq_id", "seq_heavy", "seq_light"]]))
    if not skip_ag:
        agman = pd.read_csv("extraction/cache_esm2/antigen_manifest.csv")
        agman = agman[agman.ag_seq.notna() & (agman.ag_seq != "")]
        # some manifest rows carry mutation *labels* (e.g. "SARS-CoV2_(K527T)")
        # instead of sequences; skip anything outside the plain AA alphabet
        # (missing ids zero-fill downstream, degrading those groups to baseline)
        ok = agman.ag_seq.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+")
        n_bad = int((~ok).sum())
        if n_bad:
            print(f"[ag] skipping {n_bad} non-sequence manifest rows: "
                  f"{agman.loc[~ok, 'ag_seq'].tolist()}")
        agman = agman[ok]
        jobs.append(("ag", agman[["ag_id", "ag_seq"]]))

    for kind, df in jobs:
        id_col = "seq_id" if kind == "ab" else "ag_id"
        index_path = os.path.join(cache_dir, kind, "index.csv")
        os.makedirs(os.path.join(cache_dir, kind), exist_ok=True)
        done = set()
        if os.path.exists(index_path):
            done = set(pd.read_csv(index_path, usecols=[id_col])[id_col])
        todo = df[~df[id_col].isin(done)].copy()
        if limit:
            todo = todo.head(limit)
        if len(todo) == 0:
            print(f"[{kind}] nothing to do ({len(done)} cached)")
            continue
        n_aa = (todo.seq_heavy.str.len() + todo.seq_light.str.len()
                if kind == "ab" else todo.ag_seq.str.len())
        todo = todo.assign(n_aa=n_aa).sort_values("n_aa").reset_index(drop=True)
        print(f"[{kind}] {len(todo)} to extract ({len(done)} cached)", flush=True)

        write_header = not os.path.exists(index_path)
        index_file = open(index_path, "a")
        if write_header:
            index_file.write(f"{id_col},path\n")

        n_done, t0 = 0, time.time()
        i = 0
        while i < len(todo):
            batch, budget = [], batch_tokens
            while i < len(todo):
                row = todo.iloc[i]
                cost = int(row.n_aa) + 4
                if batch and cost > budget:
                    break
                batch.append(row)
                budget -= cost
                i += 1
            # encode each chain separately (prefix/EOS per chain)
            chains, owner = [], []
            for b, r in enumerate(batch):
                if kind == "ab":
                    chains.append(r.seq_heavy)
                    owner.append((b, len(r.seq_heavy)))
                    if r.seq_light:
                        chains.append(r.seq_light)
                        owner.append((b, len(r.seq_light)))
                else:
                    chains.append(r.ag_seq)
                    owner.append((b, len(r.ag_seq)))
            per = max(1, batch_tokens // 2000)
            vecs = {}
            for j in range(0, len(chains), per):
                sub = chains[j: j + per]
                hs = enc.hidden_chains(sub)
                for k, h in enumerate(hs):
                    b = owner[j + k][0]
                    vecs.setdefault(b, []).append(h)
            for b, r in enumerate(batch):
                h = np.concatenate(vecs[b], axis=0)
                if h.shape[0] != int(r.n_aa):
                    if kind == "ab":
                        raise AssertionError((r[id_col], h.shape, int(r.n_aa)))
                    # ag: sentencepiece may merge/unk odd-but-legal strings;
                    # the pooled vector is still meaningful, keep it
                    print(f"[ag] warn: {r[id_col]} tokens {h.shape[0]} != "
                          f"len {int(r.n_aa)}; pooling anyway", flush=True)
                vec = h.mean(axis=0).astype(np.float32)
                rel = _save_npy(cache_dir, kind, r[id_col], vec)
                index_file.write(f"{r[id_col]},{rel}\n")
            index_file.flush()
            n_done += len(batch)
            rate = n_done / (time.time() - t0)
            eta = (len(todo) - n_done) / rate if rate > 0 else float("inf")
            print(f"[{kind}] {n_done}/{len(todo)}  {rate:.2f}/s  "
                  f"eta {eta/3600:.1f} h", flush=True)
        index_file.close()
    print("done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="extraction/cache_struct_prostt5")
    ap.add_argument("--model-dir", default=PROSTT5_DIR)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-tokens", type=int, default=16384)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--skip-ab", action="store_true")
    ap.add_argument("--skip-ag", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    extract(args.cache_dir, args.model_dir, args.limit, args.batch_tokens,
            args.threads, args.skip_ab, args.skip_ag)


if __name__ == "__main__":
    main()
