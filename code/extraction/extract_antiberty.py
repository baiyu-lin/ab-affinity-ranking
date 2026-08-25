#!/usr/bin/env python
"""AntiBERTy antibody-embedding extraction (Plan v4 antibody tower).

Same cache conventions as extract_embeddings.py but with the AntiBERTy MLM
encoder (no chain/species conditioning, no autoregressive NLL):

  per chain: [CLS] + chain + [SEP]  ->  last hidden state, specials stripped
  scFv/Fab:  concat(heavy, light)   ->  h_ab [L, 512] fp16
  VHH:       heavy only

Run from the repo root:  .venv/bin/python extraction/extract_antiberty.py [--limit N] [--cache-dir ...]
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import BertModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_embeddings import build_manifest, pad_batch  # noqa: E402

ANTIBERTY_DIR = "model_data/antiberty"


class AntiBERTyEncoder:
    def __init__(self):
        with open(os.path.join(ANTIBERTY_DIR, "vocab.txt")) as f:
            self.vocab = {t.strip(): i for i, t in enumerate(f) if t.strip()}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BertModel.from_pretrained(ANTIBERTY_DIR).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def encode_ids(self, chain: str) -> torch.Tensor:
        ids = [self.vocab["[CLS]"], *[self.vocab[c] for c in chain], self.vocab["[SEP]"]]
        return torch.tensor(ids, dtype=torch.long)


@torch.no_grad()
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
    todo["n_aa"] = todo.len_heavy + todo.len_light
    todo = todo.sort_values("n_aa").reset_index(drop=True)
    print(f"{len(todo)} sequences to extract ({len(done)} already cached)")

    enc = AntiBERTyEncoder()
    write_header = not os.path.exists(index_path)
    index_file = open(index_path, "a")
    if write_header:
        index_file.write("seq_id,path,len_heavy,len_light\n")

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

        chains = []  # (batch_idx, is_light, ids)
        for b, r in enumerate(batch):
            chains.append((b, False, enc.encode_ids(r.seq_heavy)))
            if r.seq_light:
                chains.append((b, True, enc.encode_ids(r.seq_light)))
        hidden = {}
        per = max(1, batch_tokens // 200)
        for j in range(0, len(chains), per):
            sub = chains[j : j + per]
            c_ids, c_mask = pad_batch([c[2] for c in sub])
            out = enc.model(c_ids.to(enc.device), attention_mask=c_mask.to(enc.device))
            h = out.last_hidden_state
            for k, (b, is_light, cids) in enumerate(sub):
                L = len(cids)
                hidden[(b, is_light)] = h[k, 1 : L - 1].to(torch.float16).cpu().numpy()

        for b, r in enumerate(batch):
            h_h = hidden[(b, False)]
            has_light = bool(r.seq_light)
            h_ab = np.concatenate([h_h, hidden[(b, True)]], axis=0) if has_light else h_h
            assert h_ab.shape[0] == int(r.n_aa), (r.seq_id, h_ab.shape)
            sub_dir = os.path.join(seqs_dir, r.seq_id[:2])
            os.makedirs(sub_dir, exist_ok=True)
            path = os.path.join(sub_dir, r.seq_id + ".npz")
            tmp = os.path.join(sub_dir, r.seq_id + ".tmp.npz")
            np.savez(tmp, h_ab=h_ab, len_heavy=np.int16(r.len_heavy))
            os.rename(tmp, path)
            rel = os.path.relpath(path, cache_dir)
            index_file.write(f"{r.seq_id},{rel},{r.len_heavy},{r.len_light}\n")
        index_file.flush()

        n_done += len(batch)
        rate = n_done / (time.time() - t0)
        eta = (len(todo) - n_done) / rate if rate > 0 else float("inf")
        print(f"{n_done}/{len(todo)} seqs  {rate:.2f} seq/s  eta {eta/3600:.1f} h", flush=True)

    index_file.close()
    print("done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="extraction/cache_antiberty")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-tokens", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    if not os.path.exists(os.path.join(args.cache_dir, "manifest.csv")):
        build_manifest(args.cache_dir)
    extract(args.cache_dir, args.limit, args.batch_tokens, args.threads)


if __name__ == "__main__":
    main()
