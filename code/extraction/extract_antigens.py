#!/usr/bin/env python
"""ESM-2 650M antigen-embedding extraction (Plan v4 antigen tower).

Collects one antigen sequence per training group from three sources:
  1. harmonized_clustered.csv.gz `ag_seq` column (AbRank + kothiwal),
  2. extraction/antigen_table.csv (curated: SAbDab PDB / manual / aux),
  3. aux files keyed `AUX::<source_file>` in the same table.

Dedups by sequence (ag_id = sha1[:16]), encodes with frozen ESM-2 650M,
stores fp16 [L_ag, 1280] per antigen + `antigen_index.csv` (group_id -> ag_id).

Antigens >1000 aa exceed ESM-2's 1024-token context: sliding windows
(1000 aa, stride 500) with per-residue averaging over overlaps.

Run from the repo root:  .venv/bin/python extraction/extract_antigens.py [--limit N]
"""

import argparse
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_embeddings import pad_batch  # noqa: E402

ESM_DIR = "model_data/esm2-650m"
PIPE_OUT = "data_pipeline/output"
WINDOW, STRIDE = 1000, 500


def ag_id_of(seq: str) -> str:
    return hashlib.sha1(seq.encode()).hexdigest()[:16]


def build_antigen_manifest(cache_dir: str) -> pd.DataFrame:
    """Write antigen_manifest.csv (ag_id, ag_seq) + antigen_index.csv (key, ag_id)."""
    df = pd.read_csv(os.path.join(PIPE_OUT, "harmonized_clustered.csv.gz"),
                     low_memory=False)[["group_id", "ag_seq"]]
    g = df.groupby("group_id").ag_seq.first().reset_index()

    table_path = "extraction/antigen_table.csv"
    table = pd.read_csv(table_path) if os.path.exists(table_path) else pd.DataFrame(
        columns=["group_id", "ag_seq", "source"])

    # main-corpus groups: harmonized ag_seq first, curated table as fallback
    g = g.merge(table[~table.group_id.str.startswith("AUX::")][["group_id", "ag_seq"]],
                on="group_id", how="left", suffixes=("", "_curated"))
    g["ag_final"] = g.ag_seq.where(g.ag_seq.notna(), g.ag_seq_curated)
    index = g[["group_id", "ag_final"]].rename(columns={"group_id": "key"})

    # aux rows from the curated table
    aux = table[table.group_id.str.startswith("AUX::")][["group_id", "ag_seq"]]
    aux = aux.rename(columns={"group_id": "key", "ag_seq": "ag_final"})
    index = pd.concat([index, aux], ignore_index=True)

    index = index[index.ag_final.notna() & (index.ag_final.str.len() > 0)].copy()
    index["ag_id"] = [ag_id_of(s) for s in index.ag_final]
    manifest = index[["ag_id", "ag_final"]].drop_duplicates("ag_id").rename(
        columns={"ag_final": "ag_seq"}).reset_index(drop=True)
    manifest["len"] = manifest.ag_seq.str.len()

    index[["key", "ag_id"]].to_csv(os.path.join(cache_dir, "antigen_index.csv"), index=False)
    manifest.to_csv(os.path.join(cache_dir, "antigen_manifest.csv"), index=False)
    n_null_main = g.ag_final.isna().sum() + int((g.ag_final.fillna("").str.len() == 0).sum())
    print(f"antigens: {len(manifest)} unique | keys mapped: {len(index)} | "
          f"main groups without antigen (null): {n_null_main} | "
          f"windowed (>1000 aa): {(manifest.len > WINDOW).sum()}")
    return manifest


@torch.no_grad()
def esm_hidden_windowed(tok, model, seq: str, device: str = "cpu") -> np.ndarray:
    """fp16 [L, 1280]; windowed average for long sequences."""
    if len(seq) <= WINDOW:
        enc = tok(seq, return_tensors="pt")
        h = model(**{k: v.to(device) for k, v in enc.items()}).last_hidden_state[0, 1 : 1 + len(seq)]
        return h.to(torch.float16).cpu().numpy()
    acc = torch.zeros(len(seq), model.config.hidden_size)
    cnt = torch.zeros(len(seq), 1)
    for start in range(0, len(seq), STRIDE):
        chunk = seq[start : start + WINDOW]
        enc = tok(chunk, return_tensors="pt")
        h = model(**{k: v.to(device) for k, v in enc.items()}).last_hidden_state[0, 1 : 1 + len(chunk)]
        acc[start : start + len(chunk)] += h.cpu()
        cnt[start : start + len(chunk)] += 1
        if start + WINDOW >= len(seq):
            break
    return (acc / cnt).to(torch.float16).numpy()


@torch.no_grad()
def extract(cache_dir: str, limit: int | None, batch_tokens: int, threads: int):
    torch.set_num_threads(threads)
    manifest = pd.read_csv(os.path.join(cache_dir, "antigen_manifest.csv"))
    ags_dir = os.path.join(cache_dir, "ags")
    done = set()
    if os.path.isdir(ags_dir):
        for sub in os.listdir(ags_dir):
            d = os.path.join(ags_dir, sub)
            if os.path.isdir(d):
                done |= {f[:-4] for f in os.listdir(d) if f.endswith(".npz")}
    todo = manifest[~manifest.ag_id.isin(done)].copy()
    if limit:
        todo = todo.head(limit)
    if len(todo) == 0:
        print("nothing to do")
        return
    todo = todo.sort_values("len").reset_index(drop=True)
    print(f"{len(todo)} antigens to extract ({len(done)} cached)")

    tok = AutoTokenizer.from_pretrained(ESM_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(ESM_DIR).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_done, t0 = 0, time.time()
    # short antigens: batched; long ones: windowed single-pass
    short = todo[todo.len <= WINDOW]
    long_ = todo[todo.len > WINDOW]

    i = 0
    while i < len(short):
        batch, budget = [], batch_tokens
        while i < len(short):
            row = short.iloc[i]
            cost = int(row.len) + 2
            if batch and cost > budget:
                break
            batch.append(row)
            budget -= cost
            i += 1
        encs = [tok(r.ag_seq, return_tensors="pt") for r in batch]
        ids, mask = pad_batch([e.input_ids[0] for e in encs])
        h = model(ids.to(device), attention_mask=mask.to(device)).last_hidden_state
        for b, r in enumerate(batch):
            L = int(r.len)
            h_ag = h[b, 1 : 1 + L].to(torch.float16).cpu().numpy()
            sub = os.path.join(ags_dir, r.ag_id[:2])
            os.makedirs(sub, exist_ok=True)
            path = os.path.join(sub, r.ag_id + ".npz")
            np.savez(os.path.join(sub, r.ag_id + ".tmp.npz"), h_ag=h_ag)
            os.rename(path.replace(".npz", ".tmp.npz"), path)
        n_done += len(batch)
        rate = n_done / (time.time() - t0)
        print(f"{n_done}/{len(todo)} antigens  {rate:.2f}/s", flush=True)

    for _, r in long_.iterrows():
        h_ag = esm_hidden_windowed(tok, model, r.ag_seq, device)
        sub = os.path.join(ags_dir, r.ag_id[:2])
        os.makedirs(sub, exist_ok=True)
        path = os.path.join(sub, r.ag_id + ".npz")
        np.savez(os.path.join(sub, r.ag_id + ".tmp.npz"), h_ag=h_ag)
        os.rename(path.replace(".npz", ".tmp.npz"), path)
        n_done += 1
        print(f"{n_done}/{len(todo)} antigens (windowed, len={int(r.len)})", flush=True)

    print("done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="extraction/cache_esm2")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-tokens", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    if not os.path.exists(os.path.join(args.cache_dir, "antigen_manifest.csv")):
        build_antigen_manifest(args.cache_dir)
    extract(args.cache_dir, args.limit, args.batch_tokens, args.threads)


if __name__ == "__main__":
    main()
