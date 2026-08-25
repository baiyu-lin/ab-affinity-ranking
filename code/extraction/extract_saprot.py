#!/usr/bin/env python
"""SaProt structure-aware embedding extraction (Arm 2: true predicted structures).

Pipeline per unique antibody sequence (streaming, PDB coordinates discarded):
    IgFold predicts VH(+VL) Fv structure (pip-bundled ckpts, no refinement)
    -> foldseek structureto3didescriptor -> per-residue 3Di tokens
    -> SaProt-650M (AF2) encodes "Aa+3di" combined tokens, per chain
    -> concat chains -> fp32 [1280] mean-pool (default) or, with
       --per-residue, fp16 [L,1280] token embeddings .npy per seq_id
       (+index.csv); --per-residue also persists the 3Di tokens to
       <cache>/3di/ so future encoders can skip the IgFold/foldseek stage

Resume-safe: seq_ids already in index.csv are skipped; failures are logged to
failed.csv and skipped (missing ids zero-fill downstream, degrading to the
sequence-only baseline contribution).

Run from the repo root:
  python extraction/extract_saprot.py [--limit N] [--batch-size 16]
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_embeddings import pad_batch  # noqa: E402

SAPROT_DIR = "model_data/saprot_650m_af2"
FOLDSEEK = "/root/bin/foldseek/bin/foldseek"
STRUCT_DIM = 1280


class _IgFoldAntiBERTy:
    """Minimal AntiBERTyRunner replacement for igfold's embed() contract.

    The pip `antiberty` package is broken under transformers v5 (its bundled
    checkpoint pickles removed classes like Trie/BasicTokenizer). This shim
    loads the HF-mirror AntiBERTy (model_data/antiberty, same weights as
    AntiBERTy_md_smooth) with plain BertModel and replicates
    AntiBERTyRunner.embed() exactly (char vocab, [CLS]/[SEP], mask-filtered
    hidden states and attentions).
    """

    def __init__(self, model_dir="model_data/antiberty"):
        from transformers import BertModel
        with open(os.path.join(model_dir, "vocab.txt")) as f:
            self.vocab = {t.strip(): i for i, t in enumerate(f) if t.strip()}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BertModel.from_pretrained(model_dir,
                                               attn_implementation="eager")
        self.model = self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def embed(self, sequences, hidden_layer=-1, return_attention=False):
        ids_list = []
        for s in sequences:
            ids = [self.vocab["[CLS]"], *[self.vocab[c] for c in s],
                   self.vocab["[SEP]"]]
            ids_list.append(torch.tensor(ids, dtype=torch.long))
        ids, mask = pad_batch(ids_list)
        out = self.model(input_ids=ids.to(self.device),
                         attention_mask=mask.to(self.device),
                         output_hidden_states=True,
                         output_attentions=return_attention)
        hs = torch.stack(out.hidden_states, dim=1)  # [B, layers, Lmax, 512]
        embeddings = []
        for i in range(len(ids_list)):
            a = mask[i].bool()
            embeddings.append(hs[i][:, a][hidden_layer])      # [L, 512]
        if not return_attention:
            return embeddings
        atts = torch.stack(out.attentions, dim=1)  # [B, layers, H, Lmax, Lmax]
        attentions = []
        for i in range(len(ids_list)):
            a = mask[i].bool()
            attentions.append(atts[i][:, :, a][:, :, :, a])  # [layers,H,L,L]
        return embeddings, attentions


def _apply_igfold_patches():
    """torch.load + legacy-tokenizer stubs needed by igfold's bundled ckpts."""
    _orig_load = torch.load

    def _patched(*a, **k):
        k.setdefault("weights_only", False)
        return _orig_load(*a, **k)

    torch.load = _patched
    import transformers.models.bert.tokenization_bert as _tb
    import transformers.tokenization_utils_sentencepiece as _tus

    def _stub_getattr(name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {})

    _tus.__getattr__ = _stub_getattr
    _tb.__getattr__ = _stub_getattr


class StructurePredictor:
    """IgFold wrapper: sequence dict -> unrefined PDB path.

    Zenodo was too slow from the training instance, so IgFold (pip-bundled
    weights) replaced ABodyBuilder2; refinement/renumbering are skipped —
    3Di tokens only need backbone geometry. num_models=1 trades a little
    accuracy for 4x throughput (measured: 2.3 s vs 8.5 s per Fv).
    """

    def __init__(self, num_models=1):
        _apply_igfold_patches()
        import sys
        from igfold import IgFoldRunner
        # NB: igfold/__init__ rebinds igfold.IgFoldRunner to the class, so the
        # module must be fetched from sys.modules to patch its global
        igr_mod = sys.modules["igfold.IgFoldRunner"]
        igr_mod.AntiBERTyRunner = _IgFoldAntiBERTy
        self.ig = IgFoldRunner(num_models=num_models)

    def predict_pdb(self, heavy, light, out_path):
        seqs = {"H": heavy}
        if light:
            seqs["L"] = light
        self.ig.fold(out_path, sequences=seqs, do_refine=False,
                     do_renum=False)


def foldseek_3di(pdb_path, workdir):
    """Return {chain_id: (aa_seq, 3di_seq)} via foldseek structureto3didescriptor.

    TSV rows: "desc \t aa_seq \t 3di_seq"; chain id is the last _-segment of
    desc after stripping the pdb basename (same parsing as SaProt's official
    utils/foldseek_util.py). No plddt masking: IgFold PDBs are not AlphaFold
    outputs, and low-confidence regions still carry backbone geometry.
    """
    out = os.path.join(workdir, "fs_out.tsv")
    subprocess.run([FOLDSEEK, "structureto3didescriptor", "-v", "0",
                    "--threads", "1", "--chain-name-mode", "1",
                    pdb_path, out], check=True, capture_output=True)
    name = os.path.basename(pdb_path)
    res = {}
    with open(out) as f:
        for line in f:
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            desc, seq, di = parts[0], parts[1], parts[2].strip()
            chain = desc.split(" ")[0].replace(name, "").split("_")[-1]
            res[chain] = (seq, di)
    return res


class SaProtEncoder:
    """Frozen SaProt-650M encoder over combined AA+3Di tokens."""

    def __init__(self, model_dir=SAPROT_DIR):
        from transformers import AutoModel, AutoTokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            model_dir, torch_dtype=dtype).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def encode_ids(self, aa_seq, di_seq) -> torch.Tensor:
        toks = " ".join(a + d.lower() for a, d in zip(aa_seq, di_seq))
        ids = self.tok.encode(toks)
        return torch.tensor(ids, dtype=torch.long)

    @torch.no_grad()
    def hidden(self, chains):
        """chains: list of (aa_seq, di_seq) -> list of fp32 [L,1280]."""
        ids_list = [self.encode_ids(a, d) for a, d in chains]
        ids, mask = pad_batch(ids_list)
        out = self.model(input_ids=ids.to(self.device),
                         attention_mask=mask.to(self.device))
        h = out.last_hidden_state
        res = []
        for k, cids in enumerate(ids_list):
            L = len(cids)
            res.append(h[k, 1: L - 1].to(torch.float32).cpu().numpy())
        return res


def _save_npy(root, key, vec):
    sub_dir = os.path.join(root, "ab", key[:2])
    os.makedirs(sub_dir, exist_ok=True)
    path = os.path.join(sub_dir, key + ".npy")
    tmp = os.path.join(sub_dir, key + ".tmp.npy")
    np.save(tmp, vec)
    os.rename(tmp, path)
    return os.path.join(key[:2], key + ".npy")


def _save_3di(root, key, chains):
    """Persist the foldseek 3Di tokens so future encoders can skip IgFold.

    chains: list of (aa_seq, di_seq) in H,L order -> root/3di/<pp>/<key>.json
    """
    import json
    sub_dir = os.path.join(root, "3di", key[:2])
    os.makedirs(sub_dir, exist_ok=True)
    path = os.path.join(sub_dir, key + ".json")
    tmp = os.path.join(sub_dir, key + ".tmp.json")
    payload = {c: [a, d] for c, (a, d) in zip(("H", "L"), chains)}
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.rename(tmp, path)


def _worker(task_q, result_q, num_models):
    """Fold+3Di worker: owns an IgFold predictor, streams (seq_id, chains)."""
    import tempfile

    import warnings
    warnings.filterwarnings("ignore")
    pred = StructurePredictor(num_models=num_models)
    with tempfile.TemporaryDirectory() as workdir:
        while True:
            item = task_q.get()
            if item is None:
                return
            seq_id, heavy, light, n_aa = item
            pdb = os.path.join(workdir, f"{seq_id}.pdb")
            try:
                pred.predict_pdb(heavy, light, pdb)
                chains = foldseek_3di(pdb, workdir)
                ordered = [chains[c] for c in ("H", "L") if c in chains]
                if not ordered:
                    raise RuntimeError(f"foldseek chains: {list(chains)}")
                got = sum(len(a) for a, _ in ordered)
                if got != n_aa:
                    raise RuntimeError(f"len mismatch {got} vs {n_aa}")
                result_q.put((seq_id, ordered, None))
            except Exception as e:  # noqa: BLE001 - logged and skipped
                result_q.put((seq_id, None, f"{type(e).__name__}: {e}"))
            finally:
                if os.path.exists(pdb):
                    os.remove(pdb)


def extract(cache_dir, model_dir, limit, encode_batch, workers, num_models,
            per_residue=False):
    man = pd.read_csv("extraction/cache_antiberty/manifest.csv")
    man.seq_light = man.seq_light.fillna("")
    os.makedirs(os.path.join(cache_dir, "ab"), exist_ok=True)
    index_path = os.path.join(cache_dir, "ab", "index.csv")
    fail_path = os.path.join(cache_dir, "ab", "failed.csv")
    done = set()
    if os.path.exists(index_path):
        done = set(pd.read_csv(index_path, usecols=["seq_id"]).seq_id)
    todo = man[~man.seq_id.isin(done)].reset_index(drop=True)
    if limit:
        todo = todo.head(limit)
    if len(todo) == 0:
        print("nothing to do")
        return
    print(f"{len(todo)} to extract ({len(done)} cached), {workers} workers",
          flush=True)

    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    task_q = ctx.Queue(maxsize=workers * 8)
    result_q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(task_q, result_q, num_models),
                         daemon=True)
             for _ in range(workers)]
    for p in procs:
        p.start()

    enc = SaProtEncoder(model_dir)
    index_file = open(index_path, "a")
    if os.path.getsize(index_path) == 0:
        index_file.write("seq_id,path\n")
    fail_file = open(fail_path, "a")
    if os.path.getsize(fail_path) == 0:
        fail_file.write("seq_id,error\n")

    n_tasks = len(todo)
    feeder_done = False
    n_sent = 0

    def feed():
        nonlocal n_sent, feeder_done
        while n_sent < n_tasks and task_q.qsize() < workers * 6:
            r = todo.iloc[n_sent]
            task_q.put((r.seq_id, r.seq_heavy, r.seq_light,
                        int(r.len_heavy) + int(r.len_light)))
            n_sent += 1
        if n_sent >= n_tasks and not feeder_done:
            for _ in range(workers):
                task_q.put(None)
            feeder_done = True

    def encode_and_store(pending):
        """pending: {seq_id: chains}; SaProt-batch them, save npy + index."""
        flat, owner = [], []
        for sid, chains in pending.items():
            for ch in chains:
                flat.append(ch)
                owner.append(sid)
        vecs = {}
        per = max(1, 16384 // 1500)
        hs = []
        for j in range(0, len(flat), per):
            hs.extend(enc.hidden(flat[j: j + per]))
        for sid, h in zip(owner, hs):
            vecs.setdefault(sid, []).append(h)
        for sid, parts in vecs.items():
            h = np.concatenate(parts, axis=0)
            if per_residue:
                vec = h.astype(np.float16)  # [L,1280], VH+VL residue order
                _save_3di(cache_dir, sid, pending[sid])
            else:
                vec = h.mean(axis=0).astype(np.float32)
            rel = _save_npy(cache_dir, sid, vec)
            index_file.write(f"{sid},{rel}\n")
        index_file.flush()

    feed()
    n_done, n_fail, n_got, t0 = 0, 0, 0, time.time()
    pending = {}
    while n_got < n_tasks:
        try:
            seq_id, chains, err = result_q.get(timeout=300)
        except Exception:
            print("result queue timeout; workers alive:",
                  [p.is_alive() for p in procs], flush=True)
            break
        n_got += 1
        if err is not None:
            fail_file.write(f"{seq_id},{err}\n")
            n_fail += 1
        else:
            pending[seq_id] = chains
            n_done += 1
        if len(pending) >= encode_batch:
            encode_and_store(pending)
            pending = {}
            fail_file.flush()
            rate = n_got / (time.time() - t0)
            eta = (n_tasks - n_got) / rate if rate > 0 else float("inf")
            print(f"{n_done} ok + {n_fail} fail / {n_tasks}  "
                  f"{rate:.2f}/s  eta {eta/3600:.1f} h", flush=True)
        if not feeder_done:
            feed()
    if pending:
        encode_and_store(pending)
    index_file.close()
    fail_file.close()
    for p in procs:
        p.join(timeout=30)
    print("done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="extraction/cache_struct_saprot")
    ap.add_argument("--model-dir", default=SAPROT_DIR)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--encode-batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--num-models", type=int, default=1)
    ap.add_argument("--per-residue", action="store_true",
                    help="save fp16 [L,1280] per-residue embeddings (plus the "
                         "3Di tokens under <cache>/3di/) instead of the "
                         "mean-pooled fp32 [1280] vector")
    args = ap.parse_args()
    extract(args.cache_dir, args.model_dir, args.limit, args.encode_batch,
            args.workers, args.num_models, per_residue=args.per_residue)


if __name__ == "__main__":
    main()
