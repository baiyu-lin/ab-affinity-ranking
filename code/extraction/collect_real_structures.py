#!/usr/bin/env python
"""Collect real experimental structures (Arm 0 validation assets).

Part 1 (antigens): for every unique ag_seq in antigen_table.csv, query the
RCSB PDB search API (sequence search), pick the hit with highest identity and
coverage >= 0.9, download its mmCIF into structures/antigens/, and write
structures/antigens/mapping.csv.

Fallback: if no hit at identity_cutoff 0.9, retry with 0.5 and mark the row
as partial-homology (部分同源).

Part 2 (antibodies): dump the SAbDab2 database via its REST API (the old
sabdab_summary_all.tsv endpoint is dead) and compute the exact-match coverage
of unique training-corpus VH sequences (seq_heavy) against SAbDab heavy
sequences. Writes structures/antibodies/matched.csv. No antibody PDB files
are downloaded.

Re-runnable: already-downloaded files are skipped.

Usage:
    python collect_real_structures.py [--skip-part1] [--skip-part2]
                                      [--corpus PATH]
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ANTIGEN_TABLE = HERE / "antigen_table.csv"
AG_DIR = HERE / "structures" / "antigens"
AB_DIR = HERE / "structures" / "antibodies"
CORPUS = HERE.parent / "data_pipeline" / "output" / "harmonized_clustered.csv.gz"

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
ENTITY_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb}/{ent}"
CIF_URL = "https://files.rcsb.org/download/{pdb}.cif"
SABDAB_URL = "https://sabdab.opig.stats.ox.ac.uk"

TIMEOUT = 30
RETRIES = 3
SLEEP = 0.4          # be gentle with the RCSB API
COVERAGE_MIN = 0.9
MAX_HITS_CHECK = 10  # align at most this many top hits per query

VH_COL_CANDIDATES = ["seq_heavy", "vh", "vh_seq", "vh_sequence",
                     "heavy", "heavy_sequence", "heavy_chain"]


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
def request_with_retry(method, url, **kw):
    kw.setdefault("timeout", TIMEOUT)
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.request(method, url, **kw)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
            if r.status_code in (400, 404):
                break  # not retryable
        except requests.RequestException as e:
            last = str(e)
        time.sleep(1.5 * attempt)
    raise RuntimeError(f"{method} {url} failed after {RETRIES} tries: {last}")


def rcsb_sequence_search(seq, identity_cutoff):
    query = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": 1e-5,
                "identity_cutoff": identity_cutoff,
                "sequence_type": "protein",
                "value": seq,
            },
        },
        "request_options": {
            "scoring_strategy": "sequence",
            "sort": [{"sort_by": "score", "direction": "desc"}],
            "paginate": {"start": 0, "rows": 25},
            "results_content_type": ["experimental"],
        },
        "return_type": "polymer_entity",
    }
    try:
        r = request_with_retry("POST", SEARCH_URL, json=query)
    except RuntimeError as e:
        # 400/204 = no results or query rejected (e.g. peptide too short)
        return [], str(e)
    data = r.json()
    hits = [x["identifier"] for x in data.get("result_set", [])]
    return hits, None


def fetch_entity_sequence(pdb, ent):
    r = request_with_retry("GET", ENTITY_URL.format(pdb=pdb, ent=ent))
    d = r.json()
    seq = d["entity_poly"]["pdbx_seq_one_letter_code_can"]
    return seq.replace("\n", "").replace(" ", "")


# --------------------------------------------------------------------------- #
# Semi-global (overlap) alignment: free end gaps on both sequences.
# Returns (identity, coverage) where
#   identity = matches / aligned_columns,
#   coverage = aligned query residues / len(query).
# --------------------------------------------------------------------------- #
def align_metrics(query, subject):
    m, n = len(query), len(subject)
    if m == 0 or n == 0:
        return 0.0, 0.0
    if query == subject:
        return 1.0, 1.0

    S = np.zeros((m + 1, n + 1), dtype=np.int32)  # free end gaps -> init 0
    T = np.zeros((m + 1, n + 1), dtype=np.uint8)  # 0=diag 1=up 2=left
    qa = np.frombuffer(query.encode(), dtype=np.uint8)
    sa = np.frombuffer(subject.encode(), dtype=np.uint8)

    for d in range(2, m + n + 1):
        i_lo, i_hi = max(1, d - n), min(m, d - 1)
        i = np.arange(i_lo, i_hi + 1)
        j = d - i
        match = np.where(qa[i - 1] == sa[j - 1], 1, -1)
        diag = S[i - 1, j - 1] + match
        up = S[i - 1, j] - 1
        left = S[i, j - 1] - 1
        best = np.maximum(np.maximum(diag, up), left)
        S[i, j] = best
        T[i, j] = np.where(best == diag, 0, np.where(best == up, 1, 2))

    # traceback starts at the best cell on the last row or last column
    last_col = S[:, n]
    last_row = S[m, :]
    if last_col.max() >= last_row.max():
        i, j = int(last_col.argmax()), n
    else:
        i, j = m, int(last_row.argmax())

    matches = aligned = q_cov = 0
    while i > 0 and j > 0:
        t = T[i, j]
        if t == 0:
            aligned += 1
            q_cov += 1
            if query[i - 1] == subject[j - 1]:
                matches += 1
            i -= 1
            j -= 1
        elif t == 1:
            aligned += 1
            q_cov += 1
            i -= 1
        else:
            aligned += 1
            j -= 1
    # leading/trailing unaligned parts are free (end gaps) -> not counted
    identity = matches / aligned if aligned else 0.0
    coverage = q_cov / m
    return identity, coverage


# --------------------------------------------------------------------------- #
# Part 1: antigens
# --------------------------------------------------------------------------- #
def find_best_hit(seq, identity_cutoff):
    """Return dict(pdb_id, entity_id, identity, coverage) or (None, err)."""
    hits, err = rcsb_sequence_search(seq, identity_cutoff)
    time.sleep(SLEEP)
    if not hits:
        return None, err or "no hits"

    best_ok, best_any = None, None
    for ident in hits[:MAX_HITS_CHECK]:
        pdb, ent = ident.split("_")
        try:
            subj = fetch_entity_sequence(pdb, ent)
        except RuntimeError as e:
            print(f"    [warn] entity fetch failed {ident}: {e}")
            continue
        time.sleep(SLEEP)
        identity, coverage = align_metrics(seq, subj)
        rec = {"pdb_id": pdb, "entity_id": ent,
               "identity": round(identity, 4), "coverage": round(coverage, 4)}
        key = (identity, coverage)
        if best_any is None or key > (best_any["identity"], best_any["coverage"]):
            best_any = rec
        if coverage >= COVERAGE_MIN and (
            best_ok is None or key > (best_ok["identity"], best_ok["coverage"])):
            best_ok = rec
    return best_ok or best_any, None


def download_cif(pdb, path):
    if path.exists() and path.stat().st_size > 0:
        return True
    r = request_with_retry("GET", CIF_URL.format(pdb=pdb))
    path.write_bytes(r.content)
    time.sleep(SLEEP)
    return True


def part1():
    AG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ANTIGEN_TABLE)
    df = df[df["ag_seq"].notna() & ~df["ag_seq"].str.startswith("null")]
    df["ag_seq"] = df["ag_seq"].str.strip()

    unique = df["ag_seq"].unique()
    print(f"[part1] {len(df)} valid rows, {len(unique)} unique antigen sequences")

    seq_result = {}  # seq -> (record|None, note)
    for idx, seq in enumerate(unique, 1):
        key = hashlib.sha1(seq.encode()).hexdigest()[:8]
        print(f"[part1] {idx}/{len(unique)} len={len(seq)} id={key}")
        rec, note = find_best_hit(seq, 0.9)
        tier = "exact"
        if rec is None or rec["coverage"] < COVERAGE_MIN or rec["identity"] < 0.9:
            rec2, note2 = find_best_hit(seq, 0.5)
            tier = "partial"
            if rec2 is not None and (
                rec is None
                or (rec2["identity"], rec2["coverage"])
                > (rec["identity"], rec["coverage"])):
                rec, note = rec2, note2
        if rec is None:
            seq_result[seq] = (None, f"未命中: {note}")
            print(f"    -> no hit ({note})")
            continue

        ok = rec["coverage"] >= COVERAGE_MIN and rec["identity"] >= 0.9
        if tier == "partial" and not ok:
            note = "部分同源(低覆盖/低一致性)"
        elif tier == "partial":
            note = "部分同源"
        else:
            note = ""

        cif = AG_DIR / f"{key}_{rec['pdb_id']}.cif"
        try:
            download_cif(rec["pdb_id"], cif)
        except RuntimeError as e:
            note = (note + ";" if note else "") + f"CIF下载失败: {e}"
        seq_result[seq] = (rec, note)
        print(f"    -> {rec['pdb_id']}_{rec['entity_id']} "
              f"identity={rec['identity']} coverage={rec['coverage']} {note}")

    rows = []
    for _, r in df.iterrows():
        rec, note = seq_result[r["ag_seq"]]
        rows.append({
            "group_id": r["group_id"],
            "pdb_id": rec["pdb_id"] if rec else "",
            "identity": rec["identity"] if rec else "",
            "coverage": rec["coverage"] if rec else "",
            "entity_id": rec["entity_id"] if rec else "",
            "备注": note,
        })
    out = AG_DIR / "mapping.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    n_exact = sum(1 for _, nt in seq_result.values() if nt == "")
    n_part = sum(1 for _, nt in seq_result.values() if nt.startswith("部分同源"))
    n_miss = sum(1 for _, nt in seq_result.values() if nt.startswith("未命中"))
    print(f"[part1] mapping -> {out}")
    print(f"[part1] unique seqs: exact={n_exact} partial={n_part} miss={n_miss}")


# --------------------------------------------------------------------------- #
# Part 2: antibodies vs SAbDab
#
# The classic sabdab_summary_all.tsv endpoint is gone (the site is now the
# SAbDab2 SPA and the old URL returns HTML). SAbDab2 exposes a paginated
# REST endpoint /api/antibodies (limit <= 500) whose records embed the
# canonical heavy segment (IMGT numbering with '-' gaps) plus all PDB
# entries realising that antibody. We page through it once, cache the raw
# JSON, and build an exact-match map: VH sequence -> set of PDB ids.
# New-style ids "pdb_00001sjx" are converted back to classic "1SJX".
# --------------------------------------------------------------------------- #
def download_sabdab():
    AB_DIR.mkdir(parents=True, exist_ok=True)
    path = AB_DIR / "sabdab2_antibodies.json"
    if path.exists() and path.stat().st_size > 0:
        print(f"[part2] SAbDab2 dump already present: {path}")
        return path
    limit, offset = 500, 0
    results, total = [], None
    while total is None or offset < total:
        url = f"{SABDAB_URL}/api/antibodies?limit={limit}&offset={offset}"
        d = request_with_retry("GET", url).json()
        total = d["total"]
        results.extend(d["results"])
        offset += limit
        print(f"[part2] SAbDab2 page: {len(results)}/{total}")
        time.sleep(SLEEP)
    path.write_text(json.dumps(results))
    print(f"[part2] SAbDab2 dump -> {path} "
          f"({len(results)} antibodies, {path.stat().st_size/1e6:.1f} MB)")
    return path


def _classic_pdb_id(pid):
    """pdb_00001sjx -> 1SJX (legacy entries are zero-padded 4-char codes)."""
    if pid.startswith("pdb_0000") and len(pid) == 12:
        return pid[8:].upper()
    return pid


def _segment_sequence(seg):
    """Reconstruct the ungapped sequence from an IMGT-numbering segment."""
    if not seg:
        return None
    numbering = seg.get("numbering") or []
    seq = "".join(res for _pos, res in numbering if res != "-")
    return seq or None


def build_sabdab_vh_map(path):
    records = json.loads(Path(path).read_text())
    sabdab_vh = {}
    for rec in records:
        seq = _segment_sequence(rec.get("heavy_like_segment"))
        if not seq:
            continue
        pdbs = {_classic_pdb_id(p["id"]) for p in rec.get("pdb_entries", [])}
        sabdab_vh.setdefault(seq, set()).update(pdbs)
    return sabdab_vh


def part2(corpus_path):
    try:
        sabdab_path = download_sabdab()
    except RuntimeError as e:
        print(f"[part2] SAbDab download failed, skipping: {e}")
        return

    sabdab_vh = build_sabdab_vh_map(sabdab_path)
    print(f"[part2] SAbDab2: {len(sabdab_vh)} unique VH sequences")

    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        print(f"[part2] corpus not found: {corpus_path} -- skipping VH match. "
              f"(expected on the training host; re-run there with --corpus)")
        return

    header = pd.read_csv(corpus_path, nrows=0)
    vh_col = next((c for c in VH_COL_CANDIDATES if c in header.columns), None)
    if vh_col is None:
        print(f"[part2] no VH column found in corpus header "
              f"(tried {VH_COL_CANDIDATES}); columns: {list(header.columns)}")
        return

    vh = pd.read_csv(corpus_path, usecols=[vh_col])[vh_col].dropna()
    unique_vh = vh.drop_duplicates()
    print(f"[part2] corpus: {len(vh):,} rows, {len(unique_vh):,} unique VH "
          f"(column: {vh_col})")

    matched = [(s, ",".join(sorted(sabdab_vh[s])))
               for s in unique_vh if s in sabdab_vh]
    out = AB_DIR / "matched.csv"
    pd.DataFrame(matched, columns=["sequence", "pdb_ids"]).to_csv(out, index=False)
    cov = 100.0 * len(matched) / max(len(unique_vh), 1)
    print(f"[part2] VH exact-match coverage: {len(matched):,}/{len(unique_vh):,} "
          f"= {cov:.2f}%  -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-part1", action="store_true")
    ap.add_argument("--skip-part2", action="store_true")
    ap.add_argument("--corpus", default=str(CORPUS),
                    help="path to harmonized_clustered.csv.gz "
                         "(default: %(default)s)")
    args = ap.parse_args()
    if not args.skip_part1:
        part1()
    if not args.skip_part2:
        part2(args.corpus)


if __name__ == "__main__":
    main()
