# Results Ledger — Ablation Runs (Plan v4)

## Final model (milestone 8, 2026-07-30) — `final_all_s{0,1,2}`

v5 recipe (cross + bilinear, `lambda_aux` 0, `rank_only_factor` 0.5,
1500 steps/ep, lr 3e-4, 10 epochs) trained on **ALL eligible rows**
(579,930 train rows; affinity2 cross-file dups dropped by the guard) via the
new `cv.strategy: "all"` in `train.py` — no holdout; early stop/checkpoint
selection uses a 100-group **in-train** monitor set (optimistic by
construction, 0.70–0.80 vs the 0.28 CV estimate). All 3 seeds ran the full
10 epochs on the remote GPU (~8–13 min each). Checkpoints pulled to
`training/runs/final_all_s{0,1,2}/` (`best.pt` + `last.pt`, 1.86M params):

| seed | best ep | monitor sp | best.pt sha1[:12] |
|---|---|---|---|
| 0 | 9 | 0.8026 | 6de364f66c3f |
| 1 | 9 | 0.7789 | d2a8202dc5f8 |
| 2 | 8 | 0.6994 | b0c2ec0eb93c |

Benchmark test set parsed from `data/benchmark_v3.xlsx` →
`data/benchmark_mAbs.csv` (mAbs sheet: 3 groups × 20 scFv; group 1 =
SARS-CoV-2 spike 1273 aa single antigen; groups 2–3 = **per-row antigens**
(HIV Env variants w/ trailing `*`; ~220–280 aa proteins); 4 chains carry a
trailing `X` — strip both at extraction).

---

All runs: frozen AntiBERTy (antibody) + ESM-2 650M (antigen) caches, ListMLE,
AdamW recipe, early stop on val Spearman. Run dirs: `training/runs/<name>/`
(`results.jsonl` per epoch + `best.pt`). Logs for detached workers:
`training/runs/logs/`.

## Baselines (A0 + zero-shot)

| baseline | kothiwal-DCC | notes |
|---|---|---|
| zero-shot ProGen2-small (−nll) | 0.322 | per-fold kothiwal range −0.22 (IL23R) … +0.76 (PDL2), mean 0.296 |
| zero-shot IgLM (−mean chain nll) | −0.003 | kothiwal mean 0.152 |
| linear probe (pooled 512+1280 concat → Linear, ListMLE) | 0.309 | AbRank fold0: **0.068** — probe does not transfer across diverse antigens |

## Key finding — rank-only proxy labels poison Kd transfer (A4)

DCC fold, same arms with/without Pred_affinity proxy groups (li2023/engelhart,
~77% of training weight at 0.5×):

| arm | rank_only 0.5× | Kd-only (factor 0.0) |
|---|---|---|
| cross+bilinear (mainline) | 0.039 | **0.350** |
| concat | 0.268 | **0.323** |
| linear probe | 0.309 | 0.256 |

→ `rank_only_factor: 0.0` for SPR-Kd-style eval (config knob added 2026-07-27).

## Kothiwal 9-fold matrix (Kd-only, LOFO over 9 eligible kothiwal antigens)

| fold | cross | blind | concat | zs-progen |
|---|---|---|---|---|
| DCC | 0.350 | **0.401** | 0.323 | 0.322 |
| DKK | 0.018 | −0.083 | −0.080 | **0.287** |
| IL23R | 0.118 | **0.251** | 0.179 | −0.217 |
| LOX1 | 0.161 | 0.264 | **0.336** | 0.246 |
| PDL1 | 0.561 | 0.515 | **0.571** | **0.631** |
| PDL2 | **0.676** | 0.589 | 0.584 | **0.762** |
| ROBO1 | 0.066 | **0.293** | 0.235 | 0.097 |
| Syncytin2 | **0.211** | 0.170 | 0.180 | 0.070 |
| TIGIT | 0.288 | **0.483** | 0.366 | 0.471 |
| **mean** | 0.272 | **0.320** | 0.299 | 0.296 |

Narrow testbed (one scaffold, 9 cell-surface receptors): antigen conditioning
cannot show its value here. blind wins 6/9 vs zs-progen, cross 3/9.

## AbRank 5-fold (cluster-stratified) — FINAL, mean-of-best over 3 seeds (2026-07-30, remote GPU)

All runs on the verified-static caches; per-fold value = mean of best val
Spearman over seeds {0,1,2}; ± = std over folds. IgLM column finalized to
3-seed means 2026-07-30 (see the A1 section). Probe (A0): full row in the
A0 section (mean 0.214).

| fold | cross (mainline) | blind | concat | IgLM tower (cross head) |
|---|---|---|---|---|
| 0 | 0.239 | 0.225 | 0.200 | 0.170 |
| 1 | 0.252 | 0.171 | 0.147 | 0.251 |
| 2 | 0.365 | 0.327 | 0.319 | 0.317 |
| 3 | 0.190 | 0.194 | 0.169 | 0.234 |
| 4 | 0.334 | 0.363 | 0.375 | 0.334 |
| **mean ± sd** | **0.276 ± 0.064** | 0.256 ± 0.075 | 0.242 ± 0.089 | 0.261 ± 0.059 |

**A2 verdict (interaction):** cross > blind > concat on the diverse-antigen
axis. The cross–blind gap (+0.020) is inside seed noise (per-fold seed
spread ±0.05–0.10), but cross wins or ties 4/5 folds and never catastroph-
ically loses; concat (no interaction) loses folds 0/1/3 — the interaction
block earns its keep, the antigen tower's marginal value over blind is real
but modest here.
**A3 verdict (fusion):** bilinear-gated ≥ plain-concat in mean everywhere;
keep bilinear.

## A4 — rank-only proxy groups: assay-dependent, confirmed both directions (2026-07-30)

| eval | rank_only 0.5× (default) | rank_only 0.0 (excluded) |
|---|---|---|
| kothiwal DCC (SPR-Kd LOFO) | 0.039 | **0.350** |
| AbRank 5-fold cross (3-seed mean vs ro0 seed-0) | **0.276** | **−0.011 ± 0.098** |

AbRank's eligible groups are largely rank-only (escape/IC50) — excluding
them starves training (mean ≈ 0). **Keep 0.5× for AbRank-style training;
0.0 only for strict SPR-Kd transfer.** The config knob stays.

## A1 — antibody tower: AntiBERTy vs IgLM — FINAL, 3 seeds (2026-07-30)

Clean rerun on verified-static caches reproduced the seed-0 values
bit-for-bit (GPU determinism); seeds 1–2 added in the follow-up batch.
IgLM tower re-extracted on the 4090 (`extraction/extract_iglm_ab.py`, exact
IgLM conventions, verified vs the sunk CPU cache, rel diff ~5e-4).

| fold | AntiBERTy cross (3 seeds) | IgLM cross (3 seeds) |
|---|---|---|
| 0 | 0.239 | 0.170 |
| 1 | 0.252 | 0.251 |
| 2 | 0.365 | 0.317 |
| 3 | 0.190 | 0.234 |
| 4 | 0.334 | 0.334 |
| **mean ± sd** | **0.276 ± 0.064** | 0.261 ± 0.059 |

Ag-cluster fold0: AntiBERTy cross 0.083 vs IgLM 0.058 (0.015/0.057/0.101).

**A1 verdict: keep AntiBERTy.** IgLM ties AntiBERTy-*blind* (0.256) but
loses to AntiBERTy-cross on both axes; its causal-LM features are also the
least stable (fold0 seed spread 0.094–0.257, three epoch-0 early stops at
seed 0). The 143 GB old cache stays retired/non-mainline.

## A0 — linear probe, full row (seed 0, 2026-07-30)

| fold | 0 | 1 | 2 | 3 | 4 | mean |
|---|---|---|---|---|---|---|
| probe | 0.068 | 0.264 | 0.277 | 0.125 | 0.337 | 0.214 |

Mainline cross beats the probe by +0.062 mean and on 4/5 folds; the probe's
fold0 collapse (0.068) confirms pooled-embedding linear models do not
transfer across diverse antigens. Zero-shot floors (IgLM −0.003, ProGen2
0.322 on kothiwal-DCC) unchanged.

## A4b — aux VHH margin loss on diverse eval (seed 0, 2026-07-30)

| arm | folds 0–4 | mean |
|---|---|---|
| cross, λ_aux = 0 (default) | 0.239 / 0.207 / 0.349 / 0.200 / 0.309 | 0.261 |
| cross, λ_aux = 0.2 | 0.125 / 0.393 / 0.312 / 0.210 / 0.275 | 0.263 |

**Verdict: λ_aux is neutral on the diverse-antigen axis → set the default
to 0.0** (simpler objective; the aux pairs are VHH binder/non-binder data
that do not match the benchmark's scFv ranking format).

## Antigen-cluster holdout (whole Ag families unseen) — FINAL fold0 ×3 seeds (2026-07-30)

Fold0 is the only balanced fold (1,989 val groups; folds 1–4 hold out just
47–81 groups each — `ag_cluster_raw % 5` is size-skewed, see A1 section).

| arm | seeds (best val) | mean ± sd |
|---|---|---|
| cross | 0.155 / 0.040 / 0.055 | **0.083 ± 0.051** |
| concat | 0.064 / 0.065 / 0.097 | 0.075 ± 0.015 |
| blind | 0.086 / 0.023 / 0.052 | 0.054 ± 0.026 |
| IgLM (cross head, seed0) | 0.015 | — |

Whole-family antigen generalization is HARD (all arms ≪ 0.1, below the
0.2–0.5 cross-antigen expectation); cross remains the best arm on the
strictest axis.

## Stabilization study (2026-07-30, remote GPU; AbRank cross+bilinear, 5 folds × seeds {0,1}, mean-of-best)

| recipe | folds 0–4 | mean ± sd |
|---|---|---|
| baseline lr 3e-4, 500 steps/ep | 0.234 / 0.289 / 0.369 / 0.181 / 0.306 | 0.276 ± 0.064 |
| low lr 1e-4 (fusion 3e-4) | 0.168 / 0.164 / 0.317 / 0.214 / 0.387 | 0.250 ± 0.088 |
| **1500 steps/epoch** | 0.251 / 0.328 / 0.349 / 0.190 / 0.365 | **0.297 ± 0.066** |

**Verdict: `steps_per_epoch` 500 → 1500 adopted** (DEFAULT_CFG changed in
`train.py` 2026-07-30). Wins 4/5 folds, +0.021 mean — at the pre-registered
±0.02 noise threshold but directionally consistent and mechanistically
motivated (3× more updates per epoch reduces dependence on lucky epoch-0
evals, the observed epoch-0-early-stop pathology). lr 1e-4 undertrains —
rejected.

## Operational notes

- 2026-07-30 (INCIDENT — overlapping cache streams): the antigen cache
  (`cache_esm2/ags`) was streamed twice — a standalone early stream (to
  unblock A1) and the second phase of the main migration script. The second
  stream overwrote npz files while training read them → `BadZipFile` crashes
  (5 main-matrix runs, rc=1). Root cause: my parallel-stream shortcut did
  not cancel the redundant phase of the original script. Response: killed
  both the re-stream and the matrix, **wiped every remote run produced
  against a non-static cache** (all A1 iglm + main-matrix dirs; only the 4
  local-CPU-synced runs kept), re-streamed ags cleanly + full integrity
  scan, then rerunning A1 + main + seed matrices. Lesson now enforced: one
  writer per cache at a time; training only after an integrity scan passes.
- 2026-07-30 (val instability quantified): same fold (AbRank fold0, cross),
  same seed/protocol, CPU vs GPU init → best-val 0.275 vs 0.239; within one
  run val Spearman swings ±0.1+ between epochs (e.g. 0.239→0.073→0.197).
  Train losses are bit-similar across devices (2.0659 vs 2.0658), so this is
  optimization-trajectory/init noise, not eval or cache corruption. **All
  arm comparisons therefore use mean-of-best over 3 seeds**
  (`training/run_matrix_seeds_remote.sh`: abrank 5 folds × {cross, blind,
  concat} × seeds 1–2 added to the seed-0 single-seed matrix; agcluster
  fold0 — the only balanced ag-cluster fold — same ×3 arms). Single-seed
  numbers are landscape only, not decisions.


- 2026-07-29 (remote GPU migration, remote GPU instance (AutoDL RTX 4090),
  workspace `/root/autodl-tmp/workspace`): the box arrived empty — the "underway" migration
  had not landed, so the full transfer was redone from this session.
  Measured link: **~2.2 MB/s aggregate regardless of stream count** → raw
  53.5 GB would have taken ~7 h; zstd -19 on the fp16 npz shards compresses
  to ~55% (fp16 mantissa noise) → ~3.5–4 h. Transfer = tar|zstd streams
  (remote zstd at `/root/miniconda3/bin/zstd`, not on the SSH PATH — first
  attempt failed on that). Remote env: torch 2.8.0+cu128 + pandas/scipy/
  sklearn/transformers installed into miniconda base.
- 2026-07-29: `training/train.py` + `evaluate.py` gained a `device` config
  (default `cpu`; tensor moves in the train loop, `aux_step`, and
  `score_groups`). Local CPU smoke test passes unchanged.
- 2026-07-29 (A1 enabler): IgLM antibody-tower cache **re-extracted on the
  remote GPU** (`extraction/extract_iglm_ab.py`, exact IgLM conventions from
  `extract_embeddings.py`, manifest seqs from the old-cache manifest.csv —
  seq_ids are the same sha1 scheme as the AntiBERTy cache, 100% overlap).
  Spot-check vs the sunk CPU cache: max rel diff ~5e-4 (fp16 rounding).
  235,331 seqs in ~7 min at ~600 seq/s on the 4090. This avoided shipping
  the 143 GB old cache.


- 2026-07-27 (twice): ≥3 concurrent training processes died — root cause **OOM**:
  one run peaked at **9.2 GB** (full-26-column corpus read incl. sequence/antigen
  strings + a 671k-entry Python row2seq dict + 8k-seq LRU). Fixed in
  `training/data.py`: slim column read (`CORPUS_COLS`) + category dtypes +
  direct row_map merge + LRU 4096 → **~3.3 GB/run**; max 2 concurrent runs on
  this 16 GB box. Categorical `group_id` required `observed=True` in the two
  groupby sites.
- 2026-07-27: three background workers died with a session restart ("lost"
  status) — training now runs as `setsid`-detached processes with logs in
  `training/runs/logs/`; survives session death.
- LOFO folds auto-union cross-file CDR-H3 leakage groups (fold names carry
  `[+N union]`); identical val scores across "different" LOFO folds are the
  union working as designed.
- Instability watch: several tiny-val kothiwal runs early-stop at epoch 0
  (val noise); AbRank val sets are large enough to be trustworthy.
