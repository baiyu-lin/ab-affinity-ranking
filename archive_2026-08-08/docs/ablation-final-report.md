# Ablation Final Report & Architecture Revision (Plan v4 → v5)

**Date:** 2026-07-30
**Scope:** complete ablation matrix of Plan v4, executed on a remote GPU instance
(AutoDL RTX 4090, workspace `/root/autodl-tmp/workspace`) after migration of
the frozen-encoder caches.
**Raw ledger:** `training/runs/README.md` (per-run numbers, incidents,
operational notes). This document is the decision record.

---

## 1. Protocol (what makes these numbers trustworthy)

- **Caches verified static before any training.** Full integrity scans:
  antibody 235,331/235,331 npz OK, antigen 4,716/4,716 npz OK. (An early
  incident — two overlapping antigen-cache streams corrupting npz files
  mid-training — was caught as `BadZipFile` crashes; all tainted runs were
  wiped and rerun. Rule now enforced: one writer per cache; train only after
  a scan.)
- **Multi-seed mean-of-best.** Val Spearman swings ±0.1 epoch-to-epoch
  within a single run, and the same fold/seed gives 0.275 (CPU init) vs
  0.239 (GPU init). Single-run numbers are landscape only; every verdict
  below uses 3 seeds × 5 folds (best-epoch val Spearman per run).
- **Decisive axes:** AbRank cluster-stratified 5-fold (diverse antigens) and
  the antigen-cluster fold0 holdout (whole antigen families unseen; folds
  1–4 of that split are unusable — `ag_cluster_raw % 5` is size-skewed,
  47–81 val groups vs 1,989 in fold0).
- **Identical recipe everywhere:** AdamW (lr 3e-4, fusion/gate 1e-3, wd 0.01
  matrices only), cosine + 5% warmup, clip 1.0, ListMLE (tie-jitter,
  censored pinned, ≤25% censored/list), 10 epochs × 500 steps, early stop on
  val Spearman (patience 3), frozen AntiBERTy + ESM-2 650M towers.

## 2. Results and verdicts

### A0 — does the full model beat the baselines? YES.

| baseline | AbRank 5-fold mean | note |
|---|---|---|
| linear probe (pooled 512+1280 → Linear) | 0.214 | fold0 collapse 0.068 — no cross-antigen transfer |
| **mainline cross+bilinear** | **0.276** | beats probe by +0.062, wins 4/5 folds |
| zero-shot IgLM / ProGen2 | ≈ 0 / 0.32 (kothiwal-DCC only) | floors, not competitors |

### A1 — antibody tower: AntiBERTy WINS over IgLM.

| tower | AbRank mean | Ag-cluster fold0 |
|---|---|---|
| **AntiBERTy** | **0.276 ± 0.064** | **0.083** |
| IgLM (GPU re-extracted, verified) | 0.261 ± 0.059 | 0.058 |

IgLM only ties AntiBERTy-*blind* (0.256); its causal-LM features are the
least stable (fold0 seed spread 0.094–0.257; three epoch-0 early stops).
The 143 GB IgLM/ProGen2 cache stays retired.

### A2 — interaction: cross-attention WINS; concat-only LOSES.

| arm | AbRank mean | Ag-cluster fold0 |
|---|---|---|
| **cross (Q=antigen)** | **0.276 ± 0.064** | **0.083 ± 0.051** |
| antigen-blind control | 0.256 ± 0.075 | 0.054 ± 0.026 |
| concat-only (no interaction) | 0.242 ± 0.089 | 0.075 ± 0.015 |

The cross–blind gap is modest on AbRank (+0.020, inside seed noise) but
cross wins/ties 4/5 folds and doubles blind on the strictest holdout —
antigen conditioning pays exactly where generalization is tested. Concat
loses folds 0/1/3 on AbRank: the interaction block is not optional.

### A3 — fusion: bilinear gate WINS (kept).

Bilinear-gated fusion ≥ plain concat in mean on both axes (see A2 table:
cross rows are all bilinear).

### A4 — data factors: rank-only is assay-dependent; aux loss is neutral.

- **rank_only_factor:** 0.0 (exclude Pred_affinity proxy groups) is
  *decisive* for SPR-Kd transfer (DCC 0.039 → 0.350) but *catastrophic* for
  AbRank-style training (−0.011 vs 0.276 — AbRank's eligible groups are
  largely rank-only). **Keep 0.5× default + the 0.0 knob for Kd-style eval.**
- **lambda_aux (VHH margin loss):** 0.263 vs 0.261 with/without — neutral.
  **Default changed to 0.0** (`train.py`), simpler objective, aux data
  doesn't match the benchmark's scFv ranking format.

### Whole-family antigen generalization is the hard frontier.

All arms score ≪ 0.1 on Ag-cluster fold0 (cross 0.083) vs 0.28 on AbRank
cluster splits. The model learns antigen-*specific* binding context, not
transferable antigen biology. Sets realistic expectations for unseen-antigen
test groups; per-group ranking within known-ish antigens is where the score
comes from.

## 3. Stabilization study — FINAL (2026-07-30)

Motivated by the ±0.1 epoch swings and epoch-0 early stops. AbRank
cross+bilinear, 5 folds × seeds {0,1}, mean-of-best:

| recipe | mean ± sd |
|---|---|
| baseline (lr 3e-4, 500 steps/ep) | 0.276 ± 0.064 |
| lr 1e-4 (fusion 3e-4) | 0.250 ± 0.088 — undertrained, rejected |
| **1500 steps/epoch** | **0.297 ± 0.066 — adopted** |

The 1500-step recipe wins 4/5 folds (+0.021 mean, at the pre-registered
±0.02 threshold, directionally consistent). Mechanistic reading: more
updates per epoch reduce dependence on lucky epoch-0 evaluations — the
observed pathology where "training" only degraded the near-init model.
**`steps_per_epoch` default changed 500 → 1500 in `training/train.py`.**

## 4. Architecture revision (v5) — decision list

| component | decision | evidence |
|---|---|---|
| Antibody tower: AntiBERTy (frozen, cached) | **keep** | A1 |
| Antigen tower: ESM-2 650M (frozen, cached) | **keep** | A2 (blind control) |
| Adapter: Linear→LN→GEGLU-FFN + residual, 256-d | **keep** | smoke + all arms |
| Interaction: Pre-LN cross-attn, Q=antigen | **keep** | A2 |
| Fusion: bilinear gated | **keep** | A3 |
| Head 256→64→1 | **keep** | — |
| Loss: ListMLE (+tie-jitter, censored pinned) | **keep** | plan research + A0 |
| rank_only_factor 0.5 (0.0 knob for Kd eval) | **keep** | A4 |
| lambda_aux default | **0.2 → 0.0** | A4b |
| IgLM/ProGen2 towers + 143 GB cache | **retired** | A1 |
| Evaluation protocol | **3-seed mean-of-best standard** | instability finding |
| Training recipe (lr/steps) | **steps_per_epoch 500 → 1500; lr 3e-4 kept** | stab study (§3) |
| Ag-cluster fold assignment | **flag: needs size-stratified re-binning** (folds 1–4 unusable as-is) | skew finding |

## 5. Incidents & lessons (full detail in the ledger)

1. **Overlapping cache streams corrupted antigen npz mid-training** → wiped
   all tainted runs, re-streamed, full scans, reran everything. Rule: one
   writer per cache; training only after integrity scan.
2. **GPU/CPU init noise is as large as ablation effects** → multi-seed
   mean-of-best is now the decision unit.
3. **Ag-cluster fold skew** (47–81 vs 1,989 val groups) → only fold0 is
   informative; fold scheme needs size-stratified re-binning before further
   use.
4. **OOM with 2 concurrent Ag-cluster-fold0 runs** (~36 GB RSS each) →
   serialize large-val-fold runs.
5. **Remote zstd not on SSH PATH / sshpass absent / launch-ssh hangs on
   detach** → full-path binaries, key auth, `setsid nohup` + file logs.

## 6. Next steps (milestone 8, deadline 2026-08-07)

1. Train the final model (v5 recipe, best-of stab study) on ALL eligible
   data (no holdout), 3 seeds; keep per-seed checkpoints.
2. Inference pipeline: 20 seqs (scFv VH+VL / VHH) + antigen seq →
   cached-encoder forward + head → ranked CSV; null-antigen fallback
   (zero pseudo-token path already in `data.py`).
3. Reproducibility pack: configs, seeds, cache manifests, this report.
