# training/ — antigen-conditioned affinity-ranking harness

Plan v4 model (see `preparation/`, `Kimi_Agent_抗体亲和力排序改进/update_architecture.txt`).
Frozen AntiBERTy (512-d) and ESM-2 (1280-d) embeddings are read from the
extraction caches; only the adapter/interaction/fusion/head blocks from the
`tools/attention.py` registry are trained. Benchmark metric: mean per-group
Spearman.

## Files

- `data.py` — cache readers (`AntibodyCache`, `AntigenCache`; lazy per-sequence
  npz loading + in-memory LRU, tolerant of still-running extraction),
  corpus loading, the ListMLE `ListSampler` (group ∝ weight, rank_only at
  0.5×, ≤20 rows/list, ≤25% censored), padding/collate, token-budget batching.
- `model.py` — `AffinityRanker(config)`. Ablation flags:
  `interaction: cross|none`, `fusion: bilinear|concat`,
  `antigen_blind: true|false`. Prints the trainable parameter count at build.
  `build_param_groups()` implements the optimizer split (fusion/gate params at
  higher lr, no weight decay; weight matrices wd 0.01; biases/norms none).
- `losses.py` — `listmle_loss` (Plackett-Luce NLL; `make_target_order` does
  tie-jitter + pins censored rows to the bottom), `aux_margin_loss` (hinge,
  margin 1.0).
- `evaluate.py` — mean per-group Spearman; scores all rows of each held-out
  group in token-budget batches, antigen loaded once per group.
- `train.py` — trains one CV fold, writes `<output_dir>/results.jsonl`
  (one line per epoch) and `best.pt` (best val Spearman; early stopping on
  Spearman, not loss).
- `probe.py` — A0 baselines: `--mode linear` (mean-pooled [h_ab‖h_ag] →
  Linear, same ListMLE protocol) and `--mode zeroshot` (IgLM / ProGen2 NLL
  from `extraction/cache/index.csv`, per-group Spearman on the same folds).
- `smoke_test.py` — synthetic-cache ablation sweep + real-cache check.
- `configs/` — example configs.

Null antigen: groups missing from `extraction/cache_esm2/antigen_index.csv`
use a fixed zero `[1,1280]` pseudo-antigen (mask all-true, length 1).

## One ablation run (single LOFO fold)

```bash
.venv/bin/python training/train.py --config training/configs/lofo_ablation.json
```

Key config fields (all have defaults in `train.py:DEFAULT_CFG`; the JSON only
needs what you override):

```json
{
  "model": {"interaction": "cross", "fusion": "bilinear", "antigen_blind": false},
  "lambda_aux": 0.0,                // aux margin-loss weight (0 disables).
                                    // Finding (2026-07-30, A4b): neutral on the
                                    // diverse-antigen axis (0.263 vs 0.261) — off by default
  "rank_only_factor": 0.5,          // sampling weight multiplier for rank_only groups;
                                    // 0.0 EXCLUDES them. Finding (2026-07-27): the
                                    // Pred_affinity proxy sets (li2023/engelhart) poison
                                    // SPR-Kd transfer — use 0.0 for Kd-style evaluation;
                                    // AbRank escape/IC50 rank-only groups tolerate 0.5.
  "epochs": 10, "steps_per_epoch": 500,
  "lr": 3e-4, "fusion_lr": 1e-3,
  "cv": {"strategy": "lofo", "holdout": ["11/kothiwal2025htp_DCC_spr.csv"]},
  "output_dir": "training/runs/lofo_kothiwal_DCC_cross_bilinear",
  "seed": 0, "num_threads": 8
}
```

Ad-hoc overrides without editing the file:

```bash
.venv/bin/python training/train.py --config training/configs/lofo_ablation.json \
    --override '{"model": {"interaction": "none"}, "output_dir": "training/runs/arm_none"}'
```

## CV runs (all three strategies reported separately)

LOFO (57 folds from `splits.json`; hold a small subset for ablation speed —
leakage-linked files from `leakage_flags_cross_file` are held out together
automatically):

```bash
for f in 11/kothiwal2025htp_DCC_spr.csv 13/phillips2021binding_cr9114_h1_kd.csv; do
  .venv/bin/python training/train.py --config training/configs/lofo_ablation.json \
      --override "{\"cv\": {\"strategy\": \"lofo\", \"holdout\": [\"$f\"]}, \
                   \"output_dir\": \"training/runs/lofo_${f//\\//_}\"}"
done
```

AbRank cluster-stratified 5-fold (`int(ab_cluster_raw) % 5`):

```bash
for k in 0 1 2 3 4; do
  .venv/bin/python training/train.py --config training/configs/abrank_fold0.json \
      --override "{\"cv\": {\"strategy\": \"abrank\", \"fold\": $k}, \
                   \"output_dir\": \"training/runs/abrank_fold$k\"}"
done
```

Antigen-cluster holdout (whole `ag_cluster_raw` clusters; NaN-cluster rows are
train-only):

```bash
for k in 0 1 2 3 4; do
  .venv/bin/python training/train.py --config training/configs/lofo_ablation.json \
      --override "{\"cv\": {\"strategy\": \"agcluster\", \"fold\": $k, \"n_folds\": 5}, \
                   \"output_dir\": \"training/runs/agcluster_fold$k\"}"
done
```

## A0 baselines

```bash
# zero-shot IgLM / ProGen2 NLL floors (no training)
.venv/bin/python training/probe.py --config training/configs/lofo_ablation.json \
    --mode zeroshot --cv lofo --max-folds 10
.venv/bin/python training/probe.py --config training/configs/lofo_ablation.json \
    --mode zeroshot --cv abrank
.venv/bin/python training/probe.py --config training/configs/lofo_ablation.json \
    --mode zeroshot --cv agcluster

# linear probe, same fold + ListMLE protocol as the main model
.venv/bin/python training/probe.py --config training/configs/lofo_ablation.json --mode linear
```

## Smoke test

```bash
.venv/bin/python training/smoke_test.py
```

Builds a synthetic cache (50 seqs, 3 antigens, 2 groups) under
`.scratch/train_smoke/`, runs 20 steps + one eval per ablation arm
(cross/none × bilinear/concat, antigen_blind), then — if the real antibody
cache has enough sequences — 50 steps on a small real group (kothiwal2025htp
DCC when cached; otherwise a fallback built from cached aux sequences).
