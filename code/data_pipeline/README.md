# Data Pipeline — Antibody–Antigen Affinity Ranking

> **Status (2026-07-25): UNIFIED.** The former v1 (`data_pipeline/` stages 01–04)
> and v2 (`data_pipeline_v2/` stages 01–05) pipelines are fused into this single
> 7-stage pipeline. v2 content is authoritative throughout; v1 survives as the
> stage-1/3 engines it already was for v2. The retired v1 harmonization script,
> v1 outputs, and the old `data_pipeline_v2/` directory are under `retired/`
> (see `retired/README.md`). Implements `preparation/dataset-loading-filtering.md`
> plus the gap fixes from `preparation/data-pipeline-research.md` §6
> (V-domain trimming, sign audit, operator censoring, clustering,
> leakage-safe splits, sampling config).

Processes the raw sequence data in
`data/raw/sequences/` (22 folders, 83 CSVs) into the
ranking-first training corpus the Plan v3 model consumes.

All scripts run from the **repo root** with the project venv:

```bash
.venv/bin/python data_pipeline/01_schema_normalization.py      # raw CSVs -> normalized
.venv/bin/python data_pipeline/02_enrich_trim.py               # + AbRank cols, V-domain trim
.venv/bin/python data_pipeline/03_cleanup.py                   # QC, flank trim, dedup
.venv/bin/python data_pipeline/04_harmonize.py                 # labels -> fitness, censoring, sign audit
.venv/bin/python data_pipeline/05_clusters_splits.py           # CDR-H3 clusters, CV splits
.venv/bin/python data_pipeline/06_sampling_config.py           # training groups + aux pairs
.venv/bin/python data_pipeline/07_distribution_plots.py        # QC figures
```

Every script has `--input` / `--output`-style arguments with the paths below as
defaults and a `--limit N` option for quick smoke tests. **A `--limit` run
overwrites the default outputs — always point smoke tests at a scratch path.**
No parquet engine is installed in the venv, so intermediate tables are
`.csv.gz`.

## Stages

### 01 — `01_schema_normalization.py`
Walks the data root, and for each CSV:

- sniffs encoding (BOM-safe, `utf-8-sig`) and skips preamble lines until the
  true header (`3/li2023machine_*_affinity2.csv` carries a 6-line MIT license
  banner that is skipped this way);
- normalizes the schema: `seq_heavy` = first of `heavy` / `HC` /
  `Ab_heavy_chain_seq`; `seq_light` = first of `light` / `LC` /
  `Ab_light_chain_seq`; `antigen` = `Antigen` / `Ag_name` / `Target` column
  else `single`; `format` = `format` column else `VHH` when no light column
  exists else `scFv`;
- captures `antigen_seq` when present (kothiwal2025htp) as `ag_seq_raw`
  (routed to `ag_seq` at stage 2);
- chooses one label column per file (spec §3 transform table): `-log…` /
  `neg_log…` columns → as-is; `Pred_affinity` (screening score, rank-only);
  `fitness` (cross-checked against unit-Kd columns — the AbRank
  `fitness = +log10(Kd[nM])` trap is detected here and the unit column used
  instead); Kd in nM; Kd in M; IC50/EC50/ADCC/ELISA → excluded corpus;
- assigns the corpus: `main` | `aux_pairwise` (binary bind/no-bind files:
  tsuruta ×2, kirby binary) | `excluded_ic50_ec50` | `excluded_rank_only`;
- skips with a manifest note: `22/proteinbase_*` (no antibody schema).
  (kothiwal2025htp is **included** since 2026-07-26: `_spr` → main with
  `neg log SPR Kd (M)` as-is, `_ec50` → `excluded_ic50_ec50`.)

**Output:** `output/normalized.csv.gz` + `output/manifest_stage1.json`.

### 02 — `02_enrich_trim.py`
Implements the P0 gaps from research §6:

- **Constant-region trimming.** AbRank (SAbDab-derived) chains carry CH1/CL
  constant regions (median heavy 217 aa) that the stage-3 length filter would
  delete wholesale. Over-long chains are cut right after the framework-4
  anchor (heavy `WGxG`+11, light `FGxG`+10 IMGT offsets; loose fallbacks for
  divergent FW4 and swapped chain columns) *before* length QC.
- **AbRank enrichment.** Re-attaches columns stage 1 dropped from the raw
  AbRank CSV: `Aff_op` (`=`/`>`/`<` — 10% of rows right-censored), `Ag_seq`,
  escape fraction, IC50 [µg/mL], and the paper's 75%-identity cluster ids.
  Join by `row_idx`, validated by heavy-sequence match rate.
- **Label recovery.** Per-row `label_kind` (`kd_nm` > `escape` > `ic50_ugml`)
  so the 247k escape-only/IC50-only AbRank rows survive as rank-only group
  members instead of dying as NaN labels (research §4.5: only within-target
  order of DMS/IC50 data is meaningful).

**Output:** `output/enriched.csv.gz` + `output/manifest_stage2.json`.

### 03 — `03_cleanup.py`
Spec §4 rules 4–6, per file (grouping never crosses files):

- **flank trimming:** signal peptide `MKYLLPTAAAGLLLLAAQPAMA` prefix plus
  leading/trailing `H{6,}` His-tags (~100% of tsuruta VHH rows trimmed);
- **alphabet QC:** `^[ACDEFGHIKLMNPQRSTVWY]+$` on heavy and light — the only
  alphabet safe for both IgLM and ProGen2 (asserted against
  `model_data/iglm_weights/vocab.txt` at runtime);
- **length QC (after trimming):** heavy 90–160 aa, light 90–130 aa;
- **missing light:** dropped unless `format == VHH`;
- **dedup** within (`source_file` × `antigen`) on `(heavy, light)`:
  identical → one copy; conflicting labels → median.

**Output:** `output/cleaned.csv.gz` + `output/manifest_stage3.json`.

### 04 — `04_harmonize.py`
Labels → `fitness` (higher = stronger binding), per (file × antigen) group:

- transforms driven by the stage-1 branch: `kd_nm` → `9 − log10(x)`;
  `kd_m` → `−log10(x)` with nM-miscuration detection (`unit_corrected`,
  catches kirby2024); `neglog_asis` / `fitness_range` with median-range
  sanity checks; `pred_affinity_rankonly` → as-is;
- **AbRank per-row kinds:** `kd_nm` → `9 − log10(Kd)`; `escape` → `−escape`;
  `ic50_ugml` → `−log10(IC50)`. Each group keeps a single dominant kind
  (mixed-kind rows dropped and counted); non-Kd groups are flagged
  `rank_only` and must never be pooled with Kd groups;
- **censoring:** modal fitness covering > 30% of a group → censored
  (phillips2021 cr9114: ~88% at fitness = 6.0), plus operator-aware censoring
  from `aff_op` (`>` → `weaker_than`, `<` → `stronger_than`, with
  `censor_dir`);
- **sign audit:** Spearman(fitness, raw label) must match the branch's
  expected sign — guards against training an inverted ranker;
- **per-group normalization:** `fitness_z` and `fitness_rank_pct` within each
  group — never pooled across groups.

`aux_pairwise` rows pass through cleaned but un-harmonized (binary 0/1).
Excluded corpora never reach this stage.

**Output:** `output/harmonized.csv.gz` (main), `output/harmonized_aux.csv.gz`
(aux), `output/group_stats.csv`, `output/manifest_stage4.json`.

### 05 — `05_clusters_splits.py`
Spec rule 7 (leakage control), following research §4.4:

- AbRank rows: the paper's own 75%-identity Lev3 cluster ids;
- all other small main-corpus files: CDR-H3 extracted by FW3/FW4-anchor
  regex, clustered **globally** at 90% length-matched identity (greedy
  Hamming within length bins), so parents shared across files (hie2023
  families, phillips cr9114/cr6261) land in one cluster and become visible
  as cross-file leakage flags;
- li2023/engelhart-scale files (>60k rows): file-level pseudo clusters
  (single-antigen screening sets handled by leave-one-file-out CV);
- li2023 affinity1/affinity2 overlap flagged by exact (heavy, light)
  cross-file match → `cross_file_dup` on the affinity2 rows.

**Output:** `output/harmonized_clustered.csv.gz` (+ `cluster_id`,
`cross_file_dup`), `output/splits.json` (leave-one-file-out folds + AbRank
cluster-stratified 5-fold + leakage flags), `output/manifest_stage5.json`.

### 06 — `06_sampling_config.py`
Benchmark-format training config (spec §6, AbRank §4.3):

- `training_groups.csv` — one row per (file × antigen) group: `eligible`
  (≥ 20 non-censored rows), `weight` = min(N, 2000) sampling weight,
  `subsample_n` = min(N, 50,000) (label-decile stratification advised above
  the cap), `m_confident_pairs` (|Δfitness| ≥ 1.0 = 10× AbRank margin,
  ≤ 2000-row subsample, capped at 1000/group);
- `aux_pairs.csv` — binder/non-binder pairs for the aux pairwise margin loss:
  all positives × up to 3 negatives, ≤ 100k pairs/file (tsuruta is 96%
  negative).

**Output:** `output/training_groups.csv`, `output/aux_pairs.csv`,
`output/manifest_stage6.json`.

### 07 — `07_distribution_plots.py`
QC figures from the final corpus into `output/plots/`: summary dashboard,
group sizes (N ≥ 20 threshold annotated), per-file fitness raw vs z-scored,
length distributions by format, censored fractions, corpus composition.

## Measured effect of the stage-2/4 fixes (full runs, 2026-07-25; refreshed 2026-07-26)

- 2026-07-26 refresh: kothiwal2025htp included (+364 SPR rows, +11 groups,
  9 training-eligible, per-row `antigen_seq`); totals 641,090 → **641,454**
  main-corpus rows, 5,686 → **5,697** antigen groups (the earlier "5,634"
  counted AbRank only), 2,274 eligible training groups (2,239 with antigen
  coverage). Row delta = exactly the 364 kothiwal rows (deterministic stages).

- AbRank rows surviving cleanup: 326,403 (former v1) → **336,460**; length
  drops 11,707 → 1,416 (the constant-region carriers).
- AbRank rows in the harmonized main corpus: 93,500 (v1) → **280,463**
  (escape + IC50 rows recovered as rank-only groups); antigen groups
  2,139 → **5,634**.
- Sign audit: all files pass (AbRank per-kind ρ = −1.0 vs raw labels).
- Total main corpus: 573,635 (v1) → **641,090** rows.

## Caveats / deliberate simplifications

- No ANARCI in the venv: CDR-H3 extraction (`05`) and V-domain trimming
  (`02`) are FW3/FW4-anchor heuristics, not IMGT numbering. 29/342k AbRank
  heavy chains have FW4 too divergent for any anchor and are dropped by
  length QC.
- AbRank rows whose only affinity information is a bare `>`/`<` operator
  with no numeric value (~34k rows) remain unusable and are dropped at
  stage 4.
- `escape` fitness is `−escape_fraction` and `ic50_ugml` is `−log10(IC50)`;
  both are only meaningful as within-group order — the groups are flagged
  `rank_only` and must feed ranking losses, never value regression.
- li2023/engelhart-scale files (>60k rows) are not sequence-clustered; they
  are single-antigen screening sets handled by leave-one-file-out CV.
- `harmonized_aux.csv.gz` rows are not clustered (binary aux loss,
  file-level holdout only).
- `Pred_affinity` groups (li2023, engelhart) are rank-only and must never be
  merged with Kd groups.

## Output layout

```
data_pipeline/
├── 01_schema_normalization.py … 07_distribution_plots.py
├── README.md                      # this file
└── output/
    ├── normalized.csv.gz          # stage 1 long table
    ├── manifest_stage1.json
    ├── enriched.csv.gz            # stage 2 (+ aff_op, ag_seq, clusters, label_kind)
    ├── manifest_stage2.json
    ├── cleaned.csv.gz             # stage 3 QC'd
    ├── manifest_stage3.json
    ├── harmonized.csv.gz          # stage 4 main corpus (+ censor_dir, rank_only)
    ├── harmonized_aux.csv.gz      # stage 4 aux pairwise (binary)
    ├── group_stats.csv            # stage 4 per-(file × antigen) stats
    ├── manifest_stage4.json
    ├── harmonized_clustered.csv.gz  # stage 5 (+ cluster_id, cross_file_dup)
    ├── splits.json                # stage 5 CV folds + leakage flags
    ├── manifest_stage5.json
    ├── training_groups.csv        # stage 6 per-group training config
    ├── aux_pairs.csv              # stage 6 sampled aux pairs
    ├── manifest_stage6.json
    ├── plots/                     # stage 7 figures
    └── stage{1..6}_full.log       # run logs (see naming note below)
```

### Naming note (2026-07-25 unification)

Outputs were produced under the former v1/v2 layout and renamed, not
re-generated; content is byte-identical to the measured-effect numbers above.
Mapping: v2 `normalized_v2` → `enriched`; v2 `cleaned_v2`/`harmonized_v2`/
`harmonized_aux_v2`/`group_stats_v2` → same names without `_v2`;
`harmonized_clustered_v2` → `harmonized_clustered`; `splits_v2.json` →
`splits.json`; `training_groups_v2.csv` → `training_groups.csv`;
`aux_pairs_v2.csv` → `aux_pairs.csv`; `manifest_v2_stageN` →
`manifest_stageN+1`. The `stage1–4_full.log` files are the original run logs —
paths mentioned inside them refer to the old layout; `stage5/6_full.log` are
from the 2026-07-25 re-run under the unified layout (outputs verified
byte-identical to the moved originals). Former v1 outputs (superseded) and
the v1 `03_label_harmonization.py` are in `retired/data_pipeline_v1/`.
