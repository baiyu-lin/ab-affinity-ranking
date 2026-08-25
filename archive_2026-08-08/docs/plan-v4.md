# Strategy Plan: Antigen-Antibody Affinity Ranking

**Date:** 2026-07-26 (**Plan v4** — antigen-conditioned overhaul per user-mandated redesign, source documents in `Kimi_Agent_抗体亲和力排序改进/`; supersedes Plan v3, whose antibody-only dual-stream design was a **fatal misstep**: the antigen never entered the model. Binding is an extrinsic property — an antibody-only model can only learn per-group antibody-quality priors and cannot generalize across antigens.)
**Project:** antigen-antibody affinity ranking benchmark (mAbs, per-group ranking of 20 antibodies)
**Metric:** Spearman correlation between predicted and ground-truth affinity ranking (per group of 20)
**Hardware constraint (binding):** CPU-only. No GPU for extraction or training.
**Local weights (verified on disk):** `model_data/antiberty/` (25.8M, 512-d, antibody tower), `model_data/esm2-650m/` (650M, 1280-d, antigen tower), `model_data/iglm_weights/` (IgLM 12.9M — ablation arm + nll bypass), `model_data/progen2-small/` (151M, **1024-d** — non-mainline; cache retained). ProGen2-BFD90 (2.7B) retired to `retired/model_data/`.
**Processed dataset (this revision):** `data_pipeline/output/` — 641,454 harmonized main-corpus rows / 5,697 antigen groups = 5,634 AbRank + 52 other + 11 kothiwal (new: kothiwal2025htp SPR, 364 rows, 9 eligible benchmark-format groups with per-row antigen sequences) + aux pairs + leakage-controlled splits.
**Antigen coverage:** 5,645/5,697 groups from the pipeline (`ag_seq`) + `extraction/antigen_table.csv` (55 curated entries: SAbDab-PDB-first, manual-known, null for small molecules/unresolvable) → 4,716 unique antigen sequences; only 20 groups remain null (li2023 AlphaNeg negative controls ×6, fluorescein ×3, mouse_Ly, misc).

**Companion documents:**

- `Kimi_Agent_抗体亲和力排序改进/` — the user-provided analysis report + target architecture (`update_architecture.txt`), source of record for Plan v4
- `preparation/data-pipeline-research.md` — the evidence base (its §6 P2 already flagged antigen conditioning as "the obvious upgrade")
- `data_pipeline/README.md` + `data_pipeline/output/manifest_stage{1..6}.json` — pipeline and provenance
- `extraction/README.md` — encoder conventions, cache layouts, transformers-5.x workarounds, baseline validation
- `extraction/antigen_table_notes.md` — per-group antigen curation provenance
- `tools/activation.py`, `tools/attention.py` + `*_params.json` — registry-driven unit/block sandbox (v4 blocks added; v3 blocks kept for ablation)

---

## Part 0: Grounding — unchanged research facts

See `data-pipeline-research.md`; the five facts from Plan v3 still hold (data curation dominates; scale doesn't systematically help; **ranking loss ≫ regression loss**; property–data match decides — binding is extrinsic and needs the antigen; split choice changes conclusions more than model choice). Plan v4 adds the report's confirmed findings:

6. **Supervised ESM-2 is the strongest sequence backbone** (AbAffinity ablation: Pearson 0.84 / Spearman 0.82; FLAb2 few-shot: general protein-LM embeddings beat antibody-specialized ones). Zero-shot PLM scores on binding are weak (mean ρ≈0.1) and germline-biased — they are baselines/bypass features, not the model.
7. **ListMLE has direct precedent** (AbLWR: ListMLE→MSE ablation collapses FRA 21.28%→3.29%; ALLM-Ab; AbRank).
8. **Dual-tower frozen PLM + cross-attention has ample precedent** (MolTrans, AttABseq, DG-Affinity, AbAgIPA); antigen-as-query is a reasonable simplification.
9. Softmax/sigmoid as *hidden activations* are an information bottleneck — v4 adapters use GEGLU-FFN + LayerNorm + residual (community standard).

---

## Part 1: Model Selection (Plan v4)

| Stream | Model | Params | Hidden dim | Role |
|---|---|---|---|---|
| Antibody specialist | **AntiBERTy** (`model_data/antiberty/`, zrollins HF mirror of the Ruffolo checkpoint, verified) | 25.8M | 512 | Antibody prior; MLM bidirectional encoder (the community-standard embedding usage, unlike causal IgLM). Trained on 558M OAS sequences. |
| Antigen generalist | **ESM-2 650M** (`model_data/esm2-650m/`, official facebook checkpoint) | 650M | 1280 | Antigen representation. Strongest supervised sequence backbone (research fact #6); general protein coverage matches arbitrary antigens (viral glycoproteins, lysozyme, peptides). |

**Frozen, extraction-cached:** both towers are fully frozen; embeddings extracted once to disk (`extraction/cache_antiberty/`, `extraction/cache_esm2/`). Trainable parameters (~1.8M) live only in the adapters, interaction, fusion, and head.

**Ablation arms (sunk assets, data will decide):** IgLM as antibody tower (218k-seq cache exists, encoder validated bit-exact vs official FLAb scores); IgLM/ProGen2 zero-shot NLLs as head-bypass features and baselines; ProGen2-on-antibody cache (131 GB) retained but non-mainline.

**Expected score (report §3, estimates):** same-antigen supervised ranking typical Spearman 0.4–0.7; cross-antigen 0.2–0.5. Floor: zero-shot ρ≈0.1.

---

## Part 2: Architecture (per `update_architecture.txt`, registry-implemented)

```
H_ab = AntiBERTy(VH), AntiBERTy(VL)  concat    # frozen, cached [B, L_ab, 512]
H_ag = ESM-2(Ag_seq)                           # frozen, cached [B, L_ag, 1280]

# Tower adapters (identical parameterization):
P_s = x + Drop(GEGLU-FFN(LN(x))),  x = Linear(d_s -> 256)(H_s)   # P_ab, P_ag [B, L, 256]

# Interaction (Pre-LN, antigen-as-query, 4 heads):
P_ag' = P_ag + Drop(CrossAttn(Q=LN(P_ag), K/V=LN(P_ab)))

# Bilinear gated fusion:
A_pool = AttnPool(P_ag')          # interacted antigen context (256)
B_pool = AttnPool(P_ab)           # raw antibody summary (not washed by antigen)
gate   = sigmoid(W(LN(A_pool) ⊙ LN(B_pool)) + b)
F      = gate ⊙ A_pool + (1-gate) ⊙ B_pool

score  = Linear(64->1)(GELU(Linear(256->64)(F)))   + Dropout
```

- **Trainable:** ≈1.83M params. CPU full-batch training in minutes.
- **Null antigen** (20 groups + any test group with unretrievable antigen): fixed zero pseudo-antigen token; the model degrades to antibody-only gracefully.
- **Registry:** all blocks in `tools/attention_block_params.json` (v4) — ablations are config changes.

---

## Part 3: Input Encoding

- **Antibody scFv (VH+VL):** each chain separately through AntiBERTy (`[CLS] chain [SEP]`, char-level vocab dict — *not* BertTokenizerFast, which maps all tokens to UNK under transformers 5.x), specials stripped, hidden states concatenated heavy→light (positions align 1:1 with the IgLM/ProGen2 caches).
- **VHH:** heavy chain only.
- **Antigen:** ESM-2 tokenizer (`<cls> … <eos>` stripped). Antigens >1000 aa (11 unique) exceed the 1024 context: sliding windows (1000 aa, stride 500) with per-residue averaging.
- Alphabet guard (unchanged): `^[ACDEFGHIKLMNPQRSTVWY]+$`.

---

## Part 4: Data — pipeline outputs (refreshed 2026-07-26)

- `harmonized_clustered.csv.gz` — **641,454 rows / 5,697 groups** (+364 kothiwal SPR rows with per-row `antigen_seq`; `_ec50` files excluded per the IC50/EC50 rule). AbRank constant regions trimmed, label sign-audited, `Aff_op` censoring parsed, `label_kind` per row, rank-only groups flagged.
- `training_groups.csv` — 2,274 eligible groups (≥ 20 non-censored; 2,239 with antigen coverage); weight = min(N, 2000); m-confident pair counts.
- `aux_pairs.csv` — binder/non-binder pairs (tsuruta hIL6 / SARS-CoV-2, kirby); antigens curated in `extraction/antigen_table.csv` (`AUX::` keys).
- `splits.json` — leave-one-file-out folds + AbRank cluster-stratified 5-fold + leakage flags. **Added evaluation axis: antigen-cluster holdout** (`ag_cluster_raw`, AbRank 75% Ag clusters) — the honest test of antigen generalization. Random splits remain banned.

**Loss (unchanged, still right):** ListMLE over size-20 lists + three amendments (tie-jitter; rank-only down-weight 0.5×; aux VHH pairwise margin loss λ 0.1–0.3). Rejected alternatives unchanged (pointwise MSE, pure RankNet, differentiable Spearman).

---

## Part 5: Training Objective, Evaluation & Ablation Plan

**Objective:** ListMLE over groups of 20 + aux pairwise margin loss (λ 0.2 default).
**Optimizer (unchanged):** AdamW — peak lr 3e-4 (gates 1e-3, own param group, no wd), β=(0.9,0.999), wd 0.01 matrices only, cosine 5% warmup, clip 1.0, early stop on **validation Spearman** (not loss).
**Evaluation:** splits.json folds; mean per-group Spearman; report LOFO / AbRank-cluster / antigen-cluster holdout separately; with/without rank-only groups.

**Baselines that must be beaten:** (0) linear probe (pooled AntiBERTy‖ESM-2 concat → Linear, same ListMLE protocol); (1) IgLM zero-shot ppl (validated bit-exact vs official FLAb scores); (2) ProGen2-small zero-shot ppl; (3) naive mean of normalized ppls.

**Ablation matrix (one variable at a time, fixed folds/seeds):**

| # | Question | Arms | Interim result (2026-07-27) |
|---|---|---|---|
| A0 | Do adapters+interaction help at all? | linear probe vs full model | probe strong on narrow sets (0.309 DCC), collapses on diverse antigens (0.068 AbRank fold0 vs 0.275 full model) → full model earns its keep |
| A1 | Antibody encoder | AntiBERTy vs IgLM (cache sunk) | pending |
| A2 | Interaction | cross-attn(Q=Ag) / concat-only / **antigen-blind control** | **context-dependent**: kothiwal 9-fold (narrow): blind 0.320 > concat 0.299 > cross 0.272; AbRank fold0 (diverse): **cross 0.275 > blind 0.204 > concat 0.194** → antigen conditioning pays on diverse-antigen eval |
| A3 | Fusion | bilinear gate vs plain concat | bilinear ≥ concat everywhere so far |
| A4 | Data | rank-only groups in/out × aux VHH loss in/out | **decisive**: Pred_affinity proxy sets poison SPR-Kd transfer (DCC cross 0.039 → 0.350 when excluded); `rank_only_factor: 0.0` for Kd-style eval, AbRank escape/IC50 stay 0.5× |

Order: A0 → A2 (the antigen question) → A1 → A3 → A4. Architecture ablations on a small fixed fold subset; winner re-confirmed on full CV. Full numbers: `training/runs/README.md` (results ledger).

---

## Part 6: Milestones & Risks

| # | Milestone | Status |
|---|---|---|
| 1–3d | FLAb analysis, model selection, IgLM zero-shot, data pipeline v1/v2/unified | ✅ |
| 4 | ProGen2-small download; extraction infra (dedup/batch/fp16 cache); conventions validated | ✅ 2026-07-25 |
| 4b | **Pivot: antigen-conditioned redesign (user report, `Kimi_Agent_抗体亲和力排序改进/`)** | ✅ 2026-07-26 |
| 5a | Pipeline refresh: kothiwal2025htp parsed (+364 rows, 9 eligible groups, per-row `antigen_seq`); stages 1–7 re-run (641,454 rows / 5,697 groups) | ✅ 2026-07-26 |
| 5b | AntiBERTy + ESM-2 650M download/verify | ✅ 2026-07-26 |
| 5c | Antigen table curation (55 entries, SAbDab-PDB-first; 20 groups null) | ✅ 2026-07-26 |
| 5d | AntiBERTy antibody cache (235,331 seqs, 48 GB) + ESM-2 antigen cache (4,716 antigens) | ✅ 2026-07-26 |
| 6 | Training harness (`training/`) + A0 probe/baselines on splits.json folds | ✅ 2026-07-26 (probe 0.309 DCC / 0.068 AbRank fold0; zero-shot floors measured) |
| 7 | Ablation matrix A1–A4 → final architecture | ✅ 2026-07-30 (remote GPU, 3-seed protocol; verdicts: AntiBERTy/cross/bilinear kept, IgLM retired, λ_aux→0, steps/ep→1500; report: `preparation/ablation-final-report.md`; ledger: `training/runs/README.md`) |
| 8 | Full-CV winner; inference pipeline (20 seqs + antigen → ranked CSV; null-antigen fallback); reproducibility pack | ⬜ (deadline 2026-08-07) |

| Risk | Mitigation |
|---|---|
| **12 days to deadline** | extraction first (day 1); A0 by day 3; ablations time-boxed; reproducibility pack scheduled 1–2 days |
| Antigen conditioning adds nothing | A2 antigen-blind control decides with data; concat-only fallback cheap |
| Antigen-cluster leakage inflates CV | antigen-cluster holdout reported alongside LOFO |
| AntiBERTy mirror checkpoint corrupt | smoke-verified; fallback ZYMScott/antiberty, then AbLang2 |
| ESM-2 650M CPU throughput | only 4,716 antigens, avg ~430 aa; 11 windowed |
| Germline-bias shortcut | per-group CV + antigen-cluster holdout exposes it; zero-shot germline confound documented in research |
| kothiwal re-run changed outputs | stages deterministic; diffs verified (641,090 → 641,454 = exactly +364 kothiwal rows) |
| Null-antigen groups (20) | zero pseudo-antigen token; antibody pathway is the default |
| Proxy-label dominance (Pred_affinity ≈77% of subsample weight) | **confirmed harmful** for SPR-Kd transfer (DCC 0.04→0.35 when excluded); `rank_only_factor: 0.0` for Kd-style eval (2026-07-27) |
| Session/worker death kills long runs | all long jobs run `setsid`-detached with file logs (`training/runs/logs/`) since 2026-07-27 |
| OOM with ≥3 concurrent training runs (9.2 GB/run) | slim corpus loading in `training/data.py` (3.3 GB/run, 2026-07-27); max 2 concurrent runs |

---

## Summary Pipeline

```
DATA (refreshed 2026-07-26):
  83 CSVs -> 641,454 harmonized rows / 5,697 groups (trimmed, sign-audited,
    censor-parsed, clustered; kothiwal SPR + per-row antigen_seq included)
    -> training_groups (2,274 eligible) + splits.json (LOFO + clustered + Ag-cluster holdout)
    -> antigens: pipeline ag_seq + curated antigen_table.csv (SAbDab-PDB-first)
      = 4,716 unique antigen sequences

EXTRACTION (CPU, one-time, cached, fp16):
  antibodies: dedup (235,331 unique) -> frozen AntiBERTy [512-d] per chain, concat
  antigens:   dedup (4,716 unique)   -> frozen ESM-2 650M [1280-d], windowed >1000 aa

TRAINING (CPU, ~1.8M trainable params):
  groups of 20 (censored pinned, tie-jitter, group-weighted)
    -> tower adapters: Linear->LN->GEGLU-FFN + residual   (256-d)
    -> interaction: Pre-LN cross-attn Q=antigen KV=antibody   [A2 ablates]
    -> fusion: bilinear gate(AttnPool(P_ag'), AttnPool(P_ab)) [A3 ablates]
    -> head 256->64->1
    -> ListMLE (+ aux VHH pairwise margin, λ 0.1–0.3) — AdamW, splits.json CV

INFERENCE (benchmark):
  20 sequences (scFv: VH+VL; VHH: VH) + antigen sequence -> cached-encoder forward
    + trained head -> Score_i -> sort descending -> Rank 1..20 -> CSV (VH/VHH, VL, Rank)
```

**Baselines:** linear probe + zero-shot IgLM/ProGen2 (validated). **Goal:** beat probe and zero-shot consistently across held-out folds incl. antigen-cluster holdout; target mean Spearman 0.4–0.7 same-antigen (report estimate), 0.2–0.5 cross-antigen.
