# Embedding Extraction — Milestones 4/5

Frozen-encoder feature extraction for the ranking model (Plan v3 Part 1–3).
One script: `extract_embeddings.py` (run from the repo root with `.venv/bin/python`).

```bash
.venv/bin/python extraction/extract_embeddings.py --build-manifest   # manifest + row_map only
.venv/bin/python extraction/extract_embeddings.py                    # extract (resumable)
.venv/bin/python extraction/extract_embeddings.py --limit 64 --cache-dir .scratch/smoke  # smoke test
```

## What it does

1. **Manifest** (`--build-manifest`): all rows of `training_groups`-eligible
   groups (`harmonized_clustered.csv.gz`) + all rows referenced by
   `aux_pairs.csv` (`harmonized_aux.csv.gz`, join on `(source_file, row_idx)`),
   deduplicated on `(seq_heavy, seq_light)` → **234,989 unique sequences**
   (190,836 main / 44,762 aux), `seq_id = sha1(heavy|light)[:16]`.
   `row_map.csv` maps all 670,794 corpus rows to their `seq_id`.
2. **Extraction**: length-sorted, token-budget batches (`--batch-tokens 4096`),
   frozen encoders, fp32 compute → fp16 cache. Resumable: `seq_id`s already in
   `index.csv` are skipped. Throughput ≈ 3 seq/s on CPU (shortest-first order;
   full run ≈ 20–30 h → milestone-5 overnight batches).

## Conventions (validated, do not change silently)

- **ProGen2-small** (`model_data/progen2-small/`, 151M, 12 × 1024-d — *not* 768-d
  as earlier drafts assumed; config verified at download):
  input `'1' + VH + (GGGGS)3 + VL + '2'` (salesforce sentinels → ids 3/4,
  char-level tokenizer, 1 token/AA). NLL over AA labels only (linker/sentinels
  excluded), logits restricted to vocab 5..29 per `progen2/likelihood.py`.
  Sanity: CE = 2.55 on the salesforce `x_uniref90bfd30` string (target ≈ 2.4).
  Hidden states: last layer, linker/sentinel positions dropped → `[L, 1024]`.
- **IgLM** (`model_data/iglm_weights/IgLM`, GPT-2, 4 × 512-d):
  per chain `[HEAVY]`/`[LIGHT]`, `[HUMAN]`, `<seq>`, `[SEP]`; token ids = line
  numbers of `vocab.txt` (**not** `BertTokenizerFast` — under transformers 5.x it
  maps every token to UNK; see below). NLL = mean CE of `labels[2:]`, full vocab,
  exactly `iglm==0.1.0 IgLM.log_likelihood`. Hidden states per chain (specials
  stripped) concatenated heavy→light → `[L, 512]`.
  Validated **bit-exact** vs the official FLAb precomputed perplexities
  (`data/FLAb-sequence-data/score/iglm/binding/Warszawski2019_d44_Kd/*_ppl.csv`,
  mean rel diff 5e-8).
- scFv/Fab rows get the linker; VHH rows are heavy-only (`nll_iglm_light = NaN`).
  Positions align 1:1 between the two streams (both `[heavy ‖ light]`).

## Cache layout

```
extraction/cache/
├── manifest.csv          seq_id, seq_heavy, seq_light, format, in_main, in_aux, len_*
├── row_map.csv           corpus, source_file, row_idx, seq_id
├── index.csv             seq_id, path, len_heavy, len_light, nll_progen, nll_iglm_heavy, nll_iglm_light
└── seqs/<xx>/<seq_id>.npz
      h_progen  fp16 [L, 1024]      h_iglm      fp16 [L, 512]
      len_heavy int16               nll_progen / nll_iglm_heavy / nll_iglm_light  fp32
```

Estimated size for the full run: ~180 GB.

## Plan v4 caches (2026-07-26, antigen-conditioned overhaul)

Two new caches join the original one (the original `extraction/cache/` —
IgLM + ProGen2-on-antibody — is retained as ablation material and the source
of the IgLM/ProGen2 NLL bypass features and zero-shot baselines):

- `extract_antiberty.py` → `extraction/cache_antiberty/`: antibody tower
  (AntiBERTy 25.8M, frozen). Per chain `[CLS] chain [SEP]` (manual vocab dict,
  same transformers-5.x BertTokenizerFast caveat), specials stripped,
  concat heavy→light. npz: `h_ab` fp16 `[L, 512]`, `len_heavy`. No NLL (MLM).
  Same manifest/row_map/index layout as the original cache (~20 seq/s CPU;
  ~600 seq/s on RTX 4090 — both extraction scripts auto-select CUDA when
  available, CPU otherwise, numerics equivalent fp32).
- `extract_antigens.py` → `extraction/cache_esm2/`: antigen tower
  (ESM-2 650M, frozen). Sources: pipeline `ag_seq` + curated
  `extraction/antigen_table.csv` (see `antigen_table_notes.md`) + `AUX::` rows.
  `antigen_index.csv` maps `group_id`/`AUX::<file>` → `ag_id`; npz per antigen:
  `h_ag` fp16 `[L_ag, 1280]`. Antigens >1000 aa (11): sliding windows
  (1000/500) with overlap averaging. 4,716 unique antigens; 20 groups null
  (small molecule / negative control / unresolvable → zero pseudo-token at
  training time).

## transformers 5.x gotchas (already worked around)

- The salesforce `modeling_progen.py` buffers (`bias`, `masked_bias`,
  `scale_attn`) are **zeroed by the meta-device load** in transformers 5.x.
  `model_data/progen2-small/` is a repaired, self-contained checkpoint
  (hugohrban mirror weights + local salesforce modeling code, buffers baked in,
  `scale_attn` made persistent — one-line local patch). A fresh
  `from_pretrained(..., trust_remote_code=True)` now loads clean and passes the
  CE sanity check. Do not "update" the checkpoint from upstream.
- Old GPT-2 checkpoints report `attn.masked_bias` as UNEXPECTED on load —
  harmless (tf 5.x GPT-2 builds causal masks itself); verified causality holds.
- `BertTokenizerFast(vocab_file=...)` silently maps everything to UNK under
  tf 5.x → replaced with a direct vocab dict.

## Baseline correction (2026-07-25)

The plan's "IgLM zero-shot r = −0.836 on a garbinski2023 20-seq subsample" is
subsample noise: re-measured r = **−0.474 on the full 81-row set** (20-seq
subsamples range −0.07…−0.78). The solid reference is the official FLAb
precomputed score on gsk2023_D25_Kd (Spearman −0.835,
`data/FLAb-sequence-data/score/iglm/binding/gsk2023_D25_Kd/`), which our encoder
reproduces exactly — raw sequences for that file are not in the repo.
