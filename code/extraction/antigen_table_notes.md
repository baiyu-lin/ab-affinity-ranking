# antigen_table.csv — per-group decisions

Covers all 52 groups in `data_pipeline/output/harmonized_clustered.csv.gz` that lack `ag_seq`,
plus 3 aux rows keyed `AUX::<source_file>` (from `harmonized_aux.csv.gz`).
Columns: `group_id,ag_seq,source`. All non-null sequences match `^[ACDEFGHIKLMNPQRSTVWY]+$`.

Sequence provenance:
- **sabdab_pdb**: SEQRES of the antigen chain(s) parsed from raw PDBs in
  `structure_data/all_structures_030526.zip`, chain assignment from `sabdab_summary_all.tsv`.
  Multi-chain antigens (HA = HA1+HA2) are concatenated in biological order. Expression tags
  stripped where clearly artificial (TSLP 5j13: N-term His tag; FXI 6aod: C-term 8xHis;
  TNFRSF9 6mi2: C-term ENLYFQG; WT-RBD 7mmo cross-check only). MSE mapped to M.
- **manual_known**: sequence stated verbatim in the source paper (or UniProt canonical for a
  clearly specified WT antigen).
- **manual_known_approx**: closest verifiable construct for the stated antigen (exact strain /
  accession cited in the paper), with boundary/tag approximations noted.
- **null_small_molecule / null_unknown**: ag_seq left empty.

## sabdab_pdb (31 groups)

| group | antigen | PDB:chain |
|---|---|---|
| 1/jain2024assessment_Hen_Lys_kd | hen egg-white lysozyme (HEL campaign, Jain 2024) | 1mlc:E |
| 8/hutchinson2023enhancement_* (8 groups) | HEL — anti-HEL D44.1 design campaign (paper states PDB 1MLC) | 1mlc:E |
| 20/warszawski2019_d44_Kd | HEL — D44.1 (1MLC) | 1mlc:E |
| 10/koenig2017mutational_kd_g6 | human VEGF-A (8–109), G6.31 Fab panning antigen | 2fjg:W |
| 13/phillips2021binding_cr9114_h1_kd | H1 HA A/New Caledonia/20/1999 (paper: NC99 H1) | 9b2m:C+D (HA1+HA2) |
| 15/rosace2023automated_kd_adalimumab | human TNF-α | 3wd5:A |
| 15/rosace2023automated_kd_golimumab | human TNF-α (5yoy chain carries FLAG tag → used clean 3wd5:A) | 3wd5:A |
| 16/...Afasevikumab-IL17A | human IL-17A (secukinumab complex; 6-aa N-term construct prefix GPIVKA retained) | 9sfx:C |
| 16/...Bimagrumab-ACVR2B | ActRIIB ECD (bimagrumab complex — exact reference Ab) | 5ngv:A |
| 16/...Eculizumab-C5 | human complement C5, full length (eculizumab complex — exact) | 5i5k:A |
| 16/...Osocimab-FXI | human factor XI catalytic domain | 6aod:C (tag stripped) |
| 16/...Tezepelumab-TSLP | human TSLP (5J13 = AMG 157/tezepelumab complex — exact) | 5j13:A (tag stripped) |
| 16/...Utomilumab-TNFRSF9 | human 4-1BB/TNFRSF9 ECD (utomilumab complex — exact) | 6mi2:C (tag stripped) |
| 17/shanker2024 Ly1404-BQ.1.1, SA58-BQ.1.1 | SARS-CoV-2 BQ.1.1 RBD (R346T verified) | 9c7s:C |
| 17/shanker2024 SA58-XBB.1.5 | SARS-CoV-2 XBB.1.5 RBD (G339H+R346T verified) | 8we4:B |
| 2/shanehsazzadeh2023unlocking_* (3 groups) | human HER2 ECD (paper: PDB 1N8Z chain C) | 1n8z:C |
| 7/hie2023 CoV2_S309 | SARS-CoV-2 Wuhan S-6P (D614 + GSAS furin + 6 Pro verified) | 7e9o:A |
| 7/hie2023 CoV2Beta_C143, CoV2Beta_REGN10987 | SARS-CoV-2 Beta (B.1.351) spike ectodomain; approximates the Beta S-6P/S2P screening antigen | 7fjo:A |
| 7/hie2023 CoV2omicron_REGN10987 | SARS-CoV-2 Omicron BA.1 RBD | 7y0c:R |
| 7/hie2023 MEDIUCA_H1Solomon | H1 HA A/Solomon Islands/3/2006 | 9oi2:A+B (HA1+HA2) |

## manual_known (4 groups/rows)

- `3/li2023...||MIT_Target` (affinity1 & affinity2) and `6/engelhart2022dataset...||single`:
  target peptide **PDVDLGDISGINAS** (HR2 region of SARS-CoV-2 spike), stated verbatim in
  Engelhart et al. 2022 (the AlphaSeq dataset paper for the same MIT-LL dataset Li et al. 2023 uses).
- `AUX::18/tsuruta2024avida-hIL6_binary.csv`: mature human IL-6, UniProt P05231 residues 30–212
  (183 aa; AVIDa-hIL6 immunized alpaca with WT human IL-6).

## manual_known_approx (10 groups/rows)

- `13/phillips2021binding_cr9114_h3_kd`: H3 HA A/Wisconsin/67/2005(H3N2), GenBank AIU46082.1,
  mature HA0 (signal peptide 1–16 removed). The assay construct is HA1 1–329 + HA2 1–176
  (H3 numbering); full mature HA given as approximation.
- `16/...Spesolimab-IL36R`: human IL-36R/IL1RL2 extracellular domain, UniProt Q9HB29 20–335.
  (Note: PDB 6A3W/6MI2-style sabdab entries do not exist for IL-36R; no AbRank entry either.)
- `7/hie2023 MEDI/MEDIUCA_H4Hubei` (2 groups): H4 HA A/swine/Hubei/06/2009(H4N1),
  GenBank AFV33926.1, mature HA0 (SP removed); exact strain per Hie et al. methods.
- `7/hie2023 MEDI_H7HK16`: H7 HA **A/Hong Kong/61/2016(H7N9)** — Hie methods explicitly define
  "H7 HK16" as HK/61/2016 (the "HK17" in the text is the other antigen, A/Hong Kong/125/2017).
  Sequence = PDB 6D7C HA1+HA2 (SEQRES via RCSB; strain-confirmed).
- `7/hie2023 ebola_mab114`: EBOV GP (Mayinga), GenBank AAG40168.1 cited by Hie et al.;
  ectodomain (residues 33–650) with the mucin-like domain deleted (Δ309–489), per paper methods.
- `9/kirby2024...kd`, `14/rawat2022abcov_kd`, `AUX::19/tsuruta2024sarscov2_binary`,
  `AUX::9/kirby2024...binary_kd`: canonical SARS-CoV-2 Wuhan RBD (spike 319–541 region,
  213 aa, from GenBank BCN86353.1 cited by Hie et al.; consistent with PDB 7mmo RBD chain).
  Approximations: kirby = Wuhan-hu-1 spike/RBD sort antigen (RBD for RBD-targeting Fabs);
  rawat = Ab-CoV database, mixed coronavirus antibodies, predominantly SARS-CoV-2 spike;
  tsuruta-sarscov2 = binary VHH data aggregated over 13 spike-variant antigens → WT RBD proxy.

## null_small_molecule (3 groups)

- `21/zimmerman2020antibody_4420_kd`, `5/adams2017measuring_4420-fluorescein_kd-flow`,
  `5/adams2017measuring_4420-fluorescein_kd-titeseq`: antigen is fluorescein (4-4-20 antibody).

## null_unknown (7 groups)

- `3/li2023...||AlphaNeg1/2/3` (6 groups): AlphaSeq **negative-control yeast strains**
  (non-specific mating/barcode-artifact measurements, Engelhart et al. methods) — no protein
  antigen exists; deliberately NOT assigned the HR2 peptide.
- `1/jain2024assessment_mouse_Ly_kd`: mouse lysozyme cross-reactivity counter-screen
  (only 2 rows); exact lysozyme species/construct not specified in Jain et al. 2024.

## Notes / anomalies

- AbRank `Ag_name_details` name matching was validated but **rejected as a source**: most
  relevant names (IL-17A, IL-6, VEGF-A, TNF, ERBB2, lysozyme) map to ≥2 distinct Ag_seq
  variants (different construct lengths), so uniqueness failed. TSLP/ACVR2B/FXI were unique in
  AbRank but the same PDB chains were used directly (sabdab_pdb) instead.
- Shanker 2024 input structures: Ly1404–Wuhan (7MMO present in sabdab zip) and SA58–BA.1; the
  Kd files measure binding to BQ.1.1/XBB.1.5 RBD, hence variant RBDs from 9c7s/8we4.
- Hie 2023 Beta antigen files are named "S2P" but the paper screened Beta S-6P; 7fjo is a
  B.1.351 spike ectodomain cryo-EM construct (stabilization scheme not stated in sabdab summary).
- IL36R ECD boundaries (20–335) follow the UniProt annotation; the spesolimab SPR antigen
  construct boundaries are not published in the IgDesign paper.
