# E2-iextended — DOGMA Pretrain Spec

**Status**: Pre-training specification. Blocks on Track A (`scripts/assemble_dogma_h5ad.py` production run).
**Supersedes**: Original E2 K562 spec (`prompts/phase6_5g_e2_k562_pretrain.md`, archived at tag `phase-6.5g-e2-abandoned-2026-04-23`).
**Author**: Ash Khan
**Date**: 2026-04-24

---

## 1. Corpus

**Primary corpus**: DOGMA-seq (GSE156478, Mimitou et al. 2021 *Nat Biotechnol*) — pooled DIG + LLL arms.

| Arm | Condition | n_cells | Source |
|---|---|---|---|
| DOGMA-LLL | CTRL | 7,624 | pairs_of_wnn.tsv (Mimitou anchor) |
| DOGMA-LLL | STIM | 6,139 | pairs_of_wnn.tsv |
| DOGMA-DIG | CTRL | 9,483 | QC filter chain (`15_LLLandDIG.R` subset) |
| DOGMA-DIG | STIM | 8,628 | QC filter chain |
| **Combined** | | **31,874** | barcode-set fidelity-gated (commit `21fb459`) |

Single donor (Mimitou 2021 Methods: "Cryopreserved healthy donor PBMCs... cells were split into 4 groups").

**Evidence anchor**: `data/phase6_5g_2/external_evidence/dogma_ncells_measured_2026-04-23/dogma_ncells_measured.json` (`pairs_of_wnn.tsv` SHA `29a7565e15e23761c41af3c3e99d24f3c1ca0acb2c8ef19323e51a23a674ee0e`).

## 2. Modalities

> **Amendment (2026-04-25)**: ATAC representation pivoted from chromVAR
> motif deviations (~700 dim) to raw peak counts (~85k–150k dim sparse).
> Rationale: pip-on-macOS + R/Bioconductor chromVAR install paths both
> exhausted dep-conflict / system-tooling debug cycles after multiple
> attempts. Raw peaks is the AIVC ATAC encoder's native input
> (`PeakLevelATACEncoder` with TF-IDF + LSI; `apply_tfidf=True` default
> from PR #28). Trade-off: lose TF-activity interpretability of motifs;
> retain finer-grained peak-level signal. Calderon 2019 eval (§6.1)
> needs corresponding amendment — peak-set harmonization to Calderon's
> peak coordinates instead of motif-space alignment. PR #28's
> `apply_tfidf=False` flag retained for future chromVAR revival without
> code change.

RNA + ATAC + Protein. **Phospho and mtDNA explicitly out-of-scope** for E2-iextended (future QuRIE-seq extension).

| Modality | Representation | Normalization | Dim |
|---|---|---|---|
| RNA | `.X` | `normalize_total(1e4)` + `log1p` | ~30k genes (deposit-dependent) |
| ATAC | `obsm['atac_peaks']` | raw peak counts (sparse, hg38-aligned) | ~85k–150k peaks (lane-dependent) |
| Protein | `obsm['protein']` | CLR (centered log-ratio) | 210 (TotalSeq-A fixed) |

## 3. Batch correction

`lysis_protocol` injected as a **scVI-style categorical batch covariate at the encoder level** (NOT Harmony, NOT post-hoc — per architectural decision 2026-04-23).

- Codes via `aivc/data/collate.py::LYSIS_PROTOCOL_CODES`: LLL=0, DIG=1, unknown=-1.
- Concatenated to encoder input as a per-cell embedding (dim TBD, default 8).
- Gradient flows through the covariate embedding — the model learns to reconcile the DIG/LLL distributional differences as part of training, not as a preprocessing step.

**Rationale**: Mimitou 2021 Ext Data Fig 5o validates joint DIG+LLL analysis *under Harmony batch correction*. scVI-style covariate injection is a stricter test (the model must learn the correction jointly with the representation) and matches the AIVC architecture's existing batch-covariate hooks.

## 4. Batch sampler

**Homogeneous-modality batches ENFORCED** (DD2 constraint from Day 3 collate).
- `aivc/data/collate.py::dogma_collate` raises `ValueError` on heterogeneous batches (mixed Protein presence, mixed `protein_panel_id`, mixed `dataset_kind`).
- Current sampler must ensure all cells in a batch have the same modality presence set.
- `lysis_protocol` mixing **within** a batch is allowed (that's the whole point of the covariate); panel/dataset_kind mixing is not.

## 5. Loss composition

`losses.py::_dogma_pretrain_loss` with `stage="pretrain"` via `LossRegistry`.

| Term | Function | Weight | Mask-gated? |
|---|---|---|---|
| `masked_rna_recon` | `pretrain_losses._masked_rna_recon` | 1.0 | Token-level (rna_mask) |
| `masked_atac_recon` | `pretrain_losses._masked_atac_recon` | 1.0 | Token-level (atac_mask) |
| `masked_protein_recon` | `pretrain_losses._masked_protein_recon` | 1.0 | Per-cell modality_mask (Day 2) |
| `cross_modal_infonce_triad` | `pretrain_losses._cross_modal_infonce_triad` | 0.5 | Pairwise silent-skip (D2) |
| `peak_to_gene_aux` | `pretrain_losses._peak_to_gene_aux` | 0.05 | Optional |

**Constraints (hard)**:
- NO Neumann L1 term (causal head not touched in pretrain)
- NO `causal_ordering` term (not pretrain-stage)
- Defense-in-depth: `_guard_pretrain_name` fires at registration time; `_PRETRAIN_FORBIDDEN_NAMES` fires at compute time (double-guarded against causal-adjacent term leakage into W)

## 6. Evaluation

### 6.1 Primary: external (Calderon 2019)
**Dataset**: GSE118189 — Calderon et al. 2019, stimulation-responsive chromatin across human immune cell types.
**Procedure**: project DOGMA encoder outputs into Calderon's ATAC-aligned space; compute cluster alignment against Calderon's published cell-type labels.
**Metric**: Adjusted Rand Index (ARI) on CD4, CD8, B cell, NK clusters.
**Pass threshold**: ARI ≥ 0.70 (conservative; Calderon clusters are coarse).

<!-- CALDERON_PREP_AMENDMENT_2026_04_27 -->
#### §6.1 amendment — Calderon 2019 external eval (raw peaks)

**Variant**: raw peak counts (consistent with DOGMA post-PR #30 chromVAR
abandonment; same rationale: chromVAR install path was unrecoverable on
macOS-x86_64 + Python 3.9/3.11, R/Bioconductor path consumed 7+ debug
iterations without convergence).

**Source**: GSE118189 — Calderon et al. 2019 *Nat Genet*, "Landscape of
stimulation-responsive chromatin across diverse human immune cells".

**File format (verified 2026-04-27 via NCBI FTP listing)**: single
supplementary file `GSE118189_ATAC_counts.txt.gz` (~111 MB). Tab-separated,
peaks as rows, samples as columns. Sample IDs encode `{donor}-{cell_type}-{stim_state}`
(e.g. `1001-CD4_Teff-S` for stimulated, `1002-Bulk_B-U` for unstimulated).

**Build script**: `scripts/prepare_calderon_2019.py` — produces an AnnData
with raw counts in `.X`, donor/cell_type/stim_state in `.obs`, peak
chrom/start/end in `.var`, and `uns = {source: GSE118189, assay: bulk_ATAC,
variant: raw_peaks}`.

**Integration test**: `tests/test_calderon_prep_integration.py` — 6
synthetic tests run by default; 1 real-data smoke test gated behind
`@pytest.mark.real_data + @pytest.mark.slow`. Opt-in via
`pytest -m real_data tests/test_calderon_prep_integration.py`.

**Peak harmonization to DOGMA**: deferred to a follow-up PR
(`scripts/harmonize_calderon_peaks.py`). Calderon and DOGMA peak sets are
called independently — overlap-matching with bedtools or pyranges is
required before joint representation eval.

### 6.2 Secondary: internal stratified holdout
**Split**: 10% per-arm-per-condition holdout, stratified by `lysis_protocol × condition` (so holdout preserves the 2×2 structure).
**Metric**: 3WNN clustering ARI on held-out cells vs trained-model projection.
**Pass threshold**: ARI ≥ 0.85.

### 6.3 CD3/CD28 stim response
**METADATA-ONLY.** Not an evaluation signal. (D2 verdict, 2026-04-23: full run × stim aliasing in the single-donor aliquot-split design precludes clean causal identification. Documented in `project_aivc_phase65_closure.md` memory.)

## 7. Divergence tripwires (pre-registered)

Computed on post-embedding cluster assignments, comparing DIG-only vs LLL-only subsets:

| Modality | Threshold | Fallback if failed |
|---|---|---|
| RNA ARI (DIG vs LLL) | ≥ 0.95 | **HALT** — fundamental representational divergence; don't pool |
| ATAC ARI (DIG vs LLL) | ≥ 0.90 | **ATAC-only DIG fallback** (see §7.1) |
| Protein ARI (DIG vs LLL) | ≥ 0.95 | **HALT** — protein reconciliation failed |

### 7.1 ATAC-only DIG fallback
Activated if ATAC ARI < 0.90:
- Retrain with ATAC branch restricted to DIG cells only (LLL cells masked out of ATAC loss)
- Keep RNA + Protein branches pooled
- Re-evaluate: eval thresholds unchanged
- If re-eval passes, proceed with the restricted-ATAC model; flag in paper/report that LLL ATAC was excluded

## 8. Checkpointing

- Save every 5 epochs.
- Keep top-3 by primary Calderon ARI.
- Retain final epoch regardless of ranking (for post-hoc inspection).
- Checkpoint naming: `checkpoints/pretrain/dogma_e2_ext_epoch{N}_ari{A:.3f}.pt`.
- Schema version 1, loadable via `aivc.training.ckpt_loader.load_pretrained_simple_rna_encoder`.

## 9. Training configuration

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Batch size | 256 per device (adjust for GPU memory) |
| Epochs | 50 |
| LR schedule | 500-step linear warmup → cosine decay |
| Early stop | Calderon ARI plateau for 5 consecutive evals |
| Compute budget | ≤ 24 GPU-h per run (2-3 runs max before decision point) |

## 10. QuRIE-seq readiness note

**Extends cleanly (no redesign)** — already 4-modality aware:
- `aivc/data/modality_mask.py`: `ModalityKey` enum includes `PHOSPHO=1`; `build_mask` handles any 4-modality subset
- `aivc/skills/fusion.py`: `TemporalCrossModalFusion` accepts 4 modalities in temporal order; `modality_mask` gating (PR #21 hotfix) correctly excludes absent modalities from attention
- `aivc/data/collate.py::dogma_collate`: mask stacking is generic over modality count
- `losses.py::combined_loss_multimodal`: mask-gated inner terms work for any modality subset

**Needs extension for 4-modality (QuRIE-seq RNA+ATAC+Protein+Phospho)**:
- `aivc/data/multiome_loader.py`: add `phospho_path` / `phospho_obsm_key` param (mirror of Protein additions)
- `aivc/training/pretrain_losses.py`: add `_masked_phospho_recon` (symmetric to `_masked_protein_recon`); extend `_cross_modal_infonce_triad` to 4-way (6 pairwise NCEs = C(4,2))
- `aivc/data/pairing_certificate.py`: `make_qurieseq()` factory with 6 PHYSICAL pairs (currently pending wet-lab confirmation in `make_quriegen_pending()`)
- `scripts/assemble_*.py`: new assembly script for QuRIE-seq (Phospho-specific preprocessing TBD)

**Summary**: ~30% of the DOGMA infrastructure requires extension for QuRIE-seq; the canonical contract (modality_mask keys, temporal order, mask semantics, collate policy, two-layer gating) extends for free.

## 11. Dependencies

Runtime:
- `pychromvar`, `pyjaspar`, `anndata`, `scanpy` (h5ad assembly, Track A)
- `torch`, `scvi-tools` (pretrain, already in AIVC stack)

External data:
- Calderon 2019 GSE118189 (eval, download separately)
- JASPAR 2022 CORE vertebrates (Homo sapiens subset, ATAC motif deviations)

## 12. Pre-training unblockers

1. **Track A complete** (PR #26 merged): `scripts/assemble_dogma_h5ad.py` produces `dogma_lll.h5ad` and `dogma_dig.h5ad`.
2. **ATAC encoder compatibility check**: verify `aivc/skills/atac_peak_encoder.py` accepts chromVAR motif deviations (dim ~700) as input — may need a lightweight input-projection layer since the encoder was designed for peak-level features.
3. **Calderon 2019 preprocessing**: separate task — download, filter, align to DOGMA chromVAR motif space.
4. **scVI-style batch covariate head**: wire `lysis_protocol` covariate into encoder input; minor `aivc/skills/rna_encoder.py` / `atac_peak_encoder.py` / Protein encoder change.
5. **Compute**: single-GPU run fits in 24h per CLAUDE.md budget.

## 13. References

- Mimitou et al. 2021, *Nat Biotechnol*, PMID 34083792, PMC8763625. DOI 10.1038/s41587-021-00927-2.
- Calderon et al. 2019, *Nat Genet*, GSE118189.
- `pairs_of_wnn.tsv` fidelity gate — `data/phase6_5g_2/external_evidence/dogma_ncells_measured_2026-04-23/`.
- Architectural decisions locked 2026-04-23 Thursday call (Thiago + Kinga).
