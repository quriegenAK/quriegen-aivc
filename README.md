# AIVC — AI Virtual Cell Platform

Multi-omics AI platform for perturbation response prediction in primary immune cells. Built on QuRIE-seq proprietary paired RNA + Protein + Phospho single-cell measurements.

## Current model status

| Component | Status | Notes |
|-----------|--------|-------|
| RNA encoder (GeneLink GAT) | Live — r=0.873 | 3,010 genes, STRING PPI, Kang 2018 |
| Protein encoder | Coded, untrained | Awaiting QuRIE-seq data (May 2026) |
| Phospho encoder | Architecture defined | No implementation class yet |
| ATAC encoder | Coded, untrained | Awaiting 10x Multiome data |
| Temporal fusion (384-dim) | Coded, causal mask enforced | Not yet trained multi-modal |
| Neumann cascade (K=3) | Coded, sparsity enforced | Awaiting H100 training run |
| Contrastive loss | Coded, gated | Activates when pairing confirmed |
| 3-critic validation gate | Live | Includes mechanistic direction checks |
| TileDB-SOMA store | Coded | USE_SOMA_STORE=False default |
| Active learning loop | Infrastructure only | Loop not yet closed |

## Benchmark

- Pearson r = 0.873 on held-out donors (Kang 2018, GSE96583, IFN-beta stimulation)
- Benchmark integrity confirmed: IFIT1 ambient RNA in ctrl cells = 3.0% (CLEAN)
- JAK-STAT recovery: 7/15 genes within 3x fold change (target: >=8/15 in v1.1)
- CD14+ monocyte r = 0.745 (target: >=0.80 in v1.1)

Compares against: CPA (r=0.856), scGEN (r=0.820) on identical test split.

## Model architecture

```
RNA (3,010 genes) --> GAT (2-layer, 4-head) --> 128-dim
Protein (200 ab.)  --> MLP + cross-attn     --> 128-dim  [stub]
Phospho (kinases)  --> MLP + kin-substrate  -->  64-dim  [planned]
ATAC (chromVAR TF) --> MLP                  -->  64-dim  [coded, no data]
                                                  |
                    Temporal cross-modal fusion (QxKV, causal mask)
                    ATAC(t=0) -> Phospho(t=1) -> RNA(t=2) -> Protein(t=3)
                                                  |
                                           384-dim fused state
                                                  |
                    ResponseDecoder (MLP) -> direct effects
                                                  |
                    NeumannPropagation (W, K=3) -> cascade effects
                                                  |
                                         delta expression per gene
```

**Causal ordering enforcement:** The temporal ordering is enforced via a lower-triangular attention mask (upper-triangle weight = 0.0 confirmed) plus a `causal_ordering_loss` penalty term. This is a temporal constraint, NOT a Pearl do-calculus structural causal model.

## Training datasets

| Dataset | Role | Perturbation IDs |
|---------|------|-----------------|
| Kang 2018 (GSE96583) | Response training + benchmark | ctrl=0, stim=1 |
| PBMC IFN-gamma (OneK1K / Dixit 2016) | Stage 2 response training | ctrl=2, stim=3 |
| Frangieh 2021 (SCP1064) | W-pretraining only (NOT response training) | W-only |
| ImmPort SDY702 | W-pretraining only (cytokine GRN) | W-only |
| Replogle 2022 (K562, essential) | W-pretraining only (housekeeping genes only) | W-only |

**Frangieh restriction:** Frangieh melanoma cells are explicitly blocked from perturbation response training (`USE_FOR_W_ONLY=True` enforced). Melanoma JAK-STAT biology (BCR-ABL1, constitutively active JAK2) is biologically incorrect for PBMC IFN response prediction.

**Replogle restriction:** Only the 267-gene housekeeping safe set is used for W pretraining. 72 genes are explicitly blocked (JAK-STAT pathway, BCR-ABL1, oncogenes, immune receptors) because K562 has constitutively active JAK2/STAT5 which would inject leukaemia-specific regulatory patterns into the PBMC W matrix.

## QuRIE-seq pairing status

**UNVERIFIED.** The pairing certificate (`data/pairing_certificates/quriegen_pending.json`) lists all modality pairs as `pairing_type: "unknown"` pending wet-lab confirmation. Contrastive loss is disabled until physically co-measured pairing is confirmed. Do not present same-cell multi-modal pairing as a validated feature until `quriegen_pending.json` is updated with confirmed types.

## Repository structure

```
.
├── README.md
├── train_week3.py              # v1.0 training (live, r=0.873)
├── train_v11.py                # v1.1 Neumann sweep (coded, awaiting GPU)
├── perturbation_model.py       # GAT + ResponseDecoder
├── losses.py                   # MSE + LFC + cosine + L1 + causal loss
├── build_ot_pairs.py           # OT cell pairing (9,616 pairs)
├── evaluate_week3.py           # Test set evaluation
├── demo_aivc_week3.ipynb       # Two-audience demo notebook
│
├── aivc/
│   ├── critics/
│   │   └── biological.py       # 3-critic gate (mechanistic checks added)
│   ├── data/
│   │   ├── multi_perturbation_loader.py  # Kang+PBMC_IFNg+Frangieh(W)+Replogle(W)
│   │   ├── pairing_certificate.py        # QuRIE-seq pairing governance
│   │   ├── housekeeping_genes.py         # 267 safe / 72 blocked for W pretrain
│   │   ├── soma_store.py                 # TileDB-SOMA (USE_SOMA_STORE=False default)
│   │   └── soma_training_bridge.py       # h5ad <-> SOMA bridge
│   ├── orchestration/
│   │   └── perturbation_curriculum.py    # 4-stage curriculum, synthetic gate
│   ├── preprocessing/
│   │   └── atac_rna_pipeline.py          # 7-step Python pipeline (Step 6: stub)
│   ├── qc/
│   │   ├── ambient_rna.py                # SoupX-equivalent decontamination
│   │   ├── benchmark_integrity.py        # r_raw vs r_clean comparison
│   │   └── domain_contamination_guard.py # Blocks melanoma from response training
│   └── skills/
│       ├── atac_encoder.py               # chromVAR TF scores -> 64-dim (coded)
│       ├── contrastive_loss.py           # InfoNCE (gated on pairing cert)
│       ├── fusion.py                     # 384-dim temporal fusion, causal mask
│       ├── neumann_propagation.py        # W cascade, sparsity enforced
│       └── neumann_w_pretrain.py         # W pretraining from Replogle
│
├── tests/
│   ├── test_neumann_sparsity.py          # 12 tests (W enforcement)
│   ├── test_ambient_rna.py               # 15 tests (decontamination)
│   ├── test_pairing_certificate.py       # 13 tests
│   ├── test_contrastive_loss.py          # 8 tests
│   ├── test_dataset_fixes.py             # 15 tests (Frangieh block)
│   ├── test_housekeeping_filter.py       # 15 tests (Replogle filter)
│   ├── test_soma_store.py                # 15 tests
│   ├── test_causal_mask.py               # 10 tests (fusion causal enforcement)
│   ├── test_synthetic_gate.py            # 8 tests (Stage 2 synthetic block)
│   └── test_mechanistic_checks.py        # 15 tests (direction checks)
│
├── scripts/
│   ├── run_soupx_equivalent.py           # Ambient RNA decontamination CLI
│   ├── ingest_to_soma.py                 # SOMA store ingestion CLI
│   └── generate_pairing_certificates.py  # Certificate JSON generation
│
└── data/pairing_certificates/
    ├── kang2018.json                     # RNA only, COMPUTATIONAL
    └── quriegen_pending.json             # All UNKNOWN — pending Thiago
```

## Running the platform

```bash
# v1.0 baseline (live)
python train_week3.py
python evaluate_week3.py

# v1.1 Neumann sweep (requires GPU, Kang 2018 h5ad in data/)
python train_v11.py

# Benchmark integrity check (no raw matrix needed — FALLBACK mode)
python scripts/run_soupx_equivalent.py \
  --filtered data/kang2018_pbmc_fixed.h5ad \
  --output   data/kang2018_pbmc_decontaminated.h5ad \
  --report   results/ambient_rna_report.csv

# Generate pairing certificates
python scripts/generate_pairing_certificates.py

# Run all tests (126 tests, 0 expected failures)
python -m pytest tests/ -v
```

## Open bugs

| Bug | File | Fix |
|-----|------|-----|
| SOMA metadata loss on ingestion | aivc/data/soma_store.py | Add `adata.obs = obs; adata.var = var` before `sio.from_anndata()` |
| tf_motif_scanner.py NotImplementedError | aivc/preprocessing/tf_motif_scanner.py | Implement chromVAR-equivalent (JASPAR 2024) |

## What is NOT yet implemented

- ProteinEncoder and PhosphoEncoder (no class files exist)
- SCM do(X) interventions (temporal mask != Pearl SCM)
- React UI / Explorer Dashboard / Python SDK
- FastAPI + Triton serving endpoint
- MLflow model registry integration
- Active learning loop (one wet-lab cycle not yet completed)
- Zero-shot validation on unseen perturbation
- Cross-cell-type generalisation testing

## Requirements

- Python 3.11+
- PyTorch 2.2+ / PyTorch Geometric 2.7+
- tiledbsoma (optional — install with `pip install tiledbsoma` for SOMA store)
- See `requirements.txt` for full list

## License

Proprietary. Copyright QurieGen.
