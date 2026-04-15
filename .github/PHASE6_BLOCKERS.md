# Phase 6 pre-start blockers

Tracking items that MUST be resolved before Phase 6 work begins.

## [ ] Run harmonize_peaks on 10x PBMC Multiome and produce peak_set artifact

- **Owner:** TBD
- **Target:** before Phase 6 kickoff
- **Why:** Phase 5 ran on mock Multiome. First real-data signal on
  whether pretraining helps Norman/Kang fine-tuning only arrives in
  Phase 6. Phase 6 merge gate requires a real-data pretraining run.
- **Command sketch:**
  ```
  python scripts/harmonize_peaks.py \
      --fragments pbmc_multiome_fragments/*.tsv.gz \
      --out artifacts/pbmc_multiome_peak_set.bed
  python scripts/pretrain_multiome.py \
      --multiome_h5ad data/pbmc_multiome.h5ad \
      --peak_set artifacts/pbmc_multiome_peak_set.bed \
      --steps 10000
  ```
- **Acceptance:** a non-mock run saved at
  `checkpoints/pretrain/pretrain_encoders.pt` with loss monotone on
  moving average and schema_version=1 validated.
