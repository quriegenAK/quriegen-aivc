# Stage 2 Unlock Checklist

## Blockers (must ALL be true before running)

- [ ] Real PBMC IFN-G dataset downloaded (not synthetic)
      Preferred: OneK1K (Yazar 2022) — large PBMC cohort
      Fallback:  Dixit 2016 GSE90063 PBMC subset (if obtainable)
      Validation: `python scripts/download_pbmc_ifng.py --output data/pbmc_ifng_real.h5ad --validate`
      Certificate: `data/validation_certificates/pbmc_ifng_real.json`

- [ ] H100 sweep complete (v1.1 checkpoint exists)
      `models/v1.1/best_model.pt`
      If not: fall back to v1.0 (weaker IFIT1 starting point)

- [ ] W scale fix confirmed active
      `w_density_large > 0.05` at epoch 50 in MLflow
      (confirmed in local 5-epoch smoke sweep — all epochs at 1.000)

- [ ] Stage 2 gate confirmed
      `perturbation_curriculum.py advance_stage()` blocks synthetic IFN-G
      Test: `pytest tests/test_synthetic_gate.py`

## Run command

```bash
./scripts/run_stage2_unlock.sh \
  --ifng-data data/pbmc_ifng_real.h5ad \
  --checkpoint models/v1.1/best_model.pt \
  --output-dir results/stage2/
```

## Expected outcomes

| Metric           | v1.0 baseline | Stage 2 target | Notes                              |
|------------------|---------------|----------------|------------------------------------|
| Kang 2018 r      | 0.9033        | >= 0.873       | Must not regress                   |
| JAK-STAT 3x      | 7/15          | >= 10/15       | Neumann cascade propagation        |
| IFIT1 FC         | 3.75x         | >= 15x         | Cascade amplifies pathway signal   |
| Norman 2019 delta| 0.0000        | > 0.000        | Any positive delta = Stage 2 works |

## If Stage 2 does NOT improve Norman 2019 zero-shot

This means the model is not learning transferable perturbation mechanisms
even with two cytokine types. Next steps in order:

1. **Add ImmPort SDY702 (Stage 3)** — 7 additional cytokine perturbation types
   (IL-2, IL-4, IL-6, IL-10, TNF-A, GM-CSF, IFN-G). More perturbation diversity
   forces the perturbation embedding to encode mechanism, not identity.

2. **Increase num_perturbations in PerturbationPredictor**
   Currently 2 (ctrl=0, stim=1). Add one embedding per perturbation type.
   The perturbation embedding must distinguish IFN-B from IFN-G from IL-6
   to learn which pathways each activates.

3. **Contrastive loss on perturbation embeddings**
   IFN-B and IFN-G embeddings should be close (shared JAK-STAT pathway).
   IFN-B and CRISPRi KO embeddings should be far (different mechanism class).
   This requires the perturbation embedding to encode mechanism, not just identity.

4. **Include CRISPR perturbation data in training**
   If the goal is to predict CRISPR effects, the model needs CRISPR training data.
   Norman 2019 K562 ctrl cells are already on disk (5,409 cells).
   This would shift the evaluation from "zero-shot" to "few-shot" but is the
   scientifically honest path if cytokine-only training proves insufficient.

## If Stage 2 REGRESSES on Kang 2018

This means adding IFN-G data degraded IFN-B prediction. Likely causes:

1. Gene universe mismatch between IFN-G dataset and Kang 2018 HVG set
2. IFN-G dataset normalisation differs from Kang 2018 preprocessing
3. Perturbation embedding confusion (pert_id=1 and pert_id=3 interfering)

Fix: check `multi_perturbation_loader.py` gene universe alignment and
normalisation pipeline. Run Stage 2 with lower learning rate or more
Stage 1 warm-up epochs.
