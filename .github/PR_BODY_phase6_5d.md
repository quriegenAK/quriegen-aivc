# Phase 6.5d — RSA objective-mismatch test (LOCKED, pre-registered)

## Summary

**Outcome: D — Active distractor bias.** Δ = RSA_real − RSA_random =
**−0.0777** with 95% CI **[−0.0832, −0.0725]** (CI excludes zero with
margin) and `CI_width_real = 0.024` (no sampling-noise override). The
pretrained `SimpleRNAEncoder` latent geometry is *anti-correlated*
with the Norman 2019 perturbation-response geometry, while the
random-init baseline (5 seeds) sits weakly positive at ≈ +0.019. This
rules in either lineage mismatch (PBMC10k → K562) or objective
mismatch (cell-identity pretraining task is anti-aligned with
perturbation-response geometry), and rules out the "harmless
mismatch" reading of the 6.5c FAIL.

## Locked contract recap

Pre-registered in `prompts/phase6_5d_rsa.md`. Thresholds were locked
before any number was observed; no post-hoc retuning.

- **Aggregation (Option A — perturbation centroid).** For each
  perturbation `p`, latent centroid `z_p` = mean of pretrained encoder
  forward pass over all cells with label `p` (dim 128); response
  centroid `r_p` = mean of `log1p(X_filtered)` over the same cells,
  minus the NTC centroid in the same space. Structural-zero filter:
  17,956 / 36,601 genes retained for response space; encoder still
  consumes raw counts in 36,601 dims (no log1p into encoder — 6.5c
  lesson preserved).
- **Metric.** Pairwise cosine distance matrices over non-NTC
  perturbations (NTC excluded from pair indices; NTC is reference,
  not a perturbation). Flatten upper triangles; RSA = Spearman
  correlation between the two flat vectors.
- **Bootstrap (95% CI).** Real arm: 1000 pair-resamples with
  replacement, percentile CI. Random arm: 5 random inits (seeds
  {0..4}) with 1000 resamples each (5,000 aggregated). Δ CI: paired
  pair-bootstrap, 1000 resamples, each computing `real_i −
  mean(random_i across 5 inits)` on the same pair indices.
- **Interpretation rule (LOCKED).** Six branches: A
  (Δ > +0.05 ∧ RSA_real > 0.15), B (Δ > +0.05 ∧ RSA_real ≤ 0.15),
  C (boundary), C_WEAK (|Δ| < 0.05 ∧ 0 ∉ CI), D (Δ < −0.05),
  INCONCLUSIVE (CI overlaps 0, or `CI_width_real > 0.30` override).

## Results

| Quantity        | Point | 95% CI | Width |
|-----------------|-------|--------|-------|
| RSA_real        | **−0.0584** | [−0.0702, −0.0467] | 0.024 |
| RSA_random      | **+0.0193** | [−0.0076, +0.0529] | 0.061 |
| Δ = real − rand | **−0.0777** | [−0.0832, −0.0725] | 0.011 |

Per-seed RSA_random: seed 0 = +0.0002, seed 1 = +0.0179, seed 2 =
+0.0264, seed 3 = +0.0053, seed 4 = +0.0467.

| Pre-flight                 | Value |
|----------------------------|-------|
| `data_sha`                 | `d4bedb53082735735993ced3cf30864b349d62f21078523b8b93137f55b333c9` |
| `ckpt_real_sha`            | `416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e` |
| `n_cells_total`            | 111,445 |
| `n_genes_total`            | 36,601 |
| `n_genes_filtered`         | 17,956 |
| `n_perturbations_included` | 236 (non-NTC) |
| `n_perturbations_dropped`  | 0 (all pass MIN_CELLS_PER_PERT = 20) |
| `n_pairs`                  | 27,730 |
| `n_boot`                   | 1000 (per arm) |
| `random_seeds`             | [0, 1, 2, 3, 4] |
| `elapsed_seconds`          | 661 (≈ 11 min on MPS) |

Pre-flight observed `n_perturbations_total = 237` vs the spec-noted
`105`. Column (`perturbation`), NTC label (`control`), and SHA all
match expectations; the count discrepancy is a documentation artifact
in the spec, not a data provenance regression. Documented in
`.github/phase6_5d_rsa.json::schema_count_warning`. The locked RSA
contract is unaffected (more perturbations → more pairs → tighter CI).

Full machine-readable result: `.github/phase6_5d_rsa.json`.
Bootstrap distributions + per-seed centroids:
`experiments/phase6_5d/artifacts/` (gitignored).

## Outcome classification (pre-registered rationale)

Step 1 — interpretability gate:
- `|Δ| = 0.0777 ≥ 0.05` ✓
- `0 ∉ CI_Δ = [−0.0832, −0.0725]` ✓
- `CI_width_real = 0.024 ≤ 0.30` ✓
- → interpretable; proceed to Step 2.

Step 2 — outcome assignment:
- `Δ = −0.0777 < −0.05` → **outcome D** (Active distractor bias).

Reading: PBMC10k pretraining is not merely orthogonal to K562
perturbation geometry (which would be outcome C, Δ ≈ 0); it is
*actively misaligned*. Random-init forward passes on the same cells
produce RSA ≈ 0 (each seed sits in [+0.0002, +0.0467] — exactly the
expected null distribution). The pretrained checkpoint reorganises
the latent space along axes that contradict perturbation responses,
not just axes that ignore them.

## 6.5e scope implication

- **Demoted.** Hypothesis 3 (architecture ceiling). Random-init at
  dim-128 already sits at ≈0; capacity is not the bottleneck.
- **Ruled out.** Probe redesign on the existing `pretrain_encoders.pt`
  latents. A negative-Δ training-free signal means no linear or MLP
  head can recover perturbation structure better than a random
  init operating on the same cells; the latents themselves must
  change.
- **Scoped in.** 6.5e is a pretraining-recipe phase, not a probe phase.
  The two viable design directions are (1) lineage-matched
  pretraining (e.g., K562 / Perturb-Seq corpus) and (2)
  perturbation-aware objective (contrastive on perturbation vs control,
  supervised perturbation prediction, or response-space
  reconstruction). RSA cannot disentangle (1) vs (2) — that is itself
  a 6.5e design question.

## What this PR does NOT do

- No retraining, no fine-tuning, no checkpoint modification.
- No probe of any kind on the latents (RSA is training-free).
- No change to `SimpleRNAEncoder` or any encoder code.
- No modification of 6.5b/6.5c branches or their result sections.
- No 6.5e implementation. The outcome scopes 6.5e; 6.5e is a separate
  PR.

## Test plan

- [x] `pytest tests/test_phase6_5d_rsa.py` — 19 tests (helpers +
      all six interpretation branches: A, B, C, C_WEAK, D,
      INCONCLUSIVE + INCONCLUSIVE-via-CI-width override).
- [x] `pytest tests/` (excluding pre-existing env-only failures
      in `tests/test_full_loop.py` and `tests/agents/` per Issue
      #11 — anthropic SDK / typer not pinned in env): no
      regression on the 415 / +19-new = 434 passing baseline.
- [x] Pipeline run completed end-to-end on real Norman 2019 (SHA
      `d4bedb53…`) + real pretrain ckpt (SHA `416e8b1a…`) on MPS,
      ~11 min, no NaN/Inf.
- [x] Pipeline tripwires verified:
      `n_pert_included = 236 ≥ 30`, `n_boot = 1000`,
      `CI_width_real = 0.024 < 0.30`, latents finite.
- [x] All required JSON keys present in `.github/phase6_5d_rsa.json`:
      `rsa_real_mean`, `rsa_real_ci95`, `rsa_random_mean`,
      `rsa_random_ci95`, `delta`, `delta_ci95`, `outcome`,
      `n_pert_included`, `n_pert_dropped`, `perturbation_col`,
      `ntc_label`, `n_boot`, `random_seeds`, `ckpt_real_sha`,
      `interpretation_notes`.
