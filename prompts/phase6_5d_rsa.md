# Phase 6.5d — RSA Objective-Mismatch Test (LOCKED)

> **STATUS:** Pre-registered analysis. No training. No new checkpoints.
> Executes a representation similarity analysis (RSA) between the
> pretrained encoder's latent space and the Norman 2019 perturbation
> response space, with 5-init random baseline and 95% bootstrap CIs.
> Outcomes are pre-registered against explicit thresholds; the
> interpretation rule is locked before any number is observed.

## CONTEXT

Phase 6.5c linear probe returned a FAIL on Norman 2019:
`Delta_real − random_de = −0.003302` (pretrained below untrained-init floor
on top-50 DE, variance-weighted R²). Both arms explained 16–18% of overall
gene variance (non-trivial capacity) but ~1% of DE-gene variance (noise
floor). Three non-mutually-exclusive hypotheses for the FAIL:

1. **Objective mismatch** — pretraining captures cell-identity axes, not
   perturbation-response axes. Linear probe cannot recover perturbation
   responses from identity features regardless of cell line or architecture.
2. **Lineage mismatch** — PBMC10k (primary immune) vs Norman 2019 (K562
   erythroleukemia). Pretrained features concentrate on PBMC-specific
   axes that are orthogonal to K562 perturbation axes.
3. **Architecture ceiling** — dim-128 latent too compressed. Demoted
   based on 6.5c overall-R² evidence (capacity clearly > probe gate floor).

RSA tests hypotheses 1 and 2 directly, without training. If RSA_real is
near zero AND below RSA_random, objective mismatch or distractor bias is
confirmed — both redirect 6.5e toward objective redesign rather than
architecture scaling.

Branch: `phase-6.5d` branched off `main` (not off `phase-6.5c` — 6.5d
must be independently mergeable while 6.5c PR #16 is in draft).

## PREREQUISITES

- `main` tip on origin is the current production baseline.
- `data/norman2019_aligned.h5ad` — shape (111,445, 36,601), integer raw counts.
  SHA `d4bedb53…` (verified in 6.5c).
- `checkpoints/pretrain/pretrain_encoders.pt` — SHA
  `416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e`.
- `adata.obs` schema (CONFIRMED via 6.5d run 2026-04-19):
  - Perturbation column: **`perturbation`** (237 distinct values; earlier
    pre-flight of 105 was truncated view from `value_counts().head(10)`).
  - Control label: literal string **`control`** (11,855 cells, ~10.6% of dataset).
  - Non-control perturbations: 236 total. Mix of single-gene (e.g. `KLF1`,
    `BAK1`) and gene-pair combinatorial (e.g. `CEBPE_RUNX1T1`, `TBX3_TBX2`).
  - No `NT` / `non-targeting` / `scramble` variants present; do not scan for them.
- `SimpleRNAEncoder` import path: `aivc.skills.rna_encoder.SimpleRNAEncoder`.
- 6.5c post-structural-zero gene filter: the 17,956-gene subset derived
  in `scripts/linear_probe_pretrain.py::_load_dataset`. Apply the same
  structural-zero filter here for consistency (no split-dependent mask —
  this is RSA, not probe).

## THE CONTRACT (LOCKED)

### Aggregation — Option A (perturbation centroid)

- For each perturbation `p` (including NTC), compute two centroid vectors:
  - **Latent centroid `z_p`:** mean of encoder forward-pass output over
    all cells with that perturbation label. Dim 128.
  - **Response centroid `r_p`:** mean of `log1p(X_filtered)` over the same
    cells, minus the NTC centroid in the same space. Dim = n_genes_filtered
    (~17,956 after structural-zero filter).
- Drop perturbations with fewer than `MIN_CELLS_PER_PERT = 20` cells
  (insufficient to estimate centroid). Log the count.

### Metric

- Pairwise cosine distance matrix in latent space: `D_z[i,j] = 1 − cos(z_i, z_j)`
  over all perturbation pairs (excluding NTC from pair indices; NTC is
  reference, not a perturbation).
- Pairwise cosine distance matrix in response space: `D_r[i,j] = 1 − cos(r_i, r_j)`.
- Flatten the upper triangle of each → vectors of length `n_pert * (n_pert − 1) / 2`.
- **RSA = Spearman correlation between flattened `D_z` and flattened `D_r`.**

### Bootstrap — 95% CIs

- **Real arm:** fixed pretrained ckpt. Pair-bootstrap the RSA:
  - `n_boot = 1000` resamples of perturbation pairs (with replacement) from
    the flattened upper-triangle vectors. Each resample produces one Spearman.
  - 95% CI = 2.5th / 97.5th percentile of the bootstrap distribution.
- **Random arm:** 5 independent random inits, seeds `{0, 1, 2, 3, 4}`.
  - For each init: instantiate a fresh `SimpleRNAEncoder` with the same
    architecture as the pretrained ckpt, run forward pass, compute centroid
    latents, compute RSA_init.
  - Pair-bootstrap each init's RSA with `n_boot = 1000`. Aggregate all
    `5 × 1000 = 5000` bootstrap values into one distribution.
  - 95% CI = 2.5th / 97.5th percentile of the aggregated distribution.
  - `RSA_random_point = mean across 5 inits` (not mean of bootstrap
    distribution — bootstrap is for CI only).
- **Delta CI:** pair-bootstrap delta directly. For each of 1000 resamples:
  resample pair indices, compute both `RSA_real` and `RSA_random[init]`
  on the same pair set (for each of the 5 inits), average the 5 random
  Spearmans for that resample, take the difference. 1000 delta samples.
  95% CI = percentile.

### Interpretation rule (locked before any number observed)

Let `Δ = RSA_real − RSA_random_point`, `CI_Δ = (lower, upper)`,
`CI_width = upper − lower`.

**Step 1 — Interpretability gate:**

- IF `|Δ| < 0.05` AND `CI_Δ` includes 0 → outcome = **INCONCLUSIVE**.
  Signal is below pre-registered magnitude threshold and not statistically
  distinguishable from zero. Action: document, halt, escalate design review
  for 6.5e.
- IF `|Δ| ≥ 0.05` AND `0 ∈ CI_Δ` (CI overlaps zero despite large point
  estimate) → outcome = **INCONCLUSIVE**. Sampling noise too high.
  Action: document, halt, escalate.
- IF `|Δ| ≥ 0.05` AND `0 ∉ CI_Δ` → interpretable. Proceed to Step 2.
- IF `|Δ| < 0.05` AND `0 ∉ CI_Δ` → statistically significant but below
  magnitude threshold. Classify as **outcome C (objective mismatch, weak)**.
  Proceed to 6.5e with mismatch hypothesis confirmed but with magnitude caveat.

**Step 2 — Outcome classification (only if Step 1 says interpretable):**

| Outcome | Primary condition | Secondary (RSA_real) | Interpretation |
|---------|-------------------|----------------------|----------------|
| **A** | Δ > +0.05 | RSA_real > 0.15 | Pretrain captures perturbation structure. 6.5c FAIL is probe-artifact. 6.5e: probe redesign. |
| **B** | Δ > +0.05 | RSA_real ≤ 0.15 | Pretrain helps but weakly. 6.5e: probe redesign + richer head. |
| **C** | −0.05 ≤ Δ ≤ +0.05 | — | Objective mismatch. Pretrain neither helps nor hurts. 6.5e: objective redesign (contrastive / perturbation-aware). |
| **D** | Δ < −0.05 | — | Active distractor bias. PBMC-pretrain erodes K562-relevant axes. 6.5e: lineage-matched pretrain OR objective redesign. |

### What's forbidden

- Fitting any linear map (Ridge, logistic, anything) on top of latents.
  This is RSA only. Pure geometric analysis.
- Changing the encoder architecture or adding normalization layers.
- Applying log1p to encoder input X (encoder expects raw counts — 6.5c lesson).
- Running on a subset of perturbations or cells to "speed up" — full
  Norman 2019 (after structural-zero + MIN_CELLS_PER_PERT filters).
- Reporting any number before the bootstrap CI has been computed.
- Hand-tuning thresholds post-hoc. The pre-registered thresholds are
  locked. Re-interpretation after seeing numbers is a protocol violation.

## TASK

### Step 1 — Branch setup

```
git fetch origin
git checkout -b phase-6.5d origin/main
# Verify off main:
git log --oneline -1  # must show main tip, not 6.5c tip
```

### Step 2 — Pre-flight checks

- Verify `data/norman2019_aligned.h5ad` SHA matches 6.5c expectation.
- Verify `checkpoints/pretrain/pretrain_encoders.pt` SHA == `416e8b1a…`.
- Schema (pre-resolved — do NOT rediscover): perturbation column =
  `perturbation`, control label = literal `control`. Assert both present
  in `adata.obs`. If either is missing, STOP (data provenance regression).
- Log: n_perturbations_total=237, n_control_cells expected ≈11,855,
  n_non_control_perturbations=236. Assert counts match (±1% tolerance).
  (Earlier spec noted 105 / 104; this was a truncated view from
  `value_counts().head(10)`. Corrected post-6.5d run on 2026-04-19.)

### Step 3 — Implement `scripts/phase6_5d_rsa.py`

Responsibilities:

- Load anndata, apply structural-zero filter (same logic as 6.5c
  `_load_dataset`).
- Resolve perturbation labels, NTC mask, perturbation list after
  MIN_CELLS_PER_PERT filter. Log dropped perturbations.
- Compute pseudobulk response centroids `r_p` in log1p space, minus NTC centroid.
- Load pretrained encoder, run forward pass on raw X, compute latent
  centroids `z_p_real`.
- For seed ∈ {0, 1, 2, 3, 4}: instantiate fresh `SimpleRNAEncoder`
  (use `torch.manual_seed(seed)` before instantiation), run forward pass,
  compute `z_p_random[seed]`.
- Compute `D_r`, `D_z_real`, `D_z_random[seed]` — all pairwise cosine
  distance matrices over perturbations (NTC excluded from pair indices).
- Flatten upper triangles. Compute Spearman RSA for real and each random
  init. Compute pair-bootstrap CIs per the contract.
- Compute Δ and CI_Δ per the contract.
- Apply interpretation rule → outcome ∈ {A, B, C, D, INCONCLUSIVE,
  C_WEAK}.
- Write structured output to `.github/phase6_5d_rsa.json`.
- Save full bootstrap distributions + latent centroids to
  `experiments/phase6_5d/artifacts/` (gitignored).

### Step 4 — Write `scripts/lib/rsa.py` helper

Small, testable, side-effect-free module:

```python
def cosine_dist_matrix(X: np.ndarray) -> np.ndarray: ...
def upper_tri(M: np.ndarray) -> np.ndarray: ...
def spearman_rsa(D_a: np.ndarray, D_b: np.ndarray) -> float: ...
def pair_bootstrap_rsa(D_a, D_b, n_boot: int, rng: np.random.Generator) -> np.ndarray: ...
```

### Step 5 — Tests: `tests/test_phase6_5d_rsa.py`

- `test_cosine_dist_matrix_self_zero`: diagonal is 0.
- `test_cosine_dist_matrix_symmetric`: M == M.T.
- `test_spearman_rsa_identity`: RSA(M, M) == 1.0.
- `test_spearman_rsa_shuffle`: RSA(M, shuffled M) ≈ 0 (within 3σ of null).
- `test_pair_bootstrap_shape`: returns array of length n_boot.
- `test_interpretation_rule`: covers each of the 6 outcome branches
  (A, B, C, C_WEAK, D, INCONCLUSIVE) with synthetic Δ / CI inputs.
- `test_ntc_reference_excluded`: NTC not in pair indices.

All tests pass locally before any RSA run.

### Step 6 — Commit 1

Title: `phase-6.5d: RSA pre-registered pipeline (locked contract)`

Files:
- `scripts/phase6_5d_rsa.py`
- `scripts/lib/rsa.py`
- `tests/test_phase6_5d_rsa.py`
- `prompts/phase6_5d_rsa.md` (this file)

`pytest tests/` must pass (no regression on 447 baseline).

### Step 7 — Run

```
python scripts/phase6_5d_rsa.py \
  --adata data/norman2019_aligned.h5ad \
  --real-ckpt checkpoints/pretrain/pretrain_encoders.pt \
  --random-seeds 0,1,2,3,4 \
  --n-boot 1000 \
  --min-cells-per-pert 20 \
  --out-json .github/phase6_5d_rsa.json \
  --art-dir experiments/phase6_5d/artifacts
```

Single forward pass run. Expected wall clock on MPS: <30 min.

### Step 8 — Docs

Files:
- `.github/PHASE6_5_RESULTS.md` — append "## Phase 6.5d — RSA (LOCKED)"
  section. Keep 6.5b BLOCKED and 6.5c FAIL sections verbatim.
- `.github/REAL_DATA_BLOCKERS.md` — append 6.5d subsection only if
  outcome == INCONCLUSIVE (blocker) or if new data-quality issues
  surface (e.g., NTC label ambiguity, dropped perturbations > 20%).
- `.github/PR_BODY_phase6_5d.md` — new, summarizes outcome, RSA_real
  (mean + CI), RSA_random (mean + CI), Δ (point + CI), classification,
  6.5e scope implication.

### Step 9 — Commit 2

Title: `phase-6.5d: RSA results + docs (outcome: <X>)`

Files: `.github/phase6_5d_rsa.json`, `.github/PHASE6_5_RESULTS.md`,
`.github/REAL_DATA_BLOCKERS.md` (if applicable),
`.github/PR_BODY_phase6_5d.md`.

### Step 10 — Push

```
git push -u origin phase-6.5d
```

STOP. Do NOT `gh pr create`. Report outcome to user first.

## VALIDATION

- `pytest tests/` passes (≥447 + new cases).
- RSA pipeline tripwires:
  - `n_pert_included >= 30` (sufficient pairs for bootstrap).
  - `n_boot == 1000` per run (not 100, not 500).
  - `CI_width_real < 0.30` (high CI widths suggest undersampling).
  - Latent centroids have no NaN / Inf.
- `.github/phase6_5d_rsa.json` has keys: `rsa_real_mean`, `rsa_real_ci95`,
  `rsa_random_mean`, `rsa_random_ci95`, `delta`, `delta_ci95`,
  `outcome`, `n_pert_included`, `n_pert_dropped`, `perturbation_col`,
  `ntc_label`, `n_boot`, `random_seeds`, `ckpt_real_sha`,
  `interpretation_notes`.

## FAILURE HANDLING

- **NTC label ambiguity:** if `adata.obs` has multiple candidate NTC
  labels (e.g., both "NT" and "ctrl"), STOP and ask user. Do not auto-resolve.
- **< 30 perturbations pass MIN_CELLS_PER_PERT:** STOP. Suggests dataset
  load issue. Do not reduce threshold.
- **CI_width_real > 0.30:** sampling-noise limited. Outcome auto-defaults
  to INCONCLUSIVE regardless of Δ. Escalate for 6.5e design.
- **RSA_real or RSA_random NaN:** indicates degenerate distance matrix
  (all perturbations mapped to same latent, or response = 0 everywhere).
  STOP. Report.
- **Encoder forward-pass OOM on full Norman 2019:** batch inference
  (batch_size=1024) — acceptable, not a contract change. Document.
- **Any outcome:** do NOT retune thresholds post-hoc. The outcome is
  the outcome. 6.5e scope adjusts to whichever branch was hit.

## PR PREPARATION

PR body template: `.github/PR_BODY_phase6_5d.md`. Required sections:
- Summary (outcome label + one-sentence interpretation).
- Locked contract recap (aggregation, metric, bootstrap, interpretation rule).
- Results table: `RSA_real`, `RSA_random`, `Δ`, all with 95% CIs.
- Outcome classification + pre-registered rationale.
- 6.5e scope implication (1–3 bullets).
- What this PR does NOT do.
- Test plan.

Open as **DRAFT** initially. Same pattern as 6.5c PR #16.
