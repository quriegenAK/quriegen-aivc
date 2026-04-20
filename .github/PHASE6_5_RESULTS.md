# Phase 6.5 results — real-data linear-probe gate

> **NOTE (2026-04-16):** This file replaces the prior synthetic-data result from
> Phase 6.5a. The Phase 6.5b rerun attempted to run all three arms against the
> real Norman 2019 aligned dataset. Run 1 (real checkpoint, pretrained arm)
> completed successfully from a data-loading standpoint (no SYNTHETIC FALLBACK
> warning; correct ckpt SHA) but produced catastrophically wrong R² values due
> to a numerical instability bug in `scripts/linear_probe_pretrain.py`. Runs 2
> and 3 were not executed — they share the same dataset and train/test split, so
> they would have produced equally meaningless results. **No gate decision can be
> issued from Phase 6.5b.** Phase 7 remains blocked. See diagnosis below.

---

## Prior result (synthetic — invalidated)

All three arms in Phase 6.5a ran on **synthetic fallback data** (no real
`norman2019_aligned.h5ad` was wired in at that time) with mismatched `n_genes`
(36601 / 2000 / 512 across the three arms). Result was FAIL with negative R²
across all arms (-0.4617 / -0.4269 / -0.2627). Not biologically interpretable.
Retained for audit trail; original W&B URLs preserved below.

| Run     | ckpt SHA (8) | top-50 DE R² | ΔR² vs random         | W&B URL                                                   |
|---------|--------------|--------------|-----------------------|-----------------------------------------------------------|
| Real    | `416e8b1a`   | -0.4617      | -0.1991 (rel -75.79%) | https://wandb.ai/quriegen/aivc-linear-probe/runs/nbc8telz |
| Mock    | `c0d9715d`   | -0.4269      | -0.1643 (rel -62.54%) | https://wandb.ai/quriegen/aivc-linear-probe/runs/5ppw0qr2 |
| Random  | n/a          | -0.2627      | 0 (baseline)          | https://wandb.ai/quriegen/aivc-linear-probe/runs/bhukayvo |

---

## Phase 6.5b attempt (real Norman 2019 — BLOCKED, numerical instability)

### Setup

- Dataset: Norman 2019 (real, aligned to PBMC10k vocabulary)
- n_cells: 111,445
- n_genes_aligned: 36,601 (full PBMC10k vocabulary)
- n_genes_intersection: 21,940 (see `data/norman2019_aligned.h5ad.meta.json`)
- n_genes_nonzero_in_dataset: 17,956 (structural zeros filtered by probe)
- Eval metric: R² on top-50 DE genes (precomputed per perturbation, union = 3,217 genes)
- Seed: 17 (80/20 train/test split → 89,156 train / 22,289 test cells)
- Probe: Ridge (alpha=1.0, solver=svd) on encoder latent embeddings (dim=128)

### SHA tripwire — PASSED

```
Real  ckpt SHA: 416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e  ✓
```

Correct checkpoint loaded. Tripwire did not fire.

### Run 1 result (real checkpoint, pretrained arm)

- W&B URL: https://wandb.ai/quriegen/aivc-linear-probe/runs/ralbtr6u
- r2_top50_de: **-24,496,977,215,488** (−2.45 × 10¹³) ← INVALID
- r2_overall: **-1,188,906,112** (−1.19 × 10⁹) ← INVALID
- pearson_overall: 2.2 × 10⁻⁶ ≈ 0
- probe_fit_seconds: 151.3
- SYNTHETIC FALLBACK warning: **not triggered** ✓

Run exited 0 but results are numerically meaningless. Runs 2 (mock) and 3
(random) were not executed — see "Root cause" below.

### Root cause — numerical instability in per-gene standardization on raw counts

The `run_condition` function in `scripts/linear_probe_pretrain.py` standardizes
Ridge targets gene-by-gene using train-set statistics:

```python
y_mu, y_sd = Y[tr].mean(0), Y[tr].std(0) + 1e-8
Y_te = (Y[te] - y_mu) / y_sd
```

For this to be well-behaved, every gene in `Y` must have nonzero variance in the
**train split**. Diagnostic run (seed=17, 80/20 split) on the real Norman 2019
data revealed:

| Statistic | Value |
|---|---|
| Total filtered genes in Y | 17,956 |
| Genes with train-set std == 0 (all-zero in train) | **164** |
| Genes with train-set std < 0.01 (near-epsilon) | 2,473 |
| X value range | 0 – 3,718 (raw counts, NOT log-normalized) |

For the 164 all-zero-in-train genes, `y_sd ≈ 1e-8`. Test-set expression values
of 1–2 raw counts become `1e8–2e8` after "standardization". Since the Ridge
model was trained on effectively-zero target values, `Y_hat ≈ 0` for these genes.
This gives `SS_res ≫ SS_tot` and, since sklearn's `r2_score` uses
`multioutput='uniform_average'`, these 164 genes dominate the mean R² and
produce the observed `−2.45 × 10¹³`.

Root causes:

1. **Raw counts, not log-normalized.** `X.max() = 3,718` vs `log1p(3718) = 8.2`.
   The probe was designed assuming roughly Gaussian-distributed inputs and
   targets. Count data with extreme right-skew violates this assumption.
2. **Train-only `y_sd`.** The nonzero-gene mask uses the full 111K-cell dataset,
   so genes that are very rarely expressed pass the structural-zero filter even
   if all their expressing cells land in the 20% test split by chance.
3. **The union DE gene set (3,217 genes) spans low-expression genes.** The
   `r2_top50_de` metric evaluated on this union is the most severely affected
   metric.

Because all three runs share the same dataset, train/test split, and Y matrix,
Runs 2 (mock) and 3 (random) would produce the same blow-up pattern. The blow-up
magnitude varies across runs (since `Y_hat` depends on the encoder), so the ΔR²
vs random values would also be meaningless rather than cancelling out.

### Required fix before Phase 6.5c

Two changes to `scripts/linear_probe_pretrain.py` (in `run_condition`):

1. **Log-normalize counts** — `X = np.log1p(X)` immediately after `_load_dataset`
   returns, before `_extract_latents` and before computing `y_mu / y_sd`. This
   compresses the 0–3718 count range to 0–8.2 and eliminates extreme outliers.
2. **Filter Y to train-expressed genes** — after the train/test split, compute
   `nonzero_in_train = (Y[tr] != 0).any(axis=0)` and subset
   `Y = Y[:, nonzero_in_train]`, with corresponding remap of `de_indices`.
   This ensures no blow-up genes enter the standardization.

Either fix alone prevents the catastrophic blow-up. Both together make the probe
robust to sparse count data.

---

## Phase 6 gate decision

**BLOCKED — numerical instability in probe script. No gate decision issued.**

Date: 2026-04-16.

The Phase 6.5b run loaded the correct checkpoint (SHA tripwire passed), used real
Norman 2019 data (no SYNTHETIC FALLBACK), and completed without crashing. However,
raw-count data combined with per-gene train-only standardization produces
`r2_top50_de = −2.45 × 10¹³`, which is numerically meaningless.

Next step: open `phase-6.5c` branch with the two-line fix in
`scripts/linear_probe_pretrain.py` described above, then rerun the three-arm
table. Gate decision (PASS / SOFT / FAIL) deferred to Phase 6.5c.

Phase 7 remains blocked.

---

## Phase 6.5c — LOCKED v2 (Option B + intersection mask)

- **Date:** 2026-04-19
- **Branch:** `phase-6.5c`
- **Commit 3 SHA:** `60624a0b8d219b03d08f03413fc4442be9fd7e9f` (code + test + intersection-gate script)
- **Commit 4 SHA:** _this commit_ (results + docs)
- **Spec:** [prompts/phase6_5c_fix.md](../prompts/phase6_5c_fix.md) (LOCKED v2)

### Evaluation contract (summary)

- Per-run train-split filter `mask_tr = Y_log1p[tr].std(0) > 1e-7`, applied
  symmetrically to Ridge fit and scoring (replaces the v1 `y_sd` assertion).
- Metric: `variance_weighted` R² on **un-standardized** `log1p(Y)`; primary
  gate is DE top-50 on the intersection of per-run masks across successful
  runs.
- Seeds `{3, 17, 42}` × arms `{real, mock, random}` = 9 runs planned.

### Run outcome (9 planned, 6 succeeded, 3 blocked)

| seed | arm    | status     | r²_top50_de_vw | r²_overall_vw | n_kept / 17956 | n_de_kept / 3217 | W&B URL |
|------|--------|------------|----------------|----------------|----------------|-------------------|---------|
| 3    | real   | ✅          | +0.010889      | +0.163044      | 17796          | 3214              | https://wandb.ai/quriegen/aivc-linear-probe/runs/rcvocklj |
| 3    | mock   | ❌ BLOCKED  | —              | —              | —              | —                 | https://wandb.ai/quriegen/aivc-linear-probe/runs/ounzfenn |
| 3    | random | ✅          | +0.014190      | +0.181075      | 17796          | 3214              | https://wandb.ai/quriegen/aivc-linear-probe/runs/jaxexby4 |
| 17   | real   | ✅          | +0.010938      | +0.162853      | 17792          | 3215              | https://wandb.ai/quriegen/aivc-linear-probe/runs/w88jxg03 |
| 17   | mock   | ❌ BLOCKED  | —              | —              | —              | —                 | https://wandb.ai/quriegen/aivc-linear-probe/runs/qmdhjerz |
| 17   | random | ✅          | +0.014133      | +0.180500      | 17792          | 3215              | https://wandb.ai/quriegen/aivc-linear-probe/runs/0riulwgo |
| 42   | real   | ✅          | +0.010689      | +0.162759      | 17762          | 3212              | https://wandb.ai/quriegen/aivc-linear-probe/runs/on93w3ls |
| 42   | mock   | ❌ BLOCKED  | —              | —              | —              | —                 | https://wandb.ai/quriegen/aivc-linear-probe/runs/dtwl6ey5 |
| 42   | random | ✅          | +0.014098      | +0.180833      | 17762          | 3212              | https://wandb.ai/quriegen/aivc-linear-probe/runs/ql2qryyz |

Runtime tripwires on every successful run: `z_sd_min > 0.81` (floor 1e-6),
`|r²_top50_de_vw| < 0.02` (cap 2.0). Both pass.

### Mock arm — BLOCKED (pre-existing ckpt / dataset feature-space mismatch)

```
RuntimeError: Pretrained encoder n_genes=2000 does not match dataset n_genes=36601.
```

- Ckpt: `checkpoints/pretrain/pretrain_encoders_mock.pt`
- SHA: `c0d9715dbc76a6ecab260fe09ca5173ee7fdf6eb640538eac0f9024399a90b4e`
- The mock ckpt was regenerated at `n_genes=2000` in Phase 6.5a (synthetic
  fallback on the pretrain script's defaults). Norman 2019 aligned is
  `n_genes=36601`. The failure is at the feature-space compatibility check
  in `scripts/linear_probe_pretrain.py:361`, **before** the LOCKED v2 code
  path — so this is not a LOCKED v2 tripwire trip and was not retried per
  the no-patching HARD RULE.
- See [REAL_DATA_BLOCKERS.md §Phase 6.5c](REAL_DATA_BLOCKERS.md) for the
  resolution plan (re-pretrain mock baseline at `n_genes=36601`).

### Intersection gate (2 arms: real, random — mock missing)

Run via `scripts/compute_intersection_gate.py`, relaxed to accept
`N ≥ 6` artifacts. Writes [`.github/phase6_5c_gate.json`](phase6_5c_gate.json)
(committed — single source of truth for the gate numerics).

- `n_runs_used`: **6** (3 real + 3 random)
- `runs_skipped`: `mock × {3, 17, 42}`
- `intersection_mask_size`: **17530** / 17956 (97.6%)
- `n_de_kept_intersection`: **3208** / 3217 (99.7%)

| arm    | G_top50_de_intersection | G_overall_intersection | n_seeds |
|--------|-------------------------|------------------------|---------|
| real   | **+0.010839**           | +0.162885              | 3       |
| mock   | BLOCKED                 | BLOCKED                | 0       |
| random | **+0.014140**           | +0.180802              | 3       |

Deltas (primary gate on real vs random):

- `Delta_real_vs_random_de`      = **−0.003302**
- `Delta_real_vs_random_overall` = −0.017917
- PASS threshold (`|G_random|=0.01414 > 0.01` → `0.05·|G_random|`) = +0.000707

### Checkpoints used

| arm    | ckpt path                                       | SHA-256 (8) | status                         |
|--------|-------------------------------------------------|-------------|--------------------------------|
| real   | `checkpoints/pretrain/pretrain_encoders.pt`     | `416e8b1a`  | ✅ loaded, tripwire passed     |
| mock   | `checkpoints/pretrain/pretrain_encoders_mock.pt`| `c0d9715d`  | ❌ n_genes=2000 ≠ 36601        |
| random | n/a                                             | n/a         | ✅ fresh `SimpleRNAEncoder` init |

### Decision — **FAIL**

`Delta_real_vs_random_de = −0.003302 ≤ 0` → **FAIL** per the LOCKED v2 gate
(see [prompts/phase6_5c_fix.md](../prompts/phase6_5c_fix.md) §Gate decision).

### Interpretation

- **G_real < G_random by 0.0033 absolute on DE top-50.** The pretrained
  encoder does not transfer to Norman 2019 perturbation prediction — it
  sits below the untrained-init floor on the primary metric.
- **Both arms are near the noise floor (~1% variance explained on top-50
  DE).** The overall-gene R² is 16–18%; DE top-50 is where perturbation
  signal should concentrate, and it does not.
- **FAIL signal is consistent with one or more of:** (i) lineage mismatch
  (PBMC10k pretrain → K562 Norman transfer), (ii) objective mismatch
  (masked-recon pretrain vs perturbation-response target), (iii) encoder
  architecture ceiling. Disambiguation is Phase 6.5d scope.

### Residual uncertainty

`variance_weighted` on pooled DE genes does not distinguish
perturbation-specific signal from cross-perturbation shared variance.
Per-perturbation evaluation is deferred to Phase 6.5d regardless of this
gate outcome.

Phase 7 remains blocked. Phase 6.5d queued.

## Phase 6.5d — RSA (LOCKED), dated 2026-04-19

### Why RSA

Phase 6.5c linear probe FAILed on Norman 2019 (`Δ_real − random_de
= −0.003302`, pretrained below untrained-init floor on top-50 DE,
variance-weighted R²). Three non-mutually-exclusive hypotheses for
the FAIL: (1) objective mismatch, (2) lineage mismatch, (3) architecture
ceiling. RSA tests (1) and (2) directly without training: it asks
whether the pretrained latent geometry agrees with the perturbation
response geometry on the *same* perturbation set, independent of any
probe head.

### Locked contract recap

- Aggregation: per-perturbation centroid (Option A). Latent centroid
  `z_p` = mean of pretrained encoder forward pass over cells with
  perturbation `p`. Response centroid `r_p` = mean of
  `log1p(X_filtered)` over the same cells, minus the NTC centroid in
  the same space. Structural-zero filter retained 17,956 / 36,601 genes
  for response space; encoder still sees the full 36,601 raw counts.
- Metric: pairwise cosine distance matrices over non-NTC perturbations,
  flatten upper triangles, Spearman correlation = RSA.
- Bootstrap: 1000 pair-resamples for real arm; 5 random inits (seeds
  0..4) with 1000 resamples each (5,000 aggregated) for random arm;
  paired pair-bootstrap of Δ with 1000 resamples (Δ_i = real_i −
  mean(random_i across 5 inits)). 95% CIs are 2.5/97.5 percentiles.
- Interpretation: pre-registered six-branch rule
  ({A, B, C, C_WEAK, D, INCONCLUSIVE}); thresholds locked before any
  number was observed.

### Results

| Quantity        | Point | 95% CI | Width |
|-----------------|-------|--------|-------|
| RSA_real        | **−0.0584** | [−0.0702, −0.0467] | 0.024 |
| RSA_random      | **+0.0193** | [−0.0076, +0.0529] | 0.061 |
| Δ = real − rand | **−0.0777** | [−0.0832, −0.0725] | 0.011 |

Per-seed RSA_random: seed 0 = +0.0002, seed 1 = +0.0179, seed 2 =
+0.0264, seed 3 = +0.0053, seed 4 = +0.0467.

- `n_perturbations_included = 236` (non-NTC, all passing
  MIN_CELLS_PER_PERT = 20). `n_perturbations_dropped = 0`.
- `n_pairs = 27,730`.
- `n_boot = 1000` per arm.
- `ckpt_real_sha = 416e8b1a…`. `data_sha = d4bedb53…`.
- Pre-flight observed `n_perturbations_total = 237` vs the spec-noted
  `105`. Column (`perturbation`), NTC label (`control`), and SHA all
  match; the count is a spec-documentation artifact, not a data
  provenance regression. Documented in
  `.github/phase6_5d_rsa.json::schema_count_warning`.

### Outcome

**D — Active distractor bias**, dated 2026-04-19.

Δ = −0.0777 < −0.05 with `0 ∉ CI_Δ = [−0.0832, −0.0725]` (CI excludes
zero with margin) and `CI_width_real = 0.024 < 0.30` (no
sampling-noise override). The pretrained encoder's latent geometry is
*anti-correlated* with the Norman 2019 perturbation-response
geometry, while the random-init baseline sits weakly positive
(≈ +0.019). PBMC10k pretraining is not merely failing to capture
K562 perturbation structure (that would be outcome C); it is
actively re-organising the latent space along axes that contradict
the perturbation geometry.

This rules out the "objective mismatch but harmless" reading of the
6.5c FAIL. It is consistent with both lineage mismatch (PBMC primary
immune vs K562 erythroleukemia) and objective mismatch (cell-identity
pretraining task is anti-aligned with perturbation-response geometry
on this dataset). RSA cannot disentangle those two — that is a 6.5e
design question.

### What this rules in / rules out for 6.5e

- **Architecture ceiling (hypothesis 3) — DEMOTED.** With dim-128
  latents the random-init baseline already sits at ≈0; capacity is not
  what is preventing pretrained features from agreeing with
  perturbation geometry. Scaling latent width is unlikely to flip Δ
  positive on its own.
- **Probe redesign on the existing checkpoint — RULED OUT.** A negative
  Δ in a training-free analysis means no linear head (Ridge, logistic,
  MLP probe) operating on the *current* latents can recover
  perturbation structure better than a random init operating on the
  same cells. 6.5e must change the latents, not the head.
- **Lineage-matched pretraining — VIABLE.** K562 / Perturb-Seq corpus
  pretraining would test whether the negative-Δ signal is lineage-
  driven.
- **Perturbation-aware objective — VIABLE.** Contrastive (perturbation
  vs control), supervised perturbation prediction, or response-space
  reconstruction objectives would test whether the negative-Δ signal
  is objective-driven.

### What this PR does NOT do

- No retraining, no fine-tuning, no checkpoint modification.
- No probe of any kind on the latents (RSA is a training-free
  similarity test, not a regression).
- No change to `SimpleRNAEncoder` or any encoder code.
- No 6.5e implementation. The outcome scopes 6.5e; 6.5e is a separate
  PR.

Full machine-readable result: `.github/phase6_5d_rsa.json`. Bootstrap
distributions and per-seed centroids: `experiments/phase6_5d/artifacts/`
(gitignored).

---

## Phase 6.5e — contrastive fine-tune (E1-rev1)

### Outcome: E1-NULL

Weight-change fine-tune of the existing `416e8b1a…` pretrain ckpt
under a pure cross-modal contrastive objective (mask
`{recon:0, recon:0, contrastive:1.0, aux:0}`, τ=0.1, B=256, 1 epoch,
LR=1e-4). Pre-registered outcome rule (Gate 2, row 3) fires on
`|R_c − R_r| < 0.05 ∧ 0 ∈ CI_Δ`.

| Quantity | Point | 95% CI | Width |
|---|---|---|---|
| R_c = RSA_real contrastive | **−0.0621** | [−0.0739, −0.0503] | 0.024 |
| R_r = RSA_real reconstruction (frozen 6.5d) | −0.0584 | [−0.0702, −0.0467] | 0.024 |
| RSA_random (frozen 6.5d) | +0.0193 | [−0.0076, +0.0529] | 0.061 |
| Δ = R_c − R_r | **−0.0037** | [−0.0154, +0.0081] | 0.024 |
| Δ = R_c − RSA_random | −0.0814 | [−0.0931, −0.0696] | 0.024 |

R_c CI excludes zero (still anti-correlated). Δ_{c−r} CI includes
zero (no meaningful shift vs reconstruction baseline). Parallel 6.5d
classifier on R_c alone still returns "D — active distractor bias"
(Δ vs random is significantly negative) — the contrastive fine-tune
did not flip the sign.

### Setup

- Parent ckpt SHA: `416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e` (T1 ✓)
- Contrastive ckpt SHA: `6084d5186cbd3dc942497d60926cda7a545931c7da5d7735ba32f555b73349ee`
- Data: `data/pbmc10k_multiome.h5ad` (11,898 cells, 36,601 genes, 115,720 peaks)
- Eval: Norman 2019 aligned (SHA `d4bedb53…`), 236 non-NTC perturbations, 27,730 pairs
- Stage: `joint_contrastive_only_e1`
- Seed: 3 (full scope: torch + numpy + random + cudnn + DataLoader)
- Projection init: `parent` (reused parent `MultiomePretrainHead.rna_proj`/`atac_proj` via deep copy)
- Optimizer: AdamW, LR 1e-4, WD 1e-4, grad clip 1.0 global
- Wall clock: 83.5s fine-tune + 689.4s RSA on MPS
- Loss: 0.588 → 0.274 (monotonically decreasing across 47 batches)
- W&B run: https://wandb.ai/quriegen/aivc-pretrain/runs/5qaltcx9

### Tripwires (all passed)

| Tripwire | Result |
|---|---|
| T1 parent SHA | pass (observed = expected) |
| T2 probe batch | pass (256 cells, seed=3, cached) |
| T3 probe drift | pass (mean\|Δz_rna\|=0.151, mean\|Δz_atac\|=0.068 ≫ 1e-4) |
| T4 collapse | pass (per-dim std min ≥ 0.60; cross-dim std mean ≥ 1.67) |
| T5 online NaN/Inf | pass (47/47 batches) |
| T6 weight mask | pass ({recon:0, recon:0, contrastive:1.0, aux:0}) |
| T7 drift ratios | rna_enc=0.56%, atac_enc=1.37%, rna_proj=1.50%, atac_proj=1.60% |

### Interpretation

The pure-contrastive weight mask on PBMC10k leaves the
cross-geometry relationship to K562 Norman 2019 essentially
unchanged: R_c is indistinguishable from R_r at the 0.05 threshold,
and Δ_{c−r}'s 95% CI straddles zero. Put differently — once the
parent was *already* trained with cross_modal_infonce (weight 0.5
of four terms), ablating the reconstruction and peak-aux terms on
top of the parent does not add a new alignment signal. The
encoders drift (T7 0.5–1.6 % parameter change) but the drift is
into neighboring, similarly anti-correlated geometry.

This is the specific hypothesis the E1-rev1 weight-change study
was designed to test — and it returned NULL.

### 6.5f scope implication

- **Scoped in.** Lineage-matched pretraining (E2 direction: K562 /
  Perturb-Seq corpus). With the E1 re-weighting falsified, lineage
  mismatch is the dominant remaining hypothesis for 6.5d's outcome D.
- **Demoted (but not dead).** Other objective-only manipulations on
  PBMC (different τ, projection head, perturbation-aware contrastive
  if we obtain pseudo-labels). E1-rev1 specifically tested the
  weight-mask variant; narrow-scope objective work is still
  technically available but strictly deprioritized vs E2.
- **Ruled out (for E1-rev1 specifically).** Weight-mask re-weighting
  of the parent's existing contrastive term as a sufficient fix —
  the pre-registered NULL outcome fired cleanly.

Full machine-readable result: `.github/phase6_5e_rsa.json`.
Tripwire record: `experiments/phase6_5e/tripwires.json`.
Bootstrap distributions + probe batch:
`experiments/phase6_5e/artifacts/` (gitignored).

## Phase 6.5f-disambig — frozen-projection disambiguation

**Outcome: F-NULL** (pre-registered). Single-variable disambiguation of
6.5e's E1-NULL outcome returned F-NULL — projection absorption was
**not** the cause. Under frozen projections (all gradient routed into
the encoders), RSA on Norman 2019 remains indistinguishable from the
6.5e contrastive-only baseline.

### Decision chain leading to 6.5f

Phase 6.5e (`joint_contrastive_only_e1`, weight mask
`{recon:0, recon:0, contrastive:1.0, aux:0}`) returned E1-NULL:
`R_c = −0.0621`, `Δ_{c−r} = −0.0037`, `0 ∈ CI_Δ`. The T7 drift-ratio
diagnostic (pre-registered follow-up rule) fired —
`drift_ratio_rna = rna_enc/rna_proj = 0.005640/0.015023 = 0.376 ≪ 1`,
indicating RNA encoder absorbed ~3× less gradient mass than `rna_proj`
during 6.5e.

| 6.5e T7 (reference) | Relative weight drift |
|---|---|
| `rna_encoder` | 0.56% |
| `atac_encoder` | 1.37% |
| `rna_proj` | 1.50% (absorbed ~3× more than encoder) |
| `atac_proj` | 1.60% (near-parity with encoder) |

Two competing interpretations of 6.5e that 6.5e alone cannot separate:
**(1) lineage interpretation** — PBMC10k manifold lacks axes aligning
with K562 perturbation response, and no objective manipulation on PBMC
can fix this (scope-in E2); **(2) projection-absorption
interpretation** — encoder direction never shifted meaningfully because
projection heads absorbed gradient, which is discarded at ckpt save
time (a cheap freeze-projections recipe could still move RSA).

6.5f is the minimal single-variable test: freeze `rna_proj` /
`atac_proj` (`requires_grad=False`, excluded from AdamW), keep
everything else bit-identical to 6.5e.

### Result

| Quantity | Point | 95% CI | Width |
|---|---|---|---|
| R_b = RSA_real frozen-proj | **−0.0621** | [−0.0738, −0.0504] | 0.0234 |
| R_c_e1 = RSA_real contrastive-only (frozen 6.5e) | −0.0621 | [−0.0739, −0.0503] | 0.0235 |
| R_r = RSA_real reconstruction-dominant (frozen 6.5d) | −0.0584 | [−0.0702, −0.0467] | 0.0236 |
| RSA_random (frozen 6.5d) | +0.0193 | [−0.0076, +0.0529] | 0.061 |
| **Δ_{b−c_e1} (primary gate)** | **−0.00005** | [−0.0117, +0.0117] | 0.023 |
| Δ_{b−r} | −0.0037 | [−0.0154, +0.0080] | 0.023 |
| Δ_{b−random} | −0.0814 | [−0.0931, −0.0697] | 0.023 |

The primary gate (`Δ_{b−c_e1}`) has a point estimate of `−5.4e-05`
(essentially zero) and a CI that straddles zero symmetrically. `|Δ|
= 0.00005 ≪ 0.05` and `0 ∈ CI_Δ` — Gate 2 row 3 (F-NULL) fires cleanly.

### Encoder drift comparison — 6.5e vs 6.5f

| Modality | 6.5e drift% | 6.5f drift% | Δ |
|---|---|---|---|
| `rna_encoder` | 0.5640% | 0.5729% | +0.0089% |
| `atac_encoder` | 1.3673% | 1.3948% | +0.0275% |
| `rna_proj` | 1.5023% | 0.0000% (frozen) | — |
| `atac_proj` | 1.6045% | 0.0000% (frozen) | — |

Freezing projections routed all gradient into the encoders as designed
(T8-a/b both pass; T3 probe drift on the encoders went **up**:
`mean|Δz_rna|` 0.151 → 0.247, `mean|Δz_atac|` 0.068 → 0.086). But the
additional encoder motion is tiny in *relative weight* terms
(~0.01–0.03 pp) and produces no detectable RSA shift. The encoder
direction is stable under contrastive pressure on PBMC regardless of
gradient routing.

### Setup

- Parent ckpt SHA: `416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e` (T1 ✓)
- 6.5e baseline ckpt SHA (reference): `6084d5186cbd3dc942497d60926cda7a545931c7da5d7735ba32f555b73349ee`
- 6.5f frozen-proj ckpt SHA: `936dfa44071322f7d62c3e698e4395e875fd02a201d171647ad8734dccfa54d7`
  (distinct from parent and 6.5e)
- Data: `data/pbmc10k_multiome.h5ad` SHA `0e1e7689…` (T2 ✓)
- Probe batch: reused 6.5e cache; index SHA `27a906d0…` (T2 ✓)
- Eval: Norman 2019 aligned (SHA `d4bedb53…`), 236 non-NTC perturbations, 27,730 pairs
- Stage: `joint_contrastive_only_e1` (reused from 6.5e; no new stage)
- Seed: 3 (full scope)
- Projection init: `parent` (byte-identical to 6.5e's; verified at runtime)
- Trainable params: 19,925,677 (encoders only under freeze)
- Optimizer: AdamW, LR 1e-4, WD 1e-4, grad clip 1.0 global
- Wall clock: 54.5s fine-tune + 479.8s RSA on MPS
- Loss: 0.588 → 0.283 (decreased across 47 batches)
- W&B run: https://wandb.ai/quriegen/aivc-pretrain/runs/5a5wspol

### Tripwires (all passed)

| # | Observed | Pass |
|---|---|---|
| T1 parent SHA | `416e8b1a…` == expected | ✓ |
| T2 probe batch | data SHA `0e1e7689…` ✓; index SHA `27a906d0…` ✓ | ✓ |
| T3 probe drift | mean\|Δz_rna\|=0.247, mean\|Δz_atac\|=0.086 (≫ 1e-4) | ✓ |
| T4 collapse | per-dim std min ≥ 0.57; cross-dim std mean ≥ 1.66; finite | ✓ |
| T5 online NaN/Inf | 47/47 batches clean | ✓ |
| T6 weight mask | `{recon:0, recon:0, contrastive:1.0, aux:0}` | ✓ |
| T7 drift ratios | rna_enc=0.573%, atac_enc=1.395%, rna_proj=0.0, atac_proj=0.0; drift_ratio_*="inf" | ✓ |
| **T8-a** optimizer exclusion | no proj param id in optimizer; no proj `requires_grad=True` | ✓ |
| **T8-b** max\|ΔW\|=0 exact | `max\|ΔW\|_rna=0.0`, `max\|ΔW\|_atac=0.0` | ✓ |

### Interpretation

Freezing the projection heads — which eliminates gradient absorption
by construction, as T7 and T8 both confirm — produces **no change in
RSA** on K562 Norman 2019. The 6.5e T7 diagnostic (projection drift ≫
encoder drift) was a real optimizer-dynamics observation, but it is
not load-bearing on the cross-cell-type geometry measurement. The
PBMC10k manifold does not contain axes that align with K562
perturbation response, and swapping which parameters absorb the
contrastive gradient does not create such axes.

Interpretation **(2)** (projection absorption) is falsified as a
sufficient explanation. Interpretation **(1)** (lineage mismatch)
stands — with higher confidence than after 6.5e alone, because the
projection-absorption alternative has now been explicitly excluded
by a direct-intervention test.

### 6.5g / E2 scope implication

- **Committed to:** E2 (K562 / Perturb-Seq lineage pretraining). With
  both within-objective (6.5e) and within-optimizer-routing (6.5f)
  manipulations on PBMC returning NULL, lineage is the dominant
  remaining hypothesis. Gate-2 outcome F-NULL triggers the
  "commit to E2 with higher confidence" branch.
- **Deprioritized:** PBMC-recipe optimization (encoder-LR sweeps,
  τ sweeps, alternative projection heads, longer training on
  PBMC10k). F-WIN or F-PARTIAL would have pivoted here; F-NULL
  does not.
- **Not ruled out:** cell-type-aware contrastive objectives that
  introduce pseudo-labels / perturbation-aware negatives. Outside
  6.5f's scope and the spec explicitly forbids new experiments
  "while we're here."

Full machine-readable result: `.github/phase6_5f_rsa.json`.
Tripwire record: `experiments/phase6_5f/tripwires.json`.
Bootstrap distributions + probe batch:
`experiments/phase6_5f/artifacts/` (gitignored).
