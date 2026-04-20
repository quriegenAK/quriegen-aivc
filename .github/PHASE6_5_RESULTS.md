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
