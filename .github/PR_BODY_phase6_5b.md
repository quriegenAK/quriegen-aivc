# Phase 6.5b: linear-probe rerun — real Norman 2019 — BLOCKED (numerical instability)

## Summary

Phase 6.5b attempted to rerun the three-arm linear probe ablation against real
Norman 2019 data (`data/norman2019_aligned.h5ad`, 111,445 cells × 36,601 genes).

**Run 1 (real checkpoint, pretrained arm) completed but produced r2_top50_de =
−2.45 × 10¹³, which is numerically meaningless. Runs 2 and 3 were not executed.
No gate decision (PASS/SOFT/FAIL) is issued. Phase 7 remains blocked.**

## What was verified

- Real Norman 2019 data loaded correctly — no SYNTHETIC FALLBACK warning
- Correct checkpoint loaded — SHA tripwire passed: `416e8b1a...`
- W&B authenticated and logged: https://wandb.ai/quriegen/aivc-linear-probe/runs/ralbtr6u
- n_cells = 111,445 (89,156 train / 22,289 test with seed=17)
- n_genes_nonzero = 17,956 (structural zeros filtered correctly)
- Union DE gene set = 3,217 genes remapped correctly

## Root cause

`scripts/linear_probe_pretrain.py` standardizes Ridge targets gene-by-gene using
train-set statistics only (`y_sd = Y[tr].std(0) + 1e-8`). With real Norman 2019
count data (0–3,718 raw counts, **not log-normalized**):

- **164 genes** have zero expression in all 89K train cells but nonzero values
  in test cells. For these, `y_sd ≈ 1e-8` and test values become `1e8–2e8`
  after "standardization".
- **2,473 genes** have train std < 0.01 (near-epsilon amplification).
- sklearn's `r2_score(multioutput='uniform_average')` is dominated by these
  blown-up genes, producing r2_top50_de = −2.45 × 10¹³.

Because all three arms share the same Y matrix and train/test split, all three
would have produced the same numerically catastrophic results.

## Required fix (phase-6.5c)

Two changes to `scripts/linear_probe_pretrain.py` in `run_condition`:

1. **Log-normalize:** `X = np.log1p(X)` after `_load_dataset` returns, before
   latent extraction and standardization.
2. **Filter Y to train-expressed genes:** after the 80/20 split, compute
   `nonzero_in_train = (Y[tr] != 0).any(axis=0)` and subset Y + remap de_indices.

## Files changed

- `.github/PHASE6_5_RESULTS.md` — overwritten with full Phase 6.5b diagnosis;
  prior synthetic-run values retained under "Prior result" header for audit trail
- `.github/REAL_DATA_BLOCKERS.md` — Phase 6.5 section updated with 6.5b attempt
  and numerical-instability blocker
- `.github/PR_BODY_phase6_5b.md` — this file

## Phase 7 status

**BLOCKED** — pending Phase 6.5c (probe script fix + valid three-arm rerun).
