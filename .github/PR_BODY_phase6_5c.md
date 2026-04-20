# Phase 6.5c — Linear probe evaluation (LOCKED v2, FAIL)

## Summary

Executed the locked linear-probe contract against Norman 2019 aligned
(111,445 cells × 36,601 genes). **6 of 9 runs succeeded** (real + random,
3 seeds each). The 3 mock-arm runs are BLOCKED by a pre-existing
checkpoint / dataset feature-space mismatch (unrelated to LOCKED v2).
**Decision: FAIL** on the primary real-vs-random gate.

## Locked contract (v2)

- **Filter:** `mask_tr = Y_log1p[tr].std(0) > 1e-7` (train-split,
  symmetric across Ridge fit and scoring). Replaces the v1 `y_sd`
  assertion that tripped on Norman 2019 seed=3 in the prior attempt.
- **Metric:** `variance_weighted` R² on **un-standardized** `log1p(Y)`.
- **Aggregation:** per-run r² (seed-level diagnostic) +
  **intersection-mask gate** computed post-hoc across all successful
  runs by AND-ing per-run `mask_tr` arrays and re-scoring on the shared
  column subset.
- **Seeds:** `{3, 17, 42}`. **Arms:** `{real, mock, random}`.
- **Tripwires:** `z_sd_min > 1e-6`, `|r²_top50_de_vw| < 2.0`,
  `n_kept > 0`, `n_de_kept ≥ 5`.

## Gate result

### Intersection gate (primary, computed across 6 successful runs)

- `n_runs_used`: 6   `runs_skipped`: mock × {3, 17, 42}
- `intersection_mask_size`: 17,530 / 17,956 (97.6% of post-structural-zero cols)
- `n_de_kept_intersection`: 3,208 / 3,217 (99.7%)

| Arm    | G_top50_de | G_overall | Seeds |
|--------|------------|-----------|-------|
| real   | +0.010839  | +0.162885 | 3/3   |
| mock   | BLOCKED    | BLOCKED   | 0/3 (n_genes=2000 vs 36601) |
| random | +0.014140  | +0.180802 | 3/3   |

- `Delta_real_vs_random_de`      = **−0.003302**
- `Delta_real_vs_random_overall` = −0.017917

**Primary gate:** `Delta_real_vs_random_de > 0` (pretrained encoder must
beat the untrained-init floor on DE top-50 `variance_weighted` R²).
Observed: **−0.003302** → **FAIL**.

**Secondary cushion** (irrelevant given `Delta ≤ 0`): had `Delta` been
positive, PASS would have additionally required
`Delta ≥ 0.05·|G_random| = +0.000707`.

### Per-run table (successful runs only)

| seed | arm    | r²_top50_de_vw | r²_overall_vw |
|------|--------|----------------|----------------|
| 3    | real   | +0.010889      | +0.163044      |
| 3    | random | +0.014190      | +0.181075      |
| 17   | real   | +0.010938      | +0.162853      |
| 17   | random | +0.014133      | +0.180500      |
| 42   | real   | +0.010689      | +0.162759      |
| 42   | random | +0.014098      | +0.180833      |

## Spec deviations (accepted)

- The `run_condition` metrics dict exposes `r2_top50_de` and
  `r2_overall` as aliases of `r2_top50_de_vw` / `r2_overall_vw`, plus
  `pearson_overall`, `probe_fit_seconds`, `condition`, `dataset`, and
  related metadata fields, for downstream compatibility with
  `scripts/exp1_linear_probe_ablation.py` (on the forbidden-edits list).
  Without these, the 9-run matrix would KeyError before reaching the
  LOCKED v2 code path.
- `scripts/compute_intersection_gate.py` was relaxed at finalization
  time to accept `N ≥ 6` artifacts (instead of hard-requiring 9) so
  that the pre-existing mock-ckpt blocker does not prevent the
  primary real-vs-random gate from being computed.

## Blockers

See [`.github/REAL_DATA_BLOCKERS.md` §Phase 6.5c](REAL_DATA_BLOCKERS.md)
for two unresolved blockers:

1. **Mock ckpt feature-space mismatch (DATA).** `pretrain_encoders_mock.pt`
   (SHA `c0d9715d`) is at `n_genes=2000`, incompatible with Norman
   aligned `n_genes=36601`. Resolution: re-pretrain mock at the correct
   feature space in Phase 6.5d.
2. **Pretrained encoder transfer gap (CAPABILITY).** `pretrain_encoders.pt`
   (SHA `416e8b1a`) sits below the untrained-init floor on DE top-50
   `variance_weighted` R². Phase 6.5d investigation scope.

## Artifacts

- **Commit 3:** `60624a0` — code + test + post-hoc script
  (`scripts/linear_probe_pretrain.py`,
   `tests/test_linear_probe_numeric_stability.py`,
   `scripts/compute_intersection_gate.py`,
   `prompts/phase6_5c_fix.md`).
- **Commit 4:** _this commit_ — docs + results
  (`.github/PHASE6_5_RESULTS.md`, `.github/REAL_DATA_BLOCKERS.md`,
   `.github/PR_BODY_phase6_5c.md`, relaxed intersection-gate script).
- **W&B project:** `aivc-linear-probe`. All 9 run URLs in
  `PHASE6_5_RESULTS.md` §Phase 6.5c.
- **Checkpoints used:** real (`416e8b1a…`) — passes; mock (`c0d9715d…`)
  — blocked; random — no checkpoint, fresh `SimpleRNAEncoder` init.
- **Gate JSON:** `.github/phase6_5c_gate.json` (committed).

## What this PR does NOT do

- Does not re-pretrain any encoder.
- Does not fix the mock ckpt feature-space mismatch.
- Does not attempt any remediation of the FAIL signal.
- All follow-up (architecture investigation, mock re-pretrain,
  per-perturbation evaluation, Phase 6.5d planning) is out of scope.

## Phase 6.5d scope preview

1. Disambiguate FAIL cause: lineage (PBMC10k → K562) vs objective
   (masked-recon vs perturbation response) vs architecture ceiling.
2. Regenerate mock ckpt at `n_genes=36601` for a complete ablation matrix.
3. Consider per-perturbation evaluation instead of pooled DE top-50.

## Test plan

- [x] `pytest tests/ → 447 passed` on Commit 3.
- [x] `scripts/compute_intersection_gate.py` runs cleanly on 6 artifacts.
- [x] Runtime tripwires (`z_sd_min > 1e-6`, `|r²| < 2.0`) pass on every
      successful run.
- [x] Every successful run loaded the correct ckpt SHA (tripwire).
- [x] Zero `SYNTHETIC FALLBACK` warnings on the successful runs.
