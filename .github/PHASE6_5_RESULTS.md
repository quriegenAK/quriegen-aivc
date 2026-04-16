# Phase 6.5 results — real-data linear-probe gate

## Setup

- dataset: norman2019
- eval metric: R² on top-50 DE genes (per `.github/PR_BODY_phase6.md`)
- seed: 17
- probe: linear (`sklearn.linear_model.Ridge(alpha=1.0)` fit on
  standardized latents; target = standardized expression;
  top-50 DE selected by train-set variance). Matches
  `scripts/linear_probe_pretrain.py::_fit_probe_and_score`.
- loader state: synthetic fallback for `norman2019` (no public
  h5ad is wired into `linear_probe_pretrain.py::_load_dataset`
  yet). `provenance = "norman2019:synthetic"` in each summary
  JSON. The Phase 6 PR body already called this out for the
  mock-checkpoint baseline; it is still the loader state for
  this rerun. See "Caveats" below.

## Results table

| Run        | ckpt_sha256 (first 8) | top-50 DE R² | ΔR² vs random         | W&B URL                                                                    |
|------------|-----------------------|--------------|-----------------------|----------------------------------------------------------------------------|
| Real       | `416e8b1a`            | -0.4617      | -0.1991 (rel -75.79%) | https://wandb.ai/quriegen/aivc-linear-probe/runs/nbc8telz                  |
| Mock       | `c0d9715d`            | -0.4269      | -0.1643 (rel -62.54%) | https://wandb.ai/quriegen/aivc-linear-probe/runs/5ppw0qr2                  |
| Random     | n/a                   | -0.2627      | 0 (baseline)          | https://wandb.ai/quriegen/aivc-linear-probe/runs/bhukayvo                  |

Full per-run summaries: `experiments/phase6_5_runs/{real,mock,random}.json`.

### SHA tripwire

```
Real  : 416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e
Mock  : c0d9715dbc76a6ecab260fe09ca5173ee7fdf6eb640538eac0f9024399a90b4e
Hist. : 10665544bd888dbeddd1c3fdd4b880515e7bb05c16f9a235b453c88b50c5a24b
real  != mock            → True   (expected; tripwire not fired)
real  == historical mock → False  (expected; tripwire not fired)
mock  == historical mock → False  (see mock-SHA divergence note)
```

**Mock-SHA divergence note.** The Phase 6.5 task spec expected the
mock-arm checkpoint to hash to `10665544...` (the Phase 6 PR body
number). The current `checkpoints/pretrain/pretrain_encoders_mock.pt`
hashes to `c0d9715d...` — already documented in
`REAL_DATA_BLOCKERS.md` appendix entry dated 2026-04-16:
`scripts/pretrain_multiome.py`'s default `n_genes` moved from 500 to
2000 between Phase 5 and Phase 6.7b, so the historical mock is no
longer reproducible from current code. What matters for the Phase
6.5 tripwire is `real != mock`, which holds. The historical
`10665544...` hash remains in `REAL_DATA_BLOCKERS.md` as the
Phase 6 PR's checkpoint-of-record; the `c0d9715d...` regenerated
mock is the current mock-arm input and is appended there.

## Phase 6 gate decision

Gate: real-checkpoint ΔR² (vs random) ≥ +5% relative → PASS;
0 < rel ΔR² < +5% → SOFT; rel ΔR² ≤ 0% → FAIL.

- Real vs Random: ΔR² = **-0.1991** (relative **-75.79%**). ≤ 0% → **FAIL**.
- Real vs Mock:   real − mock = **-0.0348** (relative **-8.15%**). Real is
  **worse** than mock, so the real-data pretraining is not contributing
  signal beyond what the architecture prior + synthetic-fallback probe
  target already yields. Flagging loudly per the task's explicit
  instruction, even though the random-vs-real gate has already failed.

### Caveats (applied to the decision, not editorialising it)

The gate decision above is the mechanical arithmetic on the three
numbers in the table. Two loader-level confounders make the
*absolute* R² values apples-to-oranges across rows, and those
confounders should drive the Phase 7 / Phase 6.5-follow-up plan
rather than overriding the FAIL above:

1. **Feature-space mismatch across runs.** `SimpleRNAEncoder.n_genes`
   is baked into the checkpoint and, for `--random_init`, defaults
   to `512`. The synthetic fallback in
   `scripts/linear_probe_pretrain.py::_synth_dataset` uses the
   probe's `n_genes` for both X and Y, so the three runs are fitting
   Ridge on three *different* synthetic datasets:
   - Real:   `n_genes = 36601` (PBMC10k real-data encoder)
   - Mock:   `n_genes = 2000`  (regenerated mock encoder)
   - Random: `n_genes = 512`   (`_build_scratch_encoder` default)
   Smaller `n_genes` ≈ easier self-recon target under fixed latent
   width (`latent_dim=128`), which is consistent with the Random row
   scoring highest in absolute R² even with zero "training".
2. **Synthetic-fallback target, not Norman 2019.** The probe's `Y`
   is a linear function of `X` under Gaussian noise, not a real
   perturbation-response target. Phase 6's PR body already caveats
   this for the mock-data numbers; it continues to apply here.

Both confounders are *pre-existing* in `linear_probe_pretrain.py` /
`exp1_linear_probe_ablation.py`. Per the Phase 6.5 task constraints,
this PR does not modify either script. A separate ablation-infra PR
is required to (a) wire a real Norman 2019 `.h5ad` path, and (b)
either fix the random-init `n_genes` to match the pretrained
encoder or run the three arms against a fixed external probe
target. The FAIL decision above is correct under the current
instrument; re-running against a dataset-aligned probe is the
obvious next step before Phase 7.

## Decision

**FAIL**, dated 2026-04-16.

Under the gate arithmetic the real-data checkpoint's top-50 DE R²
(-0.4617) is below the random-init floor (-0.2627) and below the
regenerated mock checkpoint (-0.4269), so the real-data encoder is
neither clearing the random-init gate nor outperforming the mock
architecture prior. Phase 7 (wiring the pretrained encoder into the
causal head) is blocked on this result. The next step is an
ablation-infra PR that re-runs this table against a dataset-aligned
probe (real Norman 2019 h5ad or a fixed-dimension external target)
before any further Phase 7 work; if that re-run still fails the
gate, the blocker moves upstream into pretraining recipe
(steps / encoder capacity / cell count / peak-set curation).
