# Phase 6: Linear-probe ablation on pretrained SimpleRNAEncoder

## What
- Strict checkpoint loader (`aivc/training/ckpt_loader.py`) enforcing
  `PRETRAIN_CKPT_SCHEMA.md`: `schema_version` must match exactly, every
  documented top-level key must be present, and the `rna_encoder`
  state_dict must have exactly the expected keys — no silent skips,
  no `strict=False` escape hatch. Exposes two entrypoints:
  `load_pretrained_simple_rna_encoder` and `peek_pretrain_ckpt_config`.
- `scripts/linear_probe_pretrain.py`: extracts encoder latents on a
  held-out split and fits a sklearn Ridge probe; reports R² on top-50
  DE genes + overall Pearson.
- `scripts/exp1_linear_probe_ablation.py`: thin driver that runs both
  conditions under a fixed seed, emits a summary table to stdout, and
  logs metrics + pretrained-checkpoint SHA-256 to W&B project
  `aivc-linear-probe`.
- CI enforcement: `tests/test_no_bare_torch_load.py` forbids bare
  `torch.load(` in `scripts/` and `aivc/training/` outside
  `ckpt_loader.py`.

## Why
Establish whether pretraining produces useful representations at all
before taking on the structural refactor required to compose
`SimpleRNAEncoder` into `PerturbationPredictor`'s causal path. Fast,
narrow, correctness-first signal.

## Scope narrowing (documented)
- Phase 6 does NOT run the full causal fine-tune ablation.
  `PerturbationPredictor` is a GAT over per-gene scalars with no slot
  for the pretrained encoder. That ablation is deferred to Phase 6.5
  pending either an encoder refactor or a parallel fine-tune
  entrypoint.

## Invariants preserved
- `train_week3.py`, `fusion.py`, `losses.py`, `loss_registry.py`,
  `NeumannPropagation`, `PerturbationPredictor` untouched.
- Torch global state byte-identical pre/post Phase 6 imports
  (`tests/test_phase6_no_import_side_effects.py`).
- `train_week3.py` import graph contains zero Phase 6 symbols.
- No bare `torch.load` outside `aivc/training/ckpt_loader.py` in
  policed directories (`tests/test_no_bare_torch_load.py`).

## Results

> ⚠️ **METHODOLOGY VALIDATION ONLY — pretraining checkpoint was
> produced on mock Multiome. These numbers prove the probe pipeline
> runs end-to-end. They do NOT measure whether real pretraining
> helps. Do not cite this table as biological evidence.**

Run: `scripts/exp1_linear_probe_ablation.py --seed 17`, synthetic
dataset fallback (pretrained and scratch conditions share the same
synthetic feature space, derived from the checkpoint's `n_genes=500`),
checkpoint `checkpoints/pretrain/pretrain_encoders.pt`,
SHA-256 `10665544bd888dbeddd1c3fdd4b880515e7bb05c16f9a235b453c88b50c5a24b`.

### Norman 2019 (held-out, synthetic fallback)

| condition  | R² top-50 DE | R² overall | Pearson overall | probe fit (s) |
|------------|-------------:|-----------:|----------------:|--------------:|
| pretrained |      -0.2001 |    -0.1711 |          0.2673 |        0.0186 |
| scratch    |      -0.2547 |    -0.2162 |          0.2417 |        0.0091 |
| **Δ**      |  **+0.0547** | **+0.0451**|    **+0.0256**  |             — |

### Kang 2018 (held-out, synthetic fallback)

| condition  | R² top-50 DE | R² overall | Pearson overall | probe fit (s) |
|------------|-------------:|-----------:|----------------:|--------------:|
| pretrained |      -0.2001 |    -0.1711 |          0.2673 |        0.0047 |
| scratch    |      -0.2547 |    -0.2162 |          0.2417 |        0.0054 |
| **Δ**      |  **+0.0547** | **+0.0451**|    **+0.0256**  |             — |

Negative absolute R² in both conditions reflects the synthetic-fallback
probe target (self-recon on i.i.d. Gaussian-derived features), not the
pretraining objective. The matched ΔR² across datasets is expected
under the synthetic fallback — the loader returns the same synthetic
signal for both dataset names until real h5ad artifacts are staged.
The positive pretrained-over-scratch delta confirms the plumbing is
correct: pretrained weights carry *some* information past random init
on matched-capacity encoders.

## Interpretation gate (INACTIVE for this PR)

**The interpretation gate is INACTIVE for this PR because pretraining
was on mock data.** The gate evaluates only against a real-data run.
Phase 6.5 readiness is gated on the real-data rerun listed under
"Real-data rerun plan", not on the numbers in this PR.

For reference, the gate definition (to be evaluated on the real-data
rerun, not here):
- If pretrained > scratch by ≥ 5% relative R² on real Norman 2019 top-50
  DE: proceed to Phase 6.5 (structural work to wire into causal head).
- If within noise or worse: revisit pretraining objective before
  investing in structural refactor.

## Real-data rerun plan

Phase 6.5 will not start until all items in
[`.github/REAL_DATA_BLOCKERS.md`](REAL_DATA_BLOCKERS.md) are closed
AND a real-data rerun of `scripts/exp1_linear_probe_ablation.py` has
been logged to the W&B `aivc-linear-probe` project. **This rerun's
results — not the mock-data results in this PR — gate Phase 6.5.**

`REAL_DATA_BLOCKERS.md` (renamed from `PHASE6_BLOCKERS.md` because the
blocker is no longer Phase-6-specific) records:
- owner and target date for the harmonize_peaks + real-data pretrain
  deliverable,
- the three-part definition of done,
- the current mock-data checkpoint SHA-256 (above), so the real-data
  rerun can prove "different checkpoint, different result" rather than
  "same checkpoint, magic improvement."

## Phase 6.5 tripwires
- `ckpt_loader.py` is the only legal load path. `tests/test_no_bare_torch_load.py`
  enforces this in CI.
- Any future consumer MUST use `load_pretrained_simple_rna_encoder`
  (or `peek_pretrain_ckpt_config` for metadata-only), not a bare
  `torch.load`. New artifact types should add a named function to
  `ckpt_loader` rather than bypassing it.
- Phase 6.5 numbers must come from a checkpoint whose SHA-256 is
  **different** from the mock-data checkpoint recorded in
  `REAL_DATA_BLOCKERS.md`. Matching hash = mock data reused → reject.
- Encoder refactor in `PerturbationPredictor` is a Phase 6.5 scope
  question. Do not attempt it in Phase 6.

## Tests (all passing)
- `tests/test_ckpt_loader.py` (5): valid load, schema_version
  mismatch, missing state_dict key, unexpected state_dict key,
  missing top-level key.
- `tests/test_phase6_no_import_side_effects.py` (2): torch state
  parity + `train_week3.py` symbol guard.
- `tests/test_no_bare_torch_load.py` (1): CI enforcement of the
  integration contract.
- Phase 4 + Phase 5 side-effect tests continue to pass.
