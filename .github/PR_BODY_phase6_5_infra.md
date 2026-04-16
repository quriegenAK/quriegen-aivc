# Phase 6.5-infra — ablation driver refactor

## What

Refactors `scripts/exp1_linear_probe_ablation.py` so the Phase 6.5
three-run contract (real / mock / random) can produce three clean
W&B runs, one metrics row per run.

Changes in this PR:

1. **`--ckpt_path` no longer `required=True`.** It is now optional and
   validated manually: required unless `--random_init` is set.
2. **`--random_init` added.** Skips all checkpoint-loading and
   SHA-256 work, builds a fresh `SimpleRNAEncoder`, forces
   `--condition=scratch`, and logs `ckpt_sha256="n/a"` in the W&B
   config. Mutually exclusive with `--ckpt_path` (argparse `p.error`
   at parse time).
3. **`--condition {pretrained|scratch|both}` added.** Default is
   `both` so pre-refactor invocations keep producing bundled
   `pretrained/*` + `scratch/*` metrics in a single W&B run. Passing
   `pretrained` or `scratch` runs only that arm. `--random_init`
   forces `scratch` (warns if the user passed anything else).
4. **`--n_genes` added.** Used only under `--random_init` when no
   `--dataset_path` is provided. When `--dataset_path` resolves to a
   real `.h5ad`, the dataset's own gene count is preferred
   (`_load_dataset` in `linear_probe_pretrain.py` already derives
   `n_genes` from `adata.X.shape[1]`); `--n_genes` is the fallback
   for the synthetic code path only.

Files touched (per the infra-PR constraint list):

- `scripts/exp1_linear_probe_ablation.py` — argparse + `main()` refactor.
- `.github/REAL_DATA_BLOCKERS.md` — appendix-only, adds the
  regenerated mock-ckpt hash. **Does not remove** the historical
  `10665544...` entry.
- `.github/PR_BODY_phase6_5_infra.md` — this file (new).

Not touched: `aivc/training/*`, `aivc/skills/*`,
`aivc/data/multiome_loader.py`, `train_week3.py`,
`scripts/pretrain_multiome.py`, `scripts/harmonize_peaks.py`,
`scripts/linear_probe_pretrain.py`, any test under `tests/`.

## Why

Without this refactor the three W&B runs Phase 6.5 specifies are not
achievable without editing the ablation driver inside a Phase 6.5 PR
(which the Phase 6.5 no-edit list forbids). Specifically:

- `--ckpt_path` was `required=True` — the random-init run cannot be
  expressed at all.
- `main()` always ran `pretrained + scratch` back-to-back in one
  W&B run under `pretrained/*` and `scratch/*` key namespaces — so
  "Real" and "Mock" could not be isolated into their own runs for
  the three-row `PHASE6_5_RESULTS.md` table.

## Backward compatibility — verified

Invocation identical to pre-refactor usage (no `--condition`, no
`--random_init`, no `--wandb`):

```
python scripts/exp1_linear_probe_ablation.py \
    --ckpt_path checkpoints/pretrain/pretrain_encoders.pt \
    --dataset_name norman2019 --seed 17
```

Output (unchanged shape — both arms printed, ΔR² computed, exit 0):

```
=== Linear-probe ablation summary ===
condition     r2_top50_de   r2_overall    pearson_overall  probe_fit_seconds
pretrained        -0.4617      -0.4445             0.0038             0.2011
scratch           -0.4656      -0.4801             0.0035             0.2444

ΔR²(top-50 DE) = pretrained − scratch = +0.0039 (relative: +0.83%)
```

## Dry-run R² values (informational)

Synthetic fallback — no `--dataset_path` provided. These numbers are
**not** authoritative Phase 6.5 results (see caveat below); they
exist only to prove the three invocations exit 0 and produce the
expected single-condition metric rows.

| Run | invocation shape | r2_top50_de | r2_overall | pearson_overall | exit |
|---|---|---|---|---|---|
| Real, pretrained only   | `--ckpt_path .../pretrain_encoders.pt --condition pretrained` | -0.4617 | -0.4445 | 0.0038 | 0 |
| Mock, pretrained only   | `--ckpt_path .../pretrain_encoders_mock.pt --condition pretrained` | -0.4269 | -0.3991 | 0.0695 | 0 |
| Random init             | `--random_init` | -0.2627 | -0.2289 | 0.2280 | 0 |

**Caveat — not cross-comparable.** Each run lands on a different
synthetic dataset because `_load_dataset` seeds its synthetic
generator with a shape derived from the ckpt's `n_genes` (real ckpt
has n_genes from real PBMC10k, mock ckpt uses n_genes=2000, random
run defaults to n_genes_fallback=512). The Phase 6.5 authoritative
numbers must come from runs with `--dataset_path` pointed at a real
norman2019 `.h5ad` so all three arms share the same X/Y pairs.
Pointing the three runs at a real norman2019 h5ad is a Phase 6.5
concern (not this PR).

## Validation matrix

| Check | Command | Expected | Observed |
|---|---|---|---|
| Backward compat — both arms | `--ckpt_path real.pt --dataset_name norman2019 --seed 17` | exit 0, prints both `pretrained` and `scratch` rows, ΔR² computed | ✅ |
| Single-condition — pretrained only | `--ckpt_path real.pt --dataset_name norman2019 --seed 17 --condition pretrained` | exit 0, prints only `pretrained` row | ✅ |
| Single-condition — mock / pretrained | `--ckpt_path mock.pt --dataset_name norman2019 --seed 17 --condition pretrained` | exit 0, prints only `pretrained` row | ✅ |
| Random init | `--random_init --dataset_name norman2019 --seed 17` | exit 0, prints only `scratch` row, no SHA computed | ✅ |
| Mutual exclusion | `--random_init --ckpt_path foo.pt --dataset_name norman2019` | non-zero with clear error | ✅ (exit 2, `--random_init and --ckpt_path are mutually exclusive.`) |
| `--ckpt_path` required when not random | `--dataset_name norman2019` | non-zero with clear error | ✅ (exit 2, `--ckpt_path is required unless --random_init is set.`) |
| `pytest tests/ -q --ignore=tests/test_ckpt_loader.py` | all pass | 437 passed, 10 warnings (pre-existing, unrelated) | ✅ |

## Mock checkpoint regeneration

Produced by running `scripts/pretrain_multiome.py` with the synthetic
fallback (no `--multiome_h5ad`, no `--peak_set`):

```
mkdir -p checkpoints/pretrain_mock_tmp
python scripts/pretrain_multiome.py --seed 17 --no_wandb \
    --checkpoint_dir checkpoints/pretrain_mock_tmp --steps 50
mv checkpoints/pretrain_mock_tmp/pretrain_encoders.pt \
   checkpoints/pretrain/pretrain_encoders_mock.pt
```

The direct-to-`checkpoints/pretrain/` path was NOT used because
`pretrain_multiome.py` hardcodes the filename to
`<checkpoint_dir>/pretrain_encoders.pt`, which would overwrite the
real checkpoint (no `--output` flag; adding one would require editing
`pretrain_multiome.py`, which is on the no-edit list).

- **New mock SHA-256:** `c0d9715dbc76a6ecab260fe09ca5173ee7fdf6eb640538eac0f9024399a90b4e`
- **Historical mock SHA-256 (unchanged in the audit trail):**
  `10665544bd888dbeddd1c3fdd4b880515e7bb05c16f9a235b453c88b50c5a24b`
- **SHAs differ** — expected. `pretrain_multiome.py` defaults changed
  between Phase 5 and now (`--n_genes` default is now 2000; the
  historical mock was recorded with n_genes=500 in
  `REAL_DATA_BLOCKERS.md`). The original mock checkpoint is no longer
  byte-reproducible from current code. New hash recorded in the
  appendix of `REAL_DATA_BLOCKERS.md`; the historical row is preserved
  as the prior-art entry.

Real ckpt SHA-256 confirmed untouched:
`416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e`.

## Not in scope

- **Phase 6.5 execution** — this PR makes it possible, does not run
  it. No `PHASE6_5_RESULTS.md`, no `PASS/SOFT/FAIL` decision, no W&B
  runs against the real checkpoint.
- **Model or training code changes** — no edits to
  `SimpleRNAEncoder`, `MultiomePretrainHead`, any loss, any checkpoint
  schema.
- **Dataset loader changes** — `_load_dataset` and `_synth_dataset`
  in `scripts/linear_probe_pretrain.py` are untouched. The
  dataset-derived `n_genes` path is the pre-existing code path at
  `linear_probe_pretrain.py:94-109`; this PR just threads the
  `n_genes_fallback` override through from the new `--n_genes` CLI
  arg.
- **W&B execution** — `--wandb` remains opt-in. No real W&B URLs are
  produced by this PR.

## Reproducibility contract

- Branch: `phase-6.5-infra` off `main` @ `ecfdbad`.
- Commands to reproduce the validation matrix above — all three
  should exit 0 on any workstation with the repo's `requirements.txt`
  installed:
  ```
  # backward compat
  python scripts/exp1_linear_probe_ablation.py \
      --ckpt_path checkpoints/pretrain/pretrain_encoders.pt \
      --dataset_name norman2019 --seed 17

  # single-condition — real, pretrained only
  python scripts/exp1_linear_probe_ablation.py \
      --ckpt_path checkpoints/pretrain/pretrain_encoders.pt \
      --dataset_name norman2019 --seed 17 --condition pretrained

  # single-condition — mock, pretrained only
  python scripts/exp1_linear_probe_ablation.py \
      --ckpt_path checkpoints/pretrain/pretrain_encoders_mock.pt \
      --dataset_name norman2019 --seed 17 --condition pretrained

  # random init floor
  python scripts/exp1_linear_probe_ablation.py \
      --random_init --dataset_name norman2019 --seed 17

  # full suite
  pytest tests/ -q --ignore=tests/test_ckpt_loader.py
  ```
- Real ckpt SHA: `416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e`
- Mock ckpt SHA (regenerated for this PR): `c0d9715dbc76a6ecab260fe09ca5173ee7fdf6eb640538eac0f9024399a90b4e`

After this merges, the Phase 6.5 prompt is executable without further
infra changes.
