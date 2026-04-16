# Phase 6.7b: Real-data pretraining on harmonized PBMC Multiome

## What

- `scripts/build_pbmc10k_h5ad.py` (new): builds paired-modality
  AnnData from 10x Cell Ranger ARC `filtered_feature_bc_matrix.h5`,
  aligns its ATAC peak counts to the harmonized peak set from Phase
  6.7 via pure-Python per-chromosome sorted-array `searchsorted`
  interval intersection, and writes the output with the
  `obsm_atac` schema that `aivc.data.multiome_loader.MultiomeLoader`
  expects. Sibling `.meta.json` records every input hash.
- `scripts/pretrain_multiome.py` (modified):
  - `--peak_set_path` added as canonical flag name (old `--peak_set`
    kept as alias).
  - MPS device selection (CUDA → MPS → CPU).
  - **Runtime config serialization**: `ckpt['config']` now carries
    actual runtime values for `n_genes`, `n_peaks`, `hidden_dim`,
    `latent_dim`, `device`, plus `peak_set_sha256`, instead of the
    stale argparse defaults. Closes the `vars(args)` bug — see
    "Out-of-scope fix" below.
  - W&B audit trail: `peak_set_sha256`, `peak_set_path`,
    `n_genes_runtime`, `n_peaks_runtime`, `device`, `seed`,
    `git_rev` logged at init; `ckpt_sha256` logged post-save. This
    is the provenance surface Phase 6.5's linear-probe tripwire
    will read.
- `checkpoints/pretrain/pretrain_encoders.pt` regenerated on real
  10x PBMC Multiome data. (Gitignored; hash committed as canonical
  reference in `.github/REAL_DATA_BLOCKERS.md` appendix.)

## Why

Phase 6.7b is the first real-data signal in the pretraining stack.
Phase 5 and Phase 6 both shipped on the synthetic Multiome fallback
in `pretrain_multiome.py`; every downstream ablation that claimed
a biological result therefore had a methodology-only footnote. This
PR closes that footnote by replacing the mock checkpoint with one
produced on actual 10x PBMC Multiome data through the same
`scripts/pretrain_multiome.py` entrypoint, conditioned on the
harmonized peak set artifact from Phase 6.7.

## Reproducibility contract

- **Input AnnData**: `data/pbmc10k_multiome.h5ad` (gitignored).
  - `n_cells = 11898`, `n_genes = 36601`, `n_peaks = 115720`.
  - Schema: `obsm_atac` (`.X` = RNA counts sparse, `.obsm["atac"]`
    = aligned peak counts sparse).
  - Build command: `python scripts/build_pbmc10k_h5ad.py` (defaults
    resolve to the canonical paths listed in the script).
  - Provenance: `data/pbmc10k_multiome.h5ad.meta.json` records
    `input_h5_sha256`, `input_fragments_sha256`, `peak_set_sha256`,
    `n_peaks_aligned`, `n_peaks_10x`, reindex method, and timestamp.
- **Peak set SHA-256**:
  `57b66a257e85287cd6829e38b35bd82aea795d4e96c2b074566981b298a9ef0c`
- **New (real) ckpt SHA-256**:
  **`416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e`**
- **Mock ckpt SHA-256 (for diff)**:
  `10665544bd888dbeddd1c3fdd4b880515e7bb05c16f9a235b453c88b50c5a24b`
- **Exact pretrain command**:
  ```
  AIVC_GRAD_GUARD=1 \
  python scripts/pretrain_multiome.py \
      --multiome_h5ad data/pbmc10k_multiome.h5ad \
      --peak_set_path data/peak_sets/pbmc10k_hg38_20260415.tsv \
      --steps 5000 \
      --seed 17 \
      --no_wandb
  ```
- **git rev-parse HEAD at run time**: `0a67e817d41af51`
  (`phase-6.7-exec` tip at the moment the run was launched; the
  Phase 6.7b implementation diff was present in the working tree
  but not yet committed, which is the normal local-development
  workflow before the commit that opens the PR).
- **W&B run URL**: not emitted — `wandb` is not installed in this
  local environment (`aivc_env`), so `--no_wandb` was passed. The
  equivalent audit-trail fields the W&B run would have carried
  (`peak_set_sha256`, `ckpt_sha256`, `n_genes_runtime`,
  `n_peaks_runtime`, `device`, `seed`, `git_rev`) are instead
  stamped into `ckpt['config']` and recoverable via
  `aivc.training.ckpt_loader.peek_pretrain_ckpt_config(...)`. The
  `_setup_wandb` code path is now instrumented to emit all of
  these to a real run as soon as `wandb` is available in the env
  that retrains — see
  [scripts/pretrain_multiome.py](../scripts/pretrain_multiome.py)
  around lines 201–220 (init) and 343–354 (post-save).

## Out-of-scope fix: `vars(args)` config bug

Prior `pretrain_multiome.py` wrote `torch.save({..., "config":
vars(args)})`, so the saved config dict captured argparse defaults
(`n_genes=2000`, `n_peaks=5000`) rather than the actual runtime
shapes. `aivc.training.ckpt_loader.load_pretrained_simple_rna_encoder`
then instantiates `SimpleRNAEncoder(n_genes=<from config>)` and
runs `load_state_dict(strict=True)`, which raises on size mismatch.
That is exactly the tripwire the Phase 6.7b task warned about, and
it made every real-data checkpoint produced through this entrypoint
strictly unloadable downstream. It ships in this PR because
isolating it into its own PR would produce a separate throwaway
that never emits a usable checkpoint — the fix is only observable
when the entrypoint is invoked on a real-shape input, which is
exactly what this PR does.

## Validation gates passed

- **New SHA-256 ≠ mock SHA-256** (both printed for audit):
  ```
  python -c "import hashlib; print(hashlib.sha256(open('checkpoints/pretrain/pretrain_encoders.pt','rb').read()).hexdigest())"
  # → 416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e
  # mock (for diff): 10665544bd888dbeddd1c3fdd4b880515e7bb05c16f9a235b453c88b50c5a24b
  ```
  The two strings are literally different; `assert new_sha !=
  mock_sha` holds. The real-data path was taken — no mock fallback.
- **`ckpt_loader` strict-load succeeds**:
  `from aivc.training.ckpt_loader import
  load_pretrained_simple_rna_encoder` → returns a `SimpleRNAEncoder`
  with `n_genes=36601`. `peek_pretrain_ckpt_config(...)` returns
  `schema_version=1`, `n_genes=36601`, `n_peaks=115720`,
  `hidden_dim=256`, `latent_dim=128`,
  `peak_set_sha256=57b66a25...a9ef0c`.
- **`AIVC_GRAD_GUARD=1` honored throughout**: environment variable
  set at run launch; the per-step `composite.modules()` traversal
  that asserts `NeumannPropagation.W.grad is None` never fired
  across all 5,000 steps. Training exited with status 0.
- **peak_set_sha256 stamped into ckpt['config'] matches on-disk**:
  `ckpt['config']['peak_set_sha256']` ==
  `sha256(data/peak_sets/pbmc10k_hg38_20260415.tsv)` ==
  `57b66a257e85287cd6829e38b35bd82aea795d4e96c2b074566981b298a9ef0c`.
- **73/73 phase-relevant tests pass**:
  ```
  pytest tests/test_multiome_loader.py tests/test_peak_encoder.py \
         tests/test_ckpt_loader.py tests/test_pretrain_ckpt_schema.py \
         tests/test_pretrain_loss_registration.py tests/test_pretrain_head.py \
         tests/test_loss_registry_parity.py tests/test_gradient_isolation.py \
         tests/test_no_bare_torch_load.py tests/test_phase4_no_import_side_effects.py \
         tests/test_phase5_no_import_side_effects.py tests/test_phase6_no_import_side_effects.py \
         tests/test_registry.py tests/test_causal_mask.py tests/test_neumann_sparsity.py
  # → 73 passed
  ```

## Known gaps / deferred

- Loss MA reduction is ~10× over 5,000 steps (8.56 → ~0.78–0.90
  in the converged band, stepwise-noisy but trajectory-monotone
  with occasional `peak_to_gene_aux` outlier-batch spikes).
  Whether this is "sufficient" pretraining is a decision for the
  Phase 6.5 linear-probe ΔR² gate (≥ +5% relative R² on top-50
  DE), not for additional pretraining steps inside this PR. If
  Phase 6.5 reports below-gate ΔR², the right response is to
  extend steps or adjust LR in a follow-up PR, not to retrain
  here speculatively.
- Issue #7 (test pollution in `tests/test_ckpt_loader.py`) is
  pre-existing and untouched by this PR.
- Issue #11 ([infra] environment-only test failures: pin
  `anthropic`/`typer` versions) captures the 1 collection error +
  7 broad-suite failures on `main` and PR #10 alike, none of
  which touch the phase-relevant surface. See "Validation gates
  passed" — the 73/73 phase-relevant set is fully green.

## Not in scope

- Phase 6.5 linear-probe rerun (tracked in
  `.github/REAL_DATA_BLOCKERS.md` "Real-data linear-probe rerun"
  section; a tripwire line pinning the required `ckpt_sha256`
  was added in this PR's closeout commit).
- Any change to `train_week3.py`, `fusion.py`, `losses.py`,
  `loss_registry.py`, `NeumannPropagation`, `PerturbationPredictor`,
  or any encoder implementation.
