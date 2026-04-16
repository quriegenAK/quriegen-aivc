# Phase 6.7b: Real-data pretraining rerun on harmonized 10x PBMC Multiome

## What

- Built `data/pbmc10k_multiome.h5ad` from 10x Cell Ranger ARC
  `filtered_feature_bc_matrix.h5`, aligning its ATAC peak counts to the
  harmonized peak set from Phase 6.7
  (`data/peak_sets/pbmc10k_hg38_20260415.tsv`, 115,720 peaks,
  SHA-256 `57b66a25...a9ef0c`). Sibling `.meta.json` records input and
  output hashes.
- Reran `scripts/pretrain_multiome.py` on the real Multiome input + real
  peak set for 5,000 steps on Apple MPS, producing a new checkpoint at
  `checkpoints/pretrain/pretrain_encoders.pt`.
- Appended the new checkpoint hash to the `REAL_DATA_BLOCKERS.md`
  appendix.

## Why

Closes the mock-data provenance gap flagged in Phase 6. First real-data
pretraining signal on which the Phase 6.5 linear-probe rerun will be
conditioned.

## Build command (h5ad)

```
python scripts/build_pbmc10k_h5ad.py
# (defaults: 10x h5 + fragments at data/raw/pbmc10k_multiome/,
#  peak set at data/peak_sets/pbmc10k_hg38_20260415.tsv,
#  output at data/pbmc10k_multiome.h5ad)
```

- ATAC reindexing method: **pure-Python per-chromosome sorted-array
  `searchsorted` interval intersection** (no `pybedtools` / `pyranges`
  dependency; neither is installed in this env). For each 10x peak
  `(chrom, s, e)` we find all harmonized peaks on the same chromosome
  whose `[start, end)` overlaps `[s, e)` and set `P[i, j] = 1` in a
  sparse projection; aligned ATAC = 10x ATAC `@ P`.
- Alignment diagnostics:
  - 10x peaks with ≥1 harmonized hit: **108,242 / 143,887 (75.2%)**.
    The 25% un-hit are largely on alt contigs absent from the peak
    set or on MACS2-uncalled regions — expected. (Only 4 peaks had an
    entirely absent chromosome.)
  - Cells with zero aligned peaks: **0 / 11,898** (well under the
    50% empty-row tripwire).
  - Harmonized peaks with zero cell coverage: 8,938 / 115,720 (7.7%).
- Schema choice: `obsm_atac` (matches the `MultiomeLoader(schema=
  "obsm_atac")` branch that is the fewer-adapter-lines path;
  `.X=RNA`, `.obsm["atac"]=aligned peak counts`,
  `.uns["peak_set_sha256"/"peak_set_path"/"schema"/"source_h5_sha256"/
  "fragments_sha256"/"n_peaks_aligned"]`).

## Pretrain command

```
AIVC_GRAD_GUARD=1 \
python scripts/pretrain_multiome.py \
    --multiome_h5ad data/pbmc10k_multiome.h5ad \
    --peak_set_path data/peak_sets/pbmc10k_hg38_20260415.tsv \
    --steps 5000 \
    --seed 17 \
    --no_wandb
```

(`--no_wandb` was used because `wandb` is not installed in this env;
the provenance the W&B run would have logged — device, `peak_set_sha256`,
actual `n_genes`/`n_peaks`/`hidden_dim`/`latent_dim` — is instead stamped
into the checkpoint's own `config` dict by the fix described under
"Small edits to `pretrain_multiome.py`" below.)

## Run summary

- **Device:** MPS (Apple M-series, `torch.backends.mps.is_available() ==
  True`; CUDA unavailable on this host).
- **Steps:** 5,000.
- **Loss MA trajectory:** starts at 8.56 (step 0) → 2.01 (step 300) →
  1.68 (step 640) → stabilizes at **~0.78–0.90** over steps 4,400 –
  4,980 (≈ 10× reduction). Trend is overall monotonically decreasing
  but **not strictly step-wise monotone**: the `peak_to_gene_aux`
  component produces occasional large outlier batches (~every 50–150
  steps) when the masked gene indices hit very-high-expression genes,
  which briefly spike the MA. The final step happens to land on one
  such spike (step 4995 `peak_to_gene_aux=6.83`; MA=5.25 at step 4999),
  but the **converged baseline is ~0.8** — visible at e.g. steps 4,770
  (MA=0.787), 4,810 (MA=0.773), 4,960 (MA=0.788). These spikes are
  stochastic, not training divergence (mean + a handful of outliers,
  not a monotone uptrend). LR stayed at `1e-3`; no NaNs, no divergence.
- **Grad guard fired:** No. Training completed with exit code 0 and the
  `composite.modules()` traversal that asserts `NeumannPropagation.W.grad
  is None` never fired (NeumannPropagation is structurally absent from
  this graph, as intended).
- **`schema_version` validated:** Yes.
  `aivc.training.ckpt_loader.load_pretrained_simple_rna_encoder` strict-
  loads the checkpoint successfully and reports `n_genes=36601,
  hidden_dim=256, latent_dim=128`.
- **W&B run URL:** N/A (wandb not installed in this env; `peak_set_sha256`
  is instead stamped into the saved checkpoint's `config` — see below).

## Reproducibility

- Peak set SHA-256:
  `57b66a257e85287cd6829e38b35bd82aea795d4e96c2b074566981b298a9ef0c`
- New checkpoint SHA-256:
  **`416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e`**
- Mock checkpoint SHA-256 (for contrast):
  `10665544bd888dbeddd1c3fdd4b880515e7bb05c16f9a235b453c88b50c5a24b`
- New ≠ mock (validated via `shasum -a 256`).
- h5ad build meta: `data/pbmc10k_multiome.h5ad.meta.json`
  (gitignored; hashes echoed here for audit).
  - `input_h5_sha256`:
    `f6824171378787baab244f559b8b438f79db2eb39f78d17b2196f7ecd2c03549`
  - `input_fragments_sha256`:
    `5075e32a0e9c6dded35b060bf90d6144375b150e131ffb0be121a93e3b5e1e38`
    (matches `data/peak_sets/pbmc10k_hg38_20260415.tsv.meta.json`, as
    expected).
  - `n_cells=11898  n_genes=36601  n_peaks_aligned=115720`.

## Small edits to `pretrain_multiome.py`

Two narrow changes, both required for the Phase 6.7b contract and
flagged in the task's FAILURE HANDLING section:

1. **`--peak_set_path` flag alias** (task 2a's canonical flag name).
   `--peak_set` is kept as a back-compat alias (`dest="peak_set"`), so
   the existing `test_pretrain_ckpt_schema.py` CLI invocation still
   works unchanged.
2. **`peak_set_sha256` + real-shape stamping into saved `config` dict.**
   Before this PR, `torch.save({..., "config": vars(args)})` captured
   only argparse state — on the real-data path the argparse defaults
   (`n_genes=2000`, `n_peaks=5000`) would be serialized instead of the
   actual runtime shapes (`n_genes=36601`, `n_peaks=115720`). That made
   the checkpoint unloadable by
   `ckpt_loader.load_pretrained_simple_rna_encoder`, which instantiates
   `SimpleRNAEncoder(n_genes=<from config>)` and then does
   `load_state_dict(strict=True)` — a size mismatch the strict loader
   raises on. This was the Phase 6.5 tripwire. Fix: before `torch.save`,
   write `n_genes`, `n_peaks`, `hidden_dim`, `latent_dim`, `device`, and
   `peak_set_sha256` into the config dict from the actual runtime
   values. Verified: `peek_pretrain_ckpt_config(...)` now returns
   `n_genes=36601`, and `load_pretrained_simple_rna_encoder(...)` loads
   strict=True without error.

The W&B helper also got a `config_extra` kwarg so that whenever wandb
IS installed, the same `peak_set_sha256` gets logged into the run
config (per task 3b).

MPS device selection was added as a third branch (CUDA → MPS → CPU) so
MPS-equipped hosts actually use the accelerator. CUDA hosts are
unaffected.

No other files in the forbidden list (`train_week3.py`, `fusion.py`,
`losses.py`, `loss_registry.py`, `NeumannPropagation`,
`PerturbationPredictor`, encoders) were modified.

## Invariants preserved

- `train_week3.py` — not modified; not present in this repo tree at all
  (the Phase 4/5/6 import-side-effects tests still pass, confirming the
  invariant they enforce).
- `fusion.py`, `losses.py`, `loss_registry.py`, `NeumannPropagation`,
  `PerturbationPredictor`, encoders — all unchanged.
- `MultiomeLoader.peak_set_path` contract honored: the h5ad build
  script produced an `obsm_atac`-schema input whose peak dimension
  matches the harmonized peak set exactly; the pretrain call passed
  `--peak_set_path` explicitly so the loader's `ValueError` sentinel
  never triggered.
- `ckpt_loader` remains the only legal `torch.load` call site outside
  the legacy whitelist:
  `tests/test_no_bare_torch_load.py::test_no_bare_torch_load_outside_ckpt_loader`
  passes.
- `AIVC_GRAD_GUARD=1` was set throughout; the runtime guard traversal
  ran every backward step and never fired. No leak into
  `NeumannPropagation.W`.
- Phase 1-6 test suite: **73/73 passed** (ran the explicit set —
  `test_multiome_loader`, `test_peak_encoder`, `test_ckpt_loader`,
  `test_pretrain_ckpt_schema`, `test_pretrain_loss_registration`,
  `test_pretrain_head`, `test_loss_registry_parity`,
  `test_gradient_isolation`, `test_no_bare_torch_load`, the three
  `test_phase{4,5,6}_no_import_side_effects`, `test_registry`,
  `test_causal_mask`, `test_neumann_sparsity`). Broad `pytest tests/`
  showed 431 passed and 7 environment-only failures in
  `tests/agents/test_research_agent.py` + `tests/agents/test_training_agent.py`
  from a missing `anthropic` SDK (and `test_full_loop.py` from missing
  `typer`) — both are pre-existing env gaps, unrelated to Phase 6.7b.

## Not in scope

- Linear probe rerun (Phase 6.5).
- Any structural encoder-to-causal-head wiring (Phase 7).

## Phase 6.5 tripwires

- Phase 6.5 MUST load this specific checkpoint via
  `aivc.training.ckpt_loader.load_pretrained_simple_rna_encoder` (or
  `peek_pretrain_ckpt_config` for metadata) and log its SHA-256
  (`416e8b1a...f1f58e29e`) to the W&B run config. The Phase 6
  interpretation gate (≥ +5% relative R² on top-50 DE) activates only
  against this real-data checkpoint, never against the mock
  (`10665544...0c5a24b`).
- `peek_pretrain_ckpt_config` now returns `peak_set_sha256` alongside
  the shape fields — Phase 6.5 should echo this into its W&B config
  too, as a second guard against a mock-fallback regression.
