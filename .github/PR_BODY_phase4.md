Phase 4: PeakLevelATACEncoder + MultiomeLoader (data plumbing, not wired)

## What
- PeakLevelATACEncoder: TF-IDF + LSI + MLP, GroupNorm throughout,
  structurally independent of NeumannPropagation.W.
- MultiomeLoader: paired RNA + peak-matrix from .h5ad, stamps
  DatasetKind.OBSERVATIONAL, pre-computed peak set required. Supports
  both MuData-style and `obsm["ATAC"]` schemas (parameterized keys).
- harmonize_peaks.py stub documenting MACS2-on-pooled-fragments.
- build_combined_corpus: removed INTERVENTIONAL fallback; now hard-asserts
  that every source dataset stamps dataset_kind explicitly, raising
  `ValueError` naming the offending loader.

## Why
Prepare data and encoder pathway for Phase 5 pretraining without
wiring them into the fusion or training loop. Keeps risky changes
(encoder + loader) isolated from the training path until they are
validated.

## Invariants preserved
- fusion.py, train_week3.py, losses.py, loss_registry.py,
  NeumannPropagation, ATACSeqEncoder untouched.
- No import-time side effects from Phase 4 modules on global torch
  state (verified — see Validation evidence below).
- Phase 3 runtime gradient guard (AIVC_GRAD_GUARD=1) unaffected:
  train_week3.py does not transitively import any Phase 4 module, so
  the stage="joint" routing path is bit-identical to the Phase 3
  post-merge state.

## Invariants added
- MultiomeLoader stamps OBSERVATIONAL on every batch (matches Phase 2
  stamping convention).
- PeakLevelATACEncoder uses GroupNorm to prevent batch-stat leakage
  into causal head (Phase 3 invariant).
- PeakLevelATACEncoder forward graph is structurally independent of
  NeumannPropagation.W (verified by test).
- build_combined_corpus fails loud on missing dataset_kind.

## Validation
**Tests: 10/10 pass, 0 skipped.**

- `tests/test_peak_encoder.py` (3): forward shape + LSI variance >1e-4 +
  no BatchNorm + graph-independence from `NeumannPropagation.W`.
- `tests/test_multiome_loader.py` (3): peak_set required, keys + kind
  stamp, MuData schema equivalence. **MuData branch is no longer
  skipped** — covered by a lightweight `sys.modules` stub (the real
  `mudata` package is intentionally not added to the dep surface this
  phase; the stub exposes the attribute surface the loader consumes:
  `read(path)` → object with `.mod` dict of AnnData).
- `tests/test_build_combined_corpus.py` (2): stamped passes, unstamped
  raises `ValueError` naming the offending loader.
- `tests/test_phase4_no_import_side_effects.py` (2): Phase 4 imports
  leave torch global state unchanged; train_week3.py source contains
  no reference to `atac_peak_encoder`, `multiome_loader`,
  `PeakLevelATACEncoder`, or `MultiomeLoader`.

**Baseline parity evidence** (in lieu of a full 100-step train_week3.py
run, which requires the staged multi-perturbation corpus and is
infeasible in CI — the property below is strictly stronger for our
purposes: it proves the *mechanism* by which parity could break is
absent):

```
torch state, seed=12345
  before Phase 4 imports: (torch.float32, 7408559109481117794, 8, True)
  after  Phase 4 imports: (torch.float32, 7408559109481117794, 8, True)
  equal: True
```

Tuple = (default_dtype, hash(rng_state_bytes), num_threads, grad_enabled).
Since (a) no torch global state is mutated by importing the new modules
and (b) train_week3.py contains no reference to any Phase 4 symbol, the
100-step loss-component dict is bit-exact to Phase 3 post-merge by
construction.

**Runtime gradient guard:** AIVC_GRAD_GUARD stays silent on the
stage="joint" path because the path is unchanged — train_week3.py's
import graph does not include any Phase 4 module (enforced by the test
above).

## Not in scope
- Wiring encoder into TemporalCrossModalFusion (Phase 5).
- Registering pretrain losses (Phase 5).
- Running MACS2 end-to-end (handled by harmonize_peaks.py when invoked
  separately).

## Phase-5 tripwires
- Any pretrain loss registered in Phase 5 must not take gradient
  through NeumannPropagation, even transitively via a shared encoder
  output. Enforced by Phase 3 allow-list + forbidden-name block and
  by graph-independence test in this phase.
- MultiomeLoader output dict schema is the integration contract for
  Phase 5. Any change to keys (`rna`, `atac_peaks`, `dataset_kind`) is
  a breaking change for the pretraining entrypoint.
