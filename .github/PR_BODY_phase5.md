# Phase 5: Pretraining losses + two-stage trainer entrypoint

## What
- `MultiomePretrainHead` (`aivc/training/pretrain_heads.py`): contrastive
  projection MLPs + peak-to-gene linear regressor. Validates input dict
  against the `MultiomeLoader` contract (`rna`, `atac_peaks`) at
  `forward()` entry and raises `ValueError` naming any missing key before
  any tensor op. Structurally independent of `NeumannPropagation.W`.
- Four pretrain-stage losses registered via
  `aivc/training/pretrain_losses.register_pretrain_terms`:
  - `masked_rna_recon`     (weight 1.0)
  - `masked_atac_recon`    (weight 1.0)
  - `cross_modal_infonce`  (weight 0.5)
  - `peak_to_gene_aux`     (weight 0.1)
- `scripts/pretrain_multiome.py`: standalone pretraining entrypoint.
  Runtime grad guard armed. W&B project `aivc-pretrain` (main project
  untouched). Falls back to mock Multiome when the harmonize_peaks
  artifact is absent.

## Why
Train cross-modal RNA + peak-level ATAC encoders on paired Multiome
observational data under a stage that cannot leak gradient into the
causal head. Produces checkpoints the Phase 6 fine-tuning ablation
will consume.

## Invariants preserved
- `fusion.py`, `train_week3.py`, `NeumannPropagation` untouched. Grep
  shows zero new references (`tests/test_phase5_no_import_side_effects.py`).
- Torch global state byte-identical pre/post importing `pretrain_heads`
  and `pretrain_losses`:
  ```
  before: {'default_dtype': 'torch.float32',
           'rng_hash': 4710906464937165972,
           'num_threads': 8, 'grad_enabled': True}
  after : {'default_dtype': 'torch.float32',
           'rng_hash': 4710906464937165972,
           'num_threads': 8, 'grad_enabled': True}
  identical: True
  ```
- `train_week3.py` source contains zero Phase 5 symbols
  (`MultiomePretrainHead`, `pretrain_multiome`, `masked_rna_recon`,
  `masked_atac_recon`, `cross_modal_infonce`, `peak_to_gene_aux`,
  `register_pretrain_terms`). Asserted by
  `test_train_week3_does_not_reference_phase5_symbols`.
- Phase 3 gradient-isolation guard still silent on interventional
  training (`tests/test_gradient_isolation.py` ✓).
- Phase 3 loss-registry parity still holds
  (`tests/test_loss_registry_parity.py` ✓).

## Invariants added
- `MultiomePretrainHead` validates the `MultiomeLoader` batch contract
  at `forward()` entry — raises `ValueError` naming missing keys.
- All four new pretrain terms pass the Phase 3 allow-list
  (`_is_term_active("pretrain", "pretrain") == True`) and none collide
  with `_PRETRAIN_FORBIDDEN_NAMES`.
- Registration-time substring guard (`_guard_pretrain_name`): any
  pretrain-stage name containing `causal`, `ordering`, `intervention`,
  `leak`, or `lfc` is rejected at registration rather than silently
  skipped at compute(). Synthetic-leak test registers
  `causal_ordering_leak` under stage `pretrain` and confirms the guard
  raises.
- `scripts/pretrain_multiome.py` asserts `NeumannPropagation` is ABSENT
  from the pretrain graph on setup (so "W.grad is None" after backward
  cannot pass for the wrong reason).

## Integration run
50 steps on 500-cell mock Multiome (peak_set artifact absent → mock
fallback):

```
[step 000] total=3.3463 ma=3.3463
[step 015] total=1.8457 ma=2.6317
[step 030] total=1.1816 ma=1.6735
[step 049] total=0.9725 ma=1.1132
[ckpt] saved -> checkpoints/pretrain/pretrain_encoders.pt
```

- Moving-average loss decreases monotonically.
- Runtime grad guard never fires (Neumann not in graph; assertion
  traversal runs every backward for defense-in-depth).
- Checkpoint saved.
- W&B project `aivc-pretrain` used; main project dashboards untouched.

## Phase 6 interface resolution

The task offered two paths for the encoder divergence question:

- (a) Promote `SimpleRNAEncoder` to `aivc/skills/rna_encoder.py` AND
      refactor `PerturbationPredictor` to compose it as
      `self.rna_encoder`.
- (b) Keep the encoder local and build a state_dict adapter.

**Resolution: (a) in its promotion aspect, without the
`PerturbationPredictor` refactor.**

Why: `PerturbationPredictor` operates on per-gene scalars through a
GAT on a gene-graph (`FeatureExpander: 1→feature_dim` + `GATConv`).
It has no cell×gene MLP slot. A literal Task 1(a) refactor would
either inject an unused submodule or rewrite `PerturbationPredictor`
into a different architecture; both regress existing tests and match
the task's own fallback clause. Task 1(b) fails symmetrically — a
surjective mapping onto "PerturbationPredictor's RNA encoder params"
is not definable because that encoder does not exist as an MLP.

Adopted path:

- `aivc/skills/rna_encoder.py::SimpleRNAEncoder` is now a named,
  publicly importable class (no underscore). Its state_dict schema is
  stable and documented.
- `scripts/pretrain_multiome.py` imports `SimpleRNAEncoder` from the
  promoted module; no local definition remains.
- Phase 6's fine-tuning head must instantiate `SimpleRNAEncoder`
  directly (cell-level branch), not graft it into
  `PerturbationPredictor`. This is spelled out in the schema doc.

Tests still green after promotion: 10/10 Phase 5, 14/14 upstream
invariants, 50-step smoke monotone.

## Defense-in-depth invariants

This PR enforces forbidden-term rejection at two layers:

1. **Registration-time (fail-fast)** in
   `aivc/training/pretrain_losses.py::_guard_pretrain_name`. Raises
   `ValueError` on any pretrain-stage term whose name contains
   `causal`, `ordering`, `intervention`, `leak`, or `lfc`, or collides
   with `loss_registry._PRETRAIN_FORBIDDEN_NAMES`.
2. **In-registry (runtime backstop)** in Phase 3's
   `loss_registry.LossRegistry.compute`: exact-match skip of any
   forbidden name under stage="pretrain".

**Both layers must remain in place.** Removing either "because the
other one is sufficient" is explicitly forbidden by this PR contract.
Stated in the `pretrain_losses.py` module docstring.

## Deferred to Phase 6

Real Multiome pretraining has not yet been run. First real-data signal
on whether pretraining helps Norman/Kang fine-tuning will arrive in
Phase 6. **Phase 6 merge gate must include a real-data pretraining
run, not mock.** Tracked as
[PHASE6_BLOCKERS.md](PHASE6_BLOCKERS.md) item "Run harmonize_peaks on
10x PBMC Multiome and produce peak_set artifact".

## Pretrained checkpoint schema

Frozen as a Phase 5 → Phase 6 contract at
[aivc/training/PRETRAIN_CKPT_SCHEMA.md](../aivc/training/PRETRAIN_CKPT_SCHEMA.md).

Top-level keys (`schema_version=1`):

| Key | Source |
| --- | ------ |
| `schema_version` | `PRETRAIN_CKPT_SCHEMA_VERSION` (int) |
| `rna_encoder` | `SimpleRNAEncoder.state_dict()` |
| `atac_encoder` | `PeakLevelATACEncoder.state_dict()` |
| `pretrain_head` | `MultiomePretrainHead.state_dict()` |
| `rna_encoder_class` / `atac_encoder_class` / `pretrain_head_class` | fully-qualified class strings |
| `config` | argparse namespace vars |

`tests/test_pretrain_ckpt_schema.py` runs the pretrain script for one
step and asserts every listed key is present with the expected type
and submodule state_dict keys.

## Phase-6 tripwires
- Phase 6 is expected to wire `--pretrained_ckpt` into `train_week3.py`.
  The Phase 5 dead-code invariant (zero Phase 5 symbols in
  `train_week3.py`) will change at that point. Phase 6 must snapshot
  and validate the new expected import graph.
- Checkpoint schema mismatch on load must raise, not silently skip.
- Real Multiome pretraining run is a Phase 6 merge gate.
- Both forbidden-term enforcement layers (registration-time +
  in-registry) must remain in place.

## Not in scope
- Wiring pretrained weights into the main trainer (Phase 6).
- Modifying `TemporalCrossModalFusion` or the causal mask (deferred).
- Real peak-set harmonization — still a separate deliverable; until
  the `harmonize_peaks.py` artifact exists, pretraining runs on mock
  Multiome.
