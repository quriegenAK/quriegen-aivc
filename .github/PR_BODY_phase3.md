# Phase 3: Gradient isolation guard for NeumannPropagation.W

## What
- `Stage` literal extended with `"joint_safe"` (explicit allow-list tag for
  terms safe to run under pretrain).
- Pretrain stage compute path: allow-list (`pretrain`/`joint_safe` only) +
  forbidden-name hard-block for `causal_ordering`, `causal`, `mse`, `lfc`
  (defense in depth — catches a mistagged `joint_safe` term whose name
  reveals it pulls interventional signal).
- Empty-pretrain placeholder now a freshly-allocated leaf
  `torch.zeros((), requires_grad=True)` — structurally independent of every
  model parameter, including `NeumannPropagation.W`. Preserves `.backward()`
  as a valid no-op.
- Runtime guard in the training loop, default-on via `AIVC_GRAD_GUARD=1`,
  gated EXPLICITLY on `stage == "pretrain"` (not on `W.grad` state) so it
  cannot misfire at `stage="joint"` with pretrained weights loaded.

## Why
Before Phase 4 introduces observational Multiome data, prove that
`stage="pretrain"` cannot leak gradient into causal parameters. This is the
single correctness invariant the Multiome plan depends on: observational
batches must never rewrite the causal matrix `NeumannPropagation.W`.

## Validation
- **9/9 tests pass** (3 isolation + 6 parity).
- Isolation test (`stage="pretrain"`): `W.grad is None` or
  `torch.allclose(W.grad, 0)`.
- **Counterfactual** (`stage="joint"`): `W.grad is not None` AND
  `torch.any(W.grad != 0)`. Proves the forward path actually exercises W,
  so the pretrain None/zero result is meaningful gradient-flow evidence
  rather than a structural artifact (e.g. W never participating in the
  graph at all).
- Forbidden-name block survives mistagging: a `causal_ordering` term
  registered as `joint_safe` is still skipped at pretrain (the inner term
  function raises if ever invoked — it is never invoked).
- Baseline parity on interventional training preserved: all 6
  `test_loss_registry_parity` cases pass to atol=1e-7.
- Runtime guard silent on current training (interventional-only; stage is
  always `"joint"`, so the guard body is never entered).

## Invariants documented for future phases
- **Phase 4**: observational loader must not flow through
  `NeumannPropagation` in the forward pass. If a future pretrain-only
  forward skips the Neumann step, the isolation test's None branch is
  the expected outcome; the allclose-0 branch exists only as a safety net.
- **Phase 5**: any new pretrain-stage loss term must not take gradient
  through `NeumannPropagation`, even transitively via shared encoder
  outputs (e.g. a shared BatchNorm whose running stats are touched by W's
  descendants). If adding a pretrain term makes the counterfactual-vs-
  isolation gap collapse, swap shared BatchNorm for GroupNorm in the
  encoders before merging.
- **Phase 6**: guard must remain silent when `stage="joint"` with
  pretrained weights loaded. The explicit `stage == "pretrain"` gate (not
  an implicit `W.grad is not None` check) is what makes this safe.

## Not in scope
- Registering real pretrain losses — deferred to Phase 5.
- Removing either the allow-list or the forbidden-name block — both are
  required. The allow-list prevents accidental inclusion of `joint`-tagged
  terms; the forbidden-name block catches a mistagged `joint_safe` term
  whose name reveals it targets interventional quantities. Belt and braces.
- Modifying `NeumannPropagation`, fusion, or any encoder.

## Files
- `aivc/training/loss_registry.py` — stage filter + W-disconnected
  placeholder + inline rationale comment.
- `tests/test_gradient_isolation.py` — 3 tests (isolation, counterfactual,
  forbidden-name-under-mistag).
- `train_week3.py` — post-backward guard with Phase 6 tripwire comment.
