# Phase 6.5f-disambig — Frozen-Projection Disambiguation (F-NULL)

**Outcome: F-NULL** (pre-registered).

**One-sentence interpretation:** Freezing `rna_proj` / `atac_proj` —
which routes all gradient into the encoders by construction — produces
no detectable RSA shift vs the 6.5e contrastive-only baseline, so
projection-absorption is falsified as a cause and the 6.5e E1-NULL
result is attributable to lineage, not gradient routing.

**6.5g / E2 implication:** Commit to E2 (K562 / Perturb-Seq lineage
pretraining) with higher confidence. PBMC-recipe optimization
(encoder-LR sweep, τ sweep, longer training) is deprioritized — both
within-objective (6.5e) and within-optimizer-routing (6.5f)
manipulations on PBMC returned NULL.

---

## Summary

Single-variable disambiguation of Phase 6.5e's E1-NULL outcome.

Phase 6.5e returned `R_c_e1 = −0.0621`, `Δ_{c−r} = −0.0037`,
`0 ∈ CI_Δ` → E1-NULL. The pre-registered T7 drift-ratio diagnostic
fired: `rna_enc/rna_proj = 0.376 ≪ 1` — the RNA encoder absorbed ~3×
less gradient mass than `rna_proj`. Two interpretations:
(1) lineage mismatch (PBMC → K562) — proceed to E2; (2) projection
absorption — the encoder direction never shifted meaningfully because
the projection head ate the gradient, and a cheap freeze could move
RSA on PBMC alone.

Phase 6.5f is the minimal single-variable test that separates (1)
from (2). The only change vs 6.5e: `rna_proj` and `atac_proj` have
`requires_grad=False` and are excluded from the AdamW parameter group.
All other hyperparameters, data, stage, loss mask, seed, and probe
batch are held bit-identical.

Result: freezing produces no detectable RSA shift. Interpretation (2)
is falsified; interpretation (1) stands.

---

## Locked contract recap

Single-variable: only `rna_proj.requires_grad = False` and
`atac_proj.requires_grad = False` (plus AdamW built AFTER freeze with
projection params excluded). Everything else bit-identical to 6.5e
runtime (τ=0.1, proj_dim=128, LR=1e-4, 1 epoch, batch 256, seed 3,
stage `joint_contrastive_only_e1`, loss mask
`{recon:0, recon:0, contrastive:1.0, aux:0}`, grad clip 1.0,
AIVC_GRAD_GUARD=1, projection init `parent`, reused 6.5e probe batch).

---

## 6.5e T7 diagnostic → 6.5f scope

The T7 drift-ratio pre-registered follow-up rule fired on 6.5e:

| 6.5e Module | Relative weight drift |
|---|---|
| `rna_encoder` | 0.56% |
| `atac_encoder` | 1.37% |
| `rna_proj` | 1.50% (absorbed ~3× more than encoder) |
| `atac_proj` | 1.60% (near-parity with encoder) |

`drift_ratio_rna = 0.376`, `drift_ratio_atac = 0.854`. Projection
heads absorbed more gradient mass than encoders — and projection
weights are discarded at ckpt save time. Freeze them; all gradient
must flow into encoders. 6.5f tests whether that routing change
produces an RSA shift.

---

## Result

| Quantity | Point | 95% CI | Width |
|---|---|---|---|
| **R_b = RSA_real frozen-proj** | **−0.0621** | [−0.0738, −0.0504] | 0.0234 |
| R_c_e1 (frozen 6.5e) | −0.0621 | [−0.0739, −0.0503] | 0.0235 |
| R_r (frozen 6.5d) | −0.0584 | [−0.0702, −0.0467] | 0.0236 |
| RSA_random (frozen 6.5d) | +0.0193 | [−0.0076, +0.0529] | 0.061 |
| **Δ_{b−c_e1} (primary gate)** | **−0.00005** | [−0.0117, +0.0117] | 0.0234 |
| Δ_{b−r} | −0.0037 | [−0.0154, +0.0080] | 0.0234 |
| Δ_{b−random} | −0.0814 | [−0.0931, −0.0697] | 0.0234 |

`CI_width_real = 0.0234 < 0.30` (Gate 1 not triggered).
`|Δ_{b−c_e1}| = 5.4e−05 ≪ 0.05` AND `0 ∈ CI_{b−c_e1}` →
**Gate 2 row 3 (F-NULL) fires cleanly**.

The 6.5f result tracks 6.5e to within single-bootstrap-resample noise.
R_b vs R_c_e1 differ by `5.4e-05`; CI midpoints differ by less than
half the CI half-width of either.

---

## Encoder drift comparison: 6.5e vs 6.5f

| Modality | 6.5e drift% | 6.5f drift% | Delta |
|---|---|---|---|
| `rna_encoder` | 0.5640% | 0.5729% | **+0.0089%** |
| `atac_encoder` | 1.3673% | 1.3948% | **+0.0275%** |
| `rna_proj` | 1.5023% | 0.0000% (frozen) | — |
| `atac_proj` | 1.6045% | 0.0000% (frozen) | — |

Freezing projections **did** route more gradient into the encoders —
T3 probe-drift (`mean|Δz|` on a 256-cell held-out batch) went up
measurably:

| | 6.5e | 6.5f |
|---|---|---|
| `mean|Δz_rna|` | 0.151 | **0.247** |
| `mean|Δz_atac|` | 0.068 | **0.086** |

But the additional encoder weight motion is tiny in *relative* terms
(~0.01–0.03 percentage points) and produces no detectable shift in
the cross-cell-type RSA against Norman 2019. The encoder direction is
stable under contrastive pressure on PBMC regardless of whether the
projection heads absorb gradient or not.

---

## Outcome classification (locked)

Applied in order:

**Gate 1 — Tripwire.** `CI_width_real = 0.0234 < 0.30`. Does not fire.

**Gate 2 — Outcome table.**

- **F-WIN** requires `R_b > +0.05 AND 0 ∉ CI_b`. `R_b = −0.0621` fails
  both. Not fired.
- **F-PARTIAL** requires `Δ_{b−c_e1} > +0.05 AND R_b ≤ +0.05`.
  `Δ_{b−c_e1} = −5.4e-05` fails. Not fired.
- **F-REGRESS** requires `R_b < R_c_e1 − 0.05 = −0.1121`.
  `R_b = −0.0621 ≫ −0.1121`. Not fired.
- **F-NULL** requires `|Δ_{b−c_e1}| < 0.05 AND 0 ∈ CI_Δ`.
  `|Δ| = 5.4e-05 < 0.05` ✓, `CI = [−0.0117, +0.0117]` contains 0 ✓.
  **Fires cleanly.**

Rationale: Both sides of the single variable (route gradient into
projections vs into encoders) produce the same RSA on K562. The
encoder's direction under contrastive pressure on PBMC does not
depend on where gradient is absorbed — both land in the same
anti-aligned local neighborhood of the loss surface. This is
consistent with the lineage interpretation and inconsistent with
the projection-absorption interpretation.

---

## Combined 6.5e + 6.5f conclusion

Weight motion is nearly identical between 6.5e and 6.5f, but probe
latent motion `mean|Δz_rna|` grew from 0.151 to 0.247 (**+64%**).
Despite the encoder moving substantially further in latent space in
the direction the contrastive loss prefers, RSA against Norman 2019
K562 perturbation geometry is bit-identical to 6.5e at the second
decimal (`R_b = −0.0621` vs `R_c_e1 = −0.0621`, `Δ = −5.4e-05`).

Two orthogonal PBMC-side manipulations — **objective re-weighting**
in 6.5e (recon→0, contrastive→1.0) and **gradient routing** in 6.5f
(freeze projections, force encoder-only updates) — both return NULL.
Between them, they exhaust the "keep the PBMC data and recipe, change
only one lever" hypothesis class. The PBMC data manifold does not
contain axes that align with K562 perturbation response, regardless
of objective weighting or optimizer routing.

The corollary for experimental design: further single-variable PBMC
interventions that hold the data fixed have low prior probability of
moving RSA. The remaining productive levers are **data-side** (change
the pretraining corpus — E2) or **label-side** (introduce
perturbation-aware supervision — not tested by 6.5e/6.5f; see
section below).

---

## 6.5g / E2 implication

- **Committed to:** E2 (K562 / Perturb-Seq lineage pretraining). Gate-2
  outcome F-NULL, combined with 6.5e E1-NULL, triggers the
  "commit to E2 with higher confidence" branch. Lineage mismatch is
  the dominant remaining hypothesis for 6.5d's outcome D.
- **Deprioritized:** PBMC-recipe optimization on the existing corpus
  (encoder-LR sweeps, τ sweeps, alternative projection heads, longer
  training on PBMC10k). Both single-variable manipulations on PBMC
  returned NULL, so further PBMC-only recipe tuning has low expected
  value for K562 alignment.
- **Not tested, remains available:** cell-type-aware contrastive
  objectives that introduce pseudo-labels or perturbation-aware
  negatives. This is a structurally different lever than 6.5e
  (objective re-weighting) or 6.5f (gradient routing) — it changes
  *what* the contrastive loss discriminates, not just *how* its
  gradient is applied. 6.5e/6.5f do **not** speak to this
  direction. It remains deprioritized given the combined NULL
  signal but is not ruled out by the current evidence.

---

## Checkpoint artifact

- Path: `checkpoints/pretrain/pretrain_encoders_frozen_proj.pt` (gitignored)
- SHA: `936dfa44071322f7d62c3e698e4395e875fd02a201d171647ad8734dccfa54d7`
- Parent SHA: `416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e`
- 6.5e baseline SHA (reference): `6084d5186cbd3dc942497d60926cda7a545931c7da5d7735ba32f555b73349ee`
- Distinct from parent and 6.5e (runtime guard; tripwire out_ckpt_sha256)
- Schema version: 1
- Loads cleanly via `aivc.training.ckpt_loader.load_pretrained_simple_rna_encoder`
- Metadata: `projections_frozen=true`, `projection_init="parent"`,
  `projection_dim=128`, `pretrain_stage="joint_contrastive_only_e1"`,
  `seed=3`, `loss_weights={recon:0, recon:0, contrastive:1.0, aux:0}`,
  `aivc_grad_guard=1`

---

## Tripwire log (T1–T8, all pass)

| # | Observed | Pass |
|---|---|---|
| T1 parent ckpt SHA | `416e8b1a…` == expected (full 64 chars) | ✓ |
| T2 probe batch | data SHA `0e1e7689…` ✓; index SHA `27a906d0…` ✓; cache hit on `experiments/phase6_5e/probe_batch.npz` | ✓ |
| T3 probe drift | `mean\|Δz_rna\|=0.247`, `mean\|Δz_atac\|=0.086` (≫ 1e-4) | ✓ |
| T4 collapse | per-dim std min = 0.5746 (rna), 0.7496 (atac); cross-dim std mean = 3.796 (rna), 1.656 (atac); finite | ✓ |
| T5 online NaN/Inf | 47/47 batches clean | ✓ |
| T6 weight mask | `{masked_rna_recon:0.0, masked_atac_recon:0.0, cross_modal_infonce:1.0, peak_to_gene_aux:0.0}` exact match | ✓ |
| T7 drift ratios | `rna_enc=0.5729%, atac_enc=1.3948%, rna_proj=0.0, atac_proj=0.0`; `drift_ratio_rna=inf`, `drift_ratio_atac=inf` (projection drift = 0 by construction) | ✓ |
| **T8-a** optimizer exclusion | no projection param id in any AdamW param group; every projection param has `requires_grad=False` | ✓ |
| **T8-b** max\|ΔW\| exact | `max\|ΔW\|_rna = 0.0`, `max\|ΔW\|_atac = 0.0` (exact zero, not approximate) | ✓ |

Source: `experiments/phase6_5f/tripwires.json` + merged into
`.github/phase6_5f_rsa.json::tripwires`.

---

## Test plan

New tests: `tests/test_phase6_5f_frozen_proj.py` (25 tests, all pass).

- `test_frozen_proj_requires_grad_false` — T8-a (requires_grad half)
- `test_optimizer_excludes_projections` — T8-a (optimizer-exclusion half)
- `test_optimizer_exclusion_raises_on_leak` — negative case
- `test_projection_weights_bit_identical_after_step` — T8-b (3 full
  backward+step cycles on a 4-cell tiny batch; projection tensors
  bit-identical, at least one encoder param moved)
- `test_assert_projections_equal_parent_happy` / `_raises_on_drift` —
  projection_init="parent" byte-identity invariant
- `test_reuses_6_5e_stage` / `test_freeze_flag_defaults_to_true` —
  CLI contract
- `test_ckpt_metadata_projections_frozen_true` — persisted `projections_frozen=true`
- `test_ckpt_loads_via_strict_loader` — schema-v1 loader contract
- `test_ckpt_sha_differs_from_parent_and_6_5e` — constant identity
- `test_probe_batch_reuse_indices_sha` — 27a906d0… match (skips if
  cache not present)
- `test_rsa_json_schema_keys` + 5× `test_classify_f_*` — outcome
  classifier branch coverage (F-WIN, F-PARTIAL, F-NULL, F-REGRESS,
  INCONCLUSIVE)
- `test_no_6_5d_pipeline_edits` (×3 parametrized), `test_no_6_5e_runtime_edits`
  (×3 parametrized), `test_no_aivc_training_edits_on_6_5f_branch` —
  git-diff assertions that 6.5d / 6.5e pipeline files + `aivc/training/*`
  are untouched on this branch

Full suite: **502 passed** (no regressions; all 497 post-6.5e tests +
5 `test_ckpt_loader.py`). Wall-clock `pytest tests/` ≈ 70s.

---

## Files changed on this branch

- `scripts/phase6_5f_finetune_frozen_proj.py` (new, ~570 lines) —
  runner; imports shared helpers from
  `scripts/phase6_5e_finetune_contrastive.py` with no mutation of 6.5e
  runtime paths
- `scripts/phase6_5f_enrich_rsa_json.py` (new, ~220 lines) — RSA
  wrapper over `scripts/phase6_5d_rsa.py` (monkey-patches
  `EXPECTED_CKPT_SHA` in-process only, same pattern as 6.5e)
- `tests/test_phase6_5f_frozen_proj.py` (new, 25 tests)
- `prompts/phase6_5f_frozen_proj_disambig.md` (new, locked spec)
- `.github/phase6_5f_rsa.json` (new, machine-readable result + merged tripwires)
- `.github/PHASE6_5_RESULTS.md` (appended `## Phase 6.5f-disambig`
  section; no edits to prior sections)
- `.github/PR_BODY_phase6_5f.md` (this file)

**Not touched** (verified via `test_no_*_edits`):
`aivc/training/*`, `scripts/phase6_5d_*`, `scripts/lib/rsa.py`,
`scripts/phase6_5e_finetune_contrastive.py`, `scripts/phase6_5e_rsa.py`,
`tests/test_phase6_5d_rsa.py`, `tests/test_phase6_5e_contrastive.py`.

---

## Reproducibility anchors

- PBMC10k Multiome: `data/pbmc10k_multiome.h5ad` SHA
  `0e1e7689f4d9227ab7260c1c6be584e9dbbabef1c2ef834291cb9cc054363ca2`
- Parent ckpt: `checkpoints/pretrain/pretrain_encoders.pt` SHA
  `416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e`
- 6.5e baseline ckpt (reference): SHA
  `6084d5186cbd3dc942497d60926cda7a545931c7da5d7735ba32f555b73349ee`
- 6.5f frozen-proj ckpt: SHA
  `936dfa44071322f7d62c3e698e4395e875fd02a201d171647ad8734dccfa54d7`
- Probe batch index SHA:
  `27a906d07cd3c47e294ab06bcc974351d269f97039d7dd43b94a9d6d8f215f64`

W&B run: https://wandb.ai/quriegen/aivc-pretrain/runs/5a5wspol
(run name: `phase-6.5f-disambig-frozen-proj`).

---

## Status

**DRAFT.** Per the locked contract, `git push -u origin
phase-6.5f-disambig` was completed; PR is not opened by Claude Code.
Waiting on reviewer sign-off to convert to ready-for-review.
