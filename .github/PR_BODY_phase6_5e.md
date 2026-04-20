# Phase 6.5e — Cross-modal contrastive fine-tune (E1-rev1, LOCKED, pre-registered)

## Summary

**Outcome: E1-NULL.** `R_c − R_r = −0.0037`, 95 % CI `[−0.0154,
+0.0081]` (zero inside CI). The pre-registered pure-contrastive
weight-mask fine-tune of the existing `416e8b1a…` pretrain
checkpoint on PBMC10k Multiome produced no meaningful RSA shift vs
the reconstruction baseline. Lineage mismatch (PBMC → K562) is
therefore the dominant remaining hypothesis for 6.5d's outcome D,
and 6.5f scopes forward to E2 (lineage-matched pretraining).

## Locked contract recap (E1-rev1)

Pre-registered in `prompts/phase6_5e_contrastive.md` + E1-rev1 user
brief. Thresholds locked before any number was observed.

- **Hypothesis (rev1):** the parent already contained
  `cross_modal_infonce` (weight 0.5 of four pretrain terms). E1-rev1
  tests whether *re-weighting* to pure contrastive — mask
  `{masked_rna_recon:0, masked_atac_recon:0, cross_modal_infonce:1.0,
  peak_to_gene_aux:0}` — alone shifts RSA against Norman 2019. Not
  a new objective, not a new loss module.
- **Hyperparameters (locked, unchanged from parent where applicable).**
  τ = 0.1, projection dim = 128, B = 256, LR = 1e-4, AdamW
  WD 1e-4, grad clip 1.0, epochs = 1, seed = 3 with full scope
  (torch + numpy + random + cudnn + DataLoader worker/generator).
- **Projection init (pre-registered before the run):** `parent` —
  deepcopy of the parent `MultiomePretrainHead`'s `rna_proj` /
  `atac_proj` Sequentials. Fallback `seed3_fresh` was not invoked.
- **Stage.** New `joint_contrastive_only_e1`. Registers one active
  term (`cross_modal_infonce`, weight 1.0). Parent pretrain
  registrations untouched. Reuses `_cross_modal_infonce` callable
  byte-for-byte — no new loss module.
- **Persisted ckpt.** Encoders only. `rna_proj` / `atac_proj`
  discarded. Schema v1 preserved. Loadable via
  `aivc.training.ckpt_loader.load_pretrained_simple_rna_encoder`.
- **Outcome rule (locked, applied in order).** Gate 1: tripwire
  `CI_width_real > 0.30` → INCONCLUSIVE. Gate 2: E1-WIN / E1-PARTIAL
  / E1-NULL / E1-REGRESS per the table in the spec.

## Results

### Primary RSA

| Quantity | Point | 95% CI | Width |
|---|---|---|---|
| R_c = RSA_real contrastive | **−0.0621** | [−0.0739, −0.0503] | 0.024 |
| R_r = RSA_real reconstruction (frozen 6.5d) | −0.0584 | [−0.0702, −0.0467] | 0.024 |
| RSA_random (frozen 6.5d mean) | +0.0193 | [−0.0076, +0.0529] | 0.061 |
| **Δ = R_c − R_r** | **−0.0037** | [−0.0154, +0.0081] | 0.024 |
| Δ = R_c − RSA_random | −0.0814 | [−0.0931, −0.0696] | 0.024 |

CI on Δ_{c−r} computed from the contrastive-arm bootstrap
(`boot_real.npy`) shifted by the frozen R_r point estimate. R_r and
RSA_random are frozen to 6.5d's values per the spec's FORBIDDEN rule
("No re-running 6.5d RSA against a different configuration of the
random baseline").

### Outcome classification (pre-registered rationale)

Gate 1 (tripwire): `CI_width_real = 0.024 ≤ 0.30` → not
INCONCLUSIVE; continue.

Gate 2 (outcome table, in order):
- **E1-WIN** requires `R_c > +0.05 ∧ 0 ∉ CI_c`. R_c = −0.062 → **no**.
- **E1-PARTIAL** requires `R_c − R_r > +0.05 ∧ R_c ≤ +0.05`. Δ = −0.004 → **no**.
- **E1-REGRESS** requires `R_c < R_r − 0.05`. Δ = −0.004 → **no**.
- **E1-NULL** requires `|R_c − R_r| < 0.05 ∧ 0 ∈ CI_Δ(R_c − R_r)`.
  `|Δ| = 0.0037 < 0.05` ✓; `0 ∈ [−0.0154, +0.0081]` ✓ → **E1-NULL.**

### Tripwires (T1–T7, all passing)

Record: `.github/phase6_5e_rsa.json::tripwires` and
`experiments/phase6_5e/tripwires.json`. Every tripwire wrote its
measured value regardless of pass/fail, per the rev1 brief.

| # | Check | Observed | Threshold | Pass |
|---|---|---|---|---|
| T1 | Parent ckpt SHA | `416e8b1a…` | `416e8b1a…` | ✓ |
| T2 | Probe batch (seed=3, n=256) cached | `experiments/phase6_5e/probe_batch.npz` (indices SHA `27a906d0…`) | — | ✓ |
| T3 | mean\|z_post − z_pre\| on probe (RNA / ATAC, pre-projection) | 0.151 / 0.068 | > 1e-4 | ✓ |
| T4 | per-dim std min (RNA / ATAC) | 0.599 / 0.751 | ≥ 1e-6 | ✓ |
| T4 | cross-dim std mean (RNA / ATAC) | 3.78 / 1.67 | ≥ 1e-6 | ✓ |
| T4 | Latent finiteness | finite | finite | ✓ |
| T5 | Online NaN/Inf abort | 0 triggers / 47 batches | 0 | ✓ |
| T6 | Batch-0 weight mask | `{recon:0, recon:0, contrastive:1.0, aux:0}` | locked | ✓ |
| T7 | Module drift (diagnostic) | rna_enc=0.56 %, atac_enc=1.37 %, rna_proj=1.50 %, atac_proj=1.60 % | — | diag |

### Training log

- Loss (`cross_modal_infonce`) first batch: 0.588. Last batch (step 46): 0.274.
- Loss decreased monotonically over 47 batches (see W&B sparkline).
- Wall clock: 83.5 s fine-tune + 689.4 s RSA on MPS. No deviations
  from locked hyperparameters. B = 256 held (no OOM fallback).
- W&B: https://wandb.ai/quriegen/aivc-pretrain/runs/5qaltcx9

## Artifacts

- **Contrastive checkpoint:**
  `checkpoints/pretrain/pretrain_encoders_contrastive.pt`
  - SHA-256 `6084d5186cbd3dc942497d60926cda7a545931c7da5d7735ba32f555b73349ee`
  - Parent `416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e` (SHA differs — T1 on output ✓).
  - Schema v1, loadable via `load_pretrained_simple_rna_encoder`. RNA
    encoder unchanged architecture (n_genes=36,601, hidden=256,
    latent=128). `rna_proj` / `atac_proj` not persisted.
- **RSA JSON:** `.github/phase6_5e_rsa.json` (full schema, tripwires embedded).
- **Tripwire JSON (source of truth from fine-tune):**
  `experiments/phase6_5e/tripwires.json`.
- **Bootstrap distributions + latent centroids:**
  `experiments/phase6_5e/artifacts/` (gitignored).
- **Deterministic probe batch:** `experiments/phase6_5e/probe_batch.npz`
  (gitignored, cached on first run).

### Reproducibility anchors (PBMC10k + probe batch)

- `pbmc10k_h5ad_sha256        = 0e1e7689f4d9227ab7260c1c6be584e9dbbabef1c2ef834291cb9cc054363ca2`
- `parent_ckpt_sha256         = 416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e`
- `output_ckpt_sha256         = 6084d5186cbd3dc942497d60926cda7a545931c7da5d7735ba32f555b73349ee`
- `probe_batch_index_sha256   = 27a906d07cd3c47e294ab06bcc974351d269f97039d7dd43b94a9d6d8f215f64` (seed=3, 256 cells, sorted int64, little-endian)

6.5f-disambig will regenerate the probe batch under seed=3 from the
same h5ad and assert both the h5ad SHA and the probe-index SHA match
before proceeding. Any drift = STOP in 6.5f.

## Interpretation (one line)

Pure-contrastive re-weighting on PBMC10k does not move RSA against
Norman 2019: `Δ_{c−r} = −0.0037` is indistinguishable from zero, so
the objective manipulation tested by E1-rev1 is falsified as a
sufficient fix.

## 6.5f scope implication

- **Scoped in.** E2 — lineage-matched pretraining (K562 / Perturb-Seq
  corpus). E1-NULL rules out the cheap objective-weighting
  explanation; lineage mismatch is the dominant remaining hypothesis
  for 6.5d's outcome D.
- **Deprioritized.** Other objective-only manipulations on PBMC
  (τ sweep, different projection head, perturbation-aware
  contrastive without pseudo-labels). Still technically available
  but not the highest-yield next cut.
- **Ruled out for E1-rev1 specifically.** Weight-mask re-weighting
  of the parent's existing `cross_modal_infonce` term as a standalone
  sufficient fix — pre-registered NULL outcome fired cleanly (Gate 2
  row 3) with all 7 tripwires passing.

## What this PR does NOT do

- No architecture change (SimpleRNAEncoder / PeakLevelATACEncoder /
  MultiomePretrainHead all unchanged).
- No new loss module — reuses `_cross_modal_infonce` from
  `aivc/training/pretrain_losses.py` byte-for-byte.
- No modification of `scripts/phase6_5d_rsa.py` (the 6.5e RSA
  wrapper monkey-patches `EXPECTED_CKPT_SHA` in-process; the on-disk
  6.5d module is untouched — audit trail preserved).
- No post-hoc tuning after observing the number. No retry within
  6.5e. No mid-run hyperparameter changes. No alternate random
  baseline — frozen 6.5d values used.
- No escalation to E2 inside 6.5e. 6.5f is a separate PR.

## Test plan

- [x] `pytest tests/test_phase6_5e_contrastive.py` — 11 / 11 passing
      (loss reuse contract, weight-mask lock, pretrain non-interference,
      InfoNCE identity / random / symmetric sanity, projection-init
      helpers, persisted ckpt schema + projection absence).
- [x] `pytest tests/` (excluding pre-existing env-only failures in
      `tests/test_full_loop.py` and `tests/agents/` per Issue #11) —
      **450 passed, 0 failed, 0 regressions** vs the 6.5d baseline.
- [x] Parent ckpt SHA tripwire (T1) verified at process start.
- [x] Probe-batch tripwire (T2) verified and cached.
- [x] Drift / collapse / NaN / weight-mask / drift-ratio tripwires
      (T3–T7) all pass.
- [x] Output ckpt SHA differs from parent; loads cleanly via
      `load_pretrained_simple_rna_encoder`.
- [x] `.github/phase6_5e_rsa.json` contains every required 6.5e key:
      `parent_ckpt_sha`, `contrastive_ckpt_sha`,
      `rsa_real_reconstruction`, `delta_contrastive_vs_reconstruction`,
      `delta_contrastive_vs_reconstruction_ci95`,
      `delta_contrastive_vs_random`,
      `delta_contrastive_vs_random_ci95`, `outcome_e1`,
      `interpretation_e1`, `tripwires`, `training_deviations`.
- [x] Fine-tune wall clock (83.5 s) < 1 h budget.
- [x] RSA wall clock (689 s) consistent with 6.5d baseline (661 s).

## Status

**DRAFT.** Per the locked contract, `git push -u origin phase-6.5e`
was completed; PR is not opened by Claude Code. Waiting on reviewer
sign-off to convert to ready-for-review.
