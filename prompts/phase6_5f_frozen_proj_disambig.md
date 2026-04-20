# Phase 6.5f — Frozen-Projection Disambiguation (LOCKED, F-disambig)

> **STATUS:** Pre-registered disambiguation experiment. 6.5e landed
> E1-NULL with T7 drift-ratio diagnostic showing `rna_enc/rna_proj = 0.37`
> — the pre-registered follow-up flag for "weight update was real but
> landed in discarded params." 6.5f tests whether freezing the
> projection heads (so all gradient flows into the encoders) moves RSA
> relative to the 6.5e baseline. This is a **single-variable test**:
> freeze-projections is the only change vs 6.5e. All other hyperparameters,
> data, stage, loss, seed, and probe batch are held identical.
>
> Outcome gates the 6.5g decision: F-NULL commits to E2 (K562 lineage
> pretrain) with higher confidence; F-WIN or F-PARTIAL pivots to a
> PBMC-recipe optimization phase instead of E2.

## CONTEXT

Phase 6.5e (contrastive re-weighting, recon→0, contrastive→1.0,
τ=0.1, proj_dim=128, 1 epoch, seed=3) returned:

- `R_c_e1 = rsa_real_contrastive_only = −0.0621`, CI `[−0.0739, −0.0503]`
- `Δ_{c−r} = −0.0037`, CI `[−0.0154, +0.0081]` → **E1-NULL** (pre-registered).
- Contrastive loss dropped monotonically (0.588 → 0.274) — training was real.
- T3 confirmed encoder motion: `mean|Δz_rna| = 0.151` — encoder moved
  in latent space.
- **T7 diagnostic** (pre-registered follow-up rule fired):

| Module | Relative weight drift | Notes |
|---|---|---|
| rna_encoder | 0.56% | — |
| atac_encoder | 1.37% | — |
| rna_proj | 1.50% | absorbed ~3× more than encoder |
| atac_proj | 1.60% | near-parity with encoder |

`drift_ratio_rna = 0.37 ≪ 1` — RNA encoder barely moved in *relative*
weight terms despite measurable z-level motion. The contrastive-space
geometry shift preferentially landed in `rna_proj`, which is discarded
at checkpoint-save time.

Two competing interpretations of 6.5e E1-NULL that cannot be separated
from 6.5e alone:

1. **Lineage interpretation (E2 hypothesis):** PBMC10k manifold does
   not contain axes that align with K562 perturbation response, and
   no objective manipulation on PBMC can fix this. Proceed to E2.
2. **Projection-absorption interpretation (6.5e-recipe hypothesis):**
   The encoder direction never shifted meaningfully toward the
   contrastive geometry because the projection head absorbed the
   gradient. A cheap recipe change (freeze projections) could
   still move RSA on PBMC data alone.

6.5f-disambig is the minimal test that separates (1) from (2):

- If freezing projections shifts RSA (F-WIN / F-PARTIAL): interpretation
  (2) was load-bearing; objective manipulation on PBMC still has runway;
  E2 is deprioritized.
- If freezing projections does NOT shift RSA (F-NULL): interpretation
  (1) is strengthened; lineage dominates; commit to E2.
- If freezing projections makes RSA worse (F-REGRESS): weird, unexpected;
  investigate before E2.

Branch: `phase-6.5f-disambig` branched off `main` after PR #18
(phase-6.5e) has merged. Same branching discipline as 6.5c → 6.5d → 6.5e.

## PREREQUISITES

- `main` has 6.5c (#16), 6.5d (#17), **AND 6.5e (#18) merged**.
  Verified by: `git log --oneline main | grep -E "phase-6\\.5(c|d|e)"`
  must show all three.
- `checkpoints/pretrain/pretrain_encoders.pt` — SHA `416e8b1a…` (parent,
  composite loss; same ckpt 6.5e fine-tuned from).
- `experiments/phase6_5e/probe_batch.npz` — **same probe batch as 6.5e**,
  loaded from the cache. If missing or `data_sha` mismatches, regenerate
  under seed=3 per 6.5e T2 spec (indices must match 6.5e's cached
  `27a906d0…` index SHA exactly — otherwise STOP).
- `data/pbmc10k_multiome.h5ad` — 11,898 cells × 36,601 genes × 115,720 peaks, unchanged.
- `data/norman2019_aligned.h5ad` — unchanged.
- Stage `joint_contrastive_only_e1` is registered in `loss_registry.py`
  (added in 6.5e). **Reused** — no new stage needed.
- `scripts/phase6_5e_finetune_contrastive.py` — used as the base; 6.5f
  adds a `--freeze-projections` mode OR implements a thin wrapper script.

## THE CONTRACT (LOCKED)

### Hypothesis

Freezing the RNA and ATAC projection heads forces all gradient signal
into the encoders. If the 6.5e E1-NULL outcome was caused by gradient
absorption in the (discarded) projections, freezing them will produce
a measurable RSA shift on PBMC-trained encoders evaluated against
K562 Norman 2019.

### The single variable

**Only change vs 6.5e:** all parameters of `rna_proj` and `atac_proj`
have `requires_grad = False` and are excluded from the AdamW parameter
group. Everything else is identical.

Invariants (must be bit-identical to 6.5e runtime, enforce via
assertions at runtime and via tests):

| Variable | Value | Same as 6.5e? |
|---|---|---|
| Parent ckpt SHA | `416e8b1a…` | ✓ |
| Stage | `joint_contrastive_only_e1` | ✓ |
| Loss weight mask | `{recon:0, contrastive:1.0, aux:0}` | ✓ |
| Temperature τ | 0.1 | ✓ |
| Projection dim | 128 | ✓ |
| Epochs | 1 | ✓ |
| Batch size | 256 (fallback 128 on OOM) | ✓ |
| Encoder LR | 1e-4 | ✓ |
| Optimizer | AdamW, wd=1e-4 | ✓ |
| Gradient clip | 1.0 global norm | ✓ |
| Seed | 3 | ✓ |
| Probe batch | cached `experiments/phase6_5e/probe_batch.npz` | ✓ (reuse) |
| AIVC_GRAD_GUARD | 1 | ✓ |
| `rna_proj.requires_grad` | **False** | ✗ (new) |
| `atac_proj.requires_grad` | **False** | ✗ (new) |
| AdamW param group | encoders only | ✗ (new) |

### Projection-init policy

Projection weights at fine-tune start must come from **the parent ckpt
or the 6.5e `projection_init` policy, whichever matches 6.5e's actual
runtime choice**. 6.5e reported `projection_init = "parent"`, so 6.5f
must also initialize projections from parent. Assert at runtime.
If 6.5e used `seed3_fresh`, 6.5f must also use `seed3_fresh`.
Verifiable via comparing projection weight tensor hashes at epoch-0.

### Checkpoint contract

- Output: `checkpoints/pretrain/pretrain_encoders_frozen_proj.pt`
- Schema: `schema_version = 1` (unchanged).
- Load contract: must load cleanly via
  `aivc.training.ckpt_loader.load_pretrained_simple_rna_encoder()`.
- Metadata:
  - `pretrain_stage = "joint_contrastive_only_e1"` (reused stage)
  - `parent_ckpt_sha = "416e8b1a…"`
  - `phase6_5e_baseline_ckpt_sha = "6084d5186c…"`
  - `data_sha = "<pbmc10k sha>"`
  - `epochs_finetuned = 1`, `batch_size = <B>`, `temperature = 0.1`
  - `projection_dim = 128`
  - `projections_frozen = true`   ← **new**
  - `projection_init = "parent"` (or `"seed3_fresh"` if 6.5e did)
  - `loss_weights = {"masked_rna_recon": 0.0, "masked_atac_recon": 0.0, "cross_modal_infonce": 1.0, "peak_to_gene_aux": 0.0}`
  - `aivc_grad_guard = 1`
  - `seed = 3`
- SHA must differ from parent `416e8b1a…` AND from 6.5e `6084d5186c…`.
- No `torch.load` without the loader contract — enforced by
  `tests/test_no_bare_torch_load.py`.

### Evaluation

Reuse `scripts/phase6_5d_rsa.py` exactly (frozen; do not edit from
this branch). Same random seeds, n_boot, min_cells_per_pert as 6.5d/6.5e.

### Pre-registered outcome classification

Let:
- `R_b` = `rsa_real_frozen_proj` (point estimate, this run).
- `R_c_e1` = `rsa_real_contrastive_only` = **−0.0621** (frozen 6.5e value).
- `R_r`   = `rsa_real_reconstruction_dominant` = **−0.0584** (frozen 6.5d value).
- `R_random` = **+0.0193** (frozen 6.5d value).
- `CI_b` = 95% bootstrap CI for R_b.
- `Δ_{b-c_e1}` = R_b − R_c_e1, pair-bootstrap CI (R_c_e1 frozen, subtract
  point estimate within each bootstrap sample of R_b).

Interpretation (locked, applied in order):

**Gate 1 — Tripwire**
- IF `CI_width_real(R_b) > 0.30` → outcome = **INCONCLUSIVE**
  (same rule as 6.5d/6.5e).

**Gate 2 — Outcome table** (compared against 6.5e baseline `R_c_e1`)

| Outcome | Condition | Interpretation | Next |
|---|---|---|---|
| **F-WIN** | `R_b > +0.05` AND `0 ∉ CI_b` | Projection absorption was load-bearing. Freezing projections lets the encoder shift into the perturbation-relevant geometry. | 6.5g = PBMC-recipe optimization phase (encoder-LR sweep, longer training, τ sweep). E2 deprioritized. |
| **F-PARTIAL** | `Δ_{b-c_e1} > +0.05` AND `R_b ≤ +0.05` | Freezing helps but doesn't fully recover. Both projection-absorption AND lineage contribute. | 6.5g pursues PBMC-recipe *and* queues E2 as parallel track. |
| **F-NULL** | `\|Δ_{b-c_e1}\| < 0.05` AND `0 ∈ CI_{b-c_e1}` | Projection absorption was NOT the cause. Encoder direction is stable under contrastive pressure on PBMC regardless of gradient routing. | Commit to E2 (K562 lineage pretrain) with higher confidence. Lineage is the dominant issue. |
| **F-REGRESS** | `R_b < R_c_e1 − 0.05` | Freezing projections actively harmed alignment. Unexpected — contradicts projection-absorption hypothesis. | STOP. Investigate: seed / optimizer / frozen-freeze implementation. Do NOT proceed to E2 before root cause. |

Additional reporting (not gates, but required in JSON):
- `Δ_{b-r}` = R_b − R_r, with CI (documents where we stand vs
  reconstruction-dominant baseline).
- `Δ_{b-random}` = R_b − R_random, with CI (documents whether the
  anti-alignment persists).

### Forbidden

- No change to architecture, data, loss, stage, τ, projection dim,
  LR, batch size, epochs, seed, gradient clip, or AIVC_GRAD_GUARD.
  Freezing projections is the **only** variable.
- No unfreezing partway through (scheduled unfreeze, warmup unfreeze,
  etc.). Full-run freeze.
- No re-running 6.5e or 6.5d with different configurations. Baselines
  are frozen point estimates.
- No edits to `scripts/phase6_5e_finetune_contrastive.py` that change
  6.5e's runtime behavior. The 6.5f runner may import from it, or copy
  and extend, but must not mutate 6.5e's logic.
- No edits to 6.5d pipeline files from `phase-6.5f-disambig`:
  `scripts/phase6_5d_rsa.py`, `scripts/lib/rsa.py`,
  `tests/test_phase6_5d_rsa.py`.
- No declaring F-WIN if Gate 1 tripwire fires. INCONCLUSIVE is valid.
- No post-hoc interpretation drift. F-NULL with high-confidence E2
  commitment is the most likely outcome and is a successful disambiguation.

## TASK

### Step 0 — Pre-flight

- Confirm `main` has 6.5c, 6.5d, AND 6.5e merged:
  `git log --oneline main | grep -E "phase-6\\.5(c|d|e)"` must show all three.
- If 6.5e is not merged yet: STOP, report. Do NOT proceed.
- Branch: `git checkout -b phase-6.5f-disambig main`.
- Verify `scripts/phase6_5e_finetune_contrastive.py` and the
  `joint_contrastive_only_e1` stage registration are present on main.
- Verify probe batch cache `experiments/phase6_5e/probe_batch.npz`
  exists and `data_sha` matches current PBMC10k SHA.
  - If file exists and SHA matches: reuse.
  - If file exists and SHA mismatches: STOP (data drifted).
  - If file missing: regenerate under seed=3 per 6.5e T2 spec.
    Regenerated indices must match the 6.5e-reported `27a906d0…`
    index SHA exactly — if not, STOP (non-reproducible cache).

### Step 1 — Implement `scripts/phase6_5f_finetune_frozen_proj.py`

Preferred design: thin wrapper around
`scripts/phase6_5e_finetune_contrastive.py`.

Responsibilities:
- Accept identical CLI as 6.5e (parent-ckpt, data, epochs, batch-size,
  LR, τ, projection-dim, stage, wandb args).
- Add `--freeze-projections` flag (default: True in 6.5f; flag exists
  only for test parameterization).
- Before building optimizer param group:
  ```
  for p in rna_proj.parameters(): p.requires_grad = False
  for p in atac_proj.parameters(): p.requires_grad = False
  trainable = [p for p in model.parameters() if p.requires_grad]
  assert len(trainable) > 0
  assert not any(p.requires_grad for p in rna_proj.parameters())
  assert not any(p.requires_grad for p in atac_proj.parameters())
  optimizer = AdamW(trainable, lr=lr, weight_decay=1e-4)
  ```
- Snapshot projection weights pre-training:
  ```
  proj_snapshot = {
      "rna_proj": {n: p.detach().clone() for n, p in rna_proj.named_parameters()},
      "atac_proj": {n: p.detach().clone() for n, p in atac_proj.named_parameters()},
  }
  ```
- After training epoch, verify bit-identical:
  ```
  for n, p in rna_proj.named_parameters():
      assert torch.equal(p, proj_snapshot["rna_proj"][n]), f"rna_proj {n} drifted"
  # ...same for atac_proj...
  ```
  This is **T8** (below).
- W&B run name: `phase-6.5f-disambig-frozen-proj`.
- Output: `checkpoints/pretrain/pretrain_encoders_frozen_proj.pt`
  with metadata block defined above.

### Step 2 — Tests: `tests/test_phase6_5f_frozen_proj.py`

Required tests:
- `test_frozen_proj_requires_grad_false`: after freeze, every projection
  parameter has `requires_grad == False`.
- `test_optimizer_excludes_projections`: the AdamW `param_groups` contain
  only encoder parameters (cross-check by parameter identity).
- `test_projection_weights_bit_identical`: run a tiny forward/backward
  on a 4-cell batch; assert projection tensors are bit-identical pre/post.
- `test_encoder_weights_move`: same tiny batch; assert at least one
  encoder parameter has changed (non-zero grad flow).
- `test_reuses_6_5e_stage`: runner dispatches
  `joint_contrastive_only_e1` (not a new stage).
- `test_ckpt_metadata_projections_frozen_true`: loaded ckpt metadata
  has `projections_frozen == True`.
- `test_ckpt_sha_differs_from_parent_and_6_5e`: output SHA ∉
  {`416e8b1a…`, `6084d5186c…`}.
- `test_probe_batch_reuse`: the runner loads the cached probe batch
  and asserts index SHA matches `27a906d0…`.
- `test_rsa_json_schema`: `.github/phase6_5f_rsa.json` contains
  `rsa_real_frozen_proj`, `rsa_real_contrastive_only`,
  `rsa_real_reconstruction_dominant`, `delta_frozen_proj_vs_contrastive_only`,
  `delta_frozen_proj_vs_reconstruction_dominant`,
  `delta_frozen_proj_vs_random`, `tripwires` block with T1–T8.
- `test_no_6_5d_pipeline_edits`, `test_no_6_5e_runtime_edits`:
  git-diff assertions that 6.5d and 6.5e pipeline files are unchanged
  on this branch.

### Step 3 — Commit 1

Title: `phase-6.5f-disambig: frozen-projection fine-tune pipeline (locked)`

Files:
- `scripts/phase6_5f_finetune_frozen_proj.py`
- `tests/test_phase6_5f_frozen_proj.py`
- `prompts/phase6_5f_frozen_proj_disambig.md` (this file)

No changes to `aivc/training/*`, `scripts/phase6_5e_*`, `scripts/phase6_5d_*`,
`scripts/lib/rsa.py`, or 6.5d/6.5e tests.

`pytest tests/` must pass with zero regression.

### Step 4 — Run fine-tune

```
python scripts/phase6_5f_finetune_frozen_proj.py \
  --parent-ckpt checkpoints/pretrain/pretrain_encoders.pt \
  --data data/pbmc10k_multiome.h5ad \
  --out-ckpt checkpoints/pretrain/pretrain_encoders_frozen_proj.pt \
  --stage joint_contrastive_only_e1 \
  --epochs 1 --batch-size 256 --lr 1e-4 --temperature 0.1 \
  --projection-dim 128 --seed 3 \
  --freeze-projections \
  --wandb-project aivc-pretrain --wandb-name phase-6.5f-disambig-frozen-proj
```

Wall clock budget: <5 minutes on MPS (6.5e took 83.5s; fewer trainable
params, should be similar or slightly faster).

### Step 5 — Verify checkpoint

- Compute SHA256 of output ckpt. Log.
- Verify SHA ∉ {parent, 6.5e}.
- Load via `ckpt_loader.load_pretrained_simple_rna_encoder`. No errors.
- Check encoder dim out == 128.

### Step 6 — RSA evaluation

```
python scripts/phase6_5d_rsa.py \
  --adata data/norman2019_aligned.h5ad \
  --real-ckpt checkpoints/pretrain/pretrain_encoders_frozen_proj.pt \
  --random-seeds 0,1,2,3,4 \
  --n-boot 1000 \
  --min-cells-per-pert 20 \
  --out-json .github/phase6_5f_rsa.json \
  --art-dir experiments/phase6_5f/artifacts
```

Enrich JSON with 6.5f-specific keys via
`scripts/phase6_5f_enrich_rsa_json.py` (small wrapper; reuse
6.5e's enrichment pattern):

- `parent_ckpt_sha`, `phase6_5e_baseline_ckpt_sha`, `frozen_proj_ckpt_sha`
- `parent_stage`, `child_stage` (=`joint_contrastive_only_e1`)
- `rsa_real_frozen_proj` + CI
- `rsa_real_contrastive_only` (=`−0.0621`, frozen 6.5e) + CI copy
- `rsa_real_reconstruction_dominant` (=`−0.0584`, frozen 6.5d) + CI copy
- `rsa_random_mean` (=`+0.0193`, frozen 6.5d)
- `delta_frozen_proj_vs_contrastive_only` + pair-bootstrap CI (primary gate)
- `delta_frozen_proj_vs_reconstruction_dominant` + CI
- `delta_frozen_proj_vs_random` + CI
- `loss_weights` (=6.5e's mask, unchanged)
- `projections_frozen` = true
- `tripwires` block with T1–T8

### Step 7 — Apply outcome classification

Per pre-registered rule. One of: F-WIN, F-PARTIAL, F-NULL, F-REGRESS,
INCONCLUSIVE.

### Step 8 — Docs

- `.github/PHASE6_5_RESULTS.md`: append `## Phase 6.5f-disambig —
  frozen-projection disambiguation` section. Keep all prior sections
  verbatim. Include the 6.5e T7 diagnostic table and the decision
  chain that led to 6.5f scope.
- `.github/PR_BODY_phase6_5f.md`: new. Summary, locked contract recap
  (emphasize: single-variable disambiguation), result table
  (R_b, R_c_e1, R_r, R_random with CIs; Δ_{b-c_e1}, Δ_{b-r}, Δ_{b-random}
  with CIs), outcome classification, 6.5g/E2 scope implication.
- `.github/REAL_DATA_BLOCKERS.md`: only update if a new data blocker
  surfaces.

### Step 9 — Commit 2

Title: `phase-6.5f-disambig: results + docs (outcome: <X>)`

Files: `.github/phase6_5f_rsa.json`, `.github/PHASE6_5_RESULTS.md`,
`.github/PR_BODY_phase6_5f.md`, `scripts/phase6_5f_enrich_rsa_json.py`.

### Step 10 — Push

`git push -u origin phase-6.5f-disambig`. STOP. Do NOT open PR. Report to user.

## VALIDATION

- `pytest tests/` passes with no regression on post-6.5e baseline.
- Frozen-proj ckpt loads cleanly, SHA differs from parent AND from 6.5e.
- W&B run `phase-6.5f-disambig-frozen-proj` completes; contrastive
  loss decreases (monotonic or near-monotonic); logged active weight
  vector matches 6.5e's.
- `.github/phase6_5f_rsa.json` has all required keys, including
  the `tripwires` block with T1–T8 outcomes.
- Outcome classification matches the locked rule.
- All runtime tripwires (below) pass. A failed tripwire is a hard STOP.

### Runtime tripwires (pre-registered, LOCKED)

Carry forward T1–T7 from 6.5e exactly (re-read 6.5e spec — thresholds,
probe batch, collapse criteria all identical). Add T8:

**T8 — Frozen-projection invariant**

- Before first training batch, snapshot every parameter in `rna_proj`
  and `atac_proj` (full state_dict clone).
- After the epoch completes, re-read the same parameters and assert
  bit-identical via `torch.equal(p_post, p_pre)` for every parameter.
- Log to `tripwires.projection_freeze_violation = <bool>`
  (must be `false`).
- On fail: STOP. The freeze was not actually applied. Investigate
  (param group construction, `requires_grad` flag, buffer leakage).
  Do NOT produce a ckpt.

Additional T3 adaptation: `delta_z_mean` threshold is unchanged
(`> 1e-4`). With fewer trainable params, encoder z-motion may be
larger (no gradient mass absorbed by projections), not smaller.
Tripwire semantics unchanged.

Additional T7 interpretation: in 6.5f, drift ratios should shift —
`drift_ratio_rna` and `drift_ratio_atac` become infinite (projection
drift = 0 by construction). Log as `drift_ratio_rna = "inf"` in the
tripwires block; absolute encoder drift percentages (`encoder_rna_drift`,
`encoder_atac_drift`) are the informative quantities. If encoder drift
in 6.5f is *less* than 6.5e's `0.56%` / `1.37%`, that is a scientifically
informative signal about optimizer dynamics — log in the PR body.

## FAILURE HANDLING

Tripwires T1–T7 carried forward exactly from 6.5e (see 6.5e spec for
follow-up actions on each). 6.5f-specific additions:

- **T8 (projection-freeze violation):** STOP immediately. Do NOT save
  ckpt. The optimizer or autograd graph is not respecting `requires_grad=False`
  on projections. Investigate: bufferized projections, torch functional
  calls that bypass param, optimizer created before freeze applied.
- **Probe batch regeneration mismatches cached indices:** STOP. Either
  the h5ad has drifted silently, or the probe-batch RNG path changed.
  Root-cause before proceeding.
- **Encoder drift < 1e-4 despite frozen projections:** STOP. All
  gradient is supposed to flow into encoders now. A no-op here means
  the optimizer didn't step, LR is effectively zero, or grad flow is
  broken upstream. Do NOT retry within 6.5f.
- **Contrastive loss diverges or plateaus at log(B):** STOP. Loss
  implementation or param routing bug. Run the 6.5e baseline with
  `--freeze-projections=False` on a held-out seed to confirm 6.5e is
  still reproducible — if yes, bug is freeze-specific.
- **Outcome F-REGRESS:** do NOT escalate to E2 yet. Regression under
  frozen projections contradicts the projection-absorption hypothesis
  AND implies some subtle harmful coupling. Root-cause before any
  further phase.
- **Outcome INCONCLUSIVE:** document, halt. Do NOT rerun with larger
  bootstrap or different seeds — post-result tuning violates protocol.

## PR PREPARATION

PR body: `.github/PR_BODY_phase6_5f.md`. Required sections:

- Summary (outcome + one-sentence interpretation + 6.5g / E2 implication).
- Locked contract recap (emphasize: single-variable disambiguation;
  freeze-projections is the only change vs 6.5e).
- 6.5e T7 diagnostic → 6.5f scope rationale (why we ran this).
- Result table:
  - `R_b`, `R_c_e1` (frozen), `R_r` (frozen), `R_random` (frozen), all with CIs.
  - `Δ_{b-c_e1}` (primary gate), `Δ_{b-r}`, `Δ_{b-random}`, all with CIs.
- Outcome classification + pre-registered rationale.
- Encoder drift comparison: 6.5e vs 6.5f (quantifies whether freezing
  changed optimizer dynamics as expected).
- 6.5g / E2 scope implication by outcome branch.
- Checkpoint artifact: path + SHA + parent SHA + 6.5e baseline SHA.
- Tripwire log table (T1–T8).
- Test plan (list of new tests + pass counts).

Open as **DRAFT** against main. Same pattern as 6.5c/6.5d/6.5e PRs.
Do not mark ready-for-review until the user reviews. Do not merge
before explicit approval.
