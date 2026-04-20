# Phase 6.5e — Contrastive Re-Weighting Fine-tune on PBMC10k (LOCKED, E1-rev1)

> **STATUS:** Pre-registered experiment. Revised after Step 1 audit
> surfaced that parent ckpt `416e8b1a…` already contains a cross-modal
> InfoNCE term (weight 0.5) alongside two reconstruction terms
> (weights 1.0 + 1.0) and a peak-to-gene aux term (weight 0.1). The
> original E1 framing ("swap reconstruction for contrastive") was
> false-premise. Revised hypothesis: the 6.5d anti-alignment
> (RSA = −0.058) is caused by the reconstruction terms dominating
> the 0.5-weighted contrastive term. Testing whether removing
> reconstruction (weights → 0, contrastive → 1.0) shifts RSA.
> If E1-rev1 fails (outcome NULL/REGRESS), escalate to E2 (K562
> Perturb-Seq lineage-matched pretrain).

## CONTEXT

Phase 6.5d RSA returned outcome D on `pretrain_encoders.pt` (SHA `416e8b1a…`):
- `RSA_real = −0.058`, CI [−0.070, −0.047]
- `RSA_random = +0.019`, CI [−0.008, +0.053]
- `Δ = −0.078`, CI [−0.083, −0.072] — ~13σ separation.

Pretrained encoder's latent-space geometry is **anti-correlated** with
Norman 2019 perturbation response geometry. No probe redesign can recover
an anti-aligned signal.

**6.5e Step 1 audit finding (invalidated original E1 premise):**
Parent checkpoint `416e8b1a…` was trained with a *composite* objective,
not a pure reconstruction objective. The `register_pretrain_terms()`
function in `aivc/training/pretrain_losses.py` registers four loss terms
under stage `pretrain`:

| Term                  | Weight | Role                                      |
|-----------------------|--------|-------------------------------------------|
| `masked_rna_recon`    | 1.0    | Masked RNA reconstruction                 |
| `masked_atac_recon`   | 1.0    | Masked ATAC reconstruction                |
| `cross_modal_infonce` | 0.5    | Symmetric CLIP-style RNA↔ATAC contrastive |
| `peak_to_gene_aux`    | 0.1    | Peak-to-gene regulatory auxiliary         |

Total loss ≈ `1.0·L_rna + 1.0·L_atac + 0.5·L_contrastive + 0.1·L_aux`.
The contrastive term is ~19% of the aggregate loss signal. The original
E1 framing ("swap reconstruction for contrastive") was **false-premise**
— contrastive is already present.

**Revised hypothesis space:**
1. **Weighting** — reconstruction gradients dominate contrastive
   gradients during pretrain, pushing the 128-d latent toward identity
   reconstruction axes and suppressing the cross-modal alignment axis
   that would encode regulatory (perturbation-responsive) state.
2. **Lineage** — PBMC10k (primary immune) features are orthogonal-or-inverse
   to K562 (erythroleukemia) perturbation axes, regardless of weighting.

E1-rev1 tests hypothesis 1 by **fine-tuning the existing checkpoint with
the existing cross-modal contrastive term as the only active loss**
(reconstruction weights → 0, contrastive weight → 1.0) on the same
PBMC10k Multiome data. The only independent variable is the loss
**weighting**. Architecture, data, temperature, projection dim, and
batch shape are held identical to the parent pretrain config to
isolate the weighting as the causal factor.

If RSA shifts meaningfully toward zero or positive → weighting was
load-bearing and 6.5f should explore intermediate weight ratios on
K562-lineage data. If RSA stays near −0.058 → lineage dominates and
E2 (K562 Perturb-Seq pretrain) is required regardless of objective
composition.

Branch: `phase-6.5e` branched off `main` (not off `phase-6.5d` — 6.5e
depends on 6.5d's RSA *result*, not its *code*; 6.5e will call
`scripts/phase6_5d_rsa.py` as a post-evaluation step, so 6.5d must merge
to main before 6.5e's evaluation step runs. See TASK Step 0 for ordering.)

## PREREQUISITES

- `main` has 6.5c merged (#16) AND 6.5d merged (#17). Verified by:
  - `git log --oneline main | grep -E "phase-6.5c|phase-6.5d"`
  - `scripts/phase6_5d_rsa.py` present on main.
- `checkpoints/pretrain/pretrain_encoders.pt` — SHA `416e8b1a…` (reconstruction baseline).
- `data/pbmc10k_multiome.h5ad` — 11,898 cells × 36,601 genes × 115,720 peaks.
- `data/norman2019_aligned.h5ad` — for downstream 6.5d-style RSA evaluation.
- `aivc.skills.rna_encoder.SimpleRNAEncoder`, `aivc.skills.atac_peak_encoder.ATACPeakEncoder`
  both importable.
- `aivc.training.ckpt_loader.load_pretrained_simple_rna_encoder` — used by probe/RSA.
- `aivc.training.loss_registry` — new loss will be registered here.
- W&B project `aivc-pretrain` authenticated.

## THE CONTRACT (LOCKED, E1-rev1)

### Objective

**Re-weight the existing pretrain loss so that `cross_modal_infonce`
(CLIP-style symmetric, RNA↔ATAC) is the sole active term.** Reuse the
existing registered loss (`aivc/training/pretrain_losses.py:
cross_modal_infonce`) with its parent hyperparameters. Do not
reimplement. The experiment is a weight-change study, not an
objective-swap study.

### Architecture

Unchanged from the 6.5d-tested checkpoint:
- `SimpleRNAEncoder(in_dim=36601, hidden=..., out_dim=128)` — identical.
- `ATACPeakEncoder(in_dim=115720, hidden=..., out_dim=128)` — identical.
- **Projection heads (inherited from parent pretrain):** two Linear→GELU→Linear
  projections, 128 → 128, identical to what the existing
  `cross_modal_infonce` loss module already uses. **Must match parent**;
  do NOT introduce a new 256-d projection. Projections are not persisted
  in the output checkpoint (consistent with the parent schema).

### Loss

Reuse the existing symmetric InfoNCE in `aivc/training/pretrain_losses.py`:

```
z_rna  = rna_proj(rna_encoder(x_rna))      # [B, 128]
z_atac = atac_proj(atac_encoder(x_atac))   # [B, 128]
# L2-normalize
z_rna  = z_rna  / ||z_rna||
z_atac = z_atac / ||z_atac||
logits = (z_rna @ z_atac.T) / τ            # [B, B]
targets = arange(B)
loss = 0.5 * (CE(logits, targets) + CE(logits.T, targets))
```

**Loss weights (pre-registered, the ONLY change vs parent):**

| Term                  | Parent weight | 6.5e weight |
|-----------------------|---------------|-------------|
| `masked_rna_recon`    | 1.0           | **0.0**     |
| `masked_atac_recon`   | 1.0           | **0.0**     |
| `cross_modal_infonce` | 0.5           | **1.0**     |
| `peak_to_gene_aux`    | 0.1           | **0.0**     |

Hyperparameters (pre-registered — **must match parent to isolate the
weight change as the only variable**):
- Temperature τ = **0.1** (parent value; do NOT change to 0.07).
- Projection dim = **128** (parent value; do NOT change to 256).
- Batch size B = 256 (MPS-safe). If OOM, fall back to 128 but log as
  deviation.
- Learning rate = 1e-4 (small — fine-tune, not pretrain from scratch).
- Optimizer = AdamW, weight_decay = 1e-4.
- Epochs = 1 (pre-registered for minimum experimental cost). One epoch
  over 11,898 cells / B=256 ≈ 47 batches. If outcome is E1-NULL or
  E1-PARTIAL after 1 epoch, document and escalate to 6.5e-v2 (more
  epochs) — do NOT retry within 6.5e.
- Gradient clipping = 1.0 (global norm).
- AIVC_GRAD_GUARD = 1 (no causal losses during this fine-tune).

### Stage routing

Register a new stage: **`joint_contrastive_only_e1`**. This stage
dispatches the existing four registered pretrain terms with the weight
vector above (recon=0, contrastive=1.0, aux=0). Implementation choice
(pre-registered, must not drift):

- **Preferred:** add `joint_contrastive_only_e1` to the stage registry
  and, at dispatch time, multiply each term's registered weight by the
  E1 weight-mask before summation. This reuses all existing term code
  paths unchanged.
- **Forbidden:** deleting or bypassing the recon/aux loss modules;
  zeroing weights must happen at the stage-dispatch layer, not by
  removing registrations. The registry guard remains active.

### Checkpoint contract

- Output: `checkpoints/pretrain/pretrain_encoders_contrastive.pt`
- Schema: `schema_version = 1` (unchanged).
- Load contract: must load cleanly via
  `aivc.training.ckpt_loader.load_pretrained_simple_rna_encoder()`.
- Metadata: `pretrain_stage = "joint_contrastive_only_e1"`,
  `parent_ckpt_sha = "416e8b1a…"`, `data_sha = "<pbmc10k sha>"`,
  `epochs_finetuned = 1`, `batch_size = <B>`, `temperature = 0.1`,
  `projection_dim = 128`,
  `loss_weights = {"masked_rna_recon": 0.0, "masked_atac_recon": 0.0,
  "cross_modal_infonce": 1.0, "peak_to_gene_aux": 0.0}`,
  `aivc_grad_guard = 1`.
- SHA must differ from `416e8b1a…`. Record new SHA in
  `.github/phase6_5e_rsa.json` and in 6.5e docs.
- No `torch.load` without the loader contract — enforced by
  `tests/test_no_bare_torch_load.py`.

### Evaluation

Run `scripts/phase6_5d_rsa.py` (already on main after #17 merge) against
the new checkpoint. Write JSON to `.github/phase6_5e_rsa.json` with
the identical schema as `phase6_5d_rsa.json`, plus:
- `parent_ckpt_sha`: `416e8b1a…`
- `contrastive_ckpt_sha`: new SHA
- `rsa_real_reconstruction` (from 6.5d): `−0.058`
- `rsa_real_contrastive` (this run)
- `delta_contrastive_vs_reconstruction`: `rsa_real_contrastive − rsa_real_reconstruction`
- `delta_contrastive_vs_random`: `rsa_real_contrastive − rsa_random`

### Pre-registered outcome classification

Let:
- `R_c` = `rsa_real_contrastive_only` (point estimate, 6.5e).
- `R_r` = `rsa_real_reconstruction_dominant` = −0.058 (frozen 6.5d value
  from parent composite-loss ckpt).
- `CI_c` = 95% bootstrap CI for R_c.

Interpretation (locked, applied in order):

**Gate 1 — Tripwire**
- IF `CI_width_real(R_c) > 0.30` → outcome = **INCONCLUSIVE**
  (sampling-noise limited, same rule as 6.5d).

**Gate 2 — Outcome table**
| Outcome | Condition | Interpretation | Next |
|---|---|---|---|
| **E1-WIN** | `R_c > +0.05` AND `0 ∉ CI_c` | Weighting fix strong. Removing reconstruction pressure recovers perturbation-relevant geometry even on PBMC lineage — contrastive signal alone is informative. | Re-run 6.5c probe on contrastive-only ckpt. If probe delta also flips positive, full story closed. 6.5f explores intermediate weight ratios. |
| **E1-PARTIAL** | `R_c − R_r > +0.05` AND `R_c ≤ +0.05` | Weighting helps but doesn't fully fix. Weighting + lineage both contribute. | 6.5e-v2 (more epochs) OR 6.5f (K562 + contrastive-dominant combined). |
| **E1-NULL** | `\|R_c − R_r\| < 0.05` AND `0 ∈ CI_Δ(R_c − R_r)` | Weighting is not load-bearing. Contrastive and recon both converge to the same (anti-aligned) geometry on PBMC. Lineage dominates. | 6.5f (K562 Perturb-Seq pretrain, E2 direction). |
| **E1-REGRESS** | `R_c < R_r − 0.05` | Removing reconstruction actively harmed alignment. Either reconstruction was partially protective, OR 1-epoch fine-tune destabilized latents (catastrophic forgetting). | STOP. Investigate: re-seed check, recon-only control run. Do NOT escalate to E2 before root cause identified. |

CI on `R_c − R_r` computed by pair-bootstrap of R_c only (R_r is frozen
from 6.5d; subtract point estimate within each bootstrap sample).

### Forbidden

- No architecture change to RNA or ATAC encoders.
- No change to training data (PBMC10k Multiome, 11,898 cells, unchanged peaks).
- No `log1p(X)` applied to encoder input — raw counts only (6.5c/6.5d lesson).
- No tuning of τ (must stay 0.1, parent value), projection dim (must
  stay 128, parent value), LR, batch size, or epochs mid-run.
- No re-implementing `cross_modal_infonce` — reuse the existing loss
  module in `aivc/training/pretrain_losses.py`.
- No early stopping based on validation loss — run the full 1 epoch.
- No causal loss. Stage must be `joint_contrastive_only_e1`, not `joint`
  or `joint_safe`. Register as a new stage, do NOT delete recon/aux loss
  registrations — zero them at dispatch only.
- No re-running 6.5d RSA against a different configuration of the random
  baseline — use the frozen `RSA_random` from 6.5d's JSON for
  recon-dominant-vs-contrastive-only comparison.
- **No edits to 6.5d pipeline files from the `phase-6.5e` branch**:
  `scripts/phase6_5d_rsa.py`, `scripts/lib/rsa.py`,
  `tests/test_phase6_5d_rsa.py`. If a fix is required, halt 6.5e and
  open a separate branch off `main`.
- No declaring E1-WIN if Gate 1 tripwire fires. INCONCLUSIVE is a valid outcome.
- No intermediate weight ratios in 6.5e (e.g., contrastive=0.8,
  recon=0.2). Intermediate ratios are 6.5f scope if E1-WIN or
  E1-PARTIAL. 6.5e tests the **extreme** of the weight spectrum to
  maximize signal-to-noise of the weighting-effect measurement.

## TASK

### Step 0 — Pre-flight

- Confirm `main` has both 6.5c and 6.5d merged (`git log --oneline main
  | grep -E "phase-6\\.5(c|d)"` must show both).
- If not merged: STOP, report. Do NOT proceed.
- Branch: `git checkout -b phase-6.5e main`.
- Verify `scripts/phase6_5d_rsa.py` and `scripts/lib/rsa.py` present.

### Step 1 — Audit existing pretrain objective [COMPLETED]

Audit finding (recorded in CONTEXT): parent ckpt uses a composite loss
with `cross_modal_infonce` already registered (weight 0.5), plus
`masked_rna_recon` (1.0), `masked_atac_recon` (1.0), `peak_to_gene_aux`
(0.1). Original E1 framing was false-premise. Revised framing:
weight-change study (recon → 0, contrastive → 1.0, aux → 0).

Audit artifact: `aivc/training/pretrain_losses.py:register_pretrain_terms`.
The existing `cross_modal_infonce` implementation uses τ=0.1 and
projection dim=128. These must be preserved for E1-rev1 to isolate
weighting as the only variable.

### Step 2 — Register `joint_contrastive_only_e1` stage

- Reuse existing loss implementations in
  `aivc/training/pretrain_losses.py`. **Do NOT create a new loss module.**
- In `aivc/training/loss_registry.py` (or equivalent stage registry),
  add stage `joint_contrastive_only_e1` with the weight mask:
  ```
  {
    "masked_rna_recon":     0.0,
    "masked_atac_recon":    0.0,
    "cross_modal_infonce":  1.0,
    "peak_to_gene_aux":     0.0,
  }
  ```
- Dispatch logic: stage-weighted sum over all four registered terms.
  Zeros must still call the loss fn (gradients short-circuit via
  weight=0), OR the dispatcher skips zero-weight terms — either is
  acceptable provided behavior is deterministic and documented in the
  stage registration docstring.
- Preserve registration-time + in-registry forbidden-name guards.

### Step 3 — Implement `scripts/phase6_5e_finetune_contrastive.py`

Responsibilities:
- Load existing `pretrain_encoders.pt` (SHA `416e8b1a…`). Verify SHA.
- Load ATAC encoder from same parent ckpt (or reconstruct via existing
  loader — do NOT re-initialize weights; fine-tune both encoders from
  parent state).
- Load/instantiate the projection heads used by parent
  `cross_modal_infonce` (128 → 128; whatever structure the existing loss
  module expects). **Do NOT introduce new projection dims.** If the
  parent projection weights are persisted, load them; if not, re-init
  with the same seed as parent (document which).
- Build parameter group: encoders + projections. All trainable at
  lr=1e-4.
- Load PBMC10k Multiome dataset (existing loader).
- Dispatch stage `joint_contrastive_only_e1` for 1 epoch. Pre-flight
  assertion: at batch 0, log the active weight vector and assert it
  matches the registered mask.
- Log per-batch loss to W&B project `aivc-pretrain`, run name
  `phase-6.5e-contrastive-only-e1`. Log the active weight vector in
  run config.
- After epoch: save encoders only (drop projections — parent schema
  does not persist them) to
  `checkpoints/pretrain/pretrain_encoders_contrastive.pt` using the
  schema-v1 checkpoint writer with the metadata block defined in
  "Checkpoint contract" above.
- Log new ckpt SHA.

### Step 4 — Tests: `tests/test_phase6_5e_contrastive.py`

- `test_joint_contrastive_only_e1_stage_registered`: stage name
  resolvable via the registry; weight mask matches the locked table
  (recon=0, contrastive=1.0, aux=0).
- `test_stage_dispatch_zeroes_recon`: mock the four term fns to return
  unit tensors; dispatched stage loss equals 1.0 (contrastive term
  only), confirming weight mask is applied.
- `test_reuses_existing_cross_modal_infonce`: stage dispatch calls into
  `aivc.training.pretrain_losses.cross_modal_infonce`, not a new
  implementation (assert via import graph or monkeypatch).
- `test_finetune_preserves_schema`: loading output ckpt via
  `ckpt_loader.load_pretrained_simple_rna_encoder` succeeds, returns
  encoder with same architecture as input.
- `test_finetune_ckpt_sha_differs`: output SHA != `416e8b1a…`.
- `test_finetune_ckpt_metadata`: loaded ckpt has
  `pretrain_stage == "joint_contrastive_only_e1"`, `temperature == 0.1`,
  `projection_dim == 128`, `loss_weights` dict matches the locked mask.
- `test_projections_not_persisted`: loaded ckpt has no `rna_proj` /
  `atac_proj` keys (matches parent schema).
- `test_forbidden_name_guard_intact`: registry still rejects forbidden
  names after 6.5e stage registration (regression guard).
- `test_rsa_json_schema`: `.github/phase6_5e_rsa.json` contains all
  required keys listed in Step 8, including the `tripwires` block and
  both `loss_weights_parent` / `loss_weights_e1` snapshots. Run as a
  post-result test after Step 8 completes.
- `test_probe_batch_determinism`: regenerating the probe batch under
  seed=3 from the same h5ad yields identical indices as the cached
  `experiments/phase6_5e/probe_batch.npz`.
- `test_no_6_5d_pipeline_edits`: the diff of `phase-6.5e` vs `main`
  touches none of `scripts/phase6_5d_rsa.py`, `scripts/lib/rsa.py`,
  `tests/test_phase6_5d_rsa.py`. Implement via a small subprocess
  `git diff --name-only main...HEAD` assertion.

### Step 5 — Commit 1

Title: `phase-6.5e: contrastive-only stage + fine-tune pipeline (E1-rev1 locked)`

Files:
- `aivc/training/loss_registry.py` (or stage registry file — stage added)
- `scripts/phase6_5e_finetune_contrastive.py`
- `tests/test_phase6_5e_contrastive.py`
- `prompts/phase6_5e_contrastive.md` (this file)

No new loss module. No changes to `aivc/training/pretrain_losses.py`
other than possibly exporting the existing `cross_modal_infonce` for
test import (read-only audit — edit only if export is genuinely missing).

`pytest tests/` must pass with zero regression.

### Step 6 — Run fine-tune

```
python scripts/phase6_5e_finetune_contrastive.py \
  --parent-ckpt checkpoints/pretrain/pretrain_encoders.pt \
  --data data/pbmc10k_multiome.h5ad \
  --out-ckpt checkpoints/pretrain/pretrain_encoders_contrastive.pt \
  --stage joint_contrastive_only_e1 \
  --epochs 1 --batch-size 256 --lr 1e-4 --temperature 0.1 \
  --projection-dim 128 \
  --wandb-project aivc-pretrain --wandb-name phase-6.5e-contrastive-only-e1
```

Wall clock budget: <1 hour on MPS. If exceeded, report and investigate.

### Step 7 — Verify checkpoint

- Compute SHA256 of output ckpt. Log.
- Load via `ckpt_loader.load_pretrained_simple_rna_encoder`. No errors.
- Check encoder dim out == 128.

### Step 8 — RSA evaluation

Run 6.5d's RSA script against the new checkpoint. **The 6.5d pipeline
is frozen:** no edits to `scripts/phase6_5d_rsa.py`, `scripts/lib/rsa.py`,
or `tests/test_phase6_5d_rsa.py` from the `phase-6.5e` branch. If a
change is genuinely required, halt 6.5e and open a separate fix
branch against `main`. Enforcement: the commit that publishes 6.5e
results must not touch any 6.5d pipeline file.

```
python scripts/phase6_5d_rsa.py \
  --adata data/norman2019_aligned.h5ad \
  --real-ckpt checkpoints/pretrain/pretrain_encoders_contrastive.pt \
  --random-seeds 0,1,2,3,4 \
  --n-boot 1000 \
  --min-cells-per-pert 20 \
  --out-json .github/phase6_5e_rsa.json \
  --art-dir experiments/phase6_5e/artifacts
```

Enrich JSON with 6.5e-specific keys. **Required keys** (enforced by
`tests/test_phase6_5e_contrastive.py::test_rsa_json_schema`):

- `parent_ckpt_sha` — `"416e8b1a…"` (full 64-char SHA).
- `contrastive_ckpt_sha` — new SHA of the fine-tuned ckpt.
- `parent_stage` — `"pretrain"` (composite loss).
- `child_stage` — `"joint_contrastive_only_e1"`.
- `rsa_real_reconstruction_dominant` — `−0.0584` (frozen 6.5d point
  estimate, full CI also copied under `rsa_real_reconstruction_dominant_ci`).
- `rsa_real_contrastive_only` — this run's point estimate, with
  `rsa_real_contrastive_only_ci`.
- `delta_contrastive_only_vs_reconstruction_dominant` — point estimate
  and CI via pair-bootstrap.
- `delta_contrastive_only_vs_random` — point estimate and CI.
- `loss_weights_parent` — `{"masked_rna_recon":1.0, "masked_atac_recon":1.0, "cross_modal_infonce":0.5, "peak_to_gene_aux":0.1}`.
- `loss_weights_e1` — `{"masked_rna_recon":0.0, "masked_atac_recon":0.0, "cross_modal_infonce":1.0, "peak_to_gene_aux":0.0}`.
- `tripwires` — object with outcomes of each runtime tripwire
  (see VALIDATION section): `parent_sha_match`, `delta_z_mean`,
  `latent_std_min_per_dim`, `latent_std_cross_dim_mean`, `nan_or_inf_detected`,
  plus the optional drift-ratio block.

Add a small wrapper script `scripts/phase6_5e_enrich_rsa_json.py` to
merge the training-side tripwire log with the RSA JSON output.

### Step 9 — Apply outcome classification

Per pre-registered rule. One of: E1-WIN, E1-PARTIAL, E1-NULL, E1-REGRESS,
INCONCLUSIVE.

### Step 10 — Docs

- `.github/PHASE6_5_RESULTS.md`: append "## Phase 6.5e — contrastive
  re-weighting fine-tune (E1-rev1)" section. Keep all prior sections
  verbatim. Include the Step 1 audit finding (parent composite-loss
  table) so the re-framing is reproducible from the record.
- `.github/PR_BODY_phase6_5e.md`: new. Summary, locked contract recap
  (emphasize: weight-change study, not objective swap), result table
  (R_c, R_r, R_random, Δ_{c−r}, Δ_{c−random}, all with CIs),
  outcome classification, 6.5f scope implication.
- `.github/REAL_DATA_BLOCKERS.md`: only update if a new data blocker
  surfaces.

### Step 11 — Commit 2

Title: `phase-6.5e: E1 results + docs (outcome: <X>)`

Files: `.github/phase6_5e_rsa.json`, `.github/PHASE6_5_RESULTS.md`,
`.github/PR_BODY_phase6_5e.md`, any small JSON-enrichment script.

### Step 12 — Push

`git push -u origin phase-6.5e`. STOP. Do NOT open PR. Report to user.

## VALIDATION

- `pytest tests/` passes with no regression on post-6.5d baseline.
- Contrastive ckpt loads cleanly, SHA differs from parent.
- W&B run `phase-6.5e-contrastive-only-e1` completes; logged active
  weight vector matches locked mask (recon=0, contrastive=1.0, aux=0);
  contrastive loss decreases monotonically over the epoch (not plateau,
  not diverging).
- `.github/phase6_5e_rsa.json` has all required keys, including
  `loss_weights_parent`, `loss_weights_e1`, and the `tripwires` block.
- Outcome classification matches the locked rule.
- All runtime tripwires (below) pass. A failed tripwire is a hard STOP
  and the JSON must record which tripwire fired.

### Runtime tripwires (pre-registered, LOCKED)

All thresholds are pre-registered. Do not tune after seeing a run.
Each tripwire writes its measured value into the `tripwires` block of
`.github/phase6_5e_rsa.json` regardless of pass/fail, so the run is
auditable even on abort.

**Global seed**

- `seed = 3`, applied to: `torch.manual_seed(3)`,
  `torch.cuda.manual_seed_all(3)` (also called on MPS for
  forward-compat), `numpy.random.seed(3)`, `random.seed(3)`,
  `torch.backends.cudnn.deterministic = True`,
  `torch.backends.cudnn.benchmark = False`.
- DataLoader: `worker_init_fn` seeds each worker via
  `seed + worker_id`. `generator=torch.Generator().manual_seed(3)`
  passed to DataLoader for shuffle determinism.
- If parent projection heads require re-initialization (fallback path
  when parent weights are not recoverable — see FAILURE HANDLING),
  re-init must occur under the same seed. Document `projection_init`
  in ckpt metadata as either `"parent"` or `"seed3_fresh"`.
- Note: seed=3 chosen for continuity with 6.5c (it survived the 6.5c
  y_sd tripwire). This is a weak selection-bias note, acknowledged in
  the PR body. Not a scientific confound for a single-point experiment.

**T1 — Parent checkpoint SHA match**

- Before loading weights, compute SHA256 of
  `checkpoints/pretrain/pretrain_encoders.pt`.
- Expected: `416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e`.
- On mismatch: STOP immediately. Log actual SHA to
  `tripwires.parent_sha_match = false` and STDERR. Do NOT proceed to
  training.

**T2 — Deterministic probe batch (cached)**

- Before training, materialize a probe batch for tripwires T3 and T4:
  ```
  rng = np.random.default_rng(3)
  probe_idx = sorted(rng.choice(n_cells, size=256, replace=False).tolist())
  ```
- Cache to `experiments/phase6_5e/probe_batch.npz` with keys
  `{idx, data_sha, seed, n_cells_total}`. If the file already exists
  and `data_sha` matches current PBMC10k SHA, reuse; else regenerate
  once and abort if regeneration differs from cached (prevents
  silent drift across re-runs). Do NOT read the probe batch via the
  training DataLoader — load directly from the h5ad to avoid any
  coupling to batch/shuffle state.
- `B_probe` = raw counts tensor at indices `probe_idx`.
  Shape: `[256, 36601]` for RNA, `[256, 115720]` for ATAC.

**T3 — Encoder drift (mean |Δz| > 1e-4)**

- Measure on the **pre-projection encoder output** (not the normalized
  projected output, which is bounded below by normalization geometry).
- Compute before training:
  ```
  with torch.no_grad(): z_pre = rna_encoder(B_probe_rna)   # [256, 128]
  ```
- Compute after epoch 1:
  ```
  with torch.no_grad(): z_post = rna_encoder(B_probe_rna)  # [256, 128]
  ```
- `delta_z_mean = (z_post − z_pre).abs().mean().item()`
- Assertion: `delta_z_mean > 1e-4`.
- On fail: STOP. Log to `tripwires.delta_z_mean`. Likely causes:
  optimizer not stepping, encoders accidentally frozen, grad flow
  short-circuited, LR effectively zero.
- Rationale for threshold: at LR=1e-4 × ~47 batches × grad norm ~O(1),
  encoder weights should move on the order of 5e-3 per active weight;
  z-level motion ≥ 1e-3 is expected easily. 1e-4 is a loose floor
  that only fires on a true no-op.

**T4 — Latent collapse (split criteria)**

- Measured at the same post-epoch probe batch as T3:
  ```
  z_post                                    # [256, 128], pre-projection encoder output
  per_dim_std    = z_post.std(dim=0)        # [128]  std across cells, per latent dim
  cross_dim_std  = z_post.std(dim=1).mean() # scalar avg of per-cell std across dims
  has_nan_or_inf = torch.isfinite(z_post).logical_not().any().item()
  ```
- Abort if ANY of:
  - (a) `has_nan_or_inf == True`, OR
  - (b) `per_dim_std.min().item() < 1e-6`   → partial mode collapse
        (at least one latent dim collapsed across the probe batch), OR
  - (c) `cross_dim_std < 1e-6`               → full mode collapse
        (all cells produce effectively identical latents).
- Log all three quantities to
  `tripwires.{latent_std_min_per_dim, latent_std_cross_dim_mean, nan_or_inf_detected}`
  regardless of pass/fail. Both dimensions of std matter: (b) is the
  classic InfoNCE collapse mode (one axis dies); (c) is the worst-case
  "all latents identical" degeneracy.
- Note: the tripwire runs on encoder output, NOT on normalized
  projected output. Normalized projections have per-row L2=1 and
  std geometry is bounded below by dimensionality, which would
  produce false passes.

**T5 — NaN / divergence (online)**

- Every batch: if contrastive loss is NaN or Inf, STOP immediately and
  save debug state (optimizer state, last batch, z_pre snapshot).
  Do NOT lower LR and retry within 6.5e.

**T6 — Weight-mask assertion (batch 0)**

- At batch 0: log the active weight vector returned by the stage
  dispatcher. Assert:
  `active_weights == {masked_rna_recon:0.0, masked_atac_recon:0.0, cross_modal_infonce:1.0, peak_to_gene_aux:0.0}`.
- On fail: STOP. Stage dispatch is wired wrong. Do NOT proceed.

**T7 (optional, but recommended) — Module-level drift-ratio logging**

Not a tripwire, a diagnostic. Log (W&B + `tripwires.drift_ratio`):

```
pre_weights  = {name: p.detach().clone() for name, p in model.named_parameters()}
# ... train 1 epoch ...
drift = {}
for name, p in model.named_parameters():
    w_pre = pre_weights[name]
    rel = (p.detach() - w_pre).norm() / (w_pre.norm() + 1e-12)
    drift[name] = rel.item()

# Aggregate per module:
encoder_rna_drift   = mean(rel for params in rna_encoder)
encoder_atac_drift  = mean(rel for params in atac_encoder)
proj_rna_drift      = mean(rel for params in rna_proj)
proj_atac_drift     = mean(rel for params in atac_proj)

drift_ratio_rna     = encoder_rna_drift  / (proj_rna_drift  + 1e-12)
drift_ratio_atac    = encoder_atac_drift / (proj_atac_drift + 1e-12)
```

Interpretation (logged, NOT used for pass/fail):
- `drift_ratio_rna ≫ 1` → training signal landed mostly in encoder (good
  — encoder moved meaningfully; downstream RSA measurement is valid).
- `drift_ratio_rna ≪ 0.1` → projections absorbed most of the gradient;
  encoder barely moved despite large contrastive loss drop. If combined
  with E1-NULL/REGRESS outcome, this is a diagnostic flag for "weight
  update was real but landed in discarded params" → follow-up phase
  should consider freezing projections or lowering their LR.

Record in `tripwires.drift_ratio = {rna: float, atac: float,
encoder_rna_drift: float, encoder_atac_drift: float,
proj_rna_drift: float, proj_atac_drift: float}`.

## FAILURE HANDLING

Tripwires T1–T6 are enumerated in VALIDATION. This section covers
non-tripwire degradations and tripwire follow-up actions.

- **OOM on B=256:** fall back to B=128. Log as deviation in JSON under
  `training_deviations`. Do NOT change other hyperparameters.
- **T1 (parent SHA mismatch):** STOP immediately. Do not attempt to
  locate an alternate parent ckpt. Report and halt.
- **T3 (encoder drift < 1e-4):** STOP. Save optimizer state and grad
  snapshots. Do NOT lower LR and retry within 6.5e. Investigate:
  grad flow broken, encoders unintentionally frozen, optimizer not
  stepping, LR effectively zero, gradient clipping nuking the update.
- **T4 (latent collapse — any of NaN/Inf, per-dim std < 1e-6, or
  cross-dim std < 1e-6):** STOP. Save debug state including `z_post`
  on probe batch. Do NOT tune τ or warm-start projections to recover.
  Root-cause as a 6.5e-follow-up phase.
- **T5 (NaN/Inf during training):** STOP training, save debug state.
  Do NOT lower LR and retry within 6.5e. Report.
- **T6 (weight-mask assertion at batch 0 fails):** STOP immediately.
  Stage dispatch is wired wrong. Do NOT proceed with training.
- **Contrastive loss does not decrease over the epoch (but is not NaN):**
  likely bug in dispatch (weight mask not applied post-batch-0, or
  contrastive fn not reached). Halt, run unit tests, investigate.
- **Parent projection weights not recoverable:** two allowed paths:
  (a) STOP and report — preferred for strict "only weighting changes"
  invariant; (b) re-initialize projections under seed=3, document as
  `projection_init="seed3_fresh"` in ckpt metadata and tripwires log,
  proceed. Path (b) slightly weakens the isolation claim — pre-register
  which path you took before the run starts, not after.
  E1-rev1 requires parent projections (or a documented deterministic
  init) to preserve the "only weighting changes" invariant.
- **RSA run fails for reasons unrelated to the contrastive ckpt:** report,
  do not patch 6.5d's RSA script from 6.5e branch.
- **Outcome E1-REGRESS:** do NOT escalate to E2. Root-cause the regression
  first (catastrophic forgetting? bad dispatch? LR too high? recon was
  partially protective?). Recon-only control re-run is allowed in a
  follow-up phase, NOT in 6.5e.
- **Outcome INCONCLUSIVE:** document, halt. Do NOT rerun with different
  bootstrap count or larger batch — retuning after seeing the result is
  a protocol violation.

## PR PREPARATION

PR body: `.github/PR_BODY_phase6_5e.md`. Required sections:
- Summary (outcome + one-sentence interpretation).
- Locked contract recap (objective, loss, hyperparameters, outcome rule).
- Result table: `R_c`, `R_r`, `R_random` with CIs; deltas with CIs.
- Outcome classification + pre-registered rationale.
- 6.5f scope implication (depends on outcome).
- Checkpoint artifact: path + SHA + parent SHA.
- Test plan.

Open as **DRAFT**. Same pattern as 6.5c/6.5d PRs.
