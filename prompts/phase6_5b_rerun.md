# Phase 6.5b — Linear Probe Rerun Against Real Norman 2019

## CONTEXT

Phase 6.5a merged. Real Norman 2019 data is now available at
data/norman2019_aligned.h5ad (36,601 genes, aligned to PBMC10k
encoder vocabulary).

The pretrained checkpoint:
  checkpoints/pretrain/pretrain_encoders.pt
  SHA: 416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e

Mock checkpoint:
  checkpoints/pretrain/pretrain_encoders_mock.pt
  SHA: c0d9715dbc76a6ecab260fe09ca5173ee7fdf6eb640538eac0f9024399a90b4e

This PR re-runs the Phase 6 interpretation gate on REAL data.

Branch: phase-6.5b off latest main (post 6.5a merge).

## PREREQUISITES

- wandb installed and authenticated (`wandb login` must succeed)
- WANDB_PROJECT=aivc-linear-probe
- data/norman2019_aligned.h5ad exists (produced by Phase 6.5a)
- Both checkpoints exist on disk

## TASK

### Step 1 — Run three linear probes (W&B logged)

```bash
# Run 1 — Real checkpoint, pretrained arm
python scripts/exp1_linear_probe_ablation.py \
    --ckpt_path checkpoints/pretrain/pretrain_encoders.pt \
    --dataset_name norman2019 \
    --dataset_path data/norman2019_aligned.h5ad \
    --seed 17 \
    --condition pretrained \
    --wandb

# Run 2 — Mock checkpoint, pretrained arm
python scripts/exp1_linear_probe_ablation.py \
    --ckpt_path checkpoints/pretrain/pretrain_encoders_mock.pt \
    --dataset_name norman2019 \
    --dataset_path data/norman2019_aligned.h5ad \
    --seed 17 \
    --condition pretrained \
    --wandb

# Run 3 — Random init
python scripts/exp1_linear_probe_ablation.py \
    --random_init \
    --n_genes 36601 \
    --dataset_name norman2019 \
    --dataset_path data/norman2019_aligned.h5ad \
    --seed 17 \
    --wandb
```

IMPORTANT: verify NONE of the three runs trigger the
"SYNTHETIC FALLBACK" warning. If any run prints this warning,
STOP — the dataset_path is not being respected.

Capture the three W&B run URLs.

### Step 2 — Assert SHA tripwire

From the W&B configs or run output:
- Real run ckpt_sha256 MUST = 416e8b1a...
- Mock run ckpt_sha256 MUST = c0d9715d...
- Random run ckpt_sha256 MUST = "n/a"

If real ckpt_sha256 = 10665544... (old mock) or c0d9715d...
(current mock), STOP. Wrong checkpoint loaded.

### Step 3 — Write .github/PHASE6_5_RESULTS.md

OVERWRITE the existing file (from the failed synthetic run).
Add a header noting this replaces the prior synthetic-data result.

Exact sections:

## Setup
- Dataset: Norman 2019 (real, aligned to PBMC10k vocabulary)
- n_cells: <from data>
- n_genes_intersection: <from 6.5a meta.json>
- Eval metric: R² on top-50 DE genes (precomputed per perturbation)
- Seed: 17
- Probe: Ridge (alpha=1.0) on encoder latent embeddings (dim=128)

## Prior result (synthetic — invalidated)
- All three arms ran on synthetic fallback data with mismatched
  n_genes (36601 / 2000 / 512). Result was FAIL with negative
  R² across all arms. Not biologically interpretable. Retained
  for audit trail in experiments/phase6_5_runs/.

## Results table (real Norman 2019)

| Run     | ckpt SHA (8) | top-50 DE R² | ΔR² vs random | W&B URL |
|---------|--------------|--------------|---------------|---------|
| Real    | 416e8b1a     | <value>      | <value>       | <url>   |
| Mock    | c0d9715d     | <value>      | <value>       | <url>   |
| Random  | n/a          | <value>      | 0 (baseline)  | <url>   |

## Phase 6 gate decision

Gate: real-checkpoint ΔR² (vs random) >= +5% relative.

- PASS  -> Phase 7 unblocked.
- SOFT  -> 0% < ΔR² < +5%. Multi-seed ablation needed (3-5 seeds)
           before Phase 7. Document proposed next ablation.
- FAIL  -> ΔR² <= 0%. Phase 7 blocked. Document hypotheses:
           (a) encoder learned PBMC structure but it doesn't transfer
               to K562 CRISPR perturbations (expected if biological
               programs are cell-type-specific)
           (b) pretraining learned modality alignment, not gene programs
           (c) 5000 steps insufficient
           (d) gene intersection too sparse (too many zeros in input)

Also report (real - mock) ΔR². If real <= mock, flag:
"Real data is not contributing signal beyond architecture prior."

## Decision
<PASS | SOFT | FAIL>, dated YYYY-MM-DD, with 2-3 line justification.

### Step 4 — Update REAL_DATA_BLOCKERS.md

In the "Phase 6.5 results (real-data gate)" section:
- Replace the synthetic-run values with the real Norman 2019 values
- Add: "Prior synthetic result invalidated — see PHASE6_5_RESULTS.md"
- Add W&B URLs

### Step 5 — Open PR

Title: "Phase 6.5b: linear-probe rerun — real Norman 2019 — <PASS|SOFT|FAIL>"
Body: .github/PHASE6_5_RESULTS.md

Comment on PR:
  "Phase 6.5 re-run complete on real Norman 2019 data.
   Decision: <PASS|SOFT|FAIL>.
   Real ΔR² vs random: <value>
   Real ΔR² vs mock:   <value>
   W&B runs: <url1>, <url2>, <url3>
   Phase 7 status: <UNBLOCKED | NEEDS MULTI-SEED | BLOCKED>"

Do not merge — human review required.

## CONSTRAINTS

- No edits to: aivc/training/*, aivc/skills/*, aivc/data/*,
  train_week3.py, scripts/pretrain_multiome.py,
  scripts/harmonize_peaks.py, scripts/build_norman2019_aligned.py
- Allowed edits:
    .github/PHASE6_5_RESULTS.md (overwrite)
    .github/REAL_DATA_BLOCKERS.md (update Phase 6.5 section)
    .github/PR_BODY_phase6_5b.md (new)
- No code changes in this PR — it's a pure evaluation run.

## VALIDATION

- Three W&B runs exist with distinct ckpt_sha256 values
- PHASE6_5_RESULTS.md has no <value> or <url> placeholders
- The PASS/SOFT/FAIL decision matches the table values arithmetically
- No "SYNTHETIC FALLBACK" warnings in any run output

## FAILURE HANDLING

- If any run crashes: STOP and report traceback. Do not retry blindly.
- If SYNTHETIC FALLBACK warning fires: STOP. The dataset_path flow
  is broken — 6.5a has a bug.
- If ckpt_sha256 mismatch: STOP. Wrong checkpoint.
- If wandb not authenticated: STOP. This phase requires W&B.
