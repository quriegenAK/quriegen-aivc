# AIVC GeneLink — Project Context for Claude Code

## What this is
Production-grade causal multi-omics AI platform (AI Virtual Cell).
Predicts perturbation responses in primary immune cells using
multi-modal single-cell data (RNA, ATAC, Protein, Phospho).

## Stack
- Python 3.11, PyTorch 2.2.2, PyG 2.7.0
- scanpy/anndata for single-cell data
- FastAPI backend, React/TS frontend (separate repo)
- W&B for experiment tracking (project: aivc-linear-probe, aivc-pretrain)
- GCP/AWS for production; local MPS for dev

## Repository layout
```
aivc/              # Main package
  data/            # Loaders (multiome_loader, multi_perturbation_loader)
  skills/          # Encoders (rna_encoder, atac_peak_encoder, fusion, neumann)
  training/        # Loss registry, pretrain heads/losses, ckpt_loader
scripts/           # Entrypoints (pretrain_multiome, exp1_linear_probe_ablation, etc.)
configs/           # YAML configs (agent_constraints, sweep, thresholds)
tests/             # pytest suite (single root, consolidated PR #25)
checkpoints/       # Gitignored model checkpoints
data/              # Gitignored datasets + peak_sets/ (with committed README)
.github/           # PR bodies, REAL_DATA_BLOCKERS.md, phase results
```

## Current phase: 6.5g.2 — DOGMA pretrain TRAINING LIVE
Phase 6.5 closed 2026-04-20 (tag `phase-6.5-closed`).
Phase 6.5g.1 closed 2026-04-23 via PR #20 squash (`c6551ab`) with EMPTY
outcome per external wet-lab evidence + C1–C8 catalog audit.

**Phase 6.5g.2 production training running on BSC MareNostrum5** as of 2026-04-30:
- **Job 39934660**, --arm joint, 50 epochs, H100 single-GPU, 24h time budget
- Smoke (--max-steps 100) passed: loss 11.1 → 6.1 (45% drop)
- W&B offline mode (sync logs to remote post-run)
- Pre-registered success: ≥0.70 Calderon cell-type accuracy on linear probe
- Failure handling locked: sub-0.70 → architecture-class pivot, no recipe tuning

DOGMA-seq corpus (GSE156478, Mimitou 2021, *Nat Biotechnol*):
- **31,874 cells total** (DIG 18,111 + LLL 13,763), barcode-set fidelity-gated
  against `pairs_of_wnn.tsv` (caleblareau/asap_reproducibility repo).
- Protocol: RNA + ATAC + Protein (210-D TotalSeq-A).
- Batch covariate: `lysis_protocol ∈ {DIG, LLL}`; PR #44 wires it as
  scVI-style nn.Embedding(2, 8) into RNA+ATAC encoders + additive shift
  in Protein encoder.
- Peak space: union (323,500 peaks; LLL=198,947 + DIG=124,619 - shared=66).
- CD3/CD28 stim → **metadata-only** (run × stim full aliasing in single-donor
  aliquot-split design; cannot deconfound).
- Phospho + mtDNA explicitly out-of-scope (QuRIE-seq June 2026).

Pretrain readiness PR sequence (all merged to main):
- spec amendment (`d0925ce`) — pre-registered ≥0.70 threshold
- PR #41 (`e9b9871`) — peak union harmonization
- PR #42 (`8499844`) — config + AdamW + LR schedule
- PR #43 (`65d2a04`) — tri-modal pretrain wiring (Protein + `_dogma_pretrain_loss`)
- PR #44 (`eccff9e`) — lysis covariate + joint LLL+DIG loader
- PR #45 (`9ffbdd7`) — resume mechanism (approximate granularity)
- PR #46 (`4280bda`) — pre-flight smoke + `--max-steps`
- PR #47 (`e9cde69`) — joint-arm encoder forward shape-mismatch fix
- PR #48 (`6ef02ac`) — periodic checkpointing + bootstrap placeholder
- PR #49 (`5b2af59`) — BSC cluster deployment scripts
- direct (`c813773`) — BSC MareNostrum5 module stack validated on H100

Earlier 4-day DOGMA infrastructure arc (PRs #21-#25): modality_mask
contract + tri-modal MultiomeLoader + mask-aware losses + dogma_collate
+ tests root consolidation. All merged.

## BSC MareNostrum5 production deployment

Reference: [`docs/cluster/BSC_LAUNCH_GUIDE.md`](docs/cluster/BSC_LAUNCH_GUIDE.md)
for full step-by-step (Mac → transfer node → login node → compute node).

**Production CUDA stack on BSC differs from local Mac:**
- Local Mac dev: torch 2.2.2 (requirements.txt pin), MPS device
- BSC compute: pytorch/2.4.0 (system module), CUDA 11.8 (via nvidia-hpc-sdk/23.11-cuda11.8)
- pip install on BSC compute uses `--no-deps` to avoid reinstalling torch over the system module

**Module load sequence (order is load-bearing)**:
```
gcc/11.4.0 → mkl/2024.0 → impi/2021.11 → hdf5/1.14.1-2-gcc →
openblas/0.3.27-gcc → nccl/2.19.4 → nvidia-hpc-sdk/23.11-cuda11.8 →
cudnn/9.0.0-cuda11 → tensorrt/10.0.0-cuda11 → python/3.11.5-gcc →
pytorch/2.4.0
```

**SLURM resource requirements**:
- `--account=ehpc748 --qos=acc_ehpc --partition=acc`
- `--gres=gpu:1 --cpus-per-task=20` (BSC ACC partition requires 20 CPUs per GPU)
- `--time=24:00:00` (spec section 9 budget)

**Offline pip install workflow (BSC has no pypi)**:
- pip download wheels on Mac for `manylinux2014_x86_64 + python 3.11`
- rsync wheels to scratch `/gpfs/scratch/ehpc748/quri020505/aivc_wheels/`
- `salloc` interactive compute node with full module stack
- `pip install --user --no-deps --no-index --find-links=<wheels_dir> -r requirements.txt`
- User packages land at `~/.local/lib/python3.11/site-packages`
- `PYTHONPATH=$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH` in SLURM script

**h5ad write-time compatibility**: write with `compression=None` for HDF5
attribute compatibility on BSC (system HDF5 + offline-installed h5py wheel
mismatch on compressed sparse-group attributes). Re-run `build_dogma_peak_union.py`
+ `assemble_dogma_h5ad.py` with `compression=None` if regenerating.

**W&B mode**: `WANDB_MODE=offline` for production runs (no outbound on
compute nodes). Logs sync via `wandb sync <run_dir>` post-run from a
node with internet (login or local).

**Launch command (steady-state)**:
```bash
ssh quri020505@alogin1.bsc.es
cd /gpfs/scratch/ehpc748/quri020505/aivc_genelink
export WANDB_API_KEY=<key>; export WANDB_MODE=offline
sbatch scripts/submit_pretrain.slurm
squeue -u quri020505
tail -f /gpfs/scratch/ehpc748/quri020505/logs/pretrain_<JOBID>.out
```

## DOGMA batch contract
All multi-modal data flow MUST use canonical keys from
`aivc/data/modality_mask.py`:

- `ModalityKey` IntEnum (temporal order): ATAC=0, PHOSPHO=1, RNA=2, PROTEIN=3.
  Matches `TemporalCrossModalFusion.TEMPORAL_ORDER` exactly (parity-tested).
- `TEMPORAL_ORDER: list[ModalityKey]` — canonical column ordering.
- `build_mask(present, batch_size) -> (B, 4)` — mask tensor builder.
- `mask_from_obs(obs_row) -> (4,)` — per-cell mask from `has_*` obs columns.

Canonical string constants (**NEVER hardcode**):
`RNA_KEY="rna"`, `ATAC_KEY="atac_peaks"`, `PROTEIN_KEY="protein"`,
`PHOSPHO_KEY="phospho"`, `MASK_KEY="modality_mask"`,
`LYSIS_KEY="lysis_protocol"`, `PROTEIN_PANEL_KEY="protein_panel_id"`.

Enforcement: hardcoded-string audit runs on every multi-modal PR.
Loaders stamp `has_rna`/`has_atac`/`has_protein`/`has_phospho` obs columns
via `_stamp_modality_tags` in `multi_perturbation_loader.py` at load time.

## Two-layer modality gating (DO NOT COLLAPSE)
The DOGMA flow has two complementary gating mechanisms at different layers:

1. **Collate layer — strict-raise (DD2)**: `aivc/data/collate.py::dogma_collate`
   raises `ValueError` on heterogeneous batches (mixed Protein presence, mixed
   `protein_panel_id`, mixed `dataset_kind`). Within-batch heterogeneity is a
   construction error. Upstream sampler enforces batch-level modality homogeneity.

2. **Loss layer — silent-zero (D2)**: `aivc/training/pretrain_losses.py` loss
   terms and `losses.py::combined_loss_multimodal` inner fns return 0
   contribution when the entire batch lacks a modality (mask column all-zero).
   A homogeneous RNA-only batch hitting a trimodal loss term is expected —
   the loss silently skips.

These are complementary: batch-level modality absence is a valid training
signal (silent-zero); within-batch heterogeneity is a bug (raise). Future
mixed-corpus sampler authors: segregate by modality upstream; do NOT attempt
to downgrade the collate raise.

## Key artifacts (all gitignored)
- `checkpoints/pretrain/pretrain_encoders.pt` — real-data, schema_version=1
  SHA: 416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e
- `checkpoints/pretrain/pretrain_encoders_mock.pt` — mock baseline
  SHA: c0d9715dbc76a6ecab260fe09ca5173ee7fdf6eb640538eac0f9024399a90b4e
- `data/pbmc10k_multiome.h5ad` — 11898 cells x 36601 genes x 115720 peaks
- `data/peak_sets/pbmc10k_hg38_20260415.tsv` — 115720 peaks
- `data/phase6_5g_2/external_evidence/dogma_ncells_measured_2026-04-23/`
  - `dogma_ncells_measured.json` — per-arm n_cells (LLL 7,624+6,139 /
    DIG 9,483+8,628) + barcode-set fidelity verdict vs Mimitou anchor
  - `pairs_of_wnn.tsv` — Mimitou authoritative LLL 3WNN-joined cell list,
    SHA `29a7565e15e23761c41af3c3e99d24f3c1ca0acb2c8ef19323e51a23a674ee0e`

## Checkpoint contract
All checkpoints use schema_version=1. Load via
`aivc.training.ckpt_loader.load_pretrained_simple_rna_encoder()`.
No bare `torch.load()` — enforced by tests/test_no_bare_torch_load.py.

## Architecture invariants
- Loss registry pattern (aivc/training/loss_registry.py)
- Gradient isolation: AIVC_GRAD_GUARD=1 blocks causal losses in pretrain stage
- GroupNorm (not BatchNorm) in encoders
- Stage routing: pretrain, joint, joint_safe
- Defense-in-depth: registration-time + in-registry forbidden-name guards
- **Multi-modal batch-dict keys imported from `aivc.data.modality_mask`;
  no hardcoded string literals in loaders/collate/losses.**
- **`_stamp_modality_tags` called by every loader producing a multi-modal
  AnnData; `build_combined_corpus` defaults missing `has_*` columns to
  `False` (bool, not `0`).**
- **Temporal order ATAC → PHOSPHO → RNA → PROTEIN enforced in fusion
  (`TemporalCrossModalFusion.TEMPORAL_ORDER`), mask (`TEMPORAL_ORDER`),
  and collate (`dogma_collate`).**
- **Test root is SINGLE: `tests/` at repo root (consolidated in PR #25).
  No tests under `aivc/*/tests/`.**

## Known issues

**Pre-existing (unchanged)**:
- Issue #7: test pollution in tests/test_ckpt_loader.py
- Issue #11: env-only test failures (anthropic SDK, typer unpinned)
- Two whitelisted legacy bare `torch.load` in train_hpc, evaluate_zero_shot

**Resolved during DOGMA 4-day arc (2026-04-23 → 2026-04-24)**:
- ~~fusion.py zero-fill attention leak~~ — fixed PR #21 (modality_mask hotfix)
- ~~Pre-DOGMA losses without modality_mask gating~~ — fixed PR #23
- ~~Test root bifurcation `tests/` vs `aivc/data/tests/`~~ — fixed PR #25

## Conventions
- Branch naming: phase-{N}.{sub} (e.g., phase-6.5-infra)
- PR body: .github/PR_BODY_phase{N}_{sub}.md
- SHA-256 reproducibility contracts in .github/REAL_DATA_BLOCKERS.md
- Never commit .h5ad, .pt, .tsv data files
- All phase prompts follow CONTEXT/TASK/CONSTRAINTS/VALIDATION/FAILURE HANDLING/PR PREPARATION format
