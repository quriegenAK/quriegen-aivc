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
tests/             # pytest suite (437+ tests)
checkpoints/       # Gitignored model checkpoints
data/              # Gitignored datasets + peak_sets/ (with committed README)
.github/           # PR bodies, REAL_DATA_BLOCKERS.md, phase results
```

## Current phase: 6.5 (linear probe evaluation)
Phases 1-6.7b complete. Real-data PBMC10k Multiome pretrained
checkpoint exists (SHA 416e8b1a...). Phase 6.5 linear probe
against Norman 2019 failed due to synthetic fallback data —
re-running with real Norman 2019 data.

## Key artifacts (all gitignored)
- `checkpoints/pretrain/pretrain_encoders.pt` — real-data, schema_version=1
  SHA: 416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e
- `checkpoints/pretrain/pretrain_encoders_mock.pt` — mock baseline
  SHA: c0d9715dbc76a6ecab260fe09ca5173ee7fdf6eb640538eac0f9024399a90b4e
- `data/pbmc10k_multiome.h5ad` — 11898 cells x 36601 genes x 115720 peaks
- `data/peak_sets/pbmc10k_hg38_20260415.tsv` — 115720 peaks

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

## Known issues
- Issue #7: test pollution in tests/test_ckpt_loader.py (pre-existing)
- Issue #11: env-only test failures (anthropic SDK, typer unpinned)
- Two whitelisted legacy bare torch.load in train_hpc, evaluate_zero_shot

## Conventions
- Branch naming: phase-{N}.{sub} (e.g., phase-6.5-infra)
- PR body: .github/PR_BODY_phase{N}_{sub}.md
- SHA-256 reproducibility contracts in .github/REAL_DATA_BLOCKERS.md
- Never commit .h5ad, .pt, .tsv data files
- All phase prompts follow CONTEXT/TASK/CONSTRAINTS/VALIDATION/FAILURE HANDLING/PR PREPARATION format
