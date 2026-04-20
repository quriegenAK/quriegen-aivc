"""scripts/phase6_5e_finetune_contrastive.py — Phase 6.5e (E1-rev1).

WEIGHT-CHANGE fine-tune of the existing Phase 5 pretrain checkpoint
(SHA ``416e8b1a…``) on PBMC10k Multiome under the
``joint_contrastive_only_e1`` stage — weight mask
``{masked_rna_recon: 0, masked_atac_recon: 0, cross_modal_infonce:
1.0, peak_to_gene_aux: 0}``.

REUSES:
    - ``SimpleRNAEncoder`` / ``PeakLevelATACEncoder`` / ``MultiomePretrainHead``
      — no architecture change.
    - ``_cross_modal_infonce`` from ``aivc.training.pretrain_losses``
      (same InfoNCE implementation the parent ckpt was trained with).
    - ``aivc.training.ckpt_loader.load_full_pretrain_checkpoint`` — no
      bare ``torch.load``.

TRIPWIRES (all write their measured value to
``.github/phase6_5e_rsa.json::tripwires`` regardless of pass/fail):
    T1: Parent ckpt SHA equals
        416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e.
    T2: Deterministic probe batch (seed=3, 256 cells) cached to
        experiments/phase6_5e/probe_batch.npz.
    T3: mean(|z_post - z_pre|) on probe batch, pre-projection encoder
        output, must exceed 1e-4 (training actually moved weights).
    T4: Post-epoch per-dim std and cross-dim std on probe latents must
        exceed 1e-6 each (representation did not collapse).
    T5: Online NaN/Inf abort per batch (halt if any loss / grad is
        non-finite).
    T6: Batch-0 weight mask equals {recon:0, recon:0, contrastive:1.0,
        aux:0}. Inspected from pretrain_losses.E1_WEIGHT_MASK and the
        registry state at batch 0.
    T7: Diagnostic module-level drift ratios for
        {rna_encoder, atac_encoder, rna_proj, atac_proj}.

No mid-run hyperparameter changes. If any non-diagnostic tripwire
fires, STOP and report — do NOT retry within 6.5e.

Hyperparameters (locked, matching parent):
    τ = 0.1 (parent), projection dim = 128 (parent), B = 256,
    LR = 1e-4, AdamW with weight_decay = 1e-4, epochs = 1,
    gradient clipping = 1.0 (global norm).

Seed scope (locked): ``--seed 3`` sets torch, numpy, random,
cudnn.deterministic, and the DataLoader worker seed + generator.

Projection-init policy (pre-registered): try ``parent`` (preferred)
— reuse rna_proj / atac_proj weights from the parent
``MultiomePretrainHead``; fall back to ``seed3_fresh`` (fresh init
under the same global seed) if the parent head is unavailable or
schema-incompatible. The chosen path is stamped into ckpt metadata
as ``projection_init_path``.

Usage:
    python scripts/phase6_5e_finetune_contrastive.py \\
        --parent-ckpt checkpoints/pretrain/pretrain_encoders.pt \\
        --data data/pbmc10k_multiome.h5ad \\
        --out-ckpt checkpoints/pretrain/pretrain_encoders_contrastive.pt \\
        --probe-cache experiments/phase6_5e/probe_batch.npz \\
        --tripwire-json .github/phase6_5e_tripwires.json \\
        --seed 3 --epochs 1 --batch-size 256 --lr 1e-4 \\
        --temperature 0.1 --wandb-project aivc-pretrain \\
        --wandb-name phase-6.5e-contrastive-e1
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aivc.training.ckpt_loader import load_full_pretrain_checkpoint  # noqa: E402
from aivc.training.loss_registry import LossRegistry  # noqa: E402
from aivc.training.pretrain_losses import (  # noqa: E402
    E1_STAGE,
    E1_WEIGHT_MASK,
    register_joint_contrastive_only_e1_terms,
)


PARENT_EXPECTED_SHA = (
    "416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e"
)
SCHEMA_VERSION = 1


# --------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------- #
def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _set_global_seed(seed: int) -> torch.Generator:
    """Full-scope determinism: torch (CPU + CUDA + MPS), numpy, random,
    cudnn, DataLoader worker generator.

    Returns a ``torch.Generator`` seeded with ``seed`` for passing to
    DataLoader(generator=...).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _seed_worker(worker_id: int) -> None:
    """DataLoader worker init: derive worker seed from torch initial seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _param_vector(module: nn.Module) -> torch.Tensor:
    """Detach and flatten all trainable params of ``module`` into one
    1-D CPU tensor. Used by T7 drift ratios."""
    parts = [p.detach().flatten().cpu() for p in module.parameters()]
    if not parts:
        return torch.zeros(0)
    return torch.cat(parts)


# --------------------------------------------------------------------- #
# Probe batch (T2)
# --------------------------------------------------------------------- #
def _build_or_load_probe_batch(
    adata_path: Path, cache_path: Path, seed: int, n_cells: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic (rna, atac, indices) probe batch, cached to
    ``cache_path`` as a .npz. Loads h5ad directly, not via the
    MultiomeLoader (spec: 'Load from h5ad directly')."""
    if cache_path.exists():
        print(f"[6.5e:T2] probe batch cache hit: {cache_path}", flush=True)
        data = np.load(cache_path)
        return data["rna"], data["atac"], data["indices"]

    print(f"[6.5e:T2] building probe batch (seed={seed}, n={n_cells}) ...",
          flush=True)
    import anndata as ad

    adata = ad.read_h5ad(adata_path)
    n_total = adata.shape[0]
    if n_cells > n_total:
        raise RuntimeError(
            f"Requested probe n_cells={n_cells} > dataset n={n_total}."
        )
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n_total, size=n_cells, replace=False))

    rna_X = adata.X[indices]
    if sp.issparse(rna_X):
        rna_X = rna_X.toarray()
    rna_X = np.asarray(rna_X, dtype=np.float32)

    atac_X = adata.obsm["atac"][indices]
    if sp.issparse(atac_X):
        atac_X = atac_X.toarray()
    atac_X = np.asarray(atac_X, dtype=np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, rna=rna_X, atac=atac_X, indices=indices)
    print(f"[6.5e:T2] probe batch cached: {cache_path}", flush=True)
    return rna_X, atac_X, indices


@torch.no_grad()
def _encode_probe(
    rna_enc: nn.Module,
    atac_enc: nn.Module,
    rna_probe: np.ndarray,
    atac_probe: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-projection encoder output on the probe batch.
    Returns (z_rna, z_atac) as float32 numpy arrays."""
    rna_enc.eval()
    atac_enc.eval()
    rna_t = torch.from_numpy(rna_probe).to(device)
    atac_t = torch.from_numpy(atac_probe).to(device)
    z_rna, _ = rna_enc(rna_t)
    z_atac = atac_enc(atac_t)
    return z_rna.detach().cpu().numpy(), z_atac.detach().cpu().numpy()


# --------------------------------------------------------------------- #
# Data loading (fine-tune batches)
# --------------------------------------------------------------------- #
class _InMemoryMultiome(torch.utils.data.Dataset):
    """Minimal in-memory multiome dataset from an .h5ad file.

    RNA is ``adata.X`` (raw counts, float32 dense). ATAC is
    ``adata.obsm['atac']`` (sparse CSR -> dense float32 per __getitem__).

    Loads h5ad once; materializes RNA as a dense ndarray in memory
    (11898 x 36601 float32 ≈ 1.73 GB — fits on MPS 48 GB host). ATAC
    stays sparse until __getitem__ to keep peak memory tractable.
    """

    def __init__(self, h5ad_path: Path):
        import anndata as ad
        a = ad.read_h5ad(h5ad_path)
        X = a.X
        if sp.issparse(X):
            X = X.toarray()
        self.rna = np.asarray(X, dtype=np.float32)
        atac = a.obsm["atac"]
        if not sp.issparse(atac):
            atac = sp.csr_matrix(atac)
        self.atac = atac.astype(np.float32)
        self.n = self.rna.shape[0]
        self.n_genes = self.rna.shape[1]
        self.n_peaks = self.atac.shape[1]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.rna[idx], np.asarray(self.atac[idx].toarray()[0], dtype=np.float32)


def _collate(batch):
    rna = np.stack([b[0] for b in batch], axis=0)
    atac = np.stack([b[1] for b in batch], axis=0)
    return (
        torch.from_numpy(rna),
        torch.from_numpy(atac),
    )


# --------------------------------------------------------------------- #
# Projection init (pre-registered path)
# --------------------------------------------------------------------- #
def _build_projections_from_parent(
    parent_head: nn.Module,
) -> Tuple[nn.Module, nn.Module, str]:
    """Reuse rna_proj / atac_proj Sequentials from the parent ckpt's
    MultiomePretrainHead. Returns (rna_proj, atac_proj, 'parent')."""
    rna_proj = copy.deepcopy(parent_head.rna_proj)
    atac_proj = copy.deepcopy(parent_head.atac_proj)
    return rna_proj, atac_proj, "parent"


def _build_projections_fresh(
    rna_dim: int, atac_dim: int, proj_dim: int, hidden_dim: int
) -> Tuple[nn.Module, nn.Module, str]:
    """Fallback: fresh init under the already-seeded global RNG.
    Returns (rna_proj, atac_proj, 'seed3_fresh')."""
    rna_proj = nn.Sequential(
        nn.Linear(rna_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, proj_dim),
    )
    atac_proj = nn.Sequential(
        nn.Linear(atac_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, proj_dim),
    )
    return rna_proj, atac_proj, "seed3_fresh"


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--parent-ckpt", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out-ckpt", type=Path, required=True)
    p.add_argument(
        "--probe-cache",
        type=Path,
        default=Path("experiments/phase6_5e/probe_batch.npz"),
    )
    p.add_argument(
        "--tripwire-json",
        type=Path,
        default=Path("experiments/phase6_5e/tripwires.json"),
    )
    p.add_argument("--probe-n", type=int, default=256)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument(
        "--projection-init",
        choices=["parent", "seed3_fresh"],
        default="parent",
        help="Pre-registered init path. 'parent' reuses the parent "
        "MultiomePretrainHead's rna_proj / atac_proj Sequentials; "
        "'seed3_fresh' initializes fresh under the global seed.",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--wandb-project", default="aivc-pretrain")
    p.add_argument("--wandb-name", default="phase-6.5e-contrastive-e1")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> dict:  # noqa: C901 - monolithic by design for auditability
    args = parse_args(argv)
    tripwires: Dict[str, dict] = {}
    deviations: List[str] = []

    t0 = time.time()
    generator = _set_global_seed(args.seed)
    device = _select_device()
    print(f"[6.5e] device={device} seed={args.seed}", flush=True)

    # ---- T1: parent SHA ----
    parent_sha = _sha256_file(args.parent_ckpt)
    t1_pass = parent_sha == PARENT_EXPECTED_SHA
    tripwires["T1_parent_ckpt_sha"] = {
        "pass": bool(t1_pass),
        "observed": parent_sha,
        "expected": PARENT_EXPECTED_SHA,
    }
    if not t1_pass:
        raise RuntimeError(
            f"[6.5e:T1] parent ckpt SHA mismatch. expected "
            f"{PARENT_EXPECTED_SHA}, got {parent_sha}."
        )

    # ---- Load parent ckpt ----
    print("[6.5e] loading parent ckpt ...", flush=True)
    rna_enc, atac_enc, pretrain_head, parent_cfg = load_full_pretrain_checkpoint(
        args.parent_ckpt, expected_schema_version=SCHEMA_VERSION
    )
    rna_enc.to(device)
    atac_enc.to(device)

    # ---- Projection init (pre-registered) ----
    if args.projection_init == "parent":
        rna_proj, atac_proj, proj_init_path = _build_projections_from_parent(
            pretrain_head
        )
    else:
        rna_proj, atac_proj, proj_init_path = _build_projections_fresh(
            rna_dim=rna_enc.latent_dim,
            atac_dim=atac_enc.attn_dim,
            proj_dim=int(parent_cfg["proj_dim"]),
            hidden_dim=int(parent_cfg["proj_dim"] * 2),
        )
    rna_proj.to(device)
    atac_proj.to(device)
    print(f"[6.5e] projection_init_path={proj_init_path}", flush=True)

    # ---- T2: probe batch ----
    rna_probe, atac_probe, probe_idx = _build_or_load_probe_batch(
        args.data, args.probe_cache, seed=args.seed, n_cells=args.probe_n
    )
    tripwires["T2_probe_batch"] = {
        "pass": True,
        "cache_path": str(args.probe_cache),
        "n_cells": int(rna_probe.shape[0]),
        "seed": int(args.seed),
        "rna_shape": list(rna_probe.shape),
        "atac_shape": list(atac_probe.shape),
        "indices_sha256": hashlib.sha256(probe_idx.tobytes()).hexdigest(),
    }

    # ---- Pre-training encoder output on probe batch ----
    z_rna_pre, z_atac_pre = _encode_probe(
        rna_enc, atac_enc, rna_probe, atac_probe, device
    )
    rna_pre_vec = _param_vector(rna_enc)
    atac_pre_vec = _param_vector(atac_enc)
    rna_proj_pre_vec = _param_vector(rna_proj)
    atac_proj_pre_vec = _param_vector(atac_proj)

    # ---- T6: weight mask at batch 0 ----
    registry = LossRegistry()
    register_joint_contrastive_only_e1_terms(registry)
    registered_for_e1 = {
        t.name: t.weight for t in registry.terms() if t.stage == E1_STAGE
    }
    effective_mask = {
        name: float(registered_for_e1.get(name, 0.0))
        for name in E1_WEIGHT_MASK
    }
    t6_pass = effective_mask == E1_WEIGHT_MASK
    tripwires["T6_weight_mask"] = {
        "pass": bool(t6_pass),
        "observed": effective_mask,
        "expected": E1_WEIGHT_MASK,
        "stage": E1_STAGE,
    }
    if not t6_pass:
        raise RuntimeError(
            f"[6.5e:T6] weight mask mismatch. expected {E1_WEIGHT_MASK}, "
            f"got {effective_mask}."
        )

    # ---- Optimizer ----
    params = (
        list(rna_enc.parameters())
        + list(atac_enc.parameters())
        + list(rna_proj.parameters())
        + list(atac_proj.parameters())
    )
    optim = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.weight_decay
    )

    # ---- Data ----
    print("[6.5e] loading multiome dataset ...", flush=True)
    dataset = _InMemoryMultiome(args.data)
    if dataset.n_genes != rna_enc.n_genes:
        raise RuntimeError(
            f"[6.5e] RNA n_genes mismatch: ckpt={rna_enc.n_genes} "
            f"data={dataset.n_genes}."
        )
    if dataset.n_peaks != atac_enc.n_peaks:
        raise RuntimeError(
            f"[6.5e] ATAC n_peaks mismatch: ckpt={atac_enc.n_peaks} "
            f"data={dataset.n_peaks}."
        )

    B = int(args.batch_size)
    try:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=B,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers,
            worker_init_fn=_seed_worker if args.num_workers > 0 else None,
            generator=generator,
            collate_fn=_collate,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and B > 128:
            B = 128
            deviations.append(
                f"OOM at B=256 at loader build; fell back to B=128. {e}"
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=B,
                shuffle=True,
                drop_last=False,
                num_workers=args.num_workers,
                worker_init_fn=_seed_worker if args.num_workers > 0 else None,
                generator=generator,
                collate_fn=_collate,
            )
        else:
            raise

    # ---- W&B ----
    run = None
    if not args.no_wandb:
        try:
            import wandb
            run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                job_type="phase6_5e_contrastive_finetune",
                config={
                    "parent_ckpt_sha": parent_sha,
                    "seed": int(args.seed),
                    "temperature": float(args.temperature),
                    "batch_size": int(B),
                    "lr": float(args.lr),
                    "weight_decay": float(args.weight_decay),
                    "grad_clip": float(args.grad_clip),
                    "epochs": int(args.epochs),
                    "projection_init_path": proj_init_path,
                    "stage": E1_STAGE,
                    "weight_mask": E1_WEIGHT_MASK,
                },
            )
        except Exception as e:  # pragma: no cover
            print(f"[6.5e] wandb unavailable: {e}", flush=True)
            run = None

    # ---- Training loop ----
    print(f"[6.5e] fine-tune: epochs={args.epochs} B={B} ~"
          f"{len(dataset) // B + int(len(dataset) % B > 0)} batches/epoch",
          flush=True)
    step = 0
    loss_first: float | None = None
    loss_last: float | None = None
    loss_history: List[float] = []
    for ep in range(args.epochs):
        rna_enc.train()
        atac_enc.train()
        rna_proj.train()
        atac_proj.train()
        for rna_batch, atac_batch in loader:
            rna_batch = rna_batch.to(device, non_blocking=True)
            atac_batch = atac_batch.to(device, non_blocking=True)

            # Encoder forward (pre-projection).
            z_rna_lat, _ = rna_enc(rna_batch)
            z_atac_lat = atac_enc(atac_batch)

            # Projection + L2 normalize.
            z_rna = F.normalize(rna_proj(z_rna_lat), dim=-1)
            z_atac = F.normalize(atac_proj(z_atac_lat), dim=-1)

            total, components = registry.compute(
                stage=E1_STAGE,
                z_rna=z_rna,
                z_atac=z_atac,
                infonce_temperature=float(args.temperature),
            )

            # T5: online NaN/Inf abort per batch.
            if not torch.isfinite(total).all():
                tripwires["T5_online_nan_abort"] = {
                    "pass": False,
                    "batch": int(step),
                    "loss_value": float(total.item() if total.numel() == 1 else 0.0),
                }
                _write_tripwires(args.tripwire_json, tripwires, deviations)
                raise RuntimeError(
                    f"[6.5e:T5] NaN/Inf loss at batch {step}. STOP."
                )

            optim.zero_grad(set_to_none=True)
            total.backward()

            # T5 (grad side): check grad finiteness before step.
            grad_nonfinite = any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in params
            )
            if grad_nonfinite:
                tripwires["T5_online_nan_abort"] = {
                    "pass": False,
                    "batch": int(step),
                    "reason": "non-finite gradient",
                }
                _write_tripwires(args.tripwire_json, tripwires, deviations)
                raise RuntimeError(
                    f"[6.5e:T5] non-finite gradient at batch {step}. STOP."
                )

            torch.nn.utils.clip_grad_norm_(params, max_norm=args.grad_clip)
            optim.step()

            loss_val = float(total.item())
            if loss_first is None:
                loss_first = loss_val
            loss_last = loss_val
            loss_history.append(loss_val)
            if step % 5 == 0 or step == 0:
                print(f"[6.5e] step {step:03d} loss={loss_val:.4f} "
                      f"components={components}", flush=True)
            if run is not None:
                run.log({"step": step, "total": loss_val, **components})
            step += 1

    tripwires.setdefault(
        "T5_online_nan_abort",
        {"pass": True, "n_batches": int(step)},
    )

    # ---- T3: mean(|z_post - z_pre|) on probe ----
    z_rna_post, z_atac_post = _encode_probe(
        rna_enc, atac_enc, rna_probe, atac_probe, device
    )
    drift_rna = float(np.mean(np.abs(z_rna_post - z_rna_pre)))
    drift_atac = float(np.mean(np.abs(z_atac_post - z_atac_pre)))
    t3_pass = drift_rna > 1e-4 and drift_atac > 1e-4
    tripwires["T3_probe_drift"] = {
        "pass": bool(t3_pass),
        "mean_abs_delta_z_rna": drift_rna,
        "mean_abs_delta_z_atac": drift_atac,
        "threshold": 1e-4,
    }
    if not t3_pass:
        _write_tripwires(args.tripwire_json, tripwires, deviations)
        raise RuntimeError(
            f"[6.5e:T3] probe drift below threshold: "
            f"rna={drift_rna:.2e}, atac={drift_atac:.2e} (< 1e-4). "
            "Training did not move the encoders. STOP."
        )

    # ---- T4: collapse check on post-epoch probe latents ----
    per_dim_std_rna = float(z_rna_post.std(axis=0).min())
    per_dim_std_atac = float(z_atac_post.std(axis=0).min())
    cross_dim_std_rna = float(z_rna_post.std(axis=1).mean())
    cross_dim_std_atac = float(z_atac_post.std(axis=1).mean())
    post_finite = (
        np.isfinite(z_rna_post).all() and np.isfinite(z_atac_post).all()
    )
    t4_pass = (
        post_finite
        and per_dim_std_rna >= 1e-6
        and per_dim_std_atac >= 1e-6
        and cross_dim_std_rna >= 1e-6
        and cross_dim_std_atac >= 1e-6
    )
    tripwires["T4_collapse"] = {
        "pass": bool(t4_pass),
        "finite": bool(post_finite),
        "per_dim_std_min_rna": per_dim_std_rna,
        "per_dim_std_min_atac": per_dim_std_atac,
        "cross_dim_std_mean_rna": cross_dim_std_rna,
        "cross_dim_std_mean_atac": cross_dim_std_atac,
        "threshold": 1e-6,
    }
    if not t4_pass:
        _write_tripwires(args.tripwire_json, tripwires, deviations)
        raise RuntimeError(
            f"[6.5e:T4] representation collapsed or non-finite. STOP."
        )

    # ---- T7: module-level drift ratios (diagnostic) ----
    def _drift_ratio(pre: torch.Tensor, post_module: nn.Module) -> float:
        post = _param_vector(post_module)
        if pre.numel() == 0:
            return 0.0
        denom = float(torch.linalg.norm(pre).item()) or 1.0
        return float(torch.linalg.norm(post - pre).item() / denom)

    tripwires["T7_drift_ratio"] = {
        "pass": True,  # diagnostic
        "rna_encoder": _drift_ratio(rna_pre_vec, rna_enc),
        "atac_encoder": _drift_ratio(atac_pre_vec, atac_enc),
        "rna_proj": _drift_ratio(rna_proj_pre_vec, rna_proj),
        "atac_proj": _drift_ratio(atac_proj_pre_vec, atac_proj),
    }

    # ---- Save ckpt (encoders only; projections discarded) ----
    args.out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    # Build fresh configs matching parent, plus 6.5e metadata.
    config_to_save = dict(parent_cfg)
    config_to_save.update({
        "pretrain_stage": E1_STAGE,
        "parent_ckpt_sha": parent_sha,
        "epochs_finetuned": int(args.epochs),
        "batch_size": int(B),
        "temperature": float(args.temperature),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "grad_clip": float(args.grad_clip),
        "seed": int(args.seed),
        "projection_init_path": proj_init_path,
        "weight_mask": E1_WEIGHT_MASK,
        "loss": "cross_modal_infonce",
        "aivc_grad_guard": 1,
        "device": str(device),
        "loss_first": loss_first,
        "loss_last": loss_last,
    })
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "rna_encoder": rna_enc.cpu().state_dict(),
            "atac_encoder": atac_enc.cpu().state_dict(),
            "pretrain_head": pretrain_head.state_dict(),  # frozen parent head
            "rna_encoder_class": "aivc.skills.rna_encoder.SimpleRNAEncoder",
            "atac_encoder_class": "aivc.skills.atac_peak_encoder.PeakLevelATACEncoder",
            "pretrain_head_class": "aivc.training.pretrain_heads.MultiomePretrainHead",
            "config": config_to_save,
        },
        args.out_ckpt,
    )
    out_sha = _sha256_file(args.out_ckpt)
    print(f"[6.5e] ckpt saved: {args.out_ckpt}", flush=True)
    print(f"[6.5e] ckpt_sha256={out_sha}", flush=True)
    print(f"[6.5e] parent_ckpt_sha256={parent_sha}", flush=True)

    if out_sha == parent_sha:
        _write_tripwires(args.tripwire_json, tripwires, deviations)
        raise RuntimeError(
            "[6.5e] output ckpt SHA equals parent — training did not "
            "update weights. STOP."
        )

    elapsed = time.time() - t0
    tripwires["out_ckpt_sha256"] = {
        "pass": True,
        "observed": out_sha,
        "parent": parent_sha,
    }
    tripwires["meta"] = {
        "pass": True,
        "projection_init_path": proj_init_path,
        "loss_first": loss_first,
        "loss_last": loss_last,
        "loss_decreased": bool(
            loss_first is not None and loss_last is not None
            and loss_last < loss_first
        ),
        "loss_curve_len": len(loss_history),
        "batch_size": int(B),
        "wall_clock_seconds": float(elapsed),
    }

    _write_tripwires(args.tripwire_json, tripwires, deviations)
    if run is not None:
        try:
            run.config.update(
                {"ckpt_sha256": out_sha},
                allow_val_change=True,
            )
            run.finish()
        except Exception:  # pragma: no cover
            pass

    return {
        "out_ckpt": str(args.out_ckpt),
        "out_sha": out_sha,
        "parent_sha": parent_sha,
        "tripwires": tripwires,
        "deviations": deviations,
    }


def _write_tripwires(
    path: Path, tripwires: Dict[str, dict], deviations: List[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tripwires": tripwires, "training_deviations": list(deviations)}
    path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
