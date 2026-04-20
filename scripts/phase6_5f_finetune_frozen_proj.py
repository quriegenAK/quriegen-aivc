"""scripts/phase6_5f_finetune_frozen_proj.py — Phase 6.5f-disambig.

Single-variable disambiguation of Phase 6.5e's E1-NULL outcome.

Identical runtime to 6.5e EXCEPT: ``rna_proj`` and ``atac_proj`` are
frozen (``requires_grad=False``) and excluded from the AdamW parameter
group. All other hyperparameters, data, stage, loss, seed, and probe
batch are held bit-identical. This is a contract test of the
projection-absorption hypothesis.

Helpers are imported from ``scripts.phase6_5e_finetune_contrastive``
to guarantee 6.5e's runtime is not mutated.

TRIPWIRES (all written to ``experiments/phase6_5f/tripwires.json``):
    T1: parent ckpt SHA equals
        416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e.
    T2: probe batch loaded from cache; h5ad SHA equals
        0e1e7689f4d9227ab7260c1c6be584e9dbbabef1c2ef834291cb9cc054363ca2;
        probe index SHA equals
        27a906d07cd3c47e294ab06bcc974351d269f97039d7dd43b94a9d6d8f215f64.
    T3: mean|Δz_rna|, mean|Δz_atac| on probe batch post-epoch > 1e-4.
    T4: per_dim_std.min() ≥ 1e-6; cross_dim_std.mean() ≥ 1e-6; finite.
    T5: no NaN/Inf loss / grad across 47 batches.
    T6: batch-0 weight mask == {recon:0, recon:0, contrastive:1.0, aux:0}.
    T7: module-level drift ratios (diagnostic). Projection drift is 0
        by construction under freeze; report ``drift_ratio_*="inf"``.
    T8-a: immediately after optimizer build, every projection param
          has ``requires_grad=False`` AND is NOT in any optimizer
          param group (by id()).
    T8-b: after epoch completes, BEFORE ckpt save, for every projection
          parameter ``max(|W_post - W_pre|) == 0.0`` exactly.

Projection-init policy matches 6.5e: ``projection_init="parent"``.
The parent MultiomePretrainHead's ``rna_proj`` / ``atac_proj`` are
deep-copied (via 6.5e's helper). The frozen projections therefore
start byte-identical to the parent head's projections — verified at
T8-a setup time.

Usage:
    python scripts/phase6_5f_finetune_frozen_proj.py \\
        --parent-ckpt checkpoints/pretrain/pretrain_encoders.pt \\
        --data data/pbmc10k_multiome.h5ad \\
        --out-ckpt checkpoints/pretrain/pretrain_encoders_frozen_proj.pt \\
        --probe-cache experiments/phase6_5e/probe_batch.npz \\
        --tripwire-json experiments/phase6_5f/tripwires.json \\
        --seed 3 --epochs 1 --batch-size 256 --lr 1e-4 \\
        --temperature 0.1 --freeze-projections \\
        --wandb-project aivc-pretrain \\
        --wandb-name phase-6.5f-disambig-frozen-proj
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
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
from scripts.phase6_5e_finetune_contrastive import (  # noqa: E402
    PARENT_EXPECTED_SHA,
    SCHEMA_VERSION,
    _InMemoryMultiome,
    _build_or_load_probe_batch,
    _build_projections_from_parent,
    _build_projections_fresh,
    _collate,
    _encode_probe,
    _param_vector,
    _seed_worker,
    _select_device,
    _set_global_seed,
    _sha256_file,
)


# --------------------------------------------------------------------- #
# Locked reproducibility anchors (carried from 6.5e; assert at runtime)
# --------------------------------------------------------------------- #
EXPECTED_PBMC_DATA_SHA = (
    "0e1e7689f4d9227ab7260c1c6be584e9dbbabef1c2ef834291cb9cc054363ca2"
)
EXPECTED_PROBE_INDEX_SHA = (
    "27a906d07cd3c47e294ab06bcc974351d269f97039d7dd43b94a9d6d8f215f64"
)
PHASE6_5E_CKPT_SHA = (
    "6084d5186cbd3dc942497d60926cda7a545931c7da5d7735ba32f555b73349ee"
)


def _write_tripwires(
    path: Path, tripwires: Dict[str, dict], deviations: List[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tripwires": tripwires, "training_deviations": list(deviations)}
    path.write_text(json.dumps(payload, indent=2))


# --------------------------------------------------------------------- #
# Freeze + optimizer-exclusion helpers (6.5f-specific)
# --------------------------------------------------------------------- #
def _freeze_projections(rna_proj: nn.Module, atac_proj: nn.Module) -> None:
    for p in rna_proj.parameters():
        p.requires_grad_(False)
    for p in atac_proj.parameters():
        p.requires_grad_(False)


def _snapshot_projection_state(
    rna_proj: nn.Module, atac_proj: nn.Module
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Clone projection state_dict on the current device (pre-training).

    Cloning on-device avoids dtype/rounding artifacts from a CPU round-trip
    and keeps the T8-b exact-equality check unambiguous.
    """
    return {
        "rna_proj": {
            n: p.detach().clone()
            for n, p in rna_proj.named_parameters()
        },
        "atac_proj": {
            n: p.detach().clone()
            for n, p in atac_proj.named_parameters()
        },
    }


def _assert_optimizer_excludes_projections(
    optimizer: torch.optim.Optimizer,
    rna_proj: nn.Module,
    atac_proj: nn.Module,
) -> Tuple[bool, bool]:
    """Return (optimizer_exclusion_pass, proj_requires_grad_pass).

    Both MUST be True before the first forward pass. Raises otherwise.
    """
    proj_param_ids = set()
    for p in rna_proj.parameters():
        proj_param_ids.add(id(p))
    for p in atac_proj.parameters():
        proj_param_ids.add(id(p))

    for group in optimizer.param_groups:
        for p in group["params"]:
            if id(p) in proj_param_ids:
                raise RuntimeError(
                    f"[6.5f:T8-a] projection param leaked into optimizer "
                    f"group: id={id(p)}. Freeze was not applied before "
                    "optimizer construction."
                )

    proj_params = list(rna_proj.parameters()) + list(atac_proj.parameters())
    for p in proj_params:
        if p.requires_grad:
            raise RuntimeError(
                "[6.5f:T8-a] projection param has requires_grad=True after "
                "freeze. _freeze_projections() did not take."
            )

    return True, True


def _max_abs_projection_diff(
    rna_proj: nn.Module,
    atac_proj: nn.Module,
    snapshot: Dict[str, Dict[str, torch.Tensor]],
) -> Tuple[float, float]:
    """Return (max|Δ| over rna_proj params, max|Δ| over atac_proj params).

    Both must equal 0.0 exactly for T8-b to pass. max-abs-diff is used in
    place of ``torch.equal`` because torch.equal returns False on matched
    NaN patterns, which could mask genuine numeric drift.
    """
    max_diff_rna = 0.0
    for n, p in rna_proj.named_parameters():
        diff = (p.detach() - snapshot["rna_proj"][n]).abs().max().item()
        max_diff_rna = max(max_diff_rna, float(diff))
    max_diff_atac = 0.0
    for n, p in atac_proj.named_parameters():
        diff = (p.detach() - snapshot["atac_proj"][n]).abs().max().item()
        max_diff_atac = max(max_diff_atac, float(diff))
    return max_diff_rna, max_diff_atac


def _assert_projections_equal_parent(
    rna_proj: nn.Module,
    atac_proj: nn.Module,
    parent_head: nn.Module,
) -> None:
    """Verify at start-of-training that parent-init projections are byte-
    identical to the parent head's projections (sanity check for the
    ``projection_init='parent'`` policy, matching 6.5e's runtime choice).
    """
    for (n_c, p_c), (n_p, p_p) in zip(
        rna_proj.named_parameters(),
        parent_head.rna_proj.named_parameters(),
    ):
        assert n_c == n_p, f"rna_proj name mismatch: {n_c} vs {n_p}"
        if not torch.equal(p_c.detach().cpu(), p_p.detach().cpu()):
            raise RuntimeError(
                f"[6.5f] rna_proj param {n_c!r} does not match parent head."
            )
    for (n_c, p_c), (n_p, p_p) in zip(
        atac_proj.named_parameters(),
        parent_head.atac_proj.named_parameters(),
    ):
        assert n_c == n_p, f"atac_proj name mismatch: {n_c} vs {n_p}"
        if not torch.equal(p_c.detach().cpu(), p_p.detach().cpu()):
            raise RuntimeError(
                f"[6.5f] atac_proj param {n_c!r} does not match parent head."
            )


# --------------------------------------------------------------------- #
# CLI
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
        help="Reuse 6.5e's cached probe batch. If missing, regenerate "
        "under seed=3 and the regenerated indices SHA MUST match "
        "27a906d0… exactly.",
    )
    p.add_argument(
        "--tripwire-json",
        type=Path,
        default=Path("experiments/phase6_5f/tripwires.json"),
    )
    p.add_argument("--probe-n", type=int, default=256)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--projection-dim", type=int, default=128)
    p.add_argument("--stage", default=E1_STAGE)
    p.add_argument(
        "--projection-init",
        choices=["parent", "seed3_fresh"],
        default="parent",
        help="Locked to match 6.5e runtime choice ('parent').",
    )
    p.add_argument(
        "--freeze-projections",
        action="store_true",
        default=True,
        help="6.5f's single variable vs 6.5e. Kept as an explicit flag "
        "so tests can parameterize; disabling it re-runs 6.5e exactly.",
    )
    p.add_argument(
        "--no-freeze-projections",
        dest="freeze_projections",
        action="store_false",
        help="Disable freezing (test-only — do NOT use for the 6.5f run).",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--wandb-project", default="aivc-pretrain")
    p.add_argument("--wandb-name", default="phase-6.5f-disambig-frozen-proj")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args(argv)


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #
def main(argv=None) -> dict:  # noqa: C901 - monolithic by design (auditability)
    args = parse_args(argv)
    tripwires: Dict[str, dict] = {}
    deviations: List[str] = []

    if args.stage != E1_STAGE:
        raise RuntimeError(
            f"[6.5f] --stage must be {E1_STAGE!r} (reused from 6.5e); "
            f"got {args.stage!r}."
        )

    t0 = time.time()
    generator = _set_global_seed(args.seed)
    device = _select_device()
    print(f"[6.5f] device={device} seed={args.seed}", flush=True)

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
            f"[6.5f:T1] parent ckpt SHA mismatch. expected "
            f"{PARENT_EXPECTED_SHA}, got {parent_sha}."
        )

    # ---- Load parent ckpt ----
    print("[6.5f] loading parent ckpt ...", flush=True)
    rna_enc, atac_enc, pretrain_head, parent_cfg = load_full_pretrain_checkpoint(
        args.parent_ckpt, expected_schema_version=SCHEMA_VERSION
    )
    rna_enc.to(device)
    atac_enc.to(device)
    pretrain_head.to(device)

    # Enforce projection_dim matches parent (single-variable contract).
    parent_proj_dim = int(parent_cfg["proj_dim"])
    if args.projection_dim != parent_proj_dim:
        raise RuntimeError(
            f"[6.5f] --projection-dim={args.projection_dim} differs from "
            f"parent proj_dim={parent_proj_dim}. 6.5f is single-variable."
        )

    # ---- Projection init (locked to 6.5e's runtime choice: 'parent') ----
    if args.projection_init == "parent":
        rna_proj, atac_proj, proj_init_path = _build_projections_from_parent(
            pretrain_head
        )
    else:
        rna_proj, atac_proj, proj_init_path = _build_projections_fresh(
            rna_dim=rna_enc.latent_dim,
            atac_dim=atac_enc.attn_dim,
            proj_dim=parent_proj_dim,
            hidden_dim=parent_proj_dim * 2,
        )
    rna_proj.to(device)
    atac_proj.to(device)
    print(f"[6.5f] projection_init_path={proj_init_path}", flush=True)

    # Parent-init byte-identity check (projection_init='parent' contract).
    if proj_init_path == "parent":
        _assert_projections_equal_parent(rna_proj, atac_proj, pretrain_head)

    # ---- T2: probe batch (h5ad SHA + index SHA) ----
    data_sha = _sha256_file(args.data)
    if data_sha != EXPECTED_PBMC_DATA_SHA:
        tripwires["T2_probe_batch"] = {
            "pass": False,
            "reason": "data_sha mismatch",
            "observed_data_sha": data_sha,
            "expected_data_sha": EXPECTED_PBMC_DATA_SHA,
        }
        _write_tripwires(args.tripwire_json, tripwires, deviations)
        raise RuntimeError(
            f"[6.5f:T2] h5ad SHA mismatch. expected {EXPECTED_PBMC_DATA_SHA}, "
            f"got {data_sha}. STOP."
        )

    rna_probe, atac_probe, probe_idx = _build_or_load_probe_batch(
        args.data, args.probe_cache, seed=args.seed, n_cells=args.probe_n
    )
    index_sha = hashlib.sha256(probe_idx.astype(np.int64).tobytes()).hexdigest()
    t2_pass = index_sha == EXPECTED_PROBE_INDEX_SHA
    tripwires["T2_probe_batch"] = {
        "pass": bool(t2_pass),
        "cache_path": str(args.probe_cache),
        "n_cells": int(rna_probe.shape[0]),
        "seed": int(args.seed),
        "rna_shape": list(rna_probe.shape),
        "atac_shape": list(atac_probe.shape),
        "indices_sha256": index_sha,
        "expected_indices_sha256": EXPECTED_PROBE_INDEX_SHA,
        "data_sha256": data_sha,
        "expected_data_sha256": EXPECTED_PBMC_DATA_SHA,
    }
    if not t2_pass:
        _write_tripwires(args.tripwire_json, tripwires, deviations)
        raise RuntimeError(
            f"[6.5f:T2] probe index SHA mismatch. expected "
            f"{EXPECTED_PROBE_INDEX_SHA}, got {index_sha}. STOP."
        )

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
        _write_tripwires(args.tripwire_json, tripwires, deviations)
        raise RuntimeError(
            f"[6.5f:T6] weight mask mismatch. expected {E1_WEIGHT_MASK}, "
            f"got {effective_mask}."
        )

    # =================================================================== #
    # 6.5f SINGLE VARIABLE: freeze projections, rebuild optimizer.
    # =================================================================== #
    if args.freeze_projections:
        _freeze_projections(rna_proj, atac_proj)
        print("[6.5f] rna_proj / atac_proj frozen (requires_grad=False)",
              flush=True)

    trainable = [
        p
        for p in (
            list(rna_enc.parameters())
            + list(atac_enc.parameters())
            + list(rna_proj.parameters())
            + list(atac_proj.parameters())
        )
        if p.requires_grad
    ]
    if not trainable:
        raise RuntimeError("[6.5f] no trainable parameters after freeze.")
    optim = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )

    # ---- T8-a: optimizer exclusion + requires_grad=False (IMMEDIATE) ----
    if args.freeze_projections:
        opt_pass, rg_pass = _assert_optimizer_excludes_projections(
            optim, rna_proj, atac_proj
        )
    else:
        opt_pass, rg_pass = False, False  # freeze disabled: expected False
        deviations.append(
            "freeze_projections=False — 6.5e behavior (diagnostic only)."
        )

    # ---- Snapshot projection state BEFORE first optimizer.step() ----
    proj_snapshot = _snapshot_projection_state(rna_proj, atac_proj)

    # Cross-reference projection weights to 6.5e's epoch-0 weights:
    # since projection_init='parent' and 6.5e also used 'parent', the
    # projection weights at step 0 must be byte-identical to the parent
    # head's projections (already asserted above). Record the projection
    # tensor SHA-256 fingerprint for auditability.
    _h = hashlib.sha256()
    for n in sorted(proj_snapshot["rna_proj"].keys()):
        _h.update(n.encode())
        _h.update(proj_snapshot["rna_proj"][n].detach().cpu().numpy().tobytes())
    rna_proj_init_sha = _h.hexdigest()
    _h = hashlib.sha256()
    for n in sorted(proj_snapshot["atac_proj"].keys()):
        _h.update(n.encode())
        _h.update(proj_snapshot["atac_proj"][n].detach().cpu().numpy().tobytes())
    atac_proj_init_sha = _h.hexdigest()

    # ---- Data ----
    print("[6.5f] loading multiome dataset ...", flush=True)
    dataset = _InMemoryMultiome(args.data)
    if dataset.n_genes != rna_enc.n_genes:
        raise RuntimeError(
            f"[6.5f] RNA n_genes mismatch: ckpt={rna_enc.n_genes} "
            f"data={dataset.n_genes}."
        )
    if dataset.n_peaks != atac_enc.n_peaks:
        raise RuntimeError(
            f"[6.5f] ATAC n_peaks mismatch: ckpt={atac_enc.n_peaks} "
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
                job_type="phase6_5f_frozen_proj_finetune",
                config={
                    "parent_ckpt_sha": parent_sha,
                    "phase6_5e_baseline_ckpt_sha": PHASE6_5E_CKPT_SHA,
                    "seed": int(args.seed),
                    "temperature": float(args.temperature),
                    "batch_size": int(B),
                    "lr": float(args.lr),
                    "weight_decay": float(args.weight_decay),
                    "grad_clip": float(args.grad_clip),
                    "epochs": int(args.epochs),
                    "projection_init_path": proj_init_path,
                    "projections_frozen": bool(args.freeze_projections),
                    "projection_dim": int(parent_proj_dim),
                    "stage": E1_STAGE,
                    "weight_mask": E1_WEIGHT_MASK,
                    "rna_proj_init_sha256": rna_proj_init_sha,
                    "atac_proj_init_sha256": atac_proj_init_sha,
                },
            )
        except Exception as e:  # pragma: no cover
            print(f"[6.5f] wandb unavailable: {e}", flush=True)
            run = None

    # ---- Training loop ----
    print(f"[6.5f] fine-tune: epochs={args.epochs} B={B} "
          f"trainable_params={sum(p.numel() for p in trainable)}",
          flush=True)
    step = 0
    loss_first: float | None = None
    loss_last: float | None = None
    loss_history: List[float] = []
    for ep in range(args.epochs):
        rna_enc.train()
        atac_enc.train()
        # Frozen projections stay in train() for consistency with 6.5e
        # (no dropout/BN in Sequential(Linear-GELU-Linear), so .train()
        # vs .eval() is a no-op here — mirror 6.5e semantics exactly).
        rna_proj.train()
        atac_proj.train()
        for rna_batch, atac_batch in loader:
            rna_batch = rna_batch.to(device, non_blocking=True)
            atac_batch = atac_batch.to(device, non_blocking=True)

            z_rna_lat, _ = rna_enc(rna_batch)
            z_atac_lat = atac_enc(atac_batch)

            z_rna = F.normalize(rna_proj(z_rna_lat), dim=-1)
            z_atac = F.normalize(atac_proj(z_atac_lat), dim=-1)

            total, components = registry.compute(
                stage=E1_STAGE,
                z_rna=z_rna,
                z_atac=z_atac,
                infonce_temperature=float(args.temperature),
            )

            if not torch.isfinite(total).all():
                tripwires["T5_online_nan_abort"] = {
                    "pass": False,
                    "batch": int(step),
                    "loss_value": float(
                        total.item() if total.numel() == 1 else 0.0
                    ),
                }
                _write_tripwires(args.tripwire_json, tripwires, deviations)
                raise RuntimeError(
                    f"[6.5f:T5] NaN/Inf loss at batch {step}. STOP."
                )

            optim.zero_grad(set_to_none=True)
            total.backward()

            grad_nonfinite = any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in trainable
            )
            if grad_nonfinite:
                tripwires["T5_online_nan_abort"] = {
                    "pass": False,
                    "batch": int(step),
                    "reason": "non-finite gradient",
                }
                _write_tripwires(args.tripwire_json, tripwires, deviations)
                raise RuntimeError(
                    f"[6.5f:T5] non-finite gradient at batch {step}. STOP."
                )

            torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.grad_clip)
            optim.step()

            loss_val = float(total.item())
            if loss_first is None:
                loss_first = loss_val
            loss_last = loss_val
            loss_history.append(loss_val)
            if step % 5 == 0 or step == 0:
                print(f"[6.5f] step {step:03d} loss={loss_val:.4f} "
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
            f"[6.5f:T3] probe drift below threshold: "
            f"rna={drift_rna:.2e}, atac={drift_atac:.2e} (< 1e-4). STOP."
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
            "[6.5f:T4] representation collapsed or non-finite. STOP."
        )

    # ---- T7: module-level drift ratios (diagnostic) ----
    def _drift_ratio(pre: torch.Tensor, post_module: nn.Module) -> float:
        post = _param_vector(post_module)
        if pre.numel() == 0:
            return 0.0
        denom = float(torch.linalg.norm(pre).item()) or 1.0
        return float(torch.linalg.norm(post - pre).item() / denom)

    rna_enc_drift = _drift_ratio(rna_pre_vec, rna_enc)
    atac_enc_drift = _drift_ratio(atac_pre_vec, atac_enc)
    rna_proj_drift = _drift_ratio(rna_proj_pre_vec, rna_proj)
    atac_proj_drift = _drift_ratio(atac_proj_pre_vec, atac_proj)
    # Per spec: under freeze, projection drift = 0 by construction, so
    # drift_ratio_*_proj is logged as "inf" (informationless) and only
    # encoder drift is scientifically meaningful.
    tripwires["T7_drift_ratio"] = {
        "pass": True,
        "rna_encoder": rna_enc_drift,
        "atac_encoder": atac_enc_drift,
        "rna_proj": rna_proj_drift,
        "atac_proj": atac_proj_drift,
        "drift_ratio_rna": "inf" if args.freeze_projections else (
            rna_enc_drift / rna_proj_drift if rna_proj_drift > 0 else "inf"
        ),
        "drift_ratio_atac": "inf" if args.freeze_projections else (
            atac_enc_drift / atac_proj_drift if atac_proj_drift > 0 else "inf"
        ),
        "encoder_rna_drift_percent": rna_enc_drift * 100.0,
        "encoder_atac_drift_percent": atac_enc_drift * 100.0,
    }

    # ---- T8-b: max|ΔW|=0 exact on projection params (BEFORE ckpt save) ----
    max_diff_rna, max_diff_atac = _max_abs_projection_diff(
        rna_proj, atac_proj, proj_snapshot
    )
    t8b_pass = (max_diff_rna == 0.0) and (max_diff_atac == 0.0)
    tripwires["t8"] = {
        "optimizer_exclusion_pass": bool(opt_pass),
        "proj_requires_grad_pass": bool(rg_pass),
        "max_diff_rna": max_diff_rna,
        "max_diff_atac": max_diff_atac,
    }
    # Preserve the 6.5e-style boolean key for parity with earlier tripwire logs.
    tripwires["T8_projection_freeze"] = {
        "pass": bool(opt_pass and rg_pass and t8b_pass),
        "projection_freeze_violation": not t8b_pass,
        "optimizer_exclusion_pass": bool(opt_pass),
        "proj_requires_grad_pass": bool(rg_pass),
        "max_diff_rna": max_diff_rna,
        "max_diff_atac": max_diff_atac,
    }
    if args.freeze_projections and not t8b_pass:
        _write_tripwires(args.tripwire_json, tripwires, deviations)
        raise RuntimeError(
            f"[6.5f:T8-b] projection weights drifted under freeze: "
            f"max_diff_rna={max_diff_rna}, max_diff_atac={max_diff_atac}. "
            "STOP. Do NOT save ckpt."
        )

    # ---- Save ckpt ----
    args.out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    config_to_save = dict(parent_cfg)
    config_to_save.update({
        "pretrain_stage": E1_STAGE,
        "parent_ckpt_sha": parent_sha,
        "phase6_5e_baseline_ckpt_sha": PHASE6_5E_CKPT_SHA,
        "data_sha": data_sha,
        "epochs_finetuned": int(args.epochs),
        "batch_size": int(B),
        "temperature": float(args.temperature),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "grad_clip": float(args.grad_clip),
        "seed": int(args.seed),
        "projection_init_path": proj_init_path,
        "projection_init": proj_init_path,
        "projection_dim": int(parent_proj_dim),
        "projections_frozen": bool(args.freeze_projections),
        "weight_mask": E1_WEIGHT_MASK,
        "loss_weights": E1_WEIGHT_MASK,
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
            "pretrain_head": pretrain_head.cpu().state_dict(),
            "rna_encoder_class": "aivc.skills.rna_encoder.SimpleRNAEncoder",
            "atac_encoder_class": "aivc.skills.atac_peak_encoder.PeakLevelATACEncoder",
            "pretrain_head_class": "aivc.training.pretrain_heads.MultiomePretrainHead",
            "config": config_to_save,
        },
        args.out_ckpt,
    )
    out_sha = _sha256_file(args.out_ckpt)
    print(f"[6.5f] ckpt saved: {args.out_ckpt}", flush=True)
    print(f"[6.5f] ckpt_sha256={out_sha}", flush=True)
    print(f"[6.5f] parent_ckpt_sha256={parent_sha}", flush=True)
    print(f"[6.5f] phase6_5e_baseline_ckpt_sha={PHASE6_5E_CKPT_SHA}",
          flush=True)

    if out_sha == parent_sha:
        _write_tripwires(args.tripwire_json, tripwires, deviations)
        raise RuntimeError(
            "[6.5f] output ckpt SHA equals parent — training did not "
            "update weights. STOP."
        )
    if out_sha == PHASE6_5E_CKPT_SHA:
        _write_tripwires(args.tripwire_json, tripwires, deviations)
        raise RuntimeError(
            "[6.5f] output ckpt SHA equals 6.5e baseline — 6.5f did "
            "not produce a distinct artifact. STOP."
        )

    elapsed = time.time() - t0
    tripwires["out_ckpt_sha256"] = {
        "pass": True,
        "observed": out_sha,
        "parent": parent_sha,
        "phase6_5e_baseline": PHASE6_5E_CKPT_SHA,
    }
    tripwires["meta"] = {
        "pass": True,
        "projection_init_path": proj_init_path,
        "projections_frozen": bool(args.freeze_projections),
        "loss_first": loss_first,
        "loss_last": loss_last,
        "loss_decreased": bool(
            loss_first is not None and loss_last is not None
            and loss_last < loss_first
        ),
        "loss_curve_len": len(loss_history),
        "batch_size": int(B),
        "wall_clock_seconds": float(elapsed),
        "trainable_param_count": int(sum(p.numel() for p in trainable)),
        "rna_proj_init_sha256": rna_proj_init_sha,
        "atac_proj_init_sha256": atac_proj_init_sha,
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


if __name__ == "__main__":
    main()
