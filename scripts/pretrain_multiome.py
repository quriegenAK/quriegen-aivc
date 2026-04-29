"""
scripts/pretrain_multiome.py — standalone Phase 5 multiome pretraining.

Entrypoint for training the RNA + PeakLevelATAC encoders on paired
observational multiome data. Runtime grad guard enforces Phase 3's
gradient-isolation invariant.

DOES NOT touch train_week3.py, fusion.py, or NeumannPropagation.

W&B project is "aivc-pretrain" — intentionally separate from the main
training project so pretraining runs do not pollute the main dashboard.

Mock-data fallback: if the harmonize_peaks.py artifact is missing and no
``--multiome_h5ad`` path is provided, this script synthesizes a 1k-cell
mock Multiome batch in memory. See FAILURE HANDLING in the Phase 5 PR.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aivc.skills.atac_peak_encoder import PeakLevelATACEncoder
from aivc.skills.rna_encoder import SimpleRNAEncoder
from aivc.training.loss_registry import LossRegistry
from aivc.training.pretrain_heads import MultiomePretrainHead
from aivc.training.pretrain_losses import register_pretrain_terms


# Pretrained checkpoint schema version. Phase 6's --pretrained_ckpt
# loader MUST fail loudly on mismatch. See aivc/training/PRETRAIN_CKPT_SCHEMA.md.
PRETRAIN_CKPT_SCHEMA_VERSION = 1


class _ATACDecoder(nn.Module):
    """Small linear decoder from ATAC latent back to peak space."""

    def __init__(self, latent_dim: int, n_peaks: int):
        super().__init__()
        self.decoder = nn.Linear(latent_dim, n_peaks)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


# --------------------------------------------------------------------- #
# Mock Multiome batch generator.
# --------------------------------------------------------------------- #
def _mock_multiome_batch(n_cells: int, n_genes: int, n_peaks: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    rna = rng.poisson(lam=0.5, size=(n_cells, n_genes)).astype(np.float32)
    atac = (rng.random((n_cells, n_peaks)) < 0.02).astype(np.float32) \
        * rng.integers(1, 5, size=(n_cells, n_peaks)).astype(np.float32)
    return (
        torch.from_numpy(rna),
        torch.from_numpy(atac),
    )


def _has_neumann_in_graph(model: nn.Module) -> bool:
    """True if any submodule is NeumannPropagation."""
    # Delayed import so merely importing this script does not force
    # NeumannPropagation into the module cache.
    from aivc.skills.neumann_propagation import NeumannPropagation
    return any(isinstance(m, NeumannPropagation) for m in model.modules())


def _setup_wandb(project: str, enabled: bool, config_extra: dict | None = None):
    if not enabled:
        return None
    try:
        import wandb
    except ImportError:
        print("[warn] wandb not installed; continuing without logging.")
        return None
    run = wandb.init(
        project=project,
        job_type="pretrain_multiome",
        config=config_extra or {},
    )
    return run


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """SHA-256 of an on-disk artifact; logged into W&B config for provenance."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_rev() -> str:
    """Current repo HEAD commit (short form) for audit trail. Returns
    ``'unknown'`` if git is unavailable or this is not a repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("ascii").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--multiome_h5ad", default=None, help="Path to multiome .h5ad (optional)")
    # --peak_set_path is the Phase 6.7b canonical flag name; --peak_set kept as
    # a back-compat alias so existing automation does not break.
    p.add_argument(
        "--peak_set_path",
        "--peak_set",
        dest="peak_set",
        default=None,
        help="Path to harmonized peak set",
    )
    p.add_argument("--n_cells", type=int, default=1000)
    p.add_argument("--n_genes", type=int, default=2000)
    p.add_argument("--n_peaks", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rna_latent", type=int, default=128)
    p.add_argument("--atac_latent", type=int, default=64)
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--checkpoint_dir", default="checkpoints/pretrain")
    p.add_argument("--wandb_project", default="aivc-pretrain")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    # PR #41: external YAML config + epoch-based loop.
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config; CLI args override config values")
    p.add_argument("--epochs", type=int, default=None,
                   help="Number of outer epochs (each runs --steps inner iterations)")
    args = p.parse_args(argv)

    # PR #41: load config and override defaults (CLI wins on conflict).
    # Sentinel-based detection: compare against argparse default and assume
    # any non-default value was set by the user. Crude but matches spec §9.
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        if args.lr == 1e-3:  # default; not user-set
            args.lr = float(cfg["optimizer"]["lr"])
        if args.batch_size == 64:
            args.batch_size = int(cfg["training"]["batch_size"])
        if args.epochs is None:
            args.epochs = int(cfg["training"]["epochs"])
        if args.seed == 0:
            args.seed = int(cfg["training"]["seed"])
        weight_decay = float(cfg["optimizer"].get("weight_decay", 0.01))
        warmup_steps = int(cfg["schedule"]["warmup_steps"])
        ckpt_every = int(cfg["checkpoint"].get("every_n_epochs", 5))
    else:
        weight_decay = 0.01
        warmup_steps = 500
        ckpt_every = 5

    # Epochs supersede --steps when set; back-compat default is single epoch.
    if args.epochs is None:
        args.epochs = 1
    print(f"[config] lr={args.lr} batch_size={args.batch_size} "
          f"epochs={args.epochs} steps_per_epoch={args.steps} "
          f"weight_decay={weight_decay} warmup_steps={warmup_steps} "
          f"ckpt_every_n_epochs={ckpt_every}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[device] {device}")

    # ---- Data ---- #
    if args.multiome_h5ad and args.peak_set:
        from aivc.data.multiome_loader import MultiomeLoader
        ds = MultiomeLoader(
            h5ad_path=args.multiome_h5ad,
            peak_set_path=args.peak_set,
            schema="obsm_atac",
        )
        # Materialize as tensors (demo path; real runs would use a DataLoader).
        rna_all = torch.stack([torch.from_numpy(ds[i]["rna"]) for i in range(len(ds))])
        atac_all = torch.stack([torch.from_numpy(ds[i]["atac_peaks"]) for i in range(len(ds))])
        n_genes = rna_all.shape[1]
        n_peaks = atac_all.shape[1]
        print(f"[data] real multiome: n_cells={rna_all.shape[0]} n_genes={n_genes} n_peaks={n_peaks}")
    else:
        print(
            "[data] peak_set artifact not provided; falling back to mock "
            "Multiome batch (Phase 5 PR FAILURE HANDLING)."
        )
        rna_all, atac_all = _mock_multiome_batch(
            args.n_cells, args.n_genes, args.n_peaks, seed=args.seed
        )
        n_genes, n_peaks = args.n_genes, args.n_peaks

    # ---- Model ---- #
    rna_enc = SimpleRNAEncoder(n_genes=n_genes, latent_dim=args.rna_latent).to(device)
    atac_enc = PeakLevelATACEncoder(n_peaks=n_peaks, attn_dim=args.atac_latent).to(device)
    atac_decoder = _ATACDecoder(latent_dim=args.atac_latent, n_peaks=n_peaks).to(device)
    head = MultiomePretrainHead(
        rna_dim=args.rna_latent,
        atac_dim=args.atac_latent,
        proj_dim=args.proj_dim,
        n_genes=n_genes,
    ).to(device)

    params = (
        list(rna_enc.parameters())
        + list(atac_enc.parameters())
        + list(atac_decoder.parameters())
        + list(head.parameters())
    )
    # PR #41: AdamW with weight decay + warmup-cosine LR schedule (spec §9).
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=weight_decay)

    total_steps = max(1, args.epochs * args.steps)
    from torch.optim.lr_scheduler import (  # noqa: E402
        CosineAnnealingLR,
        LinearLR,
        SequentialLR,
    )
    warmup = LinearLR(
        optim, start_factor=1e-3, end_factor=1.0,
        total_iters=max(1, warmup_steps),
    )
    cosine = CosineAnnealingLR(optim, T_max=max(1, total_steps - warmup_steps))
    scheduler = SequentialLR(
        optim, schedulers=[warmup, cosine],
        milestones=[max(1, warmup_steps)],
    )

    # ---- Setup grad-guard sentinel ---- #
    # If NeumannPropagation is NOT in this entrypoint's graph, we must
    # assert it explicitly — otherwise "W.grad is None" after backward
    # would pass for the wrong reason (the module simply does not exist).
    composite = nn.ModuleList([rna_enc, atac_enc, atac_decoder, head])
    if _has_neumann_in_graph(composite):
        raise RuntimeError(
            "Phase 5 invariant violated: NeumannPropagation unexpectedly "
            "present in pretrain model graph. Pretrain must be structurally "
            "isolated from the causal head."
        )
    print("[guard] NeumannPropagation absent from pretrain graph (expected).")

    # ---- Loss registry ---- #
    registry = LossRegistry()
    register_pretrain_terms(registry)

    # ---- W&B ---- #
    # Audit trail for Phase 6.5's linear-probe tripwire: log every
    # provenance field the downstream gate needs to assert the pretrain
    # checkpoint was produced on the real-data path — peak_set_sha256,
    # runtime shapes, device, seed, git rev. ``ckpt_sha256`` is logged
    # post-save below, once the file exists on disk.
    wandb_config: dict = {
        "device": str(device),
        "seed": int(args.seed),
        "git_rev": _git_rev(),
        "n_genes_runtime": int(n_genes),
        "n_peaks_runtime": int(n_peaks),
    }
    if args.peak_set:
        try:
            wandb_config["peak_set_sha256"] = _sha256_file(Path(args.peak_set))
            wandb_config["peak_set_path"] = str(Path(args.peak_set).resolve())
        except FileNotFoundError:
            wandb_config["peak_set_sha256"] = None
    else:
        wandb_config["peak_set_sha256"] = None
    run = _setup_wandb(args.wandb_project, enabled=not args.no_wandb, config_extra=wandb_config)

    # ---- Training loop ---- #
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    loss_ma = None
    global_step = 0
    # PR #41: outer epoch loop * inner step loop = total_steps iterations.
    # back-compat: --epochs unset -> default 1 epoch -> behavior identical to
    # pre-PR (single pass of --steps iterations).
    for epoch in range(args.epochs):
        for step in range(args.steps):
            idx = torch.randint(0, rna_all.shape[0], (args.batch_size,))
            rna_batch = rna_all[idx].to(device)
            atac_batch = atac_all[idx].to(device)

            # Random masks (15%) for reconstruction heads.
            rna_mask = (torch.rand_like(rna_batch) < 0.15).float()
            atac_mask = (torch.rand_like(atac_batch) < 0.15).float()

            rna_latent, rna_recon = rna_enc(rna_batch)
            atac_latent = atac_enc(atac_batch)
            atac_recon = atac_decoder(atac_latent)

            batch_dict = {
                "rna": rna_batch,
                "atac_peaks": atac_batch,
            }
            outputs = head(batch_dict, rna_latent, atac_latent)

            # Simple gene target: the RNA batch itself (peak_to_gene predicts
            # expression proxy from peaks). Scale-matched to rna_recon loss.
            loss_batch = {
                "rna_recon": rna_recon,
                "rna_target": rna_batch,
                "rna_mask": rna_mask,
                "atac_recon": atac_recon,
                "atac_target": atac_batch,
                "atac_mask": atac_mask,
                "z_rna": outputs["z_rna"],
                "z_atac": outputs["z_atac"],
                "gene_pred": outputs["gene_pred"],
                "gene_target": rna_batch,
                "infonce_temperature": 0.1,
            }

            total, components = registry.compute(stage="pretrain", **loss_batch)

            optim.zero_grad(set_to_none=True)
            total.backward()

            # Runtime grad guard: no NeumannPropagation in this graph, but we
            # run the traversal anyway as defense-in-depth so a future edit
            # that accidentally adds a Neumann instance will fire this check.
            from aivc.skills.neumann_propagation import NeumannPropagation
            for m in composite.modules():
                if isinstance(m, NeumannPropagation):
                    assert m.W.grad is None, (
                        "Phase 5 grad guard fired: NeumannPropagation.W "
                        "received gradient from pretrain backward."
                    )

            optim.step()
            scheduler.step()  # PR #41: warmup-cosine LR step
            global_step += 1

            loss_val = total.item()
            loss_ma = loss_val if loss_ma is None else 0.9 * loss_ma + 0.1 * loss_val
            if step % 5 == 0 or step == args.steps - 1:
                cur_lr = optim.param_groups[0]["lr"]
                print(f"[epoch {epoch:03d} step {step:03d} g{global_step:05d}] "
                      f"lr={cur_lr:.2e} total={loss_val:.4f} ma={loss_ma:.4f} "
                      f"components={components}")
            if run is not None:
                run.log({
                    "total": loss_val, "loss_ma": loss_ma, **components,
                    "epoch": epoch, "step": step, "global_step": global_step,
                    "lr": optim.param_groups[0]["lr"],
                })

    # ---- Checkpoint ---- #
    # Phase 6.7b fix: stamp actual runtime shapes into config so the
    # ckpt_loader can reconstruct SimpleRNAEncoder correctly. Without this,
    # argparse defaults (e.g. --n_genes 2000 from the mock fallback path)
    # would stay in `config` and load_pretrained_simple_rna_encoder would
    # fail with size-mismatch when instantiating the encoder. Also records
    # the peak_set SHA-256 into the checkpoint for provenance (mirrors the
    # W&B config log line added at the start of main()).
    config_to_save = dict(vars(args))
    config_to_save["n_genes"] = int(n_genes)
    config_to_save["n_peaks"] = int(n_peaks)
    config_to_save["hidden_dim"] = int(rna_enc.hidden_dim)
    config_to_save["latent_dim"] = int(rna_enc.latent_dim)
    config_to_save["device"] = str(device)
    if args.peak_set:
        try:
            config_to_save["peak_set_sha256"] = _sha256_file(Path(args.peak_set))
        except FileNotFoundError:
            config_to_save["peak_set_sha256"] = None
    else:
        config_to_save["peak_set_sha256"] = None

    ckpt_path = ckpt_dir / "pretrain_encoders.pt"
    torch.save(
        {
            "schema_version": PRETRAIN_CKPT_SCHEMA_VERSION,
            "rna_encoder": rna_enc.state_dict(),
            "atac_encoder": atac_enc.state_dict(),
            "pretrain_head": head.state_dict(),
            "rna_encoder_class": "aivc.skills.rna_encoder.SimpleRNAEncoder",
            "atac_encoder_class": "aivc.skills.atac_peak_encoder.PeakLevelATACEncoder",
            "pretrain_head_class": "aivc.training.pretrain_heads.MultiomePretrainHead",
            "config": config_to_save,
        },
        ckpt_path,
    )
    print(f"[ckpt] saved -> {ckpt_path}")

    # Post-save provenance: hash the checkpoint we just wrote and log
    # it to the W&B run config. Phase 6.5's linear-probe gate will
    # assert this value matches the ckpt it loads, so the two SHAs
    # line up in the audit trail.
    ckpt_sha256 = _sha256_file(ckpt_path)
    print(f"[ckpt] sha256={ckpt_sha256}")
    if run is not None:
        try:
            run.config.update({"ckpt_sha256": ckpt_sha256}, allow_val_change=True)
        except Exception as e:  # pragma: no cover - W&B network issue fallback
            print(f"[warn] failed to log ckpt_sha256 to W&B: {e}")
        run.finish()


if __name__ == "__main__":
    main()
