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
from aivc.skills.protein_encoder import ProteinEncoder  # logical PR #42
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


class _ProteinDecoder(nn.Module):
    """Small linear decoder from Protein latent back to ADT count space.

    PR #42 (logical): mirrors _ATACDecoder for symmetric trimodal recon
    when --arm is set. Output dim is n_proteins (210 for TotalSeq-A panel).
    """

    def __init__(self, latent_dim: int, n_proteins: int):
        super().__init__()
        self.decoder = nn.Linear(latent_dim, n_proteins)

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


# ---------------------------------------------------------------------------
# PR #44 (logical): resume mechanism — approximate granularity.
#
# Captures: optimizer state + scheduler state + epoch + global_step.
# Does NOT capture: RNG state. Decision per spec — bit-exact resume costs
# more code than it saves; sample-order divergence on resume is acceptable
# for a 24-h GPU pretrain run.
# ---------------------------------------------------------------------------

def _build_resume_state(optimizer, scheduler, epoch: int, global_step: int) -> dict:
    """Capture optimizer + scheduler + position counters for resume."""
    from datetime import datetime, timezone
    return {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }


# Critical config fields that MUST match between saved ckpt and current run
# for resume to be safe. Mismatch on any of these means encoder/loss/data
# shapes have changed; load_state_dict would fail anyway, but better to
# fail fast with a clear message.
_RESUME_CRITICAL_FIELDS = (
    "arm",
    "n_genes",
    "n_peaks",
    "n_proteins",
    "rna_latent",
    "atac_latent",
    "proj_dim",
    "protein_latent",
    "n_lysis_categories",
    "lysis_cov_dim",
)


def _validate_resume_config(saved_config: dict, current_config: dict) -> None:
    """Raise ValueError if any critical config field differs.

    Reports ALL mismatches in a single error (not just the first) so a
    debugging operator sees the full delta in one read.
    """
    mismatches = []
    for key in _RESUME_CRITICAL_FIELDS:
        sv = saved_config.get(key)
        cv = current_config.get(key)
        # None == None is fine (field genuinely absent on both sides — e.g.
        # a pre-PR-#43 single-arm ckpt has no n_lysis_categories key)
        if sv != cv:
            mismatches.append(f"  {key}: saved={sv!r}, current={cv!r}")
    if mismatches:
        raise ValueError(
            "Resume config mismatch — critical fields differ:\n"
            + "\n".join(mismatches)
            + "\n\nResume cannot proceed. Either:\n"
            + "  (a) re-invoke with matching args (same --arm, same dims), or\n"
            + "  (b) start fresh without --resume."
        )


def _apply_resume_state(ckpt: dict, optimizer, scheduler) -> tuple[int, int]:
    """Restore optimizer + scheduler from ckpt['resume_state'].

    Returns
    -------
    (start_epoch, start_global_step) : tuple[int, int]
        Caller resumes loop at start_epoch (the SAVED epoch is the LAST
        completed one, so resume starts at saved_epoch + 1). For pre-PR-#44
        ckpts (no ``resume_state`` key) returns (0, 0) → fresh start.
    """
    if "resume_state" not in ckpt:
        return 0, 0
    rs = ckpt["resume_state"]
    optimizer.load_state_dict(rs["optimizer"])
    scheduler.load_state_dict(rs["scheduler"])
    return int(rs["epoch"]) + 1, int(rs["global_step"])


# PR #48: extracted from end-of-training save block so the same payload
# structure powers periodic saves. Each ckpt is independently resume-able
# (carries the full resume_state dict). Helper does NOT touch W&B (caller
# decides whether/when to log sha + finish run).
def _save_checkpoint(
    path,
    *,
    schema_version: int,
    rna_encoder,
    atac_encoder,
    pretrain_head,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    config: dict,
    protein_encoder=None,
    protein_decoder=None,
    rna_proj=None,
    atac_proj=None,
    protein_proj=None,
) -> str:
    """Save a complete checkpoint at `path`. Returns sha256 of the file."""
    ckpt_payload = {
        "schema_version": schema_version,
        "rna_encoder": rna_encoder.state_dict(),
        "atac_encoder": atac_encoder.state_dict(),
        "pretrain_head": pretrain_head.state_dict(),
        "rna_encoder_class": "aivc.skills.rna_encoder.SimpleRNAEncoder",
        "atac_encoder_class": "aivc.skills.atac_peak_encoder.PeakLevelATACEncoder",
        "pretrain_head_class": "aivc.training.pretrain_heads.MultiomePretrainHead",
        "config": config,
    }
    if protein_encoder is not None:
        ckpt_payload["protein_encoder"] = protein_encoder.state_dict()
        ckpt_payload["protein_encoder_class"] = "aivc.skills.protein_encoder.ProteinEncoder"
    if protein_decoder is not None:
        ckpt_payload["protein_decoder"] = protein_decoder.state_dict()
    if rna_proj is not None:
        ckpt_payload["rna_proj"] = rna_proj.state_dict()
    if atac_proj is not None:
        ckpt_payload["atac_proj"] = atac_proj.state_dict()
    if protein_proj is not None:
        ckpt_payload["protein_proj"] = protein_proj.state_dict()
    ckpt_payload["resume_state"] = _build_resume_state(
        optimizer=optimizer, scheduler=scheduler,
        epoch=epoch, global_step=global_step,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt_payload, path)
    sha = _sha256_file(Path(path))
    print(f"[ckpt] saved -> {path}")
    print(f"[ckpt] sha256={sha}")
    return sha


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
    # PR #42 (logical): DOGMA tri-modal arms. PR #43 (logical) extends with "joint".
    p.add_argument("--arm", type=str, default=None,
                   choices=["lll", "dig", "joint"],
                   help="DOGMA arm to load via make_dogma_<arm>_union; "
                        "'joint' enables the lysis_protocol categorical "
                        "covariate (LLL=0, DIG=1) at the encoder level. "
                        "Supersedes --multiome_h5ad/--peak_set_path when set, "
                        "and switches the loss from MultiomePretrainHead "
                        "(bimodal) to _dogma_pretrain_loss (tri-modal).")
    # PR #45 (logical): default 128 matches rna_latent (config default).
    # ProteinEncoder.cross_attn uses a single embed_dim for q/k/v (no separate
    # kdim/vdim), so rna_emb passed at forward time must have last-dim ==
    # embed_dim. The encoder's own docstring states "embed_dim MUST be 128".
    # The prior default (64) silently broke the trimodal forward when used
    # with the canonical config (rna_latent=128); surfaced by the PR #45
    # real-data kill-and-resume integration test.
    p.add_argument("--protein_latent", type=int, default=128,
                   help="ProteinEncoder output latent dim (only used with --arm). "
                        "MUST equal rna_latent — cross-attn shares embed_dim.")
    p.add_argument("--lysis_cov_dim", type=int, default=8,
                   help="Lysis covariate embedding dim (only used with --arm joint)")
    # PR #44 (logical): resume.
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a previous checkpoint to resume from. "
                        "Encoder weights + optimizer + scheduler + epoch are "
                        "restored; config compatibility is validated at load "
                        "time (mismatch raises ValueError naming offending fields).")
    # PR #45 (logical): clean termination after N steps. Used by the
    # pre-flight kill-and-resume cycle test (deterministic, CI-friendly
    # alternative to subprocess+SIGTERM) and as a compute-budget cap.
    p.add_argument("--max-steps", "--max_steps", dest="max_steps",
                   type=int, default=None,
                   help="If set, terminate cleanly after global_step reaches "
                        "N. Existing save block fires on exit, producing a "
                        "resume-able checkpoint. Useful for compute-budget "
                        "caps and the PR #45 pre-flight kill-and-resume cycle.")
    # PR #48: periodic checkpoint cadence override. Falls back to YAML
    # config's checkpoint.every_n_epochs (default 5). 0 disables periodic
    # saves entirely; only end-of-training save fires.
    p.add_argument("--every_n_epochs", "--every-n-epochs",
                   dest="every_n_epochs", type=int, default=None,
                   help="Save a periodic checkpoint every N completed epochs "
                        "(overrides config's checkpoint.every_n_epochs). Set "
                        "to 0 to disable periodic saves.")
    args = p.parse_args(argv)
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError(f"--max-steps must be positive, got {args.max_steps}")

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
        # PR #54c: optional supcon config block. None == disabled.
        supcon_cfg = cfg.get("supcon") if isinstance(cfg.get("supcon"), dict) else None
        if supcon_cfg is not None and not supcon_cfg.get("enabled", False):
            supcon_cfg = None
    else:
        weight_decay = 0.01
        warmup_steps = 500
        ckpt_every = 5
        supcon_cfg = None

    # PR #48: --every_n_epochs CLI override wins over the config value.
    if args.every_n_epochs is not None:
        ckpt_every = int(args.every_n_epochs)

    # Epochs supersede --steps when set; back-compat default is single epoch.
    if args.epochs is None:
        args.epochs = 1
    print(f"[config] lr={args.lr} batch_size={args.batch_size} "
          f"epochs={args.epochs} steps_per_epoch={args.steps} "
          f"weight_decay={weight_decay} warmup_steps={warmup_steps} "
          f"ckpt_every_n_epochs={ckpt_every}")
    if supcon_cfg is not None:
        print(f"[supcon] enabled: lambda_supcon={supcon_cfg['lambda_supcon']} "
              f"lambda_vicreg={supcon_cfg['lambda_vicreg']} "
              f"temperature={supcon_cfg['temperature']} "
              f"masked_classes={supcon_cfg.get('masked_classes', [])} "
              f"min_confidence={supcon_cfg.get('min_confidence', 0.0)}")
        print(f"[supcon] manifest: {supcon_cfg['class_index_manifest']}")
    else:
        print("[supcon] disabled")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[device] {device}")

    # ---- Data ---- #
    # PR #42 (logical): tri-modal DOGMA path via --arm.
    # PR #43 (logical): --arm joint adds lysis covariate threading.
    # PR #54c: --supcon kwargs threaded to loader factories.
    prot_all = None       # populated only on the trimodal branch
    lysis_all = None      # populated only on the joint branch
    cell_type_idx_all = None         # PR #54c — populated when supcon_cfg is not None
    supcon_eligible_all = None       # PR #54c — populated when supcon_cfg is not None
    n_classes = None                 # PR #54c
    manifest_fingerprint = None      # PR #54c — for ckpt provenance
    n_proteins = None
    n_lysis_categories = 0

    # PR #54c: build a single supcon kwargs dict reused across factory calls.
    supcon_loader_kwargs = {}
    if supcon_cfg is not None:
        supcon_loader_kwargs = dict(
            use_labeled=True,
            labels_obs_col=supcon_cfg["labels_obs_col"],
            confidence_obs_col=supcon_cfg.get("confidence_obs_col"),
            masked_classes=supcon_cfg.get("masked_classes"),
            min_confidence=float(supcon_cfg.get("min_confidence", 0.0)),
            class_index_manifest=supcon_cfg["class_index_manifest"],
        )

    if args.arm in ("lll", "dig"):
        from aivc.data.multiome_loader import MultiomeLoader
        factory = (MultiomeLoader.make_dogma_lll_union
                   if args.arm == "lll" else MultiomeLoader.make_dogma_dig_union)
        ds = factory(**supcon_loader_kwargs)
        n_genes = ds.n_genes
        n_peaks = ds.n_peaks
        n_proteins = ds.n_proteins
        rna_all = torch.stack([torch.from_numpy(ds[i]["rna"]) for i in range(len(ds))])
        atac_all = torch.stack([torch.from_numpy(ds[i]["atac_peaks"]) for i in range(len(ds))])
        prot_all = torch.stack([torch.from_numpy(ds[i]["protein"]) for i in range(len(ds))])
        if supcon_cfg is not None:
            cell_type_idx_all = torch.tensor(
                [ds[i]["cell_type_idx"] for i in range(len(ds))], dtype=torch.long
            )
            supcon_eligible_all = torch.tensor(
                [ds[i]["supcon_eligible"] for i in range(len(ds))], dtype=torch.bool
            )
            n_classes = ds.n_classes
            manifest_fingerprint = ds.manifest_fingerprint
        print(f"[data] DOGMA {args.arm} union: n_cells={rna_all.shape[0]} "
              f"n_genes={n_genes} n_peaks={n_peaks} n_proteins={n_proteins}")
        if supcon_cfg is not None:
            print(f"[data] supcon: n_classes={n_classes} "
                  f"n_eligible={int(supcon_eligible_all.sum())} "
                  f"manifest_fingerprint={manifest_fingerprint}")
    elif args.arm == "joint":
        from aivc.data.multiome_loader import MultiomeLoader
        ds = MultiomeLoader.make_dogma_joint_union(**supcon_loader_kwargs)
        n_genes = ds.n_genes
        n_peaks = ds.n_peaks
        n_proteins = ds.n_proteins
        n_lysis_categories = 2  # LLL + DIG
        rna_all = torch.stack([torch.from_numpy(ds[i]["rna"]) for i in range(len(ds))])
        atac_all = torch.stack([torch.from_numpy(ds[i]["atac_peaks"]) for i in range(len(ds))])
        prot_all = torch.stack([torch.from_numpy(ds[i]["protein"]) for i in range(len(ds))])
        lysis_all = torch.tensor(
            [ds[i]["lysis_idx"] for i in range(len(ds))], dtype=torch.long
        )
        if supcon_cfg is not None:
            cell_type_idx_all = torch.tensor(
                [ds[i]["cell_type_idx"] for i in range(len(ds))], dtype=torch.long
            )
            supcon_eligible_all = torch.tensor(
                [ds[i]["supcon_eligible"] for i in range(len(ds))], dtype=torch.bool
            )
            # Both arms must agree on n_classes (they share the manifest).
            n_classes = ds.lll.n_classes
            manifest_fingerprint = ds.lll.manifest_fingerprint
            assert ds.dig.manifest_fingerprint == manifest_fingerprint, (
                "DIG and LLL arms have divergent class_index_manifest fingerprints"
            )
        print(f"[data] DOGMA joint union: n_cells={rna_all.shape[0]} "
              f"(LLL={ds.n_lll}, DIG={ds.n_dig}) n_genes={n_genes} "
              f"n_peaks={n_peaks} n_proteins={n_proteins} "
              f"n_lysis_categories={n_lysis_categories}")
        if supcon_cfg is not None:
            print(f"[data] supcon: n_classes={n_classes} "
                  f"n_eligible={int(supcon_eligible_all.sum())} / "
                  f"{rna_all.shape[0]} "
                  f"manifest_fingerprint={manifest_fingerprint}")
    elif args.multiome_h5ad and args.peak_set:
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
    # PR #43 (logical): pass n_lysis_categories to all three encoders so
    # they allocate their lysis_emb when the joint arm is active. Default
    # 0 keeps single-arm + bimodal back-compat untouched.
    rna_enc = SimpleRNAEncoder(
        n_genes=n_genes,
        latent_dim=args.rna_latent,
        n_lysis_categories=n_lysis_categories,
        lysis_cov_dim=args.lysis_cov_dim,
    ).to(device)
    atac_enc = PeakLevelATACEncoder(
        n_peaks=n_peaks,
        attn_dim=args.atac_latent,
        n_lysis_categories=n_lysis_categories,
        lysis_cov_dim=args.lysis_cov_dim,
    ).to(device)
    atac_decoder = _ATACDecoder(latent_dim=args.atac_latent, n_peaks=n_peaks).to(device)
    head = MultiomePretrainHead(
        rna_dim=args.rna_latent,
        atac_dim=args.atac_latent,
        proj_dim=args.proj_dim,
        n_genes=n_genes,
    ).to(device)

    # PR #42 (logical): tri-modal arm — ProteinEncoder + decoder + per-modality
    # projections to a common proj_dim for the 3-way InfoNCE in
    # _dogma_pretrain_loss. The bimodal MultiomePretrainHead is unused on this
    # branch (left in `head` for parameter symmetry / unchanged ckpt slot).
    if args.arm in ("lll", "dig", "joint"):
        protein_enc = ProteinEncoder(
            n_proteins=n_proteins,
            embed_dim=args.protein_latent,
            n_lysis_categories=n_lysis_categories,
            lysis_cov_dim=args.lysis_cov_dim,
        ).to(device)
        protein_decoder = _ProteinDecoder(
            latent_dim=args.protein_latent, n_proteins=n_proteins
        ).to(device)
        rna_proj = nn.Linear(args.rna_latent, args.proj_dim).to(device)
        atac_proj = nn.Linear(args.atac_latent, args.proj_dim).to(device)
        protein_proj = nn.Linear(args.protein_latent, args.proj_dim).to(device)
    else:
        protein_enc = None
        protein_decoder = None
        rna_proj = atac_proj = protein_proj = None

    params = (
        list(rna_enc.parameters())
        + list(atac_enc.parameters())
        + list(atac_decoder.parameters())
        + list(head.parameters())
    )
    if args.arm in ("lll", "dig", "joint"):
        params += (
            list(protein_enc.parameters())
            + list(protein_decoder.parameters())
            + list(rna_proj.parameters())
            + list(atac_proj.parameters())
            + list(protein_proj.parameters())
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

    # PR #44 (logical): optional resume from prior checkpoint. Restores
    # encoder weights + optimizer + scheduler + position. Validates that
    # critical config fields (arm, dims, covariate cardinality) match
    # before loading; raises ValueError with a multi-field diagnostic
    # if they don't.
    start_epoch = 0
    start_global_step = 0
    if args.resume:
        print(f"[resume] loading from {args.resume}")
        # Routes through aivc.training.ckpt_loader to keep the
        # test_no_bare_torch_load.py CI invariant intact (PR #44 added
        # load_pretrain_ckpt_raw specifically for the resume use case).
        from aivc.training.ckpt_loader import load_pretrain_ckpt_raw
        ckpt = load_pretrain_ckpt_raw(args.resume, map_location=str(device))
        # Mirror the same fields the save block stamps so validation
        # compares apples-to-apples. protein_latent is always present
        # in saved configs (from dict(vars(args))) regardless of arm,
        # so don't None-gate it here either.
        current_config = {
            "arm": args.arm,
            "n_genes": n_genes,
            "n_peaks": n_peaks,
            "n_proteins": (
                n_proteins if args.arm in ("lll", "dig", "joint") else None
            ),
            "rna_latent": args.rna_latent,
            "atac_latent": args.atac_latent,
            "proj_dim": args.proj_dim,
            "protein_latent": args.protein_latent,
            "n_lysis_categories": n_lysis_categories,
            "lysis_cov_dim": args.lysis_cov_dim,
        }
        _validate_resume_config(ckpt.get("config", {}), current_config)

        # Restore encoder weights (always present in v1 ckpts)
        rna_enc.load_state_dict(ckpt["rna_encoder"])
        atac_enc.load_state_dict(ckpt["atac_encoder"])
        if "pretrain_head" in ckpt:
            head.load_state_dict(ckpt["pretrain_head"])
        # Trimodal optional weights (only present for --arm in lll/dig/joint)
        if "protein_encoder" in ckpt and protein_enc is not None:
            protein_enc.load_state_dict(ckpt["protein_encoder"])
        if "protein_decoder" in ckpt and protein_decoder is not None:
            protein_decoder.load_state_dict(ckpt["protein_decoder"])
        if "rna_proj" in ckpt and rna_proj is not None:
            rna_proj.load_state_dict(ckpt["rna_proj"])
        if "atac_proj" in ckpt and atac_proj is not None:
            atac_proj.load_state_dict(ckpt["atac_proj"])
        if "protein_proj" in ckpt and protein_proj is not None:
            protein_proj.load_state_dict(ckpt["protein_proj"])

        # Restore optimizer + scheduler + position counters
        start_epoch, start_global_step = _apply_resume_state(ckpt, optim, scheduler)
        print(f"[resume] start_epoch={start_epoch}, "
              f"start_global_step={start_global_step}")

    # ---- Setup grad-guard sentinel ---- #
    # If NeumannPropagation is NOT in this entrypoint's graph, we must
    # assert it explicitly — otherwise "W.grad is None" after backward
    # would pass for the wrong reason (the module simply does not exist).
    composite_modules = [rna_enc, atac_enc, atac_decoder, head]
    if args.arm in ("lll", "dig", "joint"):
        composite_modules += [
            protein_enc, protein_decoder, rna_proj, atac_proj, protein_proj,
        ]
    composite = nn.ModuleList(composite_modules)
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
    # PR #44 (logical): both counters seed from --resume when provided.
    global_step = start_global_step
    # PR #41: outer epoch loop * inner step loop = total_steps iterations.
    # back-compat: --epochs unset -> default 1 epoch -> behavior identical to
    # pre-PR (single pass of --steps iterations).
    # PR #45 (logical): hit_max_steps sentinel propagates a clean break from
    # the inner loop out through the outer loop so the end-of-training save
    # block below still runs and produces a resume-able checkpoint at the
    # kill point. last_completed_epoch tracks how many epochs ran to
    # completion — this is what gets stamped into resume_state, not
    # args.epochs - 1 (the argparse ceiling), so a mid-epoch kill resumes
    # in the SAME epoch instead of skipping forward to args.epochs.
    hit_max_steps = False
    last_completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs):
        if hit_max_steps:
            break
        for step in range(args.steps):
            if args.max_steps is not None and global_step >= args.max_steps:
                print(f"[max-steps] reached limit "
                      f"global_step={global_step} >= {args.max_steps}; "
                      f"breaking loops, will save below.")
                hit_max_steps = True
                break
            idx = torch.randint(0, rna_all.shape[0], (args.batch_size,))
            rna_batch = rna_all[idx].to(device)
            atac_batch = atac_all[idx].to(device)

            # Random masks (15%) for reconstruction heads.
            rna_mask = (torch.rand_like(rna_batch) < 0.15).float()
            atac_mask = (torch.rand_like(atac_batch) < 0.15).float()

            # PR #46 (logical): thread the lysis covariate into the FIRST
            # encoder forward instead of running a no-covariate pre-pass and
            # then re-running with covariate. The pre-pass shape-mismatched
            # in joint mode because encoders constructed with
            # n_lysis_categories=2 expect n_genes + cov_dim input features;
            # passing only n_genes triggered RuntimeError on the input Linear.
            # Single-arm (lll/dig) and mock paths use lysis_batch=None so the
            # encoders' lysis_emb (None at construction) and forward signature
            # behave identically to the pre-PR-#46 code.
            lysis_batch = (
                lysis_all[idx].to(device) if args.arm == "joint" else None
            )
            enc_kwargs = (
                {"lysis_idx": lysis_batch} if lysis_batch is not None else {}
            )

            rna_latent, rna_recon = rna_enc(rna_batch, **enc_kwargs)
            atac_latent = atac_enc(atac_batch, **enc_kwargs)
            atac_recon = atac_decoder(atac_latent)

            if args.arm in ("lll", "dig", "joint"):
                # PR #42 (logical): tri-modal DOGMA path — _dogma_pretrain_loss
                # with masked recon + 3-way InfoNCE. ProteinEncoder.forward
                # signature is (adt, rna_emb=None, lysis_idx=None); we pass
                # rna_latent as the alignment query so the cross-attn block
                # participates.
                # PR #43 (logical): joint arm threads lysis_idx into all three
                # encoders. Single-arm (lll/dig): no covariate (lysis_emb is
                # None at construction).
                prot_batch = prot_all[idx].to(device)
                prot_token_mask = (torch.rand_like(prot_batch) < 0.15).float()

                if lysis_batch is not None:
                    protein_latent = protein_enc(
                        prot_batch, rna_emb=rna_latent, lysis_idx=lysis_batch
                    )
                else:
                    protein_latent = protein_enc(prot_batch, rna_emb=rna_latent)
                protein_recon = protein_decoder(protein_latent)

                # Project all three encoder latents to common proj_dim for
                # InfoNCE matmul (z_a @ z_b.t() requires equal hidden dim).
                z_rna = rna_proj(rna_latent)
                z_atac = atac_proj(atac_latent)
                z_protein = protein_proj(protein_latent)

                # DOGMA single-arm: PHOSPHO absent, others present.
                # ModalityKey order: ATAC=0, PHOSPHO=1, RNA=2, PROTEIN=3.
                B = rna_batch.shape[0]
                modality_mask = torch.tensor(
                    [[1, 0, 1, 1]], device=device, dtype=torch.float32
                ).expand(B, 4)

                # PR #54c: build z_supcon as L2-normalized mean of L2-normalized
                # per-modality projections. Parameter-free fusion for SupCon's
                # cell-type-aware contrastive signal.
                if supcon_cfg is not None:
                    z_supcon = F.normalize(
                        (
                            F.normalize(z_rna, dim=-1)
                            + F.normalize(z_atac, dim=-1)
                            + F.normalize(z_protein, dim=-1)
                        ) / 3.0,
                        dim=-1,
                    )
                    cell_type_batch = cell_type_idx_all[idx].to(device)
                    supcon_eligible_batch = supcon_eligible_all[idx].to(device)
                    w_supcon = float(supcon_cfg["lambda_supcon"])
                    w_vicreg = float(supcon_cfg["lambda_vicreg"])
                    supcon_temperature = float(supcon_cfg["temperature"])
                    vicreg_target_std = float(supcon_cfg.get("vicreg_target_std", 1.0))
                else:
                    z_supcon = None
                    cell_type_batch = None
                    supcon_eligible_batch = None
                    w_supcon = 0.0
                    w_vicreg = 0.0
                    supcon_temperature = 0.07
                    vicreg_target_std = 1.0

                from losses import _dogma_pretrain_loss  # noqa: E402
                total, components = _dogma_pretrain_loss(
                    rna_recon=rna_recon, rna_target=rna_batch, rna_mask=rna_mask,
                    atac_recon=atac_recon, atac_target=atac_batch, atac_mask=atac_mask,
                    protein_recon=protein_recon, protein_target=prot_batch,
                    protein_mask=prot_token_mask,
                    z_rna=z_rna, z_atac=z_atac, z_protein=z_protein,
                    modality_mask=modality_mask,
                    infonce_temperature=0.1,
                    # PR #54c: SupCon + VICReg threaded through.
                    z_supcon=z_supcon,
                    cell_type_idx=cell_type_batch,
                    supcon_eligible_mask=supcon_eligible_batch,
                    w_supcon=w_supcon,
                    w_vicreg=w_vicreg,
                    supcon_temperature=supcon_temperature,
                    vicreg_target_std=vicreg_target_std,
                )
            else:
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
        # PR #45 (logical): mark the epoch completed only if the inner loop
        # ran to its natural end (no max-steps break). This drives the
        # resume_state.epoch field — see save block below.
        if not hit_max_steps:
            last_completed_epoch = epoch

        # PR #48: periodic checkpoint at the end of every N completed epochs.
        # Naming: pretrain_encoders_epoch_{N:04d}.pt to keep history. End-of-
        # training save (after the outer loop) uses pretrain_encoders.pt.
        # Skip on a max-steps termination epoch — the unconditional final
        # save below handles that case (avoids duplicate writes).
        if (
            ckpt_every > 0
            and not hit_max_steps
            and (epoch + 1) % ckpt_every == 0
        ):
            # Stamp config keys that the helper expects (mirrors the final
            # save block; consolidated up here so periodic ckpts have the
            # same config schema as the final ckpt).
            config_periodic = dict(vars(args))
            config_periodic["n_genes"] = int(n_genes)
            config_periodic["n_peaks"] = int(n_peaks)
            config_periodic["hidden_dim"] = int(rna_enc.hidden_dim)
            config_periodic["latent_dim"] = int(rna_enc.latent_dim)
            config_periodic["device"] = str(device)
            config_periodic["peak_set_sha256"] = None  # filled below in final
            config_periodic["n_lysis_categories"] = int(n_lysis_categories)
            config_periodic["lysis_cov_dim"] = int(args.lysis_cov_dim)
            if args.arm in ("lll", "dig", "joint"):
                config_periodic["n_proteins"] = int(n_proteins)
                config_periodic["protein_latent"] = int(args.protein_latent)
                config_periodic["arm"] = str(args.arm)

            periodic_path = (
                Path(args.checkpoint_dir)
                / f"pretrain_encoders_epoch_{epoch + 1:04d}.pt"
            )
            is_trimodal = args.arm in ("lll", "dig", "joint")
            _save_checkpoint(
                periodic_path,
                schema_version=PRETRAIN_CKPT_SCHEMA_VERSION,
                rna_encoder=rna_enc,
                atac_encoder=atac_enc,
                pretrain_head=head,
                protein_encoder=protein_enc if is_trimodal else None,
                protein_decoder=protein_decoder if is_trimodal else None,
                rna_proj=rna_proj if is_trimodal else None,
                atac_proj=atac_proj if is_trimodal else None,
                protein_proj=protein_proj if is_trimodal else None,
                optimizer=optim,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                config=config_periodic,
            )

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

    # PR #42-#44 config stamps consolidated up here so they're in place for
    # both periodic AND end-of-training saves (helper passes config through
    # by value at call time — stamps must precede any save call).
    config_to_save["n_lysis_categories"] = int(n_lysis_categories)
    config_to_save["lysis_cov_dim"] = int(args.lysis_cov_dim)
    if args.arm in ("lll", "dig", "joint"):
        config_to_save["n_proteins"] = int(n_proteins)
        config_to_save["protein_latent"] = int(args.protein_latent)
        config_to_save["arm"] = str(args.arm)

    # PR #48: end-of-training save delegated to _save_checkpoint helper. The
    # same helper powers periodic saves inside the outer loop. Either
    # checkpoint file (periodic or final) is independently resume-able.
    ckpt_path = ckpt_dir / "pretrain_encoders.pt"
    is_trimodal = args.arm in ("lll", "dig", "joint")
    ckpt_sha256 = _save_checkpoint(
        ckpt_path,
        schema_version=PRETRAIN_CKPT_SCHEMA_VERSION,
        rna_encoder=rna_enc,
        atac_encoder=atac_enc,
        pretrain_head=head,
        protein_encoder=protein_enc if is_trimodal else None,
        protein_decoder=protein_decoder if is_trimodal else None,
        rna_proj=rna_proj if is_trimodal else None,
        atac_proj=atac_proj if is_trimodal else None,
        protein_proj=protein_proj if is_trimodal else None,
        optimizer=optim,
        scheduler=scheduler,
        epoch=last_completed_epoch,
        global_step=global_step,
        config=config_to_save,
    )
    if run is not None:
        try:
            run.config.update({"ckpt_sha256": ckpt_sha256}, allow_val_change=True)
        except Exception as e:  # pragma: no cover - W&B network issue fallback
            print(f"[warn] failed to log ckpt_sha256 to W&B: {e}")
        run.finish()


if __name__ == "__main__":
    main()
