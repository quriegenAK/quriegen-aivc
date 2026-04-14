"""
train_hpc.py — HPC/SLURM-ready single-config trainer for AIVC v1.1.

This is the MareNostrum5 / EuroHPC entrypoint. Key differences from train_v11.py:

  * Single (lfc_beta, neumann_K, lambda_l1) per invocation — array jobs fan out
    the sweep, not an in-process loop. Walltime kills only lose one config.
  * Full argparse CLI — no in-file hardcoded paths. All data/output/checkpoint
    locations configurable so the same binary runs under `/gpfs/scratch/...`
    or `/gpfs/projects/...` without code edits.
  * Full checkpoint state — model + optimizer + scheduler + epoch + rng_state
    + best_val_r + sparsity_history + compile flag. Tagged LATEST + BEST.
  * --resume flag — reloads everything and picks up at epoch+1.
  * SIGTERM handler — SLURM sends SIGTERM before walltime kill; we snapshot
    LATEST and exit cleanly so --resume works on the next array requeue.
  * No internet — MLflow logs to a file-based tracking URI on GPFS. No HTTP.
  * torch.compile gated behind --compile; default off on first run because
    Inductor caches to $HOME which is bandwidth-limited on Lustre/GPFS.

Usage (local):
  python scripts/train_hpc.py \
      --data-dir data \
      --output-dir outputs/v1.1 \
      --checkpoint-dir checkpoints/v1.1 \
      --sweep-index 0 \
      --epochs 200

Usage (SLURM array, MareNostrum5):
  sbatch --array=0-35 scripts/train_v11.sbatch

The sweep grid is fixed (4 x 3 x 3 = 36) to match train_v11.py for
reproducibility. Use --sweep-index 0..35 or --lfc-beta/--neumann-k/--lambda-l1
to override directly for ad-hoc runs.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

# Make project root importable no matter where the script is launched from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import anndata as ad  # noqa: E402

from perturbation_model import PerturbationPredictor, CellTypeEmbedding  # noqa: E402
from losses import combined_loss_v11  # noqa: E402
from aivc.skills.neumann_propagation import NeumannPropagation  # noqa: E402
from aivc.memory.mlflow_backend import MLflowBackend  # noqa: E402


# ============================================================================
# Sweep grid (frozen — identical to train_v11.py for reproducibility)
# ============================================================================
SWEEP_LFC_BETAS = [0.05, 0.10, 0.20, 0.50]
SWEEP_NEUMANN_KS = [2, 3, 5]
SWEEP_LAMBDA_L1S = [0.0001, 0.001, 0.01]


def sweep_index_to_config(idx: int) -> tuple[float, int, float]:
    """Map 0..35 -> (lfc_beta, neumann_K, lambda_l1)."""
    n_k = len(SWEEP_NEUMANN_KS)
    n_l1 = len(SWEEP_LAMBDA_L1S)
    total = len(SWEEP_LFC_BETAS) * n_k * n_l1
    if not 0 <= idx < total:
        raise ValueError(f"sweep_index {idx} out of range [0, {total})")
    b = idx // (n_k * n_l1)
    k = (idx // n_l1) % n_k
    l = idx % n_l1
    return SWEEP_LFC_BETAS[b], SWEEP_NEUMANN_KS[k], SWEEP_LAMBDA_L1S[l]


# ============================================================================
# Argparse
# ============================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AIVC v1.1 HPC trainer (single config, resumable)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data inputs
    p.add_argument("--data-dir", type=Path, default=Path("data"),
                   help="Root dir containing .h5ad, .npy, edge list")
    p.add_argument("--adata", type=str, default="kang2018_pbmc_fixed.h5ad")
    p.add_argument("--ot-ctrl", type=str, default="X_ctrl_ot.npy")
    p.add_argument("--ot-stim", type=str, default="X_stim_ot.npy")
    p.add_argument("--ot-cell-types", type=str, default="cell_type_ot.npy")
    p.add_argument("--ot-donors", type=str, default="donor_ot.npy")
    p.add_argument("--ctrl-paired", type=str, default="X_ctrl_paired.npy")
    p.add_argument("--stim-paired", type=str, default="X_stim_paired.npy")
    p.add_argument("--pairing-manifest", type=str, default="pairing_manifest.csv")
    p.add_argument("--edges", type=str, default="edge_list_fixed.csv")

    # Output paths
    p.add_argument("--output-dir", type=Path, default=Path("outputs/v1.1"))
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/v1.1"))
    p.add_argument("--mlflow-tracking-uri", type=str,
                   default="file:./mlruns",
                   help="MLflow URI. Use file:/gpfs/scratch/.../mlruns on HPC.")
    p.add_argument("--experiment-name", type=str, default="aivc_v11_hpc")

    # Sweep selection — mutually useful
    p.add_argument("--sweep-index", type=int, default=None,
                   help="0..35 — picks (beta, K, l1) from fixed grid")
    p.add_argument("--lfc-beta", type=float, default=None)
    p.add_argument("--neumann-k", type=int, default=None)
    p.add_argument("--lambda-l1", type=float, default=None)

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--stage1-epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--early-stop-min-epoch", type=int, default=80)
    p.add_argument("--sparsity-threshold", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subsample", type=int, default=50,
                   help="N cells per (donor, cell_type) for stochastic mini-bulk")

    # HPC plumbing
    p.add_argument("--resume", action="store_true",
                   help="Resume from LATEST checkpoint if present")
    p.add_argument("--compile", action="store_true",
                   help="Enable torch.compile (reduce-overhead). Off by default on first run.")
    p.add_argument("--checkpoint-every", type=int, default=5,
                   help="Save LATEST checkpoint every N epochs")
    p.add_argument("--baseline-r", type=float, default=0.863)
    return p


def resolve_config(args) -> tuple[float, int, float]:
    """Pick (beta, K, l1) from --sweep-index OR explicit flags."""
    if args.sweep_index is not None:
        return sweep_index_to_config(args.sweep_index)
    if None in (args.lfc_beta, args.neumann_k, args.lambda_l1):
        raise SystemExit(
            "Must provide either --sweep-index OR all three of "
            "(--lfc-beta, --neumann-k, --lambda-l1)"
        )
    return args.lfc_beta, args.neumann_k, args.lambda_l1


# ============================================================================
# Reproducibility
# ============================================================================
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_snapshot() -> dict:
    snap = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        snap["cuda"] = torch.cuda.get_rng_state_all()
    return snap


def rng_restore(snap: dict) -> None:
    random.setstate(snap["python"])
    np.random.set_state(snap["numpy"])
    torch.set_rng_state(snap["torch"])
    if torch.cuda.is_available() and "cuda" in snap:
        torch.cuda.set_rng_state_all(snap["cuda"])


# ============================================================================
# Data loading (parameterised copy of train_v11.py block 1)
# ============================================================================
@dataclass
class DataBundle:
    n_genes: int
    gene_names: list
    gene_to_idx: dict
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    X_ctrl_det: np.ndarray
    X_stim_det: np.ndarray
    cell_types_det: np.ndarray
    donors_det: np.ndarray
    ct_indices: torch.Tensor
    use_ot: bool
    X_ctrl_raw: np.ndarray | None
    X_stim_raw: np.ndarray | None
    cell_types_raw: np.ndarray | None
    donors_raw: np.ndarray | None
    train_idx: list
    val_idx: list
    test_idx: list
    train_donors: set
    jakstat_idx: dict


def load_data(args, device) -> DataBundle:
    data_dir = args.data_dir
    print(f"[AIVC] Loading data from {data_dir}")

    adata = ad.read_h5ad(data_dir / args.adata)
    gene_names = (adata.var["name"].tolist()
                  if "name" in adata.var.columns else adata.var_names.tolist())
    n_genes = len(gene_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    use_ot = False
    X_ctrl_raw = X_stim_raw = cell_types_raw = donors_raw = None
    ot_ctrl_path = data_dir / args.ot_ctrl
    if ot_ctrl_path.exists():
        X_ctrl_raw = np.load(ot_ctrl_path)
        X_stim_raw = np.load(data_dir / args.ot_stim)
        cell_types_raw = np.load(data_dir / args.ot_cell_types, allow_pickle=True)
        donors_raw = np.load(data_dir / args.ot_donors, allow_pickle=True)
        if X_ctrl_raw.shape[0] >= 200:
            use_ot = True

    if use_ot:
        print(f"  OT pairs: {X_ctrl_raw.shape[0]} x {n_genes} genes")
        groups: dict = {}
        for i in range(len(X_ctrl_raw)):
            key = (donors_raw[i], cell_types_raw[i])
            groups.setdefault(key, []).append(i)
        X_ctrl_b, X_stim_b, ct_b, donor_b = [], [], [], []
        for (d, ct) in sorted(groups.keys()):
            idxs = groups[(d, ct)]
            X_ctrl_b.append(X_ctrl_raw[idxs].mean(axis=0))
            X_stim_b.append(X_stim_raw[idxs].mean(axis=0))
            ct_b.append(ct)
            donor_b.append(d)
        X_ctrl_det = np.array(X_ctrl_b)
        X_stim_det = np.array(X_stim_b)
        cell_types_det = np.array(ct_b)
        donors_det = np.array(donor_b)
        print(f"  Mini-bulk groups: {len(X_ctrl_det)}")
    else:
        X_ctrl_det = np.load(data_dir / args.ctrl_paired)
        X_stim_det = np.load(data_dir / args.stim_paired)
        manifest = pd.read_csv(data_dir / args.pairing_manifest)
        paired = manifest[manifest["paired"]].reset_index(drop=True)
        cell_types_det = paired["cell_type"].values
        donors_det = paired["donor_id"].values
        print(f"  Pseudo-bulk pairs: {X_ctrl_det.shape[0]}")

    ct_indices = CellTypeEmbedding.encode_cell_types(cell_types_det.tolist())

    # Edges
    edge_df = pd.read_csv(data_dir / args.edges)
    edges, edge_weights = [], []
    for _, row in edge_df.iterrows():
        a = gene_to_idx.get(row["gene_a"])
        b = gene_to_idx.get(row["gene_b"])
        if a is not None and b is not None:
            edges.append([a, b])
            w = (row.get("combined_score", 700)
                 if "combined_score" in edge_df.columns else 700)
            edge_weights.append(w)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(
        device, non_blocking=True)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32).to(
        device, non_blocking=True)
    print(f"  Edges: {edge_index.shape[1]}")

    # Donor split (deterministic, same as train_v11.py)
    unique_donors = sorted(np.unique(donors_det).tolist())
    n_donors = len(unique_donors)
    n_train = max(1, int(n_donors * 0.6))
    n_val = max(1, int(n_donors * 0.2))
    train_donors = set(unique_donors[:n_train])
    val_donors = set(unique_donors[n_train:n_train + n_val])
    test_donors = set(unique_donors[n_train + n_val:])
    train_idx = [i for i in range(len(donors_det)) if donors_det[i] in train_donors]
    val_idx = [i for i in range(len(donors_det)) if donors_det[i] in val_donors]
    test_idx = [i for i in range(len(donors_det)) if donors_det[i] in test_donors]
    print(f"  Train: {len(train_idx)} Val: {len(val_idx)} Test: {len(test_idx)}")

    jakstat_genes = [
        "JAK1", "JAK2", "STAT1", "STAT2", "STAT3",
        "IRF9", "IRF1", "MX1", "MX2", "ISG15",
        "OAS1", "IFIT1", "IFIT3", "IFITM1", "IFITM3",
    ]
    jakstat_idx = {g: gene_to_idx[g] for g in jakstat_genes if g in gene_to_idx}

    return DataBundle(
        n_genes=n_genes, gene_names=gene_names, gene_to_idx=gene_to_idx,
        edge_index=edge_index, edge_attr=edge_attr,
        X_ctrl_det=X_ctrl_det, X_stim_det=X_stim_det,
        cell_types_det=cell_types_det, donors_det=donors_det,
        ct_indices=ct_indices, use_ot=use_ot,
        X_ctrl_raw=X_ctrl_raw, X_stim_raw=X_stim_raw,
        cell_types_raw=cell_types_raw, donors_raw=donors_raw,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        train_donors=train_donors, jakstat_idx=jakstat_idx,
    )


# ============================================================================
# Helpers
# ============================================================================
def compute_pearson_r(predicted, actual):
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    rs = []
    for i in range(predicted.shape[0]):
        p, a = predicted[i], actual[i]
        if np.std(p) < 1e-10 or np.std(a) < 1e-10:
            rs.append(0.0)
            continue
        r = np.corrcoef(p, a)[0, 1]
        rs.append(0.0 if np.isnan(r) else r)
    return float(np.mean(rs)), float(np.std(rs))


def compute_jakstat_fc(pred_np, actual_np, ctrl_np, jakstat_idx):
    eps = 1e-6
    results, within_3x, within_10x = {}, 0, 0
    for g, idx in jakstat_idx.items():
        c = ctrl_np[:, idx].mean()
        p = pred_np[:, idx].mean()
        a = actual_np[:, idx].mean()
        pfc = (p + eps) / (c + eps)
        afc = (a + eps) / (c + eps)
        ratio = (max(pfc, afc) / max(min(pfc, afc), eps)
                 if min(pfc, afc) > 0.01 else float("inf"))
        results[g] = {"pred_fc": pfc, "actual_fc": afc, "ratio": ratio}
        if ratio <= 3.0:
            within_3x += 1
        if ratio <= 10.0:
            within_10x += 1
    return results, within_3x, within_10x


def build_minibulk_groups(data: DataBundle) -> tuple[dict, list]:
    groups: dict = {}
    for i in range(len(data.X_ctrl_raw)):
        d = data.donors_raw[i]
        if d in data.train_donors:
            key = (d, data.cell_types_raw[i])
            groups.setdefault(key, []).append(i)
    return groups, sorted(groups.keys())


def stochastic_minibulk(data: DataBundle, groups: dict, keys: list,
                         n_sub: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X_ctrl, X_stim, ct_ = [], [], []
    for (d, ct) in keys:
        idxs = groups[(d, ct)]
        n = len(idxs)
        sub = rng.choice(idxs, min(n_sub, n), replace=(n < n_sub))
        X_ctrl.append(data.X_ctrl_raw[sub].mean(axis=0))
        X_stim.append(data.X_stim_raw[sub].mean(axis=0))
        ct_.append(ct)
    return np.array(X_ctrl), np.array(X_stim), np.array(ct_)


# ============================================================================
# Model construction
# ============================================================================
def build_model(data: DataBundle, neumann_K: int, lambda_l1: float,
                 device: torch.device) -> nn.Module:
    model = PerturbationPredictor(
        n_genes=data.n_genes, num_perturbations=2, feature_dim=64,
        hidden1_dim=64, hidden2_dim=32, num_head1=3, num_head2=2,
        decoder_hidden=256,
    ).to(device)
    model.cell_type_embedding = CellTypeEmbedding(
        num_cell_types=20, embedding_dim=model.feature_dim,
    ).to(device)
    model.neumann = NeumannPropagation(
        n_genes=data.n_genes,
        edge_index=data.edge_index.cpu(),
        edge_attr=data.edge_attr.cpu(),
        K=neumann_K,
        lambda_l1=lambda_l1,
    ).to(device)
    return model


# ============================================================================
# Checkpointing
# ============================================================================
CHECKPOINT_LATEST = "latest.pt"
CHECKPOINT_BEST = "best.pt"


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic — never leave a half-written checkpoint


def load_checkpoint(path: Path, device: torch.device) -> dict | None:
    if not path.exists():
        return None
    return torch.load(path, map_location=device, weights_only=False)


def build_state(
    *, model, optimizer, scheduler, epoch, best_val_r, best_epoch, no_improve,
    sparsity_history, lfc_beta, neumann_K, lambda_l1, args, _compiled,
) -> dict:
    # Unwrap compiled model for state_dict (torch.compile wraps in OptimizedModule)
    raw_model = getattr(model, "_orig_mod", model)
    return {
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_r": best_val_r,
        "best_epoch": best_epoch,
        "no_improve": no_improve,
        "sparsity_history": sparsity_history,
        "rng_state": rng_snapshot(),
        "lfc_beta": lfc_beta,
        "neumann_K": neumann_K,
        "lambda_l1": lambda_l1,
        "args": vars(args),
        "compiled": _compiled,
        "torch_version": torch.__version__,
        "timestamp": time.time(),
    }


# ============================================================================
# Signal handling — SLURM sends SIGTERM before walltime kill
# ============================================================================
_shutdown_requested = False


def install_signal_handlers():
    def _handler(signum, frame):
        global _shutdown_requested
        print(f"\n[AIVC] Received signal {signum} — will snapshot at next "
              "epoch boundary and exit cleanly.", flush=True)
        _shutdown_requested = True
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGUSR1, _handler)  # SLURM --signal=USR1@60


# ============================================================================
# Training
# ============================================================================
def train(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AIVC] Device: {device}")
    if device.type == "cuda":
        print(f"[AIVC] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[AIVC] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        gpu_name = torch.cuda.get_device_name(0)
        if any(k in gpu_name for k in ("H100", "A100", "H200")):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"[AIVC] Mixed precision dtype: {autocast_dtype}")

    lfc_beta, neumann_K, lambda_l1 = resolve_config(args)
    tag = f"beta{lfc_beta}_K{neumann_K}_l1{lambda_l1}"
    ckpt_dir = args.checkpoint_dir / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_path = ckpt_dir / CHECKPOINT_LATEST
    best_path = ckpt_dir / CHECKPOINT_BEST

    print(f"[AIVC] Config: {tag}")
    print(f"[AIVC] Checkpoint dir: {ckpt_dir}")

    seed_all(args.seed)
    data = load_data(args, device)
    ot_groups, ot_keys = ({}, [])
    if data.use_ot:
        ot_groups, ot_keys = build_minibulk_groups(data)

    # ── MLflow (file-based tracking URI — no HTTP required) ────────
    os.environ.setdefault("MLFLOW_TRACKING_URI", args.mlflow_tracking_uri)
    mlflow_backend = MLflowBackend()
    _run_id = mlflow_backend.start_run(
        run_name=tag,
        params={
            "lfc_beta": lfc_beta, "neumann_K": neumann_K, "lambda_l1": lambda_l1,
            "n_epochs": args.epochs, "n_genes": data.n_genes,
            "n_edges": data.edge_index.shape[1], "seed": args.seed,
            "stage1_epochs": args.stage1_epochs,
            "stage2_start": args.stage1_epochs + 1,
            "sparsity_threshold": args.sparsity_threshold,
            "dataset": args.adata, "compile": args.compile,
            "batch_size": args.batch_size, "lr": args.lr,
        },
    )

    # ── Build model + optimizer + scheduler ────────────────────────
    model = build_model(data, neumann_K, lambda_l1, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[AIVC] Parameters: {total_params:,}")
    assert next(model.parameters()).device.type == device.type

    # Stage 1: freeze W
    model.neumann.freeze_W()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_r = -1.0
    best_epoch = 0
    no_improve = 0
    start_epoch = 1
    sparsity_history: list = []
    _compiled = False

    # ── Resume ─────────────────────────────────────────────────────
    if args.resume:
        snap = load_checkpoint(latest_path, device)
        if snap is None:
            print(f"[AIVC] --resume set but {latest_path} not found. Starting fresh.")
        else:
            print(f"[AIVC] Resuming from epoch {snap['epoch']} "
                  f"(best_val_r={snap['best_val_r']:.4f} @ epoch {snap['best_epoch']})")
            # Unfreeze W if we are past stage 1 so state_dict matches
            if snap["epoch"] >= args.stage1_epochs + 1:
                model.neumann.unfreeze_W()
            model.load_state_dict(snap["model_state_dict"])
            optimizer.load_state_dict(snap["optimizer_state_dict"])
            scheduler.load_state_dict(snap["scheduler_state_dict"])
            start_epoch = snap["epoch"] + 1
            best_val_r = snap["best_val_r"]
            best_epoch = snap["best_epoch"]
            no_improve = snap["no_improve"]
            sparsity_history = snap["sparsity_history"]
            rng_restore(snap["rng_state"])

    # ── torch.compile (opt-in) ─────────────────────────────────────
    _model_eager = model
    if args.compile and device.type == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            _compiled = True
            print("[AIVC] torch.compile enabled (mode=reduce-overhead)")
        except Exception as e:
            print(f"[AIVC] torch.compile failed — eager mode: {e}")
            model = _model_eager

    pert_id_stim = torch.tensor([1], device=device)
    install_signal_handlers()
    _step_count = 0
    t_start = time.time()

    # ── Training loop ──────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        if epoch == args.stage1_epochs + 1 and not args.resume:
            model.neumann.unfreeze_W()
            print(f"  Epoch {epoch}: Unfreezing Neumann W (Stage 2)")

        # LFC beta ramp
        if epoch <= 30:
            beta_lfc = 0.0
        elif epoch <= 80:
            beta_lfc = lfc_beta * min(1.0, (epoch - 30) / 50)
        else:
            beta_lfc = lfc_beta

        model.train()
        epoch_losses = {"mse": [], "lfc": [], "cosine": [], "l1": [], "total": []}

        if data.use_ot:
            X_ctrl_ep, X_stim_ep, ct_ep_str = stochastic_minibulk(
                data, ot_groups, ot_keys, args.subsample, args.seed + epoch)
            ct_ep = CellTypeEmbedding.encode_cell_types(ct_ep_str.tolist())
        else:
            X_ctrl_ep = data.X_ctrl_det[data.train_idx]
            X_stim_ep = data.X_stim_det[data.train_idx]
            ct_ep = data.ct_indices[data.train_idx]

        X_ctrl_tensor = torch.tensor(X_ctrl_ep, dtype=torch.float32).to(
            device, non_blocking=True)
        X_stim_tensor = torch.tensor(X_stim_ep, dtype=torch.float32).to(
            device, non_blocking=True)
        ct_tensor = ct_ep.to(device, non_blocking=True)

        n_ep = len(X_ctrl_tensor)
        perm = torch.randperm(n_ep, device=device)
        X_ctrl_epoch = X_ctrl_tensor[perm]
        X_stim_epoch = X_stim_tensor[perm]
        ct_epoch = ct_tensor[perm]

        for start in range(0, n_ep, args.batch_size):
            _step_count += 1
            end = min(start + args.batch_size, n_ep)
            ctrl_b = X_ctrl_epoch[start:end]
            stim_b = X_stim_epoch[start:end]
            ct_b = ct_epoch[start:end]

            with autocast(device_type=device.type, dtype=autocast_dtype):
                pred_delta = model.forward_batch(ctrl_b, data.edge_index,
                                                 pert_id_stim, ct_b)
                predicted = (ctrl_b + pred_delta).clamp(min=0.0)
                loss, bd = combined_loss_v11(
                    predicted=predicted, actual_stim=stim_b, actual_ctrl=ctrl_b,
                    neumann_module=model.neumann,
                    alpha=1.0, beta=beta_lfc, gamma=0.1,
                )

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(
                    f"[AIVC] NaN/Inf detected in loss at epoch {epoch}, "
                    f"step {_step_count}. Loss breakdown: {bd}. "
                    "Aborting — do not waste GPU time on a corrupted run."
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if device.type == "cuda" and _step_count == 1:
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"[AIVC] GPU memory — allocated: {allocated:.2f} GB "
                      f"| reserved: {reserved:.2f} GB")
                assert allocated > 0.1, (
                    f"GPU allocated ({allocated:.3f} GB) < 0.1 GB. "
                    "Model not on GPU.")

            for k in epoch_losses:
                epoch_losses[k].append(bd[k])

        scheduler.step()
        avg = {k: float(np.mean(v)) for k, v in epoch_losses.items()}

        # ── Validation ────────────────────────────────────────────
        model.eval()
        val_preds, val_actuals = [], []
        with torch.no_grad():
            with autocast(device_type=device.type, dtype=autocast_dtype):
                for s in range(0, len(data.val_idx), args.batch_size):
                    e = min(s + args.batch_size, len(data.val_idx))
                    bi = data.val_idx[s:e]
                    ctrl_b = torch.tensor(data.X_ctrl_det[bi], dtype=torch.float32).to(
                        device, non_blocking=True)
                    stim_b = torch.tensor(data.X_stim_det[bi], dtype=torch.float32).to(
                        device, non_blocking=True)
                    ct_b = data.ct_indices[bi].to(device, non_blocking=True)
                    pd_ = model.forward_batch(ctrl_b, data.edge_index,
                                              pert_id_stim, ct_b)
                    pred = (ctrl_b + pd_).clamp(min=0.0)
                    val_preds.append(pred.detach().cpu().float())
                    val_actuals.append(stim_b.detach().cpu().float())
        val_r, val_std = compute_pearson_r(torch.cat(val_preds), torch.cat(val_actuals))

        improved = val_r > best_val_r
        if improved:
            best_val_r = val_r
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        # ── Sparsity enforcement + logging ────────────────────────
        log_this_epoch = (epoch == 1 or epoch % 10 == 0)
        n_pruned = 0
        w_density = w_density_large = None
        if log_this_epoch:
            if (epoch >= args.stage1_epochs + 1
                    and getattr(model, "_orig_mod", model).neumann.W.requires_grad):
                sr = getattr(model, "_orig_mod", model).neumann.enforce_sparsity(
                    threshold=args.sparsity_threshold)
                n_pruned = sr["n_pruned"]
            wr = getattr(model, "_orig_mod", model).neumann.get_sparsity_report()
            w_density = wr["density"]
            w_density_large = wr["density_large"]
            sparsity_history.append({
                "epoch": epoch, "density": w_density,
                "density_large": w_density_large,
                "n_pruned": n_pruned, "val_r": val_r,
            })
            mlflow_backend.log_epoch_metrics(
                epoch=epoch, train_losses=avg, val_r=val_r,
                w_density=w_density, w_density_large=w_density_large,
                n_pruned=n_pruned,
            )
            print(
                f"  E{epoch:3d} | MSE={avg['mse']:.4f} LFC={avg['lfc']:.4f} "
                f"L1={avg['l1']:.6f} | val_r={val_r:.4f} | beta={beta_lfc:.3f} | "
                f"W_dens={w_density:.3f} (>{args.sparsity_threshold:.0e}: "
                f"{w_density_large:.3f}) pruned={n_pruned}"
            )

        # ── Checkpointing ──────────────────────────────────────────
        state = build_state(
            model=model, optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, best_val_r=best_val_r, best_epoch=best_epoch,
            no_improve=no_improve, sparsity_history=sparsity_history,
            lfc_beta=lfc_beta, neumann_K=neumann_K, lambda_l1=lambda_l1,
            args=args, _compiled=_compiled,
        )
        if improved:
            save_checkpoint(best_path, state)
        if epoch % args.checkpoint_every == 0 or improved or _shutdown_requested:
            save_checkpoint(latest_path, state)

        # ── Graceful shutdown on SIGTERM ───────────────────────────
        if _shutdown_requested:
            print(f"[AIVC] Shutdown requested — snapshot saved at epoch {epoch}. "
                  f"Exiting with code 124 so SLURM can requeue.")
            mlflow_backend.end_run(status="STOPPED")
            # Exit 124 is conventional for "timed out / requeue me"
            sys.exit(124)

        # ── Early stop ─────────────────────────────────────────────
        if (no_improve >= args.patience and epoch > args.early_stop_min_epoch):
            print(f"  Early stop at epoch {epoch}. "
                  f"Best: {best_val_r:.4f} at {best_epoch}")
            break

    elapsed = time.time() - t_start

    # ── Test-set evaluation using best checkpoint ──────────────────
    print(f"\n[AIVC] Loading best checkpoint for test evaluation")
    best_snap = load_checkpoint(best_path, device)
    raw_model = getattr(model, "_orig_mod", model)
    # Ensure W is unfrozen to match state shape
    raw_model.neumann.unfreeze_W()
    raw_model.load_state_dict(best_snap["model_state_dict"])
    raw_model.eval()

    test_preds, test_actuals, test_ctrls = [], [], []
    with torch.no_grad():
        with autocast(device_type=device.type, dtype=autocast_dtype):
            for s in range(0, len(data.test_idx), args.batch_size):
                e = min(s + args.batch_size, len(data.test_idx))
                bi = data.test_idx[s:e]
                ctrl_b = torch.tensor(data.X_ctrl_det[bi], dtype=torch.float32).to(
                    device, non_blocking=True)
                stim_b = torch.tensor(data.X_stim_det[bi], dtype=torch.float32).to(
                    device, non_blocking=True)
                ct_b = data.ct_indices[bi].to(device, non_blocking=True)
                pd_ = raw_model.forward_batch(ctrl_b, data.edge_index,
                                              pert_id_stim, ct_b)
                pred = (ctrl_b + pd_).clamp(min=0.0)
                test_preds.append(pred.detach().cpu().float())
                test_actuals.append(stim_b.detach().cpu().float())
                test_ctrls.append(ctrl_b.detach().cpu().float())

    test_pred_np = torch.cat(test_preds).numpy()
    test_actual_np = torch.cat(test_actuals).numpy()
    test_ctrl_np = torch.cat(test_ctrls).numpy()
    test_r, test_std = compute_pearson_r(test_pred_np, test_actual_np)
    fc_results, within_3x, within_10x = compute_jakstat_fc(
        test_pred_np, test_actual_np, test_ctrl_np, data.jakstat_idx)

    test_cell_types = [data.cell_types_det[i] for i in data.test_idx]
    ct_r: dict = {}
    for ct in sorted(set(test_cell_types)):
        mask = [i for i, c in enumerate(test_cell_types) if c == ct]
        if mask:
            r, _ = compute_pearson_r(test_pred_np[mask], test_actual_np[mask])
            ct_r[ct] = r

    ifit1_fc = fc_results.get("IFIT1", {})

    result = {
        "lfc_beta": lfc_beta, "neumann_K": neumann_K, "lambda_l1": lambda_l1,
        "best_val_r": best_val_r, "best_epoch": best_epoch,
        "test_r": test_r, "test_std": test_std,
        "jakstat_within_3x": within_3x, "jakstat_within_10x": within_10x,
        "ifit1_pred_fc": ifit1_fc.get("pred_fc", 0),
        "ifit1_actual_fc": ifit1_fc.get("actual_fc", 0),
        "cd14_mono_r": ct_r.get("CD14+ Monocytes", 0),
        "ct_r": ct_r, "fc_results": fc_results,
        "training_time_s": elapsed,
        "save_path": str(best_path),
        "w_density": raw_model.neumann.get_effective_W_density(),
        "sparsity_history": sparsity_history,
        "top_w_edges": raw_model.neumann.get_top_edges(
            n=20, gene_names=data.gene_names),
    }

    print(f"\n[AIVC] Results [{tag}]:")
    print(f"  Test r: {test_r:.4f} +/- {test_std:.4f}")
    print(f"  JAK-STAT within 3x: {within_3x}/15")
    print(f"  JAK-STAT within 10x: {within_10x}/15")
    print(f"  IFIT1 predicted {ifit1_fc.get('pred_fc', 0):.1f}x "
          f"(actual {ifit1_fc.get('actual_fc', 0):.1f}x)")
    print(f"  CD14 mono r: {ct_r.get('CD14+ Monocytes', 0):.4f}")
    print(f"  Wall time: {elapsed:.0f}s")

    # ── Persist per-config JSON result ─────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Strip tensor-y fields for JSON
    json_result = {k: v for k, v in result.items()
                   if k not in ("fc_results", "top_w_edges", "sparsity_history")}
    json_result["fc_results"] = {
        g: {k: float(v) for k, v in fc_results[g].items()}
        for g in fc_results
    }
    (args.output_dir / f"result_{tag}.json").write_text(
        json.dumps(json_result, indent=2, default=float))

    mlflow_backend.log_final_metrics(result)
    mlflow_backend.log_sparsity_history(sparsity_history)
    mlflow_backend.log_top_edges(
        top_edges=result["top_w_edges"],
        n_jakstat_in_top20=sum(
            1 for e in result["top_w_edges"]
            if (e.get("src_name"), e.get("dst_name")) in {
                ("JAK1", "STAT1"), ("JAK2", "STAT1"),
                ("STAT1", "IFIT1"), ("STAT1", "MX1"),
                ("STAT2", "IFIT1"), ("IRF9", "IFIT1"),
            }
        ),
    )
    mlflow_backend.log_model_checkpoint(str(best_path))
    mlflow_backend.end_run(
        status="FINISHED" if test_r >= args.baseline_r else "FAILED")
    return result


# ============================================================================
# Entrypoint
# ============================================================================
def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
