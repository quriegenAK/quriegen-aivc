"""
train_week3.py — Week 3 training: OT pairs + LFC loss + cell-type embedding.

Three targeted improvements over train_perturbation.py:
    1. OT single-cell pairs instead of pseudo-bulk averages
    2. Combined loss (MSE + log-fold-change + cosine)
    3. Cell-type embedding for identity-specific responses

Training phases (3 phases):
    Phase 1 (1-30):   Warm-up: PerturbationEmbedding + CellTypeEmbedding only
    Phase 2 (31-80):  Add ResponseDecoder
    Phase 3 (81-200): Full fine-tuning with cosine annealing LR

Outputs:
    model_week3_best.pt           — best validation Pearson r checkpoint
    model_week3_epoch{N}.pt       — intermediate checkpoints every 50 epochs
    model_week3_early_stop.pt     — early stopping checkpoint (if triggered)
    training_log_week3.txt        — full training log
"""
import random
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import anndata as ad
from torch.utils.data import Dataset, DataLoader

# Reproducibility — all four seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

from perturbation_model import (
    PerturbationPredictor, CellTypeEmbedding, build_cell_type_index
)
from losses import combined_loss
from aivc.data.dataset_kind import DatasetKind

# Phase 2 dispatch: map dataset kind → loss stage.
# INTERVENTIONAL → "joint" preserves existing behavior for Kang (and any other
# paired-perturbation source). OBSERVATIONAL → "pretrain" is wired but not
# reachable in Phase 2 (no observational datasets are loaded).
_KIND_TO_STAGE = {
    DatasetKind.INTERVENTIONAL: "joint",
    DatasetKind.OBSERVATIONAL: "pretrain",
}

# =========================================================================
# 0. Device
# =========================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =========================================================================
# 1. Data loading
# =========================================================================
print("\nLoading data...")

# Gene names from adata
adata = ad.read_h5ad("data/kang2018_pbmc_fixed.h5ad")
if "name" in adata.var.columns:
    gene_names = adata.var["name"].tolist()
else:
    gene_names = adata.var_names.tolist()
n_genes = len(gene_names)
gene_to_idx = {g: i for i, g in enumerate(gene_names)}

# Try OT pairs first, fall back to pseudo-bulk
use_ot = False
if os.path.exists("data/X_ctrl_ot.npy") and os.path.exists("data/X_stim_ot.npy"):
    X_ctrl = np.load("data/X_ctrl_ot.npy")
    X_stim = np.load("data/X_stim_ot.npy")
    cell_types_arr = np.load("data/cell_type_ot.npy", allow_pickle=True)
    donors_arr = np.load("data/donor_ot.npy", allow_pickle=True)

    if X_ctrl.shape[0] >= 200:
        use_ot = True
        n_ot_pairs = X_ctrl.shape[0]
        print(f"  Loaded OT single-cell pairs: {n_ot_pairs} pairs, {n_genes} genes")

        # Index OT pairs by (donor, cell_type) group for stochastic mini-bulk.
        # Each epoch we randomly subsample cells within each group and average
        # to create fresh pseudo-bulk profiles. This gives data augmentation:
        # - 96% zeros at single-cell level → ~50% zeros at mini-bulk level
        # - Each epoch sees different mini-bulk profiles (noise regularization)
        # - OT matching quality preserved within each group
        ot_groups = {}
        for i in range(n_ot_pairs):
            key = (donors_arr[i], cell_types_arr[i])
            if key not in ot_groups:
                ot_groups[key] = []
            ot_groups[key].append(i)

        ot_group_keys = sorted(ot_groups.keys())
        print(f"  OT groups: {len(ot_group_keys)} (donor × cell_type)")
        print(f"  Group sizes: min={min(len(v) for v in ot_groups.values())}, "
              f"max={max(len(v) for v in ot_groups.values())}, "
              f"median={sorted(len(v) for v in ot_groups.values())[len(ot_groups)//2]}")

        # Create FULL mini-bulk (deterministic) for validation and test
        X_ctrl_bulk = []
        X_stim_bulk = []
        ct_bulk = []
        donor_bulk = []
        for (donor, ct) in ot_group_keys:
            indices = ot_groups[(donor, ct)]
            X_ctrl_bulk.append(X_ctrl[indices].mean(axis=0))
            X_stim_bulk.append(X_stim[indices].mean(axis=0))
            ct_bulk.append(ct)
            donor_bulk.append(donor)

        # For val/test: use deterministic full-group averages
        X_ctrl_full = np.array(X_ctrl_bulk)
        X_stim_full = np.array(X_stim_bulk)
        cell_types_arr_full = np.array(ct_bulk)
        donors_arr_full = np.array(donor_bulk)

        # Overwrite for downstream processing (val/test splits)
        X_ctrl = X_ctrl_full
        X_stim = X_stim_full
        cell_types_arr = cell_types_arr_full
        donors_arr = donors_arr_full
        print(f"  Val/test mini-bulk: {X_ctrl.shape[0]} groups, "
              f"{(X_ctrl == 0).mean():.1%} zeros")
    else:
        print(f"  WARNING: OT data has only {X_ctrl.shape[0]} pairs (< 200). Using pseudo-bulk fallback.")

if not use_ot:
    print("  WARNING: OT data not found or insufficient, using pseudo-bulk fallback")
    X_ctrl = np.load("data/X_ctrl_paired.npy")
    X_stim = np.load("data/X_stim_paired.npy")
    manifest = pd.read_csv("data/pairing_manifest.csv")
    paired = manifest[manifest["paired"]].reset_index(drop=True)
    cell_types_arr = paired["cell_type"].values
    donors_arr = paired["donor_id"].values
    print(f"  Loaded pseudo-bulk pairs: {X_ctrl.shape[0]} pairs, {n_genes} genes")

n_pairs = X_ctrl.shape[0]

# Cell type indices
ct_indices = CellTypeEmbedding.encode_cell_types(cell_types_arr.tolist())
print(f"  Cell type distribution:")
ct_counts = {}
for ct in cell_types_arr:
    ct_counts[ct] = ct_counts.get(ct, 0) + 1
for ct in sorted(ct_counts.keys()):
    print(f"    {ct:<24} {ct_counts[ct]} pairs")

# Edge list
edge_df = pd.read_csv("data/edge_list_fixed.csv")
edges = []
for _, row in edge_df.iterrows():
    a = gene_to_idx.get(row["gene_a"])
    b = gene_to_idx.get(row["gene_b"])
    if a is not None and b is not None:
        edges.append([a, b])
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
print(f"  Edges: {edge_index.shape[1]}")

# JAK-STAT genes
jakstat_genes = [
    "JAK1", "JAK2", "STAT1", "STAT2", "STAT3",
    "IRF9", "IRF1", "MX1", "MX2", "ISG15",
    "OAS1", "IFIT1", "IFIT3", "IFITM1", "IFITM3",
]
jakstat_idx = {g: gene_to_idx[g] for g in jakstat_genes if g in gene_to_idx}
jakstat_monitor = ["STAT1", "JAK1", "MX1", "ISG15", "IFIT1", "IFIT3", "IRF9", "OAS1"]

# =========================================================================
# 2. Donor-based train/val/test split
# =========================================================================
unique_donors = sorted(np.unique(donors_arr).tolist())
n_donors = len(unique_donors)
n_train = max(1, int(n_donors * 0.6))
n_val = max(1, int(n_donors * 0.2))
# Rest go to test

train_donors = set(unique_donors[:n_train])
val_donors = set(unique_donors[n_train:n_train + n_val])
test_donors = set(unique_donors[n_train + n_val:])

print(f"\n  Donor split:")
print(f"    TRAIN donors ({len(train_donors)}): {sorted(train_donors)}")
print(f"    VAL donors   ({len(val_donors)}):   {sorted(val_donors)}")
print(f"    TEST donors  ({len(test_donors)}):  {sorted(test_donors)}")

train_idx = [i for i in range(n_pairs) if donors_arr[i] in train_donors]
val_idx = [i for i in range(n_pairs) if donors_arr[i] in val_donors]
test_idx = [i for i in range(n_pairs) if donors_arr[i] in test_donors]

print(f"    Train pairs: {len(train_idx)}, Val pairs: {len(val_idx)}, Test pairs: {len(test_idx)}")

# =========================================================================
# 3. Dataset and DataLoader
# =========================================================================
class PerturbationDataset(Dataset):
    """Pairs of (ctrl_expression, stim_expression, cell_type_id, donor_id).
    Each item is one OT-paired cell pair.
    """
    def __init__(self, X_ctrl, X_stim, cell_type_ids, donor_ids):
        self.X_ctrl = torch.tensor(X_ctrl, dtype=torch.float32)
        self.X_stim = torch.tensor(X_stim, dtype=torch.float32)
        self.cell_type_ids = cell_type_ids.clone() if isinstance(cell_type_ids, torch.Tensor) else torch.tensor(cell_type_ids, dtype=torch.long)
        self.donor_ids = donor_ids

    def __len__(self):
        return self.X_ctrl.shape[0]

    def __getitem__(self, idx):
        return {
            "ctrl": self.X_ctrl[idx],
            "stim": self.X_stim[idx],
            "cell_type_id": self.cell_type_ids[idx],
            "donor_id": self.donor_ids[idx],
        }

BATCH_SIZE = 8      # Optimal for CPU GAT: 8 × 3010 = 24K nodes per forward pass
N_SUBSAMPLE = 50    # Number of cells to subsample per group for stochastic mini-bulk

# Load raw OT single-cell arrays for stochastic mini-bulk training
if use_ot:
    X_ctrl_raw = np.load("data/X_ctrl_ot.npy")
    X_stim_raw = np.load("data/X_stim_ot.npy")
    cell_types_raw = np.load("data/cell_type_ot.npy", allow_pickle=True)
    donors_raw = np.load("data/donor_ot.npy", allow_pickle=True)

    # Build train/val/test group indices for raw OT data
    ot_train_groups = {}  # (donor, ct) → [raw_indices]
    for i in range(len(X_ctrl_raw)):
        d, ct = donors_raw[i], cell_types_raw[i]
        if d in train_donors:
            key = (d, ct)
            if key not in ot_train_groups:
                ot_train_groups[key] = []
            ot_train_groups[key].append(i)
    ot_train_group_keys = sorted(ot_train_groups.keys())
    print(f"  Stochastic mini-bulk: {len(ot_train_group_keys)} train groups, "
          f"subsample {N_SUBSAMPLE} cells/group/epoch")

def generate_stochastic_minibulk(seed):
    """Generate one epoch of stochastic mini-bulk training data.

    For each (donor, cell_type) group in the training set:
    1. Randomly subsample N_SUBSAMPLE cells (with replacement if group < N_SUBSAMPLE)
    2. Average the subsampled cells to create one mini-bulk pair

    This gives the model fresh training data every epoch while maintaining
    the signal-to-noise benefits of averaging (~50% zeros vs 96%).
    """
    rng = np.random.RandomState(seed)
    X_ctrl_epoch = []
    X_stim_epoch = []
    ct_epoch = []
    donor_epoch = []

    for (donor, ct) in ot_train_group_keys:
        indices = ot_train_groups[(donor, ct)]
        n = len(indices)
        # Subsample with replacement if group is small
        sub_idx = rng.choice(indices, min(N_SUBSAMPLE, n), replace=(n < N_SUBSAMPLE))
        X_ctrl_epoch.append(X_ctrl_raw[sub_idx].mean(axis=0))
        X_stim_epoch.append(X_stim_raw[sub_idx].mean(axis=0))
        ct_epoch.append(ct)
        donor_epoch.append(donor)

    return (np.array(X_ctrl_epoch), np.array(X_stim_epoch),
            np.array(ct_epoch), np.array(donor_epoch))

# Deterministic mini-bulk for val/test
val_idx_groups = [i for i in range(n_pairs) if donors_arr[i] in val_donors]
test_idx_groups = [i for i in range(n_pairs) if donors_arr[i] in test_donors]

val_ds = PerturbationDataset(
    X_ctrl[val_idx_groups], X_stim[val_idx_groups],
    ct_indices[val_idx_groups], donors_arr[val_idx_groups]
)
test_ds = PerturbationDataset(
    X_ctrl[test_idx_groups], X_stim[test_idx_groups],
    ct_indices[test_idx_groups], donors_arr[test_idx_groups]
)

n_train_groups = len(ot_train_group_keys) if use_ot else len([i for i in range(n_pairs) if donors_arr[i] in train_donors])
print(f"  Train: {n_train_groups} groups (stochastic mini-bulk each epoch)")
print(f"  Val: {len(val_ds)} groups, Test: {len(test_ds)} groups")

# =========================================================================
# 4. Model setup
# =========================================================================
model = PerturbationPredictor(
    n_genes=n_genes,
    num_perturbations=2,
    feature_dim=64,
    hidden1_dim=64,
    hidden2_dim=32,
    num_head1=3,
    num_head2=2,
    decoder_hidden=256,
).to(device)

# Add cell type embedding
model.cell_type_embedding = CellTypeEmbedding(
    num_cell_types=20, embedding_dim=model.feature_dim
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {total_params:,}")

pert_id_stim = torch.tensor([1], device=device)

# =========================================================================
# 5. Pearson r computation
# =========================================================================
def compute_pearson_r(predicted, actual):
    """Compute mean Pearson r across pairs (gene-wise per pair)."""
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    rs = []
    for i in range(predicted.shape[0]):
        p = predicted[i]
        a = actual[i]
        if np.std(p) < 1e-10 or np.std(a) < 1e-10:
            rs.append(0.0)
            continue
        r = np.corrcoef(p, a)[0, 1]
        if np.isnan(r):
            r = 0.0
        rs.append(r)
    return float(np.mean(rs)), float(np.std(rs))


# =========================================================================
# 6. Training loop — 2-phase (warm-up MSE only, then full loss)
# =========================================================================
NUM_EPOCHS = 150
LFC_START = 30      # Introduce LFC loss after MSE stabilises
PATIENCE = 30

loss_log = []
best_val_r = -1.0
best_epoch = 0
no_improve_count = 0
early_stopped = False
ifit1_recovered_epoch = None

# Train all parameters together from the start — phased freezing caused
# dead embeddings when the frozen decoder couldn't produce useful gradients.
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
)

print(f"\n{'='*80}")
print(f"TRAINING START — {NUM_EPOCHS} epochs")
print(f"  Phase 1 (1-{LFC_START}):    MSE + cosine only (warm-up)")
print(f"  Phase 2 ({LFC_START+1}-{NUM_EPOCHS}): MSE + LFC + cosine (full loss)")
print(f"  Loss weights: MSE=1.0, LFC=0.01 (after epoch {LFC_START}), Cosine=0.1")
print(f"  Batch size: {BATCH_SIZE}, Early stopping patience: {PATIENCE}")
print(f"  LR: 5e-4 with cosine annealing to 1e-6")
print(f"{'='*80}")

for epoch in range(1, NUM_EPOCHS + 1):
    if epoch == LFC_START + 1:
        print(f"\n  Phase 2: Adding LFC loss (beta=0.01)")

    # --- LFC weight schedule ---
    # Start with MSE only to establish baseline, then gradually add LFC.
    # LFC loss is ~400x larger than MSE, so beta must be very small.
    if epoch <= LFC_START:
        beta_lfc = 0.0
    else:
        # Linear ramp from 0 to 0.01 over 20 epochs after LFC_START
        ramp = min(1.0, (epoch - LFC_START) / 20.0)
        beta_lfc = 0.01 * ramp

    # --- Generate stochastic mini-bulk training data for this epoch ---
    model.train()
    epoch_losses = {"mse": [], "lfc": [], "cosine": [], "total": []}

    if use_ot:
        X_ctrl_epoch, X_stim_epoch, ct_epoch_str, donor_epoch = generate_stochastic_minibulk(SEED + epoch)
        ct_epoch = CellTypeEmbedding.encode_cell_types(ct_epoch_str.tolist())
    else:
        # Fallback to deterministic pseudo-bulk
        train_idx_list = [i for i in range(n_pairs) if donors_arr[i] in train_donors]
        X_ctrl_epoch = X_ctrl[train_idx_list]
        X_stim_epoch = X_stim[train_idx_list]
        ct_epoch = ct_indices[train_idx_list]

    n_epoch = len(X_ctrl_epoch)

    # Shuffle training order
    perm = np.random.RandomState(SEED + epoch).permutation(n_epoch)
    X_ctrl_epoch = X_ctrl_epoch[perm]
    X_stim_epoch = X_stim_epoch[perm]
    ct_epoch = ct_epoch[perm]

    # Process in mini-batches
    for start in range(0, n_epoch, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_epoch)

        ctrl_b = torch.tensor(X_ctrl_epoch[start:end], dtype=torch.float32).to(device)
        stim_b = torch.tensor(X_stim_epoch[start:end], dtype=torch.float32).to(device)
        ct_b = ct_epoch[start:end].to(device)

        # Phase 2: dataset-kind dispatch. Kang (and every dataset currently
        # wired into this trainer) is paired ctrl/stim → INTERVENTIONAL.
        batch_kind = DatasetKind.INTERVENTIONAL
        assert batch_kind != DatasetKind.OBSERVATIONAL, (
            "Observational batch leaked into train_week3 — aborting. "
            "Phase 2 does not yet support observational datasets."
        )
        stage = _KIND_TO_STAGE[batch_kind]

        # Residual learning: model predicts perturbation delta, add to ctrl.
        # This leverages the high ctrl-stim correlation (r=0.875 at mini-bulk)
        # so the model only needs to learn the perturbation effect, not
        # replicate baseline expression from scratch.
        predicted_delta = model.forward_batch(ctrl_b, edge_index, pert_id_stim, ct_b)
        predicted = (ctrl_b + predicted_delta).clamp(min=0.0)

        # stage=="joint" is the only reachable branch in Phase 2 and matches
        # pre-phase behavior bit-exactly; combined_loss is stage-agnostic.
        assert stage == "joint", f"Unexpected stage in Phase 2: {stage}"
        loss, breakdown = combined_loss(
            predicted=predicted,
            actual_stim=stim_b,
            actual_ctrl=ctrl_b,
            alpha=1.0, beta=beta_lfc, gamma=0.1,
        )

        optimizer.zero_grad()
        loss.backward()

        # Phase 3 gradient-isolation guard: under pretrain stage (only
        # reached for OBSERVATIONAL batches), the causal parameter matrix
        # NeumannPropagation.W must not receive gradient. Default on;
        # disable by setting AIVC_GRAD_GUARD=0.
        # Phase 6 tripwire: guard must stay silent at stage="joint" with
        # pretrained weights. The branch is gated EXPLICITLY on
        # stage == "pretrain" (not on W.grad state) so it cannot misfire
        # at stage="joint" where nonzero W.grad is expected and required.
        if (
            os.environ.get("AIVC_GRAD_GUARD", "1") == "1"
            and stage == "pretrain"
            and hasattr(model, "neumann")
            and model.neumann is not None
        ):
            W_grad = model.neumann.W.grad
            if W_grad is not None and not torch.allclose(
                W_grad, torch.zeros_like(W_grad)
            ):
                raise RuntimeError(
                    "AIVC_GRAD_GUARD: NeumannPropagation.W received nonzero "
                    f"gradient under pretrain stage "
                    f"(max|grad|={W_grad.abs().max().item():.3e}). "
                    "Some term is leaking interventional signal into the "
                    "causal matrix from an observational batch. Investigate "
                    "before proceeding to Phase 4 (suspect: shared BN state "
                    "or auxiliary referencing W)."
                )

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in epoch_losses:
            epoch_losses[k].append(breakdown[k])

    if scheduler is not None:
        scheduler.step()

    avg_loss = {k: np.mean(v) for k, v in epoch_losses.items()}

    # --- Validate on fixed subset ---
    model.eval()
    val_preds = []
    val_actuals = []
    with torch.no_grad():
        for start in range(0, len(val_ds), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(val_ds))
            batch_indices = list(range(start, end))
            ctrl_b = torch.stack([val_ds[i]["ctrl"] for i in batch_indices]).to(device)
            stim_b = torch.stack([val_ds[i]["stim"] for i in batch_indices]).to(device)
            ct_b = torch.stack([val_ds[i]["cell_type_id"] for i in batch_indices]).to(device)
            pred_delta = model.forward_batch(ctrl_b, edge_index, pert_id_stim, ct_b)
            pred = (ctrl_b + pred_delta).clamp(min=0.0)
            val_preds.append(pred.cpu())
            val_actuals.append(stim_b.cpu())

    val_preds = torch.cat(val_preds, dim=0)
    val_actuals = torch.cat(val_actuals, dim=0)
    val_r_mean, val_r_std = compute_pearson_r(val_preds, val_actuals)

    # Track best
    if val_r_mean > best_val_r:
        best_val_r = val_r_mean
        best_epoch = epoch
        torch.save(model.state_dict(), "model_week3_best.pt")
        no_improve_count = 0
    else:
        no_improve_count += 1

    log_line = (
        f"Epoch {epoch:3d} | MSE={avg_loss['mse']:.4f} | LFC={avg_loss['lfc']:.4f} | "
        f"Cos={avg_loss['cosine']:.4f} | Total={avg_loss['total']:.4f} | "
        f"val_r={val_r_mean:.4f}+/-{val_r_std:.4f}"
    )
    loss_log.append(log_line)

    # --- Logging every 10 epochs ---
    if epoch == 1 or epoch % 10 == 0:
        print(f"  {log_line}")

    # --- JAK-STAT monitoring every 25 epochs ---
    if epoch % 25 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            # Use first val pair
            if len(val_ds) > 0:
                sample = val_ds[0]
                ctrl_s = sample["ctrl"].to(device)
                stim_s = sample["stim"]
                ct_s = sample["cell_type_id"].unsqueeze(0).to(device)
                pred_delta = model(ctrl_s, edge_index, pert_id_stim, ct_s).cpu()
                pred_s = (ctrl_s.cpu() + pred_delta).clamp(min=0.0).numpy()
                ctrl_s_np = sample["ctrl"].numpy()
                stim_s_np = stim_s.numpy()

                eps = 1e-6
                print(f"\n    JAK-STAT LFC monitoring (epoch {epoch}):")
                print(f"    {'Gene':<10} {'Actual LFC':<12} {'Pred LFC':<12} {'Error':<10} {'Top-50 DE?'}")
                print(f"    {'-'*55}")

                # Get top-50 predicted DE genes
                pred_delta = np.abs(pred_s - ctrl_s_np)
                top50_pred = set(np.argsort(pred_delta)[-50:])

                for g in jakstat_monitor:
                    idx = jakstat_idx.get(g)
                    if idx is None:
                        continue
                    actual_lfc = np.log2(max(stim_s_np[idx], 0) + eps) - np.log2(max(ctrl_s_np[idx], 0) + eps)
                    pred_lfc = np.log2(max(pred_s[idx], 0) + eps) - np.log2(max(ctrl_s_np[idx], 0) + eps)
                    error = abs(pred_lfc - actual_lfc)
                    in_top50 = "YES" if idx in top50_pred else "NO"
                    print(f"    {g:<10} {actual_lfc:<12.2f} {pred_lfc:<12.2f} {error:<10.2f} {in_top50}")

                    # IFIT1 milestone check
                    if g == "IFIT1" and error < 1.0 and ifit1_recovered_epoch is None:
                        ifit1_recovered_epoch = epoch
                        print(f"\n    *** IFIT1 RECOVERED at epoch {epoch}! ***")
                        print(f"    *** Predicted LFC within 1 log2 unit of actual ***\n")

                print()

    # --- Checkpoint every 50 epochs ---
    if epoch % 50 == 0:
        ckpt_path = f"model_week3_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

    # --- Early stopping ---
    if no_improve_count >= PATIENCE and epoch > LFC_START:
        print(f"\n  Early stopping at epoch {epoch}. Best val r: {best_val_r:.4f} at epoch {best_epoch}")
        torch.save(model.state_dict(), "model_week3_early_stop.pt")
        early_stopped = True
        break

# =========================================================================
# 7. Save final model and log
# =========================================================================
if not early_stopped:
    torch.save(model.state_dict(), f"model_week3_epoch{NUM_EPOCHS}.pt")
    print(f"\nFinal model saved: model_week3_epoch{NUM_EPOCHS}.pt")

# Also copy best to a standard name
import shutil
if os.path.exists("model_week3_best.pt"):
    shutil.copy2("model_week3_best.pt", "model_week3.pt")
    print(f"Best model (epoch {best_epoch}) copied to: model_week3.pt")

with open("training_log_week3.txt", "w") as f:
    f.write("\n".join(loss_log))
print("Training log saved: training_log_week3.txt")

# =========================================================================
# 8. Final test evaluation
# =========================================================================
print(f"\n{'='*80}")
print("FINAL EVALUATION ON HELD-OUT TEST SET")
print(f"{'='*80}")

# Load best model
model.load_state_dict(torch.load("model_week3_best.pt", map_location=device))
model.eval()

test_preds = []
test_actuals = []
test_ctrls = []
test_cts = []

with torch.no_grad():
    for start in range(0, len(test_ds), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(test_ds))
        batch_indices = list(range(start, end))
        ctrl_b = torch.stack([test_ds[i]["ctrl"] for i in batch_indices]).to(device)
        stim_b = torch.stack([test_ds[i]["stim"] for i in batch_indices]).to(device)
        ct_b = torch.stack([test_ds[i]["cell_type_id"] for i in batch_indices]).to(device)
        pred_delta = model.forward_batch(ctrl_b, edge_index, pert_id_stim, ct_b)
        pred = (ctrl_b + pred_delta).clamp(min=0.0)
        test_preds.append(pred.cpu())
        test_actuals.append(stim_b.cpu())
        test_ctrls.append(ctrl_b.cpu())
        test_cts.append(ct_b.cpu())

test_preds = torch.cat(test_preds, dim=0)
test_actuals = torch.cat(test_actuals, dim=0)
test_ctrls = torch.cat(test_ctrls, dim=0)
test_cts = torch.cat(test_cts, dim=0)

test_r_mean, test_r_std = compute_pearson_r(test_preds, test_actuals)
test_mse = F.mse_loss(test_preds, test_actuals).item()

print(f"  Test Pearson r: {test_r_mean:.4f} +/- {test_r_std:.4f}")
print(f"  Test MSE:       {test_mse:.4f}")
print(f"  Best val r:     {best_val_r:.4f} at epoch {best_epoch}")
print(f"  Early stopped:  {early_stopped}")
if ifit1_recovered_epoch:
    print(f"  IFIT1 recovered: epoch {ifit1_recovered_epoch}")
else:
    print(f"  IFIT1 recovered: NOT YET")

# JAK-STAT fold changes on test set
print(f"\n  JAK-STAT fold changes (test set):")
print(f"  {'Gene':<10} {'Pred FC':<10} {'Actual FC':<10} {'Error'}")
print(f"  {'-'*40}")
pred_np = test_preds.numpy()
actual_np = test_actuals.numpy()
ctrl_np = test_ctrls.numpy()
eps = 1e-6
for g in sorted(jakstat_idx.keys()):
    idx = jakstat_idx[g]
    ctrl_mean = ctrl_np[:, idx].mean()
    pred_mean = pred_np[:, idx].mean()
    actual_mean = actual_np[:, idx].mean()
    pfc = (pred_mean + eps) / (ctrl_mean + eps)
    afc = (actual_mean + eps) / (ctrl_mean + eps)
    err = abs(pfc - afc)
    print(f"  {g:<10} {pfc:<10.2f} {afc:<10.2f} {err:.2f}")

# Verdict
if test_r_mean >= 0.80:
    print(f"\n  DEMO READY. Pearson r = {test_r_mean:.3f} exceeds 0.80 target.")
elif test_r_mean >= 0.70:
    print(f"\n  APPROACHING BENCHMARK. r = {test_r_mean:.3f}. One more week needed.")
elif test_r_mean >= 0.60:
    print(f"\n  BELOW TARGET but learning. r = {test_r_mean:.3f}.")
else:
    print(f"\n  BELOW 0.60. Needs diagnosis. r = {test_r_mean:.3f}.")

# Save test info
test_info = {
    "test_donors": sorted(test_donors),
    "val_donors": sorted(val_donors),
    "train_donors": sorted(train_donors),
    "test_idx": test_idx,
    "val_idx": val_idx,
    "test_r_mean": test_r_mean,
    "test_r_std": test_r_std,
    "best_val_r": best_val_r,
    "best_epoch": best_epoch,
    "early_stopped": early_stopped,
    "ifit1_recovered_epoch": ifit1_recovered_epoch,
    "use_ot": use_ot,
    "n_train_pairs": len(train_idx),
    "n_val_pairs": len(val_idx),
    "n_test_pairs": len(test_idx),
}
torch.save(test_info, "test_split_info_week3.pt")
print(f"\n  Test info saved: test_split_info_week3.pt")
