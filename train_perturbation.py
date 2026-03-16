"""
train_perturbation.py — Train the perturbation response prediction model.

Trains PerturbationPredictor to predict stimulated gene expression from
control expression + perturbation identity on Kang 2018 PBMC data.

Training strategy:
    Phase 1 (epochs 1-50):  Train PerturbationEmbedding + ResponseDecoder only.
                            GeneLink GAT weights are frozen for stability.
    Phase 2 (epochs 51-200): Unfreeze GeneLink for fine-tuning at lower LR.

Loss: MSE + 0.1 * cosine similarity loss.
Split: by donor (no data leakage). Donors 1-6 train, 7 val, 8 test.

Outputs:
    model_perturbation.pt                — final model checkpoint
    model_perturbation_epoch{N}.pt       — intermediate checkpoints every 50 epochs
    training_log_perturbation.txt        — full training log
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import anndata as ad

# Reproducibility — set all four seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

from perturbation_model import PerturbationPredictor

# ----------------------------
# 0. Load data
# ----------------------------
print("Loading data...")
adata = ad.read_h5ad("data/kang2018_pbmc_fixed.h5ad")

# Gene names
if "name" in adata.var.columns:
    gene_names = adata.var["name"].tolist()
else:
    gene_names = adata.var_names.tolist()
n_genes = len(gene_names)
gene_to_idx = {g: i for i, g in enumerate(gene_names)}

# Load paired expression matrices
X_ctrl = np.load("data/X_ctrl_paired.npy")  # (n_pairs, n_genes)
X_stim = np.load("data/X_stim_paired.npy")  # (n_pairs, n_genes)
n_pairs = X_ctrl.shape[0]
print(f"  Loaded {n_pairs} pseudo-bulk pairs, {n_genes} genes")

# Load pairing manifest for donor-based splitting
manifest = pd.read_csv("data/pairing_manifest.csv")
paired_manifest = manifest[manifest["paired"]].reset_index(drop=True)
assert len(paired_manifest) == n_pairs, \
    f"Manifest paired rows ({len(paired_manifest)}) != data pairs ({n_pairs})"

# Load edge list
edge_df = pd.read_csv("data/edge_list_fixed.csv")
edges = []
for _, row in edge_df.iterrows():
    a = gene_to_idx.get(row["gene_a"])
    b = gene_to_idx.get(row["gene_b"])
    if a is not None and b is not None:
        edges.append([a, b])
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
print(f"  Edge index: {edge_index.shape[1]} edges")

# ----------------------------
# 1. Split by donor
# ----------------------------
donors = sorted(paired_manifest["donor_id"].unique().tolist())
print(f"  Donors ({len(donors)}): {donors}")

# Assign: first 6 donors for training, 7th for validation, 8th for test
train_donors = set(donors[:6])
val_donors = set(donors[6:7])
test_donors = set(donors[7:8])

print(f"  Train donors ({len(train_donors)}): {sorted(train_donors)}")
print(f"  Val donors   ({len(val_donors)}): {sorted(val_donors)}")
print(f"  Test donors  ({len(test_donors)}): {sorted(test_donors)}")

train_idx = [i for i, d in enumerate(paired_manifest["donor_id"]) if d in train_donors]
val_idx = [i for i, d in enumerate(paired_manifest["donor_id"]) if d in val_donors]
test_idx = [i for i, d in enumerate(paired_manifest["donor_id"]) if d in test_donors]

print(f"  Train pairs: {len(train_idx)}, Val pairs: {len(val_idx)}, Test pairs: {len(test_idx)}")

# Convert to tensors
X_ctrl_t = torch.tensor(X_ctrl, dtype=torch.float32)
X_stim_t = torch.tensor(X_stim, dtype=torch.float32)

X_ctrl_train = X_ctrl_t[train_idx]
X_stim_train = X_stim_t[train_idx]
X_ctrl_val = X_ctrl_t[val_idx]
X_stim_val = X_stim_t[val_idx]
X_ctrl_test = X_ctrl_t[test_idx]
X_stim_test = X_stim_t[test_idx]

# ----------------------------
# 2. Pearson r computation
# ----------------------------
def compute_pearson_r(predicted, actual):
    """Compute mean Pearson correlation across pairs (gene-wise per pair).

    For each (predicted, actual) pair, computes Pearson r across all genes.
    Then averages across pairs. This is the CPA benchmark metric.

    Args:
        predicted: (n_pairs, n_genes) — predicted expression.
        actual:    (n_pairs, n_genes) — actual expression.

    Returns:
        (mean_r, std_r) — mean and std Pearson r across pairs.
    """
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()

    rs = []
    for i in range(predicted.shape[0]):
        p = predicted[i]
        a = actual[i]
        # Handle constant predictions
        if np.std(p) < 1e-10 or np.std(a) < 1e-10:
            rs.append(0.0)
            continue
        r = np.corrcoef(p, a)[0, 1]
        if np.isnan(r):
            r = 0.0
        rs.append(r)
    return float(np.mean(rs)), float(np.std(rs))


# ----------------------------
# 3. JAK-STAT gene indices
# ----------------------------
jakstat_genes = [
    "JAK1", "JAK2", "STAT1", "STAT2", "STAT3",
    "IRF9", "IRF1", "MX1", "MX2", "ISG15",
    "OAS1", "IFIT1", "IFIT3", "IFITM1", "IFITM3",
]
jakstat_idx = {g: gene_to_idx[g] for g in jakstat_genes if g in gene_to_idx}
print(f"  JAK-STAT genes tracked: {len(jakstat_idx)}/{len(jakstat_genes)}")


# ----------------------------
# 4. Model setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

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

edge_index = edge_index.to(device)
pert_id_stim = torch.tensor([1], device=device)  # stim perturbation

total_params = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {total_params:,}")

# ----------------------------
# 5. Training loop
# ----------------------------
NUM_EPOCHS = 200
PHASE1_END = 50

loss_log = []
best_val_r = -1.0
best_epoch = 0
overfit_count = 0

print(f"\n{'='*80}")
print("TRAINING START — {NUM_EPOCHS} epochs, Phase 1 frozen GeneLink (1-50), Phase 2 fine-tune (51-200)")
print(f"{'='*80}")

for epoch in range(1, NUM_EPOCHS + 1):
    # Phase management
    if epoch == 1:
        # Phase 1: freeze GeneLink, train embedding + decoder
        for p in model.genelink.parameters():
            p.requires_grad = False
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3, weight_decay=1e-5,
        )
        print("  Phase 1: GeneLink frozen, training embedding + decoder (lr=1e-3)")

    elif epoch == PHASE1_END + 1:
        # Phase 2: unfreeze GeneLink with lower LR
        for p in model.genelink.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam([
            {"params": model.feature_expander.parameters(), "lr": 1e-3},
            {"params": model.pert_embedding.parameters(), "lr": 1e-3},
            {"params": model.decoder.parameters(), "lr": 1e-3},
            {"params": model.genelink.parameters(), "lr": 1e-4},
        ], weight_decay=1e-5)
        print("  Phase 2: GeneLink unfrozen, fine-tuning (GeneLink lr=1e-4)")

    # --- Train ---
    model.train()
    X_ctrl_dev = X_ctrl_train.to(device)
    X_stim_dev = X_stim_train.to(device)

    predicted = model.forward_batch(X_ctrl_dev, edge_index, pert_id_stim)

    loss_mse = F.mse_loss(predicted, X_stim_dev)
    loss_cos = 1.0 - F.cosine_similarity(predicted, X_stim_dev, dim=1).mean()
    loss = loss_mse + 0.1 * loss_cos

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    train_loss = loss.item()

    # --- Validate ---
    model.eval()
    with torch.no_grad():
        X_ctrl_v = X_ctrl_val.to(device)
        X_stim_v = X_stim_val.to(device)
        pred_val = model.forward_batch(X_ctrl_v, edge_index, pert_id_stim)
        val_mse = F.mse_loss(pred_val, X_stim_v)
        val_cos = 1.0 - F.cosine_similarity(pred_val, X_stim_v, dim=1).mean()
        val_loss = (val_mse + 0.1 * val_cos).item()

    val_r_mean, val_r_std = compute_pearson_r(pred_val, X_stim_v)

    if val_r_mean > best_val_r:
        best_val_r = val_r_mean
        best_epoch = epoch

    log_line = f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_r={val_r_mean:.4f}±{val_r_std:.4f}"
    loss_log.append(log_line)

    # --- Logging ---
    if epoch == 1 or epoch % 10 == 0:
        print(f"  {log_line}")

        # Top 5 genes by absolute prediction error (on val set)
        if pred_val.shape[0] > 0:
            mean_abs_err = (pred_val - X_stim_v).abs().mean(dim=0)
            top5_err = mean_abs_err.argsort(descending=True)[:5]
            err_genes = [(gene_names[i.item()], mean_abs_err[i.item()].item()) for i in top5_err]
            print(f"    Top 5 error genes: {', '.join(f'{g}({e:.3f})' for g, e in err_genes)}")

    # --- Overfitting detection ---
    if epoch % 50 == 0:
        if train_loss > 0 and val_loss / train_loss > 1.2:
            overfit_count += 1
            if overfit_count >= 3:
                print(f"  WARNING: Possible overfitting at epoch {epoch}.")
                print(f"    Train loss: {train_loss:.4f}. Val loss: {val_loss:.4f}. Ratio: {val_loss/train_loss:.2f}")
                print(f"    Consider: reducing capacity, increasing dropout, or early stopping.")
        else:
            overfit_count = max(0, overfit_count - 1)

    # --- JAK-STAT monitoring ---
    if epoch % 50 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            # Use first val pair for JAK-STAT monitoring
            if X_ctrl_val.shape[0] > 0:
                ctrl_sample = X_ctrl_val[0:1].to(device)
                stim_sample = X_stim_val[0:1].to(device)
                pred_sample = model.forward_batch(ctrl_sample, edge_index, pert_id_stim)

                print(f"\n    JAK-STAT fold change monitoring (epoch {epoch}):")
                print(f"    {'Gene':<10} {'Pred FC':<12} {'Actual FC':<12} {'Error'}")
                print(f"    {'-'*46}")
                for g, idx in sorted(jakstat_idx.items()):
                    ctrl_val = ctrl_sample[0, idx].item()
                    pred_val_g = pred_sample[0, idx].item()
                    actual_val = stim_sample[0, idx].item()
                    pred_fc = pred_val_g / max(ctrl_val, 0.001)
                    actual_fc = actual_val / max(ctrl_val, 0.001)
                    err = abs(pred_fc - actual_fc)
                    print(f"    {g:<10} {pred_fc:<12.2f} {actual_fc:<12.2f} {err:.2f}")
                print()

    # --- Checkpoint ---
    if epoch % 50 == 0:
        ckpt_path = f"model_perturbation_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

# ----------------------------
# 6. Save final model and log
# ----------------------------
torch.save(model.state_dict(), "model_perturbation.pt")
print(f"\nFinal model saved: model_perturbation.pt")

with open("training_log_perturbation.txt", "w") as f:
    f.write("\n".join(loss_log))
print("Training log saved: training_log_perturbation.txt")

# ----------------------------
# 7. Final test evaluation
# ----------------------------
print(f"\n{'='*80}")
print("FINAL EVALUATION ON TEST SET")
print(f"{'='*80}")

model.eval()
with torch.no_grad():
    X_ctrl_te = X_ctrl_test.to(device)
    X_stim_te = X_stim_test.to(device)
    pred_test = model.forward_batch(X_ctrl_te, edge_index, pert_id_stim)

test_r_mean, test_r_std = compute_pearson_r(pred_test, X_stim_te)
test_mse = F.mse_loss(pred_test, X_stim_te).item()

print(f"  Test MSE:      {test_mse:.4f}")
print(f"  Test Pearson r: {test_r_mean:.4f} ± {test_r_std:.4f}")
print(f"  Best val r:     {best_val_r:.4f} at epoch {best_epoch}")

if test_r_mean > 0.80:
    print("\n  BENCHMARK: EXCEEDS CPA baseline (0.856 target). Demo-ready.")
elif test_r_mean > 0.70:
    print("\n  BENCHMARK: Above minimum threshold (0.70). Needs improvement for demo.")
elif test_r_mean > 0.60:
    print("\n  BENCHMARK: Below target but learning. Check data pairing and LR schedule.")
else:
    print("\n  BENCHMARK: Below 0.60. Investigate data pairing, loss function, or model capacity.")

# Save test donor info for evaluate_model.py
test_info = {
    "test_donors": sorted(test_donors),
    "test_idx": test_idx,
    "test_r_mean": test_r_mean,
    "test_r_std": test_r_std,
}
torch.save(test_info, "test_split_info.pt")
print("  Test split info saved: test_split_info.pt")
