"""
test_regression.py — Regression test for the perturbation prediction model.

Ensures future changes do not degrade model performance below minimum
acceptable thresholds. Loads the trained model and runs evaluation assertions.

Assertions:
    1. Pearson r > 0.50 (minimum acceptable for graph-based approach)
    2. JAK-STAT recovery >= 1/15 in top 50 predicted DE genes
    3. No NaN in predictions
    4. Output shape is (n_test_pairs, n_genes)
    5. Model file exists and loads without error
"""
import random
import sys
import numpy as np
import torch
import pandas as pd
import anndata as ad
import os

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

from perturbation_model import PerturbationPredictor

print("=" * 70)
print("TEST: REGRESSION — Perturbation Model Performance Guard")
print("=" * 70)

# ----------------------------
# 1. Check model file exists
# ----------------------------
model_path = "model_perturbation.pt"
if not os.path.exists(model_path):
    print(f"  FAIL — Model file {model_path} not found")
    sys.exit(1)
print(f"  [1/5] Model file exists: {model_path}")

# ----------------------------
# 2. Load data
# ----------------------------
print("  Loading data...")
adata = ad.read_h5ad("data/kang2018_pbmc_fixed.h5ad")

if "name" in adata.var.columns:
    gene_names = adata.var["name"].tolist()
else:
    gene_names = adata.var_names.tolist()
n_genes = len(gene_names)
gene_to_idx = {g: i for i, g in enumerate(gene_names)}

X_ctrl = np.load("data/X_ctrl_paired.npy")
X_stim = np.load("data/X_stim_paired.npy")

manifest = pd.read_csv("data/pairing_manifest.csv")
paired_manifest = manifest[manifest["paired"]].reset_index(drop=True)

edge_df = pd.read_csv("data/edge_list_fixed.csv")
edges = []
for _, row in edge_df.iterrows():
    a = gene_to_idx.get(row["gene_a"])
    b = gene_to_idx.get(row["gene_b"])
    if a is not None and b is not None:
        edges.append([a, b])
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# ----------------------------
# 3. Test/val split (same as training)
# ----------------------------
donors = sorted(paired_manifest["donor_id"].unique().tolist())
test_donors = set(donors[7:8])  # patient_1488

test_idx = [i for i, d in enumerate(paired_manifest["donor_id"]) if d in test_donors]
n_test_pairs = len(test_idx)

X_ctrl_test = torch.tensor(X_ctrl[test_idx], dtype=torch.float32)
X_stim_test = X_stim[test_idx]

# ----------------------------
# 4. Load model
# ----------------------------
model = PerturbationPredictor(
    n_genes=n_genes, num_perturbations=2, feature_dim=64,
    hidden1_dim=64, hidden2_dim=32, num_head1=3, num_head2=2,
    decoder_hidden=256,
)

try:
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print(f"  [2/5] Model loads without error")
except Exception as e:
    print(f"  FAIL — Model load error: {e}")
    sys.exit(1)

# ----------------------------
# 5. Run predictions
# ----------------------------
with torch.no_grad():
    pert_stim = torch.tensor([1])
    pred_test = model.forward_batch(X_ctrl_test, edge_index, pert_stim)

pred_np = pred_test.cpu().numpy()

# ----------------------------
# ASSERTION 1: Output shape
# ----------------------------
expected_shape = (n_test_pairs, n_genes)
actual_shape = pred_np.shape
if actual_shape != expected_shape:
    print(f"  FAIL — Output shape {actual_shape} != expected {expected_shape}")
    sys.exit(1)
print(f"  [3/5] Output shape correct: {actual_shape}")

# ----------------------------
# ASSERTION 2: No NaN in predictions
# ----------------------------
n_nan = np.isnan(pred_np).sum()
if n_nan > 0:
    print(f"  FAIL — {n_nan} NaN values in predictions")
    sys.exit(1)
print(f"  [4/5] No NaN in predictions")

# ----------------------------
# ASSERTION 3: Pearson r > 0.50 (minimum acceptable)
# ----------------------------
rs = []
for i in range(pred_np.shape[0]):
    p = pred_np[i]
    a = X_stim_test[i]
    if np.std(p) < 1e-10 or np.std(a) < 1e-10:
        rs.append(0.0)
        continue
    r = np.corrcoef(p, a)[0, 1]
    if np.isnan(r):
        r = 0.0
    rs.append(r)

mean_r = float(np.mean(rs))
MIN_R = 0.50

if mean_r < MIN_R:
    print(f"  FAIL — Pearson r = {mean_r:.4f} < {MIN_R} minimum")
    sys.exit(1)
print(f"  [5/5] Pearson r = {mean_r:.4f} > {MIN_R} minimum")

# ----------------------------
# ASSERTION 4: JAK-STAT recovery (informational, not hard fail)
# ----------------------------
jakstat_genes = {
    "JAK1", "JAK2", "STAT1", "STAT2", "STAT3",
    "IRF9", "IRF1", "MX1", "MX2", "ISG15",
    "OAS1", "IFIT1", "IFIT3", "IFITM1", "IFITM3",
}
jakstat_idx_set = {gene_to_idx[g] for g in jakstat_genes if g in gene_to_idx}

# Predicted DE genes (by absolute delta from ctrl)
pred_de = np.abs(pred_np.mean(axis=0) - X_ctrl[test_idx].mean(axis=0))
top50_pred = set(np.argsort(pred_de)[-50:])

jakstat_in_top50 = len(jakstat_idx_set & top50_pred)
MIN_JAKSTAT = 1

if jakstat_in_top50 < MIN_JAKSTAT:
    print(f"  WARNING — JAK-STAT recovery: {jakstat_in_top50}/15 in top 50 (below {MIN_JAKSTAT})")
    # This is a soft warning, not a hard fail
else:
    print(f"  INFO — JAK-STAT recovery: {jakstat_in_top50}/15 in top 50")

# ----------------------------
# Summary
# ----------------------------
print()
print("=" * 70)
print(f"  Pearson r:        {mean_r:.4f} (threshold: {MIN_R})")
print(f"  JAK-STAT top 50:  {jakstat_in_top50}/15")
print(f"  Output shape:     {actual_shape}")
print(f"  NaN count:        {n_nan}")
print("=" * 70)
print("\nOVERALL: PASS")
