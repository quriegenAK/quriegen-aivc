"""
evaluate_model.py — Full evaluation and benchmarking of the perturbation model.

Evaluates the trained PerturbationPredictor on held-out test donors.
Computes Pearson r, R², DE gene recovery, cell-type stratified metrics,
and failure mode analysis.

Outputs:
    Benchmark comparison table (CPA, scGEN, AIVC)
    Cell-type stratified Pearson r
    Top 50 DE gene recovery
    Failure mode analysis (worst-predicted genes)
"""
import random
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import anndata as ad
from sklearn.metrics import r2_score

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

from perturbation_model import PerturbationPredictor

# ----------------------------
# 1. Load data
# ----------------------------
print("Loading data...")
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

# JAK-STAT genes
jakstat_genes = [
    "JAK1", "JAK2", "STAT1", "STAT2", "STAT3",
    "IRF9", "IRF1", "MX1", "MX2", "ISG15",
    "OAS1", "IFIT1", "IFIT3", "IFITM1", "IFITM3",
]
jakstat_idx = {g: gene_to_idx[g] for g in jakstat_genes if g in gene_to_idx}

# ----------------------------
# 2. Split by donor (same as training)
# ----------------------------
donors = sorted(paired_manifest["donor_id"].unique().tolist())
test_donors = set(donors[7:8])
val_donors = set(donors[6:7])

test_idx = [i for i, d in enumerate(paired_manifest["donor_id"]) if d in test_donors]
val_idx = [i for i, d in enumerate(paired_manifest["donor_id"]) if d in val_donors]

# Also get cell types for each test pair
test_cell_types = [paired_manifest.iloc[i]["cell_type"] for i in test_idx]
val_cell_types = [paired_manifest.iloc[i]["cell_type"] for i in val_idx]

print(f"  Test donors: {sorted(test_donors)}")
print(f"  Test pairs: {len(test_idx)}")

X_ctrl_test = torch.tensor(X_ctrl[test_idx], dtype=torch.float32)
X_stim_test = torch.tensor(X_stim[test_idx], dtype=torch.float32)

# ----------------------------
# 3. Load model
# ----------------------------
device = torch.device("cpu")
model = PerturbationPredictor(
    n_genes=n_genes, num_perturbations=2, feature_dim=64,
    hidden1_dim=64, hidden2_dim=32, num_head1=3, num_head2=2,
    decoder_hidden=256,
).to(device)

model.load_state_dict(torch.load("model_perturbation.pt", map_location=device))
model.eval()
print("  Model loaded from model_perturbation.pt")


# ----------------------------
# 4. Pearson r computation
# ----------------------------
def compute_pearson_r(predicted, actual):
    """Compute mean Pearson correlation across pairs (gene-wise per pair).

    Args:
        predicted: (n_pairs, n_genes) numpy array.
        actual:    (n_pairs, n_genes) numpy array.

    Returns:
        (mean_r, std_r, per_pair_rs) — mean, std, and list of per-pair Pearson r.
    """
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
    return float(np.mean(rs)), float(np.std(rs)), rs


# ----------------------------
# 5.1 Held-out test evaluation
# ----------------------------
print(f"\n{'='*80}")
print("HELD-OUT TEST EVALUATION")
print(f"{'='*80}")

with torch.no_grad():
    pert_stim = torch.tensor([1])
    pred_test = model.forward_batch(X_ctrl_test, edge_index, pert_stim)

pred_np = pred_test.cpu().numpy()
actual_np = X_stim_test.cpu().numpy()

test_r_mean, test_r_std, test_rs = compute_pearson_r(pred_np, actual_np)
test_r2 = r2_score(actual_np.flatten(), pred_np.flatten())

print(f"  Mean Pearson r: {test_r_mean:.4f} ± {test_r_std:.4f}")
print(f"  R² score:       {test_r2:.4f}")

# Top 50 DE gene recovery
actual_fc = actual_np.mean(axis=0) / np.maximum(X_ctrl[test_idx].mean(axis=0), 0.001)
pred_fc = pred_np.mean(axis=0) / np.maximum(X_ctrl[test_idx].mean(axis=0), 0.001)

actual_de = np.abs(actual_np.mean(axis=0) - X_ctrl[test_idx].mean(axis=0))
actual_top50 = set(np.argsort(actual_de)[-50:])
pred_de = np.abs(pred_np.mean(axis=0) - X_ctrl[test_idx].mean(axis=0))
pred_top50 = set(np.argsort(pred_de)[-50:])

top50_recall = len(actual_top50 & pred_top50) / 50
top200_actual = set(np.argsort(actual_de)[-200:])
top200_pred = set(np.argsort(pred_de)[-200:])
top200_recall = len(top200_actual & top200_pred) / 200

print(f"  Top 50 DE gene recall:  {top50_recall:.2%} ({len(actual_top50 & pred_top50)}/50)")
print(f"  Top 200 DE gene recall: {top200_recall:.2%} ({len(top200_actual & top200_pred)}/200)")

# JAK-STAT specific metrics
print(f"\n  JAK-STAT Gene Predictions (test set average):")
print(f"  {'Gene':<10} {'Pred FC':<10} {'Actual FC':<10} {'Abs Error'}")
print(f"  {'-'*42}")
for g in sorted(jakstat_idx.keys()):
    idx = jakstat_idx[g]
    ctrl_mean_g = X_ctrl[test_idx][:, idx].mean()
    pred_mean_g = pred_np[:, idx].mean()
    actual_mean_g = actual_np[:, idx].mean()
    pfc = pred_mean_g / max(ctrl_mean_g, 0.001)
    afc = actual_mean_g / max(ctrl_mean_g, 0.001)
    err = abs(pfc - afc)
    print(f"  {g:<10} {pfc:<10.2f} {afc:<10.2f} {err:.2f}")

# ----------------------------
# 5.2 Cell-type stratified evaluation
# ----------------------------
print(f"\n{'='*80}")
print("CELL-TYPE STRATIFIED EVALUATION")
print(f"{'='*80}")

all_cell_types = sorted(set(test_cell_types))
ct_results = {}
for ct in all_cell_types:
    ct_idx = [i for i, c in enumerate(test_cell_types) if c == ct]
    if len(ct_idx) == 0:
        continue
    ct_pred = pred_np[ct_idx]
    ct_actual = actual_np[ct_idx]
    ct_r, ct_std, _ = compute_pearson_r(ct_pred, ct_actual)
    ct_results[ct] = ct_r
    above_target = "ABOVE" if ct_r >= 0.80 else "below"
    print(f"  {ct:<24} r={ct_r:.4f}  ({above_target} 0.80 target)")

# If test set has only one donor, it may have few cell types
if not ct_results:
    # Fall back to using val set for stratification
    print("  (No stratified data in test set, using validation set)")
    X_ctrl_val = torch.tensor(X_ctrl[val_idx], dtype=torch.float32)
    X_stim_val = torch.tensor(X_stim[val_idx], dtype=torch.float32)
    with torch.no_grad():
        pred_val = model.forward_batch(X_ctrl_val, edge_index, pert_stim)
    pred_val_np = pred_val.cpu().numpy()
    actual_val_np = X_stim_val.cpu().numpy()

    all_cell_types = sorted(set(val_cell_types))
    for ct in all_cell_types:
        ct_idx_v = [i for i, c in enumerate(val_cell_types) if c == ct]
        if len(ct_idx_v) == 0:
            continue
        ct_pred = pred_val_np[ct_idx_v]
        ct_actual = actual_val_np[ct_idx_v]
        ct_r, _, _ = compute_pearson_r(ct_pred, ct_actual)
        ct_results[ct] = ct_r
        above_target = "ABOVE" if ct_r >= 0.80 else "below"
        print(f"  {ct:<24} r={ct_r:.4f}  ({above_target} 0.80 target)")

if ct_results:
    weakest_ct = min(ct_results, key=ct_results.get)
    print(f"\n  Weakest cell type: {weakest_ct} (r={ct_results[weakest_ct]:.4f})")

# ----------------------------
# 5.3 Benchmark comparison table
# ----------------------------
print(f"\n{'='*80}")
print("BENCHMARK COMPARISON")
print(f"{'='*80}")
print("""
  +----------------------+----------------+---------------+-----------------+
  | Model                | Pearson r mean | Pearson r std | Top50 DE recall |
  +----------------------+----------------+---------------+-----------------+
  | scGEN (published)    | 0.820          | --            | --              |
  | CPA (published)      | 0.856          | --            | --              |
  | AIVC Week 1 baseline | 0.000 (no pert)| --            | --              |
  | AIVC Week 2 (ours)   | {r_mean:<14.3f} | {r_std:<13.3f} | {de_recall:<15.1%} |
  +----------------------+----------------+---------------+-----------------+
""".format(r_mean=test_r_mean, r_std=test_r_std, de_recall=top50_recall))

if test_r_mean > 0.80:
    print("  ** Demo-ready. Pearson r exceeds 0.80 target. **")
elif test_r_mean > 0.70:
    print("  Above minimum threshold. Improvement needed for investor demo.")
    print("  Recommendation: increase training epochs, tune LR, or add cell-type embedding.")
elif test_r_mean < 0.70:
    print(f"  Below 0.70 target. r={test_r_mean:.3f}")
    print("  Analysis: The model with 60 pseudo-bulk pairs and a gene-graph architecture")
    print("  operates differently from CPA/scGEN which use single-cell latent spaces.")
    print("  Week 3 options: (1) add cell-type embedding, (2) use single-cell training,")
    print("  (3) increase feature_dim, (4) add skip connections in decoder.")

# ----------------------------
# 5.4 Failure mode analysis
# ----------------------------
print(f"\n{'='*80}")
print("FAILURE MODE ANALYSIS — 10 worst-predicted genes")
print(f"{'='*80}")

# Mean absolute error per gene
gene_mae = np.abs(pred_np - actual_np).mean(axis=0)
worst_10_idx = np.argsort(gene_mae)[-10:][::-1]

print(f"  {'Rank':<6} {'Gene':<14} {'MAE':<10} {'Actual FC':<12} {'Pred FC':<12} {'Likely reason'}")
print("-" * 80)

# Count edges per gene for diagnosis
edge_count = np.zeros(n_genes)
edges_np = edge_index.numpy()
for e in range(edges_np.shape[1]):
    edge_count[edges_np[0, e]] += 1
    edge_count[edges_np[1, e]] += 1

for rank, idx in enumerate(worst_10_idx):
    gene = gene_names[idx]
    mae = gene_mae[idx]
    ctrl_val = X_ctrl[test_idx][:, idx].mean()
    actual_val = actual_np[:, idx].mean()
    pred_val = pred_np[:, idx].mean()

    afc = actual_val / max(ctrl_val, 0.001)
    pfc = pred_val / max(ctrl_val, 0.001)

    n_edges_g = int(edge_count[idx])
    ctrl_std = X_ctrl[test_idx][:, idx].std()

    if n_edges_g < 3:
        reason = "few graph edges"
    elif ctrl_val < 0.01:
        reason = "low expression"
    elif ctrl_std > ctrl_val:
        reason = "high variance"
    elif abs(afc) > 10:
        reason = "extreme fold change"
    else:
        reason = "model capacity"

    print(f"  {rank+1:<6} {gene:<14} {mae:<10.4f} {afc:<12.2f} {pfc:<12.2f} {reason}")

print(f"\nOVERALL: Test Pearson r = {test_r_mean:.4f} ± {test_r_std:.4f}")
