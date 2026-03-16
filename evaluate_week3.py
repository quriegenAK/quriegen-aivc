"""
evaluate_week3.py — Full evaluation and benchmarking for Week 3 model.

Extends evaluate_model.py with:
    - Fold change accuracy (LFC error)
    - High-responder gene performance
    - Cell-type specificity metrics
    - Updated benchmark table

Outputs:
    Complete benchmark comparison table
    Cell-type stratified Pearson r and LFC error
    Fold change accuracy for JAK-STAT genes
"""
import random
import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import anndata as ad
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

from perturbation_model import PerturbationPredictor, CellTypeEmbedding

# =========================================================================
# 1. Load data
# =========================================================================
print("Loading data...")
adata = ad.read_h5ad("data/kang2018_pbmc_fixed.h5ad")

if "name" in adata.var.columns:
    gene_names = adata.var["name"].tolist()
else:
    gene_names = adata.var_names.tolist()
n_genes = len(gene_names)
gene_to_idx = {g: i for i, g in enumerate(gene_names)}

# Load OT or pseudo-bulk data
use_ot = False
if os.path.exists("data/X_ctrl_ot.npy"):
    X_ctrl = np.load("data/X_ctrl_ot.npy")
    X_stim = np.load("data/X_stim_ot.npy")
    cell_types_arr = np.load("data/cell_type_ot.npy", allow_pickle=True)
    donors_arr = np.load("data/donor_ot.npy", allow_pickle=True)
    if X_ctrl.shape[0] >= 200:
        use_ot = True
        print(f"  Using OT pairs: {X_ctrl.shape[0]} single-cell pairs")

        # Aggregate to mini-bulk (same as training)
        groups = {}
        for i in range(X_ctrl.shape[0]):
            key = (donors_arr[i], cell_types_arr[i])
            if key not in groups:
                groups[key] = {"ctrl": [], "stim": []}
            groups[key]["ctrl"].append(X_ctrl[i])
            groups[key]["stim"].append(X_stim[i])

        X_ctrl_bulk, X_stim_bulk, ct_bulk, donor_bulk = [], [], [], []
        for (d, ct) in sorted(groups.keys()):
            X_ctrl_bulk.append(np.mean(groups[(d, ct)]["ctrl"], axis=0))
            X_stim_bulk.append(np.mean(groups[(d, ct)]["stim"], axis=0))
            ct_bulk.append(ct)
            donor_bulk.append(d)

        X_ctrl = np.array(X_ctrl_bulk)
        X_stim = np.array(X_stim_bulk)
        cell_types_arr = np.array(ct_bulk)
        donors_arr = np.array(donor_bulk)
        print(f"  Aggregated to {X_ctrl.shape[0]} OT mini-bulk groups")
if not use_ot:
    X_ctrl = np.load("data/X_ctrl_paired.npy")
    X_stim = np.load("data/X_stim_paired.npy")
    manifest = pd.read_csv("data/pairing_manifest.csv")
    paired = manifest[manifest["paired"]].reset_index(drop=True)
    cell_types_arr = paired["cell_type"].values
    donors_arr = paired["donor_id"].values
    print(f"  Using pseudo-bulk pairs: {X_ctrl.shape[0]} pairs")

# Edge list
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

# =========================================================================
# 2. Split by donor (same as training)
# =========================================================================
unique_donors = sorted(np.unique(donors_arr).tolist())
n_donors = len(unique_donors)
n_train = max(1, int(n_donors * 0.6))
n_val = max(1, int(n_donors * 0.2))

train_donors = set(unique_donors[:n_train])
val_donors = set(unique_donors[n_train:n_train + n_val])
test_donors = set(unique_donors[n_train + n_val:])

test_idx = [i for i in range(len(donors_arr)) if donors_arr[i] in test_donors]
val_idx = [i for i in range(len(donors_arr)) if donors_arr[i] in val_donors]
eval_idx = test_idx if len(test_idx) > 0 else val_idx
eval_label = "test" if len(test_idx) > 0 else "val (fallback)"

print(f"  Test donors: {sorted(test_donors)}")
print(f"  Eval pairs: {len(eval_idx)} ({eval_label})")

# Cell type indices
ct_indices = CellTypeEmbedding.encode_cell_types(cell_types_arr.tolist())
eval_cell_types = [cell_types_arr[i] for i in eval_idx]

X_ctrl_eval = torch.tensor(X_ctrl[eval_idx], dtype=torch.float32)
X_stim_eval = torch.tensor(X_stim[eval_idx], dtype=torch.float32)
ct_eval = ct_indices[eval_idx]

# =========================================================================
# 3. Load model
# =========================================================================
device = torch.device("cpu")
model = PerturbationPredictor(
    n_genes=n_genes, num_perturbations=2, feature_dim=64,
    hidden1_dim=64, hidden2_dim=32, num_head1=3, num_head2=2,
    decoder_hidden=256,
).to(device)
model.cell_type_embedding = CellTypeEmbedding(
    num_cell_types=20, embedding_dim=model.feature_dim
).to(device)

model_path = "model_week3_best.pt" if os.path.exists("model_week3_best.pt") else "model_week3.pt"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"  Model loaded from {model_path}")

# =========================================================================
# 4. Predictions
# =========================================================================
with torch.no_grad():
    pert_stim = torch.tensor([1])
    # Residual learning: model predicts delta, add to ctrl
    pred_delta = model.forward_batch(X_ctrl_eval, edge_index, pert_stim, ct_eval)
    pred_eval = (X_ctrl_eval + pred_delta).clamp(min=0.0)

pred_np = pred_eval.cpu().numpy()
actual_np = X_stim_eval.numpy()
ctrl_np = X_ctrl[eval_idx]

# =========================================================================
# 5.1 Core metrics (same as Week 2)
# =========================================================================
def compute_pearson_r(predicted, actual):
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

print(f"\n{'='*80}")
print("HELD-OUT TEST EVALUATION")
print(f"{'='*80}")

test_r_mean, test_r_std, test_rs = compute_pearson_r(pred_np, actual_np)
test_r2 = r2_score(actual_np.flatten(), pred_np.flatten())

print(f"  Mean Pearson r: {test_r_mean:.4f} +/- {test_r_std:.4f}")
print(f"  R2 score:       {test_r2:.4f}")

# Top 50 DE gene recall
actual_de = np.abs(actual_np.mean(axis=0) - ctrl_np.mean(axis=0))
pred_de = np.abs(pred_np.mean(axis=0) - ctrl_np.mean(axis=0))
actual_top50 = set(np.argsort(actual_de)[-50:])
pred_top50 = set(np.argsort(pred_de)[-50:])
top50_recall = len(actual_top50 & pred_top50) / 50

actual_top200 = set(np.argsort(actual_de)[-200:])
pred_top200 = set(np.argsort(pred_de)[-200:])
top200_recall = len(actual_top200 & pred_top200) / 200

print(f"  Top 50 DE gene recall:  {top50_recall:.2%} ({len(actual_top50 & pred_top50)}/50)")
print(f"  Top 200 DE gene recall: {top200_recall:.2%} ({len(actual_top200 & pred_top200)}/200)")

# JAK-STAT predictions
print(f"\n  JAK-STAT Gene Predictions ({eval_label}):")
print(f"  {'Gene':<10} {'Pred FC':<10} {'Actual FC':<10} {'Abs Error'}")
print(f"  {'-'*42}")
for g in sorted(jakstat_idx.keys()):
    idx = jakstat_idx[g]
    c_mean = ctrl_np[:, idx].mean()
    p_mean = pred_np[:, idx].mean()
    a_mean = actual_np[:, idx].mean()
    pfc = (p_mean + 1e-6) / (c_mean + 1e-6)
    afc = (a_mean + 1e-6) / (c_mean + 1e-6)
    err = abs(pfc - afc)
    print(f"  {g:<10} {pfc:<10.2f} {afc:<10.2f} {err:.2f}")

# =========================================================================
# 5.2 Fold change accuracy (new in Week 3)
# =========================================================================
print(f"\n{'='*80}")
print("FOLD CHANGE ACCURACY (LFC)")
print(f"{'='*80}")

eps = 1e-6
actual_lfc = np.log2(actual_np + eps) - np.log2(ctrl_np + eps)
pred_lfc = np.log2(pred_np + eps) - np.log2(ctrl_np + eps)

lfc_error = np.abs(pred_lfc - actual_lfc)  # (n_pairs, n_genes)
mean_lfc_error_all = lfc_error.mean()

# JAK-STAT specific
jakstat_indices = [jakstat_idx[g] for g in jakstat_genes if g in jakstat_idx]
mean_lfc_error_jakstat = lfc_error[:, jakstat_indices].mean()

# Spearman correlation
mean_actual_lfc = actual_lfc.mean(axis=0)
mean_pred_lfc = pred_lfc.mean(axis=0)
spearman_rho, spearman_p = spearmanr(mean_actual_lfc, mean_pred_lfc)

# Percentage within thresholds
within_1_log2 = (lfc_error < 1.0).mean() * 100
within_2_log2 = (lfc_error < 2.0).mean() * 100

print(f"  Mean LFC error (all genes):     {mean_lfc_error_all:.4f}")
print(f"  Mean LFC error (JAK-STAT):      {mean_lfc_error_jakstat:.4f}")
print(f"  Spearman rho (pred vs actual):  {spearman_rho:.4f} (p={spearman_p:.2e})")
print(f"  Genes within 1 log2 unit:       {within_1_log2:.1f}%")
print(f"  Genes within 2 log2 units:      {within_2_log2:.1f}%")

# =========================================================================
# 5.3 High-responder gene performance
# =========================================================================
print(f"\n{'='*80}")
print("HIGH-RESPONDER GENE PERFORMANCE (FC > 5x)")
print(f"{'='*80}")

mean_fc = (actual_np.mean(axis=0) + eps) / (ctrl_np.mean(axis=0) + eps)
high_resp_mask = mean_fc > 5.0
n_high_resp = high_resp_mask.sum()
high_resp_genes = [gene_names[i] for i in range(n_genes) if high_resp_mask[i]]

print(f"  High-responder genes (FC > 5x): {n_high_resp}")

if n_high_resp > 0:
    hr_pred = pred_np[:, high_resp_mask]
    hr_actual = actual_np[:, high_resp_mask]
    hr_r_mean, hr_r_std, _ = compute_pearson_r(hr_pred, hr_actual)
    hr_lfc_error = lfc_error[:, high_resp_mask].mean()

    n_jakstat_hr = sum(1 for g in high_resp_genes if g in jakstat_idx)

    print(f"  Mean Pearson r (high-responders): {hr_r_mean:.4f} +/- {hr_r_std:.4f}")
    print(f"  Mean LFC error (high-responders): {hr_lfc_error:.4f}")
    print(f"  JAK-STAT genes in high-responders: {n_jakstat_hr}")
    print(f"  Week 2 comparison: r=0.633 (all genes) vs Week 3 r={test_r_mean:.3f}")

# =========================================================================
# 5.4 Cell-type specificity metrics
# =========================================================================
print(f"\n{'='*80}")
print("CELL-TYPE STRATIFIED EVALUATION")
print(f"{'='*80}")

all_cts = sorted(set(eval_cell_types))
ct_results = {}
ct_lfc_results = {}

for ct in all_cts:
    ct_mask = [i for i, c in enumerate(eval_cell_types) if c == ct]
    if len(ct_mask) == 0:
        continue
    ct_pred = pred_np[ct_mask]
    ct_actual = actual_np[ct_mask]
    ct_ctrl = ctrl_np[ct_mask]

    ct_r, ct_std, _ = compute_pearson_r(ct_pred, ct_actual)
    ct_results[ct] = ct_r

    ct_lfc_err = np.abs(
        np.log2(ct_pred + eps) - np.log2(ct_ctrl + eps) -
        (np.log2(ct_actual + eps) - np.log2(ct_ctrl + eps))
    ).mean()
    ct_lfc_results[ct] = ct_lfc_err

    above = "ABOVE" if ct_r >= 0.80 else "below"
    print(f"  {ct:<24} r={ct_r:.4f}  LFC_err={ct_lfc_err:.4f}  ({above} 0.80)")

if ct_results:
    ct_values = list(ct_results.values())
    ct_specificity = float(np.std(ct_values))
    ct_range = max(ct_values) - min(ct_values)
    weakest = min(ct_results, key=ct_results.get)
    strongest = max(ct_results, key=ct_results.get)

    print(f"\n  Cell-type specificity score (std): {ct_specificity:.4f}")
    print(f"  Week 2: 0.018 -> Week 3: {ct_specificity:.4f}")
    print(f"  Range: {ct_range:.4f} (target: > 0.10)")
    print(f"  Strongest: {strongest} (r={ct_results[strongest]:.4f})")
    print(f"  Weakest:   {weakest} (r={ct_results[weakest]:.4f})")

    # Monocyte check
    mono_r = ct_results.get("CD14+ Monocytes", None)
    if mono_r is not None:
        print(f"\n  CD14+ Monocyte Pearson r: {mono_r:.4f} {'(>= 0.80 target)' if mono_r >= 0.80 else '(below 0.80)'}")
        b_r = ct_results.get("B cells", None)
        if b_r is not None:
            print(f"  B cell Pearson r:        {b_r:.4f}")
            print(f"  Monocyte > B cell:       {'YES' if mono_r > b_r else 'NO'} (biological validity)")

# =========================================================================
# 5.5 Updated benchmark table
# =========================================================================
print(f"\n{'='*80}")
print("BENCHMARK COMPARISON")
print(f"{'='*80}")

print(f"""
  +----------------------+----------------+---------+------------+-------------+
  | Model                | Pearson r mean | Std r   | DE Recall  | LFC Error   |
  +----------------------+----------------+---------+------------+-------------+
  | scGEN (published)    | 0.820          | --      | --         | --          |
  | CPA (published)      | 0.856          | --      | --         | --          |
  | AIVC Week 2          | 0.633          | 0.018   | 24%        | --          |
  | AIVC Week 3 (ours)   | {test_r_mean:<14.3f} | {test_r_std:<7.3f} | {top50_recall:<10.1%} | {mean_lfc_error_all:<11.3f} |
  +----------------------+----------------+---------+------------+-------------+
""")

if test_r_mean >= 0.80:
    print("  ** DEMO READY — Pearson r exceeds 0.80 target **")
elif test_r_mean >= 0.70:
    print("  APPROACHING BENCHMARK — one more week needed")
else:
    print(f"  BELOW TARGET — r={test_r_mean:.3f}. Diagnostic needed.")

# =========================================================================
# 5.6 Failure mode analysis
# =========================================================================
print(f"\n{'='*80}")
print("FAILURE MODE ANALYSIS — 10 worst-predicted genes")
print(f"{'='*80}")

gene_mae = np.abs(pred_np - actual_np).mean(axis=0)
worst_10_idx = np.argsort(gene_mae)[-10:][::-1]

edge_count = np.zeros(n_genes)
edges_np = edge_index.numpy()
for e in range(edges_np.shape[1]):
    edge_count[edges_np[0, e]] += 1
    edge_count[edges_np[1, e]] += 1

print(f"  {'Rank':<6} {'Gene':<14} {'MAE':<10} {'Actual FC':<12} {'Pred FC':<12} {'Likely reason'}")
print("-" * 80)

for rank, idx in enumerate(worst_10_idx):
    gene = gene_names[idx]
    mae = gene_mae[idx]
    c_val = ctrl_np[:, idx].mean()
    a_val = actual_np[:, idx].mean()
    p_val = pred_np[:, idx].mean()
    afc = (a_val + eps) / (c_val + eps)
    pfc = (p_val + eps) / (c_val + eps)

    n_edges_g = int(edge_count[idx])
    c_std = ctrl_np[:, idx].std()

    if n_edges_g < 3:
        reason = "few graph edges"
    elif c_val < 0.01:
        reason = "low expression"
    elif c_std > c_val and c_val > 0:
        reason = "high variance"
    elif abs(afc) > 10:
        reason = "extreme fold change"
    else:
        reason = "model capacity"

    print(f"  {rank+1:<6} {gene:<14} {mae:<10.4f} {afc:<12.2f} {pfc:<12.2f} {reason}")

print(f"\nOVERALL: Test Pearson r = {test_r_mean:.4f} +/- {test_r_std:.4f}")
print(f"  LFC error = {mean_lfc_error_all:.4f}")
if ct_results:
    print(f"  Cell-type specificity = {ct_specificity:.4f}")
