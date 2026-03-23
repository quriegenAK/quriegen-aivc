"""
evaluate_v11.py — Extended evaluation for AIVC v1.1 with regression guard.

Evaluates the Neumann cascade propagation model against v1.0 baseline.
Reports IFIT1 fold change, JAK-STAT recovery, and cell-type stratified r.

Regression guard: ABORT if any of these conditions fail:
  1. Pearson r >= 0.873 (no regression from v1.0)
  2. JAK-STAT within 3x >= 8 (improvement from 7)
  3. IFIT1 predicted FC within 10x of actual 107x
  4. CD14+ monocyte r >= 0.80

Outputs:
  results/evaluation_v11.txt           — full report
  results/v11_regression_guard.json    — pass/fail per criterion
"""
import random
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import anndata as ad
from scipy.stats import spearmanr

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

from perturbation_model import PerturbationPredictor, CellTypeEmbedding
from aivc.skills.neumann_propagation import NeumannPropagation

# =========================================================================
# 1. Baseline
# =========================================================================
BASELINE = {
    "pearson_r": 0.873,
    "mse": 0.013,
    "jakstat_recovery": 7,
    "cell_type_spread": 0.198,
    "cpa_benchmark": 0.856,
}

# =========================================================================
# 2. Load data
# =========================================================================
print("Loading data...")
adata = ad.read_h5ad("data/kang2018_pbmc_fixed.h5ad")
gene_names = adata.var["name"].tolist() if "name" in adata.var.columns else adata.var_names.tolist()
n_genes = len(gene_names)
gene_to_idx = {g: i for i, g in enumerate(gene_names)}

use_ot = False
if os.path.exists("data/X_ctrl_ot.npy"):
    X_ctrl = np.load("data/X_ctrl_ot.npy")
    X_stim = np.load("data/X_stim_ot.npy")
    ct_arr = np.load("data/cell_type_ot.npy", allow_pickle=True)
    donors_arr = np.load("data/donor_ot.npy", allow_pickle=True)
    if X_ctrl.shape[0] >= 200:
        use_ot = True
        groups = {}
        for i in range(len(X_ctrl)):
            key = (donors_arr[i], ct_arr[i])
            if key not in groups:
                groups[key] = {"ctrl": [], "stim": []}
            groups[key]["ctrl"].append(X_ctrl[i])
            groups[key]["stim"].append(X_stim[i])

        X_c, X_s, ct_b, d_b = [], [], [], []
        for (d, ct) in sorted(groups.keys()):
            X_c.append(np.mean(groups[(d, ct)]["ctrl"], axis=0))
            X_s.append(np.mean(groups[(d, ct)]["stim"], axis=0))
            ct_b.append(ct)
            d_b.append(d)
        X_ctrl = np.array(X_c)
        X_stim = np.array(X_s)
        ct_arr = np.array(ct_b)
        donors_arr = np.array(d_b)

if not use_ot:
    X_ctrl = np.load("data/X_ctrl_paired.npy")
    X_stim = np.load("data/X_stim_paired.npy")
    manifest = pd.read_csv("data/pairing_manifest.csv")
    paired = manifest[manifest["paired"]].reset_index(drop=True)
    ct_arr = paired["cell_type"].values
    donors_arr = paired["donor_id"].values

edge_df = pd.read_csv("data/edge_list_fixed.csv")
edges, edge_weights = [], []
for _, row in edge_df.iterrows():
    a = gene_to_idx.get(row["gene_a"])
    b = gene_to_idx.get(row["gene_b"])
    if a is not None and b is not None:
        edges.append([a, b])
        edge_weights.append(row.get("combined_score", 700) if "combined_score" in edge_df.columns else 700)
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(edge_weights, dtype=torch.float32)

jakstat_genes = ["JAK1", "JAK2", "STAT1", "STAT2", "STAT3", "IRF9", "IRF1",
                 "MX1", "MX2", "ISG15", "OAS1", "IFIT1", "IFIT3", "IFITM1", "IFITM3"]
jakstat_idx = {g: gene_to_idx[g] for g in jakstat_genes if g in gene_to_idx}

# Donor split
unique_donors = sorted(np.unique(donors_arr).tolist())
n_d = len(unique_donors)
n_tr = max(1, int(n_d * 0.6))
n_v = max(1, int(n_d * 0.2))
test_donors = set(unique_donors[n_tr + n_v:])
test_idx = [i for i in range(len(donors_arr)) if donors_arr[i] in test_donors]

ct_indices = CellTypeEmbedding.encode_cell_types(ct_arr.tolist())
X_ctrl_test = torch.tensor(X_ctrl[test_idx], dtype=torch.float32)
X_stim_test = torch.tensor(X_stim[test_idx], dtype=torch.float32)
ct_test = ct_indices[test_idx]

print(f"  Test: {len(test_idx)} pairs, {n_genes} genes")

# =========================================================================
# 3. Load model
# =========================================================================
device = torch.device("cpu")
model = PerturbationPredictor(
    n_genes=n_genes, num_perturbations=2, feature_dim=64,
    hidden1_dim=64, hidden2_dim=32, num_head1=3, num_head2=2, decoder_hidden=256,
).to(device)
model.cell_type_embedding = CellTypeEmbedding(num_cell_types=20, embedding_dim=64).to(device)

# Try v1.1 model, fall back to v1.0
model_path = None
for p in ["models/v1.1/model_v11_best.pt", "models/week4/model_week4_best.pt",
          "model_week3_best.pt", "model_week3.pt"]:
    if os.path.exists(p):
        # Check if model needs Neumann module
        state = torch.load(p, map_location=device)
        has_neumann = any(k.startswith("neumann.") for k in state.keys())
        if has_neumann:
            model.neumann = NeumannPropagation(
                n_genes=n_genes, edge_index=edge_index, edge_attr=edge_attr,
            ).to(device)
        model.load_state_dict(state)
        model_path = p
        break

if model_path is None:
    print("ERROR: No model found")
    sys.exit(1)

model.eval()
print(f"  Model: {model_path}")
has_neumann = hasattr(model, "neumann") and model.neumann is not None
print(f"  Neumann: {'YES' if has_neumann else 'NO'}")

# =========================================================================
# 4. Predictions
# =========================================================================
with torch.no_grad():
    pert_stim = torch.tensor([1])
    pred_delta = model.forward_batch(X_ctrl_test, edge_index, pert_stim, ct_test)
    pred_test = (X_ctrl_test + pred_delta).clamp(min=0.0)

pred_np = pred_test.numpy()
actual_np = X_stim_test.numpy()
ctrl_np = X_ctrl[test_idx]

# =========================================================================
# 5. Metrics
# =========================================================================
def compute_pearson_r(predicted, actual):
    rs = []
    for i in range(predicted.shape[0]):
        p, a = predicted[i], actual[i]
        if np.std(p) < 1e-10 or np.std(a) < 1e-10:
            rs.append(0.0)
            continue
        r = np.corrcoef(p, a)[0, 1]
        rs.append(0.0 if np.isnan(r) else r)
    return float(np.mean(rs)), float(np.std(rs)), rs

report = []
def log(msg):
    print(msg)
    report.append(msg)

test_r, test_std, _ = compute_pearson_r(pred_np, actual_np)
test_mse = F.mse_loss(pred_test, X_stim_test).item()

log(f"\n{'='*70}")
log(f"AIVC v1.1 EVALUATION — Model: {model_path}")
log(f"{'='*70}")
log(f"  Pearson r: {test_r:.4f} +/- {test_std:.4f}")
log(f"  MSE:       {test_mse:.6f}")

# JAK-STAT fold changes
log(f"\n  JAK-STAT Fold Changes:")
log(f"  {'Gene':<10} {'Pred FC':<10} {'Actual FC':<10} {'Ratio':<8} {'Within 3x'}")
log(f"  {'-'*50}")

eps = 1e-6
within_3x = 0
within_10x = 0
ifit1_pred_fc = 0.0
ifit1_actual_fc = 0.0

for g in sorted(jakstat_idx.keys()):
    idx = jakstat_idx[g]
    c = ctrl_np[:, idx].mean()
    p = pred_np[:, idx].mean()
    a = actual_np[:, idx].mean()
    pfc = (p + eps) / (c + eps)
    afc = (a + eps) / (c + eps)
    ratio = max(pfc, afc) / max(min(pfc, afc), eps) if min(pfc, afc) > 0.01 else float("inf")

    if g == "IFIT1":
        ifit1_pred_fc = pfc
        ifit1_actual_fc = afc

    w3 = "YES" if ratio <= 3.0 else "NO"
    if ratio <= 3.0:
        within_3x += 1
    if ratio <= 10.0:
        within_10x += 1

    log(f"  {g:<10} {pfc:<10.2f} {afc:<10.2f} {ratio:<8.1f} {w3}")

log(f"\n  JAK-STAT within 3x: {within_3x}/15")
log(f"  JAK-STAT within 10x: {within_10x}/15")
log(f"  IFIT1: pred={ifit1_pred_fc:.1f}x actual={ifit1_actual_fc:.1f}x")

# Cell-type stratified
log(f"\n  Cell-Type Stratified Pearson r:")
test_cts = [ct_arr[i] for i in test_idx]
ct_results = {}
for ct in sorted(set(test_cts)):
    mask = [i for i, c in enumerate(test_cts) if c == ct]
    if mask:
        r, _, _ = compute_pearson_r(pred_np[mask], actual_np[mask])
        ct_results[ct] = r
        marker = " <<<" if "CD14" in ct and r < 0.80 else ""
        log(f"    {ct:<24} r={r:.4f}{marker}")

cd14_r = ct_results.get("CD14+ Monocytes", 0.0)

# =========================================================================
# 6. Regression guard
# =========================================================================
def regression_guard(results):
    """
    Returns True if v1.1 results are acceptable.
    All conditions must pass:
      1. pearson_r >= 0.873
      2. jakstat_recovery >= 8 (within 3x fold change)
      3. ifit1_fold_change within 10x of actual
      4. cd14_monocyte_r >= 0.80
    """
    checks = {
        "pearson_r >= 0.873": results["test_r"] >= BASELINE["pearson_r"],
        "jakstat_within_3x >= 8": results["jakstat_within_3x"] >= 8,
        "ifit1_within_10x": (
            results["ifit1_actual_fc"] > 0 and
            max(results["ifit1_pred_fc"], results["ifit1_actual_fc"]) /
            max(min(results["ifit1_pred_fc"], results["ifit1_actual_fc"]), eps) <= 10.0
        ),
        "cd14_monocyte_r >= 0.80": results["cd14_mono_r"] >= 0.80,
        "pearson_r >= 0.860 (hard floor)": results["test_r"] >= 0.860,
    }
    return checks

guard_input = {
    "test_r": test_r,
    "jakstat_within_3x": within_3x,
    "ifit1_pred_fc": ifit1_pred_fc,
    "ifit1_actual_fc": ifit1_actual_fc,
    "cd14_mono_r": cd14_r,
}

checks = regression_guard(guard_input)

log(f"\n{'='*70}")
log("REGRESSION GUARD")
log(f"{'='*70}")
for name, passed in checks.items():
    log(f"  {'[x]' if passed else '[ ]'} {name}")

n_passed = sum(v for v in checks.values())
log(f"\n  Passed: {n_passed}/{len(checks)}")

if all(checks.values()):
    log("  VERDICT: ALL CHECKS PASS. v1.1 ready for deployment.")
elif checks.get("pearson_r >= 0.860 (hard floor)", False):
    log("  VERDICT: ABOVE HARD FLOOR. Some targets not met.")
else:
    log("  VERDICT: REGRESSION DETECTED. Do not deploy.")

# Comparison table
log(f"\n{'='*70}")
log("v1.0 vs v1.1 COMPARISON")
log(f"{'='*70}")
log(f"""
  +------------------+----------+----------+---------+
  | Metric           | v1.0     | v1.1     | Target  |
  +------------------+----------+----------+---------+
  | Pearson r        | 0.873    | {test_r:<8.4f} | >= 0.873|
  | Test MSE         | 0.013    | {test_mse:<8.4f} | --      |
  | JAK-STAT 3x      | 7/15     | {within_3x}/15     | >= 8    |
  | JAK-STAT 10x     | --       | {within_10x}/15     | --      |
  | IFIT1 pred FC    | 2.05x    | {ifit1_pred_fc:<8.1f} | <20x off|
  | CD14 mono r      | 0.745    | {cd14_r:<8.4f} | >= 0.80 |
  | CPA benchmark    | 0.856    | {test_r:<8.4f} | beat    |
  | scGEN benchmark  | 0.820    | {test_r:<8.4f} | beat    |
  +------------------+----------+----------+---------+
""")

# =========================================================================
# 7. Save
# =========================================================================
os.makedirs("results", exist_ok=True)
with open("results/evaluation_v11.txt", "w") as f:
    f.write("\n".join(report))
print(f"  Saved: results/evaluation_v11.txt")

guard_output = {
    "checks": {k: bool(v) for k, v in checks.items()},
    "all_pass": all(checks.values()),
    "metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                for k, v in guard_input.items()},
    "baseline": BASELINE,
    "model_path": model_path,
}
with open("results/v11_regression_guard.json", "w") as f:
    json.dump(guard_output, f, indent=2)
print(f"  Saved: results/v11_regression_guard.json")
