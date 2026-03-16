"""
extract_attention_week3.py — Cell-type stratified attention analysis.

Extends extract_attention.py with:
    - Per-cell-type attention extraction
    - JAK-STAT recovery per cell type
    - Heatmap visualisation for investor demo

Outputs:
    demo_jakstat_week3.png — JAK-STAT pathway activation heatmap by cell type
"""
import random
import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# Load OT or pseudo-bulk
use_ot = False
if os.path.exists("data/X_ctrl_ot.npy"):
    X_ctrl = np.load("data/X_ctrl_ot.npy")
    X_stim = np.load("data/X_stim_ot.npy")
    cell_types_arr = np.load("data/cell_type_ot.npy", allow_pickle=True)
    donors_arr = np.load("data/donor_ot.npy", allow_pickle=True)
    if X_ctrl.shape[0] >= 200:
        use_ot = True
        # Aggregate to mini-bulk (same as training)
        groups = {}
        for i in range(X_ctrl.shape[0]):
            key = (donors_arr[i], cell_types_arr[i])
            if key not in groups:
                groups[key] = {"ctrl": [], "stim": []}
            groups[key]["ctrl"].append(X_ctrl[i])
            groups[key]["stim"].append(X_stim[i])
        X_ctrl_b, X_stim_b, ct_b, donor_b = [], [], [], []
        for (d, ct) in sorted(groups.keys()):
            X_ctrl_b.append(np.mean(groups[(d, ct)]["ctrl"], axis=0))
            X_stim_b.append(np.mean(groups[(d, ct)]["stim"], axis=0))
            ct_b.append(ct)
            donor_b.append(d)
        X_ctrl = np.array(X_ctrl_b)
        X_stim = np.array(X_stim_b)
        cell_types_arr = np.array(ct_b)
        donors_arr = np.array(donor_b)
        print(f"  Aggregated to {X_ctrl.shape[0]} mini-bulk groups")
if not use_ot:
    X_ctrl = np.load("data/X_ctrl_paired.npy")
    X_stim = np.load("data/X_stim_paired.npy")
    manifest = pd.read_csv("data/pairing_manifest.csv")
    paired = manifest[manifest["paired"]].reset_index(drop=True)
    cell_types_arr = paired["cell_type"].values
    donors_arr = paired["donor_id"].values

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
jakstat_set = set(jakstat_genes)

# =========================================================================
# 2. Load model
# =========================================================================
print("Loading model...")
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
# 3. Attention extraction
# =========================================================================
def extract_attention_weights(model, x_input, edge_index, pert_id, ct_id=None):
    """Run forward pass and capture attention weights from both GAT layers."""
    x = model.feature_expander(x_input)
    x = model.pert_embedding(x, pert_id)
    if ct_id is not None and hasattr(model, "cell_type_embedding"):
        x = model.cell_type_embedding(x, ct_id)

    x1, (ei1, alpha1) = model.genelink.conv1(x, edge_index, return_attention_weights=True)
    x1 = F.elu(x1)
    x2, (ei2, alpha2) = model.genelink.conv2(x1, edge_index, return_attention_weights=True)

    return alpha1, alpha2, ei1, ei2


def compute_centrality(alpha, edge_idx, n_nodes):
    """Compute per-node attention centrality (sum of incoming attention)."""
    alpha_mean = alpha.mean(dim=-1) if alpha.dim() > 1 else alpha
    dst = edge_idx[1]
    centrality = torch.zeros(n_nodes)
    centrality.scatter_add_(0, dst, alpha_mean)
    return centrality


# =========================================================================
# 4. Cell-type stratified attention
# =========================================================================
print(f"\n{'='*80}")
print("CELL-TYPE STRATIFIED ATTENTION ANALYSIS")
print(f"{'='*80}")

cell_types_list = sorted(set(cell_types_arr.tolist()))
# Exclude Megakaryocytes (too few cells)
cell_types_eval = [ct for ct in cell_types_list if ct != "Megakaryocytes"]

ct_map = CellTypeEmbedding.CELL_TYPE_MAP
pert_ctrl = torch.tensor([0])
pert_stim = torch.tensor([1])

# Store per-cell-type attention deltas for all genes
ct_deltas = {}  # {cell_type: (n_genes,) attention delta}
ct_top10 = {}
ct_jakstat_recovery = {}

for ct in cell_types_eval:
    ct_mask = cell_types_arr == ct
    n_ct = ct_mask.sum()
    if n_ct < 5:
        print(f"  Skipping {ct}: only {n_ct} pairs")
        continue

    # Mean ctrl and stim expression for this cell type
    x_ctrl_ct = torch.tensor(X_ctrl[ct_mask].mean(axis=0), dtype=torch.float32)
    x_stim_ct = torch.tensor(X_stim[ct_mask].mean(axis=0), dtype=torch.float32)

    ct_idx_val = ct_map.get(ct, 8)
    ct_id = torch.tensor([ct_idx_val])

    with torch.no_grad():
        _, alpha2_ctrl, _, ei2_ctrl = extract_attention_weights(
            model, x_ctrl_ct, edge_index, pert_ctrl, ct_id
        )
        _, alpha2_stim, _, ei2_stim = extract_attention_weights(
            model, x_stim_ct, edge_index, pert_stim, ct_id
        )

    ctrl_cent = compute_centrality(alpha2_ctrl, ei2_ctrl, n_genes)
    stim_cent = compute_centrality(alpha2_stim, ei2_stim, n_genes)
    delta = stim_cent - ctrl_cent

    ct_deltas[ct] = delta.numpy()

    # Top 10 genes
    sorted_idx = delta.argsort(descending=True)
    top10 = [(gene_names[sorted_idx[i].item()], delta[sorted_idx[i].item()].item())
             for i in range(min(10, len(sorted_idx)))]
    ct_top10[ct] = top10

    # JAK-STAT recovery
    top50_set = set(gene_names[sorted_idx[i].item()] for i in range(min(50, len(sorted_idx))))
    jakstat_in = jakstat_set & top50_set
    ct_jakstat_recovery[ct] = len(jakstat_in)

    print(f"\n  {ct} (n={n_ct} pairs):")
    print(f"    Top 5 activated: {', '.join(f'{g}({d:.4f})' for g, d in top10[:5])}")
    print(f"    JAK-STAT in top 50: {len(jakstat_in)}/15")
    if jakstat_in:
        print(f"    Found: {', '.join(sorted(jakstat_in))}")

# =========================================================================
# 5. JAK-STAT recovery summary
# =========================================================================
print(f"\n{'='*80}")
print("JAK-STAT RECOVERY PER CELL TYPE")
print(f"{'='*80}")

print(f"  {'Cell type':<24} {'Recovery':<12} {'Top activated gene'}")
print(f"  {'-'*55}")
for ct in cell_types_eval:
    if ct in ct_jakstat_recovery:
        top_gene = ct_top10[ct][0][0] if ct in ct_top10 else "N/A"
        print(f"  {ct:<24} {ct_jakstat_recovery[ct]}/15        {top_gene}")

# Overall recovery: union across cell types
all_recovered = set()
for ct in cell_types_eval:
    if ct in ct_deltas:
        delta = ct_deltas[ct]
        sorted_idx = np.argsort(delta)[::-1][:50]
        for i in sorted_idx:
            if gene_names[i] in jakstat_set:
                all_recovered.add(gene_names[i])

overall_recovery = len(all_recovered)
print(f"\n  Overall recovery (union): {overall_recovery}/15")
if all_recovered:
    print(f"  Recovered genes: {', '.join(sorted(all_recovered))}")
not_recovered = jakstat_set - all_recovered
if not_recovered:
    print(f"  Not recovered:   {', '.join(sorted(not_recovered))}")

# Monocyte vs B cell check
mono_recovery = ct_jakstat_recovery.get("CD14+ Monocytes", 0)
b_recovery = ct_jakstat_recovery.get("B cells", 0)
print(f"\n  Biological validity check:")
print(f"    CD14+ Monocytes JAK-STAT recovery: {mono_recovery}/15")
print(f"    B cells JAK-STAT recovery:         {b_recovery}/15")
print(f"    Monocytes > B cells:               {'YES' if mono_recovery > b_recovery else 'NO (expected YES)'}")

# =========================================================================
# 6. Heatmap visualisation
# =========================================================================
print(f"\n{'='*80}")
print("GENERATING JAK-STAT HEATMAP")
print(f"{'='*80}")

# Build heatmap data matrix: rows = JAK-STAT genes, cols = cell types
heatmap_cts = [ct for ct in cell_types_eval if ct in ct_deltas]

# Order cell types by total JAK-STAT activation (strongest left)
ct_total_activation = {}
for ct in heatmap_cts:
    delta = ct_deltas[ct]
    jakstat_indices = [gene_to_idx[g] for g in jakstat_genes if g in gene_to_idx]
    ct_total_activation[ct] = sum(delta[i] for i in jakstat_indices)

heatmap_cts = sorted(heatmap_cts, key=lambda ct: ct_total_activation[ct], reverse=True)

# Build matrix
jakstat_present = [g for g in jakstat_genes if g in gene_to_idx]
heatmap_data = np.zeros((len(jakstat_present), len(heatmap_cts)))

for j, ct in enumerate(heatmap_cts):
    delta = ct_deltas[ct]
    for i, g in enumerate(jakstat_present):
        idx = gene_to_idx[g]
        heatmap_data[i, j] = delta[idx]

# Normalise per column for better visualisation
for j in range(heatmap_data.shape[1]):
    col_std = heatmap_data[:, j].std()
    if col_std > 1e-10:
        heatmap_data[:, j] = heatmap_data[:, j] / col_std

# Order genes by mean activation
gene_order = np.argsort(heatmap_data.mean(axis=1))[::-1]
heatmap_data = heatmap_data[gene_order]
jakstat_ordered = [jakstat_present[i] for i in gene_order]

# Plot
fig, ax = plt.subplots(figsize=(12, 8))
vmax = max(abs(heatmap_data.min()), abs(heatmap_data.max()))
if vmax < 1e-6:
    vmax = 1.0
im = ax.imshow(heatmap_data, cmap="RdBu_r", aspect="auto",
               vmin=-vmax, vmax=vmax)

# Gold highlights for activation > 2 std above column mean
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        col_mean = heatmap_data[:, j].mean()
        col_std_val = heatmap_data[:, j].std()
        if col_std_val > 1e-10 and heatmap_data[i, j] > col_mean + 2 * col_std_val:
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                 fill=False, edgecolor="gold", linewidth=2)
            ax.add_patch(rect)

# Short labels for cell types
ct_short = {
    "CD4 T cells": "CD4 T",
    "CD8 T cells": "CD8 T",
    "NK cells": "NK",
    "B cells": "B cells",
    "CD14+ Monocytes": "CD14 Mono",
    "Dendritic cells": "DC",
    "FCGR3A+ Monocytes": "FCGR3A Mono",
}

ax.set_xticks(range(len(heatmap_cts)))
ax.set_xticklabels([ct_short.get(ct, ct) for ct in heatmap_cts],
                   fontsize=11, rotation=30, ha="right")
ax.set_yticks(range(len(jakstat_ordered)))
ax.set_yticklabels(jakstat_ordered, fontsize=11, fontfamily="DejaVu Sans")

ax.set_title("IFN-\u03B2 Response \u2014 JAK-STAT Pathway Activation by Cell Type",
             fontsize=14, fontweight="bold", pad=15, fontfamily="DejaVu Sans")
ax.text(0.5, 1.02, "AIVC Model Attention Weights | Kang 2018 PBMC | GSE96583",
        transform=ax.transAxes, fontsize=10, ha="center", color="gray",
        fontfamily="DejaVu Sans")

cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Attention delta (normalised)", fontsize=10)

ax.text(0.5, -0.12,
        "Attention delta = model activation under IFN-\u03B2 vs control\n"
        "Gold border = activation > 2 std above column mean",
        transform=ax.transAxes, fontsize=9, ha="center", color="gray",
        fontfamily="DejaVu Sans")

plt.tight_layout()
plt.savefig("demo_jakstat_week3.png", dpi=300, bbox_inches="tight")
print("  Saved: demo_jakstat_week3.png (300 dpi)")

# Pass/fail
print(f"\n{'='*80}")
if overall_recovery >= 8:
    print(f"OVERALL: PASS — {overall_recovery}/15 JAK-STAT genes recovered (union across cell types)")
elif overall_recovery >= 4:
    print(f"OVERALL: PARTIAL — {overall_recovery}/15 JAK-STAT genes recovered")
else:
    print(f"OVERALL: FAIL — {overall_recovery}/15 JAK-STAT genes recovered")
    print("  Note: attention centrality measures structural changes in the graph.")
    print("  The model may still predict expression changes correctly.")
