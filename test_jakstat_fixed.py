"""
test_jakstat_fixed.py — Validate JAK-STAT pathway recovery in the fixed model.
Runs ctrl vs stim forward passes and computes activation delta per gene.
"""
import os
import torch
import torch.nn.functional as F
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np

from torch_geometric.data import Data
from torch_geometric.nn import GATConv

# ----------------------------
# GeneLink model (same architecture as for_now.py)
# ----------------------------
class GeneLink(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden1_dim,
        hidden2_dim,
        hidden3_dim,
        output_dim,
        num_head1,
        num_head2,
        alpha,
        device,
        type,
        reduction,
    ):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden1_dim, heads=num_head1, dropout=0.01)
        self.conv2 = GATConv(
            hidden1_dim * num_head1, hidden2_dim, heads=num_head2, dropout=0.01
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.01, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# ----------------------------
# 1. Load fixed data (full dataset, both conditions)
# ----------------------------
print("Loading fixed data...")
adata = ad.read_h5ad("data/kang2018_pbmc_fixed.h5ad")
print(f"  Shape: {adata.shape[0]} cells x {adata.shape[1]} genes")

# The data is already normalized and log-transformed from fix_gene_selection.py
# But we need to verify — check if layers["counts"] exists
if "counts" not in adata.layers:
    print("  Re-normalizing (counts layer not found)...")
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

# 2. Build gene name mapping
if "name" in adata.var.columns:
    gene_names = adata.var["name"].tolist()
else:
    gene_names = adata.var_names.tolist()
gene_to_idx = {g: i for i, g in enumerate(gene_names)}

# 3. Load edge list and build edge_index
edge_df = pd.read_csv("data/edge_list_fixed.csv")
edges = []
for _, row in edge_df.iterrows():
    a = gene_to_idx.get(row["gene_a"])
    b = gene_to_idx.get(row["gene_b"])
    if a is not None and b is not None:
        edges.append([a, b])
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
print(f"  Edges: {edge_index.shape[1]}")

# 4. Get expression matrices for ctrl and stim
X = adata.X
if hasattr(X, "toarray"):
    X = X.toarray()

ctrl_mask = (adata.obs["label"] == "ctrl").values
stim_mask = (adata.obs["label"] == "stim").values

ctrl_expr = X[ctrl_mask, :]  # [n_ctrl_cells, n_genes]
stim_expr = X[stim_mask, :]  # [n_stim_cells, n_genes]

# Build node features: mean expression per gene (transpose to genes x cells, then mean)
# For each pass, the input is the gene-level feature vector derived from cell expression
# Pass A: ctrl cells only -> genes x n_ctrl_cells
# Pass B: stim cells only -> genes x n_stim_cells
# But the model expects x of shape [n_genes, n_features].
# In training, features = all ctrl cell expressions (genes x n_ctrl_cells).
# For a fair comparison, we use the mean expression vector per gene as a 1D feature.
# However, the model was trained with input_dim = n_ctrl_cells. We need matching dims.

# Load model to check expected input_dim
n_ctrl = ctrl_mask.sum()
n_stim = stim_mask.sum()
print(f"  Ctrl cells: {n_ctrl}, Stim cells: {n_stim}")

# The model was trained with input_dim = n_ctrl_cells (each gene's feature = its
# expression across all ctrl cells). For stim comparison, we need to project stim
# expression to the same dimensionality. The simplest approach:
# Use the full ctrl expression as Pass A input, and for Pass B, sample n_ctrl cells
# from stim to match dimensions.

# Pass A input: genes x ctrl_cells
x_ctrl = torch.tensor(ctrl_expr.T, dtype=torch.float)  # [n_genes, n_ctrl_cells]

# Pass B input: genes x stim_cells — sample to match ctrl size
np.random.seed(42)
if n_stim >= n_ctrl:
    stim_sample_idx = np.random.choice(n_stim, size=n_ctrl, replace=False)
else:
    stim_sample_idx = np.random.choice(n_stim, size=n_ctrl, replace=True)
x_stim = torch.tensor(stim_expr[stim_sample_idx, :].T, dtype=torch.float)  # [n_genes, n_ctrl_cells]

print(f"  Ctrl input shape: {x_ctrl.shape}")
print(f"  Stim input shape: {x_stim.shape}")

# 5. Load model checkpoint
ckpt_path = "model_fixed.pt"
if not os.path.exists(ckpt_path):
    print(f"\nERROR: {ckpt_path} not found. Run train_fixed.py first.")
    exit(1)

device = torch.device("cpu")
model = GeneLink(
    input_dim=x_ctrl.shape[1],
    hidden1_dim=64,
    hidden2_dim=32,
    hidden3_dim=16,
    output_dim=8,
    num_head1=3,
    num_head2=2,
    alpha=0.2,
    device=device,
    type="MLP",
    reduction="mean",
).to(device)

model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()
print("  Model loaded from model_fixed.pt")

# 6-7. Forward passes
print("\nRunning forward passes...")
with torch.no_grad():
    z_ctrl = model(x_ctrl, edge_index)
    z_stim = model(x_stim, edge_index)

print(f"  Ctrl embedding: {z_ctrl.shape}, mean={z_ctrl.mean():.4f}, std={z_ctrl.std():.4f}")
print(f"  Stim embedding: {z_stim.shape}, mean={z_stim.mean():.4f}, std={z_stim.std():.4f}")

# 8. Compute activation delta
delta = (z_stim - z_ctrl).abs()
delta_magnitude = delta.norm(dim=1)  # L2 norm per gene

# 9. Print top 20 genes by delta magnitude
must_include = {
    "JAK1", "JAK2", "STAT1", "STAT2", "STAT3",
    "IRF9", "IRF1", "MX1", "MX2", "ISG15",
    "OAS1", "IFIT1", "IFIT3", "IFITM1", "IFITM3",
}

sorted_idx = delta_magnitude.argsort(descending=True)

print()
print("=" * 70)
print("TOP 20 GENES BY ACTIVATION DELTA (|stim_embed - ctrl_embed|)")
print("=" * 70)
print(f"  {'Rank':<6} {'Gene':<12} {'Delta':<12} {'JAK-STAT?'}")
print("-" * 70)

for rank in range(min(20, len(sorted_idx))):
    idx = sorted_idx[rank].item()
    gene = gene_names[idx]
    d = delta_magnitude[idx].item()
    is_jakstat = "YES **" if gene in must_include else ""
    print(f"  {rank+1:<6} {gene:<12} {d:<12.4f} {is_jakstat}")

# 10. JAK-STAT recovery score: how many in top 50?
top50_genes = set()
for rank in range(min(50, len(sorted_idx))):
    idx = sorted_idx[rank].item()
    top50_genes.add(gene_names[idx])

jakstat_in_top50 = must_include & top50_genes
recovery_score = len(jakstat_in_top50)

print()
print("=" * 70)
print("JAK-STAT RECOVERY SCORE")
print("=" * 70)
print(f"  JAK-STAT genes in top 50 delta: {recovery_score}/{len(must_include)}")
if jakstat_in_top50:
    print(f"    Found: {', '.join(sorted(jakstat_in_top50))}")
jakstat_missing_top50 = must_include - jakstat_in_top50
if jakstat_missing_top50:
    print(f"    Not in top 50: {', '.join(sorted(jakstat_missing_top50))}")

# Also show where each JAK-STAT gene ranks
print()
print("  Individual JAK-STAT gene rankings:")
for gene in sorted(must_include):
    idx = gene_to_idx.get(gene)
    if idx is not None:
        # Find rank
        rank_pos = (sorted_idx == idx).nonzero(as_tuple=True)[0]
        if len(rank_pos) > 0:
            r = rank_pos[0].item() + 1
            d = delta_magnitude[idx].item()
            marker = " <-- top 50" if r <= 50 else ""
            print(f"    {gene:<10} rank {r:>5}/{len(gene_names)}, delta={d:.4f}{marker}")
        else:
            print(f"    {gene:<10} (not ranked)")
    else:
        print(f"    {gene:<10} (not in gene set)")

# 11. Pass/fail
print()
if recovery_score >= 5:
    print(f"OVERALL: PASS — {recovery_score}/15 JAK-STAT genes in top 50 activated nodes")
elif recovery_score >= 2:
    print(f"OVERALL: PARTIAL — {recovery_score}/15 JAK-STAT genes in top 50 (model needs more epochs or perturbation embedding)")
else:
    print(f"OVERALL: FAIL — Only {recovery_score}/15 JAK-STAT genes in top 50")
