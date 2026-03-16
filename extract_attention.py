"""
extract_attention.py — Extract GAT attention weights and compute JAK-STAT recovery.

Runs the trained PerturbationPredictor on ctrl and stim inputs, extracts
attention weights from the GAT layers, and computes per-gene attention
centrality. The delta between stim and ctrl centrality reveals which genes
are most activated by the perturbation.

This produces the biological output for demos and publications.

Outputs:
    Top 50 genes by attention delta (stim - ctrl)
    JAK-STAT recovery score: N/15 pathway genes in top 50
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import anndata as ad
from torch_geometric.nn import GATConv

# Reproducibility — all four seeds
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

# Load paired data
X_ctrl = np.load("data/X_ctrl_paired.npy")
X_stim = np.load("data/X_stim_paired.npy")

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

# ----------------------------
# 2. Load model
# ----------------------------
print("Loading model...")
device = torch.device("cpu")
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

model.load_state_dict(torch.load("model_perturbation.pt", map_location=device))
model.eval()
print("  Model loaded from model_perturbation.pt")


# ----------------------------
# 3. Extract attention weights via monkey-patching
# ----------------------------
def extract_attention_weights(model, x_input, edge_index, pert_id):
    """Run forward pass and capture attention weights from both GAT layers.

    Monkey-patches the GATConv layers to return attention weights,
    then restores original behaviour. Does not modify for_now.py.

    Args:
        model: PerturbationPredictor instance.
        x_input: (n_genes,) — per-gene expression.
        edge_index: (2, n_edges) — graph connectivity.
        pert_id: (1,) — perturbation index.

    Returns:
        (attention_weights_layer1, attention_weights_layer2,
         edge_index_layer1, edge_index_layer2)
        Each attention tensor has shape (n_edges_with_self_loops, n_heads).
    """
    # Expand features and add perturbation
    x = model.feature_expander(x_input)
    x = model.pert_embedding(x, pert_id)

    # Layer 1: get attention weights
    x1, (ei1, alpha1) = model.genelink.conv1(x, edge_index, return_attention_weights=True)
    x1 = F.elu(x1)
    # No dropout in eval mode

    # Layer 2: get attention weights
    x2, (ei2, alpha2) = model.genelink.conv2(x1, edge_index, return_attention_weights=True)

    return alpha1, alpha2, ei1, ei2


# ----------------------------
# 4. Compute attention centrality for ctrl and stim
# ----------------------------
print("\nComputing attention centrality...")

# Use mean of all ctrl pairs as representative ctrl input
x_ctrl_mean = torch.tensor(X_ctrl.mean(axis=0), dtype=torch.float32)
x_stim_mean = torch.tensor(X_stim.mean(axis=0), dtype=torch.float32)

pert_ctrl = torch.tensor([0])
pert_stim = torch.tensor([1])

with torch.no_grad():
    # Ctrl pass
    alpha1_ctrl, alpha2_ctrl, ei1_ctrl, ei2_ctrl = extract_attention_weights(
        model, x_ctrl_mean, edge_index, pert_ctrl
    )
    # Stim pass
    alpha1_stim, alpha2_stim, ei1_stim, ei2_stim = extract_attention_weights(
        model, x_stim_mean, edge_index, pert_stim
    )


def compute_centrality(alpha, edge_idx, n_nodes, n_heads):
    """Compute per-node attention centrality (sum of incoming attention).

    For each node, sums the attention weights of all edges pointing to it.
    Averages across attention heads first.

    Args:
        alpha: (n_edges, n_heads) — attention weights.
        edge_idx: (2, n_edges) — edge index (source, target).
        n_nodes: int — number of nodes.
        n_heads: int — number of attention heads.

    Returns:
        (n_nodes,) — attention centrality per node.
    """
    # Average across heads
    alpha_mean = alpha.mean(dim=-1) if alpha.dim() > 1 else alpha  # (n_edges,)
    dst = edge_idx[1]  # target nodes

    centrality = torch.zeros(n_nodes)
    centrality.scatter_add_(0, dst, alpha_mean)
    return centrality


# Use layer 2 (deeper, more informative) attention
n_heads2 = 2
ctrl_centrality = compute_centrality(alpha2_ctrl, ei2_ctrl, n_genes, n_heads2)
stim_centrality = compute_centrality(alpha2_stim, ei2_stim, n_genes, n_heads2)

# Delta: stim - ctrl (positive = activated under IFN-b)
delta = stim_centrality - ctrl_centrality

# ----------------------------
# 5. Rank genes by delta
# ----------------------------
sorted_idx = delta.argsort(descending=True)

print()
print("=" * 75)
print("TOP 50 GENES BY ATTENTION DELTA (stim_centrality - ctrl_centrality)")
print("=" * 75)
print(f"  {'Rank':<6} {'Gene':<14} {'Delta':<12} {'Ctrl cent.':<12} {'Stim cent.':<12} {'JAK-STAT?'}")
print("-" * 75)

jakstat_in_top50 = set()
for rank in range(min(50, len(sorted_idx))):
    idx = sorted_idx[rank].item()
    gene = gene_names[idx]
    d = delta[idx].item()
    cc = ctrl_centrality[idx].item()
    sc_val = stim_centrality[idx].item()
    is_js = gene in jakstat_set
    if is_js:
        jakstat_in_top50.add(gene)
    marker = "YES **" if is_js else ""
    print(f"  {rank+1:<6} {gene:<14} {d:<12.6f} {cc:<12.6f} {sc_val:<12.6f} {marker}")

# ----------------------------
# 6. JAK-STAT recovery score
# ----------------------------
recovery_score = len(jakstat_in_top50)

print()
print("=" * 75)
print("JAK-STAT RECOVERY SCORE")
print("=" * 75)
print(f"  Recovery: {recovery_score}/15 pathway genes in top 50 attention delta")
if jakstat_in_top50:
    print(f"  Found in top 50: {', '.join(sorted(jakstat_in_top50))}")
not_found = jakstat_set - jakstat_in_top50
if not_found:
    print(f"  Not in top 50: {', '.join(sorted(not_found))}")

# Individual gene rankings
print()
print("  Individual JAK-STAT gene rankings:")
for gene in sorted(jakstat_genes):
    idx_g = gene_to_idx.get(gene)
    if idx_g is not None:
        rank_pos = (sorted_idx == idx_g).nonzero(as_tuple=True)[0]
        if len(rank_pos) > 0:
            r = rank_pos[0].item() + 1
            d = delta[idx_g].item()
            marker = " <-- top 50" if r <= 50 else ""
            print(f"    {gene:<12} rank {r:>5}/{n_genes}, delta={d:.6f}{marker}")

# Pass/fail
print()
if recovery_score >= 8:
    print(f"OVERALL: PASS — {recovery_score}/15 JAK-STAT genes in top 50")
elif recovery_score >= 4:
    print(f"OVERALL: PARTIAL — {recovery_score}/15 JAK-STAT genes in top 50")
else:
    print(f"OVERALL: FAIL — {recovery_score}/15 JAK-STAT genes in top 50")
    print("  Note: attention delta measures structural graph centrality changes.")
    print("  The perturbation model may still predict expression correctly")
    print("  even if attention centrality does not concentrate on JAK-STAT genes.")
    print("  Check evaluate_model.py for expression prediction accuracy.")
