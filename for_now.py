#################################################
#
# AIVC GeneLink — GAT link prediction
# Feb 18, 2026
#################################################

import os
import anndata as ad
import scanpy as sc
import torch
import pandas as pd
from torch import nn, optim
from torch.nn import functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.transforms import RandomLinkSplit

# ----------------------------
# 0. Load Kang 2018 PBMC data and build the graph
# ----------------------------

adata = ad.read_h5ad("data/kang2018_pbmc.h5ad")

# Use control cells only
adata = adata[adata.obs["label"] == "ctrl"].copy()

# Normalize and log transform
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# Select top 2000 HVGs (same as used to build edge_list.csv)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)

# Build gene name to index mapping
# Use the 'name' column if available, otherwise var_names
if "name" in adata.var.columns:
    gene_names = adata.var["name"].tolist()
else:
    gene_names = adata.var_names.tolist()
gene_to_idx = {g: i for i, g in enumerate(gene_names)}

# Expression matrix: genes x cells (transpose from cells x genes)
# Convert sparse to dense if needed
X = adata.X
if hasattr(X, "toarray"):
    X = X.toarray()
x = torch.tensor(X.T, dtype=torch.float)  # shape: [n_genes, n_cells]

print(f"Expression matrix: {x.shape[0]} genes x {x.shape[1]} cells")

# Load edge list
edge_df = pd.read_csv("data/edge_list.csv")

# Map gene names to indices, drop edges where either gene is missing
edges = []
for _, row in edge_df.iterrows():
    a = gene_to_idx.get(row["gene_a"])
    b = gene_to_idx.get(row["gene_b"])
    if a is not None and b is not None:
        edges.append([a, b])

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # shape: [2, num_edges]
print(f"Edge index: {edge_index.shape[1]} edges")

# Build the graph data object
data = Data(x=x, edge_index=edge_index)
print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges")


# ----------------------------
# 1. Model: 2-layer GAT (architecture unchanged)
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
# 2. Dot-product decoder for link prediction
# ----------------------------

def decode_dot(z, edge_label_index):
    src, dst = edge_label_index
    return (z[src] * z[dst]).sum(dim=-1)


# ----------------------------
# 3. Train/val/test split with negative sampling
# ----------------------------

transform = RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=True,
    neg_sampling_ratio=1.0,
)

train_data, val_data, test_data = transform(data)


# ----------------------------
# 4. Define model, optimizer, device
# ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# output dim of conv2 = hidden2_dim * num_head2
model = GeneLink(
    input_dim=x.shape[1],
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

train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


# ----------------------------
# 5. Training loop
# ----------------------------

loss_log = []

model.train()
for epoch in range(200):
    optimizer.zero_grad()

    z = model(train_data.x, train_data.edge_index)
    logits = decode_dot(z, train_data.edge_label_index)
    y = train_data.edge_label.float()

    loss = F.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    optimizer.step()

    loss_val = loss.item()
    loss_log.append(f"Epoch {epoch+1}, Loss: {loss_val:.4f}")
    print(loss_log[-1])

# Save training log
with open("training_log.txt", "w") as f:
    f.write("\n".join(loss_log))
print("\nTraining log saved to training_log.txt")


# ----------------------------
# 6. Evaluate on test set
# ----------------------------

@torch.no_grad()
def evaluate(split_data):
    model.eval()
    z = model(split_data.x, split_data.edge_index)
    logits = decode_dot(z, split_data.edge_label_index)
    y = split_data.edge_label

    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y.cpu(), logits.cpu())

test_auc = evaluate(test_data)
print(f"\nTest AUC: {test_auc:.4f}")
