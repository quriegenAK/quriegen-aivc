"""
train_fixed.py — Retrain the GeneLink model on the fixed gene set.
Uses data/kang2018_pbmc_fixed.h5ad and data/edge_list_fixed.csv.
Saves checkpoint as model_fixed.pt and log as training_log_fixed.txt.
"""
import sys
import os
import anndata as ad
import scanpy as sc
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

# Import GeneLink and decode_dot from for_now.py
# We cannot import for_now.py directly because it runs training at module level.
# Instead, we redefine the identical model class here (architecture unchanged).
from torch_geometric.nn import GATConv


class GeneLink(torch.nn.Module):
    """Same architecture as for_now.py — 2-layer GAT, unchanged."""
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


def decode_dot(z, edge_label_index):
    src, dst = edge_label_index
    return (z[src] * z[dst]).sum(dim=-1)


# ----------------------------
# 0. Load fixed data
# ----------------------------
print("Loading fixed data...")
adata = ad.read_h5ad("data/kang2018_pbmc_fixed.h5ad")

# Use control cells only for training
adata = adata[adata.obs["label"] == "ctrl"].copy()
print(f"  Control cells: {adata.shape[0]} x {adata.shape[1]} genes")

# Gene name mapping
if "name" in adata.var.columns:
    gene_names = adata.var["name"].tolist()
else:
    gene_names = adata.var_names.tolist()
gene_to_idx = {g: i for i, g in enumerate(gene_names)}

# Expression matrix: genes x cells
X = adata.X
if hasattr(X, "toarray"):
    X = X.toarray()
x = torch.tensor(X.T, dtype=torch.float)  # [n_genes, n_cells]
print(f"  Expression matrix: {x.shape[0]} genes x {x.shape[1]} cells")

# Load fixed edge list
edge_df = pd.read_csv("data/edge_list_fixed.csv")
edges = []
for _, row in edge_df.iterrows():
    a = gene_to_idx.get(row["gene_a"])
    b = gene_to_idx.get(row["gene_b"])
    if a is not None and b is not None:
        edges.append([a, b])
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
print(f"  Edge index: {edge_index.shape[1]} edges")

# Build graph
data = Data(x=x, edge_index=edge_index)
print(f"  Graph: {data.num_nodes} nodes, {data.num_edges} edges")

# ----------------------------
# 1. Train/val/test split
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
# 2. Model, optimizer, device (same hyperparameters as original)
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# 3. Training loop — 200 epochs
# ----------------------------
print("\nTraining...")
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

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  {loss_log[-1]}")

# Save checkpoint
torch.save(model.state_dict(), "model_fixed.pt")
print(f"\nCheckpoint saved: model_fixed.pt")

# Save training log
with open("training_log_fixed.txt", "w") as f:
    f.write("\n".join(loss_log))
print("Training log saved: training_log_fixed.txt")

# ----------------------------
# 4. Evaluate on test set
# ----------------------------
from sklearn.metrics import roc_auc_score

model.eval()
with torch.no_grad():
    z = model(test_data.x, test_data.edge_index)
    logits = decode_dot(z, test_data.edge_label_index)
    y = test_data.edge_label.float()

auc = roc_auc_score(y.cpu().numpy(), logits.cpu().numpy())
print(f"\nTest AUC: {auc:.4f}")

if auc > 0.70:
    print("OVERALL: PASS — Model trained successfully on fixed gene set")
else:
    print(f"OVERALL: MARGINAL — AUC {auc:.4f} below 0.70, but model is training")
