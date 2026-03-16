#
# AIVC GeneLink — Full-batch GAT
# February 18, 2026
#
# Implementation of a simple full-batch GAT for link prediction on directed gene regulatory networks.
# https://academic.oup.com/bioinformatics/article/38/19/4522/6663989
# 
# This GAT is not cell type specific

# Full-batch, transductive link prediction with a 2-layer GAT in PyTorch Geometric.
#   - Directed graph
#   - RandomLinkSplit for train/val/test (with negative sampling)
#   - No loaders (full-batch message passing)
#
# Assumes you already have:
#   data: torch_geometric.data.Data with data.x, data.edge_index
# Optional:
#   data.edge_attr => edge features are not used here

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv
from torch_geometric.transforms import RandomLinkSplit


import pandas as pd
from torch_geometric.data import Data

# ----------------------------
# 0. Setup the data and graph
# ----------------------------

# 910 x 758 expression matrix (genes x cells)
expression = pd.read_csv("data/Expression.csv", index_col=0)
edge_list = pd.read_csv("data/Label.csv")[["TF", "Target"]]

# convert edge list to edge index of shape: [2, num_edges]
edge_index = torch.tensor(edge_list.to_numpy(), dtype=torch.long)

# this is the expression data
x = torch.tensor(expression.values, dtype=torch.float)


# this is the graph data
data = Data(x=x, edge_index=edge_index.t().contiguous())

data.num_nodes
data.num_edges
data.has_self_loops()
data.is_directed()


# ----------------------------
# 1. Split directed edges into train/val/test with negative sampling 
# ----------------------------
split = RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected=False,              # directed graph
    add_negative_train_samples=True,  # adds negatives to train split
    neg_sampling_ratio=1.0,           # 1:1 neg:pos
    # disjoint_train_ratio=0.0,       # optional; keep 0.0 for simplest transductive setup
)

train_data, val_data, test_data = split(data)

# ----------------------------
# 2) Model: 2-layer GAT encoder
# ----------------------------
class GeneLink(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.dropout = float(dropout)

        # Layer 1: multi-head attention (concat heads)
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=self.dropout,
            add_self_loops=False,
        )

        # Layer 2: single head, no concat -> out_channels exactly
        self.conv2 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,
            concat=False,
            dropout=self.dropout,
            add_self_loops=False,
        )

    def forward(self, x, edge_index, return_attention_weights=False):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if return_attention_weights:
           x, (edge_idx, alpha) = self.conv2(x, edge_index, return_attention_weights=True)
           return x, edge_idx, alpha
        else:
            x = self.conv2(x, edge_index)
        return x





def decode_dot(z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
    """Dot-product decoder for directed edges (src -> dst)."""
    src, dst = edge_label_index
    return (z[src] * z[dst]).sum(dim=-1)

# ----------------------------
# 3.  Train/eval utilities
# ----------------------------
@torch.no_grad()
def eval_split(model: nn.Module, split_data, device: torch.device):
    """
    Returns: (loss, auc, ap)
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    model.eval()
    split_data = split_data.to(device)

    z = model(split_data.x, split_data.edge_index)
    logits = decode_dot(z, split_data.edge_label_index)
    y = split_data.edge_label.float()

    loss = F.binary_cross_entropy_with_logits(logits, y).item()

    y_np = y.detach().cpu().numpy()
    p_np = logits.detach().cpu().numpy()

    auc = roc_auc_score(y_np, p_np)
    ap = average_precision_score(y_np, p_np)
    return loss, auc, ap

def train_one_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, train_data, device: torch.device):
    model.train()
    train_data = train_data.to(device)

    z = model(train_data.x, train_data.edge_index)
    logits = decode_dot(z, train_data.edge_label_index)
    y = train_data.edge_label.float()

    loss = F.binary_cross_entropy_with_logits(logits, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return float(loss.item())

# ----------------------------
# 4) Run training
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GeneLink(
    in_channels=train_data.num_features,
    hidden_channels=64,
    out_channels=64,
    heads=3,
    dropout=0.2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

best_val_auc = -1.0
best_state = None

for epoch in range(1, 401):
    train_loss = train_one_epoch(model, optimizer, train_data, device)
    val_loss, val_auc, val_ap = eval_split(model, val_data, device)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_auc={val_auc:.4f} | val_ap={val_ap:.4f}"
        )

# Load best model (by val AUC) and evaluate on test
if best_state is not None:
    model.load_state_dict(best_state)


model.eval()
z, edge_idx, alpha = model(test_data.x, test_data.edge_index, return_attention_weights=True)

test_loss, test_auc, test_ap = eval_split(model, test_data, device)
print(f"\nBest val AUC: {best_val_auc:.4f}")
print(f"Test | loss={test_loss:.4f} | auc={test_auc:.4f} | ap={test_ap:.4f}")


# why are there more edges coming out than in?
test_data.edge_index.shape


df = pd.DataFrame({
    "src": edge_idx[0].cpu().numpy(),
    "dst": edge_idx[1].cpu().numpy(),
    "attention": alpha.detach().cpu().numpy().flatten(),
})

#  4091 => you get more edges if you add self-loops