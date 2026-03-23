"""
perturbation_model.py — Perturbation-aware extension of the GeneLink model.

This module adds perturbation response prediction to the existing GeneLink
architecture WITHOUT modifying for_now.py. It wraps GeneLink and adds:
  - PerturbationEmbedding: learned condition vector added to node features
  - ResponseDecoder: projects GNN embedding back to gene expression space
  - PerturbationPredictor: full pipeline from ctrl expression to predicted stim

Architecture overview:
    ctrl_expression (n_genes,)
         |
    FeatureExpander (expand scalar per-gene to embedding_dim features)
         |
    PerturbationEmbedding (add learned perturbation vector)
         |
    GeneLink.forward (GAT message passing on gene graph)
         |
    ResponseDecoder (project back to n_genes expression values)
         |
    predicted_stim_expression (n_genes,)

Design rationale:
    The pretrained GeneLink was trained with input_dim = n_cells (12315).
    For perturbation prediction, each pseudo-bulk pair gives us 1 scalar per gene.
    We cannot reuse the pretrained weights directly (dimension mismatch).
    Instead, we instantiate a new GeneLink with input_dim = feature_dim and
    train it end-to-end for the perturbation task. The GAT architecture is
    preserved — same class, new weights learned for the new task.
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------
# GeneLink: same architecture as for_now.py (not imported to avoid side effects)
# ----------------------------
class GeneLink(nn.Module):
    """2-layer GAT for gene graph message passing. Identical to for_now.py."""

    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim,
                 output_dim, num_head1, num_head2, alpha, device, type, reduction):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden1_dim, heads=num_head1, dropout=0.01)
        self.conv2 = GATConv(
            hidden1_dim * num_head1, hidden2_dim, heads=num_head2, dropout=0.01
        )

    def forward(self, x, edge_index):
        """Forward pass through 2-layer GAT.

        Args:
            x: (n_genes, input_dim) — node features
            edge_index: (2, n_edges) — graph connectivity

        Returns:
            (n_genes, hidden2_dim * num_head2) — node embeddings
        """
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.01, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class FeatureExpander(nn.Module):
    """Expands scalar per-gene expression to a richer feature vector.

    Each gene node in the graph has a single scalar (its mean expression).
    This module projects it to embedding_dim features so the GAT has
    enough capacity to learn gene-specific patterns.

    Architecture:
        Linear(1, embedding_dim) → LayerNorm → GELU
    """

    def __init__(self, embedding_dim: int):
        """Initialise feature expander.

        Args:
            embedding_dim: output feature dimension per gene node.
        """
        super().__init__()
        self.expand = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand scalar features to embedding_dim.

        Args:
            x: (n_genes,) or (n_genes, 1) — scalar expression per gene.

        Returns:
            (n_genes, embedding_dim) — expanded features.
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # (n_genes, 1)
        return self.expand(x)


class PerturbationEmbedding(nn.Module):
    """Learned condition embedding that makes the GNN perturbation-aware.

    Encodes perturbation identity (e.g., IFN-b) as a learned vector that is
    added to node features before GAT message passing. This additive design
    preserves the original expression signal while shifting the feature space
    toward the perturbation context.

    The perturbation vector is the same for all genes — the model learns which
    genes respond to this shift via the GAT attention weights.

    This mechanism is analogous to the perturbation embeddings in CPA
    (Lotfollahi et al. 2023) and scGEN (Lotfollahi et al. 2019).

    Attributes:
        embedding: nn.Embedding(num_perturbations, embedding_dim)
    """

    def __init__(self, num_perturbations: int, embedding_dim: int):
        """Initialise perturbation embedding.

        Args:
            num_perturbations: number of distinct perturbations.
                2 for ctrl/stim in Week 2; extensible to 90+ for Parse 10M.
            embedding_dim: must match the feature dimension after FeatureExpander.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_perturbations, embedding_dim)

    def forward(self, x: torch.Tensor, pert_id: torch.Tensor) -> torch.Tensor:
        """Add perturbation embedding to node features.

        Args:
            x: (n_genes, embedding_dim) — expanded gene features.
            pert_id: scalar or (batch,) — perturbation index (0=ctrl, 1=stim).

        Returns:
            (n_genes, embedding_dim) — features with perturbation signal added.
        """
        if pert_id.dim() == 0:
            pert_id = pert_id.unsqueeze(0)
        # Get perturbation vector: (1, embedding_dim), broadcast across all genes
        pert_vec = self.embedding(pert_id[0:1])  # (1, embedding_dim)
        return x + pert_vec  # broadcast: (n_genes, embedding_dim)


class ResponseDecoder(nn.Module):
    """Projects GNN embedding back to gene expression space.

    Architecture:
        Linear(gnn_out_dim, hidden_dim) → LayerNorm → GELU
        Linear(hidden_dim, hidden_dim)  → LayerNorm → GELU
        Linear(hidden_dim, 1)           → output per gene

    Uses LayerNorm (not BatchNorm) because we have ~60 training pairs.
    Uses GELU (not ReLU) for smoother gradients on small positive values.
    """

    def __init__(self, gnn_out_dim: int, hidden_dim: int = 256):
        """Initialise response decoder.

        Args:
            gnn_out_dim: dimensionality of GNN output per node.
            hidden_dim: width of hidden layers in decoder.
        """
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(gnn_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode GNN embeddings to per-gene expression predictions.

        Args:
            z: (n_genes, gnn_out_dim) — GNN node embeddings.

        Returns:
            (n_genes,) — predicted expression per gene.
        """
        return self.decoder(z).squeeze(-1)


class PerturbationPredictor(nn.Module):
    """Full perturbation response prediction model.

    Pipeline:
        ctrl_expression (n_genes,)
             ↓
        FeatureExpander → (n_genes, feature_dim)
             ↓
        PerturbationEmbedding → (n_genes, feature_dim)  [shifted toward stim]
             ↓
        GeneLink.forward → (n_genes, gnn_out_dim)       [GAT message passing]
             ↓
        ResponseDecoder → (n_genes,)                     [predicted stim expression]

    The model predicts per-gene stimulated expression given control expression
    and a perturbation index. The GeneLink encoder learns gene-gene relationships
    through graph attention, and the perturbation embedding informs which
    relationships are relevant under the given condition.

    Attributes:
        feature_expander: FeatureExpander — scalar to vector
        pert_embedding: PerturbationEmbedding — condition-aware shift
        genelink: GeneLink — 2-layer GAT
        decoder: ResponseDecoder — embedding to expression
    """

    def __init__(self, n_genes: int, num_perturbations: int = 2,
                 feature_dim: int = 64, hidden1_dim: int = 64,
                 hidden2_dim: int = 32, num_head1: int = 3,
                 num_head2: int = 2, decoder_hidden: int = 256):
        """Initialise the full perturbation predictor.

        Args:
            n_genes: number of gene nodes in the graph.
            num_perturbations: number of perturbation conditions (2 for Week 2).
            feature_dim: dimension of expanded per-gene features.
            hidden1_dim: GAT layer 1 output dimension.
            hidden2_dim: GAT layer 2 output dimension.
            num_head1: number of attention heads in GAT layer 1.
            num_head2: number of attention heads in GAT layer 2.
            decoder_hidden: hidden dimension in response decoder.
        """
        super().__init__()
        self.n_genes = n_genes
        self.feature_dim = feature_dim

        # Feature expansion: scalar expression → feature_dim
        self.feature_expander = FeatureExpander(feature_dim)

        # Perturbation embedding: adds condition signal
        self.pert_embedding = PerturbationEmbedding(num_perturbations, feature_dim)

        # GeneLink encoder: GAT message passing
        # input_dim = feature_dim (from expander + perturbation)
        gnn_out_dim = hidden2_dim * num_head2
        self.genelink = GeneLink(
            input_dim=feature_dim,
            hidden1_dim=hidden1_dim,
            hidden2_dim=hidden2_dim,
            hidden3_dim=16,  # unused in forward but required by signature
            output_dim=8,    # unused in forward but required by signature
            num_head1=num_head1,
            num_head2=num_head2,
            alpha=0.2,
            device="cpu",
            type="MLP",
            reduction="mean",
        )

        # Response decoder: GNN embedding → predicted expression
        self.decoder = ResponseDecoder(gnn_out_dim, decoder_hidden)

    def forward(self, x_ctrl: torch.Tensor, edge_index: torch.Tensor,
                pert_id: torch.Tensor,
                cell_type_ids: torch.Tensor = None) -> torch.Tensor:
        """Predict stimulated expression from control expression.

        Args:
            x_ctrl: (n_genes,) or (n_genes, 1) — mean ctrl expression per gene.
            edge_index: (2, n_edges) — graph connectivity.
            pert_id: scalar or (1,) — perturbation index (1 for stim).
            cell_type_ids: optional scalar or (1,) — cell type index.
                          If None: cell type embedding is zero (backward compatible).
                          If provided: adds cell type embedding to features.

        Returns:
            (n_genes,) — predicted stimulated expression per gene.
        """
        # 1. Expand scalar features
        x = self.feature_expander(x_ctrl)  # (n_genes, feature_dim)

        # 2. Add perturbation signal
        x = self.pert_embedding(x, pert_id)  # (n_genes, feature_dim)

        # 3. Add cell type signal (if available and embedding exists)
        if cell_type_ids is not None and hasattr(self, "cell_type_embedding"):
            x = self.cell_type_embedding(x, cell_type_ids)

        # 4. GAT message passing
        z = self.genelink(x, edge_index)  # (n_genes, gnn_out_dim)

        # 5. Decode to expression (direct effects)
        pred = self.decoder(z)  # (n_genes,)

        # 6. Neumann cascade propagation (v1.1 — if module attached)
        if hasattr(self, "neumann") and self.neumann is not None:
            pred = self.neumann(pred)

        return pred

    def forward_batch(self, X_ctrl: torch.Tensor, edge_index: torch.Tensor,
                      pert_id: torch.Tensor,
                      cell_type_ids: torch.Tensor = None) -> torch.Tensor:
        """Predict stimulated expression for a batch of pairs.

        Uses PyG graph batching to run ALL pairs through the GAT in a single
        forward pass. Each pair becomes a separate subgraph with the same
        topology (edge_index) but different node features (expression values).

        This is O(1) GNN forward passes per batch instead of O(n_pairs).
        Critical for OT training with thousands of pairs.

        Args:
            X_ctrl: (n_pairs, n_genes) — ctrl expression for each pair.
            edge_index: (2, n_edges) — shared graph connectivity.
            pert_id: scalar or (n_pairs,) — perturbation index per pair.
            cell_type_ids: optional (n_pairs,) — cell type index per pair.
                          If None: cell type embedding is zero (backward compatible).

        Returns:
            (n_pairs, n_genes) — predicted stim expression for each pair.
        """
        n_pairs = X_ctrl.shape[0]
        n_genes = X_ctrl.shape[1]

        # --- Vectorised feature expansion for all pairs at once ---
        # X_ctrl: (n_pairs, n_genes) → reshape to (n_pairs * n_genes, 1) for Linear
        x_flat = X_ctrl.reshape(n_pairs * n_genes, 1)
        x_expanded = self.feature_expander.expand(x_flat)  # (n_pairs * n_genes, feature_dim)
        x_expanded = x_expanded.reshape(n_pairs, n_genes, -1)  # (n_pairs, n_genes, feature_dim)

        # --- Add perturbation embedding (broadcast to all genes, per pair) ---
        if pert_id.dim() == 0:
            pert_id = pert_id.unsqueeze(0)
        if pert_id.numel() == 1:
            # Same pert for all pairs
            pert_vec = self.pert_embedding.embedding(pert_id[:1])  # (1, feature_dim)
            x_expanded = x_expanded + pert_vec.unsqueeze(1)  # broadcast: (n_pairs, n_genes, feature_dim)
        else:
            pert_vecs = self.pert_embedding.embedding(pert_id)  # (n_pairs, feature_dim)
            x_expanded = x_expanded + pert_vecs.unsqueeze(1)

        # --- Add cell type embedding (if available) ---
        if cell_type_ids is not None and hasattr(self, "cell_type_embedding"):
            if cell_type_ids.dim() == 0:
                cell_type_ids = cell_type_ids.unsqueeze(0)
            ct_vecs = self.cell_type_embedding.embedding(cell_type_ids)  # (n_pairs, feature_dim)
            x_expanded = x_expanded + ct_vecs.unsqueeze(1)

        # --- Build batched graph for PyG ---
        # Stack all pairs' node features: (n_pairs * n_genes, feature_dim)
        x_batched = x_expanded.reshape(n_pairs * n_genes, -1)

        # Offset edge_index for each subgraph: pair i's nodes are [i*n_genes, (i+1)*n_genes)
        n_edges = edge_index.shape[1]
        offsets = torch.arange(n_pairs, device=edge_index.device) * n_genes  # (n_pairs,)
        # Repeat edge_index n_pairs times, add per-pair offsets
        edge_index_batched = edge_index.repeat(1, n_pairs)  # (2, n_pairs * n_edges)
        offset_repeated = offsets.repeat_interleave(n_edges)  # (n_pairs * n_edges,)
        edge_index_batched = edge_index_batched + offset_repeated.unsqueeze(0)

        # --- Single GAT forward pass on batched graph ---
        z_batched = self.genelink(x_batched, edge_index_batched)  # (n_pairs * n_genes, gnn_out_dim)

        # --- Decode all at once (direct effects) ---
        pred_batched = self.decoder(z_batched)  # (n_pairs * n_genes,)

        # Reshape back to (n_pairs, n_genes)
        pred = pred_batched.reshape(n_pairs, n_genes)

        # --- Neumann cascade propagation (v1.1 — if module attached) ---
        if hasattr(self, "neumann") and self.neumann is not None:
            pred = self.neumann(pred)

        return pred


# =========================================================================
# Cell Type Embedding (Week 3 extension)
# =========================================================================

class CellTypeEmbedding(nn.Module):
    """Learned cell-type-specific embedding.

    Same design as PerturbationEmbedding but encodes biological cell identity.
    A monocyte and a B cell respond differently to IFN-b.
    This embedding gives the model that biological context.

    Cell types in Kang 2018:
        0: CD4 T cells        1: CD8 T cells       2: NK cells
        3: B cells            4: CD14+ Monocytes   5: Dendritic cells
        6: Megakaryocytes     7: FCGR3A+ Monocytes 8: unknown

    Extensibility: designed for 20 cell types to accommodate future data.
    When QurieGen Phase 2 data arrives (25 donors, more conditions),
    this embedding handles new cell types without architectural changes.

    Architecture:
        nn.Embedding(num_cell_types, embedding_dim)
        Combined with features via addition (not concatenation).
        Reason: addition preserves dimensionality, keeps input_dim compatible
                with pretrained GeneLink weights.
    """

    CELL_TYPE_MAP = {
        "CD4 T cells": 0,
        "CD8 T cells": 1,
        "NK cells": 2,
        "B cells": 3,
        "CD14+ Monocytes": 4,
        "Dendritic cells": 5,
        "Megakaryocytes": 6,
        "FCGR3A+ Monocytes": 7,
    }

    def __init__(self, num_cell_types: int = 20, embedding_dim: int = 64):
        """Initialise cell type embedding.

        Args:
            num_cell_types: number of cell type slots (20 for future expansion).
            embedding_dim: must match the feature dimension after FeatureExpander.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_cell_types, embedding_dim)

    def forward(self, x: torch.Tensor, cell_type_id: torch.Tensor) -> torch.Tensor:
        """Add cell type embedding to node features.

        Args:
            x: (n_genes, embedding_dim) — expanded gene features.
            cell_type_id: scalar or (1,) — cell type index.

        Returns:
            (n_genes, embedding_dim) — features with cell type signal added.
        """
        if cell_type_id.dim() == 0:
            cell_type_id = cell_type_id.unsqueeze(0)
        ct_vec = self.embedding(cell_type_id[0:1])  # (1, embedding_dim)
        return x + ct_vec  # broadcast across all genes

    @classmethod
    def encode_cell_types(cls, cell_type_labels) -> torch.Tensor:
        """Convert string cell type labels to embedding indices.

        Unknown cell types: map to index 8 ('unknown') with a logged warning.
        Does not crash on unseen cell types — this will be called with Phase 2 data.

        Args:
            cell_type_labels: list of string cell type names.

        Returns:
            (n_cells,) long tensor of cell type indices.
        """
        indices = []
        unknown_types = set()
        for label in cell_type_labels:
            idx = cls.CELL_TYPE_MAP.get(label, 8)  # 8 = unknown
            if idx == 8 and label not in cls.CELL_TYPE_MAP:
                unknown_types.add(label)
            indices.append(idx)
        if unknown_types:
            print(f"  WARNING: Unknown cell types mapped to index 8: {unknown_types}")
        return torch.tensor(indices, dtype=torch.long)


def build_cell_type_index(adata) -> tuple:
    """Extract cell type indices from AnnData obs.

    Checks these columns in order: 'cell_type', 'celltype', 'cell.type',
    'predicted.celltype.l2', 'leiden', 'louvain'.
    Logs which column was found and used.
    Maps string labels to integers using CellTypeEmbedding.CELL_TYPE_MAP.
    Unknown labels: map to 8, log the unknown label and its frequency.

    Args:
        adata: AnnData object with obs columns.

    Returns:
        (cell_type_indices, cell_type_labels, column_name)
        cell_type_indices: np.ndarray of shape (n_cells,) with integer indices.
        cell_type_labels: list of string labels.
        column_name: which obs column was used.
    """
    import numpy as np

    candidates = ["cell_type", "celltype", "cell.type",
                  "predicted.celltype.l2", "leiden", "louvain"]
    col_name = None
    for c in candidates:
        if c in adata.obs.columns:
            col_name = c
            break
    if col_name is None:
        raise ValueError(
            f"No cell type column found in adata.obs. "
            f"Checked: {candidates}. "
            f"Available: {list(adata.obs.columns)}"
        )
    print(f"  Cell type column: '{col_name}'")

    labels = adata.obs[col_name].tolist()
    indices = []
    unknown_counts = {}
    for label in labels:
        idx = CellTypeEmbedding.CELL_TYPE_MAP.get(label, 8)
        if idx == 8 and label not in CellTypeEmbedding.CELL_TYPE_MAP:
            unknown_counts[label] = unknown_counts.get(label, 0) + 1
        indices.append(idx)

    if unknown_counts:
        print(f"  WARNING: Unknown cell types mapped to index 8:")
        for lab, count in sorted(unknown_counts.items()):
            print(f"    {lab}: {count} cells")

    return np.array(indices), labels, col_name


# =========================================================================
# Unit test
# =========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("UNIT TEST: PerturbationPredictor")
    print("=" * 60)

    torch.manual_seed(SEED)
    n_genes = 3010
    n_pairs = 10
    n_edges = 500

    # Mock data
    x_ctrl = torch.randn(n_pairs, n_genes)
    edge_index = torch.randint(0, n_genes, (2, n_edges))
    pert_id = torch.tensor([1])  # stim

    # Instantiate model
    model = PerturbationPredictor(n_genes=n_genes)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward pass (single pair)
    try:
        out_single = model(x_ctrl[0], edge_index, pert_id)
        assert out_single.shape == (n_genes,), \
            f"Single output shape {out_single.shape} != expected ({n_genes},)"
        assert not torch.isnan(out_single).any(), "NaN in single output"
        assert not torch.isinf(out_single).any(), "Inf in single output"
        print(f"  Single pair: PASS — output shape {out_single.shape}")
    except Exception as e:
        print(f"  Single pair: FAIL — {e}")
        raise

    # Forward pass (batch)
    try:
        out_batch = model.forward_batch(x_ctrl, edge_index, pert_id)
        assert out_batch.shape == (n_pairs, n_genes), \
            f"Batch output shape {out_batch.shape} != expected ({n_pairs}, {n_genes})"
        assert not torch.isnan(out_batch).any(), "NaN in batch output"
        assert not torch.isinf(out_batch).any(), "Inf in batch output"
        print(f"  Batch ({n_pairs} pairs): PASS — output shape {out_batch.shape}")
    except Exception as e:
        print(f"  Batch: FAIL — {e}")
        raise

    # Gradient flow test
    try:
        loss = out_batch.sum()
        loss.backward()
        n_grad = sum(1 for p in model.parameters() if p.grad is not None)
        n_total = sum(1 for p in model.parameters())
        assert n_grad == n_total, f"Only {n_grad}/{n_total} params have gradients"
        print(f"  Gradient flow: PASS — {n_grad}/{n_total} params have gradients")
    except Exception as e:
        print(f"  Gradient flow: FAIL — {e}")
        raise

    # Backward compatibility: forward without cell_type_ids
    try:
        model.zero_grad()
        out_no_ct = model(x_ctrl[0], edge_index, pert_id, cell_type_ids=None)
        assert out_no_ct.shape == (n_genes,)
        print(f"  Backward compat (no cell_type): PASS")
    except Exception as e:
        print(f"  Backward compat: FAIL — {e}")
        raise

    # CellTypeEmbedding test
    print("\n  CellTypeEmbedding tests:")
    try:
        ct_emb = CellTypeEmbedding(num_cell_types=20, embedding_dim=64)
        labels = ["CD14+ Monocytes", "B cells", "CD4 T cells", "UnknownType"]
        ct_idx = CellTypeEmbedding.encode_cell_types(labels)
        assert ct_idx.shape == (4,)
        assert ct_idx[0].item() == 4  # CD14+ Monocytes
        assert ct_idx[1].item() == 3  # B cells
        assert ct_idx[2].item() == 0  # CD4 T cells
        assert ct_idx[3].item() == 8  # unknown
        print(f"    encode_cell_types: PASS — {ct_idx.tolist()}")

        # Forward
        x_test = torch.randn(100, 64)
        ct_id = torch.tensor([4])
        out_ct = ct_emb(x_test, ct_id)
        assert out_ct.shape == (100, 64)
        print(f"    forward: PASS — output shape {out_ct.shape}")
    except Exception as e:
        print(f"    FAIL — {e}")
        raise

    # Model with cell type embedding
    print("\n  PerturbationPredictor with cell type embedding:")
    try:
        model_ct = PerturbationPredictor(n_genes=n_genes)
        model_ct.cell_type_embedding = CellTypeEmbedding(
            num_cell_types=20, embedding_dim=model_ct.feature_dim
        )
        ct_ids = torch.tensor([4, 3, 0, 1, 2, 5, 6, 7, 4, 3])
        out_with_ct = model_ct.forward_batch(x_ctrl, edge_index, pert_id, ct_ids)
        assert out_with_ct.shape == (n_pairs, n_genes)
        assert not torch.isnan(out_with_ct).any()
        print(f"    Batch with cell_type_ids: PASS — shape {out_with_ct.shape}")
    except Exception as e:
        print(f"    FAIL — {e}")
        raise

    print()
    print("OVERALL: PASS — All unit tests passed")
