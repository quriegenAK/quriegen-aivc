"""
Neumann Series Cascade Propagation for AIVC v1.1.

Fixes the IFIT1 fold-change compression problem (predicted 2x vs actual 107x).

Root cause: ResponseDecoder MLP predicts direct drug effects only.
It does not propagate indirect cascade effects.
IFN-B hits JAK1 -> JAK1 activates STAT1 -> STAT1 activates IFIT1.
That is 3 hops. The MLP sees only the direct signal.

Solution (Tahoe TX1-CD approach): Neumann Series propagation.
Formula: effect = (I - W)^-1 * d_p ~ sum(W^k * d_p) for k=0..K

Where:
  W   = sparse gene regulatory network matrix
  d_p = direct perturbation effect vector (from ResponseDecoder MLP)
  K   = propagation steps (3 is sufficient for JAK-STAT cascade)

For IFIT1:
  d_p[JAK1] is large (direct IFN-B target)
  W[JAK1, STAT1] > 0 (JAK1 phosphorylates STAT1)
  W[STAT1, IFIT1] > 0 (STAT1 transcribes IFIT1)
  After 2 Neumann steps: IFIT1 receives signal proportional
  to d_p[JAK1] * W[JAK1,STAT1] * W[STAT1,IFIT1]
  This is what produces 107x, not 2x.

Training schedule (Tahoe TX1-CD two-stage):
  Stage 1 (epochs 1-10):  Freeze W, train ResponseDecoder only
  Stage 2 (epochs 11+):   Unfreeze W, train jointly with L1 penalty
"""
import torch
import torch.nn as nn
import numpy as np


class NeumannPropagation(nn.Module):
    """
    Propagates direct perturbation effects through the gene regulatory
    network using truncated Neumann Series approximation.

    Implements: effect = sum(W^k * d_p) for k=0..K
    Where W is a learned sparse GRN matrix.

    Architecture:
      - W is initialised from STRING PPI edge weights (confidence scores)
      - W is learnable (fine-tuned during training)
      - W is masked by the edge_index (only valid edges can have non-zero W)
      - L1 sparsity regularisation on W (matches Tahoe approach)
      - K=3 propagation steps by default (sufficient for 3-hop JAK-STAT)

    Args:
        n_genes: Number of genes (nodes in the GRN).
        edge_index: (2, n_edges) long tensor of graph edges.
        edge_attr: (n_edges,) float tensor of edge weights (STRING confidence).
            If None, all edges initialised with weight 0.01.
        K: Number of propagation steps. Default 3, max 5.
        lambda_l1: L1 sparsity penalty weight. Default 0.001.
    """

    def __init__(
        self,
        n_genes: int,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
        K: int = 3,
        lambda_l1: float = 0.001,
    ):
        super().__init__()

        if K > 5:
            raise ValueError(f"K={K} exceeds maximum 5. Computational cost scales as O(K * n_edges).")

        self.n_genes = n_genes
        self.K = K
        self.lambda_l1 = lambda_l1

        # Store edge structure (not learnable)
        self.register_buffer("edge_src", edge_index[0].long())
        self.register_buffer("edge_dst", edge_index[1].long())
        n_edges = edge_index.shape[1]

        # Initialise W from STRING PPI weights
        if edge_attr is not None:
            # Normalise confidence scores to small initial weights
            # STRING scores are 0-1000; normalise to ~0.001-0.01
            init_weights = edge_attr.float() / 100000.0
        else:
            init_weights = torch.full((n_edges,), 0.01)

        # Learnable edge weights (masked by edge_index structure)
        self.W = nn.Parameter(init_weights.clone())

    def forward(self, direct_effects: torch.Tensor) -> torch.Tensor:
        """
        Propagate direct effects through the GRN.

        Args:
            direct_effects: (batch, n_genes) or (n_genes,) — output of
                ResponseDecoder MLP (the delta/residual prediction).

        Returns:
            (batch, n_genes) or (n_genes,) — propagated effects after K
            Neumann steps. Same shape as input.
        """
        squeezed = False
        if direct_effects.dim() == 1:
            direct_effects = direct_effects.unsqueeze(0)
            squeezed = True

        propagated = direct_effects.clone()

        for k in range(self.K):
            propagated = propagated + self._sparse_matmul(propagated)

        if squeezed:
            propagated = propagated.squeeze(0)

        return propagated

    def _sparse_matmul(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sparse matrix-vector multiply using edge_index.
        Only computes over valid edges (masked by STRING PPI structure).

        Computes: result[dst] += W[edge] * x[src] for each edge

        Args:
            x: (batch, n_genes) — current state vector.

        Returns:
            (batch, n_genes) — propagated signal.
        """
        batch_size = x.shape[0]

        # Gather source values: (batch, n_edges)
        src_vals = x[:, self.edge_src]  # (batch, n_edges)

        # Multiply by edge weights: (batch, n_edges)
        weighted = src_vals * self.W.unsqueeze(0)  # broadcast W across batch

        # Scatter-add to destination nodes
        result = torch.zeros_like(x)  # (batch, n_genes)
        result.scatter_add_(1, self.edge_dst.unsqueeze(0).expand(batch_size, -1), weighted)

        return result

    def l1_penalty(self) -> torch.Tensor:
        """
        Compute L1 sparsity penalty on W.

        Returns:
            Scalar: lambda_l1 * ||W||_1
        """
        return self.lambda_l1 * self.W.abs().sum()

    def get_effective_W_density(self) -> float:
        """Report fraction of edges with |W| > 1e-6."""
        with torch.no_grad():
            active = (self.W.abs() > 1e-6).float().mean().item()
        return active

    def freeze_W(self):
        """Freeze W (Stage 1: train decoder only)."""
        self.W.requires_grad = False

    def unfreeze_W(self):
        """Unfreeze W (Stage 2: joint training with L1)."""
        self.W.requires_grad = True
