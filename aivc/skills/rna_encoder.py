"""
rna_encoder.py — cell-level RNA MLP encoder shared between Phase 5
pretraining and Phase 6 fine-tuning.

Architecture: (batch, n_genes) -> Linear -> GELU -> Linear -> (batch, latent_dim),
with a companion Linear decoder back to gene-expression space for masked
reconstruction.

Phase 5 promotes this class from scripts/pretrain_multiome.py so its
state_dict schema is stable and publicly importable. Phase 6 will
instantiate SimpleRNAEncoder directly in its fine-tuning entrypoint and
load the pretrained weights via torch.nn.Module.load_state_dict.

Structural independence
-----------------------
Forward path allocates only its own nn.Linear + nn.GELU parameters. No
reference to NeumannPropagation.W or any causal-head parameter. Graph
independence is identical to PeakLevelATACEncoder (Phase 4 invariant).

Architectural note for Phase 6
------------------------------
PerturbationPredictor (perturbation_model.py) operates on per-gene
scalar features through a GAT over a gene-graph: its input is
(n_genes,) scalar-per-gene, not (batch, n_genes). SimpleRNAEncoder
is therefore NOT composed into PerturbationPredictor; Phase 6 must
introduce a cell-level fine-tuning head that calls SimpleRNAEncoder
directly. Forcing a refactor of PerturbationPredictor to add a
self.rna_encoder slot would either (a) create an unused submodule
or (b) rewrite PerturbationPredictor into a different architecture,
both of which would regress existing tests — see Phase 5 PR body,
"Phase 6 interface resolution".
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SimpleRNAEncoder(nn.Module):
    """Cell-level RNA MLP encoder + linear reconstruction decoder.

    Parameters
    ----------
    n_genes : int
        Input width (number of genes).
    hidden_dim : int
        Hidden layer width.
    latent_dim : int
        Output latent dimensionality.
    """

    def __init__(self, n_genes: int, hidden_dim: int = 256, latent_dim: int = 128):
        super().__init__()
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Linear(latent_dim, n_genes)

    def forward(self, x: torch.Tensor):
        """Return (latent, reconstruction). Structurally W-independent."""
        z = self.net(x)
        return z, self.decoder(z)
