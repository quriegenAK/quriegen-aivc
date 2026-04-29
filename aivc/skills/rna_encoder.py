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

    def __init__(
        self,
        n_genes: int,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        n_lysis_categories: int = 0,
        lysis_cov_dim: int = 8,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_lysis_categories = n_lysis_categories
        self.lysis_cov_dim = lysis_cov_dim
        # PR #43 (logical): scVI-style categorical batch covariate. Opt-in
        # via n_lysis_categories>0; default 0 preserves pre-PR behavior so
        # existing checkpoints (no lysis_emb in state_dict) load unchanged.
        if n_lysis_categories > 0:
            self.lysis_emb = nn.Embedding(n_lysis_categories, lysis_cov_dim)
            input_dim = n_genes + lysis_cov_dim
        else:
            self.lysis_emb = None
            input_dim = n_genes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Decoder reconstructs raw n_genes (no covariate); the latent z
        # captures cov structure as needed.
        self.decoder = nn.Linear(latent_dim, n_genes)

    def forward(self, x: torch.Tensor, lysis_idx: torch.Tensor | None = None):
        """Return (latent, reconstruction). Structurally W-independent.

        lysis_idx : optional (B,) LongTensor of categorical batch covariates.
            Only consumed when self.lysis_emb is not None (i.e.
            n_lysis_categories > 0 at construction). Negative indices
            (-1 unknown sentinel) must be filtered upstream — nn.Embedding
            does not accept them.
        """
        if self.lysis_emb is not None and lysis_idx is not None:
            cov = self.lysis_emb(lysis_idx)
            x = torch.cat([x, cov], dim=1)
        z = self.net(x)
        return z, self.decoder(z)
