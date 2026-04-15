"""
MultiomePretrainHead — cross-modal pretraining head for paired RNA + ATAC.

Two projection MLPs (one per modality) + a linear peak-to-gene regressor
support InfoNCE contrastive alignment and a peak-to-gene auxiliary loss.

Structural independence
-----------------------
This module is structurally independent of ``NeumannPropagation.W`` and any
causal-head parameter, mirroring the Phase 4 PeakLevelATACEncoder contract.
The forward path allocates only its own ``nn.Linear`` weights and does not
touch the causal-head parameter matrix — a backward() through this head on
a sibling-style model leaves ``NeumannPropagation.W.grad`` as ``None``
(verified by tests/test_pretrain_head.py).

Dead code until scripts/pretrain_multiome.py wires it in. train_week3.py
remains unaware of this module (Phase 4 dead-code invariant).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


# Integration contract with aivc.data.multiome_loader.MultiomeLoader.
# Keep in sync with MultiomeLoader.__getitem__ — any future key changes
# must update both sides together.
REQUIRED_BATCH_KEYS: Tuple[str, ...] = ("rna", "atac_peaks")


class MultiomePretrainHead(nn.Module):
    """Two projection MLPs + peak-to-gene linear regressor.

    Parameters
    ----------
    rna_dim : int
        Dimensionality of the RNA encoder latent.
    atac_dim : int
        Dimensionality of the PeakLevelATACEncoder latent.
    proj_dim : int
        Shared contrastive projection dimensionality.
    n_genes : int
        Output width of the peak-to-gene regressor (gene count on the
        RNA side of the multiome pair).
    hidden_dim : int
        Hidden width of each projection MLP.
    """

    def __init__(
        self,
        rna_dim: int,
        atac_dim: int,
        proj_dim: int = 128,
        n_genes: int = 2000,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.rna_dim = rna_dim
        self.atac_dim = atac_dim
        self.proj_dim = proj_dim
        self.n_genes = n_genes

        # Contrastive projections (MLP), symmetric per modality.
        self.rna_proj = nn.Sequential(
            nn.Linear(rna_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )
        self.atac_proj = nn.Sequential(
            nn.Linear(atac_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

        # Peak-to-gene auxiliary regressor — linear only (no nonlinearity)
        # so the contract with Phase 6 (peak accessibility -> gene proxy)
        # stays simple.
        self.peak_to_gene = nn.Linear(atac_dim, n_genes)

    @staticmethod
    def _validate_batch_keys(batch: Dict[str, Any]) -> None:
        """Validate that the batch dict matches the MultiomeLoader contract.

        Raises
        ------
        ValueError
            With a clear message naming every missing key. This MUST raise
            before any tensor op so failures surface at the contract
            boundary, not deep in an encoder or matmul.
        """
        if not isinstance(batch, dict):
            raise ValueError(
                f"MultiomePretrainHead expects a dict (MultiomeLoader "
                f"contract). Got {type(batch).__name__}."
            )
        missing = [k for k in REQUIRED_BATCH_KEYS if k not in batch]
        if missing:
            raise ValueError(
                f"MultiomePretrainHead batch missing required key(s) "
                f"{missing!r}. Expected MultiomeLoader contract keys: "
                f"{list(REQUIRED_BATCH_KEYS)!r}."
            )

    def forward(
        self,
        batch: Dict[str, Any],
        rna_latent: torch.Tensor,
        atac_latent: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Structural independence: this forward path does NOT reference
        # NeumannPropagation.W or any causal-head parameter.
        # (Same pattern as PeakLevelATACEncoder.forward.)
        self._validate_batch_keys(batch)

        z_rna = self.rna_proj(rna_latent)
        z_atac = self.atac_proj(atac_latent)

        # L2-normalize for cosine-based InfoNCE.
        z_rna = nn.functional.normalize(z_rna, dim=-1)
        z_atac = nn.functional.normalize(z_atac, dim=-1)

        gene_pred = self.peak_to_gene(atac_latent)

        return {
            "z_rna": z_rna,
            "z_atac": z_atac,
            "gene_pred": gene_pred,
        }
