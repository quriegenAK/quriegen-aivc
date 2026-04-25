"""
PeakLevelATACEncoder — peak-matrix (sparse) encoder for the multiome
pretrain pathway.

Pipeline: TF-IDF normalization -> Linear LSI projection (svd_dim=50)
-> MLP -> latent of dim `attn_dim` (default 64 to match fusion).

This module is intentionally structurally independent of
`NeumannPropagation.W` and any causal-head parameter. It is dead code
until Phase 5 wires it into the fusion.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PeakLevelATACEncoder(nn.Module):
    """Encode a (batch, n_peaks) sparse-friendly peak matrix into a
    (batch, attn_dim) latent.

    Notes
    -----
    - GroupNorm required: BatchNorm leaks observational batch stats into
      causal head via shared running statistics (Phase 3 invariant).
    - Forward path MUST NOT reference NeumannPropagation.W or any
      causal-head parameter. Graph independence is verified by
      tests/test_peak_encoder.py.
    """

    def __init__(
        self,
        n_peaks: int,
        svd_dim: int = 50,
        hidden_dim: int = 128,
        attn_dim: int = 64,
        dropout: float = 0.1,
        groupnorm_groups: int = 8,
        tfidf_eps: float = 1e-6,
        apply_tfidf: bool = True,
    ):
        super().__init__()
        self.n_peaks = n_peaks
        self.svd_dim = svd_dim
        self.attn_dim = attn_dim
        self.tfidf_eps = tfidf_eps
        # apply_tfidf: True for raw peak counts (default, count-based semantics);
        # False for pre-normalized features like chromVAR motif deviations
        # (z-score-like, can be negative). TF-IDF on chromVAR scores produces
        # silent garbage — clamp_min(eps) saves the divisor but TF/IDF lose
        # their count-based biological meaning.
        self.apply_tfidf = apply_tfidf

        # Linear LSI projection (learnable surrogate for truncated SVD).
        self.lsi = nn.Linear(n_peaks, svd_dim, bias=False)

        # MLP head with GroupNorm (NOT BatchNorm).
        # GroupNorm required: BatchNorm leaks observational batch stats into
        # causal head via shared running statistics (Phase 3 invariant).
        g1 = min(groupnorm_groups, svd_dim)
        while svd_dim % g1 != 0 and g1 > 1:
            g1 -= 1
        g2 = min(groupnorm_groups, hidden_dim)
        while hidden_dim % g2 != 0 and g2 > 1:
            g2 -= 1

        self.mlp = nn.Sequential(
            nn.GroupNorm(g1, svd_dim),
            nn.Linear(svd_dim, hidden_dim),
            nn.GELU(),
            nn.GroupNorm(g2, hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, attn_dim),
        )

    def _tfidf(self, x: torch.Tensor) -> torch.Tensor:
        """TF-IDF normalization over (batch, n_peaks) counts.

        TF  = per-cell relative frequency (row-normalized).
        IDF = log(1 + N / df) computed per batch, where df is the number
              of cells in the batch that have nonzero counts at the peak.
        """
        x = x.float()
        # TF: row-normalize.
        row_sum = x.sum(dim=1, keepdim=True).clamp_min(self.tfidf_eps)
        tf = x / row_sum
        # IDF: per-batch.
        n_cells = x.shape[0]
        df = (x > 0).float().sum(dim=0, keepdim=True)  # (1, n_peaks)
        idf = torch.log1p(n_cells / (df + 1.0))
        return tf * idf

    def forward(self, peaks: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, n_peaks) -> (batch, attn_dim).

        Structural independence: this forward path does NOT reference
        NeumannPropagation.W or any causal-head parameter. Verified by
        tests/test_peak_encoder.py::test_graph_independence_from_W.
        """
        if peaks.is_sparse:
            peaks = peaks.to_dense()
        # apply_tfidf=False is required for chromVAR motif deviations or any
        # pre-normalized score-based input (negative values + non-count semantics).
        x = self._tfidf(peaks) if self.apply_tfidf else peaks.float()
        x = self.lsi(x)
        z = self.mlp(x)
        return z
