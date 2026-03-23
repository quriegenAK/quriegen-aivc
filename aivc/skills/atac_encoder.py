"""
ATACSeqEncoder — Encodes per-cell TF motif enrichment scores for AIVC v3.0.

INPUT: chromvar_scores (batch, n_tfs) — continuous chromVAR deviation scores
       from mdata['atac'].obsm['chromvar_scores']
       n_tfs ~ 700-900 (JASPAR 2024 CORE vertebrates)

NOT: raw peak counts. Peaks are 100,000+ sparse binary values.
     chromVAR scores are ~800 continuous dense values.
     The SCM needs named TF signals, not genomic coordinates.

Architecture:
  Input (n_tfs) -> LayerNorm
  -> Linear(n_tfs, 256) -> GELU -> Dropout(0.1)
  -> Linear(256, 128) -> GELU -> Dropout(0.1)
  -> Linear(128, 64) -> LayerNorm
  Output: 64-dim ATAC embedding

Why MLP not GAT for ATAC?
  - TF motif scores are already aggregated signals (not per-peak)
  - Graph structure is captured via peak->gene edges in the SCM
  - GAT on 800 TF nodes would overfit on small QuRIE-seq batches
  - Matches PhosphoEncoder architecture (also MLP->64-dim)
"""
import torch
import torch.nn as nn


class ATACSeqEncoder(nn.Module):
    """
    Encodes per-cell TF motif enrichment scores into a 64-dim representation.

    Args:
        n_tfs: Number of TF motif scores (input dimension).
        embed_dim: Output embedding dimension. Default 64 (matches PhosphoEncoder).
        hidden_dim: Hidden layer width. Default 256.
        dropout: Dropout rate. Default 0.1.
    """

    def __init__(
        self,
        n_tfs: int,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_tfs = n_tfs
        self.embed_dim = embed_dim

        self.encoder = nn.Sequential(
            nn.LayerNorm(n_tfs),
            nn.Linear(n_tfs, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Xavier uniform init
        self._init_weights()

    def _init_weights(self):
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        chromvar_scores: torch.Tensor,
        atac_quality_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Encode TF motif scores to 64-dim ATAC embedding.

        Args:
            chromvar_scores: (batch, n_tfs) — chromVAR deviation scores.
            atac_quality_weights: (batch,) — WNN ATAC modality weight per cell.
                If weight < 0.2: poor ATAC quality. Output multiplied by weight
                to de-emphasise. Derived from preprocessing step04.

        Returns:
            (batch, embed_dim) — ATAC embedding.
        """
        embedding = self.encoder(chromvar_scores)  # (batch, embed_dim)

        # De-emphasise cells with poor ATAC quality
        if atac_quality_weights is not None:
            # Clamp weights to [0, 1] for safety
            weights = atac_quality_weights.clamp(0.0, 1.0)
            embedding = embedding * weights.unsqueeze(-1)  # (batch, embed_dim)

        return embedding
