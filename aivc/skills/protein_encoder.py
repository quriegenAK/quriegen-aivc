"""
aivc/skills/protein_encoder.py — Surface protein encoder for AIVC.

Encodes per-cell antibody-derived tag (ADT) counts from CITE-seq or
QuRIE-seq into a 128-dim representation aligned with the RNA encoder.

Biology:
  QuRIE-seq measures ~200 surface proteins per cell simultaneously with
  RNA, Phospho, and ATAC. Surface proteins (CD3, CD14, CD19, CD4, CD8,
  etc.) provide direct cell-type identity signal and activation state.
  Unlike RNA (minutes latency), surface protein levels reflect stable
  cell-type identity and slowly changing activation state (hours-days).
  In the temporal fusion module: Protein is at t=3 (latest in causal order).

Architecture:
  ADT matrix (batch, n_proteins)
      -> CLR normalisation (Centred Log-Ratio, standard for ADT data)
      -> LayerNorm
      -> Linear(n_proteins, hidden) + GELU + Dropout
      -> Linear(hidden, hidden) + GELU + Dropout
      -> Linear(hidden, 128)
      -> Cross-attention alignment to RNA (optional, for contrastive loss)
      -> (batch, 128)

Status: CODED. Awaiting QuRIE-seq ADT data (May 2026).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CLRNorm(nn.Module):
    """
    Centred Log-Ratio normalisation for ADT count data.

    CLR is the standard for CITE-seq / QuRIE-seq antibody counts.
    Formula: CLR(x_i) = log(x_i) - mean(log(x_i)) per cell.

    Why NOT log1p (which works for RNA):
      ADT counts are NOT sparse (every protein has counts > 0).
      log1p biases small values; CLR treats all proteins symmetrically.
      Reference: Stoeckius et al. 2017 (CITE-seq paper).
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, n_proteins) -> (batch, n_proteins) CLR-normalised."""
        x_safe = x.clamp(min=self.eps)
        log_x = torch.log(x_safe)
        geo_mean_log = log_x.mean(dim=-1, keepdim=True)
        return log_x - geo_mean_log


class ProteinEncoder(nn.Module):
    """
    Surface protein (ADT) encoder for QuRIE-seq multi-modal data.
    Output dimension is 128 to match RNA encoder and TemporalCrossModalFusion.

    The cross-attention module aligns protein embeddings to the RNA
    embedding space when rna_emb is provided — required for contrastive loss.

    Args:
        n_proteins: ADT panel size (~200 for QuRIE-seq).
        embed_dim:  Output dimension. MUST be 128. Default: 128.
        hidden_dim: Hidden layer width. Default: 256.
        n_heads:    Cross-attention heads. Default: 4.
        dropout:    Dropout rate. Default: 0.1.
    """

    def __init__(
        self,
        n_proteins: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
        n_lysis_categories: int = 0,
        lysis_cov_dim: int = 8,
    ):
        super().__init__()
        self.n_proteins = n_proteins
        self.embed_dim = embed_dim
        self.n_lysis_categories = n_lysis_categories
        self.lysis_cov_dim = lysis_cov_dim

        self.clr_norm = CLRNorm()
        self.layer_norm = nn.LayerNorm(n_proteins)

        self.encoder = nn.Sequential(
            nn.Linear(n_proteins, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(embed_dim)

        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)

        # PR #43 (logical): scVI-style categorical batch covariate.
        # Cross-attn here uses single-token (B, 1, embed_dim), so
        # token-prepend doesn't apply. Use additive shift: project
        # lysis_emb -> embed_dim and add to encoder output before
        # cross-attn. Back-compat: lysis_emb is None when
        # n_lysis_categories=0 (default), forward path unchanged.
        if n_lysis_categories > 0:
            self.lysis_emb = nn.Embedding(n_lysis_categories, lysis_cov_dim)
            self.lysis_to_embed = nn.Linear(lysis_cov_dim, embed_dim)
        else:
            self.lysis_emb = None
            self.lysis_to_embed = None

    def forward(
        self,
        adt: torch.Tensor,
        rna_emb: Optional[torch.Tensor] = None,
        lysis_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode ADT counts into a 128-dim protein embedding.

        Args:
            adt:     (batch, n_proteins) — raw ADT counts.
            rna_emb: (batch, 128) — RNA encoder output (optional).
                     When provided: cross-attention aligns to RNA space.
                     When None: pure MLP encoding.
            lysis_idx: optional (batch,) LongTensor — categorical batch
                     covariate (LLL=0, DIG=1). Only consumed when this
                     encoder was constructed with n_lysis_categories>0.
        Returns:
            (batch, 128)
        """
        x = self.clr_norm(adt)
        x = self.layer_norm(x)
        x = self.encoder(x)

        # PR #43 (logical): additive batch-covariate shift.
        if self.lysis_emb is not None and lysis_idx is not None:
            cov = self.lysis_emb(lysis_idx)            # (B, cov_dim)
            cov_proj = self.lysis_to_embed(cov)        # (B, embed_dim)
            x = x + cov_proj

        if rna_emb is not None:
            x_q = x.unsqueeze(1)
            rna_k = rna_emb.unsqueeze(1)
            attn_out, _ = self.cross_attn(query=x_q, key=rna_k, value=rna_k)
            x = self.cross_attn_norm(x + attn_out.squeeze(1))

        x = self.output_proj(x)
        x = self.output_norm(x)
        return x

    def get_protein_attention_weights(
        self,
        adt: torch.Tensor,
        rna_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Return cross-attention weights (protein -> RNA alignment)."""
        self.eval()
        with torch.no_grad():
            x = self.clr_norm(adt)
            x = self.layer_norm(x)
            x = self.encoder(x)
            x_q = x.unsqueeze(1)
            rna_k = rna_emb.unsqueeze(1)
            _, attn_weights = self.cross_attn(
                query=x_q, key=rna_k, value=rna_k,
                need_weights=True, average_attn_weights=True,
            )
        return attn_weights
