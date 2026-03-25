"""
aivc/skills/phospho_encoder.py — Phosphoproteomic encoder for AIVC.

Encodes per-cell phosphopeptide intensities into a 64-dim representation
that captures active kinase signalling state.

Biology:
  In IFN-B stimulation: within 2 minutes of receptor binding, JAK1 and
  TYK2 phosphorylate STAT1 at Y701 and STAT2 at Y690. Phospho data
  captures this within minutes; RNA after ~30 min; Protein after hours.
  In the temporal fusion module: Phospho is at t=1.

Architecture:
  Phosphopeptide intensities (batch, n_sites)
      -> Log-normalise + z-score (phospho-specific normalisation)
      -> LayerNorm
      -> Linear(n_sites, hidden) + GELU + Dropout
      -> KS graph attention (optional — PhosphoSitePlus edges)
      -> Linear(hidden, 64)
      -> (batch, 64)

Status: CODED. Awaiting QuRIE-seq phospho panel data (May 2026).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PhosphoNorm(nn.Module):
    """
    Log-normalisation + z-score for phosphopeptide intensity data.

    Why not CLR (used for ADT):
      Phospho intensities are NOT compositional — total phospho signal
      per cell is biologically meaningful. CLR would remove this signal.
      Z-score normalises each site independently.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, n_sites) -> (batch, n_sites) log-normalised, z-scored."""
        x = torch.log1p(x.clamp(min=0.0))
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True).clamp(min=self.eps)
        return (x - mean) / std


class KSGraphAttention(nn.Module):
    """
    Kinase-substrate graph attention layer.
    Uses PhosphoSitePlus kinase->substrate edges as structural prior.

    When edge_index=None: returns input unchanged (clean fallback).
    """

    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:          (batch, seq, hidden_dim) — site embeddings.
            edge_index: (2, n_edges) — kinase->substrate edges. None = identity.
        Returns:
            (batch, seq, hidden_dim)
        """
        if edge_index is None:
            return x

        batch, n, d = x.shape
        head_dim = self.head_dim

        Q = self.q_proj(x).view(batch, n, self.n_heads, head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch, n, self.n_heads, head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch, n, self.n_heads, head_dim).transpose(1, 2)

        scale = head_dim ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # Adjacency mask: only attend over known K->S edges
        mask = torch.full((n, n), float('-inf'), device=x.device, dtype=x.dtype)
        # Self-attention always allowed
        mask.fill_diagonal_(0.0)
        src, dst = edge_index[0], edge_index[1]
        valid_src = src[src < n]
        valid_dst = dst[dst < n]
        n_valid = min(len(valid_src), len(valid_dst))
        if n_valid > 0:
            mask[valid_dst[:n_valid], valid_src[:n_valid]] = 0.0

        scores = scores + mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch, n, d)
        out = self.out_proj(out)
        return self.norm(x + out)


class PhosphoEncoder(nn.Module):
    """
    Phosphoproteomic encoder for QuRIE-seq multi-modal data.
    Output dimension is 64 to match TemporalCrossModalFusion phospho_dim.

    Args:
        n_sites:      Number of phosphosites. Default: 400.
        embed_dim:    Output dim. MUST be 64. Default: 64.
        hidden_dim:   Hidden layer width. Default: 256.
        n_heads:      Graph attention heads. Default: 4.
        dropout:      Dropout rate. Default: 0.1.
        use_ks_graph: Include KSGraphAttention. Default: True.
        align_to_rna: If True, add projection to 128-dim for contrastive
                      loss with RNA encoder. Default: False.
    """

    def __init__(
        self,
        n_sites: int = 400,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_ks_graph: bool = True,
        align_to_rna: bool = False,
    ):
        super().__init__()
        self.n_sites = n_sites
        self.embed_dim = embed_dim
        self.use_ks_graph = use_ks_graph

        self.phospho_norm = PhosphoNorm()
        self.layer_norm = nn.LayerNorm(n_sites)

        self.input_proj = nn.Sequential(
            nn.Linear(n_sites, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        if use_ks_graph:
            self.ks_graph = KSGraphAttention(
                hidden_dim=hidden_dim, n_heads=n_heads, dropout=dropout,
            )
        else:
            self.ks_graph = None

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Optional RNA alignment projection for contrastive loss
        self.rna_alignment_proj = (
            nn.Linear(embed_dim, 128) if align_to_rna else None
        )

    def forward(
        self,
        phospho: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode phosphopeptide intensities into a 64-dim embedding.

        Args:
            phospho:    (batch, n_sites) — raw phospho intensities.
            edge_index: (2, n_edges) — PhosphoSitePlus K->S edges (optional).
        Returns:
            (batch, 64)
        """
        x = self.phospho_norm(phospho)
        x = self.layer_norm(x)
        x = self.input_proj(x)

        if self.use_ks_graph and self.ks_graph is not None:
            x_seq = x.unsqueeze(1)
            x_seq = self.ks_graph(x_seq, edge_index=edge_index)
            x = x_seq.squeeze(1)

        x = self.output_proj(x)
        return x

    def get_rna_aligned_embedding(
        self,
        phospho: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Return 128-dim embedding aligned to RNA space for contrastive loss.
        Requires align_to_rna=True in __init__.

        Returns:
            (batch, 128) — projected phospho embedding for InfoNCE loss.
        """
        emb_64 = self.forward(phospho, edge_index=edge_index)
        if self.rna_alignment_proj is not None:
            return self.rna_alignment_proj(emb_64)
        return emb_64
