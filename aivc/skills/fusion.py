"""
Cross-modal fusion for AIVC — 4 modalities, enforced causal ordering.

Architecture: RNA (128) + Protein (128) + Phospho (64) + ATAC (64) = 384-dim output

Temporal encoding:
  ATAC    = t=0 (chromatin state, baseline)
  Phospho = t=1 (kinase signalling, seconds)
  RNA     = t=2 (transcription, minutes)
  Protein = t=3 (translation, hours)

Causal ordering enforcement (Phase 3):
  A lower-triangular attention mask is applied before softmax.
  This forces: modality at time t attends ONLY to t' <= t.
  Protein cannot attend to RNA (future cannot influence past).
  Upper-triangle attention weight = 0.0 (confirmed by test).
  use_causal_mask=True is the default. Set False only for ablation.

This is a TEMPORAL ORDERING CONSTRAINT, not a Pearl do-calculus structural
causal model. do(X) interventions are not implemented.

Loss: Combined loss includes causal_ordering_loss() term (lambda_causal=0.1)
which penalises residual upper-triangular attention weight.
See losses.py: combined_loss_multimodal(attn_weights=..., lambda_causal=...).

Status: Coded and tested. Not yet trained on real multi-modal data.
Protein/Phospho encoders are not yet implemented (no class files).
ATAC encoder coded but awaiting 10x Multiome data. When only RNA is
available: zero-filled modality embeddings are passed; the causal mask
applies but has no effect on RNA-only training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class TemporalCrossModalFusion(nn.Module):
    """
    Cross-modal fusion with temporal causal ordering.

    Combines embeddings from up to 4 modalities via Q*KV attention,
    where temporal position encodings enforce the biological causal order:
      ATAC (t=0) -> Phospho (t=1) -> RNA (t=2) -> Protein (t=3)

    Args:
        rna_dim: RNA encoder output dimension. Default 128.
        protein_dim: Protein encoder output dimension. Default 128.
        phospho_dim: Phospho encoder output dimension. Default 64.
        atac_dim: ATAC encoder output dimension. Default 64.
        n_heads: Number of attention heads. Default 4.
        dropout: Attention dropout. Default 0.1.
        pert_dim: Perturbation embedding dimension. Default 32.
    """

    # Temporal order: ATAC is always t=0
    TEMPORAL_ORDER = {
        "atac": 0,
        "phospho": 1,
        "rna": 2,
        "protein": 3,
    }

    def __init__(
        self,
        rna_dim: int = 128,
        protein_dim: int = 128,
        phospho_dim: int = 64,
        atac_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
        pert_dim: int = 32,
    ):
        super().__init__()

        self.rna_dim = rna_dim
        self.protein_dim = protein_dim
        self.phospho_dim = phospho_dim
        self.atac_dim = atac_dim

        # Project each modality to shared attention dim
        self.attn_dim = 64  # per-modality attention dimension
        self.proj_rna = nn.Linear(rna_dim, self.attn_dim)
        self.proj_protein = nn.Linear(protein_dim, self.attn_dim)
        self.proj_phospho = nn.Linear(phospho_dim, self.attn_dim)
        self.proj_atac = nn.Linear(atac_dim, self.attn_dim)

        # Temporal position encoding (4 positions)
        self.temporal_encoding = nn.Embedding(4, self.attn_dim)

        # Multi-head self-attention over modalities
        self.n_heads = n_heads
        self.head_dim = self.attn_dim // n_heads
        self.qkv = nn.Linear(self.attn_dim, 3 * self.attn_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(self.attn_dim, self.attn_dim)

        # Output: concatenate all modality outputs + perturbation
        # 4 modalities * attn_dim + pert_dim
        self.total_dim = 4 * self.attn_dim + pert_dim  # 4*64 + 32 = 288
        self.output_dim = rna_dim + protein_dim + phospho_dim + atac_dim  # 384

        self.output_proj = nn.Sequential(
            nn.Linear(self.total_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.GELU(),
        )

        # Perturbation embedding
        self.pert_embedding = nn.Embedding(10, pert_dim)  # up to 10 perturbations

        # Causal masking: set False to ablate for experiments
        self.use_causal_mask = True

    def forward(
        self,
        rna_emb: torch.Tensor,
        protein_emb: torch.Tensor = None,
        phospho_emb: torch.Tensor = None,
        atac_emb: torch.Tensor = None,
        pert_id: torch.Tensor = None,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse modality embeddings via temporal cross-attention.

        Args:
            rna_emb: (batch, rna_dim) — required.
            protein_emb: (batch, protein_dim) — optional, zero-filled if absent.
            phospho_emb: (batch, phospho_dim) — optional, zero-filled if absent.
            atac_emb: (batch, atac_dim) — optional, zero-filled if absent.
            pert_id: (batch,) or scalar — perturbation index.
            modality_mask: (batch, 4) bool/float tensor in TEMPORAL order
                [ATAC, Phospho, RNA, Protein]. 1 = modality present for that
                cell and contributes to attention; 0 = modality absent (row
                and column masked to -inf, effectively excluded from
                softmax). Default None = all 4 modalities present for all
                cells (backward-compatible).

        Returns:
            (batch, output_dim) — fused 384-dim representation.
        """
        batch_size = rna_emb.shape[0]
        device = rna_emb.device

        # Project each modality to attention dim
        rna_proj = self.proj_rna(rna_emb)  # (batch, attn_dim)

        if protein_emb is not None:
            protein_proj = self.proj_protein(protein_emb)
        else:
            protein_proj = torch.zeros(batch_size, self.attn_dim, device=device)

        if phospho_emb is not None:
            phospho_proj = self.proj_phospho(phospho_emb)
        else:
            phospho_proj = torch.zeros(batch_size, self.attn_dim, device=device)

        if atac_emb is not None:
            atac_proj = self.proj_atac(atac_emb)
        else:
            atac_proj = torch.zeros(batch_size, self.attn_dim, device=device)

        # Stack modalities: (batch, 4, attn_dim) in temporal order
        modalities = torch.stack([
            atac_proj,      # t=0
            phospho_proj,   # t=1
            rna_proj,       # t=2
            protein_proj,   # t=3
        ], dim=1)  # (batch, 4, attn_dim)

        # Add temporal position encoding
        temporal_ids = torch.arange(4, device=device)
        temporal_enc = self.temporal_encoding(temporal_ids)  # (4, attn_dim)
        modalities = modalities + temporal_enc.unsqueeze(0)  # broadcast

        # Multi-head self-attention
        n_mod = 4
        qkv = self.qkv(modalities)  # (batch, 4, 3*attn_dim)
        qkv = qkv.reshape(batch_size, n_mod, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, n_heads, 4, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (batch, n_heads, 4, 4)

        # ── Causal mask ─────────────────────────────────────────
        # Temporal order: ATAC(0) → Phospho(1) → RNA(2) → Protein(3)
        # Lower-triangular: modality at time t attends to t' <= t only.
        #   ATAC    attends to: [ATAC]
        #   Phospho attends to: [ATAC, Phospho]
        #   RNA     attends to: [ATAC, Phospho, RNA]
        #   Protein attends to: [ATAC, Phospho, RNA, Protein]
        # Base attention validity mask (n_mod, n_mod). Start with all-True;
        # then AND in causal mask (if enabled) and modality_mask (if given).
        if self.use_causal_mask:
            base_mask = torch.tril(
                torch.ones(n_mod, n_mod, device=device, dtype=torch.bool)
            ).unsqueeze(0).unsqueeze(0)  # (1, 1, 4, 4)
        else:
            base_mask = torch.ones(
                1, 1, n_mod, n_mod, device=device, dtype=torch.bool
            )

        # Modality mask: absent modalities cannot be attended to (column)
        # AND cannot attend out (row). Outer-product of presence vector.
        if modality_mask is not None:
            m = modality_mask.bool()
            if m.shape != (batch_size, n_mod):
                raise ValueError(
                    f"modality_mask shape {tuple(m.shape)} != "
                    f"({batch_size}, {n_mod}). Order is "
                    f"[ATAC, Phospho, RNA, Protein]."
                )
            # (batch, n_mod, n_mod): (i, j) present iff both i and j present
            m2d = m.unsqueeze(-1) & m.unsqueeze(-2)
            m2d = m2d.unsqueeze(1)  # (batch, 1, n_mod, n_mod), broadcasts over heads
            full_mask = base_mask & m2d
        else:
            full_mask = base_mask

        attn_scores = attn_scores.masked_fill(~full_mask, float('-inf'))
        # ─────────────────────────────────────────────────────────

        attn_weights = F.softmax(attn_scores, dim=-1)
        # Handle NaN from all-inf rows (ATAC row has only self-attention)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (batch, n_heads, 4, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, n_mod, self.attn_dim)
        attn_output = self.out_proj(attn_output)  # (batch, 4, attn_dim)

        # Flatten all modality outputs
        fused = attn_output.reshape(batch_size, n_mod * self.attn_dim)  # (batch, 256)

        # Add perturbation embedding
        if pert_id is not None:
            if pert_id.dim() == 0:
                pert_id = pert_id.unsqueeze(0).expand(batch_size)
            pert_vec = self.pert_embedding(pert_id)  # (batch, pert_dim)
        else:
            pert_vec = torch.zeros(batch_size, 32, device=device)

        fused = torch.cat([fused, pert_vec], dim=-1)  # (batch, 288)

        # Project to output dimension (384)
        output = self.output_proj(fused)  # (batch, 384)

        return output

    def get_attention_weights(
        self,
        rna_emb: torch.Tensor,
        protein_emb: torch.Tensor = None,
        phospho_emb: torch.Tensor = None,
        atac_emb: torch.Tensor = None,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract cross-modal attention weights for interpretability.

        Returns:
            (batch, n_heads, 4, 4) attention weight matrix.
            Rows: query modality, Cols: key modality.
            Order: [ATAC, Phospho, RNA, Protein]
        """
        batch_size = rna_emb.shape[0]
        device = rna_emb.device

        rna_proj = self.proj_rna(rna_emb)
        protein_proj = self.proj_protein(protein_emb) if protein_emb is not None else torch.zeros(batch_size, self.attn_dim, device=device)
        phospho_proj = self.proj_phospho(phospho_emb) if phospho_emb is not None else torch.zeros(batch_size, self.attn_dim, device=device)
        atac_proj = self.proj_atac(atac_emb) if atac_emb is not None else torch.zeros(batch_size, self.attn_dim, device=device)

        modalities = torch.stack([atac_proj, phospho_proj, rna_proj, protein_proj], dim=1)
        temporal_ids = torch.arange(4, device=device)
        modalities = modalities + self.temporal_encoding(temporal_ids).unsqueeze(0)

        qkv = self.qkv(modalities).reshape(batch_size, 4, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]

        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if self.use_causal_mask:
            base_mask = torch.tril(
                torch.ones(4, 4, device=device, dtype=torch.bool)
            ).unsqueeze(0).unsqueeze(0)
        else:
            base_mask = torch.ones(1, 1, 4, 4, device=device, dtype=torch.bool)
        if modality_mask is not None:
            m = modality_mask.bool()
            if m.shape != (batch_size, 4):
                raise ValueError(
                    f"modality_mask shape {tuple(m.shape)} != ({batch_size}, 4)."
                )
            m2d = (m.unsqueeze(-1) & m.unsqueeze(-2)).unsqueeze(1)
            full_mask = base_mask & m2d
        else:
            full_mask = base_mask
        attn_scores = attn_scores.masked_fill(~full_mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        return attn_weights  # (batch, n_heads, 4, 4)
