"""Stage 3 — pathway-aware per-modality decoders.

Per docs/specs/stage3_part2_architecture_proposal_2026_05_06.md §3.5.

Three decoders are provided:
  - PathwayAwareRNADecoder: latent → (gene_logits, pathway_scores)
      Per-gene linear head + static gene-to-pathway pool for pathway-level
      summary scores. Used in Stage 3b training (QurieSeq RNA readouts).
  - PathwayAwareProteinDecoder: latent → (protein_expression, pathway_scores_or_None)
      Per-antibody linear head. Pathway pool optional — Mimitou's 38
      antibodies don't usefully aggregate to pathway scores; defaults to
      no pathway alignment unless explicitly provided.
  - PathwayAwarePhosphoDecoder: latent → (phospho_logits, pathway_scores)
      Per-phospho-site linear head with explicit phospho-to-pathway map.
      Used in Stage 3c (QurieSeq Phase 2 phospho integration). Stub here;
      full alignment lands when phospho labels arrive.

Pathway pool matrix: built by scripts/build_pathway_pool_matrix.py from
Report 3's gene_to_pathway_map.csv. Shape (n_pathways, n_genes), row-
normalized (sum=1.0 per pathway). Loaded as a non-trainable buffer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


def load_pathway_pool(npz_path: Path) -> torch.Tensor:
    """Load the sparse pool matrix (n_pathways, n_genes) and convert to
    a dense torch tensor. Sparse multiplication is supported but adds
    autograd complexity; dense is fine at the (58, ~5000) target shape.
    """
    pool_sp = sp.load_npz(str(npz_path))
    pool_dense = torch.from_numpy(pool_sp.toarray()).float()
    return pool_dense  # (n_pathways, n_genes)


# --- Schema_version=1 checkpoint envelope shared across decoders ---

DECODER_CKPT_SCHEMA_VERSION = 1


class _DecoderBase(nn.Module):
    """Shared init + state_dict skeleton for the 3 decoders."""

    def __init__(self, d: int, out_dim: int, hidden_mult: int = 0):
        super().__init__()
        if d <= 0 or out_dim <= 0:
            raise ValueError(f"d and out_dim must be positive; got d={d}, out_dim={out_dim}")
        self.d = d
        self.out_dim = out_dim
        if hidden_mult > 0:
            hidden = hidden_mult * d
            self.head = nn.Sequential(
                nn.Linear(d, hidden), nn.GELU(),
                nn.Linear(hidden, out_dim),
            )
        else:
            self.head = nn.Linear(d, out_dim)

    def _project(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[-1] != self.d:
            raise ValueError(f"decoder input last dim {z.shape[-1]} != d={self.d}")
        return self.head(z)


class PathwayAwareRNADecoder(_DecoderBase):
    """RNA expression decoder with pathway-level pooling.

    forward(z): returns (gene_logits, pathway_scores) where
        gene_logits     ∈ (B, n_genes)        — per-gene linear logits
        pathway_scores  ∈ (B, n_pathways)     — mean-pooled over members

    Args:
        d:         input latent dimension
        n_genes:   number of genes in the per-gene head
        gene_to_pathway_pool: (n_pathways, n_genes) dense tensor or None.
                  If None, pathway_scores is None on forward (decoder is
                  effectively gene-only).
        hidden_mult: if > 0, head is 2-layer MLP (Linear → GELU → Linear).
                    Default 0 = single linear head.
    """

    def __init__(
        self,
        d: int = 256,
        n_genes: int = 36601,
        gene_to_pathway_pool: Optional[torch.Tensor] = None,
        hidden_mult: int = 0,
    ):
        super().__init__(d=d, out_dim=n_genes, hidden_mult=hidden_mult)
        self.n_genes = n_genes
        if gene_to_pathway_pool is not None:
            if gene_to_pathway_pool.shape[1] != n_genes:
                raise ValueError(
                    f"gene_to_pathway_pool shape {gene_to_pathway_pool.shape} "
                    f"second dim != n_genes={n_genes}"
                )
            # Non-trainable buffer; survives state_dict() save/load
            self.register_buffer("g2p_W", gene_to_pathway_pool)
            self.has_pathway_pool = True
        else:
            self.has_pathway_pool = False

    def forward(self, z: torch.Tensor):
        gene_logits = self._project(z)  # (B, n_genes)
        if self.has_pathway_pool:
            # pathway_scores = gene_logits @ g2p_W.T  →  (B, n_pathways)
            pathway_scores = gene_logits @ self.g2p_W.T
        else:
            pathway_scores = None
        return gene_logits, pathway_scores


class PathwayAwareProteinDecoder(_DecoderBase):
    """Protein (ADT) expression decoder with optional pathway alignment.

    Mimitou's 38 antibodies are sparse pathway members and aggregation
    doesn't yield interpretable pathway scores. By default, this
    decoder is protein-level only (pathway alignment off).

    For QurieSeq (Stage 3b/3c) where the TotalSeq-A panel has 210
    antibodies and many cover signaling-relevant proteins, a protein-to-
    pathway pool can be supplied — then pathway_scores becomes non-None.

    forward(z): returns (protein_expr, pathway_scores_or_None)
        protein_expr    ∈ (B, n_proteins)
        pathway_scores  ∈ (B, n_pathways) or None
    """

    def __init__(
        self,
        d: int = 256,
        n_proteins: int = 38,
        protein_to_pathway_pool: Optional[torch.Tensor] = None,
        hidden_mult: int = 0,
    ):
        super().__init__(d=d, out_dim=n_proteins, hidden_mult=hidden_mult)
        self.n_proteins = n_proteins
        if protein_to_pathway_pool is not None:
            if protein_to_pathway_pool.shape[1] != n_proteins:
                raise ValueError(
                    f"protein_to_pathway_pool shape "
                    f"{protein_to_pathway_pool.shape} second dim != "
                    f"n_proteins={n_proteins}"
                )
            self.register_buffer("p2p_W", protein_to_pathway_pool)
            self.has_pathway_pool = True
        else:
            self.has_pathway_pool = False

    def forward(self, z: torch.Tensor):
        protein_expr = self._project(z)
        if self.has_pathway_pool:
            pathway_scores = protein_expr @ self.p2p_W.T
        else:
            pathway_scores = None
        return protein_expr, pathway_scores


class PathwayAwarePhosphoDecoder(_DecoderBase):
    """Phospho readout decoder with pathway-readout alignment.

    For Stage 3c (QurieSeq Phase 2). Each phospho site is associated with
    one pathway (pJAK1 → JAK_STAT_signaling, pERK → MAPK_signaling, etc.)
    via a static mapping. The decoder produces both per-site logits AND
    pathway-level scores (one per pathway, computed as mean over sites).

    forward(z): returns (phospho_logits, pathway_scores)
    """

    def __init__(
        self,
        d: int = 256,
        n_sites: int = 0,  # placeholder until Phase 2 lands
        phospho_to_pathway_pool: Optional[torch.Tensor] = None,
        hidden_mult: int = 0,
    ):
        super().__init__(d=d, out_dim=max(n_sites, 1), hidden_mult=hidden_mult)
        self.n_sites = n_sites
        if phospho_to_pathway_pool is not None and n_sites > 0:
            if phospho_to_pathway_pool.shape[1] != n_sites:
                raise ValueError(
                    f"phospho_to_pathway_pool shape "
                    f"{phospho_to_pathway_pool.shape} second dim != "
                    f"n_sites={n_sites}"
                )
            self.register_buffer("ph2p_W", phospho_to_pathway_pool)
            self.has_pathway_pool = True
        else:
            self.has_pathway_pool = False

    def forward(self, z: torch.Tensor):
        if self.n_sites == 0:
            # Stub — Phase 2 not yet wired
            return None, None
        phospho_logits = self._project(z)
        if self.has_pathway_pool:
            pathway_scores = phospho_logits @ self.ph2p_W.T
        else:
            pathway_scores = None
        return phospho_logits, pathway_scores


# --- Save/load helpers (schema_version=1 envelope) ---

def save_decoder(
    decoder: _DecoderBase,
    path,
    decoder_kind: str,
    extra_meta: Optional[dict] = None,
) -> None:
    """Save a decoder with the schema_version=1 envelope.

    Args:
        decoder_kind: one of {'rna', 'protein', 'phospho'}.
    """
    if decoder_kind not in {"rna", "protein", "phospho"}:
        raise ValueError(f"decoder_kind must be 'rna'/'protein'/'phospho'; got {decoder_kind!r}")
    payload = {
        "schema_version": DECODER_CKPT_SCHEMA_VERSION,
        "kind": f"stage3_decoder_{decoder_kind}",
        "config": {
            "d": decoder.d,
            "out_dim": decoder.out_dim,
            "has_pathway_pool": getattr(decoder, "has_pathway_pool", False),
        },
        "state_dict": decoder.state_dict(),
        "meta": extra_meta or {},
    }
    torch.save(payload, path)


def load_decoder(path, decoder_cls, map_location: str = "cpu", **decoder_kwargs):
    """Load a decoder from a schema_version=1 checkpoint.

    Caller must pass the decoder class (PathwayAwareRNADecoder etc.) +
    any non-state-dict constructor args (e.g., gene_to_pathway_pool).
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "schema_version" not in payload:
        raise ValueError(f"{path}: not a versioned checkpoint")
    if payload["schema_version"] != DECODER_CKPT_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version {payload['schema_version']} != {DECODER_CKPT_SCHEMA_VERSION}"
        )
    cfg = payload["config"]
    # Reconstruct with constructor kwargs (typically d, n_genes/n_proteins, pool)
    decoder = decoder_cls(d=cfg["d"], **decoder_kwargs)
    decoder.load_state_dict(payload["state_dict"])
    return decoder
