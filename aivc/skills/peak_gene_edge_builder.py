"""
PeakGeneEdgeBuilder — Converts peak-gene links into PyG edge_index.

Adds DIRECTED peak->gene edges to existing STRING PPI graph.
Peak->gene edges are DIRECTED (causal: accessibility enables transcription).
STRING PPI edges are UNDIRECTED (correlation-based).
Never mix directionality.

Edge weight = correlation * motif_score (from peak_gene_linker).
"""
import torch
import pandas as pd
import numpy as np
from typing import Optional


def build_peak_gene_edges(
    peak_gene_links: pd.DataFrame,
    gene_to_idx: dict,
    n_genes: int,
    motif_scores: dict = None,
) -> tuple:
    """
    Convert peak-gene links DataFrame into PyG-compatible edge tensors.

    Args:
        peak_gene_links: DataFrame with columns [peak_id, gene_name,
            correlation, pval, fdr, distance_to_tss].
        gene_to_idx: Dict mapping gene names to indices in the graph.
        n_genes: Total number of gene nodes.
        motif_scores: Optional dict mapping peak_id -> TF motif strength.
            If None, uses correlation as sole edge weight.

    Returns:
        Tuple of (atac_edge_index, atac_edge_weight):
            atac_edge_index: (2, n_atac_edges) long tensor.
                Row 0 = source (peak node index, offset by n_genes).
                Row 1 = target (gene node index).
            atac_edge_weight: (n_atac_edges,) float tensor.
    """
    if peak_gene_links is None or len(peak_gene_links) == 0:
        return (
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
        )

    # Assign unique indices to peaks (offset by n_genes to avoid collision)
    unique_peaks = sorted(peak_gene_links["peak_id"].unique())
    peak_to_idx = {p: i + n_genes for i, p in enumerate(unique_peaks)}

    src_indices = []
    dst_indices = []
    weights = []

    for _, row in peak_gene_links.iterrows():
        peak_id = row["peak_id"]
        gene_name = row["gene_name"]

        if gene_name not in gene_to_idx:
            continue
        if peak_id not in peak_to_idx:
            continue

        src_idx = peak_to_idx[peak_id]
        dst_idx = gene_to_idx[gene_name]

        # Edge weight = correlation * motif_score (if available)
        corr = float(row.get("correlation", 0.1))
        if motif_scores is not None and peak_id in motif_scores:
            weight = abs(corr) * motif_scores[peak_id]
        else:
            weight = abs(corr)

        src_indices.append(src_idx)
        dst_indices.append(dst_idx)
        weights.append(weight)

    if len(src_indices) == 0:
        return (
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
        )

    atac_edge_index = torch.tensor(
        [src_indices, dst_indices], dtype=torch.long
    )
    atac_edge_weight = torch.tensor(weights, dtype=torch.float32)

    return atac_edge_index, atac_edge_weight


def combine_graphs(
    ppi_edge_index: torch.Tensor,
    ppi_edge_weight: torch.Tensor,
    atac_edge_index: torch.Tensor,
    atac_edge_weight: torch.Tensor,
) -> dict:
    """
    Combine STRING PPI and ATAC peak->gene edges into a heterogeneous graph.

    Returns dict with separate edge types (NOT concatenated):
      - "gene_gene": undirected STRING PPI edges
      - "peak_gene": directed ATAC peak->gene edges

    Args:
        ppi_edge_index: (2, n_ppi_edges) — STRING PPI edges (undirected).
        ppi_edge_weight: (n_ppi_edges,) — PPI confidence scores.
        atac_edge_index: (2, n_atac_edges) — peak->gene edges (directed).
        atac_edge_weight: (n_atac_edges,) — correlation * motif weights.

    Returns:
        Dict with edge type keys mapping to (edge_index, edge_weight) tuples.
    """
    return {
        "gene_gene": {
            "edge_index": ppi_edge_index,
            "edge_weight": ppi_edge_weight,
            "directed": False,
            "n_edges": ppi_edge_index.shape[1],
        },
        "peak_gene": {
            "edge_index": atac_edge_index,
            "edge_weight": atac_edge_weight,
            "directed": True,
            "n_edges": atac_edge_index.shape[1],
        },
    }
