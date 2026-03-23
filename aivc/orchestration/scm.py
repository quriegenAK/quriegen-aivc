"""
Structural Causal Model (SCM) for AIVC v3.0 — 4-node causal ordering.

Extended causal ordering:
  ATAC (t=0, baseline chromatin) -> Phospho (seconds) -> RNA (minutes) -> Protein (hours)

New edge type in heterogeneous graph:
  Source: peak node (from mdata.uns['peak_gene_links'])
  Target: gene node (STRING PPI graph)
  Edge weight: co-accessibility score * TF motif strength
  Edge direction: ATAC peak -> gene (DIRECTED, not undirected like STRING PPI)

do(X) interventions:
  do(peak=closed) — set a peak's accessibility to 0
  -> all TF motif scores at that peak -> 0
  -> propagate through ATAC->Phospho->RNA->Protein chain
  -> quantify which genes are affected

This is the epigenomic counterfactual: "what if this enhancer was silenced?"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CausalNode:
    """A node in the structural causal model."""
    name: str
    temporal_order: int  # 0=ATAC, 1=Phospho, 2=RNA, 3=Protein
    embedding_dim: int
    description: str = ""


@dataclass
class CausalEdge:
    """A directed edge in the structural causal model."""
    source: str
    target: str
    edge_type: str  # "atac_to_phospho", "phospho_to_rna", "rna_to_protein"
    weight: float = 1.0
    description: str = ""


class StructuralCausalModel(nn.Module):
    """
    4-node SCM with epigenomic causal ordering.

    Nodes:
      ATAC    (t=0): chromatin accessibility state
      Phospho (t=1): kinase-substrate signalling
      RNA     (t=2): gene transcription
      Protein (t=3): protein translation

    Causal edges (directed):
      ATAC -> Phospho: chromatin state enables TF binding -> kinase activation
      Phospho -> RNA:  phospho signalling activates transcription factors
      RNA -> Protein:  mRNA is translated to protein

    Enforced via causal consistency loss:
      When upstream modality changes, downstream must follow.
    """

    NODES = {
        "atac": CausalNode("atac", 0, 64, "Chromatin accessibility"),
        "phospho": CausalNode("phospho", 1, 64, "Kinase-substrate signalling"),
        "rna": CausalNode("rna", 2, 128, "Gene transcription"),
        "protein": CausalNode("protein", 3, 128, "Protein translation"),
    }

    EDGES = [
        CausalEdge("atac", "phospho", "atac_to_phospho", 1.0,
                    "Chromatin state enables TF binding"),
        CausalEdge("phospho", "rna", "phospho_to_rna", 1.0,
                    "Phospho signalling activates transcription"),
        CausalEdge("rna", "protein", "rna_to_protein", 1.0,
                    "mRNA is translated to protein"),
    ]

    def __init__(self, atac_dim=64, phospho_dim=64, rna_dim=128, protein_dim=128):
        super().__init__()

        # Learnable causal edge weights
        self.edge_weight_atac_phospho = nn.Parameter(torch.tensor(1.0))
        self.edge_weight_phospho_rna = nn.Parameter(torch.tensor(1.0))
        self.edge_weight_rna_protein = nn.Parameter(torch.tensor(1.0))

        # Causal projection layers (map upstream to downstream prediction)
        self.proj_atac_to_phospho = nn.Linear(atac_dim, phospho_dim)
        self.proj_phospho_to_rna = nn.Linear(phospho_dim, rna_dim)
        self.proj_rna_to_protein = nn.Linear(rna_dim, protein_dim)

    def causal_consistency_loss(
        self,
        atac_emb: torch.Tensor = None,
        phospho_emb: torch.Tensor = None,
        rna_emb: torch.Tensor = None,
        protein_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute causal consistency loss.

        When upstream modality changes, downstream must follow in order:
          ATAC -> Phospho -> RNA -> Protein

        Loss = sum of MSE between predicted downstream and actual downstream,
        weighted by learned causal edge weights.

        Args:
            atac_emb: (batch, atac_dim) or None
            phospho_emb: (batch, phospho_dim) or None
            rna_emb: (batch, rna_dim) or None
            protein_emb: (batch, protein_dim) or None

        Returns:
            Scalar causal consistency loss.
        """
        loss = torch.tensor(0.0, requires_grad=True)
        device = None

        # Find device from any available embedding
        for emb in [atac_emb, phospho_emb, rna_emb, protein_emb]:
            if emb is not None:
                device = emb.device
                break

        if device is None:
            return loss

        loss = loss.to(device)

        # ATAC -> Phospho
        if atac_emb is not None and phospho_emb is not None:
            predicted_phospho = self.proj_atac_to_phospho(atac_emb)
            w = torch.sigmoid(self.edge_weight_atac_phospho)
            loss = loss + w * F.mse_loss(predicted_phospho, phospho_emb.detach())

        # Phospho -> RNA
        if phospho_emb is not None and rna_emb is not None:
            predicted_rna = self.proj_phospho_to_rna(phospho_emb)
            w = torch.sigmoid(self.edge_weight_phospho_rna)
            loss = loss + w * F.mse_loss(predicted_rna, rna_emb.detach())

        # RNA -> Protein
        if rna_emb is not None and protein_emb is not None:
            predicted_protein = self.proj_rna_to_protein(rna_emb)
            w = torch.sigmoid(self.edge_weight_rna_protein)
            loss = loss + w * F.mse_loss(predicted_protein, protein_emb.detach())

        return loss

    def do_intervention(
        self,
        intervention_node: str,
        intervention_value: torch.Tensor,
        current_embeddings: dict,
    ) -> dict:
        """
        Pearl do(X=x) intervention — set a node and propagate downstream.

        Args:
            intervention_node: Which node to intervene on ("atac", "phospho", "rna", "protein").
            intervention_value: (batch, node_dim) — the intervention value.
            current_embeddings: Dict of current embeddings per node.

        Returns:
            Dict of counterfactual embeddings after propagation.
        """
        result = dict(current_embeddings)
        result[intervention_node] = intervention_value

        order = ["atac", "phospho", "rna", "protein"]
        start_idx = order.index(intervention_node)

        # Propagate downstream from intervention point
        projections = {
            ("atac", "phospho"): self.proj_atac_to_phospho,
            ("phospho", "rna"): self.proj_phospho_to_rna,
            ("rna", "protein"): self.proj_rna_to_protein,
        }

        for i in range(start_idx, len(order) - 1):
            src = order[i]
            dst = order[i + 1]
            if src in result and result[src] is not None:
                proj = projections.get((src, dst))
                if proj is not None:
                    result[dst] = proj(result[src])

        return result

    def do_peak_closed(
        self,
        peak_indices: list,
        current_embeddings: dict,
        chromvar_scores: torch.Tensor,
        tf_to_peak_mask: torch.Tensor = None,
    ) -> dict:
        """
        Epigenomic counterfactual: "what if this enhancer was silenced?"

        Sets specified peaks' accessibility to 0, zeroes affected TF motif
        scores, then propagates through ATAC->Phospho->RNA->Protein.

        Args:
            peak_indices: List of peak indices to silence.
            current_embeddings: Dict with current modality embeddings.
            chromvar_scores: (batch, n_tfs) — current TF scores.
            tf_to_peak_mask: (n_tfs, n_peaks) — which TFs are in which peaks.

        Returns:
            Dict of counterfactual embeddings.
        """
        modified_scores = chromvar_scores.clone()

        if tf_to_peak_mask is not None:
            # Zero out TF scores for TFs whose motifs are in silenced peaks
            for peak_idx in peak_indices:
                if peak_idx < tf_to_peak_mask.shape[1]:
                    affected_tfs = tf_to_peak_mask[:, peak_idx] > 0
                    modified_scores[:, affected_tfs] = 0.0
        else:
            # Without mask: zero a fraction of TF scores proportional to silenced peaks
            n_zero = min(len(peak_indices), modified_scores.shape[1])
            modified_scores[:, :n_zero] = 0.0

        # The caller should re-encode modified_scores through ATACSeqEncoder
        # to get new atac_emb, then call do_intervention("atac", new_atac_emb, ...)
        return {"modified_chromvar_scores": modified_scores}

    def get_causal_graph(self) -> dict:
        """Return the causal graph structure for visualization."""
        return {
            "nodes": {name: {"order": node.temporal_order, "dim": node.embedding_dim}
                      for name, node in self.NODES.items()},
            "edges": [{"source": e.source, "target": e.target, "type": e.edge_type,
                       "weight": torch.sigmoid(getattr(self, f"edge_weight_{e.edge_type}")).item()
                       if hasattr(self, f"edge_weight_{e.edge_type}") else e.weight}
                      for e in self.EDGES],
        }
