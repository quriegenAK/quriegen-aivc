"""
eval/benchmarks/norman_eval.py — Norman 2019 CRISPR zero-shot evaluation.

Evaluates a Kang-trained checkpoint on Norman 2019 K562 CRISPR perturbations.

# V1.1 KNOWN CEILING — scalar pert_id
# PerturbationPredictor is instantiated with num_perturbations=2 (ctrl/stim).
# PerturbationEmbedding.forward broadcasts a single learned vector across
# all genes regardless of which Norman perturbation is being evaluated
# (only pert_id[0:1] is consumed — perturbation_model.py:140).
# Every Norman perturbation therefore receives the same predicted delta.
#
# What this eval measures in v1.1:
#   - delta_nonzero_pct: whether the model predicts ANY response vs ctrl
#   - ctrl_memorisation_score: whether the model has collapsed to copying ctrl
#   - pearson_r_ctrl_sub: correlation of the single predicted delta
#     with the MEAN ground-truth delta across all perturbations
#
# What this eval does NOT measure:
#   - Per-perturbation accuracy (requires gene-conditioned pert_id)
#   - Top-k gene overlap per perturbation (uninformative in v1.1)
#
# v1.2 fix: expand num_perturbations to >=237, add per-row pert_id
# indexing in PerturbationEmbedding.forward, retrain from Kang checkpoint.

Data layout (norman2019.h5ad):
  obs["nperts"]:       0 = ctrl, 1 = single, 2 = double perturbation
  obs["perturbation"]: perturbation name (237 categories)
  obs["gemgroup"]:     replicate group (integer)
  var: 33694 genes — intersect with gene_names.txt (3010)
  Cell line: K562 (all cells)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from pydantic import BaseModel, Field, model_validator

from eval.metrics import (
    ctrl_memorisation_score,
    delta_nonzero_pct,
    pearson_r_ctrl_subtracted,
    top_k_gene_overlap,
)

logger = logging.getLogger("aivc.eval.norman")

N_GENES = 3010
MIN_CELLS_PER_PSEUDOBULK = 30
DELTA_NONZERO_THRESHOLD = 1.0  # v1 — raise to 2-5% in v2


class NormanEvalReport(BaseModel):
    """Evaluation report for Norman 2019 zero-shot benchmark."""

    run_id: str
    dataset: str = "norman2019"
    n_genes: int = N_GENES
    n_perturbations: int = 0
    n_pseudobulk_pairs: int = 0
    n_dropped_groups: int = 0

    pearson_r_ctrl_sub: float = 0.0
    delta_nonzero_pct: float = 0.0
    ctrl_memorisation_score: float = 0.0
    top_k_overlap: float = 0.0

    passed: bool = False
    failure_reason: Optional[str] = None

    @model_validator(mode="after")
    def _n_genes_is_3010(self) -> NormanEvalReport:
        if self.n_genes != N_GENES:
            raise ValueError(
                f"Norman eval requires exactly {N_GENES} genes, got {self.n_genes}"
            )
        return self

    @model_validator(mode="after")
    def _collapse_invariant(self) -> NormanEvalReport:
        """If delta_nonzero_pct == 0, ctrl_memorisation_score must be >= 0.99."""
        if self.delta_nonzero_pct == 0.0 and self.ctrl_memorisation_score < 0.99:
            raise ValueError(
                f"Collapse invariant violated: delta_nonzero_pct=0.0 but "
                f"ctrl_memorisation_score={self.ctrl_memorisation_score:.4f} < 0.99. "
                f"If delta is zero, the model must be memorising ctrl."
            )
        return self


def _load_gene_order(path: str = "data/gene_names.txt") -> list[str]:
    """Load the canonical 3010-gene order."""
    with open(path) as f:
        genes = [line.strip() for line in f if line.strip()]
    assert len(genes) == N_GENES, f"Expected {N_GENES} genes, got {len(genes)}"
    return genes


def _load_norman_data(
    adata_path: str,
    gene_order: list[str],
    min_cells: int = MIN_CELLS_PER_PSEUDOBULK,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Load and pseudo-bulk Norman 2019 data.

    Returns:
        (X_ctrl_pb, X_pert_pb, n_perturbations, n_pairs, n_dropped)
    """
    import anndata as ad

    logger.info(f"Loading Norman data: {adata_path}")
    adata = ad.read_h5ad(adata_path)

    # Gene alignment: Norman has ~33k genes, intersect with model's 3010
    gene_set = set(gene_order)
    adata_genes = list(adata.var_names)
    intersection = gene_set & set(adata_genes)
    assert len(intersection) == N_GENES, (
        f"Gene alignment failed: {len(intersection)}/{N_GENES} genes found in Norman. "
        f"Missing from Norman: {sorted(gene_set - set(adata_genes))[:10]}..."
    )
    gene_idx = [adata_genes.index(g) for g in gene_order]

    # Control cells: nperts == 0
    ctrl_mask = adata.obs["nperts"].values == 0
    X_ctrl_all = np.asarray(adata.X[ctrl_mask][:, gene_idx].todense()
                            if hasattr(adata.X[ctrl_mask], "todense")
                            else adata.X[ctrl_mask][:, gene_idx])

    # Pseudo-bulk ctrl per gemgroup
    ctrl_obs = adata.obs[ctrl_mask]
    gemgroups = sorted(ctrl_obs["gemgroup"].unique())

    # Perturbation cells
    pert_mask = adata.obs["nperts"].values > 0
    pert_obs = adata.obs[pert_mask]
    perturbations = sorted(pert_obs["perturbation"].unique())

    X_ctrl_list, X_pert_list = [], []
    n_dropped = 0

    for pert in perturbations:
        for gg in gemgroups:
            # Ctrl cells for this gemgroup
            ctrl_gg_mask = ctrl_mask & (adata.obs["gemgroup"].values == gg)
            n_ctrl = ctrl_gg_mask.sum()

            # Pert cells for this (perturbation, gemgroup)
            pert_gg_mask = (
                pert_mask
                & (adata.obs["perturbation"].values == pert)
                & (adata.obs["gemgroup"].values == gg)
            )
            n_pert = pert_gg_mask.sum()

            if n_ctrl < min_cells or n_pert < min_cells:
                n_dropped += 1
                continue

            X_c = np.asarray(
                adata.X[ctrl_gg_mask][:, gene_idx].todense()
                if hasattr(adata.X[ctrl_gg_mask], "todense")
                else adata.X[ctrl_gg_mask][:, gene_idx]
            ).mean(axis=0).flatten()

            X_p = np.asarray(
                adata.X[pert_gg_mask][:, gene_idx].todense()
                if hasattr(adata.X[pert_gg_mask], "todense")
                else adata.X[pert_gg_mask][:, gene_idx]
            ).mean(axis=0).flatten()

            X_ctrl_list.append(X_c)
            X_pert_list.append(X_p)

    logger.info(
        f"Norman pseudo-bulk: {len(X_ctrl_list)} pairs from "
        f"{len(perturbations)} perturbations, {n_dropped} groups dropped "
        f"(< {min_cells} cells)"
    )

    X_ctrl_pb = np.array(X_ctrl_list) if X_ctrl_list else np.zeros((0, N_GENES))
    X_pert_pb = np.array(X_pert_list) if X_pert_list else np.zeros((0, N_GENES))

    return X_ctrl_pb, X_pert_pb, len(perturbations), len(X_ctrl_list), n_dropped


def _load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load PerturbationPredictor from checkpoint."""
    from perturbation_model import PerturbationPredictor, CellTypeEmbedding

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model = PerturbationPredictor(n_genes=N_GENES)
    model.cell_type_embedding = CellTypeEmbedding(
        num_cell_types=20, embedding_dim=model.feature_dim
    )
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def run_norman_eval(
    checkpoint_path: str,
    adata_path: str = "data/norman2019.h5ad",
    *,
    run_id: str,
    device: str | None = None,
    min_cells_per_pseudobulk: int = MIN_CELLS_PER_PSEUDOBULK,
) -> NormanEvalReport:
    """Run Norman 2019 zero-shot evaluation.

    Args:
        checkpoint_path: Path to model checkpoint.
        adata_path: Path to Norman h5ad file.
        run_id: Unique run identifier.
        device: "cuda", "cpu", or None for auto-detect.
        min_cells_per_pseudobulk: Minimum cells per pseudo-bulk group.

    Returns:
        NormanEvalReport with all metrics and pass/fail status.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Norman eval: device={dev}, checkpoint={checkpoint_path}")

    gene_order = _load_gene_order()
    X_ctrl, X_pert, n_perturbations, n_pairs, n_dropped = _load_norman_data(
        adata_path, gene_order, min_cells_per_pseudobulk
    )

    if n_pairs == 0:
        logger.warning("No pseudo-bulk pairs survived filtering")
        return NormanEvalReport(
            run_id=run_id,
            n_perturbations=n_perturbations,
            n_pseudobulk_pairs=0,
            n_dropped_groups=n_dropped,
            ctrl_memorisation_score=1.0,
            failure_reason="No pseudo-bulk pairs survived min_cells filtering",
        )

    # Load model and run inference
    model = _load_model(checkpoint_path, dev)

    # Build edge index
    import pandas as pd

    edge_df = pd.read_csv("data/edge_list_fixed.csv")
    gene_to_idx = {g: i for i, g in enumerate(gene_order)}
    edges = []
    for _, row in edge_df.iterrows():
        a = gene_to_idx.get(row["gene_a"])
        b = gene_to_idx.get(row["gene_b"])
        if a is not None and b is not None:
            edges.append([a, b])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(dev)

    # pert_id = 1 for all perturbations (stim token)
    # K562 ct_id=0 is a placeholder — see v1.2 cell-type flag
    pert_id = torch.tensor([1], device=dev)

    X_ctrl_t = torch.tensor(X_ctrl, dtype=torch.float32).to(dev)
    with torch.no_grad():
        pred = model.forward_batch(X_ctrl_t, edge_index, pert_id)
    pred_np = pred.cpu().numpy()

    # Compute metrics
    r_ctrl_sub = pearson_r_ctrl_subtracted(pred_np, X_ctrl, X_pert)
    if np.isnan(r_ctrl_sub):
        logger.warning(
            f"pearson_r_ctrl_sub is NaN for run {run_id} — "
            "coercing to 0.0. Ground truth delta is degenerate."
        )
        r_ctrl_sub = 0.0

    dnz = delta_nonzero_pct(pred_np, X_ctrl)
    cms = ctrl_memorisation_score(pred_np, X_ctrl)
    tkv = top_k_gene_overlap(pred_np - X_ctrl, X_pert - X_ctrl)

    # Pass gate
    passed = (dnz > DELTA_NONZERO_THRESHOLD) and (r_ctrl_sub > 0.0)

    failure_reason = None
    if not passed:
        failure_reason = (
            f"Norman pass gate: delta_nonzero_pct={dnz:.2f}% "
            f"(threshold={DELTA_NONZERO_THRESHOLD}%), "
            f"pearson_r_ctrl_sub={r_ctrl_sub:.4f}"
        )

    return NormanEvalReport(
        run_id=run_id,
        n_perturbations=n_perturbations,
        n_pseudobulk_pairs=n_pairs,
        n_dropped_groups=n_dropped,
        pearson_r_ctrl_sub=r_ctrl_sub,
        delta_nonzero_pct=dnz,
        ctrl_memorisation_score=cms,
        top_k_overlap=tkv,
        passed=passed,
        failure_reason=failure_reason,
    )
