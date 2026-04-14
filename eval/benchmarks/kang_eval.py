"""
eval/benchmarks/kang_eval.py — Kang 2018 PBMC IFN-b evaluation.

Regression guard: pearson_r must be >= 0.873 (v1.0 baseline).
A checkpoint that fails the regression guard is never registered.

Data layout (kang2018_pbmc_fixed.h5ad):
  obs["label"]:     "ctrl" | "stim"
  obs["replicate"]: donor ID (e.g. "patient_101")
  obs["cell_type"]: PBMC cell type (e.g. "CD14+ Monocytes")
"""
from __future__ import annotations

import logging
import os
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

logger = logging.getLogger("aivc.eval.kang")

BASELINE_R = 0.873
N_GENES = 3010
DELTA_NONZERO_THRESHOLD = 1.0  # v1 — raise to 2-5% in v2


class KangEvalReport(BaseModel):
    """Evaluation report for Kang 2018 benchmark."""

    run_id: str
    dataset: str = "kang2018_pbmc"
    n_genes: int = N_GENES
    n_test_pairs: int = 0

    pearson_r: float = 0.0
    pearson_r_std: float = 0.0
    pearson_r_ctrl_sub: float = 0.0
    delta_nonzero_pct: float = 0.0
    ctrl_memorisation_score: float = 0.0
    top_k_overlap: float = 0.0

    regression_guard_passed: bool = False
    passed: bool = False
    failure_reason: Optional[str] = None

    @model_validator(mode="after")
    def _n_genes_is_3010(self) -> KangEvalReport:
        if self.n_genes != N_GENES:
            raise ValueError(
                f"Kang eval requires exactly {N_GENES} genes, got {self.n_genes}"
            )
        return self


def _load_gene_order(path: str = "data/gene_names.txt") -> list[str]:
    """Load the canonical 3010-gene order."""
    with open(path) as f:
        genes = [line.strip() for line in f if line.strip()]
    assert len(genes) == N_GENES, f"Expected {N_GENES} genes, got {len(genes)}"
    return genes


def _load_test_data(
    adata_path: str,
    gene_order: list[str],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load Kang test split data.

    Returns:
        (X_ctrl, X_stim, n_test_pairs) — pseudo-bulk arrays aligned to gene_order.
    """
    import anndata as ad

    # Try pre-paired arrays first
    if (
        os.path.exists("data/X_ctrl_paired.npy")
        and os.path.exists("data/X_stim_paired.npy")
    ):
        X_ctrl = np.load("data/X_ctrl_paired.npy")
        X_stim = np.load("data/X_stim_paired.npy")
        logger.info(f"Loaded pre-paired arrays: {X_ctrl.shape[0]} pairs")

        # Use test split if available
        if os.path.exists("data/test_split_info.pt"):
            split = torch.load("data/test_split_info.pt", map_location="cpu", weights_only=False)
            test_idx = split.get("test_idx", list(range(X_ctrl.shape[0])))
            X_ctrl = X_ctrl[test_idx]
            X_stim = X_stim[test_idx]
            logger.info(f"Applied test split: {len(test_idx)} test pairs")
        else:
            logger.warning(
                "data/test_split_info.pt not found — using last 2 donors as fallback"
            )
            # Load manifest to get donors
            import pandas as pd

            if os.path.exists("data/pairing_manifest.csv"):
                manifest = pd.read_csv("data/pairing_manifest.csv")
                paired = manifest[manifest["paired"]].reset_index(drop=True)
                donors = sorted(paired["donor_id"].unique())
                test_donors = set(donors[-2:])
                test_mask = paired["donor_id"].isin(test_donors).values
                X_ctrl = X_ctrl[test_mask]
                X_stim = X_stim[test_mask]
                logger.info(f"Fallback test split: {X_ctrl.shape[0]} pairs from donors {test_donors}")

        return X_ctrl, X_stim, X_ctrl.shape[0]

    # Fall back to re-pairing from h5ad
    logger.info(f"Loading from h5ad: {adata_path}")
    adata = ad.read_h5ad(adata_path)

    gene_set = set(gene_order)
    adata_genes = set(adata.var_names if "name" not in adata.var.columns else adata.var["name"])
    intersection = gene_set & adata_genes
    if len(intersection) != N_GENES:
        # Try var["name"] column
        if "name" in adata.var.columns:
            adata_genes = set(adata.var["name"])
            intersection = gene_set & adata_genes
    assert len(intersection) == N_GENES, (
        f"Gene alignment failed: {len(intersection)}/{N_GENES} genes found. "
        f"Missing: {gene_set - adata_genes}"
    )

    # Use var["name"] for indexing if available
    if "name" in adata.var.columns:
        gene_idx = [list(adata.var["name"]).index(g) for g in gene_order]
    else:
        gene_idx = [list(adata.var_names).index(g) for g in gene_order]

    ctrl_mask = adata.obs["label"] == "ctrl"
    stim_mask = adata.obs["label"] == "stim"
    donors = sorted(adata.obs["replicate"].unique())

    # Last 2 donors as test
    test_donors = set(donors[-2:])
    logger.info(f"Test donors: {test_donors}")

    X_ctrl_list, X_stim_list = [], []
    for donor in test_donors:
        for ct in adata.obs["cell_type"].unique():
            donor_ct_ctrl = (
                ctrl_mask
                & (adata.obs["replicate"] == donor)
                & (adata.obs["cell_type"] == ct)
            )
            donor_ct_stim = (
                stim_mask
                & (adata.obs["replicate"] == donor)
                & (adata.obs["cell_type"] == ct)
            )
            if donor_ct_ctrl.sum() < 5 or donor_ct_stim.sum() < 5:
                continue
            X_c = np.asarray(adata.X[donor_ct_ctrl][:, gene_idx].mean(axis=0)).flatten()
            X_s = np.asarray(adata.X[donor_ct_stim][:, gene_idx].mean(axis=0)).flatten()
            X_ctrl_list.append(X_c)
            X_stim_list.append(X_s)

    X_ctrl = np.array(X_ctrl_list)
    X_stim = np.array(X_stim_list)
    return X_ctrl, X_stim, X_ctrl.shape[0]


def _load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load PerturbationPredictor from checkpoint."""
    from perturbation_model import PerturbationPredictor, CellTypeEmbedding

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle both direct state_dict and wrapped checkpoint formats
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


def run_kang_eval(
    checkpoint_path: str,
    adata_path: str = "data/kang2018_pbmc_fixed.h5ad",
    *,
    run_id: str,
    device: str | None = None,
) -> KangEvalReport:
    """Run Kang 2018 evaluation with regression guard.

    Args:
        checkpoint_path: Path to model checkpoint.
        adata_path: Path to Kang h5ad file.
        run_id: Unique run identifier.
        device: "cuda", "cpu", or None for auto-detect.

    Returns:
        KangEvalReport with all metrics and pass/fail status.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Kang eval: device={dev}, checkpoint={checkpoint_path}")

    gene_order = _load_gene_order()
    X_ctrl, X_stim, n_test_pairs = _load_test_data(adata_path, gene_order)
    logger.info(f"Test data: {n_test_pairs} pairs, {N_GENES} genes")

    # Load model and run inference
    model = _load_model(checkpoint_path, dev)

    # Build edge index (needed for forward pass)
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

    pert_id = torch.tensor([1], device=dev)

    # Inference
    X_ctrl_t = torch.tensor(X_ctrl, dtype=torch.float32).to(dev)
    with torch.no_grad():
        pred = model.forward_batch(X_ctrl_t, edge_index, pert_id)
    pred_np = pred.cpu().numpy()

    # Compute metrics
    from eval.metrics import _per_cell_pearson

    rs = _per_cell_pearson(pred_np, X_stim)
    r_mean = float(np.mean(rs))
    r_std = float(np.std(rs))

    r_ctrl_sub = pearson_r_ctrl_subtracted(pred_np, X_ctrl, X_stim)
    if np.isnan(r_ctrl_sub):
        logger.warning(
            f"pearson_r_ctrl_sub is NaN for run {run_id} — "
            "coercing to 0.0. Ground truth delta is degenerate."
        )
        r_ctrl_sub = 0.0

    dnz = delta_nonzero_pct(pred_np, X_ctrl)
    cms = ctrl_memorisation_score(pred_np, X_ctrl)
    tkv = top_k_gene_overlap(pred_np - X_ctrl, X_stim - X_ctrl)

    # Regression guard
    regression_passed = r_mean >= BASELINE_R

    # Pass gate
    passed = regression_passed and (dnz > DELTA_NONZERO_THRESHOLD) and (r_ctrl_sub > 0.0)

    failure_reason = None
    if not regression_passed:
        failure_reason = (
            f"Kang regression guard: r={r_mean:.4f} < {BASELINE_R}"
        )
    elif not passed:
        failure_reason = (
            f"Kang pass gate: delta_nonzero_pct={dnz:.2f}% <= {DELTA_NONZERO_THRESHOLD}% "
            f"or r_ctrl_sub={r_ctrl_sub:.4f} <= 0.0"
        )

    return KangEvalReport(
        run_id=run_id,
        n_test_pairs=n_test_pairs,
        pearson_r=r_mean,
        pearson_r_std=r_std,
        pearson_r_ctrl_sub=r_ctrl_sub,
        delta_nonzero_pct=dnz,
        ctrl_memorisation_score=cms,
        top_k_overlap=tkv,
        regression_guard_passed=regression_passed,
        passed=passed,
        failure_reason=failure_reason,
    )
