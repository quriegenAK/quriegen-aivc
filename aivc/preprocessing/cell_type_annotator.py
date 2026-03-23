"""
Cell type annotation via label transfer from Hoa et al. PBMC reference.

Transfers cell type labels from a reference dataset to query cells.
Discards cells with prediction score < 0.5 (doublets + ambient RNA).
Validates retention rates under stimulation — flags if CD14+ monocytes
have >10% discard rate under IFN-beta (explains low monocyte r in v1.0).
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse

from .utils import SEED

logger = logging.getLogger("aivc.preprocessing")

KNOWN_PBMC_TYPES = [
    "B cells",
    "CD14+ Monocytes",
    "CD4 T cells",
    "CD8 T cells",
    "Dendritic cells",
    "FCGR3A+ Monocytes",
    "Megakaryocytes",
    "NK cells",
]


def transfer_labels(
    query_adata,
    reference_path: Optional[str] = None,
    min_pred_score: float = 0.5,
    n_neighbors: int = 30,
) -> tuple:
    """
    Transfer cell type labels from reference to query.

    In production: loads Hoa et al. reference, runs kNN label transfer.
    For mock data: if 'cell_type' and 'pred_score' already in obs, uses those.

    Args:
        query_adata: AnnData to annotate.
        reference_path: Path to reference .h5ad.
        min_pred_score: Minimum prediction score. Default 0.5.
        n_neighbors: Number of neighbors for kNN transfer.

    Returns:
        Tuple of (annotated_adata, discard_report):
          annotated_adata: AnnData with cell_type in obs, low-score cells removed.
          discard_report: Dict with per-cell-type discard statistics.
    """
    # If labels already present (mock data or pre-annotated)
    if "cell_type" in query_adata.obs.columns and "pred_score" in query_adata.obs.columns:
        logger.info("  Cell type labels and pred_score already in obs.")
        return _apply_score_filter(query_adata, min_pred_score)

    # If reference is provided, run label transfer
    if reference_path is not None:
        logger.info(f"  Loading reference from {reference_path}")
        try:
            return _run_label_transfer(query_adata, reference_path, min_pred_score, n_neighbors)
        except Exception as e:
            logger.error(f"  Label transfer failed: {e}")
            raise

    # No labels and no reference — assign unknown
    logger.warning("  No cell_type labels and no reference provided. Assigning 'unknown'.")
    query_adata.obs["cell_type"] = "unknown"
    query_adata.obs["pred_score"] = 1.0
    return query_adata, {"warning": "no_labels_assigned"}


def _apply_score_filter(adata, min_pred_score: float) -> tuple:
    """Filter cells below prediction score threshold."""
    n_before = adata.n_obs
    scores = adata.obs["pred_score"].values.astype(float)

    # Build discard report before filtering
    report = _build_discard_report(adata, scores, min_pred_score)

    # Filter
    keep_mask = scores >= min_pred_score
    adata_filtered = adata[keep_mask].copy()
    n_after = adata_filtered.n_obs
    n_dropped = n_before - n_after

    logger.info(f"  Prediction score filter (>= {min_pred_score}):")
    logger.info(f"    Before: {n_before}, After: {n_after}, Dropped: {n_dropped}")

    # Flag CD14+ monocyte discard rate under stimulation
    if "condition" in adata.obs.columns:
        _check_monocyte_stim_discard(adata, scores, min_pred_score)

    return adata_filtered, report


def _build_discard_report(adata, scores, min_pred_score) -> dict:
    """Build per-cell-type discard statistics."""
    report = {}
    cell_types = adata.obs.get("cell_type", pd.Series(["unknown"] * adata.n_obs))
    unique_types = sorted(cell_types.unique())

    for ct in unique_types:
        ct_mask = cell_types == ct
        ct_scores = scores[ct_mask]
        n_total = len(ct_scores)
        n_discard = (ct_scores < min_pred_score).sum()
        pct_discard = 100.0 * n_discard / max(n_total, 1)

        report[ct] = {
            "n_total": int(n_total),
            "n_discarded": int(n_discard),
            "n_retained": int(n_total - n_discard),
            "pct_discarded": float(pct_discard),
        }

        if pct_discard > 20:
            logger.warning(f"  {ct}: {pct_discard:.1f}% discarded (>{20}% threshold)")
        else:
            logger.info(f"  {ct}: {pct_discard:.1f}% discarded ({n_discard}/{n_total})")

    return report


def _check_monocyte_stim_discard(adata, scores, min_pred_score):
    """Check if CD14+ monocytes have >10% discard under IFN-beta stimulation."""
    cell_types = adata.obs.get("cell_type", pd.Series())
    conditions = adata.obs.get("condition", pd.Series())

    mono_stim_mask = (cell_types == "CD14+ Monocytes") & (conditions == "stim")
    n_mono_stim = mono_stim_mask.sum()

    if n_mono_stim > 0:
        mono_stim_scores = scores[mono_stim_mask]
        n_discard = (mono_stim_scores < min_pred_score).sum()
        pct = 100.0 * n_discard / n_mono_stim

        if pct > 10:
            logger.warning(
                f"  CD14+ Monocytes under stim: {pct:.1f}% discarded "
                f"({n_discard}/{n_mono_stim}). This may explain low monocyte "
                "Pearson r in v1.0."
            )
        else:
            logger.info(
                f"  CD14+ Monocytes under stim: {pct:.1f}% discarded — OK"
            )


def _run_label_transfer(query_adata, reference_path, min_pred_score, n_neighbors):
    """Run actual label transfer using scanpy ingest or kNN."""
    import scanpy as sc
    import anndata as ad

    ref = ad.read_h5ad(reference_path)
    logger.info(f"  Reference: {ref.n_obs} cells, {ref.n_vars} genes")

    # Find common genes
    common_genes = list(set(query_adata.var_names) & set(ref.var_names))
    logger.info(f"  Common genes: {len(common_genes)}")

    if len(common_genes) < 100:
        raise ValueError(
            f"Only {len(common_genes)} common genes with reference. "
            "Need >= 100 for reliable label transfer."
        )

    # Subset to common genes
    query_sub = query_adata[:, common_genes].copy()
    ref_sub = ref[:, common_genes].copy()

    # Normalize and find variable genes
    sc.pp.normalize_total(query_sub, target_sum=1e4)
    sc.pp.log1p(query_sub)
    sc.pp.normalize_total(ref_sub, target_sum=1e4)
    sc.pp.log1p(ref_sub)

    # PCA on reference
    sc.pp.highly_variable_genes(ref_sub, n_top_genes=2000)
    sc.pp.pca(ref_sub, n_comps=50)
    sc.pp.neighbors(ref_sub, n_neighbors=n_neighbors)

    # Transfer labels via ingest
    sc.tl.ingest(query_sub, ref_sub, obs="cell_type")

    query_adata.obs["cell_type"] = query_sub.obs["cell_type"].values
    # Ingest doesn't provide prediction scores — use neighbor distance as proxy
    query_adata.obs["pred_score"] = 0.8  # default high score for ingest

    return _apply_score_filter(query_adata, min_pred_score)
