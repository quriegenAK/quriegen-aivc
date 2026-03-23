"""
Shared helpers for the ATAC-RNA preprocessing pipeline.
"""
import time
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("aivc.preprocessing")

SEED = 42

JAKSTAT_GENES = [
    "JAK1", "JAK2", "STAT1", "STAT2", "STAT3",
    "IRF9", "IRF1", "MX1", "MX2", "ISG15",
    "OAS1", "IFIT1", "IFIT3", "IFITM1", "IFITM3",
]

IFN_BETA_TFS = ["STAT1", "STAT2", "IRF3", "IRF7", "IRF1", "IRF9"]


class StepTimer:
    """Context manager for timing pipeline steps."""

    def __init__(self, step_name: str):
        self.step_name = step_name
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.time()
        logger.info(f"[START] {self.step_name}")
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        logger.info(f"[DONE]  {self.step_name} ({self.elapsed:.1f}s)")


def log_cell_counts(mdata, step_name: str, modality: str = "rna") -> dict:
    """Log cell and feature counts after a pipeline step."""
    n_cells = mdata.n_obs
    info = {"step": step_name, "n_cells": n_cells}

    if "rna" in mdata.mod:
        info["n_rna_cells"] = mdata["rna"].n_obs
        info["n_genes"] = mdata["rna"].n_vars
    if "atac" in mdata.mod:
        info["n_atac_cells"] = mdata["atac"].n_obs
        info["n_peaks"] = mdata["atac"].n_vars

    logger.info(
        f"  {step_name}: {n_cells} cells"
        + (f", {info.get('n_genes', '?')} genes" if "n_genes" in info else "")
        + (f", {info.get('n_peaks', '?')} peaks" if "n_peaks" in info else "")
    )
    return info


def compute_mt_fraction(adata):
    """Compute mitochondrial gene fraction for an RNA AnnData."""
    import scanpy as sc

    mt_mask = adata.var_names.str.startswith("MT-") | adata.var_names.str.startswith("mt-")
    if mt_mask.sum() == 0:
        logger.warning("No MT- genes found. Setting pct_mt = 0 for all cells.")
        adata.obs["pct_mt"] = 0.0
    else:
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
        )
        adata.obs["pct_mt"] = adata.obs.get("pct_counts_mt", 0.0)
    return adata


def compute_tss_enrichment(adata, fragments_path: Optional[str] = None):
    """
    Compute TSS enrichment score for ATAC data.

    In production: uses fragment file + TSS annotations.
    For mock data: if 'tss_score' already in obs, keep it.
    """
    if "tss_score" in adata.obs.columns:
        logger.info("  TSS scores already present in obs.")
        return adata

    if fragments_path is not None:
        try:
            import muon as mu
            mu.atac.tl.tss_enrichment(adata, features=None)
        except Exception as e:
            logger.warning(f"TSS enrichment computation failed: {e}. Setting default=5.0")
            adata.obs["tss_score"] = 5.0
    else:
        logger.info("  No fragments file — setting default tss_score=5.0")
        adata.obs["tss_score"] = 5.0

    return adata
