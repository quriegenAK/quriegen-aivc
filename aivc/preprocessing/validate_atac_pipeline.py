"""
Validation suite for the ATAC-RNA preprocessing pipeline.

Three validation checks:
  1. JAK-STAT coverage: >= 10/15 genes must have peak-gene links
  2. Motif enrichment direction: STAT1/IRF3 higher in stim vs ctrl
  3. Cell type retention: no cell type loses >20% of cells
"""
import logging

import numpy as np
import pandas as pd

from .utils import JAKSTAT_GENES, IFN_BETA_TFS
from .tf_motif_scanner import validate_ifn_beta_enrichment

logger = logging.getLogger("aivc.preprocessing")


def validate_jakstat_coverage(
    peak_gene_links: pd.DataFrame,
    min_coverage: int = 10,
) -> dict:
    """
    Check that JAK-STAT pathway genes have peak-gene links.

    Minimum: 10/15 JAK-STAT genes must have at least one co-accessible peak.

    Args:
        peak_gene_links: DataFrame with 'gene_name' column from peak_gene_linker.
        min_coverage: Minimum number of JAK-STAT genes that must have links.

    Returns:
        Dict with coverage details and pass/fail status.
    """
    if peak_gene_links is None or len(peak_gene_links) == 0:
        logger.warning("No peak-gene links provided. JAK-STAT coverage: 0/15")
        return {
            "passed": False,
            "coverage": 0,
            "total": 15,
            "covered_genes": [],
            "missing_genes": JAKSTAT_GENES.copy(),
        }

    linked_genes = set(peak_gene_links["gene_name"].unique())

    covered = [g for g in JAKSTAT_GENES if g in linked_genes]
    missing = [g for g in JAKSTAT_GENES if g not in linked_genes]
    n_covered = len(covered)

    passed = n_covered >= min_coverage

    if passed:
        logger.info(f"  JAK-STAT coverage: {n_covered}/15 (PASS, >= {min_coverage})")
    else:
        logger.error(
            f"  JAK-STAT coverage: {n_covered}/15 (FAIL, need >= {min_coverage}). "
            f"Missing: {missing}"
        )

    return {
        "passed": passed,
        "coverage": n_covered,
        "total": 15,
        "min_required": min_coverage,
        "covered_genes": covered,
        "missing_genes": missing,
    }


def validate_motif_enrichment_direction(
    chromvar_scores: np.ndarray,
    tf_names: list,
    conditions: np.ndarray,
    ctrl_label: str = "ctrl",
    stim_label: str = "stim",
) -> dict:
    """
    Validate that IFN-beta stimulated cells show higher STAT1, IRF3 motif
    scores than control cells (t-test, p < 0.05).

    This is the ATAC equivalent of the JAK-STAT RNA recovery check.
    If STAT1 motif is NOT enriched in stim: TF scanning failed.

    Args:
        chromvar_scores: Matrix (n_cells, n_TFs).
        tf_names: TF name list matching columns.
        conditions: Per-cell condition labels.
        ctrl_label: Control condition label.
        stim_label: Stimulation condition label.

    Returns:
        Dict with per-TF validation and overall pass/fail.
    """
    results = validate_ifn_beta_enrichment(
        chromvar_scores, tf_names, conditions, ctrl_label, stim_label
    )

    # Check critical TFs
    critical_tfs = ["STAT1", "IRF3"]
    critical_pass = all(
        results.get(tf, {}).get("enriched", False) for tf in critical_tfs
        if tf in {name for name in tf_names}
    )

    results["critical_tfs_pass"] = critical_pass

    if critical_pass:
        logger.info("  Motif enrichment direction: PASS (STAT1, IRF3 enriched in stim)")
    else:
        logger.error(
            "  Motif enrichment direction: FAIL. "
            "STAT1 and/or IRF3 not enriched in stimulated cells. "
            "TF motif scanning may have failed."
        )

    return results


def validate_cell_type_retention(
    obs_before: pd.DataFrame,
    obs_after: pd.DataFrame,
    cell_type_col: str = "cell_type",
    max_loss_pct: float = 20.0,
    monocyte_max_loss_pct: float = 10.0,
) -> dict:
    """
    Report per-cell-type: n_cells before QC, n_cells after QC.
    Flag if any cell type loses >20% of cells.
    CD14+ monocytes: flag if >10% lost under stim.

    Args:
        obs_before: obs DataFrame before QC filtering.
        obs_after: obs DataFrame after QC filtering.
        cell_type_col: Column name for cell type labels.
        max_loss_pct: Maximum acceptable loss percentage per cell type.
        monocyte_max_loss_pct: Special threshold for CD14+ monocytes.

    Returns:
        Dict with per-cell-type retention stats and flags.
    """
    if cell_type_col not in obs_before.columns:
        logger.warning(f"  '{cell_type_col}' not in obs_before. Cannot validate retention.")
        return {"passed": True, "warning": "no_cell_types"}

    before_counts = obs_before[cell_type_col].value_counts().to_dict()

    if cell_type_col in obs_after.columns:
        after_counts = obs_after[cell_type_col].value_counts().to_dict()
    else:
        after_counts = {}

    results = {}
    all_pass = True

    for ct in sorted(before_counts.keys()):
        n_before = before_counts[ct]
        n_after = after_counts.get(ct, 0)
        n_lost = n_before - n_after
        pct_lost = 100.0 * n_lost / max(n_before, 1)

        threshold = monocyte_max_loss_pct if ct == "CD14+ Monocytes" else max_loss_pct
        passed = pct_lost <= threshold

        results[ct] = {
            "n_before": n_before,
            "n_after": n_after,
            "n_lost": n_lost,
            "pct_lost": round(pct_lost, 1),
            "threshold": threshold,
            "passed": passed,
        }

        if not passed:
            all_pass = False
            logger.warning(
                f"  {ct}: lost {pct_lost:.1f}% ({n_lost}/{n_before}) — "
                f"EXCEEDS {threshold}% threshold"
            )
        else:
            logger.info(
                f"  {ct}: lost {pct_lost:.1f}% ({n_lost}/{n_before}) — OK"
            )

    results["all_pass"] = all_pass
    return results
