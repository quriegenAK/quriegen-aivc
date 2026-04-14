"""
eval/benchmarks/replogle_eval.py — Replogle 2022 evaluation (STUB).

Replogle v1.2: data/replogle2022_safe_267.h5ad not yet on disk.
This stub defines the ReplogleEvalReport schema and a placeholder
run_replogle_eval() that raises NotImplementedError.

Enable when:
  1. data/replogle2022_safe_267.h5ad is staged and validated
  2. num_perturbations is expanded to >=267
  3. EvalDataBlocked governance check is implemented

Data governance: the Replogle dataset contains perturbations outside
the cleared safe-set. When activated, this module will enforce the
safe-set via EvalDataBlocked before running inference.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from eval.exceptions import EvalDataBlocked


class ReplogleEvalReport(BaseModel):
    """Evaluation report for Replogle 2022 benchmark (v1.2)."""

    run_id: str
    dataset: str = "replogle2022"
    n_genes: int = 3010
    n_perturbations: int = 0
    n_pseudobulk_pairs: int = 0
    n_dropped_groups: int = 0

    pearson_r_ctrl_sub: float = 0.0
    delta_nonzero_pct: float = 0.0
    ctrl_memorisation_score: float = 0.0
    top_k_overlap: float = 0.0

    passed: bool = False
    failure_reason: Optional[str] = None


def run_replogle_eval(
    checkpoint_path: str,
    adata_path: str = "data/replogle2022_safe_267.h5ad",
    *,
    run_id: str,
    device: str | None = None,
) -> ReplogleEvalReport:
    """Run Replogle 2022 evaluation.

    NOT IMPLEMENTED in v1. Raises NotImplementedError.
    """
    raise NotImplementedError(
        "Replogle eval disabled in v1. See v1.2 roadmap. "
        "Requires data/replogle2022_safe_267.h5ad and expanded num_perturbations."
    )
