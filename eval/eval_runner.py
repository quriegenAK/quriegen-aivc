"""
eval/eval_runner.py — AIVC evaluation suite orchestrator.

Runs benchmarks in fixed sequence:
  1. Kang 2018 (regression guard) — halts suite if failed
  2. Norman 2019 (zero-shot transfer)
  3. Replogle 2022 — skipped in v1

Replogle v1.2: data/replogle2022_safe_267.h5ad not yet on disk.
ReplogleEvalReport schema lives in eval/benchmarks/replogle_eval.py (stub).
Enable when file is staged and num_perturbations is expanded to 267.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel

from eval.benchmarks.kang_eval import KangEvalReport, run_kang_eval
from eval.benchmarks.norman_eval import NormanEvalReport, run_norman_eval
from eval.benchmarks.replogle_eval import ReplogleEvalReport
from eval.exceptions import CheckpointRejected

if TYPE_CHECKING:
    from aivc_platform.tracking.schemas import RunMetadata

logger = logging.getLogger("aivc.eval.runner")


class EvalSuite(BaseModel):
    """Complete evaluation suite results."""

    kang: KangEvalReport
    norman: Optional[NormanEvalReport] = None
    # D1: Replogle skipped — data/replogle2022_safe_267.h5ad
    #     not on disk. Enable in v1.2.
    replogle: Optional[ReplogleEvalReport] = None
    overall_passed: bool = False
    halt_reason: Optional[str] = None


def run_eval_suite(
    checkpoint_path: str,
    *,
    run_id: str,
    kang_adata: str = "data/kang2018_pbmc_fixed.h5ad",
    norman_adata: str = "data/norman2019.h5ad",
    device: str | None = None,
) -> EvalSuite:
    """Run the full evaluation suite.

    Sequence:
      1. Kang regression guard — if fails, halt (Norman + Replogle skipped)
      2. Norman zero-shot — always runs if Kang passes
      3. Replogle — skipped in v1

    Args:
        checkpoint_path: Path to model checkpoint.
        run_id: Unique run identifier.
        kang_adata: Path to Kang h5ad.
        norman_adata: Path to Norman h5ad.
        device: "cuda", "cpu", or None for auto-detect.

    Returns:
        EvalSuite with all benchmark results and overall pass/fail.
    """
    logger.info(f"Starting eval suite: run_id={run_id}")

    # Step 1: Kang regression guard
    kang = run_kang_eval(
        checkpoint_path, kang_adata, run_id=run_id, device=device
    )

    if not kang.regression_guard_passed:
        logger.warning(f"Kang regression guard FAILED: {kang.failure_reason}")
        return EvalSuite(
            kang=kang,
            norman=None,
            replogle=None,
            overall_passed=False,
            halt_reason=kang.failure_reason,
        )

    # Step 2: Norman zero-shot
    norman: Optional[NormanEvalReport] = None
    try:
        norman = run_norman_eval(
            checkpoint_path, norman_adata, run_id=run_id, device=device
        )
    except Exception as e:
        logger.error(f"Norman eval failed with exception: {e}")
        norman = None

    # Step 3: Replogle — skipped in v1
    # D1: Replogle skipped — data/replogle2022_safe_267.h5ad
    #     not on disk. Enable in v1.2.

    # Overall pass: Kang AND Norman must pass
    overall = kang.passed and (norman.passed if norman else False)

    return EvalSuite(
        kang=kang,
        norman=norman,
        replogle=None,
        overall_passed=overall,
        halt_reason=None if overall else (
            kang.failure_reason or
            (norman.failure_reason if norman else "Norman eval not available")
        ),
    )


def populate_run_metadata(
    meta: RunMetadata,
    suite: EvalSuite,
) -> RunMetadata:
    """Populate RunMetadata eval fields from EvalSuite.

    Caller must call this before ExperimentLogger.finish().
    """
    meta.pearson_r = suite.kang.pearson_r
    if suite.norman:
        meta.delta_nonzero_pct = suite.norman.delta_nonzero_pct
        meta.ctrl_memorisation_score = suite.norman.ctrl_memorisation_score
    return meta


if __name__ == "__main__":
    # python -m eval.eval_runner --checkpoint <path> --run-id <id>
    parser = argparse.ArgumentParser(description="AIVC evaluation suite")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--run-id", required=True, help="Unique run identifier")
    parser.add_argument("--kang-adata", default="data/kang2018_pbmc_fixed.h5ad")
    parser.add_argument("--norman-adata", default="data/norman2019.h5ad")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")

    suite = run_eval_suite(
        args.checkpoint,
        run_id=args.run_id,
        kang_adata=args.kang_adata,
        norman_adata=args.norman_adata,
        device=args.device,
    )

    print(f"\n{'='*60}")
    print(f"EVAL SUITE: {'PASSED' if suite.overall_passed else 'FAILED'}")
    print(f"{'='*60}")
    print(f"  Kang:    passed={suite.kang.passed}  r={suite.kang.pearson_r:.4f}")
    if suite.norman:
        print(
            f"  Norman:  passed={suite.norman.passed}  "
            f"delta_nz={suite.norman.delta_nonzero_pct:.1f}%  "
            f"r_ctrl_sub={suite.norman.pearson_r_ctrl_sub:.4f}"
        )
    else:
        print("  Norman:  skipped")
    print(f"  Replogle: skipped (v1)")
    if suite.halt_reason:
        print(f"  Halt:    {suite.halt_reason}")

    sys.exit(0 if suite.overall_passed else 1)
