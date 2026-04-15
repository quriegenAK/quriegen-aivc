"""scripts/validate_run.py — Classify a run as SUCCESS / REGRESSION / FAILURE.

Per AIVC hardening spec §3.2 and §9 status matrix.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel

if TYPE_CHECKING:
    from aivc_platform.tracking.schemas import RunMetadata
    from eval.eval_runner import EvalSuite

from scripts.build_signature import (
    EXPERIMENTS_DIR,
    REPO_ROOT,
    RunSignature,
    get_best_signature,
    load_signature,
)

logger = logging.getLogger("aivc.scripts.validate_run")

# Defaults from D1 (overridden by configs/thresholds.yaml if present).
PEARSON_R_FLOOR = 0.873
REGRESSION_TOLERANCE = 0.005
DELTA_NONZERO_FLOOR = 1.0
PROMOTION_MARGIN = 0.001

_THRESHOLDS_YAML = REPO_ROOT / "configs" / "thresholds.yaml"
if _THRESHOLDS_YAML.exists():
    try:
        import yaml
        _t = yaml.safe_load(_THRESHOLDS_YAML.read_text(encoding="utf-8")) or {}
        PEARSON_R_FLOOR = float(_t.get("pearson_r_floor", PEARSON_R_FLOOR))
        REGRESSION_TOLERANCE = float(_t.get("pearson_r_regression_tolerance", REGRESSION_TOLERANCE))
        DELTA_NONZERO_FLOOR = float(_t.get("delta_nonzero_pct_floor", DELTA_NONZERO_FLOOR))
        PROMOTION_MARGIN = float(_t.get("best_checkpoint_promotion_margin", PROMOTION_MARGIN))
    except Exception as e:
        logger.warning(f"Could not parse thresholds.yaml, using D1 defaults: {e}")


class RunStatus(str, Enum):
    SUCCESS = "SUCCESS"
    REGRESSION = "REGRESSION"
    FAILURE = "FAILURE"


class ValidationResult(BaseModel):
    status: RunStatus
    run_id: str
    pearson_r: float
    delta_nonzero_pct: float
    best_pearson_r: Optional[float] = None
    regression_detected: bool = False
    failure_reason: Optional[str] = None
    promote_checkpoint: bool = False


def _kang_guard_passed(suite) -> bool:
    try:
        return bool(suite.kang.regression_guard_passed)
    except AttributeError:
        return False


def classify_run(meta, suite, best_sig: Optional[RunSignature]) -> ValidationResult:
    """Classify a run per spec §3.2 / §9."""
    pearson_r = float(getattr(meta, "pearson_r", 0.0))
    dnz = float(getattr(meta, "delta_nonzero_pct", 0.0))
    best_r = best_sig.metrics.pearson_r if best_sig else None

    # NaN check first
    if math.isnan(pearson_r) or math.isnan(dnz):
        return ValidationResult(
            status=RunStatus.FAILURE,
            run_id=meta.run_id,
            pearson_r=pearson_r,
            delta_nonzero_pct=dnz,
            best_pearson_r=best_r,
            failure_reason="nan_metrics",
        )

    kang_ok = _kang_guard_passed(suite)

    # FAILURE conditions (highest priority among non-success after NaN)
    if not kang_ok:
        return ValidationResult(
            status=RunStatus.FAILURE,
            run_id=meta.run_id,
            pearson_r=pearson_r,
            delta_nonzero_pct=dnz,
            best_pearson_r=best_r,
            failure_reason="kang_guard_failed",
        )

    if pearson_r < PEARSON_R_FLOOR:
        return ValidationResult(
            status=RunStatus.FAILURE,
            run_id=meta.run_id,
            pearson_r=pearson_r,
            delta_nonzero_pct=dnz,
            best_pearson_r=best_r,
            failure_reason="below_floor",
        )

    if dnz <= DELTA_NONZERO_FLOOR:
        return ValidationResult(
            status=RunStatus.FAILURE,
            run_id=meta.run_id,
            pearson_r=pearson_r,
            delta_nonzero_pct=dnz,
            best_pearson_r=best_r,
            failure_reason="delta_collapse",
        )

    # REGRESSION
    if best_r is not None and pearson_r < best_r - REGRESSION_TOLERANCE:
        return ValidationResult(
            status=RunStatus.REGRESSION,
            run_id=meta.run_id,
            pearson_r=pearson_r,
            delta_nonzero_pct=dnz,
            best_pearson_r=best_r,
            regression_detected=True,
            failure_reason="pearson_regression",
        )

    # SUCCESS
    promote = best_r is None or pearson_r > best_r + PROMOTION_MARGIN
    return ValidationResult(
        status=RunStatus.SUCCESS,
        run_id=meta.run_id,
        pearson_r=pearson_r,
        delta_nonzero_pct=dnz,
        best_pearson_r=best_r,
        promote_checkpoint=promote,
    )


def promote_checkpoint(checkpoint_path: str) -> None:
    """Atomically symlink experiments/best/best.pt → checkpoint_path."""
    best_dir = EXPERIMENTS_DIR / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    target = Path(checkpoint_path)
    if not target.is_absolute():
        target = REPO_ROOT / target
    link_path = best_dir / "best.pt"
    tmp_link = best_dir / f".best.pt.tmp.{os.getpid()}"
    try:
        if tmp_link.exists() or tmp_link.is_symlink():
            tmp_link.unlink()
    except OSError:
        pass
    try:
        os.symlink(str(target), str(tmp_link))
        os.replace(str(tmp_link), str(link_path))
    except OSError as e:
        logger.warning(f"promote_checkpoint symlink failed: {e}")
        try:
            tmp_link.unlink()
        except OSError:
            pass


def _save_winning_signature(sig: RunSignature) -> None:
    best_dir = EXPERIMENTS_DIR / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    dst = best_dir / "signature.json"
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=str(best_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(sig.model_dump_json(indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dst)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def classify_run_from_sig(run_id: str) -> Optional[ValidationResult]:
    """Re-classify a stored run via signature + cached eval_suite.json."""
    sig = load_signature(run_id)
    if sig is None:
        return None
    suite_path = EXPERIMENTS_DIR / run_id / "eval_suite.json"
    suite = None
    if suite_path.exists():
        try:
            from eval.eval_runner import EvalSuite
            suite = EvalSuite.model_validate_json(suite_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"could not parse eval_suite.json: {e}")

    class _Meta:
        pass
    m = _Meta()
    m.run_id = run_id
    m.pearson_r = sig.metrics.pearson_r
    m.delta_nonzero_pct = sig.metrics.delta_nonzero_pct
    m.checkpoint_path = sig.model.checkpoint_path

    if suite is None:
        # Fall back to empty stub: kang.regression_guard_passed inferred from sig status
        class _K:
            regression_guard_passed = sig.status != "FAILURE"
        class _S:
            kang = _K()
        suite = _S()

    best = get_best_signature()
    # Avoid comparing against self
    if best and best.run_id == run_id:
        best = None
    return classify_run(m, suite, best)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    res = classify_run_from_sig(args.run_id)
    if res is None:
        print(json.dumps({"error": "no signature", "run_id": args.run_id}))
        sys.exit(1)
    print(res.model_dump_json(indent=2))
    sys.exit(0 if res.status == RunStatus.SUCCESS else 1)
