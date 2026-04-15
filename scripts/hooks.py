"""scripts/hooks.py — Central hook dispatcher for train + agent lifecycles.

Per AIVC hardening spec §3.6. Called by cli.py; Makefile also calls sub-modules.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.build_signature import (
    EXPERIMENTS_DIR,
    INDEX_PATH,
    REPO_ROOT,
    build_signature,
    get_best_signature,
    save_signature,
)
from scripts.sync_context import (
    DEFAULT_SNAPSHOT_PATH,
    sync_context,
    verify_snapshot_fresh,
)
from scripts.validate_run import (
    RunStatus as VRunStatus,
    classify_run,
    promote_checkpoint,
)
from scripts.validate_proposal import validate_proposal

if TYPE_CHECKING:
    from aivc_platform.tracking.schemas import RunMetadata
    from eval.eval_runner import EvalSuite
    from agents.base_agent import AgentResult

logger = logging.getLogger("aivc.scripts.hooks")


# ─── Helpers ──────────────────────────────────────────────────────────

def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _run_in_index(run_id: str) -> bool:
    if not INDEX_PATH.exists():
        return False
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                if json.loads(line).get("run_id") == run_id:
                    return True
            except Exception:
                continue
    return False


def _dataset_h5ad_path(dataset_name: str) -> Path:
    name_lower = (dataset_name or "").lower()
    if "kang2018" in name_lower or name_lower == "kang":
        return REPO_ROOT / "data" / "kang2018_pbmc_fixed.h5ad"
    if "norman" in name_lower:
        return REPO_ROOT / "data" / "norman2019.h5ad"
    return REPO_ROOT / "data" / f"{dataset_name}.h5ad"


# ─── Hooks ────────────────────────────────────────────────────────────

def pre_train() -> None:
    """Run before training subprocess. Aborts only on missing dataset."""
    # Freshness check (warn only)
    try:
        context_path = Path(os.environ.get(
            "AIVC_CONTEXT_PATH",
            str(Path("~/Documents/Cowork/AshKhan/Projects/aivc_genelink/context.md").expanduser()),
        ))
        fresh = verify_snapshot_fresh(DEFAULT_SNAPSHOT_PATH, context_path, 24)
        if not fresh:
            logger.warning("pre_train: context snapshot stale or missing (non-fatal).")
    except Exception as e:
        logger.warning(f"pre_train: freshness check failed: {e}")

    # Env checks
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("pre_train: CUDA not available (warn only).")
    except Exception as e:
        logger.warning(f"pre_train: torch import failed: {e}")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("pre_train: ANTHROPIC_API_KEY not set (warn only).")
    if not os.environ.get("WANDB_API_KEY"):
        logger.warning("pre_train: WANDB_API_KEY not set (warn only).")

    # Dataset — hard gate
    dataset = os.environ.get("AIVC_DATASET", "kang2018_pbmc")
    h5ad = _dataset_h5ad_path(dataset)
    if not h5ad.exists():
        logger.error(f"pre_train ABORT: dataset missing: {h5ad}")
        sys.exit(1)


def post_train(meta, suite) -> None:
    """Run after training+eval. Idempotent per run_id."""
    try:
        if _run_in_index(meta.run_id):
            logger.info(f"post_train: run_id={meta.run_id} already in index, skipping.")
            return
    except Exception as e:
        logger.warning(f"post_train: idempotency check failed: {e}")

    # Build & classify
    try:
        sig = build_signature(meta, suite)
    except Exception as e:
        logger.error(f"post_train: build_signature failed: {e}")
        return

    try:
        best = get_best_signature()
    except Exception as e:
        logger.warning(f"post_train: get_best_signature failed: {e}")
        best = None

    try:
        result = classify_run(meta, suite, best)
    except Exception as e:
        logger.error(f"post_train: classify_run failed: {e}")
        return

    sig.status = result.status.value
    # Save signature + index
    try:
        save_signature(sig)
    except Exception as e:
        logger.error(f"post_train: save_signature failed: {e}")
        return

    # Persist eval_suite.json alongside
    try:
        suite_path = EXPERIMENTS_DIR / meta.run_id / "eval_suite.json"
        _atomic_write_json(suite_path, json.loads(suite.model_dump_json()))
    except Exception as e:
        logger.warning(f"post_train: eval_suite.json write failed: {e}")

    if result.status == VRunStatus.SUCCESS:
        # Promote
        try:
            if result.promote_checkpoint and meta.checkpoint_path:
                promote_checkpoint(meta.checkpoint_path)
                # Save winning signature
                try:
                    best_sig_path = EXPERIMENTS_DIR / "best" / "signature.json"
                    _atomic_write_json(best_sig_path, json.loads(sig.model_dump_json()))
                except Exception as e:
                    logger.warning(f"post_train: write best signature: {e}")
        except Exception as e:
            logger.warning(f"post_train: promote_checkpoint failed: {e}")

        # Context sync
        try:
            sync_context()
        except Exception as e:
            logger.warning(f"post_train: sync_context failed: {e}")

        # Auto-commit
        try:
            from scripts.commit_run import commit_run
            commit_run(sig)
        except Exception as e:
            logger.warning(f"post_train: commit_run failed: {e}")
    else:
        # REGRESSION or FAILURE — write failure.json
        try:
            fail_path = EXPERIMENTS_DIR / meta.run_id / "failure.json"
            _atomic_write_json(fail_path, json.loads(result.model_dump_json()))
        except Exception as e:
            logger.warning(f"post_train: failure.json write failed: {e}")

        if result.status == VRunStatus.FAILURE:
            # Quarantine checkpoint
            try:
                ckpt = meta.checkpoint_path
                if ckpt:
                    src = Path(ckpt)
                    if not src.is_absolute():
                        src = REPO_ROOT / src
                    dst = EXPERIMENTS_DIR / meta.run_id / "quarantine.pt"
                    if src.exists() and not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(src), str(dst))
            except Exception as e:
                logger.warning(f"post_train: quarantine failed: {e}")


def pre_agent(agent: str, run_id: str) -> None:
    """Hard-abort if context snapshot is stale."""
    try:
        context_path = Path(os.environ.get(
            "AIVC_CONTEXT_PATH",
            str(Path("~/Documents/Cowork/AshKhan/Projects/aivc_genelink/context.md").expanduser()),
        ))
        fresh = verify_snapshot_fresh(DEFAULT_SNAPSHOT_PATH, context_path, 24)
    except Exception as e:
        logger.warning(f"pre_agent: freshness check error: {e}")
        fresh = False

    if not fresh:
        out_dir = REPO_ROOT / "artifacts" / "failure_notes"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(
                out_dir / f"stale_context_{run_id}.json",
                {
                    "run_id": run_id,
                    "agent": agent,
                    "reason": "stale_context",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception as e:
            logger.warning(f"pre_agent: could not write stale note: {e}")
        logger.error(f"pre_agent: stale_context for run_id={run_id}, aborting.")
        sys.exit(1)


def post_agent(agent: str, run_id: str, result) -> None:
    """Validate agent proposal, write approved/rejected note, exit 1 if rejected."""
    if agent not in ("training_agent", "research_agent"):
        return
    try:
        proposal = None
        extra = getattr(result, "extra", None)
        if isinstance(extra, dict):
            proposal = extra.get("proposal")
    except Exception as e:
        logger.warning(f"post_agent: extra read failed: {e}")
        return
    if not proposal:
        return

    try:
        pv = validate_proposal(proposal)
    except Exception as e:
        logger.error(f"post_agent: validate_proposal failed: {e}")
        return

    queue_dir = REPO_ROOT / "artifacts" / "agent_queue"
    try:
        queue_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning(f"post_agent: mkdir queue: {e}")

    if not pv.approved:
        try:
            _atomic_write_json(
                queue_dir / f"rejected_{run_id}_{pv.proposal_id}.json",
                json.loads(pv.model_dump_json()),
            )
        except Exception as e:
            logger.warning(f"post_agent: write rejected: {e}")
        logger.warning(f"post_agent: proposal rejected ({pv.rejection_reason})")
        sys.exit(1)
    else:
        try:
            _atomic_write_json(
                queue_dir / f"approved_{run_id}_{pv.proposal_id}.json",
                json.loads(pv.model_dump_json()),
            )
        except Exception as e:
            logger.warning(f"post_agent: write approved: {e}")
