"""scripts/commit_run.py — Auto-commit on SUCCESS to experiments/auto branch.

Per AIVC hardening spec §3.4. SUCCESS-only (D3). Author Ash Khan <a.khan@quriegen.com> (D1).
"""
from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger("aivc.scripts.commit_run")

REPO_ROOT = Path(__file__).resolve().parent.parent
AUTHOR_NAME = "Ash Khan"
AUTHOR_EMAIL = "a.khan@quriegen.com"
DEFAULT_BRANCH = "experiments/auto"

WHITELIST_PATTERNS = [
    "configs/",
    "context.md",
    "docs/context_snapshot.md",
    "experiments/index.jsonl",
    "experiments/best/signature.json",
    "scripts/",
    ".gitignore",
]
BLACKLIST_PATTERNS = [
    "*.pt", "*.pth", "*.ckpt",
    "*.h5ad", "*.h5",
    "wandb/", "mlruns/", "artifacts/",
]


def _git(*args, check=True, capture=True):
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT,
        capture_output=capture, text=True, check=check,
    )


def ensure_experiments_branch(branch: str = DEFAULT_BRANCH) -> None:
    try:
        r = _git("rev-parse", "--verify", branch, check=False)
        if r.returncode != 0:
            _git("checkout", "-b", branch)
        else:
            cur = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
            if cur != branch:
                _git("checkout", branch)
    except subprocess.CalledProcessError as e:
        logger.warning(f"ensure_experiments_branch: {e}")


def _staged_diff_only_timestamp_bump() -> bool:
    """True iff the only staged change is snapshot_ts in context_snapshot.md."""
    r = _git("diff", "--cached", "--name-only", check=False)
    if r.returncode != 0:
        return False
    files = [f for f in r.stdout.strip().splitlines() if f]
    if files != ["docs/context_snapshot.md"]:
        return False
    diff = _git("diff", "--cached", "--", "docs/context_snapshot.md", check=False)
    if diff.returncode != 0:
        return False
    # Look at +/- lines (excluding @@/diff headers)
    changed = []
    for line in diff.stdout.splitlines():
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+") or line.startswith("-"):
            changed.append(line)
    sha_changed = any("snapshot_sha" in l for l in changed)
    only_ts = all(("snapshot_ts" in l) for l in changed) and len(changed) > 0
    return only_ts and not sha_changed


def commit_run(sig, branch: str = DEFAULT_BRANCH) -> Optional[str]:
    """Commit signed run on SUCCESS. Returns commit sha or None if skipped."""
    if sig.status != "SUCCESS":
        logger.info(f"commit_run: status={sig.status}, skipping (SUCCESS-only).")
        return None

    try:
        ensure_experiments_branch(branch)
    except Exception as e:
        logger.warning(f"commit_run: ensure_branch failed: {e}")
        return None

    # Stage whitelist
    paths = list(WHITELIST_PATTERNS) + [f"experiments/{sig.run_id}/"]
    # Filter to existing paths to avoid `git add` noise on missing
    add_args = []
    for p in paths:
        full = REPO_ROOT / p.rstrip("/")
        if full.exists() or p.endswith("/"):
            add_args.append(p)
    if add_args:
        _git("add", "--", *add_args, check=False)

    # Reset blacklist
    for pat in BLACKLIST_PATTERNS:
        _git("reset", "HEAD", "--", pat, check=False)

    # Timestamp-only bump skip
    if _staged_diff_only_timestamp_bump():
        logger.info("commit_run: staged diff is timestamp-only bump, skipping.")
        _git("reset", "HEAD", check=False)
        return None

    # Anything staged?
    r = _git("diff", "--cached", "--name-only", check=False)
    if not r.stdout.strip():
        logger.info("commit_run: nothing staged.")
        return None

    pearson = sig.metrics.pearson_r
    best_delta_str = ""
    msg_summary = f"pearson_r={pearson:.4f}"
    msg = (
        f"run({sig.run_id}): {msg_summary}\n"
        f"\n"
        f"run_id: {sig.run_id}\n"
        f"pearson_r: {pearson:.4f}\n"
        f"delta_nonzero_pct: {sig.metrics.delta_nonzero_pct:.2f}\n"
        f"dataset: {sig.dataset.name} (sha256:{sig.dataset.sha256[:12]})\n"
        f"config: {sig.config.path}\n"
        f"status: {sig.status}\n"
        f"wandb: -\n"
    )

    try:
        r = _git(
            "-c", f"user.name={AUTHOR_NAME}",
            "-c", f"user.email={AUTHOR_EMAIL}",
            "commit", "-m", msg, check=False,
        )
        if r.returncode != 0:
            logger.warning(f"git commit failed: {r.stderr}")
            return None
        sha = _git("rev-parse", "HEAD").stdout.strip()
        return sha
    except Exception as e:
        logger.warning(f"commit_run failed: {e}")
        return None


if __name__ == "__main__":
    import sys
    from scripts.build_signature import load_signature
    import argparse as _ap
    _p = _ap.ArgumentParser(description="Manually commit a signed run to experiments/auto branch.")
    _p.add_argument("--run-id", required=True, help="run_id to commit")
    _args = _p.parse_args()
    _sig = load_signature(_args.run_id)
    if not _sig:
        print(f"No signature found for run_id={_args.run_id}")
        print(f"Expected: experiments/{_args.run_id}/signature.json")
        sys.exit(1)
    _sha = commit_run(_sig)
    if _sha:
        print(f"Committed: {_sha}")
    else:
        print("Skipped: timestamp-only bump or no staged changes.")
    sys.exit(0)
