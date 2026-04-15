"""scripts/build_signature.py — Sign a training run.

Per AIVC hardening spec §3.1. Atomic write of experiments/{run_id}/signature.json
plus flock-protected append to experiments/index.jsonl.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from aivc_platform.tracking.schemas import RunMetadata
    from eval.eval_runner import EvalSuite

logger = logging.getLogger("aivc.scripts.build_signature")

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
INDEX_PATH = EXPERIMENTS_DIR / "index.jsonl"
CONTEXT_SNAPSHOT = REPO_ROOT / "docs" / "context_snapshot.md"

DATASET_PATH_MAP = {
    "kang2018": "data/kang2018_pbmc_fixed.h5ad",
    "kang2018_pbmc": "data/kang2018_pbmc_fixed.h5ad",
    "norman2019": "data/norman2019.h5ad",
}


# ─── Schema ───────────────────────────────────────────────────────────

class GitInfo(BaseModel):
    commit: str
    branch: str
    dirty: bool
    diff_sha: Optional[str] = None


class ConfigInfo(BaseModel):
    path: str
    sha256: str
    snapshot_path: str


class DatasetInfo(BaseModel):
    name: str
    version: str
    sha256: str
    dvc_path: Optional[str] = None


class EnvironmentInfo(BaseModel):
    python: str
    cuda: Optional[str] = None
    torch: str
    pip_freeze_sha: str


class ModelInfo(BaseModel):
    checkpoint_path: str
    checkpoint_sha256: str
    architecture: Literal["GeneLink-GAT-Neumann-v1.1"] = "GeneLink-GAT-Neumann-v1.1"


class MetricsInfo(BaseModel):
    pearson_r: float
    delta_nonzero_pct: float
    ctrl_memorisation_score: float


class RunSignature(BaseModel):
    schema_version: Literal["aivc/run_signature/v1"] = "aivc/run_signature/v1"
    run_id: str
    timestamp_utc: datetime
    git: GitInfo
    config: ConfigInfo
    dataset: DatasetInfo
    environment: EnvironmentInfo
    model: ModelInfo
    metrics: MetricsInfo
    context_snapshot_sha: str
    status: Literal["SUCCESS", "REGRESSION", "FAILURE"] = "FAILURE"
    config_diff_sha: Optional[str] = None


# ─── Helpers ──────────────────────────────────────────────────────────

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    out = subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, timeout=10
    )
    return out.decode("utf-8", errors="replace").strip()


def _git_info() -> GitInfo:
    try:
        commit = _git("rev-parse", "--short=7", "HEAD")
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        diff = subprocess.check_output(
            ["git", "diff", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, timeout=10
        )
        dirty = len(diff.strip()) > 0
        diff_sha = _sha256_bytes(diff) if dirty else None
        return GitInfo(commit=commit, branch=branch, dirty=dirty, diff_sha=diff_sha)
    except Exception as e:
        logger.warning(f"git info unavailable: {e}")
        return GitInfo(commit="unknown", branch="unknown", dirty=True, diff_sha=None)


def _pip_freeze_sha() -> str:
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.DEVNULL, timeout=5,
        )
        return _sha256_bytes(out)
    except Exception:
        return "timeout"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _safe_relative(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _resolve_dataset_path(name: str) -> Path:
    name_lower = name.lower()
    if "kang2018" in name_lower:
        rel = DATASET_PATH_MAP["kang2018_pbmc"]
    elif "norman" in name_lower:
        rel = DATASET_PATH_MAP["norman2019"]
    else:
        rel = f"data/{name}.h5ad"
    return REPO_ROOT / rel


def _dataset_info(name: str) -> DatasetInfo:
    p = _resolve_dataset_path(name)
    dvc_p = p.parent / (p.name + ".dvc")
    dvc_path = str(dvc_p.relative_to(REPO_ROOT)) if dvc_p.exists() else None
    if p.exists():
        sha = _sha256_file(p)
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
        except Exception:
            mtime = "unknown"
        return DatasetInfo(name=name, version=mtime, sha256=sha, dvc_path=dvc_path)
    return DatasetInfo(name=name, version="missing", sha256="unknown", dvc_path=dvc_path)


def _checkpoint_info(path: Optional[str]) -> ModelInfo:
    if not path:
        return ModelInfo(checkpoint_path="", checkpoint_sha256="missing")
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if p.exists():
        return ModelInfo(checkpoint_path=str(path), checkpoint_sha256=_sha256_file(p))
    return ModelInfo(checkpoint_path=str(path), checkpoint_sha256="missing")


def _context_snapshot_sha() -> str:
    if CONTEXT_SNAPSHOT.exists():
        return _sha256_file(CONTEXT_SNAPSHOT)
    return "missing"


def _config_info(run_id: str, config_path: Optional[str]) -> ConfigInfo:
    snapshot = EXPERIMENTS_DIR / run_id / "config.yaml"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if config_path:
        src = Path(config_path)
        if not src.is_absolute():
            src = REPO_ROOT / src
        if src.exists():
            data = src.read_bytes()
            sha = _sha256_bytes(data)
            shutil.copy2(src, snapshot)
            return ConfigInfo(path=str(config_path), sha256=sha, snapshot_path=_safe_relative(snapshot))
    # No config: write empty marker
    snapshot.write_text("", encoding="utf-8")
    return ConfigInfo(path=config_path or "", sha256="missing",
                      snapshot_path=_safe_relative(snapshot))


def _config_diff_sha(config_path: Optional[str]) -> Optional[str]:
    if not os.environ.get("AIVC_PROPOSAL_ID"):
        return None
    if not config_path:
        return None
    src = Path(config_path)
    if not src.is_absolute():
        src = REPO_ROOT / src
    if not src.exists():
        return None
    try:
        import yaml
        data = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return _sha256_bytes(canonical.encode("utf-8"))
    except Exception:
        return None


# ─── Public API ───────────────────────────────────────────────────────

def build_signature(meta: "RunMetadata", suite: "EvalSuite") -> RunSignature:
    """Assemble a RunSignature from finalised RunMetadata + EvalSuite."""
    import torch  # local import

    # Try to recover the original config path from env (set by cli.py train)
    config_path = os.environ.get("AIVC_CONFIG_PATH") or os.environ.get("AIVC_CONFIG")
    cfg = _config_info(meta.run_id, config_path)

    return RunSignature(
        run_id=meta.run_id,
        timestamp_utc=datetime.now(timezone.utc),
        git=_git_info(),
        config=cfg,
        dataset=_dataset_info(meta.dataset),
        environment=EnvironmentInfo(
            python=sys.version.split()[0],
            cuda=getattr(torch.version, "cuda", None),
            torch=torch.__version__,
            pip_freeze_sha=_pip_freeze_sha(),
        ),
        model=_checkpoint_info(meta.checkpoint_path),
        metrics=MetricsInfo(
            pearson_r=float(meta.pearson_r),
            delta_nonzero_pct=float(meta.delta_nonzero_pct),
            ctrl_memorisation_score=float(meta.ctrl_memorisation_score),
        ),
        context_snapshot_sha=_context_snapshot_sha(),
        status="FAILURE",  # caller overwrites after classify_run
        config_diff_sha=_config_diff_sha(config_path),
    )


def save_signature(sig: RunSignature) -> Path:
    """Atomically write signature.json and append one line to index.jsonl (flocked)."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = EXPERIMENTS_DIR / sig.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    sig_path = run_dir / "signature.json"
    _atomic_write_text(sig_path, sig.model_dump_json(indent=2))

    # Flocked append to index.jsonl
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.touch(exist_ok=True)
    line = sig.model_dump_json() + "\n"
    deadline = time.monotonic() + 30.0
    with open(INDEX_PATH, "a", encoding="utf-8") as f:
        while True:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"index.jsonl flock timeout (30s): {INDEX_PATH}")
                time.sleep(0.1)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return sig_path


def load_signature(run_id: str) -> Optional[RunSignature]:
    p = EXPERIMENTS_DIR / run_id / "signature.json"
    if not p.exists():
        return None
    try:
        return RunSignature.model_validate_json(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"load_signature {run_id} failed: {e}")
        return None


def get_best_signature() -> Optional[RunSignature]:
    """Stream index.jsonl, return the SUCCESS entry with max pearson_r."""
    if not INDEX_PATH.exists():
        return None
    best: Optional[RunSignature] = None
    best_r = -float("inf")
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sig = RunSignature.model_validate_json(line)
            except Exception:
                continue
            if sig.status != "SUCCESS":
                continue
            if sig.metrics.pearson_r > best_r:
                best_r = sig.metrics.pearson_r
                best = sig
    return best


# ─── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    sig = load_signature(args.run_id)
    if sig is None:
        print(f"No signature for {args.run_id}", file=sys.stderr)
        sys.exit(1)
    print(sig.model_dump_json(indent=2))
