"""scripts/sync_context.py — Atomic context.md → docs/context_snapshot.md sync.

Per AIVC hardening spec §3.3.
"""
from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("aivc.scripts.sync_context")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONTEXT_PATH = Path("~/Documents/Cowork/AshKhan/Projects/aivc_genelink/context.md").expanduser()
DEFAULT_SNAPSHOT_PATH = REPO_ROOT / "docs" / "context_snapshot.md"
DEFAULT_LOCK_PATH = REPO_ROOT / ".context.lock"

REQUIRED_SECTIONS = [
    "Project",
    "Mission",
    "Stack snapshot",
    "Architecture pattern",
    "Key files / entry points",
    "Active APIs",
    "Models / ML components",
    "Data sources",
    "Open questions",
    "Do not touch",
    "GitHub",
]


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[end + 5:]
    return text


def _parse_frontmatter(text: str) -> dict:
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}
    fm = text[4:end]
    out = {}
    for line in fm.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def validate_context_md(path) -> Tuple[bool, Optional[str]]:
    p = Path(path)
    if not p.exists():
        return False, f"context file missing: {p}"
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        return False, f"could not read: {e}"

    body = _strip_frontmatter(text)

    # Check required ### sections
    headers = set()
    for line in body.splitlines():
        if line.startswith("### "):
            headers.add(line[4:].split(":")[0].strip())

    for section in REQUIRED_SECTIONS:
        if section not in headers:
            return False, f"missing required section: ### {section}"

    # Last updated date (YYYY-MM-DD parseable)
    m = re.search(r"Last updated[:\s]+(\d{4}-\d{2}-\d{2})", body)
    if not m:
        return False, "missing or unparseable 'Last updated: YYYY-MM-DD' field"
    try:
        datetime.strptime(m.group(1), "%Y-%m-%d")
    except ValueError as e:
        return False, f"invalid Last updated date: {e}"

    return True, None


def sync_context(
    context_path=None,
    snapshot_path=None,
    lock_path=None,
) -> bool:
    cp = Path(context_path) if context_path else DEFAULT_CONTEXT_PATH
    sp = Path(snapshot_path) if snapshot_path else DEFAULT_SNAPSHOT_PATH
    lp = Path(lock_path) if lock_path else DEFAULT_LOCK_PATH

    sp.parent.mkdir(parents=True, exist_ok=True)
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.touch(exist_ok=True)

    deadline = time.monotonic() + 30.0
    with open(lp, "a+", encoding="utf-8") as lock_f:
        while True:
            try:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    logger.error(f"sync_context: lock timeout (30s) on {lp}")
                    return False
                time.sleep(0.1)
        try:
            ok, reason = validate_context_md(cp)
            if not ok:
                logger.error(f"sync_context: validate_context_md failed: {reason}")
                return False
            raw = cp.read_bytes()
            sha = _sha256_bytes(raw)
            ts = datetime.now(timezone.utc).isoformat()
            run_id = os.environ.get("AIVC_RUN_ID", "manual")
            body = raw.decode("utf-8")
            # Strip any existing frontmatter on the source before re-prefixing
            body = _strip_frontmatter(body)
            new_text = (
                f"---\n"
                f"snapshot_sha: {sha}\n"
                f"snapshot_ts: {ts}\n"
                f"synced_from_run: {run_id}\n"
                f"---\n"
                f"{body}"
            )
            fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=str(sp.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(new_text)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, sp)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
            return True
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def verify_snapshot_fresh(snapshot_path=None, context_path=None, max_age_hours: float = 24) -> bool:
    sp = Path(snapshot_path) if snapshot_path else DEFAULT_SNAPSHOT_PATH
    cp = Path(context_path) if context_path else DEFAULT_CONTEXT_PATH
    if not sp.exists():
        return False
    text = sp.read_text(encoding="utf-8")
    fm = _parse_frontmatter(text)
    snap_sha = fm.get("snapshot_sha")
    snap_ts_str = fm.get("snapshot_ts")
    if not snap_sha or not snap_ts_str:
        return False
    if cp.exists():
        cur_sha = _sha256_bytes(cp.read_bytes())
        if cur_sha != snap_sha:
            return False
    try:
        snap_ts = datetime.fromisoformat(snap_ts_str)
    except ValueError:
        return False
    if (datetime.now(timezone.utc) - snap_ts) > timedelta(hours=max_age_hours):
        return False
    return True


if __name__ == "__main__":
    ok = sync_context()
    print("sync_context:", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)
