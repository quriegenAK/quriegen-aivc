"""aivc_platform/memory/context_updater.py — safe edits to the user's context.md.

Updatable fields: last_updated, best_pearson_r (only if higher),
latest_checkpoint, last_wandb_run, last_run_date.

Forbidden sections (header-to-next-header spans) — never modified:
  Mission, Stack snapshot, Open questions, Do not touch, GitHub.

The real context.md uses ### headers (verified against fixture). Regex
matches both ## and ### for resilience.

Atomic writes via .tmp → .replace(). Missing fields are appended under
`## Architecture decisions` (created at EOF if absent).
"""
from __future__ import annotations

import difflib
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from aivc_platform.tracking.schemas import RunMetadata
from aivc_platform.tracking.wandb_config import WANDB_URL

if TYPE_CHECKING:
    from eval.eval_runner import EvalSuite

logger = logging.getLogger("aivc.memory.context_updater")

DEFAULT_CONTEXT_PATH = "~/Documents/Cowork/AshKhan/Projects/aivc_genelink/context.md"

ARCHITECTURE_SECTION = "## Architecture decisions"

# ── Patterns ────────────────────────────────────────────────────────────
# Validated against docs/context_snapshot.md:
#   Line 3: "Last updated: 2026-04-10"  → last_updated matches
# Other fields not present in fixture → will be appended under ## Architecture decisions.

_PAT_LAST_UPDATED = re.compile(
    r"^(?P<prefix>[-*]?\s*\*{0,2}Last updated\*{0,2}\s*[:\-]\s*)(?P<val>.+?)[ \t]*$",
    re.MULTILINE | re.IGNORECASE,
)
_PAT_BEST_PEARSON = re.compile(
    r"^(?P<prefix>[-*]?\s*\*{0,2}best pearson[_ ]?r\*{0,2}\s*[:\-]\s*)(?P<val>[0-9.]+)[ \t]*$",
    re.MULTILINE | re.IGNORECASE,
)
_PAT_LATEST_CHECKPOINT = re.compile(
    r"^(?P<prefix>[-*]?\s*\*{0,2}latest checkpoint\*{0,2}\s*[:\-]\s*)(?P<val>`?[^`\n]+`?)[ \t]*$",
    re.MULTILINE | re.IGNORECASE,
)
_PAT_LAST_WANDB = re.compile(
    r"^(?P<prefix>[-*]?\s*\*{0,2}last W&?B( run)?\*{0,2}\s*[:\-]\s*)(?P<val>\S.*?)[ \t]*$",
    re.MULTILINE | re.IGNORECASE,
)
_PAT_LAST_RUN_DATE = re.compile(
    r"^(?P<prefix>[-*]?\s*\*{0,2}last run date\*{0,2}\s*[:\-]\s*)(?P<val>.+?)[ \t]*$",
    re.MULTILINE | re.IGNORECASE,
)

# NB: real file uses ### headers; accept 2 or 3 hashes.
_SECTION_RE = re.compile(
    r"^(?P<hashes>#{2,3})\s+(?P<name>.+?)\s*$(?P<body>.*?)(?=^#{2,3}\s|\Z)",
    re.MULTILINE | re.DOTALL,
)

FORBIDDEN_SECTIONS = {
    "Open questions",
    "Do not touch",
    "GitHub",
    "Stack snapshot",
    "Mission",
}


def _forbidden_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for m in _SECTION_RE.finditer(text):
        if m.group("name").strip() in FORBIDDEN_SECTIONS:
            spans.append((m.start(), m.end()))
    return spans


def _in_forbidden(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(s <= pos < e for s, e in spans)


def _safe_sub(text: str, pattern: re.Pattern[str], new_value: str,
              spans: list[tuple[int, int]]) -> tuple[str, int]:
    """Replace matches outside forbidden spans. Returns (new_text, n_replaced)."""
    count = 0

    def _repl(match: re.Match[str]) -> str:
        nonlocal count
        if _in_forbidden(match.start(), spans):
            return match.group(0)
        count += 1
        return f"{match.group('prefix')}{new_value}"

    return pattern.sub(_repl, text), count


def _current_best_pearson(text: str, spans: list[tuple[int, int]]) -> Optional[float]:
    for m in _PAT_BEST_PEARSON.finditer(text):
        if _in_forbidden(m.start(), spans):
            continue
        try:
            return float(m.group("val"))
        except ValueError:
            continue
    return None


def _ensure_architecture_section(text: str, lines_to_add: list[str]) -> str:
    """Append lines under ## Architecture decisions, creating the section if absent."""
    if not lines_to_add:
        return text
    # Search for existing ## or ### Architecture decisions header.
    arch_re = re.compile(
        r"^(#{2,3})\s+Architecture decisions[ \t]*$", re.MULTILINE | re.IGNORECASE
    )
    m = arch_re.search(text)
    addition = "\n".join(lines_to_add) + "\n"
    if m:
        # Insert after the header line.
        insert_at = m.end()
        # Ensure a preceding newline.
        prefix = "" if text[insert_at:insert_at + 1] == "\n" else "\n"
        return text[:insert_at] + prefix + addition + text[insert_at:]
    # Create section at EOF.
    sep = "" if text.endswith("\n") else "\n"
    return f"{text}{sep}\n{ARCHITECTURE_SECTION}\n\n{addition}"


def _fmt_now_date() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _build_values(meta: RunMetadata, suite: "EvalSuite | None") -> dict[str, str]:
    run_date = meta.finished_at or meta.started_at
    last_run_date = run_date.strftime("%Y-%m-%d") if run_date else _fmt_now_date()
    return {
        "last_updated": _fmt_now_date(),
        "best_pearson_r": f"{meta.pearson_r:.4f}",
        "latest_checkpoint": f"`{meta.checkpoint_path}`" if meta.checkpoint_path else "n/a",
        "last_wandb_run": meta.wandb_url or WANDB_URL,
        "last_run_date": last_run_date,
    }


_FIELD_LABELS = {
    "last_updated": "Last updated",
    "best_pearson_r": "best pearson_r",
    "latest_checkpoint": "latest checkpoint",
    "last_wandb_run": "last W&B run",
    "last_run_date": "last run date",
}

_PATTERNS = {
    "last_updated": _PAT_LAST_UPDATED,
    "best_pearson_r": _PAT_BEST_PEARSON,
    "latest_checkpoint": _PAT_LATEST_CHECKPOINT,
    "last_wandb_run": _PAT_LAST_WANDB,
    "last_run_date": _PAT_LAST_RUN_DATE,
}


def update_context(
    meta: RunMetadata,
    suite: "EvalSuite | None" = None,
    context_path: str = DEFAULT_CONTEXT_PATH,
    *,
    dry_run: bool = False,
) -> None:
    """Patch the user's context.md with latest run state.

    Raises FileNotFoundError if context_path does not exist.
    best_pearson_r is updated only if meta.pearson_r exceeds the current value.
    """
    path = Path(context_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"context.md not found at {path}")

    original = path.read_text(encoding="utf-8")
    text = original
    spans = _forbidden_spans(text)
    values = _build_values(meta, suite)

    # Conditional best_r: only bump if strictly higher than current.
    current_best = _current_best_pearson(text, spans)
    if current_best is not None and meta.pearson_r <= current_best:
        values.pop("best_pearson_r")

    missing: list[str] = []
    for field, pattern in _PATTERNS.items():
        if field not in values:
            continue
        text, n = _safe_sub(text, pattern, values[field], spans)
        if n == 0:
            missing.append(field)
        # Recompute spans when the text shifts. This is safe because
        # forbidden sections are header-anchored, and we only substitute
        # within already-safe regions — sizes may shift but region
        # boundaries remain meaningful line-wise.
        spans = _forbidden_spans(text)

    if missing:
        lines = [f"- **{_FIELD_LABELS[f]}**: {values[f]}" for f in missing]
        text = _ensure_architecture_section(text, lines)

    if text == original:
        return

    if dry_run:
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            text.splitlines(keepends=True),
            fromfile="before",
            tofile="after",
        )
        import sys
        sys.stdout.writelines(diff)
        return

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
