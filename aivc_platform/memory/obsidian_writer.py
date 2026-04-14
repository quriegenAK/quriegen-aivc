"""aivc_platform/memory/obsidian_writer.py — Obsidian note renderer.

Writes experiment / failure notes using Jinja2 templates under templates/.
Jinja2 is a hard dependency; ImportError surfaces at import time.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import jinja2
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "aivc_platform.memory requires Jinja2. "
        "Install with: pip install 'jinja2>=3.1,<4'"
    ) from e

from aivc_platform.memory.vault import ObsidianConfig
from aivc_platform.tracking.schemas import PostRunDecision, RunMetadata
from aivc_platform.tracking.wandb_config import WANDB_URL

logger = logging.getLogger("aivc.memory.obsidian_writer")

_TEMPLATES_DIR = Path(__file__).parent / "templates"

_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)


def _fmt_float(v) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def _fmt_pct(v) -> str:
    return "n/a" if v is None else f"{v:.2f}%"


def _fmt_dt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return str(v)


_ENV.filters["fmt_float"] = _fmt_float
_ENV.filters["fmt_pct"] = _fmt_pct
_ENV.filters["fmt_dt"] = _fmt_dt


# ── Failure classification (deterministic, no LLM) ──────────────────────

class _Cause:
    __slots__ = ("label", "rationale")

    def __init__(self, label: str, rationale: str) -> None:
        self.label = label
        self.rationale = rationale


_CAUSES_DELTA = [
    _Cause("w_scale_range too aggressive", "Sparsity collapse — Neumann cascade zeroed out propagation."),
    _Cause("neumann_k too small", "Insufficient propagation depth — signal does not reach target genes."),
    _Cause("pert_id broadcast bug", "Known v1.1 ceiling: single learned vector applied across all perturbations."),
    _Cause("data label swap", "Possible ctrl/stim mislabelling in training loader."),
]

_CAUSES_REGRESSION = [
    _Cause("GAT unfreeze leaked", "frozen_modules does not include 'genelink' — check meta.frozen_modules."),
    _Cause("checkpoint corruption", "Wrong run loaded or state_dict mismatch."),
    _Cause("LR too high on feature_expander", "Decoder overfit degrading Kang regression."),
]

_CAUSES_NORMAN_UNAVAIL = [
    _Cause("AnnData missing / wrong path", "data/norman2019.h5ad not on disk or path mis-configured."),
    _Cause("gene alignment failure", "3010-gene intersection failed — check gene_names.txt alignment."),
    _Cause("insufficient pseudo-bulk pairs", "Below MIN_CELLS_PER_PSEUDOBULK=30 for most perturbations."),
]

_CAUSES_DEFAULT = [
    _Cause("unclassified", "Automated triage could not rank causes — human review required."),
]


_ACTIONS_DELTA = [
    "Shrink w_scale_range upper bound (try [0.001, 0.05]).",
    "Increase neumann_k to 5 and re-run sweep.",
    "Audit perturbation_model.py:140 for pert_id broadcast — confirm v1.1 ceiling applies.",
    "Verify ctrl/stim label distribution in training loader.",
]

_ACTIONS_REGRESSION = [
    "Assert 'genelink' in meta.frozen_modules before retraining.",
    "Re-validate checkpoint hash against artifacts/registry/latest_checkpoint.json.",
    "Lower feature_expander LR by 10×; re-run Kang regression guard.",
]

_ACTIONS_NORMAN_UNAVAIL = [
    "Stage data/norman2019.h5ad into data/; confirm path in eval_runner.",
    "Re-run gene intersection against data/gene_names.txt; expect 3010.",
    "Check MIN_CELLS_PER_PSEUDOBULK=30 filter and n_dropped_groups.",
]

_ACTIONS_DEFAULT = [
    "Human triage: review failure_note_path JSON and W&B run.",
]


def _extract_failure_reason(suite) -> str:
    """Pick the most informative failure string from the eval suite."""
    if getattr(suite, "halt_reason", None):
        return str(suite.halt_reason)
    kang = getattr(suite, "kang", None)
    if kang is not None and getattr(kang, "failure_reason", None):
        return str(kang.failure_reason)
    norman = getattr(suite, "norman", None)
    if norman is not None and getattr(norman, "failure_reason", None):
        return str(norman.failure_reason)
    if norman is None:
        return "Norman eval not available"
    return "unknown"


def _classify_failure(suite, meta: RunMetadata):
    """Return (bucket, causes, actions) for a failure.

    Matches substring (case-insensitive) on the failure reason.
    """
    reason = _extract_failure_reason(suite).lower()

    if "regression guard" in reason or "regression" in reason:
        return "regression_guard", _CAUSES_REGRESSION, _ACTIONS_REGRESSION
    if "delta_nonzero" in reason or "ctrl memorisation" in reason or "ctrl_memorisation" in reason:
        return "delta_collapse", _CAUSES_DELTA, _ACTIONS_DELTA
    if "norman eval not available" in reason or "norman" in reason:
        return "norman_unavailable", _CAUSES_NORMAN_UNAVAIL, _ACTIONS_NORMAN_UNAVAIL

    # Fallback: infer from metric state
    if meta.delta_nonzero_pct <= 0.0:
        return "delta_collapse", _CAUSES_DELTA, _ACTIONS_DELTA

    return "unclassified", _CAUSES_DEFAULT, _ACTIONS_DEFAULT


_FRONTMATTER_TAGS_RE = re.compile(
    r"^---\s*\n(?P<body>.*?)\n---\s*\n",
    re.DOTALL,
)


def _find_related_failures(vault: Path, bucket: str, limit: int = 5) -> list[str]:
    """Scan vault/failures/ for prior failures matching the same bucket tag.

    Fail-soft: return [] on any error.
    Returns up to `limit` stem names sorted by mtime descending.
    """
    try:
        fdir = vault / "failures"
        if not fdir.exists():
            return []
        entries: list[tuple[float, str]] = []
        for p in fdir.glob("*.md"):
            try:
                text = p.read_text(encoding="utf-8")
            except OSError:
                continue
            m = _FRONTMATTER_TAGS_RE.match(text)
            if not m:
                continue
            body = m.group("body")
            if bucket in body:
                entries.append((p.stat().st_mtime, p.stem))
        entries.sort(reverse=True)
        return [stem for _, stem in entries[:limit]]
    except Exception as e:  # pragma: no cover
        logger.debug(f"_find_related_failures failed: {e}")
        return []


def _render(name: str, **ctx) -> str:
    return _ENV.get_template(name).render(**ctx)


def _write_or_print(path: Path, content: str, dry_run: bool) -> Path:
    if dry_run:
        print(f"---\n# DRY RUN: {path}\n---\n{content}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def write_experiment_note(
    meta: RunMetadata,
    suite,
    config: ObsidianConfig,
) -> Path:
    """Render and write an experiment note to vault/experiments/{run_id}.md."""
    vault = config.resolved_vault()
    out = vault / "experiments" / f"{meta.run_id}.md"
    content = _render(
        "experiment_note.md.j2",
        meta=meta,
        suite=suite,
        config=config,
        now=datetime.now(timezone.utc),
        wandb_url_default=WANDB_URL,
        anndata_hash="n/a (v1.2)",
        mlflow_run_id="n/a (file store, no registry)",
    )
    return _write_or_print(out, content, config.dry_run)


def write_failure_note(
    meta: RunMetadata,
    decision: PostRunDecision,
    suite,
    config: ObsidianConfig,
) -> Path:
    """Render and write a failure note to vault/failures/{run_id}.md."""
    vault = config.resolved_vault()
    out = vault / "failures" / f"{meta.run_id}.md"
    bucket, causes, actions = _classify_failure(suite, meta)
    related: list[str] = []
    if config.auto_link and not config.dry_run:
        related = _find_related_failures(vault, bucket)
    content = _render(
        "failure_note.md.j2",
        meta=meta,
        suite=suite,
        decision=decision,
        config=config,
        now=datetime.now(timezone.utc),
        wandb_url_default=WANDB_URL,
        failure_reason=_extract_failure_reason(suite),
        failure_bucket=bucket,
        suspected_causes=causes,
        next_actions=actions,
        related_failures=related,
        anndata_hash="n/a (v1.2)",
    )
    return _write_or_print(out, content, config.dry_run)
