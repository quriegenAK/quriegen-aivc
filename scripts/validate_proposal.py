"""scripts/validate_proposal.py — Safety gate for agent proposals.

Per AIVC hardening spec §3.5.
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

logger = logging.getLogger("aivc.scripts.validate_proposal")

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = REPO_ROOT / "experiments" / "index.jsonl"
CONSTRAINTS_YAML = REPO_ROOT / "configs" / "agent_constraints.yaml"


class ProposalValidation(BaseModel):
    approved: bool
    proposal_id: str
    rejection_reason: Optional[str] = None
    duplicate_of: Optional[str] = None
    boundary_violations: List[str] = []


def _load_constraints() -> dict:
    if not CONSTRAINTS_YAML.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(CONSTRAINTS_YAML.read_text(encoding="utf-8")) or {}
    except Exception as e:
        logger.warning(f"could not load constraints: {e}")
        return {}


def _canonical_json(d: dict) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def _config_diff_sha(config_diff: dict) -> str:
    return hashlib.sha256(_canonical_json(config_diff).encode("utf-8")).hexdigest()


def _read_last_n_index(n: int) -> list:
    if not INDEX_PATH.exists():
        return []
    lines = []
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    out = []
    for ln in lines[-n:]:
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def validate_proposal(proposal: dict) -> ProposalValidation:
    proposal_id = proposal.get("proposal_id") or f"prop_{uuid.uuid4().hex[:8]}"

    # 1. Schema check
    for field in ("hypothesis", "config_diff", "expected_metric_delta"):
        v = proposal.get(field)
        if v is None or (isinstance(v, (dict, str, list)) and len(v) == 0):
            return ProposalValidation(
                approved=False,
                proposal_id=proposal_id,
                rejection_reason=f"schema_missing:{field}",
            )

    config_diff = proposal["config_diff"]
    if not isinstance(config_diff, dict):
        return ProposalValidation(
            approved=False,
            proposal_id=proposal_id,
            rejection_reason="schema_invalid:config_diff_not_dict",
        )

    # 2. Duplicate check (last 50)
    target_sha = _config_diff_sha(config_diff)
    recent = _read_last_n_index(50)
    for past in recent:
        past_sha = past.get("config_diff_sha")
        if past_sha and past_sha == target_sha:
            return ProposalValidation(
                approved=False,
                proposal_id=proposal_id,
                rejection_reason="duplicate",
                duplicate_of=past.get("run_id"),
            )

    # 3. Boundary check
    constraints = _load_constraints()
    allowed = constraints.get("allowed_hyperparams", {})
    forbidden = set(constraints.get("forbidden", []) or [])
    violations: List[str] = []

    for key, val in config_diff.items():
        if key in forbidden:
            violations.append(f"forbidden:{key}")
            continue
        if key in allowed:
            bounds = allowed[key]
            try:
                lo = float(bounds.get("min"))
                hi = float(bounds.get("max"))
                fv = float(val)
                if fv < lo or fv > hi:
                    violations.append(f"out_of_bounds:{key}={val} (must be in [{lo}, {hi}])")
            except (TypeError, ValueError):
                violations.append(f"non_numeric:{key}={val}")

    # Forbidden semantics — explicit mutations
    if "frozen_modules" in config_diff:
        fm = config_diff["frozen_modules"]
        if isinstance(fm, list) and "genelink" not in fm:
            violations.append("forbidden:unfreeze_genelink")
    if "n_genes" in config_diff:
        violations.append("forbidden:change_n_genes")
    if "edge_index" in config_diff:
        violations.append("forbidden:change_edge_index")

    if violations:
        return ProposalValidation(
            approved=False,
            proposal_id=proposal_id,
            rejection_reason="boundary_violations",
            boundary_violations=violations,
        )

    # 4. Regression loop check (last 3)
    rl = constraints.get("regression_loop", {}) or {}
    window = int(rl.get("window", 3))
    last_runs = _read_last_n_index(window)
    if len(last_runs) >= window and all(r.get("status") == "REGRESSION" for r in last_runs):
        return ProposalValidation(
            approved=False,
            proposal_id=proposal_id,
            rejection_reason="regression_loop_detected",
        )

    return ProposalValidation(approved=True, proposal_id=proposal_id)
