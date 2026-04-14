"""agents/data_agent.py — deterministic h5ad + pairing-cert gate.

See docs/aivc_spec_agents_v1.md §3.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from agents.base_agent import AgentResult, AgentTask, BaseAgent

logger = logging.getLogger("aivc.agents.data_agent")


class DataReport(BaseModel):
    valid: bool
    n_genes: int
    n_cells: int
    cert_present: bool
    cert_path: Optional[str] = None
    blocked_reason: Optional[str] = None


# Verified Phase 2B column names.
REQUIRED_OBS_BY_DATASET: dict[str, set[str]] = {
    "kang2018": {"label", "replicate", "cell_type"},
    "kang2018_pbmc": {"label", "replicate", "cell_type"},
    "norman2019": {"perturbation", "nperts", "gemgroup"},
}


class DataAgent(BaseAgent):
    agent_name = "data_agent"
    EXPECTED_N_GENES = 3010

    def __init__(
        self,
        cert_dir: str = "data/pairing_certificates",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cert_dir = Path(cert_dir)

    def _required_obs(self, dataset: str) -> set[str]:
        return REQUIRED_OBS_BY_DATASET.get(dataset, set())

    def _find_cert(self, dataset: str) -> Optional[Path]:
        stem = dataset.split("_")[0]
        primary = self.cert_dir / f"{stem}.json"
        if primary.exists():
            return primary
        # Fallback glob (e.g., kang2018 → kang*.json; quriegen → quriegen_pending.json)
        matches = list(self.cert_dir.glob(f"*{stem}*.json"))
        return matches[0] if matches else None

    def run(self, task: AgentTask) -> AgentResult:
        payload = task.payload
        h5ad_path = payload.get("h5ad_path")
        dataset = payload.get("dataset", "")
        require_cert = payload.get("require_cert", True)
        run_id = task.run_id

        self.queue_dir.mkdir(parents=True, exist_ok=True)

        n_genes = 0
        n_cells = 0
        blocked_reason: Optional[str] = None
        cert_path_obj: Optional[Path] = None

        # 1. Load with backed="r" — never touch X.
        try:
            import anndata
            adata = anndata.read_h5ad(h5ad_path, backed="r")
            n_genes = int(adata.n_vars)
            n_cells = int(adata.n_obs)
            obs_cols = set(adata.obs.columns)
        except Exception as e:
            blocked_reason = f"failed to open h5ad: {e}"
            obs_cols = set()

        # 2. Gene count
        if blocked_reason is None and n_genes != self.EXPECTED_N_GENES:
            blocked_reason = (
                f"n_vars={n_genes} != EXPECTED_N_GENES={self.EXPECTED_N_GENES}"
            )

        # 3. Required obs columns
        if blocked_reason is None:
            required = self._required_obs(dataset)
            missing = required - obs_cols
            if missing:
                blocked_reason = f"missing required obs columns: {sorted(missing)}"

        # 4. Pairing certificate
        cert_present = False
        if blocked_reason is None and require_cert:
            cert_path_obj = self._find_cert(dataset)
            if cert_path_obj is None:
                blocked_reason = "pairing certificate missing"
            else:
                cert_present = True
                if cert_path_obj.name == "quriegen_pending.json" or "qurie" in dataset.lower():
                    logger.warning(
                        "Certificate quriegen_pending.json found but status is PENDING. "
                        "QuRIE-seq multimodal training is blocked until cert is finalised."
                    )
                    blocked_reason = "cert_pending"

        valid = blocked_reason is None
        report = DataReport(
            valid=valid,
            n_genes=n_genes,
            n_cells=n_cells,
            cert_present=cert_present,
            cert_path=str(cert_path_obj) if cert_path_obj else None,
            blocked_reason=blocked_reason,
        )

        ts = int(time.time())
        out_path = self.queue_dir / f"data_agent_{run_id}_{ts}.json"
        try:
            out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        except OSError as e:
            logger.warning(f"DataAgent: could not write report: {e}")

        summary = (
            f"DataAgent: dataset={dataset} n_genes={n_genes} n_cells={n_cells} "
            f"valid={valid}"
            + (f" blocked_reason={blocked_reason}" if blocked_reason else "")
        )

        return AgentResult(
            agent_name="data_agent",
            run_id=run_id,
            success=valid,
            output_path=str(out_path),
            summary=summary,
            error=None if valid else blocked_reason,
            extra={"blocked_reason": blocked_reason} if blocked_reason else {},
        )
