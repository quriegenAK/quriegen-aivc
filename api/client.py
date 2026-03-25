"""
api/client.py — AIVC Python client SDK.

Usage:
    from api.client import AIVCClient
    client = AIVCClient("http://localhost:8000")
    result = client.predict(ctrl_expression=my_ctrl_array)
    print(result["top_genes"][:5])
"""
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PredictResult:
    model_version: str
    is_trained: bool
    predicted_expression: list
    top_genes: list
    jakstat_predictions: Optional[dict]
    pearson_r_benchmark: float
    latency_ms: float
    warnings: list


@dataclass
class InterveneResult:
    is_valid: bool
    intervention_type: str
    intervened_genes: list
    top_affected_genes: list
    ifit1_delta: Optional[float]
    jakstat_deltas: dict
    w_density: float
    latency_ms: float
    warnings: list


class AIVCClient:
    """
    Python client for the AIVC inference API.

    Args:
        base_url: API server URL. Default: "http://localhost:8000"
        timeout:  Request timeout in seconds. Default: 30.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = None
        try:
            import requests
            self._session = requests.Session()
            self._session.headers["Content-Type"] = "application/json"
            self._session.headers["X-Client"] = "aivc-python-sdk/0.1"
        except ImportError:
            pass

    def _post(self, path: str, data: dict) -> dict:
        url = f"{self.base_url}{path}"
        if self._session is not None:
            r = self._session.post(url, json=data, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        # urllib fallback
        import urllib.request
        req = urllib.request.Request(
            url, data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def _get(self, path: str, params: dict = None) -> dict:
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
        if self._session is not None:
            r = self._session.get(url, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        import urllib.request
        with urllib.request.urlopen(url, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def health(self) -> dict:
        return self._get("/health")

    def model_info(self) -> dict:
        return self._get("/model/info")

    def predict(
        self,
        ctrl_expression,
        perturbation_id: int = 1,
        cell_type_id: int = 1,
        return_top_k: int = 50,
        include_jakstat: bool = True,
    ) -> PredictResult:
        if hasattr(ctrl_expression, "tolist"):
            ctrl_expression = ctrl_expression.tolist()
        resp = self._post("/predict", {
            "ctrl_expression": ctrl_expression,
            "perturbation_id": perturbation_id,
            "cell_type_id": cell_type_id,
            "return_top_k": return_top_k,
            "include_jakstat": include_jakstat,
        })
        return PredictResult(
            model_version=resp["model_version"],
            is_trained=resp["is_trained"],
            predicted_expression=resp["predicted_expression"],
            top_genes=resp["top_genes"],
            jakstat_predictions=resp.get("jakstat_predictions"),
            pearson_r_benchmark=resp["pearson_r_benchmark"],
            latency_ms=resp["latency_ms"],
            warnings=resp["warnings"],
        )

    def intervene_ko(self, gene: str, ctrl_expression, top_k: int = 20) -> InterveneResult:
        if hasattr(ctrl_expression, "tolist"):
            ctrl_expression = ctrl_expression.tolist()
        resp = self._post("/intervene", {
            "intervention_type": "gene_ko",
            "target_genes": [gene],
            "ctrl_expression": ctrl_expression,
            "top_k": top_k,
        })
        return self._parse_intervene(resp)

    def intervene_pathway_block(self, genes: list, ctrl_expression, top_k: int = 20) -> InterveneResult:
        if hasattr(ctrl_expression, "tolist"):
            ctrl_expression = ctrl_expression.tolist()
        resp = self._post("/intervene", {
            "intervention_type": "pathway_block",
            "target_genes": genes,
            "ctrl_expression": ctrl_expression,
            "top_k": top_k,
        })
        return self._parse_intervene(resp)

    def intervene_oe(self, gene: str, fold_change: float, ctrl_expression) -> InterveneResult:
        if hasattr(ctrl_expression, "tolist"):
            ctrl_expression = ctrl_expression.tolist()
        resp = self._post("/intervene", {
            "intervention_type": "gene_oe",
            "target_genes": [gene],
            "ctrl_expression": ctrl_expression,
            "fold_change": fold_change,
        })
        return self._parse_intervene(resp)

    def _parse_intervene(self, resp: dict) -> InterveneResult:
        return InterveneResult(
            is_valid=resp["is_valid"],
            intervention_type=resp["intervention_type"],
            intervened_genes=resp["intervened_genes"],
            top_affected_genes=resp["top_affected_genes"],
            ifit1_delta=resp.get("ifit1_delta"),
            jakstat_deltas=resp.get("jakstat_deltas", {}),
            w_density=resp["w_density"],
            latency_ms=resp["latency_ms"],
            warnings=resp["warnings"],
        )

    def top_edges(self, n: int = 20) -> list:
        resp = self._get("/model/top_edges", {"n": n})
        return resp.get("edges", [])

    def jakstat_weights(self) -> dict:
        return self._get("/model/jakstat")
