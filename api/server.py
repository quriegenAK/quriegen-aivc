"""
api/server.py — AIVC inference API.

FastAPI service exposing the trained perturbation response model
and SCM intervention engine as REST endpoints.

Endpoints:
  GET  /health           — liveness check
  GET  /model/info       — model version, benchmark, capabilities
  POST /predict          — perturbation response prediction
  POST /intervene        — SCM do(X) causal intervention
  GET  /model/top_edges  — top-N Neumann W matrix edges
  GET  /model/jakstat    — JAK-STAT pathway edge weights
"""
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

logger = logging.getLogger("aivc.api")

N_GENES = 3010
SEED = 42

# ─── Module-level state (loaded once at startup) ──────────────────
_state: Dict[str, Any] = {}

JAKSTAT_GENES = [
    "JAK1", "JAK2", "TYK2", "STAT1", "STAT2", "STAT3",
    "IRF9", "IRF1", "IFIT1", "IFIT3", "MX1", "MX2",
    "ISG15", "OAS1", "IFITM1", "IFITM3",
]


# ─── Lifespan ─────────────────────────────────────────────────────

def _load_gene_names() -> List[str]:
    path = "data/gene_names.txt"
    if os.path.exists(path):
        with open(path) as f:
            names = [line.strip() for line in f if line.strip()]
        if len(names) == N_GENES:
            logger.info(f"Gene names loaded from {path} ({len(names)} genes)")
            return names
        logger.warning(f"{path} has {len(names)} genes, expected {N_GENES}. Using fallback.")
    logger.info("Gene names: using fallback GENE_0..GENE_3009")
    return [f"GENE_{i}" for i in range(N_GENES)]


def _load_edge_index() -> torch.Tensor:
    import pandas as pd
    for path in ["data/edge_list_fixed.csv", "data/edge_list.csv"]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if "src_idx" in df.columns and "dst_idx" in df.columns:
                    result = torch.tensor(
                        df[["src_idx", "dst_idx"]].values.T, dtype=torch.long
                    )
                    logger.info(f"Edge index loaded from {path} ({result.shape[1]} edges)")
                    return result
                elif "gene_a" in df.columns:
                    # Need gene_to_idx mapping — defer to mock
                    continue
                else:
                    continue
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
    # Fallback: mock edges
    torch.manual_seed(SEED)
    edges = torch.randint(0, N_GENES, (2, 100))
    logger.info(f"Edge index: using mock ({edges.shape[1]} random edges)")
    return edges


def _load_registry_checkpoint() -> str:
    """Load checkpoint_path from registry. Returns '' on any error."""
    from pathlib import Path
    reg_path = Path("artifacts/registry/latest_checkpoint.json")
    if reg_path.exists():
        try:
            import json
            reg = json.loads(reg_path.read_text())
            ckpt = reg.get("checkpoint_path", "")
            if ckpt:
                logger.info(f"Registry checkpoint: {ckpt}")
            return ckpt
        except Exception as e:
            logger.warning(f"Registry load failed: {e}")
    return ""


def _load_model(edge_index: torch.Tensor) -> dict:
    """Load best model checkpoint or instantiate untrained demo model."""
    from perturbation_model import PerturbationPredictor, CellTypeEmbedding
    from aivc.skills.neumann_propagation import NeumannPropagation

    checkpoint_paths = [
        os.getenv("AIVC_CHECKPOINT", ""),
        _load_registry_checkpoint(),
        "models/v1.1/model_v11_best.pt",
        "models/v1.0/aivc_v1.0_best.pt",
        "model_week3_best.pt",
        "model_week3.pt",
    ]

    model = PerturbationPredictor(
        n_genes=N_GENES, num_perturbations=2, feature_dim=64,
        hidden1_dim=64, hidden2_dim=32, num_head1=3, num_head2=2,
        decoder_hidden=256,
    )
    model.cell_type_embedding = CellTypeEmbedding(
        num_cell_types=20, embedding_dim=model.feature_dim,
    )

    is_trained = False
    version = "demo-untrained"
    loaded_path = None

    for path in checkpoint_paths:
        if path and os.path.exists(path):
            try:
                model.load_state_dict(torch.load(path, map_location="cpu"))
                is_trained = True
                loaded_path = path
                if "v1.1" in path:
                    version = "v1.1"
                elif "v1.0" in path:
                    version = "v1.0"
                else:
                    version = "v1.0-week3"
                logger.info(f"Model loaded: {path} (version={version})")
                break
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

    if not is_trained:
        logger.info("No checkpoint found. Running in demo mode (untrained).")

    model.eval()

    # Neumann module
    neumann = getattr(model, "neumann", None)
    if neumann is None:
        neumann = NeumannPropagation(
            n_genes=N_GENES, edge_index=edge_index, K=3, lambda_l1=0.001,
        )

    return {
        "model": model,
        "neumann": neumann,
        "is_trained": is_trained,
        "version": version,
        "loaded_path": loaded_path,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup."""
    torch.manual_seed(SEED)

    gene_names = _load_gene_names()
    edge_index = _load_edge_index()
    model_info = _load_model(edge_index)

    # SCM engine — wrap the trained model (minus Neumann) as the response decoder.
    # model.decoder expects 64-dim GNN embeddings, not raw gene expression.
    # This wrapper runs the full pipeline (expander → pert_embedding → GAT → decoder)
    # to produce direct effects in gene space, using trained weights throughout.
    from aivc.skills.scm_engine import SCMEngine

    class _TrainedDirectEffectDecoder(nn.Module):
        """Wraps PerturbationPredictor pipeline (minus Neumann) for SCM engine."""
        def __init__(self, model, edge_index):
            super().__init__()
            self.model = model
            self.edge_index = edge_index

        def forward(self, x):
            # x: (batch, n_genes) — ctrl expression in gene space
            # Run per-sample through the model pipeline, skipping Neumann
            results = []
            pert_id = torch.tensor([1])  # stim
            for i in range(x.shape[0]):
                xi = x[i]  # (n_genes,)
                feat = self.model.feature_expander(xi)
                feat = self.model.pert_embedding(feat, pert_id)
                z = self.model.genelink(feat, self.edge_index)
                pred = self.model.decoder(z)
                results.append(pred.unsqueeze(0))
            return torch.cat(results, dim=0)

    direct_effect_decoder = _TrainedDirectEffectDecoder(
        model_info["model"], edge_index,
    )
    engine = SCMEngine(
        neumann=model_info["neumann"],
        response_decoder=direct_effect_decoder,
        gene_names=gene_names,
    )

    _state.update({
        "model": model_info["model"],
        "neumann": model_info["neumann"],
        "engine": engine,
        "gene_names": gene_names,
        "gene_to_idx": {g: i for i, g in enumerate(gene_names)},
        "edge_index": edge_index,
        "is_trained": model_info["is_trained"],
        "version": model_info["version"],
        "n_edges": edge_index.shape[1],
    })

    logger.info(
        f"AIVC API ready | model={model_info['version']} | "
        f"trained={model_info['is_trained']} | n_genes={N_GENES} | "
        f"n_edges={edge_index.shape[1]} | endpoints: /predict /intervene"
    )

    yield

    _state.clear()


# ─── App ──────────────────────────────────────────────────────────

app = FastAPI(
    title="AIVC — AI Virtual Cell API",
    description="Perturbation response prediction and causal intervention engine.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Middleware ───────────────────────────────────────────────────

@app.middleware("http")
async def add_headers(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{(time.time() - start) * 1000:.1f}ms"
    response.headers["X-Request-ID"] = request_id
    return response


# ─── Exception handlers ──────────────────────────────────────────

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "type": "validation_error"},
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    logger.error(f"RuntimeError: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": "model_error"},
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal error", "type": "unknown"},
    )


# ─── Request / Response schemas ───────────────────────────────────

class PredictRequest(BaseModel):
    ctrl_expression: List[float]
    perturbation_id: int = 1
    cell_type_id: int = 1
    return_top_k: int = 50
    include_jakstat: bool = True

    @field_validator("ctrl_expression")
    @classmethod
    def validate_expression_length(cls, v):
        if len(v) != N_GENES:
            raise ValueError(
                f"ctrl_expression must have exactly {N_GENES} values. Got {len(v)}."
            )
        if any(x < 0 for x in v):
            raise ValueError("ctrl_expression values must be non-negative.")
        return v

    @field_validator("return_top_k")
    @classmethod
    def validate_top_k(cls, v):
        if v < 1 or v > 500:
            raise ValueError("return_top_k must be between 1 and 500.")
        return v


class PredictResponse(BaseModel):
    model_version: str
    is_trained: bool
    perturbation_id: int
    cell_type_id: int
    predicted_expression: List[float]
    top_genes: List[Dict[str, Any]]
    jakstat_predictions: Optional[Dict[str, Any]] = None
    pearson_r_benchmark: float
    latency_ms: float
    warnings: List[str]


class InterveneRequest(BaseModel):
    intervention_type: str
    target_genes: List[str]
    ctrl_expression: List[float]
    fold_change: float = 5.0
    top_k: int = 20

    @field_validator("intervention_type")
    @classmethod
    def validate_type(cls, v):
        allowed = {"gene_ko", "pathway_block", "gene_oe"}
        if v not in allowed:
            raise ValueError(f"intervention_type must be one of {allowed}. Got '{v}'.")
        return v

    @field_validator("ctrl_expression")
    @classmethod
    def validate_expression(cls, v):
        if len(v) != N_GENES:
            raise ValueError(f"ctrl_expression must have {N_GENES} values. Got {len(v)}.")
        return v

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v):
        if v < 1 or v > 100:
            raise ValueError("top_k must be between 1 and 100.")
        return v


class InterveneResponse(BaseModel):
    model_version: str
    is_valid: bool
    intervention_type: str
    intervened_genes: List[str]
    top_affected_genes: List[Dict[str, Any]]
    causal_path: List[Dict[str, Any]]
    ifit1_delta: Optional[float] = None
    jakstat_deltas: Dict[str, float]
    w_density: float
    latency_ms: float
    warnings: List[str]


class ModelInfoResponse(BaseModel):
    model_version: str
    is_trained: bool
    n_genes: int
    n_edges: int
    benchmark_r: float
    jakstat_recovery: str
    what_is_live: List[str]
    what_is_not_live: List[str]
    endpoints: List[str]


# ─── Endpoints ────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": "model" in _state,
        "version": _state.get("version", "not loaded"),
    }


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    return ModelInfoResponse(
        model_version=_state.get("version", "unknown"),
        is_trained=_state.get("is_trained", False),
        n_genes=N_GENES,
        n_edges=_state.get("n_edges", 0),
        benchmark_r=0.873,
        jakstat_recovery="7/15 (v1.0 baseline)",
        what_is_live=[
            "RNA encoder (GeneLink GAT, r=0.873)",
            "OT cell pairing (9,616 pairs)",
            "3-critic validation gate",
            "Neumann cascade propagation (K=3)",
            "SCM do(X) intervention engine",
            "Ambient RNA decontamination",
        ],
        what_is_not_live=[
            "Protein encoder (awaiting QuRIE-seq data)",
            "Phospho encoder (awaiting QuRIE-seq data)",
            "ATAC encoder (awaiting 10x Multiome data)",
            "Contrastive loss (pairing unconfirmed)",
            "Active learning loop (not yet closed)",
        ],
        endpoints=["/predict", "/intervene", "/model/top_edges", "/model/jakstat"],
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if "model" not in _state:
        raise HTTPException(503, "Model not loaded.")

    t0 = time.time()
    warnings = []

    model = _state["model"]
    edge_index = _state["edge_index"]
    gene_names = _state["gene_names"]

    ctrl = torch.tensor(req.ctrl_expression, dtype=torch.float32)
    pert_id = torch.tensor([req.perturbation_id])

    try:
        with torch.no_grad():
            from perturbation_model import CellTypeEmbedding
            ct_ids = CellTypeEmbedding.encode_cell_types(["CD14+ Monocytes"])
            ct_id = ct_ids[0:1]
            ct_id[0] = req.cell_type_id

            pred_delta = model.forward_batch(
                ctrl.unsqueeze(0), edge_index, pert_id, ct_id,
            )
            pred = (ctrl.unsqueeze(0) + pred_delta).clamp(min=0.0).squeeze(0)
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(500, f"Prediction error: {e}")

    pred_np = pred.numpy()
    ctrl_np = ctrl.numpy()

    # Top genes by |log2 fold change|
    import numpy as np
    eps = 1e-6
    log2_fc = np.log2(pred_np + eps) - np.log2(ctrl_np + eps)
    abs_lfc = np.abs(log2_fc)
    top_idx = np.argsort(abs_lfc)[::-1][:req.return_top_k]

    top_genes = []
    for rank, idx in enumerate(top_idx):
        top_genes.append({
            "rank": rank + 1,
            "gene_name": gene_names[idx],
            "gene_idx": int(idx),
            "pred_expr": round(float(pred_np[idx]), 4),
            "ctrl_expr": round(float(ctrl_np[idx]), 4),
            "log2_fc": round(float(log2_fc[idx]), 4),
            "direction": "up" if log2_fc[idx] > 0 else "down",
        })

    # JAK-STAT predictions
    jakstat_predictions = None
    if req.include_jakstat:
        jakstat_predictions = {}
        gene_to_idx = _state["gene_to_idx"]
        for g in JAKSTAT_GENES:
            if g in gene_to_idx:
                idx = gene_to_idx[g]
                c = float(ctrl_np[idx])
                p = float(pred_np[idx])
                jakstat_predictions[g] = {
                    "ctrl": round(c, 4),
                    "pred": round(p, 4),
                    "pred_fc": round((p + eps) / (c + eps), 2),
                }

    if not _state.get("is_trained"):
        warnings.append("Model is untrained (demo mode). Predictions are random.")

    latency = (time.time() - t0) * 1000

    return PredictResponse(
        model_version=_state.get("version", "unknown"),
        is_trained=_state.get("is_trained", False),
        perturbation_id=req.perturbation_id,
        cell_type_id=req.cell_type_id,
        predicted_expression=[round(float(v), 6) for v in pred_np],
        top_genes=top_genes,
        jakstat_predictions=jakstat_predictions,
        pearson_r_benchmark=0.873,
        latency_ms=round(latency, 1),
        warnings=warnings,
    )


@app.post("/intervene", response_model=InterveneResponse)
async def intervene(req: InterveneRequest):
    if "engine" not in _state:
        raise HTTPException(503, "SCM engine not loaded.")

    t0 = time.time()
    engine = _state["engine"]
    gene_to_idx = _state["gene_to_idx"]
    ctrl = torch.tensor(req.ctrl_expression, dtype=torch.float32)

    if req.intervention_type == "gene_ko":
        result = engine.do_gene_ko(req.target_genes[0], ctrl, top_k=req.top_k)
    elif req.intervention_type == "pathway_block":
        result = engine.do_pathway_block(req.target_genes, ctrl, top_k=req.top_k)
    elif req.intervention_type == "gene_oe":
        result = engine.do_gene_oe(
            req.target_genes[0], req.fold_change, ctrl, top_k=req.top_k,
        )
    else:
        raise ValueError(f"Unknown intervention_type: {req.intervention_type}")

    # Extract IFIT1 delta
    ifit1_delta = None
    if "IFIT1" in gene_to_idx:
        ifit1_delta = round(float(result.delta[gene_to_idx["IFIT1"]]), 4)

    # JAK-STAT deltas
    jakstat_deltas = {}
    for g in JAKSTAT_GENES:
        if g in gene_to_idx:
            jakstat_deltas[g] = round(float(result.delta[gene_to_idx[g]]), 4)

    latency = (time.time() - t0) * 1000

    return InterveneResponse(
        model_version=_state.get("version", "unknown"),
        is_valid=result.is_valid,
        intervention_type=result.intervention_type,
        intervened_genes=result.intervened_genes,
        top_affected_genes=result.top_affected_genes,
        causal_path=result.causal_path,
        ifit1_delta=ifit1_delta,
        jakstat_deltas=jakstat_deltas,
        w_density=result.w_density_at_inference,
        latency_ms=round(latency, 1),
        warnings=result.warnings,
    )


@app.get("/model/top_edges")
async def top_edges(n: int = Query(default=20, ge=1, le=100)):
    neumann = _state.get("neumann")
    if neumann is None:
        raise HTTPException(503, "Neumann module not loaded.")

    gene_names = _state.get("gene_names")
    edges = neumann.get_top_edges(n=n, gene_names=gene_names)

    jakstat_set = set(JAKSTAT_GENES)
    for e in edges:
        e["is_jakstat"] = (
            e.get("src_name", "") in jakstat_set
            and e.get("dst_name", "") in jakstat_set
        )

    return {"edges": edges, "n": len(edges)}


@app.get("/model/jakstat")
async def jakstat_weights():
    neumann = _state.get("neumann")
    if neumann is None:
        raise HTTPException(503, "Neumann module not loaded.")

    gene_to_idx = _state.get("gene_to_idx", {})
    top_edges = neumann.get_top_edges(n=20, gene_names=_state.get("gene_names"))
    top_set = {(e.get("src_name"), e.get("dst_name")) for e in top_edges}

    known_pairs = [
        ("JAK1", "STAT1"), ("JAK1", "STAT2"), ("JAK2", "STAT1"),
        ("STAT1", "IFIT1"), ("STAT1", "MX1"), ("STAT2", "IFIT1"),
        ("IRF9", "IFIT1"), ("STAT1", "ISG15"),
    ]

    edges = []
    for src, dst in known_pairs:
        src_idx = gene_to_idx.get(src)
        dst_idx = gene_to_idx.get(dst)
        weight = 0.0
        if src_idx is not None and dst_idx is not None:
            with torch.no_grad():
                mask = ((neumann.edge_src == src_idx) &
                        (neumann.edge_dst == dst_idx))
                if mask.any():
                    weight = float(neumann.W[mask][0].item())

        edges.append({
            "src": src, "dst": dst,
            "weight": round(weight, 6),
            "in_top_20": (src, dst) in top_set,
        })

    n_in_top20 = sum(1 for e in edges if e["in_top_20"])
    density = neumann.get_effective_W_density()

    return {
        "edges": edges,
        "jakstat_recovery_top20": n_in_top20,
        "w_density": round(density, 4),
    }
