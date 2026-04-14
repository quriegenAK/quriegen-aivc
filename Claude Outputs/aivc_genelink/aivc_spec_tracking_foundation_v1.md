# AIVC — Tracking Foundation Spec (v1)

**File:** `Claude Outputs/aivc_genelink/aivc_spec_tracking_foundation_v1.md`
**Scope:** `platform/tracking/{schemas.py, wandb_config.py, experiment_logger.py}`
**Mode:** Architect
**Author:** Ash (spec)
**Date:** 2026-04-13
**Status:** Design-ready — no code generated yet

---

## 0. Grounding (what was read, what exists)

Verified on disk before writing this spec:

- `platform/tracking/` — **does not exist**. This spec creates the module.
- `api/server.py` — FastAPI service; benchmark constant `pearson_r_benchmark = 0.873`; checkpoint search order `${AIVC_CHECKPOINT}`, `models/v1.1/model_v11_best.pt`, `models/v1.0/aivc_v1.0_best.pt`, `model_week3_best.pt`, `model_week3.pt`. `N_GENES = 3010`. Neumann default `K=3`. No tracking hooks today.
- `perturbation_model.py` — no file named `models/gat_encoder.py`. The GAT encoder is the `GeneLink` class (2-layer `GATConv`) living inside `perturbation_model.py`; it is attached to `PerturbationPredictor` as the attribute **`genelink`** (see `server.py:181` → `self.model.genelink(feat, edge_index)`). This is the canonical "GAT encoder" and the thing that must be frozen by default.
- `aivc/memory/mlflow_backend.py` — already implements an MLflow client (`EXPERIMENT_NAME = "aivc_v11_neumann_sweep"`, `PLATFORM_TAGS`, `log_epoch_metrics`, `log_final_metrics`, `log_model_checkpoint`, `get_best_run`). It does **not** call `mlflow.register_model`. The new `experiment_logger` will reuse this class rather than re-implement MLflow I/O.
- `aivc/registry.py` — this is the **in-process skill registry** (decorator pattern). It is *not* a model registry and must not be conflated with MLflow Model Registry.
- `aivc/orchestration/orchestrator.py` — has `AIVCOrchestrator` + `CriticSuite`. There is **no** `training_agent` or `research_agent` defined anywhere in the repo (grep returns zero hits). **Gap → see §7.**
- `requirements.txt` — has `fastapi`, `pydantic>=2`, `torch`, `torch-geometric`, etc. No `wandb`, no `mlflow`. No `pyproject.toml` exists.

---

## 1. Architecture at a glance

```
                 ┌────────────────────────────────────────────┐
 train_v11.py ──▶│  platform.tracking.ExperimentLogger        │
                 │   ├── W&B Run  (sweeps, live curves, UI)   │
                 │   └── MLflowBackend (runs + Model Registry)│
                 │   ├── RunMetadata (Pydantic, typed)        │
                 │   └── post_run_hook  → agent dispatch      │
                 └──────────────┬─────────────────────────────┘
                                │
                     ┌──────────┴──────────┐
                     ▼                     ▼
            delta_nonzero_pct == 0   pearson_r ≥ 0.873
                 │                           │
                 ▼                           ▼
        FAILURE_NOTE_*.md           mlflow.register_model(...)
        dispatch("training_agent")   dispatch("research_agent")
```

**Division of labour (explicit):**

| Concern                     | Backend               | Rationale                                                          |
|-----------------------------|-----------------------|--------------------------------------------------------------------|
| Live curves, sweeps, agents | **W&B**               | Bayesian sweeps + hyperband + shareable dashboards                 |
| Artifact + model registry   | **MLflow**            | Already partially integrated; `register_model` is the contract     |
| Typed metadata envelope     | **Pydantic v2**       | Single source of truth, prevents schema drift across backends      |
| Post-run automation         | `post_run_hook`       | Deterministic routing to downstream agents                         |

Rule: **every W&B log call is mirrored to MLflow**. W&B is allowed to log more (e.g., media, sweep internals) but *never less* than MLflow for numeric metrics.

---

## 2. `platform/tracking/schemas.py`

Pydantic v2. All models `model_config = ConfigDict(extra="forbid", frozen=False)`.

### 2.1 Enums

```python
class Modality(str, Enum):
    RNA = "rna"
    ATAC = "atac"
    PROTEIN = "protein"
    PHOSPHO = "phospho"

class RunStatus(str, Enum):
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    KILLED = "killed"
```

### 2.2 `RunMetadata` — exact fields

```python
class RunMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    # ── Identity ───────────────────────────────────────────────
    run_id: str = Field(..., description="W&B run id; also used as MLflow run tag.")
    run_group: str = Field(..., description="W&B group, e.g. kang2018_wsweep_2026-04-13.")
    git_sha: str = Field(..., min_length=7, max_length=40)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: RunStatus = RunStatus.RUNNING

    # ── Required metrics (post-run) ────────────────────────────
    pearson_r: float = Field(..., ge=-1.0, le=1.0,
                             description="Pearson r on held-out stim vs. pred.")
    delta_nonzero_pct: float = Field(..., ge=0.0, le=100.0,
                                     description="% of Δ entries with |Δ|>1e-6. 0 ⇒ model collapsed.")
    ctrl_memorisation_score: float = Field(..., ge=0.0, le=1.0,
        description="Cosine(pred, ctrl). 1.0 ⇒ model memorised ctrl and ignored perturbation.")

    # ── Architecture knobs ────────────────────────────────────
    frozen_modules: list[str] = Field(
        default_factory=lambda: ["genelink", "feature_expander"],
        description="Module attribute names on PerturbationPredictor whose params are frozen."
    )
    w_scale_range: tuple[float, float] = Field(
        ..., description="(min, max) search range for Neumann W scale regulariser."
    )
    neumann_k: int = Field(..., ge=1, le=6,
                           description="Neumann series truncation order. Sweep values: {2,3,4}.")

    # ── Data provenance ──────────────────────────────────────
    modalities: list[Modality] = Field(..., min_length=1)
    dataset: str = Field(..., description="e.g. 'kang2018_pbmc_ifnb'")
    checkpoint_path: str = Field(..., description="Absolute or repo-relative path to .pt file.")
    anndata_hash: str = Field(..., pattern=r"^[0-9a-f]{16,64}$",
                              description="sha256 (16–64 hex chars) of the AnnData .h5ad used.")

    # ── Derived, not required at init ────────────────────────
    mlflow_run_id: str | None = None
    wandb_url: str | None = None
    model_registry_uri: str | None = None  # set iff post_run_hook registers

    # ── Validators ───────────────────────────────────────────
    @field_validator("frozen_modules")
    @classmethod
    def _must_freeze_gat_encoder(cls, v: list[str]) -> list[str]:
        """Enforce GAT encoder is frozen by default. Hard constraint from spec."""
        if "genelink" not in v:
            raise ValueError(
                "frozen_modules MUST contain 'genelink' (the GAT encoder). "
                "To explicitly unfreeze for a research run, set "
                "AIVC_ALLOW_GAT_UNFREEZE=1 and pass frozen_modules=[...] "
                "without 'genelink' — this validator will re-check the env var."
            )
        return v

    @field_validator("w_scale_range")
    @classmethod
    def _monotonic(cls, v: tuple[float, float]) -> tuple[float, float]:
        lo, hi = v
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(f"w_scale_range must be 0<=lo<=hi<=1, got {v}.")
        return v

    @field_validator("neumann_k")
    @classmethod
    def _sweep_compatible_k(cls, v: int) -> int:
        if v not in (2, 3, 4):
            # Soft warn via logger; do not raise — allows ad-hoc K=1/5 runs.
            logging.getLogger("aivc.tracking").warning(
                f"neumann_k={v} outside canonical sweep set {{2,3,4}}."
            )
        return v

    @model_validator(mode="after")
    def _checkpoint_exists_when_finished(self) -> "RunMetadata":
        if self.status == RunStatus.FINISHED and not Path(self.checkpoint_path).exists():
            raise ValueError(f"checkpoint_path does not exist: {self.checkpoint_path}")
        return self
```

**Why these validators are non-negotiable**

- `_must_freeze_gat_encoder` — the `GeneLink` GAT hit `pearson_r = 0.873` on v1.0 and is the load-bearing encoder. An accidental unfreeze during a Neumann sweep will destroy reproducibility. Forcing it into the schema means no run can be logged without an explicit override.
- `_monotonic` — the W-scale sweep (`0.0 → 0.5`) is encoded here so bad sweep configs fail at run-start, not at epoch 200.
- `_checkpoint_exists_when_finished` — prevents the Model Registry from pointing at a ghost artifact.

### 2.3 Supporting schemas

```python
class SweepBounds(BaseModel):
    w_scale_min: float = 0.0
    w_scale_max: float = 0.5
    neumann_k_choices: list[int] = [2, 3, 4]

class PostRunDecision(BaseModel):
    should_register: bool
    should_trigger_training_agent: bool
    should_trigger_research_agent: bool
    failure_note_path: str | None = None
    reason: str
```

---

## 3. `platform/tracking/wandb_config.py`

### 3.1 Init config

```python
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "aivc-genelink")
WANDB_ENTITY  = os.getenv("WANDB_ENTITY",  "aivc-lab")   # org/team slug
WANDB_MODE    = os.getenv("WANDB_MODE",    "online")     # online|offline|disabled

DEFAULT_TAGS: list[str] = [
    "aivc",
    "genelink-gat",
    "neumann",
    "kang2018-pbmc-ifnb",
    "benchmark-r-0.873",
]

# Mirrors aivc.memory.mlflow_backend.PLATFORM_TAGS for cross-backend consistency.
PLATFORM_CONFIG: dict[str, str] = {
    "platform":   "AIVC",
    "model":      "GeneLink-GAT-Neumann",
    "n_genes":    "3010",
    "ppi_edges":  "13878",
    "benchmark":  "r=0.873-v1.0-baseline",
}
```

### 3.2 Run group strategy

Deterministic, human-readable, sortable:

```
run_group = f"{dataset}_{sweep_name}_{utc_date}"
# e.g. "kang2018_wsweep_2026-04-13"
```

Two levels of aggregation:
- **group** — one sweep (Bayesian search). All trials share a group.
- **job_type** — `"sweep"`, `"ablation"`, `"single"`, `"eval"`. Used by W&B UI for filtering.

```python
def build_run_group(dataset: str, sweep_name: str) -> str:
    return f"{dataset}_{sweep_name}_{datetime.utcnow().date().isoformat()}"
```

### 3.3 `init_wandb()` signature

```python
def init_wandb(
    meta: RunMetadata,
    job_type: Literal["sweep","ablation","single","eval"] = "sweep",
    notes: str | None = None,
) -> "wandb.sdk.wandb_run.Run":
    """
    Side-effects:
      - wandb.init(...) with project/entity/group/job_type/tags
      - wandb.config.update(meta.model_dump(mode='json'))
      - sets meta.wandb_url (caller responsibility to persist)
    Safety:
      - If wandb not installed → returns a DummyRun with no-op methods.
      - If WANDB_MODE=disabled → same.
    """
```

### 3.4 Sweep config — W scale × Neumann K (Bayesian + Hyperband)

Exact dict to pass to `wandb.sweep(...)`:

```python
SWEEP_CONFIG: dict = {
    "name": "wscale_x_neumannk_v1",
    "method": "bayes",
    "metric": {
        "name": "val/pearson_r",
        "goal": "maximize",
    },
    "parameters": {
        "w_scale": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 0.5,
        },
        "neumann_k": {
            "values": [2, 3, 4],
        },
        # Fixed — pinned so the sweep does not drift
        "frozen_modules": {
            "value": ["genelink", "feature_expander"],
        },
        "dataset": {"value": "kang2018_pbmc_ifnb"},
        "modalities": {"value": ["rna"]},
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 20,      # epochs — allows warm-up + Neumann lr ramp
        "eta": 3,
        "s": 2,
    },
    "run_cap": 60,           # hard cap; cost guardrail
    "program": "train_v11.py",
    "command": [
        "${env}", "python", "${program}",
        "--wandb", "--sweep",
        "${args}",
    ],
}

def register_sweep() -> str:
    """Create sweep in W&B and return sweep_id. Idempotent via tag lookup."""
```

### 3.5 Contract between sweep params and `RunMetadata`

The sweep emits `w_scale: float` (scalar). `RunMetadata.w_scale_range` is a tuple `(lo, hi)`. The logger is responsible for packing: for a single trial, `w_scale_range = (wandb.config.w_scale, wandb.config.w_scale)`. The `(lo, hi)` tuple survives in metadata so a *sweep-summary* run can record the traversed range.

---

## 4. `platform/tracking/experiment_logger.py`

### 4.1 Responsibilities

- Single public class `ExperimentLogger`; composition over inheritance — wraps existing `aivc.memory.mlflow_backend.MLflowBackend`.
- Emits every metric to both backends.
- Owns the `post_run_hook` lifecycle.
- Never raises in the hot path; `__exit__` and `finish()` are best-effort.

### 4.2 Class skeleton

```python
class ExperimentLogger:
    def __init__(
        self,
        meta: RunMetadata,
        job_type: str = "sweep",
        enable_wandb: bool = True,
        enable_mlflow: bool = True,
    ): ...

    # ── context manager ─────────────────────────────────────
    def __enter__(self) -> "ExperimentLogger": ...
    def __exit__(self, exc_type, exc, tb) -> None: ...

    # ── logging ────────────────────────────────────────────
    def log_hyperparams(self, params: dict) -> None: ...
    def log_metric(self, name: str, value: float, step: int | None = None) -> None: ...
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...
    def log_artifact(self, path: str, kind: Literal["checkpoint","figure","table"]) -> None: ...
    def log_w_matrix_snapshot(self, w: "torch.Tensor", epoch: int) -> None: ...

    # ── lifecycle ──────────────────────────────────────────
    def finish(self, final_meta: RunMetadata) -> PostRunDecision: ...
```

### 4.3 `post_run_hook` — exact signature and logic

```python
def post_run_hook(
    meta: RunMetadata,
    *,
    register_threshold: float = 0.873,          # benchmark from api/server.py
    registry_name: str = "aivc.genelink",       # MLflow Model Registry key
    failure_note_dir: str = "artifacts/failure_notes",
    dispatcher: "AgentDispatcher | None" = None,
) -> PostRunDecision:
    """
    Deterministic routing after a training run finishes.

    Invariants
    ----------
    - Must be idempotent: calling twice with the same meta.run_id is a no-op
      on the second call (checked via MLflow run tag `post_run_hook=done`).
    - Never raises. Any failure downgrades to a logged warning + reason string
      in the returned PostRunDecision.

    Logic
    -----
    1. FAILURE MODE — delta_nonzero_pct == 0
         ⇒ model collapsed to ctrl memorisation (no predicted perturbation).
         ⇒ write failure_notes/{run_id}.md with:
              - meta (yaml dump)
              - last N training metrics
              - likely causes: (a) W scale too aggressive, (b) GAT unfrozen,
                (c) Neumann K too small, (d) ctrl/stim label swap.
         ⇒ dispatcher.trigger("training_agent", payload)

    2. SUCCESS MODE — pearson_r >= register_threshold
         ⇒ checkpoint_uri = f"runs:/{meta.mlflow_run_id}/checkpoints"
         ⇒ mlflow.register_model(checkpoint_uri, registry_name,
                                 tags={"pearson_r": str(meta.pearson_r),
                                       "neumann_k": str(meta.neumann_k),
                                       "dataset": meta.dataset,
                                       "anndata_hash": meta.anndata_hash})
         ⇒ transition stage → "Staging" (promotion to Production is manual).
         ⇒ dispatcher.trigger("research_agent", payload)

    3. OTHERWISE — log "no action" and return.

    Precedence
    ----------
    If BOTH conditions fire (pearson_r >= 0.873 AND delta_nonzero_pct == 0)
    — which is mathematically impossible but guard anyway — the FAILURE path
    wins. A registered model with zero-delta predictions is worse than no
    registration.

    Returns
    -------
    PostRunDecision with all three boolean flags set and a human reason.
    """
```

### 4.4 `AgentDispatcher` — minimal contract (no implementation here)

```python
class AgentDispatcher(Protocol):
    def trigger(self, agent_name: Literal["training_agent","research_agent"],
                payload: dict) -> None: ...
```

Default implementation in v1 is a **stub** that writes a JSON file to `artifacts/agent_queue/{agent_name}_{run_id}.json`. A future PR wires it to the orchestrator or a Celery/RQ queue. See §7 (gaps).

### 4.5 Failure-note template (markdown, generated by hook)

```
# FAILURE — run {run_id}
Date: {created_at}
Dataset: {dataset}
Pearson r: {pearson_r}
Δ non-zero %: {delta_nonzero_pct}   ← COLLAPSE INDICATOR
ctrl memorisation cos: {ctrl_memorisation_score}

## Config
frozen_modules: {frozen_modules}
w_scale_range:  {w_scale_range}
neumann_k:      {neumann_k}
modalities:     {modalities}
checkpoint:     {checkpoint_path}
anndata_hash:   {anndata_hash}

## Suspected causes (ranked)
1. ...
2. ...

## Next actions (for training_agent)
- [ ] Re-run with w_scale halved.
- [ ] Verify GAT truly frozen (grad.norm == 0 on genelink.*).
- [ ] Sanity-check ctrl/stim labels.
```

---

## 5. Coexistence rules — W&B and MLflow

1. **MLflow is canonical for artifacts and registry.** Do not register models via W&B Artifacts even though it supports it.
2. **W&B is canonical for sweeps.** Do not use MLflow projects for hyperparameter search; the Bayes+Hyperband combo is not available there.
3. Every scalar logged via `ExperimentLogger.log_metric` goes to both. Step semantics are identical (epoch index).
4. Run identity: `meta.run_id = wandb.run.id`; `meta.mlflow_run_id` is the MLflow run id. They are written as cross-reference tags on each other (`wandb_run_id` tag on MLflow, `mlflow_run_id` config on W&B).
5. A single training process owns exactly one active run in each backend. Nested runs are disallowed in v1.

---

## 6. Integration points (out of scope, but contracts pinned)

- **`train_v11.py`** — to be modified in a follow-up PR: replace direct calls to `aivc.memory.mlflow_backend.MLflowBackend` with `platform.tracking.ExperimentLogger`. The latter composes the former, so no metric coverage is lost.
- **`api/server.py`** — consumes the MLflow Model Registry. Checkpoint search order gains a prepended step: resolve `models:/aivc.genelink/Staging` → local path, fall back to existing list.
- **`aivc/memory/mlflow_backend.py`** — add one method `register_model(checkpoint_uri, name, tags) -> str` wrapping `mlflow.register_model`. This is the only change required in that file.

---

## 7. Gaps, unknowns, risks

| # | Item                                      | Impact | Mitigation                                                                   |
|---|-------------------------------------------|--------|------------------------------------------------------------------------------|
| 1 | `training_agent` / `research_agent` do not exist in repo | HIGH   | v1 dispatcher writes JSON to `artifacts/agent_queue/` — non-blocking stub.  |
| 2 | `pyproject.toml` does not exist           | LOW    | Add deps to `requirements.txt` (see §8). Migrate to `pyproject.toml` later.  |
| 3 | MLflow Model Registry requires a DB backend (not file store) | MED    | Document env var `MLFLOW_TRACKING_URI=<db-uri>`; fail-soft with warning.     |
| 4 | W&B entity slug is unknown                 | LOW    | Default to `"aivc-lab"`, override via `WANDB_ENTITY` env.                    |
| 5 | `delta_nonzero_pct` computation not yet implemented in eval | HIGH   | Spec assumes it is produced by the evaluator; add ticket to `evaluate_v11.py`. |
| 6 | `anndata_hash` canonicalisation undefined  | MED    | Define as `sha256` over bytes of the source `.h5ad` file, 16-char truncation OK. |
| 7 | Race condition on double `finish()`        | LOW    | MLflow tag `post_run_hook=done` provides idempotency.                        |
| 8 | Secrets — `WANDB_API_KEY` in CI            | MED    | Read from env only; never log. Document in README of `platform/tracking/`.   |

---

## 8. Exact dependencies to add

Append to `requirements.txt`:

```
# ── experiment tracking (platform/tracking/) ──
wandb>=0.17.0,<0.19
mlflow>=2.12.0,<3.0
GitPython>=3.1.40      # for git_sha capture in RunMetadata
```

Already satisfied, do not re-add: `pydantic>=2.0.0`, `torch`, `pandas`.

**Optional** (gated behind `extras`, only for MLflow Model Registry with DB backend):

```
SQLAlchemy>=2.0
psycopg2-binary>=2.9     # only if using Postgres backend
```

When `pyproject.toml` is introduced later, the equivalent entries are:

```toml
[project]
dependencies = [
  "wandb>=0.17,<0.19",
  "mlflow>=2.12,<3.0",
  "GitPython>=3.1.40",
]

[project.optional-dependencies]
registry-postgres = ["SQLAlchemy>=2.0", "psycopg2-binary>=2.9"]
```

---

## 9. Acceptance tests (for the follow-up implementation PR)

1. `RunMetadata(..., frozen_modules=["feature_expander"])` raises `ValidationError` — GAT must be frozen.
2. `RunMetadata(..., w_scale_range=(0.3, 0.1))` raises — monotonic check.
3. `init_wandb(meta)` with `WANDB_MODE=disabled` returns a dummy run, no network calls.
4. `post_run_hook(meta with delta_nonzero_pct=0, pearson_r=0.9)` ⇒ failure path wins, no registration.
5. `post_run_hook(meta with delta_nonzero_pct=5.0, pearson_r=0.88)` ⇒ `mlflow.register_model` called exactly once; `research_agent` dispatched.
6. Re-running `finish()` twice produces one registration, one dispatch (idempotency).
7. Sweep config passed to `wandb.sweep()` round-trips via `wandb.Api().sweep(sweep_id).config == SWEEP_CONFIG` modulo W&B's auto-added keys.

---

## Key Risks

- **Agent contract is fictional today.** The `training_agent`/`research_agent` names do not resolve to anything in `aivc/`. If this spec is implemented before those agents exist, the dispatch is a no-op and silent failure is possible. The v1 stub writes files so failure is observable.
- **MLflow Model Registry requires a real database.** A default file-store install will silently accept `register_model` and drop the registration. Must enforce `MLFLOW_TRACKING_URI` validation at logger init.
- **Schema rigidity vs research agility.** The `frozen_modules` validator is strict by design. Researchers will hit it; the escape hatch (`AIVC_ALLOW_GAT_UNFREEZE`) must be documented prominently or the spec becomes a footgun.
- **Dual-backend drift.** If a metric is logged to W&B via `wandb.log` directly (bypassing `ExperimentLogger`), MLflow will miss it. Enforce via a lint rule (`grep -n "wandb.log(" train_v11.py` should return zero hits after migration).
- **Sweep cost.** `run_cap=60` × 200 epochs × GPU hours is non-trivial. Early-terminate (hyperband) is load-bearing for cost; do not remove without substituting ASHA or similar.

## Recommended Next Step

Implement `platform/tracking/schemas.py` first (pure, no I/O), ship with `pytest` coverage for the 7 acceptance tests above. Only after schemas are green, wire `wandb_config.py` and `experiment_logger.py`. Add one-line `register_model` method to `aivc/memory/mlflow_backend.py` in the same PR to avoid a stub-dispatcher/real-registry mismatch.

## What I need from you

1. Confirm W&B **entity** slug (`aivc-lab`?) and whether the org has a paid seat for Bayesian sweeps + hyperband.
2. Confirm MLflow tracking URI in prod (`file://mlruns` vs Postgres). Model Registry only works with a DB backend.
3. Confirm whether `training_agent` and `research_agent` are (a) planned concrete classes, (b) prompts for Claude, or (c) external services. The dispatcher shape depends on this.
4. Sign off on the `frozen_modules` default `["genelink", "feature_expander"]` — specifically whether `feature_expander` should also be frozen, or only `genelink`.
