# AIVC Repo Interrogation — v1

**Date:** 2026-04-13
**Scope:** Answer 5 targeted questions strictly from repo contents. No guessing.

---

## Q1 — W&B entity slug

**Answer:** not found in repo.

**Evidence**
- `requirements.txt` (lines 1–19): no `wandb` package listed.
- No `.env` or `.env.example` present at repo root (glob returned no matches).
- No `Makefile` at repo root (only a vendored `nbconvert` template inside `aivc_env/`).
- Grep for `wandb|WANDB` across the project (excluding `aivc_env/`) returns only one hit — inside `Claude Outputs/aivc_genelink/aivc_spec_tracking_foundation_v1.md` (a prior design doc), never in runnable code, no yaml config.

---

## Q2 — W&B account tier

**Answer:** `wandb` is not a dependency at all — no tier needed, no tier configured.

**Evidence**
- `wandb` absent from `requirements.txt`.
- Zero `wandb.sweep()` / `hyperband` / `bayes` references anywhere in the codebase.
- Experiment tracking is implemented via **MLflow** instead (`aivc/memory/mlflow_backend.py`), which is also not pinned in `requirements.txt` — it is imported defensively with ImportError fallback (`mlflow_backend.py:19–28`) and simply no-ops if absent.

---

## Q3 — MLflow tracking URI

**Answer:** Default `http://localhost:5000`, with automatic fallback to local file store `./mlruns`.

**Evidence**
- `aivc/memory/mlflow_backend.py:32` — `TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")`
- `aivc/memory/mlflow_backend.py:33` — `FALLBACK_URI = "./mlruns"`
- `aivc/memory/mlflow_backend.py:82` — `mlflow.set_tracking_uri(self.tracking_uri)` (resolved value)
- `aivc/memory/mlflow_backend.py:90` — fallback invocation on unreachable server: `mlflow.set_tracking_uri(FALLBACK_URI)`
- No `.env` / `.env.example` exists, so `os.getenv("MLFLOW_TRACKING_URI", ...)` resolves to the default string above at process start.
- No `sqlite://` or `postgresql://` URI is referenced anywhere — **backend type is HTTP (MLflow server)** with degrade path to a **local-file store** (no DB backend).

---

## Q4 — Agent architecture

**Answer:** There is no agent subsystem. The platform runs a **synchronous in-process workflow orchestrator** that dispatches via direct Python method calls.

**What exists today**
- **No `aivc/agents/` directory** (Glob returned zero files).
- **No class named `*Agent`** in the codebase.
- **No imports** of `langchain`, `autogen`, `crewai`, `anthropic`, or `openai`.
- **No task queue** — no `celery`, `rq`, `redis`, or `dramatiq`.
- `aivc/orchestration/orchestrator.py` defines `AIVCOrchestrator` — a plain synchronous class:
  - `run_workflow()` (line 54) pulls a named workflow from the `WORKFLOWS` dict and iterates its steps in a for-loop (line 85).
  - Dispatch is in-process function call: `skill = self.registry.get(step.skill)` (line 86) → `skill.execute(step_inputs, context)` (line 114).
  - A cost gate runs before GPU-intensive steps (lines 89–108).
  - Three critics (`statistical`, `methodological`, `biological`) validate every step result (lines 150–160); on failure it attempts one inline replan via `_replan` / `_replan_validation` (lines 214–296).
- The only "queue" is `aivc/memory/active_learning.py` — a JSON-backed list of uncertainty-ranked genes for experiment prioritisation (`STORAGE_PATH = "memory/active_learning_queue.json"`). It is **not** a task queue; it is a data queue of candidate experiments.

**Dispatch mechanism (summary)**
Named workflow → ordered `Step` list → per-step cost gate → per-step `skill.execute(...)` synchronous call → per-step 3-critic validation → next step. All in-process, single thread of control, no async/await, no broker.

---

## Q5 — `feature_expander` frozen status

**Answer:** **No, it is not frozen anywhere.** It is a standalone submodule (no shared weights with `genelink`). The only freeze operation in the training loop targets the Neumann `W` matrix — `genelink` itself is also never frozen, so there is no "alongside genelink" freeze to mirror.

**Attribute definition**
- `perturbation_model.py:248` — `self.feature_expander = FeatureExpander(feature_dim)`
- `perturbation_model.py:93–97` — `FeatureExpander` is `nn.Sequential(Linear(1, d), LayerNorm(d), GELU)` with its own parameters.

**Separation from `genelink`**
- `perturbation_model.py:256` — `self.genelink = GeneLink(input_dim=feature_dim, ...)` — separately instantiated `GeneLink` (2-layer `GATConv` stack, `perturbation_model.py:47–72`).
- Forward path (`perturbation_model.py:290 → 293 → 300`): `feature_expander → pert_embedding → genelink → decoder`. Distinct modules, distinct `nn.Parameter` sets. **No shared weights.**

**Freeze operations in `train_v11.py`**
- `train_v11.py:323` — `model.neumann.freeze_W()` — Stage 1 (epochs 1–10), freezes Neumann `W` only.
- `train_v11.py:359` — `model.neumann.unfreeze_W()` — Stage 2 (epoch 11+).
- Grep of `feature_expander|genelink|requires_grad|requires_grad_` in `train_v11.py` returns:
  - line 266 — `model.cell_type_embedding = CellTypeEmbedding(...)`
  - line 480 — `if epoch >= 11 and model.neumann.W.requires_grad:`
  - (zero occurrences of `param.requires_grad = False` or `.requires_grad_(False)` anywhere in the file)

**Implication**
- Neither `feature_expander` nor `genelink` is ever frozen in the current training loop.
- If a future stage freezes `genelink`, `feature_expander` **must be frozen independently** — it is a distinct `nn.Module` and it sits directly upstream in the forward path, so leaving it trainable while freezing `genelink` would shift the encoder's input distribution and silently degrade the GAT's learned attention.

---

## Key Risks
- **Observability gap:** MLflow defaults to `http://localhost:5000` with a silent fallback to `./mlruns`. In production/CI this masks tracking-server outages as "everything looks fine locally" — run-id continuity across machines is lost.
- **No async/agent layer:** The orchestrator blocks on every skill. Any GPU-heavy step stalls the whole workflow. There is no retry-with-backoff, no cross-run concurrency, no work stealing.
- **Training correctness tripwire:** If `genelink` is frozen in a future stage without also freezing `feature_expander`, gradients will still flow into the upstream encoder and the frozen GAT will see drifting features — a subtle, non-erroring failure mode. Tests do not currently guard this.

## Recommended Next Step
Add two guards in one small PR:
1. `MLFLOW_TRACKING_URI` to a committed `.env.example` with an explicit comment ("Do not ship localhost default"), and an assertion in `MLflowBackend.__init__` that logs a WARNING at WARN level whenever the fallback file store is activated.
2. A `freeze_encoder()` helper on `PerturbationPredictor` that freezes `feature_expander`, `pert_embedding`, and `genelink` together, plus a unit test asserting `sum(p.requires_grad for p in model.parameters())` drops to expected count after the call.

## What I need from you
Confirm whether `wandb` is intended to replace MLflow (the design doc in `Claude Outputs/` mentions it) or run alongside it — so the tracking-URI fix targets the right backend.
