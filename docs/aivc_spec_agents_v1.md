# AIVC Agent Layer — Design Spec v1

Status: design only. No code written yet. This spec defines the `agents/`
package that wraps the existing `AgentDispatcher` stub in
`aivc_platform/tracking/experiment_logger.py` and closes the
train → eval → diagnose → retry loop.

## 0. Scope and non-goals

In scope:
- `agents/base_agent.py` — `AgentTask`, `AgentResult`, `BaseAgent`, `MCPTool`.
- `agents/eval_agent.py` — wraps `eval_runner.run_eval_suite`.
- `agents/training_agent.py` — diagnoses failure, emits retry sweep config.
- `agents/research_agent.py` — runs after success, writes next-experiment note.
- `agents/data_agent.py` — h5ad integrity + QuRIE-seq pairing certificate gate.

Explicitly out of scope for v1:
- Async queue / job runner. Execution is synchronous, in-process.
- Agent-to-agent messaging. Chaining goes through `AgentDispatcher` or direct
  Python calls.
- New dependencies. Only `anthropic` (already in `requirements.txt`).
- Replacing `AIVCOrchestrator.run_workflow()`. Agents live alongside it and
  are invoked from `post_run_hook`, not from the skill loop.

## 1. Architecture

```
                 ┌────────────────────────────┐
                 │  pipeline / CLI / notebook │
                 └─────────────┬──────────────┘
                               │
                      AgentTask(data_agent)
                               ▼
                     ┌──────────────────┐
                     │  DataAgent       │   gate: h5ad + pairing cert
                     └───────┬──────────┘
                             │ AgentResult.success
                             ▼
                     training loop (existing)
                             │
                             ▼
                 ExperimentLogger.finish(meta, suite)
                             │
                             ▼
                     post_run_hook(meta, suite)
                             │
         ┌───────────────────┴───────────────────┐
         │                                       │
  delta_nonzero_pct == 0                 pearson_r >= 0.873
  (ctrl memorisation)                    AND delta_nonzero_pct > 0
         │                                       │
         ▼                                       ▼
  AgentDispatcher.trigger(              AgentDispatcher.trigger(
    "training_agent", payload)            "research_agent", payload)
         │                                       │
         │ prompt file in                        │ prompt file in
         │ artifacts/agent_queue/                │ artifacts/agent_queue/
         ▼                                       ▼
  TrainingAgent.run(task)               ResearchAgent.run(task)
         │                                       │
         │ emits                                 │ emits
         │ configs/sweep_w_scale_retry.yaml      │ Claude Outputs/.../
         │ + {run_id}_response.md                │   {run_id}_next_experiment.md
         ▼                                       │ + Obsidian note
  EvalAgent.run(task)  ← optional re-eval        │
  on a newly-produced checkpoint from the        │
  retry sweep (closes the loop)                  │
         │                                       │
         ▼                                       ▼
   AgentResult                             AgentResult
```

All four agents inherit `BaseAgent` and return `AgentResult`. Only the
training and research agents make Claude API calls; `EvalAgent` and
`DataAgent` are deterministic.

## 2. `agents/base_agent.py`

### 2.1 Schemas

```python
# agents/base_agent.py
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field


AgentName = Literal["data_agent", "training_agent", "eval_agent", "research_agent"]


class AgentTask(BaseModel):
    agent_name: AgentName
    run_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentResult(BaseModel):
    agent_name: AgentName
    run_id: str
    success: bool
    output_path: Optional[str] = None   # absolute path to primary artifact
    summary: str = ""                   # one-paragraph human summary
    error: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)
```

### 2.2 `MCPTool` protocol

```python
@runtime_checkable
class MCPTool(Protocol):
    name: str
    description: str
    def execute(self, input: dict) -> dict: ...
```

Not wired to any MCP server in v1. The protocol exists so EvalAgent,
TrainingAgent etc. can accept future tool objects without refactor.

### 2.3 `BaseAgent`

```python
class BaseAgent(ABC):
    agent_name: AgentName
    tools: list[MCPTool]

    def __init__(
        self,
        tools: Optional[list[MCPTool]] = None,
        context_path: str = "docs/context_snapshot.md",
        registry_path: str = "artifacts/registry/latest_checkpoint.json",
        queue_dir: str = "artifacts/agent_queue",
    ) -> None:
        self.tools = tools or []
        self.context_path = Path(context_path)
        self.registry_path = Path(registry_path)
        self.queue_dir = Path(queue_dir)

    # ---- memory -------------------------------------------------
    def read_context(self) -> str:
        return self.context_path.read_text() if self.context_path.exists() else ""

    def read_latest_checkpoint(self) -> dict:
        import json
        if not self.registry_path.exists():
            return {}
        return json.loads(self.registry_path.read_text())

    # ---- history ------------------------------------------------
    def query_run_history(self, min_pearson: float = 0.0) -> list[dict]:
        """Scan artifacts/agent_queue/ for past dispatches and return runs
        whose payload recorded pearson_r >= min_pearson.

        v1 implementation: parse the `Pearson r:` line from prompt files
        written by AgentDispatcher. Returns list of
        {"run_id", "pearson_r", "file"} dicts, newest first.
        """
        ...

    # ---- main entrypoint ---------------------------------------
    @abstractmethod
    def run(self, task: AgentTask) -> AgentResult: ...
```

Memory contract: all agents read the same two files. No agent mutates
`context.md` or `latest_checkpoint.json` directly — only the memory layer
(`context_updater.update_context`, `ExperimentLogger._handle_success`) does.

## 3. `agents/data_agent.py`

Deterministic gate. No Claude call. Must pass before any training run.

```python
class DataReport(BaseModel):
    valid: bool
    n_genes: int
    n_cells: int
    cert_present: bool
    cert_path: Optional[str] = None
    blocked_reason: Optional[str] = None
```

### 3.1 Signature

```python
class DataAgent(BaseAgent):
    agent_name = "data_agent"

    REQUIRED_OBS = {"cell_type", "stim", "donor"}  # minimum Kang cols
    EXPECTED_N_GENES = 3010

    def __init__(
        self,
        cert_dir: str = "data/pairing_certificates",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cert_dir = Path(cert_dir)

    def run(self, task: AgentTask) -> AgentResult: ...
```

### 3.2 `AgentTask.payload` required keys

| key | type | notes |
|---|---|---|
| `h5ad_path` | `str` | absolute or repo-relative path to AnnData file |
| `dataset` | `str` | `"kang2018_pbmc"` / `"norman2019"` |
| `require_cert` | `bool` | default `True`; set `False` for public datasets |

### 3.3 Logic

1. Load with `anndata.read_h5ad(h5ad_path, backed="r")` — do not load X.
2. Validate `adata.n_vars == EXPECTED_N_GENES`. Otherwise → `valid=False`.
3. Validate `REQUIRED_OBS ⊆ set(adata.obs.columns)`.
4. If `require_cert=True`: search `cert_dir` for any file matching
   `*{dataset}*quriepair*.json` (or equivalent). Missing cert →
   `valid=False, blocked_reason="pairing certificate missing"`.
5. Write `DataReport` JSON to
   `artifacts/agent_queue/data_agent_{run_id}_{ts}.json`.
6. Return `AgentResult(success=valid, output_path=<json>, summary=...)`.

### 3.4 Failure behaviour

`DataAgent` never raises on data invalidity — it returns `success=False`
with `blocked_reason`. Callers (training pipeline, orchestrator) MUST check
`AgentResult.success` and halt ingestion before any TileDB-SOMA write.

### 3.5 Connection to `AgentDispatcher`

DataAgent is **not** triggered by `AgentDispatcher`. It runs pre-training,
outside the post-run hook. It is invoked directly by the pipeline entry
point (e.g. `scripts/train.py`) before `ExperimentLogger.start()`.

## 4. `agents/eval_agent.py`

Deterministic. Wraps `eval.eval_runner.run_eval_suite`. No Claude call.

### 4.1 Signature

```python
class EvalAgent(BaseAgent):
    agent_name = "eval_agent"

    def run(self, task: AgentTask) -> AgentResult: ...
```

### 4.2 `AgentTask.payload` required keys

| key | type | notes |
|---|---|---|
| `checkpoint_path` | `str` | required |
| `kang_adata` | `str` | default `data/kang2018_pbmc_fixed.h5ad` |
| `norman_adata` | `str` | default `data/norman2019.h5ad` |
| `device` | `str \| None` | auto-detect if None |

### 4.3 Logic

1. Resolve `checkpoint_path`. If missing, fall back to
   `self.read_latest_checkpoint()["checkpoint_path"]`.
2. Call `suite = run_eval_suite(checkpoint_path, run_id=task.run_id, ...)`.
3. Serialise `suite.model_dump()` to
   `artifacts/agent_queue/eval_agent_{run_id}_{ts}.json`. This is
   `AgentResult.output_path`. **Always** written, including on failure.
4. Success criteria:
   - Kang regression guard passed → required.
     `not suite.kang.regression_guard_passed` → `success=False`,
     `error="kang regression guard failed: {reason}"`.
   - Norman delta collapse (`delta_nonzero_pct == 0`) →
     `success=False, error="norman delta collapse"` but suite JSON still
     written (used by TrainingAgent downstream).
   - Otherwise `success = suite.overall_passed`.

### 4.4 Connection to `AgentDispatcher`

EvalAgent is invoked in two places:

1. Inside the training pipeline, immediately after training. Its
   `AgentResult` populates `RunMetadata` via `populate_run_metadata`, then
   `ExperimentLogger.finish()` routes to success/failure.
2. By `TrainingAgent` — after TrainingAgent writes a retry sweep config
   and (externally) a new checkpoint is produced, EvalAgent re-runs to
   close the loop.

EvalAgent does not itself call `AgentDispatcher.trigger`. It feeds data
into the existing `post_run_hook` path.

## 5. `agents/training_agent.py`

### 5.1 Signature

```python
class TrainingAgent(BaseAgent):
    agent_name = "training_agent"
    claude_model = "claude-sonnet-4-20250514"

    def __init__(
        self,
        retry_config_path: str = "configs/sweep_w_scale_retry.yaml",
        **kwargs,
    ) -> None: ...

    def run(self, task: AgentTask) -> AgentResult: ...
```

### 5.2 `AgentTask.payload` required keys

Populated by `AgentDispatcher.trigger("training_agent", payload)`
(see `experiment_logger.py:470-482`):

| key | type |
|---|---|
| `run_id` | `str` |
| `dataset` | `str` |
| `pearson_r` | `float` |
| `delta_nonzero_pct` | `float` |
| `ctrl_memorisation_score` | `float` |
| `w_scale_range` | `str` (tuple repr) |
| `neumann_k` | `int` |
| `failure_note_path` | `str` |
| `eval_suite_path` | `str` (optional — path to EvalAgent output JSON) |

### 5.3 Logic

1. Load prompt file: most recent `training_agent_{run_id}_*.md` in
   `artifacts/agent_queue/` (already written by `AgentDispatcher`).
2. Load `EvalSuite` JSON from `payload["eval_suite_path"]` if present.
3. Diagnose cause (rule-based pre-prompt, then Claude refines):
   - `w_scale_range[1] >= 0.2` AND `delta_nonzero_pct == 0` →
     "W scale collapse"
   - `neumann_k <= 2` AND `pearson_r < 0.5` → "K too small"
   - `ctrl_memorisation_score > 0.95` → "ctrl label memorisation"
   - else → "unknown; defer to Claude"
4. Call Claude API with the prompt from `AGENT_PROMPTS["training_agent"]`
   (already templated by AgentDispatcher) — exact template shown in §5.4.
   Model: `claude-sonnet-4-20250514`, `max_tokens=4096`.
5. Parse response for a ```yaml fenced block`. Write YAML to
   `configs/sweep_w_scale_retry.yaml` (overwrites). If no YAML block →
   emit a stub with conservative defaults and flag `extra["no_yaml_from_claude"]=True`.
6. Write full Claude response to
   `artifacts/agent_queue/training_agent_{run_id}_{ts}_response.md`.
7. Return `AgentResult` with `output_path=configs/sweep_w_scale_retry.yaml`.

### 5.4 Claude API prompt template (exact)

Training agent reuses the exact template in
`experiment_logger.AgentDispatcher.AGENT_PROMPTS["training_agent"]`
(verbatim copy — do not fork):

```
# Training Agent Task

## Context
Run ID: {run_id}
Dataset: {dataset}
Pearson r: {pearson_r}
Delta non-zero %: {delta_nonzero_pct}
ctrl memorisation score: {ctrl_memorisation_score}
Neumann K: {neumann_k}
Failure note: {failure_note_path}

## Problem
The model collapsed to ctrl memorisation (delta_nonzero_pct == 0).
Predicted perturbation response is indistinguishable from ctrl.

## Your task
1. Diagnose the most likely cause from:
   a. W scale too aggressive (sparsity collapse)
   b. Neumann K too small (insufficient propagation depth)
   c. Learning rate too high (GAT overfits to ctrl distribution)
   d. Data issue (ctrl/stim label swap or contamination)
2. Recommend exact hyperparameter changes for the next run.
3. Output a configs/sweep_w_scale_retry.yaml with corrected bounds.

## Constraints
- genelink must remain frozen (requires_grad=False)
- Do not change n_genes=3010 or edge index
- Write output to Claude Outputs/aivc_genelink/
```

### 5.5 Failure behaviour

- `ANTHROPIC_API_KEY` absent → `AgentResult.success=False, error="no api key"`.
  Prompt file already on disk (written by AgentDispatcher).
- Claude API 4xx/5xx → retry once with 2 s backoff. Second failure →
  `success=False, error=str(exc)`. Prompt file retained.
- YAML parse failure → `success=True` (response was written) but
  `extra["no_yaml_from_claude"]=True`; human must review.

### 5.6 Connection to `AgentDispatcher`

`AgentDispatcher.trigger("training_agent", payload)` already writes the
prompt file and, if key is set, calls Claude. TrainingAgent replaces the
inline Claude call: the dispatcher should be reconfigured to **only write
prompt files**, and a separate process (or the pipeline) instantiates
`TrainingAgent` and calls `agent.run(AgentTask(...))`. This keeps the
dispatcher as pure-IO and agents responsible for reasoning. Backwards
compat: if the pipeline does not instantiate TrainingAgent, the dispatcher
falls back to its v1 inline behaviour — no regression.

## 6. `agents/research_agent.py`

### 6.1 Signature

```python
class ResearchAgent(BaseAgent):
    agent_name = "research_agent"
    claude_model = "claude-sonnet-4-20250514"

    def run(self, task: AgentTask) -> AgentResult: ...
```

### 6.2 `AgentTask.payload` required keys

Populated by `AgentDispatcher.trigger("research_agent", payload)`
(see `experiment_logger.py:527-539`):

| key | type |
|---|---|
| `run_id` | `str` |
| `dataset` | `str` |
| `pearson_r` | `float` |
| `delta_nonzero_pct` | `float` |
| `w_scale_range` | `str` |
| `neumann_k` | `int` |
| `checkpoint_path` | `str` |
| `wandb_url` | `str` |

### 6.3 Logic

1. Fetch W&B run summary via `wandb.Api().run(wandb_url)` — best-effort;
   on failure proceed without extra metrics.
2. Idempotent re-trigger of memory writes (the primary write already
   happened inside `_handle_success`). ResearchAgent calls
   `write_experiment_note(meta, suite, obsidian_config)` and
   `update_context(meta, suite)` from `aivc_platform.memory` **only if**
   the Obsidian note for `run_id` does not yet exist — this lets a manual
   `agent.run()` re-populate missing notes without duplicating.
3. Call Claude API with `AGENT_PROMPTS["research_agent"]` (§6.4). Model
   `claude-sonnet-4-20250514`, `max_tokens=4096`.
4. Parse Claude output into a `next_experiment` markdown block. Write to
   `Claude Outputs/aivc_genelink/{run_id}_next_experiment.md`.
5. Also write raw response to
   `artifacts/agent_queue/research_agent_{run_id}_{ts}_response.md`
   (mirrors AgentDispatcher).
6. Return `AgentResult` with `output_path` pointing to
   `{run_id}_next_experiment.md`.

### 6.4 Claude API prompt template (exact)

Reuses `AGENT_PROMPTS["research_agent"]` verbatim:

```
# Research Agent Task

## Context
Run ID: {run_id}
Dataset: {dataset}
Pearson r: {pearson_r}
Delta non-zero %: {delta_nonzero_pct}
W scale range: {w_scale_range}
Neumann K: {neumann_k}
Checkpoint: {checkpoint_path}
W&B run: {wandb_url}

## Your task
1. Summarise what changed vs the previous best checkpoint (r=0.873).
2. Interpret the delta_nonzero improvement biologically:
   what does a higher % mean for perturbation prediction in PBMCs?
3. Write an Obsidian experiment note to:
   ~/Documents/Obsidian/aivc_genelink/experiments/{run_id}.md
   Use the standard experiment note template.
4. Suggest one concrete next experiment (hypothesis + config change).
5. Update context.md field "best pearson_r" if this run exceeds current best.

## Constraints
- Write only to Claude Outputs/ and Obsidian vault
- Do not modify any source file
```

### 6.5 Failure behaviour

- Missing `wandb_url` → proceed with payload-only context.
- Obsidian vault not mounted → skip memory call; still write next-experiment
  markdown.
- Claude API failure → `success=False, error=str(exc)`. The success-path
  registry write has already happened upstream, so no training-state
  corruption.

### 6.6 Connection to `AgentDispatcher`

Same pattern as TrainingAgent (§5.6). The dispatcher writes the prompt;
ResearchAgent consumes it and performs the reasoning + follow-up writes.

## 7. Agent chain — end-to-end

Textual sequence for a single training run:

```
1. pipeline → DataAgent.run(task)                 # gate
   └─ fail  → halt; no TileDB-SOMA writes; exit
   └─ pass  → continue
2. pipeline → train model → checkpoint            # existing code
3. pipeline → EvalAgent.run(task)                 # deterministic eval
   └─ populate_run_metadata(meta, suite)
4. pipeline → ExperimentLogger.finish(meta, suite)
   └─ post_run_hook(meta, suite)
        ├─ FAILURE path
        │    AgentDispatcher.trigger("training_agent", payload)
        │         └─ writes artifacts/agent_queue/training_agent_*.md
        │    TrainingAgent.run(AgentTask("training_agent", ...))
        │         ├─ diagnose + Claude call
        │         ├─ writes configs/sweep_w_scale_retry.yaml
        │         └─ returns AgentResult
        │    [external scheduler or human kicks off retry sweep]
        │    retry sweep produces new checkpoint
        │    EvalAgent.run(...)   ← loop closes here
        │
        └─ SUCCESS path
             AgentDispatcher.trigger("research_agent", payload)
                  └─ writes artifacts/agent_queue/research_agent_*.md
             ResearchAgent.run(AgentTask("research_agent", ...))
                  ├─ Claude call
                  ├─ writes Claude Outputs/.../{run_id}_next_experiment.md
                  └─ returns AgentResult
```

Invariants:
- `genelink` stays frozen across every retry (`RunMetadata` validator).
- No agent writes `artifacts/registry/latest_checkpoint.json`. Only
  `ExperimentLogger._handle_success` does.
- Every agent output is addressable: either under `artifacts/agent_queue/`
  or `Claude Outputs/aivc_genelink/`.

## 8. File / directory conventions

| artifact | path |
|---|---|
| Prompt files | `artifacts/agent_queue/{agent}_{run_id}_{ts}.md` |
| Claude responses | `artifacts/agent_queue/{agent}_{run_id}_{ts}_response.md` |
| EvalSuite JSON | `artifacts/agent_queue/eval_agent_{run_id}_{ts}.json` |
| DataReport JSON | `artifacts/agent_queue/data_agent_{run_id}_{ts}.json` |
| Retry sweep config | `configs/sweep_w_scale_retry.yaml` |
| Next-experiment md | `Claude Outputs/aivc_genelink/{run_id}_next_experiment.md` |
| Obsidian notes | `~/Documents/Obsidian/aivc_genelink/experiments/{run_id}.md` |
| Failure notes | `artifacts/failure_notes/failure_{run_id}.json` |
| Registry | `artifacts/registry/latest_checkpoint.json` |
| Pairing certs | `data/pairing_certificates/*quriepair*.json` |

## 9. Testing strategy (v1)

- `tests/agents/test_base_agent.py` — memory reads, history query, schemas.
- `tests/agents/test_data_agent.py` — fixture with 3010 genes passes;
  3009 fails; missing cert fails; `require_cert=False` passes without cert.
- `tests/agents/test_eval_agent.py` — monkeypatch `run_eval_suite` to return
  canned `EvalSuite`s for (Kang-fail, delta-collapse, all-pass); assert
  JSON is always written.
- `tests/agents/test_training_agent.py` — monkeypatch `anthropic.Anthropic`;
  assert YAML is extracted and written to
  `configs/sweep_w_scale_retry.yaml`.
- `tests/agents/test_research_agent.py` — monkeypatch W&B + `anthropic`;
  assert `{run_id}_next_experiment.md` is produced.

No live Claude API calls in CI. Use `ANTHROPIC_API_KEY` guards.

## 10. Key Risks

- **Dispatcher double-execution.** Current `AgentDispatcher` calls Claude
  inline when `ANTHROPIC_API_KEY` is set. If TrainingAgent is also wired,
  Claude will be called twice per failure. Mitigation: add a
  `AgentDispatcher(execute=False)` flag (new kwarg) and flip it to `False`
  once agents are live; keep default `True` until rollout.
- **Prompt drift.** The `AGENT_PROMPTS` constants are duplicated in this
  spec. Source of truth remains `experiment_logger.py`. Do not fork; agents
  should import the dict.
- **YAML extraction fragility.** Relying on fenced ```yaml blocks means a
  Claude formatting change silently breaks retries. Add a structural-output
  check and surface `no_yaml_from_claude` loudly.
- **W&B API dependency.** ResearchAgent will fail cleanly without W&B, but
  losing run summaries degrades the next-experiment suggestion quality.
- **Pairing cert schema unspecified.** `data/pairing_certificates/*.json`
  format is not yet locked. v1 DataAgent only checks presence, not schema.
- **No async / queue.** A training failure blocks until Claude returns.
  Acceptable for v1 (one run at a time on a single GPU); will not scale
  to the sweep batch-mode case.

## 11. Recommended Next Step

Implement `agents/base_agent.py` + `agents/data_agent.py` first. These are
deterministic, require no Claude wiring, and unblock the TileDB-SOMA
ingest gate — the hardest compliance requirement. Follow with `EvalAgent`
(also deterministic), then TrainingAgent and ResearchAgent once the
dispatcher `execute=False` flag is in place.

## 12. What I need from you

1. Confirm the pairing-certificate filename pattern
   (`*{dataset}*quriepair*.json` is a guess).
2. Confirm whether `docs/context_snapshot.md` or a different path is the
   canonical `context.md` for `BaseAgent.read_context()`.
3. Sign-off on the dispatcher `execute=False` migration — this is the
   only backwards-incompatible change in the spec.
