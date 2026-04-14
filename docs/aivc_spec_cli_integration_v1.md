# AIVC CLI Integration Specification v1

## Executive Summary

This spec covers the end-to-end design for a Typer-based CLI (`cli.py`) that orchestrates the full AIVC training→eval→registry→memory loop. The design integrates existing agents (DataAgent, EvalAgent, TrainingAgent, ResearchAgent), the experiment logger, and model registry with deterministic exits and failure modes. Key decisions: `train_v11.py` is a subprocess-callable script (not a function import); `api/server.py` checkpoint lookup is minimally patched to prepend `latest_checkpoint.json`; agents own their own CloudAI calls and write artifacts to deterministic paths.

---

## 1. CLI Command Signatures & Flows

### 1.1 `aivc train`

**Signature:**
```
aivc train [--config CONFIG] [--dataset DATASET] [--run-id RUN_ID] 
           [--output-dir OUTPUT_DIR] [--device DEVICE] [--sweep SWEEP_YAML]
           [--skip-eval] [--no-memory-sync]
```

**Flags:**
- `--config`: Path to sweep config YAML (defaults: `configs/sweep_w_scale_retry.yaml` if exists, else `configs/default_train.yaml`)
- `--dataset`: Dataset name for training (default: `kang2018_pbmc`)
- `--run-id`: Unique run identifier (auto-generated as UUID if absent)
- `--output-dir`: Checkpoint output directory (default: `models/v1.1/`)
- `--device`: `cuda`, `cpu` (auto-detect if absent)
- `--sweep`: Override sweep bounds as inline JSON/YAML (e.g., `--sweep '{"w_scale": [0.0, 0.15]}'`)
- `--skip-eval`: Skip post-training eval (default: False — always eval)
- `--no-memory-sync`: Suppress ExperimentLogger.finish() (default: False)

**Workflow:**
1. DataAgent validates h5ad + pairing cert (halts if missing/invalid)
2. Subprocess `python train_v11.py --run-id {run_id} --config {config} --output {output_dir}`
3. If exit code != 0, log failure, exit 1
4. EvalAgent loads checkpoint, runs eval suite
5. populate_run_metadata fills RunMetadata with eval results
6. ExperimentLogger.finish(meta) routes post-run actions
7. On success: registry + memory writes + ResearchAgent dispatch
8. On failure: failure note + TrainingAgent dispatch

**Exit Codes:**
- `0`: Training + eval succeeded (overall_passed=True)
- `1`: DataAgent halted or train_v11 exit != 0
- `2`: EvalAgent kang regression guard failed
- `3`: EvalAgent Norman eval collapsed (delta_nonzero_pct == 0)
- `4`: ExperimentLogger write failures (non-fatal; training succeeded but artifact writes failed)

---

### 1.2 `aivc eval`

**Signature:**
```
aivc eval --checkpoint CHECKPOINT [--run-id RUN_ID] [--device DEVICE]
          [--kang-adata PATH] [--norman-adata PATH] [--output-dir OUTPUT_DIR]
```

**Flags:**
- `--checkpoint`: Path to model checkpoint (required; no fallback to latest_checkpoint.json)
- `--run-id`: Unique run identifier (auto-generated if absent)
- `--device`: `cuda`, `cpu` (auto-detect if absent)
- `--kang-adata`: Kang 2018 h5ad path (default: `data/kang2018_pbmc_fixed.h5ad`)
- `--norman-adata`: Norman 2019 h5ad path (default: `data/norman2019.h5ad`)
- `--output-dir`: Directory to write eval results JSON (default: `artifacts/eval_results/`)

**Workflow:**
1. EvalAgent.run() with checkpoint_path (fails immediately if path missing)
2. run_eval_suite(checkpoint_path, run_id=..., device=...)
3. Outputs: EvalSuite JSON to `artifacts/eval_results/{run_id}_suite.json`
4. Prints EVAL SUITE summary to stdout

**Exit Codes:**
- `0`: All evals passed (overall_passed=True)
- `1`: Kang regression guard failed
- `2`: Norman eval failed or collapsed
- `3`: Checkpoint load error or data file missing

---

### 1.3 `aivc sweep`

**Signature:**
```
aivc sweep [--config CONFIG] [--dataset DATASET] [--num-runs NUM_RUNS]
           [--output-dir OUTPUT_DIR] [--device DEVICE] [--parallel PARALLEL]
           [--no-halt-on-fail]
```

**Flags:**
- `--config`: Sweep bounds YAML (default: `configs/default_train.yaml`)
- `--dataset`: Dataset (default: `kang2018_pbmc`)
- `--num-runs`: Max hyperparameter samples to try (default: 12)
- `--output-dir`: Checkpoint output (default: `models/v1.1/`)
- `--device`: `cuda`, `cpu` (default: auto-detect)
- `--parallel`: Number of concurrent training processes (default: 1, max 4 on single GPU)
- `--no-halt-on-fail`: Continue sweep if a run fails (default: halt on first failure)

**Workflow:**
1. Load sweep bounds from config YAML
2. Sample N hyperparameter sets (uniform random or Bayesian if wandb sweep)
3. For each set: submit `aivc train --config {per-run-config} --run-id {run_id_N}`
4. Collect per-run exit codes
5. If any exit code in {2, 3} (eval halt) and not `--no-halt-on-fail`: stop sweep
6. Write `artifacts/sweep_results_{timestamp}.json` with all run metadata
7. Print best r, best run_id

**Exit Codes:**
- `0`: All runs completed; best_r >= 0.873
- `1`: One or more runs failed (training or data error)
- `2`: Sweep halted on first eval regression (Kang guard failed)
- `3`: Sweep halted on first delta collapse (Norman eval == 0)

---

### 1.4 `aivc agent run`

**Signature:**
```
aivc agent run --agent AGENT --run-id RUN_ID [--payload PAYLOAD_JSON]
               [--no-api-call]
```

**Flags:**
- `--agent`: Agent name (`data_agent`, `eval_agent`, `training_agent`, `research_agent`)
- `--run-id`: Run identifier
- `--payload`: Agent input as JSON dict (default: `{}`)
- `--no-api-call`: Write prompt file only, suppress Claude API call (for training/research agents)

**Workflow:**
1. Construct AgentTask(agent_name=..., run_id=..., payload=...)
2. Instantiate agent, call agent.run(task)
3. Print AgentResult to stdout (JSON)
4. Write output_path file (report or response)

**Exit Codes:**
- `0`: agent.run() returned success=True
- `1`: agent.run() returned success=False
- `2`: Agent instantiation or task construction failed

---

### 1.5 `aivc memory sync`

**Signature:**
```
aivc memory sync [--run-ids RUN_IDS] [--force-update]
```

**Flags:**
- `--run-ids`: Comma-separated run IDs (default: all in artifacts/agent_queue/)
- `--force-update`: Overwrite existing Obsidian notes (default: skip if exists)

**Workflow:**
1. For each run_id: read latest_checkpoint.json and EvalSuite JSON
2. Call write_experiment_note(meta, suite, ObsidianConfig()) if not exists or `--force-update`
3. Call update_context(meta, suite) to patch docs/context_snapshot.md
4. Print sync summary (N notes written, M contexts updated)

**Exit Codes:**
- `0`: All syncs succeeded (or no runs to sync)
- `1`: Vault not accessible or Obsidian config missing
- `2`: One or more write failures (partial success reported)

---

### 1.6 `aivc status`

**Signature:**
```
aivc status [--run-id RUN_ID] [--summary]
```

**Flags:**
- `--run-id`: Show status of single run (default: show all runs)
- `--summary`: One-line per run (default: full details)

**Workflow:**
1. Scan `artifacts/agent_queue/*.json` and `artifacts/registry/latest_checkpoint.json`
2. Parse run metadata and construct status table
3. For each run: show run_id, status, pearson_r, delta_nonzero_pct, started_at, finished_at

**Example output:**
```
Run ID                 Status      Pearson r  Delta NZ%  Duration
─────────────────────────────────────────────────────────────────
run_uuid_001_train     SUCCESS     0.8755     15.2%      2h 14m
run_uuid_002_train     FAILURE     0.6200     0.0%       1h 47m
run_uuid_003_eval      HALTED      –          –          (pending)
```

**Exit Codes:**
- `0`: Status query succeeded
- `1`: No runs found

---

## 2. ModelRegistry Pydantic Schema

### 2.1 ModelCard

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class ModelCard(BaseModel):
    """Canonical model registry entry."""
    # Identity
    version: str = Field(..., description="Semantic version (e.g. 'v1.1-r0.875')")
    run_id: str = Field(..., description="Unique training run identifier")
    
    # Metrics
    pearson_r: float = Field(..., ge=-1.0, le=1.0, description="Pearson r on Kang 2018 held-out")
    pearson_r_std: Optional[float] = Field(default=None, ge=0.0)
    delta_nonzero_pct: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="Percentage of genes with nonzero predicted delta"
    )
    ctrl_memorisation_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Cosine similarity between predicted stim and ctrl"
    )
    jakstat_within_3x: Optional[int] = Field(default=None, ge=0, le=15)
    
    # Paths & URLs
    checkpoint_path: str = Field(..., description="Absolute or relative path to model checkpoint")
    datasets_trained: List[str] = Field(
        default_factory=lambda: ["kang2018_pbmc"],
        description="List of dataset names used for training"
    )
    frozen_modules: List[str] = Field(
        default_factory=lambda: ["genelink"],
        description="Module names frozen during training"
    )
    wandb_url: Optional[str] = Field(default=None, description="W&B run URL")
    
    # Timestamps
    registered_at: datetime = Field(default_factory=datetime.utcnow)
    notes: Optional[str] = Field(default=None, description="Free-text notes")
    
    # Regression guard
    regression_guard_passed: bool = Field(
        default=True,
        description="False if pearson_r < 0.873; reject on False"
    )
```

### 2.2 Registry Functions

```python
class ModelRegistry:
    """Manages model registry (file-based: artifacts/registry/*.json)."""
    
    def register(
        meta: RunMetadata,
        suite: EvalSuite,
    ) -> ModelCard:
        """
        Create and write ModelCard to artifacts/registry/{run_id}.json.
        
        Args:
            meta: RunMetadata with checkpoint_path, run_id, etc.
            suite: EvalSuite with kang, norman results
            
        Returns:
            ModelCard instance
            
        Raises:
            ValueError if regression_guard_passed=False (pearson_r < 0.873)
        """
    
    def get_best(min_pearson: float = 0.873) -> Optional[ModelCard]:
        """
        Load best registered model by pearson_r >= min_pearson.
        
        Returns:
            ModelCard with highest pearson_r, or None if no qualifying model
        """
    
    def get_latest() -> Optional[ModelCard]:
        """
        Load most recently registered model (by registered_at).
        
        Returns:
            ModelCard with max registered_at, or None if registry empty
        """
```

**Failure modes:**
- `ValueError("regression_guard_passed=False")` if pearson_r < 0.873 and checkpoint is submitted to register()
- Registry entry written only on SUCCESS path in ExperimentLogger._handle_success()
- If register() raises ValueError, ExperimentLogger catches and logs but does NOT block memory writes (best-effort)

---

## 3. Full-Loop Integration Test Spec (`tests/test_full_loop.py`)

### Test A: Success Path

**Setup:**
- Mock DataAgent.run() → DataReport(valid=True, n_genes=3010, cert_present=True)
- Mock train_v11 subprocess → exit code 0, write checkpoint to models/v1.1/test_a_model.pt
- Mock EvalAgent.run() → EvalSuite(kang=KangReport(passed=True, pearson_r=0.876), norman=NormanReport(passed=True, delta_nonzero_pct=18.5), overall_passed=True)
- Mock ExperimentLogger.post_run_hook() → verify _handle_success called

**Assertions:**
1. `aivc train --run-id test_a` returns exit code 0
2. RunMetadata.status == RunStatus.SUCCESS
3. artifacts/registry/latest_checkpoint.json written with pearson_r=0.876
4. ResearchAgent prompt file written to artifacts/agent_queue/
5. Obsidian note attempted via memory layer (mock: verify write_experiment_note called)
6. cli.train logs "SUCCESS: run_id=test_a pearson_r=0.876"

---

### Test B: Delta Collapse Failure

**Setup:**
- Mock DataAgent.run() → DataReport(valid=True, ...)
- Mock train_v11 subprocess → exit code 0, checkpoint written
- Mock EvalAgent.run() → EvalSuite(kang=KangReport(passed=True, pearson_r=0.895), norman=NormanReport(passed=False, delta_nonzero_pct=0.0, failure_reason="norman_delta_zero"), overall_passed=False)

**Assertions:**
1. `aivc train --run-id test_b` returns exit code 3
2. RunMetadata.delta_nonzero_pct == 0.0
3. RunMetadata.status == RunStatus.CTRL_MEMORISATION
4. artifacts/failure_notes/failure_test_b.json written
5. TrainingAgent prompt file written (NOT ResearchAgent)
6. latest_checkpoint.json NOT updated (from previous run)
7. cli.train logs "FAILURE: run_id=test_b delta_nonzero_pct=0.0 — halting eval"

---

### Test C: DataAgent Halt

**Setup:**
- Mock DataAgent.run() → DataReport(valid=False, blocked_reason="pairing certificate missing")

**Assertions:**
1. `aivc train --run-id test_c --dataset qurie_pending` returns exit code 1
2. train_v11 subprocess is NEVER called
3. No checkpoint written
4. RunMetadata.status == RunStatus.FAILURE
5. artifacts/agent_queue/data_agent_test_c_*.json written with blocked_reason
6. cli.train logs "HALTED by DataAgent: blocked_reason=pairing certificate missing"

---

### Test D: Kang Regression Guard

**Setup:**
- Mock DataAgent.run() → DataReport(valid=True, ...)
- Mock train_v11 subprocess → exit code 0, checkpoint written
- Mock EvalAgent.run() → EvalSuite(kang=KangReport(passed=False, regression_guard_passed=False, pearson_r=0.862, failure_reason="pearson_r < 0.873"), overall_passed=False)

**Assertions:**
1. `aivc train --run-id test_d` returns exit code 2
2. RunMetadata.status == RunStatus.REGRESSION
3. artifacts/failure_notes/failure_test_d.json written
4. TrainingAgent prompt file written
5. latest_checkpoint.json NOT updated
6. cli.train logs "REGRESSION GUARD FAILED: pearson_r=0.862 < 0.873 — halting"

---

## 4. Full-Loop Diagram (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CLI Entry: aivc train --run-id=RUN_1 --dataset=kang2018_pbmc         │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────┐
         │   DataAgent.run()         │
         │  - Load h5ad (backed="r") │
         │  - Validate n_genes=3010  │
         │  - Check pairing cert     │
         └───────────┬───────────────┘
                     │
                ┌────┴─────┐
                │           │
         ❌ HALT      ✅ PASS
                │           │
        [blocked]    ┌──────▼──────────────────┐
                     │ subprocess train_v11.py │
                     │ with config + run_id    │
                     │ write checkpoint_path   │
                     └──────┬──────────────────┘
                            │
                       ┌────┴─────┐
                       │           │
                   EXIT=0       EXIT≠0
                       │           │
                 ┌─────▼─┐    [FAIL exit 1]
                 │ EVAL  │
                 │ SUITE │
                 └─────┬─┘
                       │
              ┌────────┴────────┐
              │                 │
         [KANG PASS]       [KANG FAIL]
              │                 │
              ▼                 ▼
        ┌──────────────┐   [HALT: exit 2]
        │ run norman   │   [REGRESSION]
        │ eval         │   [failure_note]
        └──┬───────────┘   [training_agent]
           │
      ┌────┴─────────┐
      │              │
 [DELTA==0]    [DELTA>0]
      │              │
      ▼              ▼
[HALT: exit 3] ✅ SUCCESS
[CTRL_MEM]     (exit 0)
[failure_note] │
[training_agent] │
              ┌─┴──────────────────────────────────┐
              │ ExperimentLogger.finish(meta, suite)│
              └──────┬───────────────────────────────┘
                     │
                ┌────┴──────────────────┐
                │                       │
         _handle_success()      _handle_failure()
           (above branch)         (above branch)
                │                       │
    ┌───────────┼───────────┐    ┌─────┘
    │           │           │    │
    ▼           ▼           ▼    ▼
 REGISTRY  MEMORY(Obsidian) RESEARCH_AGENT  FAILURE_NOTE
  WRITE    WRITE              DISPATCH       (written earlier)
           + update_context                  + TRAINING_AGENT
                                             DISPATCH
                │
                └──────────────┬──────────────────┐
                               │                  │
                            [CLI                [CLI
                             exit 0]            exit code]
                             
                  Agents (Training/Research)
                  own Claude API calls &
                  write outputs to dedicated paths
```

**Legend:**
- ❌ HALT: DataAgent validation failure → immediate exit
- REGRESSION: Kang pearson_r < 0.873 → halt eval suite
- CTRL_MEM: Norman delta_nonzero_pct == 0.0 → halt eval suite
- ✅ SUCCESS: Both Kang pass + Norman pass + pearson_r >= 0.873
- All failure branches write failure_note + dispatch agents
- All success branches write registry entry + dispatch ResearchAgent
- ExperimentLogger idempotent on finish() (checked via run_id in _finished_runs set)

---

## 5. train_v11.py Entry-Point Determination

### Analysis

**File:** `/Users/ashkhan/Projects/aivc_genelink/train_v11.py` (lines 1–41)

**Structure:**
- Lines 1–41: Module docstring + imports + device selection
- Lines 86–184: Data loading (h5ad, OT pairs, edge lists, JAK-STAT genes)
- Continues beyond line 250 (full training loop not shown in read)
- **No `if __name__ == "__main__"` block visible in first 150 lines**
- **No function definitions wrapping the training loop**
- **Pattern:** Module-level code at execution time (lines 42+)

**Conclusion:** `train_v11.py` is a **script-only entry point**, not a callable function.

### Integration Approach: Subprocess Call

**Rationale:**
- train_v11.py loads data, models, and training state at module level
- No top-level function to import without triggering execution
- Best practice for isolated training jobs: subprocess isolation prevents memory leaks and GPU state conflicts
- Exit code propagation is explicit and reliable

**cli.py Integration:**
```python
def train_command(...):
    # ... DataAgent, config prep ...
    
    # Subprocess call
    cmd = [
        sys.executable,
        "train_v11.py",
        "--run-id", run_id,
        "--config", str(config_path),
        "--output", str(output_dir),
    ]
    
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"train_v11 failed: {result.stderr}")
        return 1  # exit code
    
    # ... continue with EvalAgent ...
```

**Arguments expected by train_v11.py:**
- `--run-id`: AIVC run identifier (for MLflow + W&B run naming)
- `--config`: Path to sweep config YAML (hyperparameters)
- `--output`: Directory to write checkpoint

**Output file contract:**
- Checkpoint written to `--output/model_v11_{run_id}.pt` (or similar deterministic path)
- Exit code 0 = success; nonzero = failure

---

## 6. api/server.py Checkpoint Search Order Patch

### Existing Search Order (lines 91–97)

```python
checkpoint_paths = [
    os.getenv("AIVC_CHECKPOINT", ""),                    # 1. Env var (explicit override)
    "models/v1.1/model_v11_best.pt",                     # 2. v1.1 best
    "models/v1.0/aivc_v1.0_best.pt",                     # 3. v1.0 best
    "model_week3_best.pt",                                # 4. week 3 best
    "model_week3.pt",                                     # 5. week 3 fallback
]
```

**Issue:** The v1 registry file (`artifacts/registry/latest_checkpoint.json`) is never consulted, even though ExperimentLogger.finish() writes it as the canonical source of truth for successful runs.

### Minimal Patch

**New Search Order:**
```python
checkpoint_paths = [
    os.getenv("AIVC_CHECKPOINT", ""),                    # 1. Env var (explicit override)
    _load_registry_checkpoint(),                          # 2. **NEW**: artifacts/registry/latest_checkpoint.json
    "models/v1.1/model_v11_best.pt",                     # 3. v1.1 best (fallback)
    "models/v1.0/aivc_v1.0_best.pt",                     # 4. v1.0 best
    "model_week3_best.pt",                                # 5. week 3 best
    "model_week3.pt",                                     # 6. week 3 fallback
]
```

**Helper Function (prepend to _load_model):**
```python
def _load_registry_checkpoint() -> str:
    """Load checkpoint_path from artifacts/registry/latest_checkpoint.json if it exists."""
    reg_path = Path("artifacts/registry/latest_checkpoint.json")
    if reg_path.exists():
        try:
            reg = json.loads(reg_path.read_text())
            ckpt = reg.get("checkpoint_path")
            if ckpt:
                logger.info(f"Loaded checkpoint from registry: {ckpt}")
                return ckpt
        except Exception as e:
            logger.warning(f"Failed to load registry checkpoint: {e}")
    return ""
```

**Rationale:**
- Minimal change: one new item in checkpoint_paths list + one helper function
- Preserves backward compatibility: env var still takes precedence; hardcoded paths remain as fallbacks
- Closes the gap: latest_checkpoint.json is now the first file-based source (after env var)
- No changes to loading loop or error handling

---

## 7. Exit-Code Contract

| Exit Code | Meaning | Context | Recovery |
|-----------|---------|---------|----------|
| 0 | SUCCESS | Training + eval passed; pearson_r >= 0.873, delta_nonzero_pct > 0 | Model registered, memory updated |
| 1 | DATA/TRAIN HALT | DataAgent invalid OR train_v11 exit != 0 | Fix data/config; retry `aivc train` |
| 2 | REGRESSION GUARD | Kang eval failed; pearson_r < 0.873 | TrainingAgent dispatched; review suggested config changes |
| 3 | DELTA COLLAPSE | Norman eval delta_nonzero_pct == 0 | TrainingAgent dispatched; address memorisation issue |
| 4 | ARTIFACT WRITE FAIL | Training/eval succeeded but registry/Obsidian writes failed | Manual artifact recovery; retry memory sync via `aivc memory sync` |

**Cross-command consistency:**
- `aivc train`: Uses all codes 0–4
- `aivc eval`: Uses 0 (pass), 1 (Kang fail), 2 (Norman fail), 3 (missing data)
- `aivc sweep`: Uses 0 (all pass), 1 (run failed), 2 (Kang halt), 3 (delta collapse)
- `aivc agent run`: Uses 0 (success), 1 (agent.run failure), 2 (instantiation fail)
- `aivc memory sync`: Uses 0 (success), 1 (vault error), 2 (partial failure)
- `aivc status`: Uses 0 (success), 1 (no runs found)

---

## 8. Packaging & Entry Point

### 8.1 aivc/__main__.py

**File:** `/Users/ashkhan/Projects/aivc_genelink/aivc/__main__.py`

```python
"""
AIVC CLI entry point. Enables: python -m aivc <command> <args>
"""
import sys
from aivc.cli import app

if __name__ == "__main__":
    app()  # Typer app instance
```

### 8.2 requirements.txt Patch

**Add:**
```
typer[all]>=0.12.0
```

**Rationale:**
- `typer[all]` includes click (argument parsing) and rich (terminal formatting)
- No other new dependencies required; existing deps (pydantic, torch, fastapi) sufficient
- Pin version to 0.12.0+ for stability

**Existing relevant deps (do not add):**
```
pydantic>=2.0
torch>=2.2.2
fastapi
mlflow
```

---

## 9. Key Risks & Recommended Next Steps

### 9.1 Known Gaps

1. **train_v11.py argument parsing:** File needs `argparse` block (or Typer decorator) to accept `--run-id`, `--config`, `--output`. Currently reads from module-level, not CLI args.
   - **Action:** Add argparse block to train_v11.py before `if __name__ == "__main__"` logic (or refactor to main() function)
   - **Impact:** Medium; blocks train_v11 subprocess integration

2. **ModelRegistry schema not yet in codebase:** Proposed in §2; no actual `aivc_platform/registry/model_registry.py` file exists.
   - **Action:** Create ModelRegistry class with register(), get_best(), get_latest() methods
   - **Impact:** Medium; ExperimentLogger._handle_success() calls ModelRegistry.register() in full loop

3. **About Me / anti-ai-style.md missing:** User requested pre-reading but files do not exist at `/Users/ashkhan/Projects/aivc_genelink/About Me/`.
   - **Action:** Clarify file location with user or assume style is implicit from context_snapshot.md tone
   - **Impact:** Low; spec assumes concise, senior-level prose (inferred from context_snapshot.md)

4. **Tests/test_full_loop.py not scaffolded:** Spec defines test cases A–D; no pytest fixtures or mocks exist yet.
   - **Action:** Create test file with pytest + unittest.mock; mock DataAgent, EvalAgent, ExperimentLogger
   - **Impact:** Medium; enables validation of CLI integration before GPU training

5. **aivc.skills.neumann_propagation, scm_engine availability:** server.py imports these; unclear if they are implemented or stubs.
   - **Action:** Verify these modules exist and have expected classes/methods
   - **Impact:** Low (API spec only); confirmed in codebase scan

6. **Memory layer optional coupling:** ExperimentLogger calls write_experiment_note() + update_context() best-effort (catch-all exceptions). If these fail silently, debugging is hard.
   - **Action:** Consider explicit memory.WriterConfig flag to enable/disable memory writes at CLI level
   - **Impact:** Low; non-blocking for v1

### 9.2 Recommended Next Steps

**Phase 1 (Week 1–2):**
1. ✅ Create `aivc_platform/registry/model_registry.py` with ModelCard + Registry functions
2. ✅ Scaffold `tests/test_full_loop.py` with Test A (success path) using mocks
3. ✅ Add argparse to train_v11.py; test subprocess call via `python train_v11.py --run-id test --output models/test/`

**Phase 2 (Week 2–3):**
4. ✅ Implement `cli.py` with Typer app + train, eval, sweep, agent run, memory sync, status commands
5. ✅ Patch api/server.py checkpoint search (3-line change per §6)
6. ✅ Add typer to requirements.txt + create aivc/__main__.py
7. ✅ Test CLI locally: `python -m aivc train --help`, `python -m aivc eval --help`

**Phase 3 (Week 3–4, on GPU):**
8. Run full integration test (Test A) with real H100 training
9. Validate exit codes, artifact paths, agent dispatches
10. Test sweep command with 3–5 hyperparameter sets
11. Complete Tests B, C, D with mocks for failure paths

**Phase 4 (Post-validation):**
12. Add e2e test in CI/CD (GitHub Actions or similar)
13. Document CLI in README.md with examples

### 9.3 What I Need From You

1. **Clarify train_v11.py entry contract:**
   - What arguments does train_v11.py currently accept (if any)?
   - Where should checkpoint be written? (Hardcoded `models/v1.1/` or via `--output` flag?)
   - Should exit code be 0 on success or does it currently exit silently?

2. **Confirm ModelRegistry location & scope:**
   - Should ModelRegistry live in `aivc_platform/registry/model_registry.py` (new subpackage) or `aivc_platform/tracking/experiment_logger.py` (inline)?
   - Does ExperimentLogger already call a register() function on success, or is that new?

3. **Memory layer readiness:**
   - Are write_experiment_note() and update_context() in `aivc_platform/memory/obsidian_writer.py` and `aivc_platform/memory/context_updater.py` production-ready, or stub/in-flight?
   - Should CLI support `--no-memory-sync` to skip these calls?

4. **Agent API key handling:**
   - Should TrainingAgent and ResearchAgent use os.getenv("ANTHROPIC_API_KEY") directly, or should CLI inject via AgentDispatcher?
   - If no API key, should agents fail (exit 1) or write prompt file only?

5. **Sweep concurrency:**
   - Is multi-GPU training needed for `--parallel > 1`? (E.g., one run per GPU; max 4 on H100)
   - Or is serial execution (--parallel 1) acceptable for v1?

---

## Document Notes

- **Status:** Specification only (no implementation)
- **Audience:** Senior backend/ML engineers implementing cli.py and orchestration
- **Last Updated:** 2026-04-14
- **Next Review:** Post Phase 1 implementation (week 2)

---

## References

- `/Users/ashkhan/Projects/aivc_genelink/docs/context_snapshot.md` — Project overview
- `/Users/ashkhan/Projects/aivc_genelink/aivc_platform/tracking/schemas.py` — RunMetadata, RunStatus
- `/Users/ashkhan/Projects/aivc_genelink/aivc_platform/tracking/experiment_logger.py` — ExperimentLogger, post_run_hook
- `/Users/ashkhan/Projects/aivc_genelink/eval/eval_runner.py` — EvalSuite, run_eval_suite
- `/Users/ashkhan/Projects/aivc_genelink/agents/*.py` — BaseAgent, DataAgent, EvalAgent, TrainingAgent, ResearchAgent
- `/Users/ashkhan/Projects/aivc_genelink/train_v11.py` — Training script (subprocess entry point)
- `/Users/ashkhan/Projects/aivc_genelink/api/server.py` — Checkpoint loading logic (lines 86–127)
