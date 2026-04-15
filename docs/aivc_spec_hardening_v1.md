# AIVC Hardening Specification v1

**Status:** design only — no code in this doc
**Audience:** self / engineers wiring the reproducibility + safety layer
**Scope:** six new `scripts/*.py` modules, two config files, Makefile targets, CLI extensions, and hook points into existing `cli.py` / `ExperimentLogger.finish`
**Mode:** architect

---

## 0. Purpose & Non-Goals

### Purpose
Close the reproducibility, regression, and agent-safety gaps that the current `cli.py` + `ExperimentLogger` loop leaves open:

- runs are not signed (no single artifact binds config + data + code + env + metrics)
- regression logic is binary-ish (`pearson_r < 0.873` only) — no margin, no best-tracking, no promotion rule
- `context_snapshot.md` is written by memory layer best-effort; no validator, no atomicity, no freshness contract
- agent proposals land straight in `artifacts/agent_queue/` with no schema / duplicate / boundary / loop check
- success is never committed; experiment history is not reconstructable from git alone

### Non-Goals
- No orchestration system (Prefect / Airflow). D3: single-box cron + Makefile only.
- No new DB / service. Everything is files, committed alongside the repo.
- No rewrite of `ExperimentLogger`, `EvalSuite`, agents, or registry. This layer wraps them.
- No auto-push. `commit_run` never pushes; remotes are manual.

---

## 1. Confirmed Decisions (binding)

**D1 — Validation metrics**
- Primary: `pearson_r`, regression tolerance ε = 0.005
- Secondary: `delta_nonzero_pct`, floor = 1.0 %
- SUCCESS ⇔ `pearson_r ≥ 0.873 AND pearson_r ≥ best_pearson_r − 0.005 AND delta_nonzero_pct > 1.0`

**D2 — Dataset versioning: DVC**
- `data/*.h5ad` tracked via DVC; `.dvc` files committed, `*.h5ad` in `.gitignore`.
- SHA-256 of each `.h5ad` recorded in `signature.json` (independent of DVC hash; DVC path stored too).

**D3 — Orchestration: cron + Makefile**
- All hooks run on the training box. `hooks.py` is the dispatcher, reused by `cli.py` and Makefile.

**D4 — `context.md` validator reads the real file**
- `validate_context_md` infers required sections from `docs/context_snapshot.md` contents, not from the hardening plan doc.
- Sections present in the live snapshot (2026-04-10): `Project`, `Last updated`, `Mission`, `Stack snapshot`, `Architecture pattern`, `Key files / entry points`, `Active APIs`, `Models / ML components`, `Data sources`, `Open questions`, `Do not touch`, `GitHub`.

---

## 2. Architecture

### 2.1 Layering
```
┌───────────────────────────────────────────────────────────────┐
│  cli.py (Typer)                        Makefile (cron-safe)    │
│  aivc train | validate-run |           make train | validate-  │
│  sync-context | repro                  run | sync-context | …  │
└───────────────────────────┬───────────────────────────────────┘
                            │           (both call ↓)
                    ┌───────▼────────┐
                    │  scripts/hooks │   single dispatcher
                    │  .py           │   pre_/post_ train + agent
                    └───────┬────────┘
        ┌──────────┬────────┼────────┬───────────┬────────────┐
        ▼          ▼        ▼        ▼           ▼            ▼
 build_signature validate sync    commit     validate    (future)
    .py          _run.py  _context _run.py   _proposal
                          .py               .py
        │          │        │        │           │
        │          │        │        │           │
        ▼          ▼        ▼        ▼           ▼
 experiments/  experiments/ docs/   git       configs/
 {run_id}/     best/        context experiments agent_
 signature.json signature   _snapshot /auto     constraints.yaml
 experiments/  .json        .md      branch    experiments/
 index.jsonl                                   index.jsonl (read)
```

### 2.2 Filesystem contract
```
experiments/
  index.jsonl                          # append-only, 1 line per run
  {run_id}/
    signature.json                     # RunSignature
    config.yaml                        # verbatim snapshot of --config
    eval_suite.json                    # EvalSuite dump
    failure.json                       # only if REGRESSION/FAILURE
  best/
    best.pt -> ../{run_id}/…/best.pt   # symlink, atomic via ln -sfn
    signature.json                     # mirror of winning run's signature

configs/
  thresholds.yaml
  agent_constraints.yaml

docs/
  context_snapshot.md                  # frontmatter: snapshot_sha, snapshot_ts, synced_from_run
context.md                             # working doc authored by memory layer

.context.lock                          # fcntl lock, repo root
```

### 2.3 Integration points (existing code, one-line deltas)
- `cli.py train` (line ~96 and after `ExperimentLogger().finish(...)` at ~239): call `hooks.pre_train()` before the `DataAgent` gate is fine, and `hooks.post_train(meta, suite)` immediately after `ExperimentLogger.finish` on the success branch AND on both failure branches (exits 2/3). `hooks.post_train` is idempotent via `(run_id, status)` dedupe against `index.jsonl`.
- `cli.py agent run` (`agent_run`): call `hooks.pre_agent(agent, run_id)` before `agents_map[agent]()` instantiation; `hooks.post_agent(agent, run_id, result)` before `typer.echo(result.model_dump_json(...))`.
- `ExperimentLogger.finish` is **not** modified. Hook calls live at the CLI layer so standalone `python train_v11.py` runs still work (unhooked path, acceptable).

---

## 3. Component Specs

All components obey:
- standalone CLI (`python -m scripts.X …`) AND importable by `hooks.py`
- atomic writes (`tempfile.NamedTemporaryFile` in same dir + `os.replace`)
- `fcntl.flock` with 30 s timeout, never blocks indefinitely
- no new deps except `dvc` (added to `requirements.txt`)
- fail-soft on non-critical paths; `exit 1` only where the spec says "abort"

### 3.1 `scripts/build_signature.py`

Assembles the signed record of a run. One write per run.

**Schema — `RunSignature` (Pydantic v2):**
```
schema_version: Literal["aivc/run_signature/v1"]
run_id: str
timestamp_utc: datetime
git:
  commit: str                 # 7-char
  branch: str
  dirty: bool
  diff_sha: str | None        # sha256(git diff) if dirty, else None
config:
  path: str                   # original --config path
  sha256: str                 # of the YAML bytes
  snapshot_path: str          # experiments/{run_id}/config.yaml
dataset:
  name: str                   # e.g. kang2018_pbmc
  version: str                # free-form; default = dvc rev or mtime iso
  sha256: str                 # of the .h5ad file bytes
  dvc_path: str | None        # "data/kang2018_pbmc_fixed.h5ad.dvc" if tracked
environment:
  python: str                 # sys.version.split()[0]
  cuda: str | None            # torch.version.cuda
  torch: str                  # torch.__version__
  pip_freeze_sha: str         # sha256(pip freeze stdout)
model:
  checkpoint_path: str
  checkpoint_sha256: str
  architecture: Literal["GeneLink-GAT-Neumann-v1.1"]
metrics:
  pearson_r: float
  delta_nonzero_pct: float
  ctrl_memorisation_score: float
context_snapshot_sha: str     # sha256(docs/context_snapshot.md)
status: Literal["SUCCESS", "REGRESSION", "FAILURE"]
```

**Functions:**
```
build_signature(meta: RunMetadata, suite: EvalSuite) -> RunSignature
save_signature(sig: RunSignature) -> Path
    # atomic write experiments/{run_id}/signature.json
    # atomic append one JSON line to experiments/index.jsonl (flock)
load_signature(run_id: str) -> RunSignature | None
get_best_signature() -> RunSignature | None
    # stream index.jsonl, return entry with max pearson_r where status == SUCCESS
```

**Atomicity:** `index.jsonl` append uses flock + write + fsync. If fsync fails → raise; caller decides.

**Failure modes:**
- `.h5ad` missing → `FileNotFoundError`, caller logs and continues without signature (run is unsigned; `post_train` then classifies as `FAILURE`).
- `git` unavailable → `commit="unknown", dirty=True, diff_sha=None`. Do not abort.
- `pip freeze` > 5 s → timeout, store `pip_freeze_sha="timeout"`.

---

### 3.2 `scripts/validate_run.py`

Single gate between "run finished" and "we act on the result".

**Schema — `ValidationResult`:**
```
status: RunStatus                # SUCCESS | REGRESSION | FAILURE
run_id: str
pearson_r: float
delta_nonzero_pct: float
best_pearson_r: float | None
regression_detected: bool
failure_reason: str | None
promote_checkpoint: bool
```

**Classification (from `configs/thresholds.yaml`, D1):**
```
SUCCESS:
    pearson_r >= PEARSON_R_FLOOR (0.873)
 AND (best is None OR pearson_r >= best - REGRESSION_TOLERANCE)
 AND delta_nonzero_pct > DELTA_NONZERO_FLOOR (1.0)
 AND suite.kang.regression_guard_passed

REGRESSION (not SUCCESS):
    best is not None AND pearson_r < best - REGRESSION_TOLERANCE
    → failure_reason = "pearson_regression"

FAILURE (everything else):
    NaN in any metric                → "nan_metrics"
    not suite.kang.regression_guard_passed → "kang_guard_failed"
    delta_nonzero_pct <= DELTA_NONZERO_FLOOR → "delta_collapse"
    pearson_r < PEARSON_R_FLOOR      → "below_floor"
```

**Promotion rule:**
```
promote_checkpoint = (
    status == SUCCESS
    AND (best is None OR pearson_r > best + PROMOTION_MARGIN (0.001))
)
```

**Functions:**
```
classify_run(meta, suite, best_sig) -> ValidationResult
promote_checkpoint(checkpoint_path: str) -> None
    # 1. mkdir -p experiments/best
    # 2. os.symlink(tmp); os.replace(tmp, experiments/best/best.pt)   # atomic
    # 3. write experiments/best/signature.json (atomic)
```

**Note:** intentional overlap with existing `RunStatus` enum in `aivc_platform/tracking/schemas.py` — `validate_run` returns its own `RunStatus` from `configs/thresholds.yaml`, and hooks reconcile by writing both to `signature.json.status` and `meta.status`. The CTRL_MEMORISATION subtype from the existing enum is not used by this layer; `delta_collapse` failure covers it.

---

### 3.3 `scripts/sync_context.py`

Atomic `context.md` → `docs/context_snapshot.md` sync with freshness check.

**`validate_context_md(path) -> tuple[bool, str | None]`**
- Read `docs/context_snapshot.md` once to discover required section headers (Markdown lines starting with `###`). Cache result per-process.
- Required sections (from the 2026-04-10 snapshot, D4):
  `Project`, `Mission`, `Stack snapshot`, `Architecture pattern`,
  `Key files / entry points`, `Active APIs`, `Models / ML components`,
  `Data sources`, `Open questions`, `Do not touch`, `GitHub`
  plus top-level fields: `Last updated:` (parseable `YYYY-MM-DD`) and a `best pearson_r` float somewhere in the file (regex `r"best pearson_r[:\s]+([0-9.]+)"`, tolerated absent if no SUCCESS run yet).
- Return `(False, reason)` on any missing piece; otherwise `(True, None)`.

**`sync_context(context_path, snapshot_path="docs/context_snapshot.md", lock_path=".context.lock") -> bool`**
```
1. acquire flock(lock_path, timeout=30s)             — on timeout: return False, log
2. ok, reason = validate_context_md(context_path)    — if not ok: release, return False
3. sha = sha256(context_path bytes)
4. body = read(context_path)
   prepend frontmatter:
     ---
     snapshot_sha: {sha}
     snapshot_ts: {utc_iso}
     synced_from_run: {os.environ.get("AIVC_RUN_ID", "manual")}
     ---
5. write tmp in same dir; os.replace(tmp, snapshot_path)
6. release lock; return True
```

**`verify_snapshot_fresh(snapshot_path, context_path, max_age_hours=24) -> bool`**
- Parse frontmatter; compare `snapshot_sha` to current `sha256(context_path)` — stale if mismatch.
- Stale if `(now - snapshot_ts) > max_age_hours`.

**Concurrency contract:** `sync_context` is the **only** writer of `docs/context_snapshot.md`. Memory layer's `update_context` continues to write `context.md` (working doc); the snapshot is downstream.

---

### 3.4 `scripts/commit_run.py`

Auto-commit on SUCCESS (and optionally REGRESSION, with flag).

**Commit message (machine-parseable):**
```
run({run_id}): {one-line summary, e.g. "pearson_r=0.8762 +0.0032 over best"}

run_id: {run_id}
pearson_r: {value:.4f}
delta_nonzero_pct: {value:.2f}
dataset: {name} (sha256:{sha[:12]})
config: {path}
status: {SUCCESS|REGRESSION}
wandb: {url or "-"}
```

**Staging whitelist** (nothing else is staged, even if dirty):
```
configs/
context.md
docs/context_snapshot.md
experiments/{run_id}/
experiments/index.jsonl
experiments/best/signature.json           # symlink target change is a git mv
scripts/
.gitignore
```

**Staging blacklist** (explicitly never):
```
*.pt   *.pth   *.ckpt
*.h5ad *.h5
wandb/   mlruns/   artifacts/
```
Enforced by an explicit `git reset HEAD -- <pattern>` after `git add`.

**Functions:**
```
commit_run(sig: RunSignature, branch: str = "experiments/auto") -> str | None
    # 1. ensure_experiments_branch()
    # 2. git add --intent selected paths
    # 3. git reset for blacklist
    # 4. if `git diff --cached --stat` is only a timestamp bump in context_snapshot.md
    #    (snapshot_sha unchanged): skip commit, return None
    # 5. git -c user.name="Ash Khan" -c user.email="a.khan@quriegen.com" commit …
    # 6. return sha
ensure_experiments_branch() -> None
    # create experiments/auto from current HEAD if missing; checkout; never push
```

**Never-push invariant:** no `git push` anywhere in this module. Push is manual.

---

### 3.5 `scripts/validate_proposal.py`

Safety gate between an agent proposal and any downstream `aivc train --config`.

**Input contract** (matches current `AgentResult.extra` shape + proposal fields agents already emit):
```
{
  "hypothesis": str,
  "config_diff": dict,              # e.g. {"neumann_k": 4, "lambda_l1": 0.02}
  "expected_metric_delta": {        # required, used for loop detection & docs
     "pearson_r": float,
     "delta_nonzero_pct": float
  },
  "proposal_id": str | None         # auto-generated if absent
}
```

**Schema — `ProposalValidation`:**
```
approved: bool
proposal_id: str
rejection_reason: str | None
duplicate_of: str | None            # run_id
boundary_violations: list[str]
```

**Checks (in order; first failure sets `approved=False` and returns):**

1. **Schema**: `hypothesis`, `config_diff`, `expected_metric_delta` present and non-empty.
2. **Duplicate**: `h = sha256(canonical_json(config_diff))`. Scan last 50 entries in `experiments/index.jsonl`; if any past run's `config_diff_sha` matches, set `duplicate_of = past.run_id`, reject.
3. **Boundary** against `configs/agent_constraints.yaml` (see §4):
   - `w_scale` max <= 0.5
   - `neumann_k` in `[1, 6]`
   - `lfc_beta` in `[0.0, 2.0]`
   - `lambda_l1` in `[0.0, 1.0]`
   - forbidden keys: `frozen_modules` mutation that removes `genelink`, `n_genes`, `edge_index`
   Every violation appended to `boundary_violations`. Any non-empty list → reject.
4. **Regression loop**: read last 3 completed runs from `index.jsonl`. If all 3 `status == REGRESSION` against the same `best_pearson_r` baseline → reject with `rejection_reason = "regression_loop_detected"` and set env flag `AIVC_FORCE_DIAGNOSTIC=1` on the returned `extra` for the next run.

**Function:**
```
validate_proposal(proposal: dict) -> ProposalValidation
```

Note: this module **reads** `experiments/index.jsonl`. For that read to be useful, `build_signature.save_signature` must include `config_diff_sha` for runs that were themselves agent-originated. Add that field opportunistically (nullable) — omitted from §3.1 schema to keep it minimal but computed and stored when `AIVC_PROPOSAL_ID` env var is set by `cli agent run`.

---

### 3.6 `scripts/hooks.py`

Central dispatcher. Single source of truth for ordering.

**`pre_train()`**
```
1. verify_snapshot_fresh(docs/context_snapshot.md, context.md, max_age_hours=24)
   if stale: log WARN (do not abort — training itself doesn't consume snapshot)
2. check_env():
   - torch.cuda.is_available() (warn only)
   - ANTHROPIC_API_KEY / WANDB_API_KEY presence (warn only)
   - dataset h5ad at expected path (ABORT if missing, exit 1)
```

**`post_train(meta, suite)` — idempotent per `run_id`:**
```
1. sig = build_signature(meta, suite)
2. result = classify_run(meta, suite, best_sig=get_best_signature())
3. sig.status = result.status; save_signature(sig)
4. if result.status == SUCCESS:
     if result.promote_checkpoint:
         promote_checkpoint(meta.checkpoint_path)
     sync_context(context_path="context.md")          # logs on failure, does not abort
     commit_run(sig)                                   # logs on failure, does not abort
     # Obsidian experiment note: delegated to existing memory layer (no change)
6. elif result.status in {REGRESSION, FAILURE}:
     write experiments/{run_id}/failure.json with result.dict()
     if result.status == FAILURE:
         quarantine: mv meta.checkpoint_path → experiments/{run_id}/quarantine.pt
```

**`pre_agent(agent, run_id)`:**
```
1. fresh = verify_snapshot_fresh(docs/context_snapshot.md, context.md, 24h)
2. if not fresh:
     write artifacts/failure_notes/stale_context_{run_id}.json
     log ERROR "stale_context"
     sys.exit(1)                                       # hard abort
```

**`post_agent(agent, run_id, result)`:**
```
1. if agent not in {"training_agent", "research_agent"}: return
2. proposal = result.extra.get("proposal")
   if not proposal: return                             # agent produced no actionable proposal
3. pv = validate_proposal(proposal)
4. if not pv.approved:
     write artifacts/agent_queue/rejected_{run_id}_{pv.proposal_id}.json
     log WARN
     sys.exit(1)
5. else:
     write artifacts/agent_queue/approved_{run_id}_{pv.proposal_id}.json
```

**All hooks:** wrap each sub-step in `try/except` and log; only the abort points called out above `sys.exit(1)`. Never raise into `cli.py`.

**Single-call invariant:** `sync_context` and `commit_run` are invoked **only** from `hooks.post_train`. Anywhere else that mutates `context_snapshot.md` or stages commits is a bug.

---

## 4. Config files

### 4.1 `configs/thresholds.yaml`
```yaml
# Regression + promotion thresholds. Consumed by scripts/validate_run.py.
pearson_r_floor: 0.873
pearson_r_regression_tolerance: 0.005
delta_nonzero_pct_floor: 1.0
delta_nonzero_regression_tolerance: 0.5   # reserved; not in D1 gate
best_checkpoint_promotion_margin: 0.001
```

### 4.2 `configs/agent_constraints.yaml`
```yaml
# Hyperparameter bounds for agent proposals. Consumed by scripts/validate_proposal.py.
allowed_hyperparams:
  w_scale:
    min: 0.0
    max: 0.5
  neumann_k:
    min: 1
    max: 6
  lfc_beta:
    min: 0.0
    max: 2.0
  lambda_l1:
    min: 0.0
    max: 1.0
forbidden:
  - unfreeze_genelink        # frozen_modules must contain "genelink"
  - change_n_genes           # n_genes must stay 3010
  - change_edge_index        # STRING PPI edge list is frozen
regression_loop:
  window: 3                  # last N runs
  action: force_diagnostic   # sets AIVC_FORCE_DIAGNOSTIC=1
```

---

## 5. Makefile (cron-safe)

```
.PHONY: train validate-run sync-context commit-run agent-propose status repro

train:
	python cli.py train --config $(CONFIG) --run-id $(RUN_ID)

validate-run:
	python -m scripts.validate_run --run-id $(RUN_ID)

sync-context:
	python -m scripts.sync_context

commit-run:
	python -m scripts.commit_run --run-id $(RUN_ID)

agent-propose:
	python cli.py agent run --agent training_agent --run-id $(RUN_ID) --payload @$(PAYLOAD)

status:
	python cli.py status

repro:
	python cli.py repro --run-id $(RUN_ID)
```

Cron example (daily sweep, same box):
```
0 2 * * *  cd /repo && make train CONFIG=configs/sweep_w_scale.yaml RUN_ID=nightly_$(date +\%s) >> logs/cron.log 2>&1
```

---

## 6. CLI extensions (`cli.py`)

Three new commands; all are thin wrappers over `scripts/*`.

```
@app.command("validate-run")
def validate_run(run_id: str = typer.Option(...)):
    """Re-run classify_run for a past run_id using index.jsonl + suite.json."""

@app.command("sync-context")
def sync_context_cmd():
    """Manually sync context.md → docs/context_snapshot.md."""

@app.command("repro")
def repro(run_id: str = typer.Option(...)):
    """
    Reproduce a past run:
      1. read experiments/{run_id}/signature.json
      2. git checkout sig.git.commit (warn if dirty)
      3. dvc pull sig.dataset.dvc_path
      4. exec `python cli.py train --config experiments/{run_id}/config.yaml --run-id repro_{run_id}`
    """
```

### Hook wiring in existing `cli.py train` (minimal diff)

- Line after `run_id = run_id or str(uuid.uuid4())[:8]`:
  `hooks.pre_train()`
- After the SUCCESS memory block (before `typer.echo("SUCCESS: …")`):
  `hooks.post_train(meta, suite)`
- In both failure branches (before `raise typer.Exit(2)` and `raise typer.Exit(3)`):
  `hooks.post_train(meta, suite)`   # classifies as REGRESSION/FAILURE, writes failure.json, no promote, no commit

### Hook wiring in `agent_run`
- After `task = AgentTask(...)` and before `result = instance.run(task)`:
  `hooks.pre_agent(agent, run_id)`
- After `result = instance.run(task)`:
  `hooks.post_agent(agent, run_id, result)`

Idempotency: `hooks.post_train` first checks `index.jsonl` for `run_id`; if present, skip. That guards against double-wiring via `ExperimentLogger.finish` path and the CLI success block.

---

## 7. Atomicity, locking, failure semantics

| Resource                           | Writer             | Mechanism                        |
|------------------------------------|--------------------|----------------------------------|
| `experiments/{run_id}/signature.json` | `build_signature`   | tmp + `os.replace`               |
| `experiments/index.jsonl`          | `build_signature`   | `fcntl.flock` 30 s + append+fsync |
| `experiments/best/best.pt`         | `promote_checkpoint`| tmp symlink + `os.replace`       |
| `docs/context_snapshot.md`         | `sync_context`      | flock `.context.lock` 30 s + tmp + `os.replace` |
| git tree                           | `commit_run`        | `git add` whitelist + `git reset` blacklist; no push |
| `artifacts/agent_queue/*`          | `post_agent`        | tmp + `os.replace`               |

**Lock timeout = 30 s.** On timeout, the writer returns `False` / logs and moves on. No operation blocks indefinitely.

**Quarantine on FAILURE:** checkpoint is moved, not deleted. Recoverable if a future run refs it by sha.

---

## 8. requirements.txt delta

Add (nothing else):
```
dvc>=3.0
```
`pydantic`, `typer`, `torch`, `PyYAML` are already present via existing spec.

---

## 9. Status matrix (decision routing summary)

| Pearson r vs best / floor                    | delta_nz | kang_guard | status     | promote | commit | dispatch agent  |
|-----------------------------------------------|----------|------------|------------|---------|--------|-----------------|
| `>= floor ∧ >= best − ε ∧ > best + 0.001`      | `> 1.0`  | ✓          | SUCCESS    | ✓       | ✓      | research_agent  |
| `>= floor ∧ >= best − ε ∧ ≤ best + 0.001`      | `> 1.0`  | ✓          | SUCCESS    | –       | ✓      | research_agent  |
| `>= floor ∧ < best − ε`                        | any      | ✓          | REGRESSION | –       | –      | training_agent  |
| `< floor`                                      | any      | any        | FAILURE    | –       | –      | training_agent  |
| any                                           | `≤ 1.0`  | ✓          | FAILURE    | –       | –      | training_agent  |
| any                                           | any      | ✗          | FAILURE    | –       | –      | training_agent  |
| metrics NaN                                   | –        | –          | FAILURE    | –       | –      | training_agent  |

Dispatch is already owned by `ExperimentLogger.finish`; this layer only *gates* whether promote/commit happen and writes `signature.json`.

---

## 10. Test plan (minimal)

`tests/hardening/`:
- `test_build_signature.py` — dirty repo yields `diff_sha`; clean repo `diff_sha=None`; missing h5ad raises.
- `test_validate_run.py` — matrix from §9, one case per row, using fixture `best_sig`.
- `test_sync_context.py` — stale detection (sha mismatch; ts > 24 h); lock contention yields `False` within 30 s.
- `test_commit_run.py` — blacklist enforcement (staging a `.pt` in a dirty tree is rejected); timestamp-only bump is skipped.
- `test_validate_proposal.py` — duplicate detection against fixture `index.jsonl`; boundary violations; regression-loop trigger.
- `test_hooks_idempotency.py` — double-call of `post_train` writes one `index.jsonl` entry.

No GPU required; all fixtures use tiny fake `EvalSuite` + `RunMetadata`.

---

## Key Risks

- **`RunStatus` enum drift.** `aivc_platform/tracking/schemas.py` already defines SUCCESS/FAILURE/CTRL_MEMORISATION/REGRESSION. `validate_run` re-classifies using D1 floors, which may disagree with the existing `ExperimentLogger.post_run_hook` thresholds (`PEARSON_R_THRESHOLD=0.873`, `DELTA_NONZERO_THRESHOLD=0.0`). Disagreement is harmless today because `post_run_hook` only drives agent dispatch, not promotion — but the next person will trip on this. **Mitigation:** single docstring in both files pointing at `configs/thresholds.yaml` as the source of truth; optional follow-up to make `ExperimentLogger` read the YAML too.
- **Snapshot freshness blocks agents hard (`pre_agent` exits 1).** If `context.md` is edited by hand without a sync, `aivc agent run` dies. Mitigation: `aivc sync-context` is a one-liner documented in the rejection note.
- **Duplicate detection is config-only.** Two runs with identical `config_diff` on different dataset versions hash the same. Accept this for now (single-dataset regime) and extend the hash to include `dataset.sha256` once QuRIE-seq lands.
- **DVC assumed installed on cron box.** If `dvc` is not on `PATH`, `build_signature` silently sets `dvc_path=None` and the repro command degrades to "checkout + hope". Fail-louder in `repro` if `dvc_path` is None.
- **`commit_run` on a dirty working tree.** Whitelist/blacklist reduces blast radius but cannot prevent a user from having an unrelated staged change. Mitigation: `commit_run` aborts (does not reset) if `git diff --cached` contains paths outside the whitelist at entry.
- **No guard on concurrent `post_train` from two CLI invocations.** `index.jsonl` append is flocked, but `promote_checkpoint` is not globally locked. Two near-simultaneous SUCCESS runs can race the symlink. Acceptable on single-box D3; revisit if we ever parallelise sweeps.

## Recommended Next Step

Implement in this order, each a self-contained PR on `hardening/*` branches:
1. `configs/thresholds.yaml`, `configs/agent_constraints.yaml`, `requirements.txt` bump.
2. `scripts/build_signature.py` + `tests/hardening/test_build_signature.py`. Wire into `ExperimentLogger.finish` via a one-line optional call gated by `AIVC_SIGN=1` env so it can be rolled out dark.
3. `scripts/validate_run.py` + tests; expose `aivc validate-run`.
4. `scripts/sync_context.py` + tests; expose `aivc sync-context`. Manual-only first.
5. `scripts/hooks.py` with **only** `pre_train` / `post_train` wired. Leave agent hooks stubbed.
6. `scripts/commit_run.py` behind `AIVC_AUTO_COMMIT=1` for one week before turning on by default.
7. `scripts/validate_proposal.py` + `pre_agent` / `post_agent` wiring.
8. `aivc repro` last — requires (1)–(6) to be trustworthy.

## What I need from you

1. Confirm the commit author identity (`Ash Khan <a.khan@quriegen.com>`) vs. the about-me email (`kinga@quriegen.com`) — which should `commit_run` use?
2. Confirm `experiments/` path (not `runs/` or `models/…`) — this is a new top-level directory; OK to add to repo root?
3. `context.md` working doc — does this file exist today, or is the memory layer's `update_context` writing to `docs/context_snapshot.md` directly? If the latter, we need a one-time split before `sync_context` is useful.
4. Do you want `REGRESSION` runs committed (with `AIVC_COMMIT_REGRESSIONS=1`) so the history is complete, or strictly SUCCESS-only in `experiments/auto`?
