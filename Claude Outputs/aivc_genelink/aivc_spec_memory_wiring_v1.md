# AIVC Memory Wiring — Design Spec v1

**Author:** Architect Mode · **Date:** 2026-04-14
**Scope:** `aivc_platform/memory/` — Obsidian note writer, context.md patcher, vault initialiser, and wiring into `ExperimentLogger.post_run_hook`.

---

## 0. Critical gaps & schema divergences (read first)

The task prompt lists "locked schemas" that do **not match** the schemas currently in `aivc_platform/tracking/schemas.py`. Per the rule "read from schemas.py", this spec is authored against the **on-disk** schemas and every divergence is called out below. Do not paper over these — they determine correctness of wiring.

| Prompt-declared "locked" field | Actual `schemas.py` field | Resolution in this spec |
| --- | --- | --- |
| `RunMetadata.modalities` (plural) | `RunMetadata.modality: Modality` (singular enum) | Spec uses `meta.modality.value`. Template renders a single modality. If plural is truly intended, schema change is a prerequisite (out of scope here). |
| `RunMetadata.mlflow_run_id` | **not present** | Spec renders `mlflow_run_id` as `"n/a (file store, no registry)"` — matches the actual MLflow file-backend behaviour. Do **not** add this field unless registry is enabled. |
| `PostRunDecision` as a BaseModel with `should_register / should_trigger_training_agent / should_trigger_research_agent / failure_note_path / reason` | `PostRunDecision` is an `Enum` (`REGISTER`, `TRIGGER_TRAINING_AGENT`, `TRIGGER_RESEARCH_AGENT`, `NOOP`) | Spec consumes the enum. Wiring uses the enum value to decide write path. `failure_note_path` already lives on `RunMetadata`. |
| `NormanEvalReport.top20_gene_overlap` | `NormanEvalReport.top_k_overlap` (k configurable; currently k=20 in `metrics.top_k_gene_overlap`) | Template renders `top_k_overlap` with a footnote `(k=20 default)`. |

Additionally:

- **`context.md` is not accessible to this session.** The prompt requires regex patterns to be tested against actual content. Those patterns are therefore specified as **candidate templates** keyed off the field names the prompt enumerates; they MUST be validated against the real file before merging. A `tests/test_context_updater.py` fixture strategy is specified below.
- `wandb_config.WANDB_URL` exists and is imported by `experiment_logger.py`. `meta.wandb_url` may be `None`; fall back to `WANDB_URL` for rendering.

---

## 1. Module layout

```
aivc_platform/
└── memory/
    ├── __init__.py              # re-exports public API
    ├── obsidian_writer.py       # write_experiment_note, write_failure_note
    ├── context_updater.py       # update_context
    ├── vault.py                 # init_vault, ObsidianConfig
    ├── templates/
    │   ├── experiment_note.md.j2
    │   └── failure_note.md.j2
    └── tests/
        ├── test_obsidian_writer.py
        ├── test_context_updater.py
        └── fixtures/
            ├── context_sample.md
            └── expected_patched_context.md
```

Rationale for splitting `vault.py` out of `obsidian_writer.py`: `init_vault` is idempotent filesystem scaffolding; the writer should import config but not own vault lifecycle. Keeps `obsidian_writer.py` under ~200 LoC and unit-testable without touching directories outside tmp.

---

## 2. `ObsidianConfig` — Pydantic schema

```python
# aivc_platform/memory/vault.py
from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel, Field, field_validator

class ObsidianConfig(BaseModel):
    """Configuration for Obsidian vault writes."""

    vault_path: str = Field(
        default="~/Documents/Obsidian/aivc_genelink",
        description="Root of the Obsidian vault. `~` is expanded at resolution time.",
    )
    project: str = Field(default="aivc_genelink")
    auto_link: bool = Field(
        default=True,
        description="Emit wikilinks to related experiments/failures/papers.",
    )
    dry_run: bool = Field(
        default=False,
        description="If True, render and print to stdout but do not write to disk.",
    )

    @field_validator("vault_path")
    @classmethod
    def _no_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    def resolved_vault(self) -> Path:
        """Expand ~ and resolve. Does NOT create the directory."""
        return Path(self.vault_path).expanduser().resolve()
```

**Non-negotiables (enforced by this schema):**

- `vault_path` is stored as a `str` so it remains JSON-serialisable for logging.
- Expansion of `~` happens **only** through `resolved_vault()` / `Path.expanduser()` — never string interpolation.
- `dry_run=True` short-circuits every write path in this module.

---

## 3. `vault.py::init_vault`

```python
# aivc_platform/memory/vault.py
def init_vault(config: ObsidianConfig) -> None:
    """Create vault subdirectories and stub paper notes.

    Idempotent. Safe to call on every session start.
    Raises no exceptions for existing directories.

    If resolved_vault() does not exist, it is created (NOT raised).
    """
```

### Behaviour

1. `root = config.resolved_vault()`
2. `root.mkdir(parents=True, exist_ok=True)` — creates vault_path if absent. **Never raises** for missing vault_path (prompt requirement).
3. For each sub in `["experiments", "hypotheses", "failures", "insights", "papers", "models"]`:
   `(root / sub).mkdir(parents=True, exist_ok=True)`.
4. Stub papers (write only if file does not already exist — never clobber user edits):

   `papers/kang2018.md`:
   ```markdown
   # Kang 2018 — PBMC IFN-β single-cell perturbation atlas
   Kang, H. M., et al. (2018). "Multiplexed droplet single-cell RNA-sequencing using natural genetic variation." *Nat Biotechnol* 36, 89–94. https://doi.org/10.1038/nbt.4042
   ```

   `papers/norman2019.md`:
   ```markdown
   # Norman 2019 — Combinatorial CRISPRa in K562
   Norman, T. M., et al. (2019). "Exploring genetic interaction manifolds constructed from rich single-cell phenotypes." *Science* 365, 786–793. https://doi.org/10.1126/science.aax4438
   ```

5. `dry_run=True` → print the directory plan and stub contents to stdout; perform no filesystem mutation.

---

## 4. `obsidian_writer.py`

### 4.1 Public API

```python
from pathlib import Path
from aivc_platform.tracking.schemas import RunMetadata, PostRunDecision
from eval.eval_runner import EvalSuite
from aivc_platform.memory.vault import ObsidianConfig

def write_experiment_note(
    meta: RunMetadata,
    suite: EvalSuite,
    config: ObsidianConfig,
) -> Path: ...

def write_failure_note(
    meta: RunMetadata,
    decision: PostRunDecision,       # Enum, not BaseModel
    suite: EvalSuite,
    config: ObsidianConfig,
) -> Path: ...
```

### 4.2 Jinja2 dependency handling

```python
try:
    import jinja2
except ImportError as e:
    raise ImportError(
        "aivc_platform.memory requires Jinja2. "
        "Install with: pip install 'jinja2>=3.1,<4'"
    ) from e
```

Import is performed at **module top level**, not lazily, so the failure surfaces at `post_run_hook` bootstrap time rather than mid-run.

### 4.3 Rendering rules

- Templates are loaded from `aivc_platform/memory/templates/` via `jinja2.FileSystemLoader`, with `autoescape=False` (markdown output, no HTML escaping desired).
- Environment options: `trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True`.
- Custom filters:
  - `fmt_float` → `"{:.4f}".format(v) if v is not None else "n/a"`
  - `fmt_pct` → `"{:.2f}%".format(v) if v is not None else "n/a"`
  - `fmt_dt` → ISO-8601 UTC, second precision
- If `config.resolved_vault()` does not exist at write time, it is created (no raise).
- Writes use `Path.write_text(..., encoding="utf-8")`.
- `dry_run=True` → print `---\n# DRY RUN: {output_path}\n---\n{rendered}` to stdout, return the intended path without writing.
- Output is **overwritten** if it exists (idempotent by `run_id`).

### 4.4 Experiment note template (`templates/experiment_note.md.j2`)

```jinja
---
run_id: {{ meta.run_id }}
date: {{ now | fmt_dt }}
dataset: {{ meta.dataset }}
modality: {{ meta.modality.value }}
status: {{ meta.status.value }}
wandb_url: {{ meta.wandb_url or wandb_url_default }}
tags: [experiment, {{ config.project }}, {{ meta.modality.value | lower }}]
---

# Experiment · {{ meta.run_id }}

**Date:** {{ now | fmt_dt }}
**W&B:** {{ meta.wandb_url or wandb_url_default }}
**Checkpoint:** `{{ meta.checkpoint_path or "n/a" }}`
**AnnData hash:** `{{ anndata_hash or "n/a" }}`
**MLflow run:** {{ mlflow_run_id or "n/a (file store, no registry)" }}

## Results

### Kang 2018 (regression guard)
- Pearson r (raw):            {{ suite.kang.pearson_r | fmt_float }}  (± {{ suite.kang.pearson_r_std | fmt_float }})
- Pearson r (ctrl-subtracted): {{ suite.kang.pearson_r_ctrl_sub | fmt_float }}
- Regression guard passed:    {{ suite.kang.regression_guard_passed }}
- Delta non-zero %:           {{ suite.kang.delta_nonzero_pct | fmt_pct }}

### Norman 2019 (zero-shot)
{% if suite.norman %}
- Pearson r (ctrl-subtracted): {{ suite.norman.pearson_r_ctrl_sub | fmt_float }}
- Delta non-zero %:           {{ suite.norman.delta_nonzero_pct | fmt_pct }}
- Ctrl memorisation score:    {{ suite.norman.ctrl_memorisation_score | fmt_float }}
- Top-k gene overlap (k=20):  {{ suite.norman.top_k_overlap | fmt_float }}
- Passed:                     {{ suite.norman.passed }}
{% else %}
- *Norman eval not available (halted at Kang regression guard).*
{% endif %}

## Configuration

- `frozen_modules`:  {{ meta.frozen_modules | join(", ") }}
- `w_scale_range`:   [{{ meta.w_scale_range[0] }}, {{ meta.w_scale_range[1] }}]
- `neumann_k`:       {{ meta.neumann_k }}
- `lfc_beta`:        {{ meta.lfc_beta }}
- `lambda_l1`:       {{ meta.lambda_l1 }}
- `dataset`:         {{ meta.dataset }}
- `modality`:        {{ meta.modality.value }}

## What changed

<!-- human-fill: describe the single axis of change vs the previous run -->

## What it means biologically

<!-- human-fill: interpret the result in terms of perturbation response, JAK/STAT, etc -->

## Next experiment

<!-- human-fill: one sentence hypothesis + one config change -->

---

**Related:** [[models/v1.x_state]] · [[papers/kang2018]] · [[papers/norman2019]]
```

**Template contract (verified against spec requirements):**

- run_id, date, W&B url ✓
- Kang pearson_r (raw + ctrl-subtracted) ✓
- Norman pearson_r_ctrl_sub, delta_nonzero_pct, ctrl_memorisation_score, top20_gene_overlap (rendered as `top_k_overlap` with k=20 note) ✓
- frozen_modules, w_scale_range, neumann_k ✓
- checkpoint_path, anndata_hash ✓ (anndata_hash passed in by writer)
- dataset, modalities ✓ (single `modality` — see §0)
- Three human-fill sections ✓ (no placeholder prose beyond the HTML comment)
- Wikilinks: `[[models/v1.x_state]] [[papers/kang2018]] [[papers/norman2019]]` ✓

### 4.5 Failure note template (`templates/failure_note.md.j2`)

```jinja
---
run_id: {{ meta.run_id }}
date: {{ now | fmt_dt }}
dataset: {{ meta.dataset }}
status: {{ meta.status.value }}
decision: {{ decision.value }}
tags: [failure, {{ config.project }}, {{ failure_bucket }}]
---

# Failure · {{ meta.run_id }}

**Date:** {{ now | fmt_dt }}
**Decision:** `{{ decision.value }}`
**Failure reason:** {{ failure_reason }}
**Failure note (JSON):** `{{ meta.failure_note_path or "n/a" }}`

## RunMetadata

| field | value |
| --- | --- |
| run_id | {{ meta.run_id }} |
| dataset | {{ meta.dataset }} |
| modality | {{ meta.modality.value }} |
| pearson_r | {{ meta.pearson_r | fmt_float }} |
| pearson_r_std | {{ meta.pearson_r_std | fmt_float }} |
| delta_nonzero_pct | {{ meta.delta_nonzero_pct | fmt_pct }} |
| ctrl_memorisation_score | {{ meta.ctrl_memorisation_score | fmt_float }} |
| jakstat_within_3x | {{ meta.jakstat_within_3x }} |
| ifit1_pred_fc | {{ meta.ifit1_pred_fc | fmt_float }} |
| lfc_beta | {{ meta.lfc_beta }} |
| neumann_k | {{ meta.neumann_k }} |
| lambda_l1 | {{ meta.lambda_l1 }} |
| w_scale_range | [{{ meta.w_scale_range[0] }}, {{ meta.w_scale_range[1] }}] |
| frozen_modules | {{ meta.frozen_modules | join(", ") }} |
| checkpoint_path | `{{ meta.checkpoint_path or "n/a" }}` |
| wandb_url | {{ meta.wandb_url or wandb_url_default }} |
| started_at | {{ meta.started_at | fmt_dt }} |
| finished_at | {{ (meta.finished_at or now) | fmt_dt }} |
| training_time_s | {{ meta.training_time_s }} |

## EvalSuite summary

- Kang r:                 {{ suite.kang.pearson_r | fmt_float }}  (guard passed: {{ suite.kang.regression_guard_passed }})
- Norman delta_nonzero %: {% if suite.norman %}{{ suite.norman.delta_nonzero_pct | fmt_pct }}{% else %}n/a (skipped){% endif %}
- Halt reason:            {{ suite.halt_reason or "n/a" }}

## Suspected causes (ranked by likelihood)

{% for cause in suspected_causes %}
{{ loop.index }}. **{{ cause.label }}** — {{ cause.rationale }}
{% endfor %}

## Next actions — training_agent checklist

{% for action in next_actions %}
- [ ] {{ action }}
{% endfor %}

---

{% if config.auto_link and related_failures %}
**Related failures:**
{% for rf in related_failures %}
- [[failures/{{ rf }}]]
{% endfor %}
{% endif %}
**See also:** [[papers/kang2018]] · [[papers/norman2019]]
```

**Cause-ranking heuristic** (inferred from `failure_reason` — deterministic, no LLM call):

| trigger (substring match, case-insensitive) | ranked causes |
| --- | --- |
| `"regression guard"` | 1. GAT unfreeze leaked (assert `"genelink" in meta.frozen_modules`) · 2. checkpoint corruption or wrong run loaded · 3. LR too high on feature_expander |
| `"delta_nonzero"` or `"ctrl memorisation"` | 1. `w_scale_range` too aggressive → sparsity collapse · 2. `neumann_k` too small → insufficient propagation · 3. pert_id broadcast bug (known v1.1 ceiling) · 4. data label swap |
| `"Norman eval not available"` | 1. AnnData missing / wrong path · 2. gene alignment failure (3010 intersection) · 3. insufficient pseudo-bulk pairs |
| default | 1. unclassified — human triage required |

**Next-actions list** is derived from the top-ranked cause (closed set of strings — no freeform generation).

**`related_failures`**: scan `{vault_path}/failures/` for files whose YAML frontmatter `tags` includes the same `failure_bucket`; return up to 5, sorted by mtime descending. Skipped if `auto_link=False`.

### 4.6 Writer implementation notes

```python
# obsidian_writer.py — shape only
from datetime import datetime, timezone
from pathlib import Path
import jinja2
from aivc_platform.tracking.wandb_config import WANDB_URL

_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(Path(__file__).parent / "templates"),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)
_ENV.filters["fmt_float"] = lambda v: "n/a" if v is None else f"{v:.4f}"
_ENV.filters["fmt_pct"]   = lambda v: "n/a" if v is None else f"{v:.2f}%"
_ENV.filters["fmt_dt"]    = lambda v: v.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _render(name: str, **ctx) -> str:
    return _ENV.get_template(name).render(**ctx)

def _write_or_print(path: Path, content: str, dry_run: bool) -> Path:
    if dry_run:
        print(f"---\n# DRY RUN: {path}\n---\n{content}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path

def write_experiment_note(meta, suite, config) -> Path:
    vault = config.resolved_vault()
    out = vault / "experiments" / f"{meta.run_id}.md"
    content = _render(
        "experiment_note.md.j2",
        meta=meta, suite=suite, config=config,
        now=datetime.now(timezone.utc),
        wandb_url_default=WANDB_URL,
        anndata_hash=getattr(meta, "anndata_hash", None),
        mlflow_run_id=None,  # v1: no registry
    )
    return _write_or_print(out, content, config.dry_run)

def write_failure_note(meta, decision, suite, config) -> Path:
    vault = config.resolved_vault()
    out = vault / "failures" / f"{meta.run_id}.md"
    bucket, causes, actions = _classify_failure(suite, meta)
    related = _find_related_failures(vault, bucket) if config.auto_link else []
    content = _render(
        "failure_note.md.j2",
        meta=meta, suite=suite, decision=decision, config=config,
        now=datetime.now(timezone.utc),
        wandb_url_default=WANDB_URL,
        failure_reason=_extract_failure_reason(suite),
        failure_bucket=bucket,
        suspected_causes=causes,
        next_actions=actions,
        related_failures=related,
    )
    return _write_or_print(out, content, config.dry_run)
```

---

## 5. `context_updater.py::update_context`

### 5.1 Public API

```python
def update_context(
    meta: RunMetadata,
    suite: EvalSuite,
    context_path: str = "~/Documents/Cowork/AshKhan/Projects/aivc_genelink/context.md",
    *,
    dry_run: bool = False,
) -> None: ...
```

### 5.2 Contract

- **Update only** these four lines (plus conditional best_r bump):
  - `Last updated`
  - `best pearson_r` (only if `meta.pearson_r > current_value`)
  - `latest checkpoint`
  - `last W&B run`
  - `last run date`
- **Never modify** any line between the start of any of these section headers and the next `## ` header at the same level:
  - `## Open questions`
  - `## Do not touch`
  - `## GitHub`
  - `## Stack snapshot`
  - `## Mission`
- If `context_path` does not exist: raise `FileNotFoundError`. This is intentional — silent creation would produce a file diverging from the user's template. `init_vault` handles Obsidian side; `context.md` is authored by the user.
- **Idempotency:** running twice with the same `meta` / `suite` MUST produce byte-identical output.
- **Dry run:** emit a unified diff to stdout via `difflib.unified_diff`; do not write.

### 5.3 Regex patterns (candidate — MUST be validated)

`context.md` was **not readable by this session** (see §0). These patterns are specified against the prompt-declared field names and the common `key: value` / markdown-list conventions seen in this repo's existing context files (e.g., `Claude Outputs/aivc_genelink/aivc_repo_q1q2_v1.md`). Before merging, run the test fixture in §5.5 against the real file and adjust anchors if needed.

```python
# aivc_platform/memory/context_updater.py
import re

# Each pattern captures the full line in group 0 and the value in group "val".
# Patterns are line-anchored (MULTILINE) and tolerate optional bold/italic,
# optional backticks on the value, and trailing whitespace.

PATTERNS: dict[str, re.Pattern[str]] = {
    "last_updated": re.compile(
        r"^(?P<prefix>[-*]?\s*\*{0,2}Last updated\*{0,2}\s*[:\-]\s*)(?P<val>.+?)\s*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    "best_pearson_r": re.compile(
        r"^(?P<prefix>[-*]?\s*\*{0,2}best pearson_?r\*{0,2}\s*[:\-]\s*)(?P<val>[0-9.]+)\s*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    "latest_checkpoint": re.compile(
        r"^(?P<prefix>[-*]?\s*\*{0,2}latest checkpoint\*{0,2}\s*[:\-]\s*)(?P<val>`?[^`\n]+`?)\s*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    "last_wandb_run": re.compile(
        r"^(?P<prefix>[-*]?\s*\*{0,2}last W&?B run\*{0,2}\s*[:\-]\s*)(?P<val>\S+)\s*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    "last_run_date": re.compile(
        r"^(?P<prefix>[-*]?\s*\*{0,2}last run date\*{0,2}\s*[:\-]\s*)(?P<val>.+?)\s*$",
        re.MULTILINE | re.IGNORECASE,
    ),
}

# Forbidden section ranges (header → next ## header).
FORBIDDEN_SECTIONS = (
    "Open questions",
    "Do not touch",
    "GitHub",
    "Stack snapshot",
    "Mission",
)

FORBIDDEN_SECTION_RE = re.compile(
    r"^##\s+(?P<name>.+?)\s*$(?P<body>.*?)(?=^##\s|\Z)",
    re.MULTILINE | re.DOTALL,
)
```

### 5.4 Algorithm

1. Read `context_path` (expand `~`). Raise `FileNotFoundError` if absent.
2. Compute forbidden ranges:
   ```python
   forbidden_spans = [
       (m.start(), m.end())
       for m in FORBIDDEN_SECTION_RE.finditer(text)
       if m.group("name").strip() in FORBIDDEN_SECTIONS
   ]
   ```
3. For each updatable field, attempt a regex substitution in **regions not inside a forbidden span** only. Implementation:
   ```python
   def _safe_sub(text, pattern, replacement, forbidden_spans):
       def repl(match):
           if any(s <= match.start() < e for s, e in forbidden_spans):
               return match.group(0)  # leave untouched
           return replacement(match)
       return pattern.sub(repl, text)
   ```
4. Conditional best_r: parse current value, only replace if `meta.pearson_r > current`.
5. **If a pattern has zero matches outside forbidden spans**, append under `## Architecture decisions` (create the section at EOF if it is missing). Each appended line uses canonical form: `- **<Field Name>**: <value>`.
6. If no bytes changed, return without writing (true idempotency).
7. `dry_run=True` → print `difflib.unified_diff(original, patched, "before", "after")` to stdout; skip write.
8. Write atomically: `tmp = context_path + ".tmp"; tmp.write_text(...); tmp.replace(context_path)`.

### 5.5 Test strategy for regex (prerequisite before merge)

Add `tests/fixtures/context_sample.md` — a sanitised copy of the real `context.md`. Add `tests/test_context_updater.py` covering:

- Updating each of the five fields independently.
- Double-run idempotency (byte-equal after second call).
- Forbidden-section protection: tests that inject the updatable strings *inside* `## Open questions` and assert they remain untouched.
- `best_pearson_r`: no update when `meta.pearson_r < current`; update when `>`.
- Missing-field path: appends under `## Architecture decisions`, which is itself created if absent.
- `dry_run=True`: no write; diff present on stdout.

If any pattern in §5.3 fails against the real file, adjust the `prefix` group only — never weaken the value group enough to over-match.

---

## 6. Wiring into `ExperimentLogger.post_run_hook`

### 6.1 Insertion points

`aivc_platform/tracking/experiment_logger.py` — two injection points:

```python
# top of file, alongside existing imports:
from aivc_platform.memory.obsidian_writer import (
    write_experiment_note, write_failure_note,
)
from aivc_platform.memory.context_updater import update_context
from aivc_platform.memory.vault import ObsidianConfig, init_vault
from aivc_platform.tracking.schemas import PostRunDecision
```

`ExperimentLogger.__init__`:

```python
self._obsidian_config = ObsidianConfig()  # read from env in v1.2
try:
    init_vault(self._obsidian_config)
except Exception as e:
    logger.warning(f"init_vault failed (non-fatal): {e}")
```

### 6.2 `post_run_hook` wiring — new signature

`post_run_hook` must accept the `EvalSuite` so the memory layer sees full Kang + Norman results, not just the lossy projection onto `RunMetadata`. Minimal change:

```python
def post_run_hook(self, meta: RunMetadata, suite: "EvalSuite | None" = None) -> None:
```

Callers that already have the suite (training scripts call `populate_run_metadata(meta, suite)` just before `finish`) pass it through `finish(meta, suite=suite)`. `finish` forwards to `post_run_hook`.

### 6.3 Success / failure branching

```python
# inside _handle_success(self, meta, suite):
...  # existing MLflow + registry code unchanged
try:
    write_experiment_note(meta, suite, self._obsidian_config)
    update_context(meta, suite)
    meta.decision = PostRunDecision.REGISTER
except Exception as e:
    logger.warning(f"memory write (success path) failed: {e}")
# existing: self._dispatcher.trigger("research_agent", ...)

# inside _handle_failure(self, meta, suite):
...  # existing JSON failure note unchanged
try:
    write_failure_note(
        meta,
        PostRunDecision.TRIGGER_TRAINING_AGENT,
        suite,
        self._obsidian_config,
    )
    # explicitly DO NOT call update_context on failure
    meta.decision = PostRunDecision.TRIGGER_TRAINING_AGENT
except Exception as e:
    logger.warning(f"memory write (failure path) failed: {e}")
# existing: self._dispatcher.trigger("training_agent", ...)
```

### 6.4 Failure-mode table

| Failure | Effect on training run | Effect on agent dispatch |
| --- | --- | --- |
| Jinja2 not installed | `ImportError` at module load → training script fails at import (surfaces early) | n/a |
| Vault path not writable | logged warning; obsidian write is skipped | unaffected — agent dispatch still runs |
| `context.md` missing | `FileNotFoundError` from `update_context`, caught by wrapper → warning | unaffected |
| Suite is `None` (caller didn't pass it) | memory write is skipped with warning; existing JSON failure note still written | unaffected |

Memory writes are **best-effort, non-fatal** — training completeness and agent dispatch are load-bearing; Obsidian notes are not.

### 6.5 Caller update

In `populate_run_metadata` callers (e.g. `train_week4.py`, `evaluate_week4.py`):

```python
meta = populate_run_metadata(meta, suite)
logger.finish(meta, suite=suite)   # add suite kwarg
```

---

## 7. Error-handling summary (one place)

| Condition | Behaviour |
| --- | --- |
| `vault_path` does not exist | Created via `mkdir(parents=True, exist_ok=True)`. No raise. |
| Jinja2 not installed | `ImportError` at module import with explicit install instruction. |
| Template file missing | `jinja2.TemplateNotFound` bubbles up (indicates broken install). |
| `config.dry_run=True` | Print rendered content or diff; no writes. Return the intended path. |
| `context.md` missing | `FileNotFoundError`. Wired call site logs and continues. |
| Concurrent writers to same `run_id` | Last-writer-wins — acceptable (runs are serial per logger). |
| Unicode in user notes | UTF-8 throughout. No normalisation. |

---

## 8. Module public API (`memory/__init__.py`)

```python
from aivc_platform.memory.vault import ObsidianConfig, init_vault
from aivc_platform.memory.obsidian_writer import (
    write_experiment_note,
    write_failure_note,
)
from aivc_platform.memory.context_updater import update_context

__all__ = [
    "ObsidianConfig",
    "init_vault",
    "write_experiment_note",
    "write_failure_note",
    "update_context",
]
```

---

## Key Risks

1. **Schema drift — high.** The prompt's "locked schemas" list diverges from `schemas.py` (`modalities` vs `modality`, `mlflow_run_id` missing, `PostRunDecision` is an Enum). This spec reads from the actual code. If the intent is to migrate the schemas, that is a separate ticket and must land first — otherwise the memory layer will render `AttributeError` at runtime.
2. **Regex patterns unvalidated against real `context.md`.** The file is not accessible in this session. Patterns are defensive but there is non-zero probability they mis-anchor on real content (e.g., a line like `Last updated by Ash: ...` would match). The fixture-based test in §5.5 is a **pre-merge prerequisite**, not a nice-to-have.
3. **`post_run_hook` signature change.** Adding `suite` to `post_run_hook` / `finish` is a breaking change for any external caller. Mitigation: make `suite` optional and degrade gracefully, which this spec does.
4. **`related_failures` scanning doesn't scale.** Linear directory scan with YAML parse is fine at 10²–10³ failures; past that, add an index file. Not urgent for v1.
5. **Jinja2 autoescape disabled.** Intentional (markdown output) but means any template author who inserts user-controlled HTML gets it rendered raw. Low risk today — all inputs are internal types — flag if `meta.dataset` or `run_id` ever becomes user-provided.

## Recommended Next Step

Land in two PRs to isolate risk:

- **PR 1** — `aivc_platform/memory/{vault.py, obsidian_writer.py, templates/, __init__.py}` + tests that mock the filesystem (`tmp_path`). Does not touch `experiment_logger.py`. Can be merged and validated in isolation by running `init_vault` + `write_experiment_note` on a past successful `RunMetadata` captured from `artifacts/registry/latest_checkpoint.json`.
- **PR 2** — `context_updater.py` + regex fixture test (after copying the real `context.md` into `tests/fixtures/`) + the `experiment_logger.py` wiring (§6.2–6.3) + caller updates (§6.5).

## What I need from you

1. Confirm the schema divergences in §0 — should the spec adapt (current behaviour) or should `schemas.py` be migrated first (`modalities: list[Modality]`, `mlflow_run_id: Optional[str]`, `PostRunDecision` promoted to a BaseModel)?
2. Drop the real `context.md` into `tests/fixtures/context_sample.md` so the regex in §5.3 can be validated before I hand over the implementation.
3. Confirm `anndata_hash` sourcing — it is referenced in the experiment note template but is not a field on `RunMetadata` today. Options: (a) add to `RunMetadata`, (b) compute at write time from `meta.dataset` path, (c) render `"n/a"`. Current spec defaults to (c).
