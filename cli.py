"""cli.py — AIVC Typer CLI orchestrator.

Commands: train, eval, sweep, agent run, memory sync, status.
All heavy imports are lazy (inside command bodies) to keep --help fast
and to let tests patch module-level symbols cleanly.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import typer

app = typer.Typer(name="aivc", help="AIVC GeneLink CLI", no_args_is_help=True)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _default_h5ad_for_dataset(dataset: str) -> str:
    mapping = {
        "kang2018_pbmc": "data/kang2018_pbmc_fixed.h5ad",
        "kang2018": "data/kang2018_pbmc_fixed.h5ad",
        "norman2019": "data/norman2019.h5ad",
    }
    return mapping.get(dataset, f"data/{dataset}.h5ad")


def _build_run_metadata(run_id: str, dataset: str, checkpoint_path: str, suite):
    """Construct a RunMetadata populated from an EvalSuite."""
    from aivc_platform.tracking.schemas import RunMetadata, RunStatus
    from eval.eval_runner import populate_run_metadata

    meta = RunMetadata(run_id=run_id, dataset=dataset, checkpoint_path=checkpoint_path)
    populate_run_metadata(meta, suite)
    if suite.overall_passed:
        meta.status = RunStatus.SUCCESS
    elif suite.kang is not None and not suite.kang.regression_guard_passed:
        meta.status = RunStatus.REGRESSION
    elif suite.norman is not None and suite.norman.delta_nonzero_pct == 0.0:
        meta.status = RunStatus.CTRL_MEMORISATION
    else:
        meta.status = RunStatus.FAILURE
    return meta


def _dispatch_agent(agent_name: str, payload: dict) -> None:
    """Direct AgentDispatcher call — writes prompt file to agent_queue/."""
    try:
        from aivc_platform.tracking.experiment_logger import AgentDispatcher
        AgentDispatcher(execute=False).trigger(agent_name, payload)
    except Exception as e:  # pragma: no cover
        typer.echo(f"[WARN] dispatch {agent_name} failed: {e}")


def _write_failure_note(meta) -> None:
    from datetime import datetime, timezone
    failure_dir = Path("artifacts/failure_notes")
    failure_dir.mkdir(parents=True, exist_ok=True)
    note = {
        "run_id": meta.run_id,
        "dataset": meta.dataset,
        "pearson_r": meta.pearson_r,
        "delta_nonzero_pct": meta.delta_nonzero_pct,
        "ctrl_memorisation_score": meta.ctrl_memorisation_score,
        "w_scale_range": list(meta.w_scale_range),
        "neumann_k": meta.neumann_k,
        "status": meta.status.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = failure_dir / f"failure_{meta.run_id}.json"
    path.write_text(json.dumps(note, indent=2), encoding="utf-8")
    meta.failure_note_path = str(path)


# ─────────────────────────────────────────────────────────────────────
# train
# ─────────────────────────────────────────────────────────────────────

@app.command()
def train(
    config: Path = typer.Option(None, help="Sweep config YAML."),
    dataset: str = typer.Option("kang2018_pbmc", help="Dataset name."),
    run_id: str = typer.Option(None, help="Run id (auto-generated if absent)."),
    output_dir: Path = typer.Option(Path("models/v1.1/"), help="Checkpoint output dir."),
    device: str = typer.Option(None, help="cuda|cpu (auto-detect if absent)."),
    sweep: str = typer.Option(None, help="Inline JSON sweep override."),
    skip_eval: bool = typer.Option(False),
    no_memory_sync: bool = typer.Option(False),
    dry_run: bool = typer.Option(False),
) -> None:
    """Run the full train → eval → register → memory loop."""
    from agents.base_agent import AgentTask
    from agents.data_agent import DataAgent
    from agents.eval_agent import EvalAgent

    run_id = run_id or str(uuid.uuid4())[:8]

    # 1. DataAgent gate
    data_result = DataAgent().run(AgentTask(
        agent_name="data_agent",
        run_id=run_id,
        payload={
            "h5ad_path": _default_h5ad_for_dataset(dataset),
            "dataset": dataset,
            "require_cert": True,
        },
    ))
    if not data_result.success:
        typer.echo(f"HALTED by DataAgent: {data_result.error}")
        raise typer.Exit(1)

    if dry_run:
        typer.echo("Dry run OK")
        raise typer.Exit(0)

    # 2. Maybe write inline sweep override to a tmp config
    effective_config = config
    if sweep:
        import tempfile
        import yaml
        try:
            data = json.loads(sweep)
        except json.JSONDecodeError as e:
            typer.echo(f"Invalid --sweep JSON: {e}")
            raise typer.Exit(1)
        tmp = Path(tempfile.mkstemp(suffix=".yaml")[1])
        tmp.write_text(yaml.safe_dump(data), encoding="utf-8")
        effective_config = tmp

    # 3. Subprocess: train_v11.py
    env = os.environ.copy()
    env["AIVC_RUN_ID"] = run_id
    cmd = [sys.executable, "train_v11.py"]
    if effective_config is not None:
        cmd += ["--config", str(effective_config)]
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    except Exception as e:
        typer.echo(f"Training subprocess failed to start: {e}")
        raise typer.Exit(1)
    if result.returncode != 0:
        typer.echo(f"Training failed:\n{result.stderr}")
        raise typer.Exit(1)

    checkpoint = Path("models/v1.1/model_v11_best.pt")
    if not checkpoint.exists():
        typer.echo("Checkpoint not found after training")
        raise typer.Exit(1)

    if skip_eval:
        typer.echo(f"SUCCESS (skip-eval): run_id={run_id}")
        raise typer.Exit(0)

    # 4. Eval
    eval_result = EvalAgent().run(AgentTask(
        agent_name="eval_agent",
        run_id=run_id,
        payload={"checkpoint_path": str(checkpoint), "device": device},
    ))

    suite = None
    if eval_result.output_path and Path(eval_result.output_path).exists():
        from eval.eval_runner import EvalSuite
        try:
            suite = EvalSuite.model_validate_json(
                Path(eval_result.output_path).read_text(encoding="utf-8")
            )
        except Exception as e:
            typer.echo(f"Could not parse EvalSuite: {e}")
            raise typer.Exit(1)

    if suite is None:
        typer.echo(f"Eval produced no suite JSON: {eval_result.error}")
        raise typer.Exit(1)

    meta = _build_run_metadata(run_id, dataset, str(checkpoint), suite)

    # 5. Failure routing
    if not suite.kang.regression_guard_passed:
        typer.echo(
            f"REGRESSION GUARD FAILED: pearson_r={suite.kang.pearson_r:.4f} < 0.873 — halting"
        )
        try:
            _write_failure_note(meta)
        except Exception as e:
            typer.echo(f"[WARN] failure note write failed: {e}")
        _dispatch_agent("training_agent", {
            "run_id": run_id, "dataset": dataset,
            "pearson_r": meta.pearson_r, "delta_nonzero_pct": meta.delta_nonzero_pct,
            "ctrl_memorisation_score": meta.ctrl_memorisation_score,
            "w_scale_range": str(meta.w_scale_range), "neumann_k": meta.neumann_k,
            "failure_note_path": meta.failure_note_path or "",
        })
        if not no_memory_sync:
            try:
                from aivc_platform.memory.obsidian_writer import write_failure_note as _wfn
                from aivc_platform.memory.vault import ObsidianConfig
                from aivc_platform.tracking.schemas import PostRunDecision
                _wfn(meta, PostRunDecision.TRIGGER_TRAINING_AGENT, suite, ObsidianConfig())
            except Exception as e:
                typer.echo(f"[WARN] failure memory write failed: {e}")
        raise typer.Exit(2)

    if suite.norman is not None and suite.norman.delta_nonzero_pct == 0.0:
        typer.echo(
            f"FAILURE: run_id={run_id} delta_nonzero_pct=0.0 — halting eval"
        )
        try:
            _write_failure_note(meta)
        except Exception as e:
            typer.echo(f"[WARN] failure note write failed: {e}")
        _dispatch_agent("training_agent", {
            "run_id": run_id, "dataset": dataset,
            "pearson_r": meta.pearson_r, "delta_nonzero_pct": meta.delta_nonzero_pct,
            "ctrl_memorisation_score": meta.ctrl_memorisation_score,
            "w_scale_range": str(meta.w_scale_range), "neumann_k": meta.neumann_k,
            "failure_note_path": meta.failure_note_path or "",
        })
        if not no_memory_sync:
            try:
                from aivc_platform.memory.obsidian_writer import write_failure_note as _wfn
                from aivc_platform.memory.vault import ObsidianConfig
                from aivc_platform.tracking.schemas import PostRunDecision
                _wfn(meta, PostRunDecision.TRIGGER_TRAINING_AGENT, suite, ObsidianConfig())
            except Exception as e:
                typer.echo(f"[WARN] failure memory write failed: {e}")
        raise typer.Exit(3)

    # 6. Success path: memory + registry. ExperimentLogger.finish owns dispatch.
    if not no_memory_sync:
        try:
            from aivc_platform.tracking.experiment_logger import ExperimentLogger
            ExperimentLogger().finish(meta, suite=suite)
        except Exception as e:
            typer.echo(f"[WARN] ExperimentLogger.finish failed: {e}")
        try:
            from aivc_platform.memory.obsidian_writer import write_experiment_note
            from aivc_platform.memory.context_updater import update_context
            from aivc_platform.memory.vault import ObsidianConfig
            write_experiment_note(meta, suite, ObsidianConfig())
            try:
                update_context(meta, suite)
            except Exception as e:
                typer.echo(f"[WARN] update_context failed: {e}")
        except Exception as e:
            typer.echo(f"[WARN] memory write failed: {e}")

    try:
        from aivc_platform.registry.model_registry import ModelRegistry
        ModelRegistry().register(meta, suite)
    except Exception as e:
        typer.echo(f"[WARN] Registry write failed: {e}")
        raise typer.Exit(4)

    typer.echo(f"SUCCESS: run_id={run_id} pearson_r={meta.pearson_r:.4f}")
    raise typer.Exit(0)


# ─────────────────────────────────────────────────────────────────────
# eval
# ─────────────────────────────────────────────────────────────────────

@app.command("eval")
def eval_cmd(
    checkpoint: Path = typer.Argument(...),
    run_id: str = typer.Option(None),
    device: str = typer.Option(None),
    kang_adata: Path = typer.Option(Path("data/kang2018_pbmc_fixed.h5ad")),
    norman_adata: Path = typer.Option(Path("data/norman2019.h5ad")),
    output_dir: Path = typer.Option(Path("artifacts/eval_results/")),
) -> None:
    """Run the evaluation suite on a checkpoint."""
    from agents.base_agent import AgentTask
    from agents.eval_agent import EvalAgent

    run_id = run_id or str(uuid.uuid4())[:8]
    if not checkpoint.exists():
        typer.echo(f"Checkpoint not found: {checkpoint}")
        raise typer.Exit(3)

    result = EvalAgent().run(AgentTask(
        agent_name="eval_agent",
        run_id=run_id,
        payload={
            "checkpoint_path": str(checkpoint), "device": device,
            "kang_adata": str(kang_adata), "norman_adata": str(norman_adata),
        },
    ))

    if not result.output_path:
        typer.echo(f"Eval failed: {result.error}")
        raise typer.Exit(3)

    from eval.eval_runner import EvalSuite
    suite = EvalSuite.model_validate_json(Path(result.output_path).read_text(encoding="utf-8"))

    typer.echo("=" * 60)
    typer.echo(f"EVAL SUITE: {'PASSED' if suite.overall_passed else 'FAILED'}")
    typer.echo("=" * 60)
    typer.echo(f"  Kang:   passed={suite.kang.passed} r={suite.kang.pearson_r:.4f}")
    if suite.norman:
        typer.echo(
            f"  Norman: passed={suite.norman.passed} "
            f"delta_nz={suite.norman.delta_nonzero_pct:.1f}%"
        )
    else:
        typer.echo("  Norman: skipped")

    if not suite.kang.regression_guard_passed:
        raise typer.Exit(1)
    if suite.norman is not None and not suite.norman.passed:
        raise typer.Exit(2)
    raise typer.Exit(0)


# ─────────────────────────────────────────────────────────────────────
# sweep
# ─────────────────────────────────────────────────────────────────────

@app.command()
def sweep(
    config: Path = typer.Option(Path("configs/sweep_w_scale.yaml")),
    dataset: str = typer.Option("kang2018_pbmc"),
    count: int = typer.Option(20),
    output_dir: Path = typer.Option(Path("models/v1.1/")),
    device: str = typer.Option(None),
    no_halt_on_fail: bool = typer.Option(False),
) -> None:
    """Run a serial hyperparameter sweep (v1: parallel=1 only)."""
    try:
        import torch
        if not torch.cuda.is_available():
            typer.echo("[WARN] CUDA not available — sweep will run on CPU.")
    except Exception:
        pass

    if not config.exists():
        typer.echo(f"Sweep config not found: {config}")
        raise typer.Exit(1)

    import time
    results = []
    best_r = -1.0
    best_run_id = None

    for i in range(count):
        rid = f"sweep_{int(time.time())}_{i:03d}"
        cmd = [sys.executable, "cli.py", "train",
               "--config", str(config),
               "--dataset", dataset,
               "--run-id", rid,
               "--output-dir", str(output_dir)]
        if device:
            cmd += ["--device", device]
        typer.echo(f"[{i + 1}/{count}] run_id={rid}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        results.append({"run_id": rid, "exit_code": proc.returncode})

        from aivc_platform.registry.model_registry import ModelRegistry
        latest = ModelRegistry().get_latest()
        if latest and latest.run_id == rid and latest.pearson_r > best_r:
            best_r = latest.pearson_r
            best_run_id = rid

        if proc.returncode in (2, 3) and not no_halt_on_fail:
            typer.echo(f"Sweep halted after {i + 1} runs (exit={proc.returncode})")
            break

    Path("artifacts").mkdir(parents=True, exist_ok=True)
    out = Path(f"artifacts/sweep_results_{int(time.time())}.json")
    out.write_text(json.dumps({
        "best_pearson_r": best_r, "best_run_id": best_run_id, "runs": results,
    }, indent=2), encoding="utf-8")
    typer.echo(f"Best pearson_r={best_r:.4f} run_id={best_run_id}")
    raise typer.Exit(0 if best_r >= 0.873 else 1)


# ─────────────────────────────────────────────────────────────────────
# agent run
# ─────────────────────────────────────────────────────────────────────

agent_app = typer.Typer(help="Agent subcommands.")
app.add_typer(agent_app, name="agent")


@agent_app.command("run")
def agent_run(
    agent: str = typer.Option(...),
    run_id: str = typer.Option(...),
    payload: str = typer.Option("{}", help="JSON dict or @path/to/file.json"),
    no_api_call: bool = typer.Option(False),
) -> None:
    """Run a single agent by name."""
    from agents.base_agent import AgentTask
    from agents.data_agent import DataAgent
    from agents.eval_agent import EvalAgent
    from agents.training_agent import TrainingAgent
    from agents.research_agent import ResearchAgent

    if payload.startswith("@"):
        payload_dict = json.loads(Path(payload[1:]).read_text(encoding="utf-8"))
    else:
        try:
            payload_dict = json.loads(payload)
        except json.JSONDecodeError as e:
            typer.echo(f"Invalid --payload JSON: {e}")
            raise typer.Exit(2)

    agents_map = {
        "data_agent": DataAgent,
        "eval_agent": EvalAgent,
        "training_agent": TrainingAgent,
        "research_agent": ResearchAgent,
    }
    if agent not in agents_map:
        typer.echo(f"Unknown agent: {agent}")
        raise typer.Exit(2)

    if no_api_call and agent in ("training_agent", "research_agent"):
        os.environ["ANTHROPIC_API_KEY"] = ""

    try:
        instance = agents_map[agent]()
    except Exception as e:
        typer.echo(f"Agent instantiation failed: {e}")
        raise typer.Exit(2)

    task = AgentTask(agent_name=agent, run_id=run_id, payload=payload_dict)
    result = instance.run(task)
    typer.echo(result.model_dump_json(indent=2))
    raise typer.Exit(0 if result.success else 1)


# ─────────────────────────────────────────────────────────────────────
# memory sync
# ─────────────────────────────────────────────────────────────────────

memory_app = typer.Typer(help="Memory subcommands.")
app.add_typer(memory_app, name="memory")


@memory_app.command("sync")
def memory_sync(
    run_ids: str = typer.Option(None, help="Comma-separated (default: scan queue)"),
    force_update: bool = typer.Option(False),
) -> None:
    """Backfill Obsidian notes + context.md from queue artifacts."""
    from aivc_platform.memory.obsidian_writer import write_experiment_note
    from aivc_platform.memory.context_updater import update_context
    from aivc_platform.memory.vault import ObsidianConfig
    from aivc_platform.tracking.schemas import RunMetadata
    from eval.eval_runner import EvalSuite

    queue = Path("artifacts/agent_queue")
    if run_ids:
        ids = [r.strip() for r in run_ids.split(",") if r.strip()]
    else:
        if not queue.exists():
            typer.echo("No queue files found.")
            raise typer.Exit(0)
        ids = sorted({p.stem.split("_")[-2] for p in queue.glob("eval_agent_*.json")
                      if len(p.stem.split("_")) >= 3})

    config = ObsidianConfig()
    errors = 0
    for rid in ids:
        try:
            eval_files = list(queue.glob(f"eval_agent_{rid}_*.json"))
            if not eval_files:
                continue
            suite = EvalSuite.model_validate_json(eval_files[-1].read_text(encoding="utf-8"))
            meta = RunMetadata(run_id=rid, checkpoint_path="")
            from eval.eval_runner import populate_run_metadata
            populate_run_metadata(meta, suite)

            vault_note = config.resolved_vault() / "experiments" / f"{rid}.md"
            if vault_note.exists() and not force_update:
                continue
            write_experiment_note(meta, suite, config)
            try:
                update_context(meta, suite)
            except Exception as e:
                typer.echo(f"[WARN] update_context {rid}: {e}")
        except Exception as e:
            typer.echo(f"[WARN] sync {rid} failed: {e}")
            errors += 1

    typer.echo(f"memory sync done: {len(ids)} runs, {errors} errors")
    raise typer.Exit(0 if errors == 0 else 2)


# ─────────────────────────────────────────────────────────────────────
# status
# ─────────────────────────────────────────────────────────────────────

@app.command()
def status() -> None:
    """Print latest checkpoint, best registered, pending prompts, GPU status."""
    from aivc_platform.registry.model_registry import ModelRegistry
    from aivc_platform.tracking.wandb_config import WANDB_URL

    reg = ModelRegistry()
    latest = reg.get_latest()
    best = reg.get_best()

    queue = Path("artifacts/agent_queue")
    pending = 0
    if queue.exists():
        md_files = list(queue.glob("*.md"))
        for p in md_files:
            if not p.stem.endswith("_response"):
                resp = p.with_name(p.stem + "_response.md")
                if not resp.exists():
                    pending += 1

    try:
        import torch
        gpu = torch.cuda.is_available()
    except Exception:
        gpu = False

    typer.echo(f"Latest checkpoint: {latest.checkpoint_path if latest else 'none'}")
    typer.echo(f"Latest pearson_r:  {latest.pearson_r:.4f}" if latest else "Latest pearson_r:  n/a")
    typer.echo(f"Best registered:   {best.version if best else 'none'}")
    typer.echo(f"Pending prompts:   {pending}")
    typer.echo(f"W&B URL:           {WANDB_URL}")
    typer.echo(f"GPU available:     {gpu}")
    raise typer.Exit(0 if latest else 1)


if __name__ == "__main__":
    app()
