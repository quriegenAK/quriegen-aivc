"""aivc_platform.memory — Obsidian vault writer, context.md patcher."""
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
