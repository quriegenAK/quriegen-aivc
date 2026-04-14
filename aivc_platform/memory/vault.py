"""aivc_platform/memory/vault.py — Obsidian vault scaffolding.

ObsidianConfig: write configuration (vault path, dry-run flag).
init_vault: idempotent creation of 6 subdirs + 2 stub paper notes.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator

SUBDIRS = ("experiments", "hypotheses", "failures", "insights", "papers", "models")

_KANG_STUB = """# Kang 2018 — PBMC IFN-β single-cell perturbation atlas
Kang, H. M., et al. (2018). "Multiplexed droplet single-cell RNA-sequencing using natural genetic variation." *Nat Biotechnol* 36, 89–94. https://doi.org/10.1038/nbt.4042
"""

_NORMAN_STUB = """# Norman 2019 — Combinatorial CRISPRa in K562
Norman, T. M., et al. (2019). "Exploring genetic interaction manifolds constructed from rich single-cell phenotypes." *Science* 365, 786–793. https://doi.org/10.1126/science.aax4438
"""


class ObsidianConfig(BaseModel):
    """Configuration for Obsidian vault writes."""

    vault_path: str = Field(
        default="~/Documents/Obsidian/aivc_genelink",
        description="Root of the Obsidian vault. `~` expanded at resolution time.",
    )
    project: str = Field(default="aivc_genelink")
    auto_link: bool = Field(default=True)
    dry_run: bool = Field(default=False)

    @field_validator("vault_path")
    @classmethod
    def _no_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    def resolved_vault(self) -> Path:
        """Expand ~ and resolve. Does NOT create the directory."""
        return Path(self.vault_path).expanduser().resolve()


def init_vault(config: ObsidianConfig) -> None:
    """Create vault subdirectories and stub paper notes.

    Idempotent. Existing files are never clobbered.
    dry_run=True → print plan, no filesystem mutation.
    """
    root = config.resolved_vault()

    if config.dry_run:
        print(f"[DRY RUN] init_vault root={root}")
        for sub in SUBDIRS:
            print(f"[DRY RUN] mkdir -p {root / sub}")
        print(f"[DRY RUN] write {root / 'papers' / 'kang2018.md'}")
        print(f"[DRY RUN] write {root / 'papers' / 'norman2019.md'}")
        return

    root.mkdir(parents=True, exist_ok=True)
    for sub in SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)

    papers = root / "papers"
    kang_path = papers / "kang2018.md"
    norman_path = papers / "norman2019.md"
    if not kang_path.exists():
        kang_path.write_text(_KANG_STUB, encoding="utf-8")
    if not norman_path.exists():
        norman_path.write_text(_NORMAN_STUB, encoding="utf-8")
