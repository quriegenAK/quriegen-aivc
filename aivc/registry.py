"""
aivc/registry.py — Skill registration and discovery system.

Singleton registry. All skills register here at import time.
The orchestrator queries this registry to discover available skills.

Extension mechanism: to add a new skill (e.g., ATAC-seq, IL-6 perturbation),
simply create a new file in aivc/skills/, decorate the class with
@registry.register(...), and it becomes available to all workflows.
Zero changes to orchestrator, critics, or memory layers required.
"""

from aivc.interfaces import (
    AIVCSkill, BiologicalDomain, ComputeProfile,
)


class AIVCSkillRegistry:
    """
    Singleton registry. All skills register here at import time.
    The orchestrator queries this registry to discover available skills.
    """

    def __init__(self):
        self._skills: dict[str, AIVCSkill] = {}
        self._metadata: dict[str, dict] = {}

    def register(self, name: str, domain: BiologicalDomain,
                 version: str, requires: list[str],
                 compute_profile: ComputeProfile):
        """
        Decorator factory. Usage:

            @registry.register(
                name="scRNA_preprocessor",
                domain=BiologicalDomain.TRANSCRIPTOMICS,
                version="1.0.0",
                requires=["adata_path"],
                compute_profile=ComputeProfile.CPU_LIGHT
            )
            class ScRNAPreprocessor(AIVCSkill):
                ...
        """
        def decorator(cls):
            if name in self._skills:
                raise ValueError(
                    f"Skill '{name}' already registered. "
                    f"Duplicate registration is not allowed."
                )
            # Set class attributes from registration metadata
            cls.name = name
            cls.version = version
            cls.biological_domain = domain
            cls.compute_profile = compute_profile

            # Instantiate and store
            instance = cls()
            self._skills[name] = instance
            self._metadata[name] = {
                "name": name,
                "domain": domain.value,
                "version": version,
                "requires": requires,
                "profile": compute_profile.value,
            }
            return cls

        return decorator

    def get(self, name: str) -> AIVCSkill:
        """Return skill by name. Raises KeyError if not found."""
        if name not in self._skills:
            raise KeyError(
                f"Skill '{name}' not found in registry. "
                f"Available skills: {list(self._skills.keys())}"
            )
        return self._skills[name]

    def list_skills(self, domain: BiologicalDomain = None,
                    profile: ComputeProfile = None) -> list[dict]:
        """List all registered skills with optional filtering."""
        results = []
        for name, meta in self._metadata.items():
            if domain is not None and meta["domain"] != domain.value:
                continue
            if profile is not None and meta["profile"] != profile.value:
                continue
            results.append(meta)
        return results

    def check_requirements(self, name: str,
                           available_data: dict) -> tuple[bool, list[str]]:
        """
        Check if all required data is available before dispatching.
        Returns (can_run, list_of_missing_requirements)
        """
        if name not in self._metadata:
            return False, [f"Skill '{name}' not found in registry"]
        required = self._metadata[name]["requires"]
        missing = [r for r in required if r not in available_data]
        return len(missing) == 0, missing

    def get_requires(self, name: str) -> list[str]:
        """Return the list of required inputs for a skill."""
        if name not in self._metadata:
            raise KeyError(f"Skill '{name}' not found in registry")
        return self._metadata[name]["requires"]


# Global singleton
registry = AIVCSkillRegistry()
