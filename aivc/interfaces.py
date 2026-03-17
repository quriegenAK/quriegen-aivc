"""
aivc/interfaces.py — Base classes and schemas for the AIVC platform.

All skills, critics, and orchestration components depend on these definitions.
No external dependencies beyond standard library and dataclasses.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum
import time


class BiologicalDomain(Enum):
    TRANSCRIPTOMICS = "transcriptomics"
    PROTEOMICS = "proteomics"
    EPIGENOMICS = "epigenomics"
    MULTIMODAL = "multimodal"
    EVALUATION = "evaluation"
    REPORTING = "reporting"


class ComputeProfile(Enum):
    CPU_LIGHT = "cpu_light"          # < 1 minute, no GPU needed
    GPU_REQUIRED = "gpu_required"    # requires GPU, < 10 minutes
    GPU_INTENSIVE = "gpu_intensive"  # requires GPU, > 10 minutes


@dataclass
class SkillResult:
    skill_name: str
    version: str
    success: bool
    outputs: dict
    metadata: dict          # timing, GPU used, dataset size
    warnings: list[str]     # non-fatal issues
    errors: list[str]       # fatal issues
    timestamp: float = field(default_factory=time.time)


@dataclass
class ValidationReport:
    passed: bool
    critic_name: str
    checks_passed: list[str]
    checks_failed: list[str]
    biological_score: Optional[float] = None  # 0 to 1
    uncertainty_flags: list[str] = field(default_factory=list)
    quarantined_outputs: list[str] = field(default_factory=list)
    recommendation: Optional[str] = None


@dataclass
class ComputeCost:
    estimated_minutes: float
    gpu_memory_gb: float
    profile: ComputeProfile
    estimated_usd: float    # based on $2/hr H100 estimate
    can_run_on_cpu: bool


@dataclass
class WorkflowResult:
    success: bool
    outputs: Optional[dict] = None
    error_message: Optional[str] = None
    failed_step: Optional[str] = None
    diagnostic_report: Optional[dict] = None
    total_cost: float = 0.0
    pearson_r: float = 0.0
    demo_ready: bool = False
    output_paths: list[str] = field(default_factory=list)

    @classmethod
    def failed(cls, message: str, step: str = None,
               diagnostic: dict = None) -> "WorkflowResult":
        return cls(
            success=False,
            error_message=message,
            failed_step=step,
            diagnostic_report=diagnostic,
        )

    @classmethod
    def success(cls, outputs: dict, total_cost: float) -> "WorkflowResult":
        return cls(
            success=True,
            outputs=outputs,
            total_cost=total_cost,
            pearson_r=outputs.get("pearson_r_mean", 0.0),
            demo_ready=outputs.get("demo_ready", False),
            output_paths=outputs.get("output_paths", []),
        )


class AIVCSkill:
    """
    Base class for all AIVC skills.
    Every skill must implement execute(), validate(), and estimate_cost().
    """
    name: str = "base_skill"
    version: str = "0.0.0"
    biological_domain: BiologicalDomain = BiologicalDomain.TRANSCRIPTOMICS
    compute_profile: ComputeProfile = ComputeProfile.CPU_LIGHT
    validation_required: bool = True
    uncertainty_aware: bool = False

    def execute(self, inputs: dict, context: Any) -> SkillResult:
        raise NotImplementedError(
            f"{self.name} v{self.version}: execute() not implemented"
        )

    def validate(self, result: SkillResult) -> ValidationReport:
        raise NotImplementedError(
            f"{self.name} v{self.version}: validate() not implemented"
        )

    def estimate_cost(self, inputs: dict) -> ComputeCost:
        raise NotImplementedError(
            f"{self.name} v{self.version}: estimate_cost() not implemented"
        )

    def _check_inputs(self, inputs: dict, required: list[str]) -> None:
        """
        Called at the start of every execute().
        Raises ValueError with a precise message if any required key is missing.
        Never silently proceed with missing inputs.
        """
        missing = [k for k in required if k not in inputs]
        if missing:
            raise ValueError(
                f"{self.name} v{self.version}: "
                f"missing required inputs: {missing}"
            )
