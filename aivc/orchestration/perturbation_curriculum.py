"""
Perturbation Curriculum — staged multi-perturbation training.

Adds perturbations incrementally, blocking progression if Pearson r drops.

Stage 1: Kang 2018 only (IFN-B, baseline r=0.873)
Stage 2: + IFN-G from Frangieh (shared JAK-STAT, 2 cytokines)
Stage 3: + ImmPort cytokines (IL-6, TNF-a, W training only)
Stage 4: + Frangieh CRISPR KOs (JAK1-KO, STAT1-KO — causal ground truth)

RULES:
  - Stages advance in order, never skip.
  - Pearson r on Kang 2018 test set must stay >= 0.873 to advance.
  - JAK-STAT recovery must not decrease between stages.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("aivc.orchestration")

BASELINE_R = 0.873


@dataclass
class StageResult:
    """Result of a curriculum stage."""
    stage: int
    pearson_r: float
    jakstat_recovery: int
    ifit1_fold_change: float
    w_jak1_stat1: float = 0.0
    passed: bool = False
    reason: str = ""


class PerturbationCurriculum:
    """
    Manages curriculum training across 4 stages.
    Tracks stage-specific metrics. Blocks progression if r drops.
    """

    STAGES = {
        1: {
            "name": "kang_baseline",
            "datasets": ["kang"],
            "perturbation_ids": [0, 1],
            "description": "Kang 2018 IFN-B only (v1.0 baseline)",
        },
        2: {
            "name": "add_ifng",
            "datasets": ["kang", "frangieh_ifng"],
            "perturbation_ids": [0, 1, 2, 3],
            "description": "Add IFN-G from Frangieh (shared JAK-STAT)",
        },
        3: {
            "name": "add_cytokines",
            "datasets": ["kang", "frangieh_ifng", "immport"],
            "perturbation_ids": [0, 1, 2, 3, 753, 754, 755, 756, 757, 758, 759],
            "description": "Add ImmPort cytokines (IL-6, TNF-a — W training)",
        },
        4: {
            "name": "add_crispr_ko",
            "datasets": ["kang", "frangieh_ifng", "immport", "frangieh_crispr_jak_stat"],
            "perturbation_ids": [
                0, 1, 2, 3, 753, 754, 755, 756, 757, 758, 759,
                "jak1_ko", "stat1_ko", "stat2_ko", "irf9_ko", "jak2_ko",
            ],
            "description": "Add JAK/STAT CRISPR KOs (causal ground truth)",
        },
    }

    def __init__(self):
        self.history = []
        self.current_stage = 0

    def advance_stage(self, current_stage: int, current_pearson_r: float,
                      jakstat_recovery: int = 0,
                      ifit1_fold_change: float = 0.0,
                      w_jak1_stat1: float = 0.0) -> int:
        """
        Advance to next stage if conditions are met.

        Conditions:
          1. current Pearson r >= 0.873
          2. JAK-STAT recovery >= previous stage recovery

        Args:
            current_stage: Stage just completed (1-4).
            current_pearson_r: Pearson r on Kang 2018 test set.
            jakstat_recovery: Number of JAK-STAT genes within 3x FC.
            ifit1_fold_change: Predicted IFIT1 fold change.
            w_jak1_stat1: W[JAK1, STAT1] weight.

        Returns:
            Next stage number, or current_stage if blocked.
        """
        result = StageResult(
            stage=current_stage,
            pearson_r=current_pearson_r,
            jakstat_recovery=jakstat_recovery,
            ifit1_fold_change=ifit1_fold_change,
            w_jak1_stat1=w_jak1_stat1,
        )

        # Check Pearson r
        if current_pearson_r < BASELINE_R:
            result.passed = False
            result.reason = (
                f"Pearson r {current_pearson_r:.4f} < baseline {BASELINE_R}. "
                "Stage advance BLOCKED."
            )
            logger.error(f"  Stage {current_stage}: {result.reason}")
            self.history.append(result)
            return current_stage

        # Check JAK-STAT recovery doesn't decrease
        if self.history:
            prev_js = max(r.jakstat_recovery for r in self.history)
            if jakstat_recovery < prev_js:
                result.passed = False
                result.reason = (
                    f"JAK-STAT recovery {jakstat_recovery} < previous {prev_js}. "
                    "Stage advance BLOCKED."
                )
                logger.warning(f"  Stage {current_stage}: {result.reason}")
                self.history.append(result)
                return current_stage

        # All checks pass
        result.passed = True
        next_stage = min(current_stage + 1, 4)
        result.reason = f"Advanced to stage {next_stage}"
        logger.info(
            f"  Stage {current_stage} PASSED: r={current_pearson_r:.4f}, "
            f"JAK-STAT={jakstat_recovery}/15, IFIT1={ifit1_fold_change:.1f}x, "
            f"W[JAK1,STAT1]={w_jak1_stat1:.4f}"
        )
        self.history.append(result)
        self.current_stage = next_stage
        return next_stage

    def get_stage_config(self, stage: int) -> dict:
        """Get configuration for a given stage."""
        if stage not in self.STAGES:
            raise ValueError(f"Unknown stage {stage}. Valid: 1-4.")
        return self.STAGES[stage]

    def get_report(self) -> str:
        """Generate a curriculum progress report."""
        lines = ["Perturbation Curriculum Report", "=" * 50]
        for result in self.history:
            status = "PASS" if result.passed else "BLOCKED"
            lines.append(
                f"  Stage {result.stage}: {status} | "
                f"r={result.pearson_r:.4f} JS={result.jakstat_recovery}/15 "
                f"IFIT1={result.ifit1_fold_change:.1f}x "
                f"W[J1,S1]={result.w_jak1_stat1:.4f}"
            )
            if not result.passed:
                lines.append(f"    Reason: {result.reason}")
        return "\n".join(lines)
