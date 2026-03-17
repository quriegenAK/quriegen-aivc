"""
aivc/critics/ — Independent validation layer.

Three critics validate every SkillResult:
  - StatisticalCritic: numerical correctness
  - MethodologicalCritic: scientific methodology
  - BiologicalCritic: biological plausibility
"""

from aivc.critics.statistical import StatisticalCritic
from aivc.critics.methodological import MethodologicalCritic
from aivc.critics.biological import BiologicalCritic

__all__ = ["StatisticalCritic", "MethodologicalCritic", "BiologicalCritic"]
