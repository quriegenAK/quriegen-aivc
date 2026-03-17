"""
aivc/skills/ — All callable skill modules.

Importing this package triggers registration of all skills
with the global registry via their @registry.register decorators.
"""

from aivc.skills.preprocessing import ScRNAPreprocessor
from aivc.skills.graph_builder import GraphBuilder
from aivc.skills.ot_pairing import OTCellPairer
from aivc.skills.gat_training import GATTrainer
from aivc.skills.perturbation import PerturbationPredictor
from aivc.skills.attention import AttentionExtractor
from aivc.skills.uncertainty import UncertaintyEstimator
from aivc.skills.plausibility import BiologicalPlausibilityScorer
from aivc.skills.evaluation import BenchmarkEvaluator
from aivc.skills.reporting import TwoAudienceRenderer

__all__ = [
    "ScRNAPreprocessor",
    "GraphBuilder",
    "OTCellPairer",
    "GATTrainer",
    "PerturbationPredictor",
    "AttentionExtractor",
    "UncertaintyEstimator",
    "BiologicalPlausibilityScorer",
    "BenchmarkEvaluator",
    "TwoAudienceRenderer",
]
