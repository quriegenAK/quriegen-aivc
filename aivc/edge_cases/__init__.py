"""
aivc/edge_cases/ — Failure mode handlers.
"""

from aivc.edge_cases.handlers import (
    MissingDataHandler,
    NoisySignalHandler,
    ModelFailureHandler,
    DistributionShiftHandler,
    AmbiguousQueryHandler,
)

__all__ = [
    "MissingDataHandler",
    "NoisySignalHandler",
    "ModelFailureHandler",
    "DistributionShiftHandler",
    "AmbiguousQueryHandler",
]
