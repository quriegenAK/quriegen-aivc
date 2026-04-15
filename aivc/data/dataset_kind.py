"""
DatasetKind enum — distinguishes interventional vs observational data sources.

INTERVENTIONAL: paired ctrl/stim (or ctrl/KO) data; drives joint-stage training.
OBSERVATIONAL:  unpaired expression; reserved for pretrain stage.

Phase 2 wires the enum through the loader + train-loop dispatch without
introducing any observational datasets. The OBSERVATIONAL branch is not
yet reachable but is in place for future phases.
"""
from enum import Enum


class DatasetKind(str, Enum):
    INTERVENTIONAL = "interventional"
    OBSERVATIONAL = "observational"
