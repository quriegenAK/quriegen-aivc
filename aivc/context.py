"""
aivc/context.py — SessionContext wrapping all memory stores.

Passed to every skill.execute() call to provide access to session state,
experimental history, biological knowledge, and active learning queue.
"""

from dataclasses import dataclass
from aivc.memory.session import SessionMemory
from aivc.memory.experimental import ExperimentalMemory
from aivc.memory.knowledge import KnowledgeBase
from aivc.memory.active_learning import ActiveLearningMemory


@dataclass
class MemoryStores:
    """Container for all memory stores."""
    session: SessionMemory
    experimental: ExperimentalMemory
    active_learning: ActiveLearningMemory
    knowledge: KnowledgeBase


@dataclass
class SessionContext:
    """
    Session context passed to every skill execution.
    Wraps all memory stores and provides convenience accessors.
    """
    memory: MemoryStores
    device: str = "cpu"
    random_seed: int = 42
    data_dir: str = "data/"

    @classmethod
    def create_default(cls, data_dir: str = "data/",
                       device: str = "cpu") -> "SessionContext":
        """Create a SessionContext with default memory stores."""
        return cls(
            memory=MemoryStores(
                session=SessionMemory(),
                experimental=ExperimentalMemory(),
                active_learning=ActiveLearningMemory(),
                knowledge=KnowledgeBase(),
            ),
            device=device,
            data_dir=data_dir,
        )
