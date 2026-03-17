"""
aivc/memory/active_learning.py — Uncertainty-driven experiment queue.

The bridge between the model and the wet lab.
Stores high-uncertainty pathways ranked by expected information gain.
This list is the input to the next QuRIE-seq experimental batch.

This is QurieGen's competitive moat in memory form:
the model directs the experiments, the experiments improve the model.
"""

import json
import os
import time
from typing import Optional


class ActiveLearningMemory:
    """
    Stores high-uncertainty pathways ranked by expected information gain.
    This list is the input to the next QuRIE-seq experimental batch.
    """

    STORAGE_PATH = "memory/active_learning_queue.json"

    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or self.STORAGE_PATH
        self._queue: list[dict] = []
        self._measured: list[dict] = []
        self._load()

    def _load(self) -> None:
        """Load active learning queue from disk."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    self._queue = data.get("queue", [])
                    self._measured = data.get("measured", [])
            except (json.JSONDecodeError, IOError):
                self._queue = []
                self._measured = []
        else:
            self._queue = []
            self._measured = []

    def _save(self) -> None:
        """Persist queue to disk."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "queue": self._queue,
                "measured": self._measured,
                "last_updated": time.time(),
            }, f, indent=2, default=str)

    def update_queue(self, uncertainty_result) -> None:
        """
        Takes output from UncertaintyEstimator.
        Ranks by epistemic uncertainty (Gini entropy).
        Saves ranked list with: gene, pathway, uncertainty_score,
        recommended_antibody_addition, priority_rank.
        """
        outputs = uncertainty_result.outputs if hasattr(
            uncertainty_result, "outputs"
        ) else uncertainty_result

        # Extract high-uncertainty recommendations
        recommendations = outputs.get("active_learning_recommendations", [])

        # Sort by uncertainty score descending
        recommendations.sort(
            key=lambda x: x.get("uncertainty_score", 0),
            reverse=True,
        )

        # Assign priority ranks
        for i, rec in enumerate(recommendations):
            rec["priority_rank"] = i + 1
            rec["added_timestamp"] = time.time()

        # Replace queue with fresh rankings
        self._queue = recommendations
        self._save()

    def get_next_experiment_recommendations(self, n: int = 10) -> list[dict]:
        """
        Returns top N pathway recommendations for next QuRIE-seq batch.
        Format matches Thiago's antibody panel specification.
        """
        return self._queue[:n]

    def mark_measured(self, genes: list[str]) -> None:
        """
        Called when QuRIE-seq data arrives.
        Removes measured genes from queue.
        Logs the measurement closure date.
        """
        measured_set = set(genes)
        moved = []
        remaining = []

        for item in self._queue:
            if item.get("gene") in measured_set:
                item["measured_timestamp"] = time.time()
                moved.append(item)
            else:
                remaining.append(item)

        self._measured.extend(moved)
        self._queue = remaining

        # Re-rank remaining items
        for i, item in enumerate(self._queue):
            item["priority_rank"] = i + 1

        self._save()

    def get_queue_size(self) -> int:
        """Return number of items in the active learning queue."""
        return len(self._queue)

    def get_measured_count(self) -> int:
        """Return number of genes that have been measured."""
        return len(self._measured)

    def get_queue(self) -> list[dict]:
        """Return the full queue."""
        return list(self._queue)
