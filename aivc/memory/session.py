"""
aivc/memory/session.py — Current run state. Cleared between sessions.

Tracks: steps completed, errors, validation results, donor splits.
Donor splits logged here to prevent test donor contamination
across multiple runs in the same session.
"""

import time
from typing import Optional


class SessionMemory:
    """
    Current run state. Cleared between sessions.
    """

    def __init__(self):
        self.steps_completed: list[dict] = []
        self.errors: list[dict] = []
        self.validation_failures: list[dict] = []
        self.test_donors: set[str] = set()  # locked after first split
        self._test_donors_locked: bool = False
        self.start_time: float = time.time()

    def log_step_complete(self, skill_name: str, result) -> None:
        """Log a successfully completed step."""
        self.steps_completed.append({
            "skill_name": skill_name,
            "timestamp": time.time(),
            "elapsed_since_start": time.time() - self.start_time,
            "success": result.success if hasattr(result, "success") else True,
            "warnings": result.warnings if hasattr(result, "warnings") else [],
        })

    def log_error(self, skill_name: str, error_msg: str) -> None:
        """Log an error with skill name and context."""
        self.errors.append({
            "skill_name": skill_name,
            "error": error_msg,
            "timestamp": time.time(),
            "elapsed_since_start": time.time() - self.start_time,
        })

    def log_validation_failure(self, skill_name: str,
                               failed_critics: list[str]) -> None:
        """Log a validation failure."""
        self.validation_failures.append({
            "skill_name": skill_name,
            "failed_critics": failed_critics,
            "timestamp": time.time(),
        })

    def lock_test_donors(self, test_donors: set[str]) -> None:
        """
        Lock the test donor set. Can only be called once per session.
        After locking, no training step may use these donors.
        """
        if self._test_donors_locked:
            raise RuntimeError(
                "Test donors already locked for this session. "
                "Cannot change test set mid-session."
            )
        self.test_donors = set(test_donors)
        self._test_donors_locked = True

    def assert_test_donors_clean(self, candidate_donors: set[str]) -> None:
        """
        Raises ValueError if any candidate donor is in the locked test set.
        Called before training to prevent data leakage.
        """
        if not self._test_donors_locked:
            return  # No test set locked yet
        overlap = set(candidate_donors) & self.test_donors
        if overlap:
            raise ValueError(
                f"DATA LEAKAGE DETECTED: donors {overlap} appear in both "
                f"training candidates and locked test set {self.test_donors}. "
                f"This would invalidate all evaluation results."
            )

    def get_summary(self) -> dict:
        """Return a summary of the session state."""
        return {
            "steps_completed": len(self.steps_completed),
            "errors": len(self.errors),
            "validation_failures": len(self.validation_failures),
            "test_donors_locked": self._test_donors_locked,
            "test_donors": list(self.test_donors),
            "elapsed_seconds": time.time() - self.start_time,
        }
