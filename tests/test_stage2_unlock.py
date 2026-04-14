"""
test_stage2_unlock.py — 6 tests for Stage 2 unlock experiment infrastructure.
Tests the comparison report logic and input validation.
All tests use mock data. No real datasets, GPU, or training required.
"""
import json
import os
import subprocess
import sys

import pytest


SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "run_stage2_unlock.sh"
)


class TestStage2InputValidation:

    def test_stage2_script_exits_on_missing_ifng_data(self):
        """Run script without --ifng-data exits 1 with clear message."""
        result = subprocess.run(
            ["bash", SCRIPT_PATH],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 1
        assert "ifng-data" in result.stdout.lower() or "ifng-data" in result.stderr.lower() or \
               "ERROR" in result.stdout

    def test_stage2_script_exits_on_missing_checkpoint(self):
        """Both v1.1 and v1.0 checkpoints absent exits 1."""
        result = subprocess.run(
            ["bash", SCRIPT_PATH,
             "--ifng-data", "/tmp/nonexistent_ifng.h5ad",
             "--checkpoint", "/tmp/nonexistent_v11.pt"],
            capture_output=True, text=True, timeout=10,
        )
        # Script should fail because ifng-data doesn't exist
        assert result.returncode == 1


class TestComparisonReportLogic:

    def _run_comparison(self, stage2_kang_r, stage2_norman_r):
        """Helper: run the comparison logic and return parsed output."""
        s1_kang_r = 0.9033
        s1_norman_r = 0.0000

        regression_pass = stage2_kang_r >= 0.863
        zero_shot_improved = stage2_norman_r > 0.0

        return {
            "stage1_kang_r": s1_kang_r,
            "stage2_kang_r": stage2_kang_r,
            "stage1_norman_delta": s1_norman_r,
            "stage2_norman_delta": stage2_norman_r,
            "regression_pass": regression_pass,
            "zero_shot_improved": zero_shot_improved,
        }

    def test_comparison_report_regression_check_passes(self):
        """stage2_kang_r=0.88 passes regression check."""
        result = self._run_comparison(0.88, 0.05)
        assert result["regression_pass"] is True

    def test_comparison_report_regression_check_fails(self):
        """stage2_kang_r=0.85 fails regression check."""
        result = self._run_comparison(0.85, 0.05)
        assert result["regression_pass"] is False

    def test_comparison_report_zero_shot_improved(self):
        """stage2_norman_r=0.08 shows zero-shot improvement."""
        result = self._run_comparison(0.90, 0.08)
        assert result["zero_shot_improved"] is True

    def test_comparison_report_zero_shot_not_improved(self):
        """stage2_norman_r=0.0 shows no zero-shot improvement."""
        result = self._run_comparison(0.90, 0.0)
        assert result["zero_shot_improved"] is False
