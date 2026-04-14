"""
eval/exceptions.py — Exceptions for the AIVC evaluation pipeline.
"""


class CheckpointRejected(Exception):
    """
    Raised when Kang regression guard fails (pearson_r < 0.873).
    Never caught inside eval_runner — propagates to post_run_hook.
    A rejected checkpoint is never registered regardless of Norman score.
    """


class EvalDataBlocked(Exception):
    """
    Reserved for Replogle data governance enforcement (v1.2).
    Raised when a dataset outside the cleared safe-set is passed to eval.
    """
