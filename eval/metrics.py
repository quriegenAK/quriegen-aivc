"""
eval/metrics.py — Pure metric functions for AIVC evaluation.

NumPy only. No model I/O. No torch imports.

All functions accept arrays of shape (n_cells, n_genes) representing
pseudo-bulk expression profiles.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("aivc.eval.metrics")

DEFAULT_DELTA_EPSILON: float = 1e-6


def _per_cell_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row Pearson correlation between two (n_cells, n_genes) arrays.

    Rows where either side has std < 1e-10 return 0.0 (matches train_v11.py).

    Returns:
        (n_cells,) array of per-row Pearson r values.
    """
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    if a.ndim == 1:
        a = a.reshape(1, -1)
        b = b.reshape(1, -1)

    a_mean = a.mean(axis=1, keepdims=True)
    b_mean = b.mean(axis=1, keepdims=True)
    a_centered = a - a_mean
    b_centered = b - b_mean

    a_std = a.std(axis=1)
    b_std = b.std(axis=1)

    # Rows with near-zero std on either side → 0.0
    valid = (a_std > 1e-10) & (b_std > 1e-10)

    numerator = (a_centered * b_centered).sum(axis=1)
    denominator = a_std * b_std * a.shape[1]

    r = np.where(valid, numerator / np.maximum(denominator, 1e-30), 0.0)
    # Clamp NaN to 0.0
    r = np.where(np.isnan(r), 0.0, r)
    return r


def pearson_r_ctrl_subtracted(
    pred: np.ndarray,
    ctrl: np.ndarray,
    truth: np.ndarray,
) -> float:
    """Per-cell mean Pearson on (pred - ctrl) vs (truth - ctrl).

    Args:
        pred:  (n_cells, n_genes) predicted expression.
        ctrl:  (n_cells, n_genes) control expression.
        truth: (n_cells, n_genes) ground-truth stimulated expression.

    Returns:
        Mean Pearson r across cells, or float('nan') if truth delta is
        degenerate (zero std across all entries).
    """
    pred_delta = pred - ctrl
    truth_delta = truth - ctrl

    # Check for degenerate ground truth delta
    if truth_delta.std() < 1e-10:
        logger.warning("degenerate ground truth delta — all genes at ctrl level")
        return float("nan")

    rs = _per_cell_pearson(pred_delta, truth_delta)
    return float(np.mean(rs))


def delta_nonzero_pct(
    pred: np.ndarray,
    ctrl: np.ndarray,
    epsilon: float = DEFAULT_DELTA_EPSILON,
) -> float:
    """Percentage of (gene, cell) entries where |pred - ctrl| > epsilon.

    Returns:
        Float in [0.0, 100.0].
    """
    delta = np.abs(pred - ctrl)
    nonzero = (delta > epsilon).sum()
    total = delta.size
    if total == 0:
        return 0.0
    return float(nonzero / total * 100.0)


def ctrl_memorisation_score(
    pred: np.ndarray,
    ctrl: np.ndarray,
) -> float:
    """Mean cosine similarity between pred and ctrl across cells.

    Assumes non-negative log1p expression.

    Returns:
        Float in [0, 1]. 1.0 = perfect memorisation (pred == ctrl).
    """
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
        ctrl = ctrl.reshape(1, -1)

    scores = []
    for i in range(pred.shape[0]):
        p_norm = np.linalg.norm(pred[i])
        c_norm = np.linalg.norm(ctrl[i])
        if p_norm < 1e-30 or c_norm < 1e-30:
            scores.append(0.0)
            continue
        cos = np.dot(pred[i], ctrl[i]) / (p_norm * c_norm)
        scores.append(float(np.clip(cos, 0.0, 1.0)))
    return float(np.mean(scores))


def top_k_gene_overlap(
    pred_delta: np.ndarray,
    truth_delta: np.ndarray,
    k: int = 20,
) -> float:
    """Jaccard overlap of top-k upregulated genes. Supplementary metric.

    Computes mean delta across cells, then ranks genes by magnitude.

    Args:
        pred_delta:  (n_cells, n_genes) predicted delta (pred - ctrl).
        truth_delta: (n_cells, n_genes) ground-truth delta (truth - ctrl).
        k: Number of top genes to compare.

    Returns:
        Jaccard index in [0.0, 1.0].
    """
    pred_mean = pred_delta.mean(axis=0) if pred_delta.ndim > 1 else pred_delta
    truth_mean = truth_delta.mean(axis=0) if truth_delta.ndim > 1 else truth_delta

    n_genes = pred_mean.shape[0]
    k = min(k, n_genes)

    pred_top_k = set(np.argsort(np.abs(pred_mean))[-k:])
    truth_top_k = set(np.argsort(np.abs(truth_mean))[-k:])

    intersection = len(pred_top_k & truth_top_k)
    union = len(pred_top_k | truth_top_k)
    if union == 0:
        return 0.0
    return float(intersection / union)
