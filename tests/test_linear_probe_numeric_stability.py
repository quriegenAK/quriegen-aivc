"""Regression tests for the Phase 6.5c LOCKED v2 contract.

Contract under test (see prompts/phase6_5c_fix.md):

* Per-run train-split variance filter: ``mask_tr = Y_log1p[tr].std(0) > 1e-7``.
* Mask applied symmetrically to Ridge fit and to scoring.
* DE indices are remapped from original gene space into masked space.
* ``z_sd`` floor assertion (encoder-collapse tripwire) preserved.
* ``|r²_top50_de_vw| < 2.0`` tripwire preserved.
* ``y_sd`` assertion and ``MIN_TRAIN_NONZERO`` filter REMOVED.
* Per-run artifacts persisted to ``experiments/phase6_5c/artifacts``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


_SEED = 17
_N_CELLS = 512
_N_GENES = 128


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


class _StubAnnData:
    """Minimal AnnData shim that satisfies ``_load_dataset``'s interface."""

    def __init__(self, X: np.ndarray, uns: dict | None = None):
        self.X = X
        self.obs = pd.DataFrame({})  # no 'perturbation' column → zeros
        self.uns = uns or {}


def _make_train_split(n_cells: int = _N_CELLS, seed: int = _SEED):
    """Exact replica of run_condition's train/test split (seed-local RNG)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_cells)
    split = int(0.8 * n_cells)
    return idx[:split], idx[split:]


def _make_fixture_with_train_zero_cols(
    n_cells: int = _N_CELLS,
    n_genes: int = _N_GENES,
    seed: int = _SEED,
    forced_cols=(0, 1, 2),
):
    """Build a Poisson(λ=0.3) count matrix with ``forced_cols`` zeroed
    in the train split. A single test-side nonzero is placed in each so
    the upstream structural-zero filter in ``_load_dataset`` keeps them
    — the per-run train-variance mask is what must drop them.
    """
    rng = np.random.default_rng(seed)
    X = rng.poisson(lam=0.3, size=(n_cells, n_genes)).astype(np.float32)
    tr, te = _make_train_split(n_cells=n_cells, seed=seed)
    for i, col in enumerate(forced_cols):
        X[tr, col] = 0.0
        X[te[i], col] = 1.0
    return X, tr, te, list(forced_cols)


# --------------------------------------------------------------------- #
# (a) mask drops all-train-zero columns symmetrically (fit + score)
# --------------------------------------------------------------------- #
def test_mask_drops_train_zero_columns(monkeypatch):
    import scripts.linear_probe_pretrain as lpp

    X, _tr, _te, forced_cols = _make_fixture_with_train_zero_cols()
    monkeypatch.setattr(lpp, "_load_anndata", lambda p: _StubAnnData(X))

    m = lpp.run_condition(
        condition="scratch",
        ckpt_path=None,
        dataset_name="test_mask",
        dataset_path=Path(__file__),
        seed=_SEED,
        hidden_dim=32,
        latent_dim=16,
        n_genes_fallback=_N_GENES,
    )

    # forced_cols must be dropped; natural all-train-zero in Poisson(0.3)×409
    # is astronomically rare, so the lower-bound assertion is tight.
    assert m["n_dropped"] >= len(forced_cols), (
        f"n_dropped={m['n_dropped']} < forced={len(forced_cols)}"
    )
    assert m["n_kept"] + m["n_dropped"] == _N_GENES
    assert 0.0 <= m["pct_dropped"] < 1.0

    # Cross-check via the persisted artifact: forced cols must be False
    # in mask_tr at their original gene-space indices.
    art = np.load(
        _repo_root() / "experiments/phase6_5c/artifacts"
                     / f"run_random_seed{_SEED}.npz",
        allow_pickle=True,
    )
    mask = art["mask_tr"]
    for col in forced_cols:
        assert not mask[col], (
            f"forced-train-zero col {col} survived mask_tr"
        )


# --------------------------------------------------------------------- #
# (b) DE index remapping into masked space
# --------------------------------------------------------------------- #
def test_de_index_remapping_respects_mask(monkeypatch):
    import scripts.linear_probe_pretrain as lpp

    X, _tr, _te, forced_cols = _make_fixture_with_train_zero_cols()
    # Include all 3 forced-train-zero indices (must be dropped) and 6
    # regular indices (must survive). n_de_kept >= 5 required by the
    # LOCKED v2 tripwire in run_condition.
    uns = {
        "top50_de_per_perturbation": {
            "A": [0, 1, 10, 20, 50],
            "B": [2, 30, 40, 60, 70],
        }
    }
    monkeypatch.setattr(lpp, "_load_anndata",
                        lambda p: _StubAnnData(X, uns=uns))
    m = lpp.run_condition(
        condition="scratch",
        ckpt_path=None,
        dataset_name="test_deremap",
        dataset_path=Path(__file__),
        seed=_SEED,
        hidden_dim=32,
        latent_dim=16,
        n_genes_fallback=_N_GENES,
    )

    union = sorted({0, 1, 2, 10, 20, 30, 40, 50, 60, 70})
    assert m["n_de_total"] == len(union)
    # Exactly the 3 forced-train-zero DE genes must be excluded.
    assert m["n_de_kept"] == len(union) - len(forced_cols)
    assert np.isfinite(m["r2_top50_de_vw"])
    assert abs(m["r2_top50_de_vw"]) < 2.0


# --------------------------------------------------------------------- #
# (c) z_sd assertion still fires on collapsed encoder latents
# --------------------------------------------------------------------- #
def test_z_sd_assertion_fires_on_collapsed_latents(monkeypatch):
    import scripts.linear_probe_pretrain as lpp

    X, *_ = _make_fixture_with_train_zero_cols()
    monkeypatch.setattr(lpp, "_load_anndata", lambda p: _StubAnnData(X))

    # Collapsed latents: std is 1e-8 (floor), below the 1e-6 tripwire.
    def _collapsed_latents(encoder, X_in, device="cpu"):
        return np.zeros((X_in.shape[0], 16), dtype=np.float32)

    monkeypatch.setattr(lpp, "_extract_latents", _collapsed_latents)

    with pytest.raises(AssertionError, match="z_sd floor violated"):
        lpp.run_condition(
            condition="scratch",
            ckpt_path=None,
            dataset_name="test_zsd_trip",
            dataset_path=Path(__file__),
            seed=_SEED,
            hidden_dim=32,
            latent_dim=16,
            n_genes_fallback=_N_GENES,
        )


# --------------------------------------------------------------------- #
# (d) |r²| < 2.0 tripwire fires on pathological r²
# --------------------------------------------------------------------- #
def test_r2_tripwire_fires_on_pathological_value(monkeypatch):
    import sklearn.metrics as skm
    import scripts.linear_probe_pretrain as lpp

    X, *_ = _make_fixture_with_train_zero_cols()
    monkeypatch.setattr(lpp, "_load_anndata", lambda p: _StubAnnData(X))

    original_r2 = skm.r2_score
    calls = {"n": 0}

    def _bad_r2(y_true, y_pred, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # First call is r2_top50_de_vw — force pathological value.
            return 10.0
        return original_r2(y_true, y_pred, **kwargs)

    monkeypatch.setattr(skm, "r2_score", _bad_r2)

    with pytest.raises(AssertionError, match=r"r²_top50_de_vw"):
        lpp.run_condition(
            condition="scratch",
            ckpt_path=None,
            dataset_name="test_r2_trip",
            dataset_path=Path(__file__),
            seed=_SEED,
            hidden_dim=32,
            latent_dim=16,
            n_genes_fallback=_N_GENES,
        )


# --------------------------------------------------------------------- #
# (e) per-run artifact written with expected keys + dtypes
# --------------------------------------------------------------------- #
def test_artifact_file_written_with_expected_keys(monkeypatch):
    import scripts.linear_probe_pretrain as lpp

    X, *_ = _make_fixture_with_train_zero_cols()
    monkeypatch.setattr(lpp, "_load_anndata", lambda p: _StubAnnData(X))

    m = lpp.run_condition(
        condition="scratch",
        ckpt_path=None,
        dataset_name="test_artifact",
        dataset_path=Path(__file__),
        seed=_SEED,
        hidden_dim=32,
        latent_dim=16,
        n_genes_fallback=_N_GENES,
    )

    art_path = (
        _repo_root() / "experiments/phase6_5c/artifacts"
                      / f"run_random_seed{_SEED}.npz"
    )
    assert art_path.exists(), f"artifact not written: {art_path}"

    arr = np.load(art_path, allow_pickle=True)
    expected = {
        "mask_tr", "te_idx", "Y_hat_log1p_te_full",
        "de_idx_orig", "seed", "arm",
    }
    assert set(arr.files) == expected, (
        f"artifact keys mismatch: {set(arr.files) ^ expected}"
    )
    assert int(arr["seed"]) == _SEED
    assert str(arr["arm"]) == "random"
    assert arr["mask_tr"].dtype == bool
    assert arr["Y_hat_log1p_te_full"].shape[1] == _N_GENES
    # Un-masked positions must be NaN (the _scatter_back invariant).
    pred = arr["Y_hat_log1p_te_full"]
    assert np.isnan(pred[:, ~arr["mask_tr"]]).all()
    assert np.isfinite(pred[:, arr["mask_tr"]]).all()
    # Return-value/metric plumbing sanity.
    assert m["arm"] == "random"
    assert m["seed"] == _SEED
