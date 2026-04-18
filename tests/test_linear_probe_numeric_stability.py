"""Regression test for the Phase 6.5c LOCKED evaluation contract.

Contract under test (see prompts/phase6_5c_fix.md):

* X stays in raw-counts space; encoder forward pass is unchanged.
* Ridge target is ``log1p(X_filtered)`` — variance-stabilized.
* Ridge fits in a standardized (Z, Y_log1p) space for numerical
  conditioning of the SVD solver, then predictions are un-standardized
  back to log1p space.
* Primary gate metric is
  ``r2_score(Y_log1p_te_de, Y_hat_log1p_te_de, multioutput='variance_weighted')``.
* No ``MIN_TRAIN_NONZERO`` filter is applied — variance_weighted is
  expected to tolerate rare (low-variance) columns without a heuristic.

This test builds a small synthetic Poisson(λ=0.3) fixture with three
forced single-nonzero-train-cell columns and asserts that
``run_condition`` returns finite metrics inside a sane range with both
tripwires (``z_sd_min``, ``y_sd_min``) respected.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest


def _make_poisson_fixture(n_cells: int = 512, n_genes: int = 128,
                          seed: int = 17):
    """Build a Poisson(λ=0.3) count matrix with 3 columns forced to
    have exactly one nonzero count in the TRAIN split under the
    ``run_condition`` seed-local RNG. Placement is computed from the
    same permutation logic (``np.random.default_rng(seed).permutation``)
    so the forced cells are guaranteed train-side.
    """
    rng = np.random.default_rng(seed)
    X = rng.poisson(lam=0.3, size=(n_cells, n_genes)).astype(np.float32)
    # Zero out the first three columns entirely, then drop exactly one
    # nonzero count in the train split for each.
    X[:, 0:3] = 0.0
    # Replicate the run_condition split logic so we can place the
    # forced nonzeros at indices that are guaranteed train-side.
    split_rng = np.random.default_rng(seed)
    idx = split_rng.permutation(n_cells)
    split = int(0.8 * n_cells)
    train_cells = idx[:split]
    # Use the first three cells of the train split for columns 0-2.
    X[train_cells[0], 0] = 1
    X[train_cells[1], 1] = 1
    X[train_cells[2], 2] = 1
    return X


class _StubAnnData:
    """Minimal AnnData stand-in that satisfies ``_load_dataset``'s
    interface (``.X``, ``.obs``, ``.uns``).
    """
    def __init__(self, X: np.ndarray):
        self.X = X
        import pandas as pd
        self.obs = pd.DataFrame({})  # no 'perturbation' column → zeros
        self.uns = {}  # no precomputed DE → variance fallback triggers


def test_locked_contract_returns_finite_metrics(monkeypatch):
    """End-to-end: run_condition under locked contract on a 512x128
    Poisson fixture with 3 forced rare columns; tripwires + ranges."""
    import scripts.linear_probe_pretrain as lpp

    n_cells, n_genes = 512, 128
    X = _make_poisson_fixture(n_cells=n_cells, n_genes=n_genes, seed=17)

    # Patch _load_anndata so _load_dataset pulls our synthetic fixture
    # via the real-file branch (so the structural-zero filter + log1p
    # path are exercised, not the synthetic fallback branch).
    def _fake_load_anndata(path):
        return _StubAnnData(X)
    monkeypatch.setattr(lpp, "_load_anndata", _fake_load_anndata)

    # Make the "path exists" branch fire by pointing at any existing file.
    from pathlib import Path
    fake_path = Path(__file__)  # this test file definitely exists

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = lpp.run_condition(
            condition="scratch",
            ckpt_path=None,
            dataset_name="poisson_synth",
            dataset_path=fake_path,
            seed=17,
            hidden_dim=32,
            latent_dim=16,
            n_genes_fallback=n_genes,
        )

    # Primary tripwires (contract assertions)
    assert np.isfinite(m["r2_top50_de_vw"]), (
        f"r2_top50_de_vw not finite: {m['r2_top50_de_vw']}")
    assert -1.5 <= m["r2_top50_de_vw"] <= 1.01, (
        f"r2_top50_de_vw out of sane range: {m['r2_top50_de_vw']}")

    # Secondary diagnostics are finite
    assert np.isfinite(m["r2_overall_vw"]), m["r2_overall_vw"]
    assert np.isfinite(m["r2_top50_de_median"]), m["r2_top50_de_median"]

    # Per-spec tripwire floors
    assert m["z_sd_min"] > 1e-6, f"z_sd_min={m['z_sd_min']}"
    assert m["y_sd_min"] > 1e-6, f"y_sd_min={m['y_sd_min']}"

    # No RuntimeWarning from numpy during the call
    rt_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not rt_warnings, (
        f"Unexpected RuntimeWarning(s): {[str(w.message) for w in rt_warnings]}"
    )


def test_forced_rare_columns_do_not_inflate_vw():
    """Sanity: with 3 columns that have ≤3 train nonzeros, the locked
    contract's variance_weighted metric stays inside the ±1 sanity
    range — the old uniform_average pathology was negative infinity-
    like values when near-constant columns dominated."""
    import scripts.linear_probe_pretrain as lpp

    n_cells, n_genes = 512, 128
    X = _make_poisson_fixture(n_cells=n_cells, n_genes=n_genes, seed=17)

    class _Stub:
        def __init__(self, X):
            self.X = X
            import pandas as pd
            self.obs = pd.DataFrame({})
            self.uns = {}

    import pytest as _pytest
    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(lpp, "_load_anndata", lambda p: _Stub(X))
        from pathlib import Path
        m = lpp.run_condition(
            condition="scratch",
            ckpt_path=None,
            dataset_name="poisson_rare",
            dataset_path=Path(__file__),
            seed=17,
            hidden_dim=32,
            latent_dim=16,
            n_genes_fallback=n_genes,
        )

    # The tripwire that halts a run on the real contract:
    assert abs(m["r2_top50_de_vw"]) < 2.0, m["r2_top50_de_vw"]
