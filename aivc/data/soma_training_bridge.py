"""
Bridge between SOMA store and existing AIVC training loop.

Enables gradual migration:
  Stage 1 (now): load from SOMA, return numpy arrays (same interface)
  Stage 2 (May): train_v11.py reads directly from SOMA DataLoader
  Stage 3 (June): remove numpy array intermediaries

Controlled by USE_SOMA_STORE flag (default False).
"""
import logging

import numpy as np

logger = logging.getLogger("aivc.data")

USE_SOMA_STORE = False  # Set True once SOMA store is validated


def load_training_arrays(
    store_path: str = None,
    h5ad_path: str = "data/kang2018_pbmc_fixed.h5ad",
    gene_universe: list = None,
    obs_query: str = None,
    use_soma: bool = None,
) -> dict:
    """
    Load training data from SOMA store or legacy h5ad.

    Both paths return identical data structures.

    Returns dict with: adata, X, obs, gene_names, n_cells, n_genes, source, store_path
    """
    _use_soma = use_soma if use_soma is not None else USE_SOMA_STORE

    if _use_soma:
        return _load_from_soma(store_path, gene_universe, obs_query)
    else:
        return _load_from_h5ad(h5ad_path, gene_universe)


def _load_from_h5ad(h5ad_path: str, gene_universe: list = None) -> dict:
    """Load from legacy h5ad (current behaviour)."""
    import anndata as ad
    import scipy.sparse as sp

    logger.info(f"Loading from h5ad: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)

    if gene_universe is not None:
        common = [g for g in gene_universe if g in adata.var_names]
        adata = adata[:, common].copy()

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()

    return {
        "adata": adata,
        "X": X.astype(np.float32),
        "obs": adata.obs,
        "gene_names": adata.var_names.tolist(),
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "source": "h5ad",
        "store_path": h5ad_path,
    }


def _load_from_soma(
    store_path: str,
    gene_universe: list = None,
    obs_query: str = None,
) -> dict:
    """Load from SOMA store with column pruning."""
    from aivc.data.soma_store import AIVCSomaStore
    import scipy.sparse as sp

    if store_path is None:
        raise ValueError("store_path is required when use_soma=True")

    logger.info(f"Loading from SOMA store: {store_path}")

    var_query = "is_hvg == True" if gene_universe is not None else None

    with AIVCSomaStore(store_path) as store:
        q = store.query(obs_query=obs_query, var_query=var_query)
        adata = q.to_anndata(X_name="raw")

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()

    logger.info(f"SOMA load complete: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    return {
        "adata": adata,
        "X": X.astype(np.float32),
        "obs": adata.obs,
        "gene_names": adata.var_names.tolist(),
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "source": "soma",
        "store_path": store_path,
    }


def benchmark_load_time(
    h5ad_path: str,
    store_path: str = None,
    n_repeats: int = 3,
) -> dict:
    """
    Benchmark load time: h5ad vs SOMA for the same dataset.

    Returns dict with h5ad_mean_s, soma_mean_s, speedup.
    """
    import time

    h5ad_times = []
    for _ in range(n_repeats):
        t = time.time()
        _load_from_h5ad(h5ad_path)
        h5ad_times.append(time.time() - t)

    soma_times = []
    if store_path:
        for _ in range(n_repeats):
            t = time.time()
            _load_from_soma(store_path)
            soma_times.append(time.time() - t)

    h5ad_mean = np.mean(h5ad_times)
    soma_mean = np.mean(soma_times) if soma_times else None
    speedup = h5ad_mean / soma_mean if soma_mean else None

    return {
        "h5ad_mean_s": h5ad_mean,
        "soma_mean_s": soma_mean,
        "speedup": speedup,
    }
