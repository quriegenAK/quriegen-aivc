"""Phase 4: hard-assert dataset_kind stamping in build_combined_corpus."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aivc.data.dataset_kind import DatasetKind
from aivc.data.multi_perturbation_loader import MultiPerturbationLoader


def _stub_adata(n_cells=5, n_genes=6, stamp_kind=True, dataset_id=0):
    import anndata as ad
    X = np.random.RandomState(0).rand(n_cells, n_genes).astype(np.float32)
    obs = pd.DataFrame({
        "perturbation_id": np.zeros(n_cells, dtype=int),
        "dataset_id": np.full(n_cells, dataset_id, dtype=int),
        "donor_id": [f"d{i}" for i in range(n_cells)],
        "in_test_set": np.zeros(n_cells, dtype=bool),
        "USE_FOR_W_ONLY": np.zeros(n_cells, dtype=bool),
    })
    if stamp_kind:
        obs["dataset_kind"] = DatasetKind.INTERVENTIONAL.value
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    return ad.AnnData(X=X, obs=obs, var=var)


def test_concat_all_stamped_passes():
    loader = MultiPerturbationLoader()
    loader._datasets["kang"] = _stub_adata(stamp_kind=True, dataset_id=0)
    loader._datasets["frangieh"] = _stub_adata(stamp_kind=True, dataset_id=1)
    loader.gene_universe = None  # skip gene-universe subsetting

    combined = loader.build_combined_corpus(include_frangieh=True, include_immport=False)
    assert combined is not None
    assert "dataset_kind" in combined.obs.columns


def test_concat_one_unstamped_raises_with_loader_name():
    loader = MultiPerturbationLoader()
    loader._datasets["kang"] = _stub_adata(stamp_kind=True, dataset_id=0)
    loader._datasets["frangieh"] = _stub_adata(stamp_kind=False, dataset_id=1)
    loader.gene_universe = None

    with pytest.raises(ValueError, match="frangieh"):
        loader.build_combined_corpus(include_frangieh=True, include_immport=False)
