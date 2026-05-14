"""Unit tests for aivc.data.mimitou_loader.

Covers:
  - Arm → roles mapping (the canonical synergy 4-arm structure)
  - PerturbationEmbedder (shape + padding-idx zero-emit)
  - MimitouDataset on a synthetic h5ad
  - ArmBalancedBatchSampler composition guarantees
  - sparse_collate output shapes
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from aivc.data.mimitou_loader import (
    ArmAssignment,
    MimitouArmMap,
    MimitouDataset,
    ArmBalancedBatchSampler,
    PerturbationEmbedder,
    map_arm_to_roles,
    sparse_collate,
    STIM_VOCAB, INH_VOCAB, DOUBLE_KO_NAME,
)


# ---------- map_arm_to_roles ----------

def test_map_ntc():
    a = map_arm_to_roles("NTC")
    assert a == ArmAssignment(has_stim=False, has_inh=False, stim_id=-1, inh_id=-1)


def test_map_cd3e():
    a = map_arm_to_roles("CD3E")
    assert a.has_stim and not a.has_inh
    assert a.stim_id >= 0 and a.inh_id == -1


def test_map_cd4():
    a = map_arm_to_roles("CD4")
    assert not a.has_stim and a.has_inh
    assert a.stim_id == -1 and a.inh_id >= 0


def test_map_double_ko():
    a = map_arm_to_roles(DOUBLE_KO_NAME)
    assert a.has_stim and a.has_inh
    assert a.stim_id >= 0 and a.inh_id >= 0


def test_map_unknown_raises():
    with pytest.raises(ValueError, match="Unknown arm"):
        map_arm_to_roles("CD8_KO")


def test_mimitou_arm_map_completeness():
    """All 6 Mimitou arms must be in MimitouArmMap."""
    expected = {"NTC", "CD3E", "CD4", "ZAP70", "NFKB2", DOUBLE_KO_NAME}
    assert set(MimitouArmMap.keys()) == expected


# ---------- PerturbationEmbedder ----------

def test_perturbation_embedder_shape():
    emb = PerturbationEmbedder(n_stim=3, n_inh=1, pert_dim=16)
    stim_id = torch.tensor([0, 1, 2, -1])    # 3 real ids + 1 absent
    inh_id  = torch.tensor([-1, 0, -1, 0])
    stim_emb, inh_emb = emb(stim_id, inh_id)
    assert stim_emb.shape == (4, 16)
    assert inh_emb.shape == (4, 16)


def test_perturbation_embedder_padding_zero():
    """stim_id=-1 maps to zero vector (padding_idx behavior)."""
    torch.manual_seed(0)
    emb = PerturbationEmbedder(n_stim=3, n_inh=1, pert_dim=8)
    stim_id = torch.tensor([0, -1, 1])
    inh_id  = torch.tensor([0, -1, -1])
    stim_emb, inh_emb = emb(stim_id, inh_id)
    # Index 1 (-1 → padding idx 0) should be all zeros
    assert torch.allclose(stim_emb[1], torch.zeros(8))
    assert torch.allclose(inh_emb[1], torch.zeros(8))
    assert torch.allclose(inh_emb[2], torch.zeros(8))
    # Non-padding indices should NOT be zero (with high probability)
    assert stim_emb[0].abs().sum() > 0


# ---------- MimitouDataset (synthetic h5ad) ----------

@pytest.fixture
def synthetic_h5ad(tmp_path):
    """Build a tiny Mimitou-shaped h5ad for testing.

    Layout matches the real prep output:
      obs['perturbation'] over 6 arms with non-trivial per-arm counts
      obsm['atac_peaks'] sparse (n_cells, 100)
      obsm['protein'] sparse (n_cells, 8)
    """
    n_per_arm = [50, 30, 30, 30, 20, 10]  # NTC, CD3E, CD4, ZAP70, NFKB2, double
    arm_names = ["NTC", "CD3E", "CD4", "ZAP70", "NFKB2", DOUBLE_KO_NAME]
    perturbations = sum(([a] * n for a, n in zip(arm_names, n_per_arm)), [])
    n_total = len(perturbations)

    obs = pd.DataFrame({"perturbation": pd.Categorical(perturbations)},
                       index=[f"cell_{i}" for i in range(n_total)])
    X = sp.csr_matrix((n_total, 0), dtype=np.float32)
    adata = ad.AnnData(X=X, obs=obs)
    rng = np.random.default_rng(0)
    adata.obsm["atac_peaks"] = sp.csr_matrix(
        rng.binomial(1, 0.05, size=(n_total, 100)).astype(np.float32)
    )
    adata.obsm["protein"] = sp.csr_matrix(
        rng.poisson(2.0, size=(n_total, 8)).astype(np.float32)
    )

    path = tmp_path / "synth_mimitou.h5ad"
    adata.write_h5ad(path, compression=None)
    return path, n_per_arm, arm_names


def test_dataset_loads_full_arm_set(synthetic_h5ad):
    path, n_per_arm, arm_names = synthetic_h5ad
    ds = MimitouDataset(path)
    assert len(ds) == sum(n_per_arm)
    counts = ds.per_arm_counts()
    for n_expected, arm in zip(n_per_arm, arm_names):
        assert counts[arm] == n_expected


def test_dataset_exclude_double(synthetic_h5ad):
    path, n_per_arm, arm_names = synthetic_h5ad
    ds = MimitouDataset(path, exclude_double=True)
    # n_per_arm[5] is double-KO; should be dropped
    assert len(ds) == sum(n_per_arm) - n_per_arm[5]
    assert DOUBLE_KO_NAME not in ds.per_arm_counts()


def test_dataset_getitem_keys(synthetic_h5ad):
    path, _, _ = synthetic_h5ad
    ds = MimitouDataset(path)
    item = ds[0]
    required = {"cell_idx", "atac_row", "perturbation", "has_stim", "has_inh",
                "stim_id", "inh_id", "arm_label", "time"}
    assert required.issubset(item.keys())
    assert sp.issparse(item["atac_row"])


def test_dataset_protein_present_when_in_obsm(synthetic_h5ad):
    path, _, _ = synthetic_h5ad
    ds = MimitouDataset(path)
    item = ds[0]
    assert "protein" in item
    assert item["protein"].shape == (8,)


def test_dataset_arm_label_is_dense_idx(synthetic_h5ad):
    """arm_label is a dense index over included_arms — useful for SupCon."""
    path, _, _ = synthetic_h5ad
    ds = MimitouDataset(path)
    labels = [ds[i]["arm_label"] for i in range(len(ds))]
    # Should cover the full range [0, n_arms)
    assert min(labels) == 0
    assert max(labels) == len(ds.included_arms) - 1


def test_dataset_missing_obsm_raises(tmp_path):
    """h5ad without obsm['atac_peaks'] is rejected."""
    obs = pd.DataFrame({"perturbation": pd.Categorical(["NTC"] * 10)},
                       index=[f"c{i}" for i in range(10)])
    adata = ad.AnnData(X=sp.csr_matrix((10, 0), dtype=np.float32), obs=obs)
    p = tmp_path / "bad.h5ad"
    adata.write_h5ad(p, compression=None)
    with pytest.raises(ValueError, match="atac_peaks"):
        MimitouDataset(p)


# ---------- ArmBalancedBatchSampler ----------

def test_sampler_batch_size_too_small_raises():
    labels = np.array([0, 0, 1, 1, 2, 2])
    with pytest.raises(ValueError, match="too small"):
        # 3 arms * 2 min_per_arm = 6 > batch_size 4
        ArmBalancedBatchSampler(labels, batch_size=4, min_per_arm=2)


def test_sampler_min_per_arm_satisfied():
    """Every emitted batch must have at least min_per_arm cells per arm."""
    labels = np.array([0]*20 + [1]*20 + [2]*5)   # arm 2 is sparse
    sampler = ArmBalancedBatchSampler(labels, batch_size=12, min_per_arm=2, n_batches=5, seed=0)
    for batch in sampler:
        assert len(batch) == 12
        batch_labels = labels[batch]
        # 2 must appear at least min_per_arm=2 times in the batch
        for arm in (0, 1, 2):
            assert (batch_labels == arm).sum() >= 2, (
                f"arm {arm} not represented enough times in batch: "
                f"{(batch_labels == arm).sum()}"
            )


def test_sampler_yields_expected_count():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    sampler = ArmBalancedBatchSampler(labels, batch_size=4, min_per_arm=2, n_batches=3, seed=0)
    batches = list(sampler)
    assert len(batches) == 3
    assert all(len(b) == 4 for b in batches)


def test_sampler_underresourced_arm_with_replacement():
    """Sparse arm (less than min_per_arm cells) still gets min_per_arm via replacement."""
    labels = np.array([0]*10 + [1])   # arm 1 has only 1 cell
    sampler = ArmBalancedBatchSampler(labels, batch_size=4, min_per_arm=2,
                                      with_replacement=True, n_batches=2, seed=0)
    for batch in sampler:
        batch_labels = labels[batch]
        assert (batch_labels == 1).sum() >= 2


# ---------- sparse_collate ----------

def test_collate_output_shapes(synthetic_h5ad):
    path, _, _ = synthetic_h5ad
    ds = MimitouDataset(path)
    batch = [ds[i] for i in range(4)]
    out = sparse_collate(batch)
    assert sp.issparse(out["atac"])
    assert out["atac"].shape == (4, 100)
    assert out["has_stim"].shape == (4,) and out["has_stim"].dtype == torch.bool
    assert out["has_inh"].shape == (4,)
    assert out["arm_label"].shape == (4,) and out["arm_label"].dtype == torch.long
    assert out["protein"].shape == (4, 8)


def test_collate_perturbation_strings_retained(synthetic_h5ad):
    path, _, _ = synthetic_h5ad
    ds = MimitouDataset(path)
    batch = [ds[i] for i in range(4)]
    out = sparse_collate(batch)
    assert isinstance(out["perturbation"], list)
    assert len(out["perturbation"]) == 4
    assert all(isinstance(s, str) for s in out["perturbation"])
