"""Tests for aivc.eval.calderon_probe.

6 synthetic tests + 1 real_data smoke that runs full pipeline against
on-disk Path A artifacts.
"""
from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from aivc.eval.calderon_probe import (
    MockEncoder,
    encode_samples,
    project_calderon_to_dogma_space,
    run_linear_probe,
)


def test_project_shape_and_dtype():
    X = sp.csr_matrix(np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]]))  # (2, 3)
    M = sp.csr_matrix(np.array([[0.5, 0.0], [1.0, 0.0], [0.0, 1.0]]))  # (3, 2)
    Y = project_calderon_to_dogma_space(X, M)
    assert Y.shape == (2, 2)
    assert sp.issparse(Y)
    # Sample 0: [1, 0, 2] @ M = [0.5, 2.0]
    np.testing.assert_allclose(Y.toarray()[0], [0.5, 2.0], atol=1e-6)


def test_project_dim_mismatch_raises():
    X = sp.csr_matrix(np.zeros((2, 5)))
    M = sp.csr_matrix(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="!= M.shape\\[0\\]"):
        project_calderon_to_dogma_space(X, M)


def test_mock_encoder_output_shape():
    enc = MockEncoder(n_peaks=100, latent_dim=16, seed=42)
    x = torch.randn(5, 100)
    z = enc(x)
    assert z.shape == (5, 16)


def test_mock_encoder_deterministic_with_seed():
    enc1 = MockEncoder(n_peaks=20, latent_dim=8, seed=7)
    enc2 = MockEncoder(n_peaks=20, latent_dim=8, seed=7)
    x = torch.randn(3, 20)
    z1 = enc1(x); z2 = enc2(x)
    torch.testing.assert_close(z1, z2)


def test_encode_samples_batched_equals_full():
    """Batched and full-pass produce identical results."""
    enc = MockEncoder(n_peaks=50, latent_dim=8, seed=1)
    X = sp.csr_matrix(np.random.RandomState(0).randn(20, 50))
    z_batched = encode_samples(X, enc, batch_size=4)
    z_full = encode_samples(X, enc, batch_size=999)
    np.testing.assert_allclose(z_batched, z_full, atol=1e-5)


def test_run_linear_probe_separable_data_high_accuracy():
    """Linearly separable synthetic → mean_accuracy > 0.9."""
    rng = np.random.RandomState(0)
    n_per = 30
    means = np.array([[0.0, 0.0], [5.0, 5.0], [-5.0, 5.0]])
    embeds, labels = [], []
    for cls, mu in enumerate(means):
        embeds.append(rng.randn(n_per, 2) + mu)
        labels.extend([cls] * n_per)
    embeds = np.vstack(embeds)
    labels = np.asarray(labels)
    out = run_linear_probe(embeds, labels, cv_folds=3)
    assert out["mean_accuracy"] > 0.9, f"separable data should be easy: {out['mean_accuracy']}"


def test_run_linear_probe_leave_one_group_out():
    """Group CV produces n_groups folds; each fold's test set is one group."""
    rng = np.random.RandomState(0)
    embeds = rng.randn(40, 4)
    labels = rng.randint(0, 3, size=40)
    groups = np.array(["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10)
    out = run_linear_probe(embeds, labels, groups=groups)
    assert out["n_folds"] == 4
    test_groups_per_fold = [m["test_groups"] for m in out["fold_metrics"]]
    # Each fold's held-out group should be a singleton
    for tg in test_groups_per_fold:
        assert len(tg) == 1
    # All 4 groups should appear exactly once across folds
    held = sorted([tg[0] for tg in test_groups_per_fold])
    assert held == ["A", "B", "C", "D"]


# --- Real-data smoke ----------------------------------------------------------

@pytest.mark.real_data
@pytest.mark.slow
def test_real_data_smoke_calderon_probe_pipeline():
    """End-to-end against Path A artifacts. Mock encoder → linear probe.

    Asserts pipeline runs without errors and produces sane metrics shape.
    Does NOT assert high accuracy — mock encoder is random-init, expected
    near-chance (1/n_classes ≈ 4% for ~25 cell types).
    """
    calderon_path = Path(os.environ.get(
        "CALDERON_HG38_H5AD",
        "data/calderon2019/calderon_atac_hg38.h5ad"))
    proj_path = Path(os.environ.get(
        "CALDERON_PROJ_NPZ",
        "data/calderon2019/calderon_to_dogma_lll_M.npz"))
    if not calderon_path.exists() or not proj_path.exists():
        pytest.skip("Path A artifacts missing")

    calderon = ad.read_h5ad(calderon_path)
    M = sp.load_npz(proj_path)

    X_proj = project_calderon_to_dogma_space(calderon.X, M)
    assert X_proj.shape == (calderon.shape[0], M.shape[1])

    enc = MockEncoder(n_peaks=X_proj.shape[1], latent_dim=64, seed=0)
    embeds = encode_samples(X_proj, enc, batch_size=32)
    assert embeds.shape == (calderon.shape[0], 64)
    assert np.isfinite(embeds).all(), "embeddings contain NaN/Inf"

    labels = calderon.obs["cell_type"].astype(str).values
    groups = calderon.obs["donor"].astype(str).values
    out = run_linear_probe(embeds, labels, groups=groups)

    print(f"\nMock encoder probe: mean_acc={out['mean_accuracy']:.4f}, "
          f"chance={1.0/out['n_classes']:.4f}, n_folds={out['n_folds']}, "
          f"n_classes={out['n_classes']}")

    # Empirical baseline: Calderon is BULK ATAC (175 samples = 25 cell types ×
    # 7 donors), not single-cell. Each sample is a strongly-distinct cell-type
    # pseudobulk. Random Linear projection from 199k DOGMA peaks → 64-dim
    # preserves enough variance (Johnson-Lindenstrauss) that LogReg recovers
    # cell type at ~0.64 accuracy with leave-one-donor-out CV. Chance ≈ 0.04.
    #
    # Real-encoder gain (PR #41) is measured against this mock baseline; if a
    # trained encoder doesn't beat ~0.64, the encoder isn't carrying cell-type
    # signal beyond what random projection already exposes.
    #
    # Bounds rationale:
    #   lower (chance * 1.2): pipeline must be at least marginally above chance
    #   upper (0.95): catches obvious label leakage / non-mock encoder
    chance = 1.0 / out["n_classes"]
    assert chance * 1.2 <= out["mean_accuracy"] <= 0.95, (
        f"Mock probe accuracy {out['mean_accuracy']:.4f} outside expected "
        f"range [chance*1.2={chance*1.2:.3f}, 0.95]. "
        "Below: pipeline broken (worse than chance); above: suspect label leakage "
        "or non-mock encoder. Empirical mock baseline ≈ 0.64."
    )
    assert out["n_folds"] >= 2  # at least 2 donors in any sane Calderon


def _minimal_pretrain_ckpt_dict(atac_state, config):
    """Build a synthetic ckpt dict satisfying ckpt_loader's strict schema.

    ckpt_loader._validate_top_level requires all 8 top-level keys; the
    encoder loader only consumes 'atac_encoder' + 'config'. Other slots
    are set to placeholder values whose only job is to clear validation.
    """
    return {
        "schema_version": 1,
        "rna_encoder": {},
        "atac_encoder": atac_state,
        "pretrain_head": {},
        "rna_encoder_class": "aivc.skills.rna_encoder.SimpleRNAEncoder",
        "atac_encoder_class": "aivc.skills.atac_peak_encoder.PeakLevelATACEncoder",
        "pretrain_head_class": "aivc.training.pretrain_heads.MultiomePretrainHead",
        "config": config,
    }


def test_load_atac_encoder_from_ckpt_synthetic(tmp_path):
    """Round-trip: build fake ckpt, load it back, verify forward identical."""
    import torch
    from aivc.skills.atac_peak_encoder import PeakLevelATACEncoder
    from aivc.eval.calderon_probe import load_atac_encoder_from_ckpt

    n_peaks = 100
    attn_dim = 8
    enc = PeakLevelATACEncoder(n_peaks=n_peaks, attn_dim=attn_dim)
    ckpt_path = tmp_path / "fake_pretrain.pt"
    torch.save(
        _minimal_pretrain_ckpt_dict(
            enc.state_dict(),
            {
                "arm": "joint",
                "n_peaks": n_peaks,
                "atac_latent": attn_dim,
                "n_lysis_categories": 0,
            },
        ),
        ckpt_path,
    )

    loaded, cfg = load_atac_encoder_from_ckpt(ckpt_path)
    assert loaded.attn_dim == attn_dim
    enc.eval()  # match loaded.eval() so dropout doesn't desync the forward
    x = torch.rand(4, n_peaks)
    with torch.no_grad():
        z_orig = enc(x)
        z_loaded = loaded(x)
    torch.testing.assert_close(z_orig, z_loaded, atol=1e-6, rtol=1e-6)


def test_load_atac_encoder_n_peaks_mismatch_raises(tmp_path):
    """Mismatched expected_n_peaks surfaces a clear error."""
    import torch
    import pytest
    from aivc.skills.atac_peak_encoder import PeakLevelATACEncoder
    from aivc.eval.calderon_probe import load_atac_encoder_from_ckpt

    enc = PeakLevelATACEncoder(n_peaks=100, attn_dim=8)
    ckpt_path = tmp_path / "fake.pt"
    torch.save(
        _minimal_pretrain_ckpt_dict(
            enc.state_dict(),
            {"n_peaks": 100, "atac_latent": 8},
        ),
        ckpt_path,
    )

    with pytest.raises(ValueError, match="n_peaks mismatch"):
        load_atac_encoder_from_ckpt(ckpt_path, expected_n_peaks=99999)


def test_load_atac_encoder_recovers_n_peaks_from_lsi_weight(tmp_path):
    """If config lacks n_peaks, recover from LSI weight shape."""
    import torch
    from aivc.skills.atac_peak_encoder import PeakLevelATACEncoder
    from aivc.eval.calderon_probe import load_atac_encoder_from_ckpt

    enc = PeakLevelATACEncoder(n_peaks=200, attn_dim=4)
    ckpt_path = tmp_path / "no_n_peaks.pt"
    torch.save(
        _minimal_pretrain_ckpt_dict(
            enc.state_dict(),
            {"atac_latent": 4},
        ),
        ckpt_path,
    )

    loaded, cfg = load_atac_encoder_from_ckpt(ckpt_path)
    assert loaded.n_peaks == 200


def test_eval_script_default_projection_is_union(tmp_path):
    """Default --projection points at union M, not LLL-only.

    Source-text grep — fails if the default ever reverts.
    """
    from pathlib import Path
    SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_calderon_linear_probe.py"
    src = SCRIPT.read_text()
    assert "calderon_to_dogma_union_M.npz" in src, (
        "Default --projection must point at union peak-set M (323,500 peaks). "
        "LLL-only M (198,947 peaks) shape-mismatches with union-trained encoder."
    )
