"""Tests for PeakLevelATACEncoder apply_tfidf flag (PR #28)."""
import torch
import pytest

from aivc.skills.atac_peak_encoder import PeakLevelATACEncoder


def test_apply_tfidf_true_is_default():
    """Backward compat: apply_tfidf defaults to True."""
    enc = PeakLevelATACEncoder(n_peaks=100)
    assert enc.apply_tfidf is True


def test_apply_tfidf_true_unchanged_behavior():
    """apply_tfidf=True (explicit) bitwise-identical to default-constructed
    encoder. Backward-compat guarantee for raw-peak callers."""
    torch.manual_seed(0)
    enc_default = PeakLevelATACEncoder(n_peaks=100)
    torch.manual_seed(0)
    enc_explicit = PeakLevelATACEncoder(n_peaks=100, apply_tfidf=True)

    x = torch.poisson(torch.full((4, 100), 0.5))
    enc_default.eval(); enc_explicit.eval()
    with torch.no_grad():
        a = enc_default(x)
        b = enc_explicit(x)
    assert torch.allclose(a, b, atol=1e-6)


def test_apply_tfidf_false_skips_normalization_no_nan():
    """chromVAR-like input (z-scores with negatives) must not NaN/Inf
    when apply_tfidf=False."""
    torch.manual_seed(42)
    enc = PeakLevelATACEncoder(n_peaks=700, apply_tfidf=False)
    enc.eval()
    x = torch.randn(8, 700)
    with torch.no_grad():
        z = enc(x)
    assert z.shape == (8, 64)
    assert not torch.isnan(z).any(), "NaN in output"
    assert not torch.isinf(z).any(), "Inf in output"


def test_apply_tfidf_false_handles_extreme_negatives():
    """Robustness: large-magnitude negative inputs (heavy z-score outliers)
    pass through cleanly without NaN."""
    torch.manual_seed(1)
    enc = PeakLevelATACEncoder(n_peaks=700, apply_tfidf=False)
    enc.eval()
    x = torch.randn(4, 700) * 10.0
    with torch.no_grad():
        z = enc(x)
    assert not torch.isnan(z).any()
    assert not torch.isinf(z).any()


def test_apply_tfidf_true_vs_false_diverge_on_negative_input():
    """Sanity check: TF-IDF gating actually changes output when input has
    negative values (i.e., the flag isn't a no-op)."""
    torch.manual_seed(7)
    enc_t = PeakLevelATACEncoder(n_peaks=200, apply_tfidf=True)
    torch.manual_seed(7)
    enc_f = PeakLevelATACEncoder(n_peaks=200, apply_tfidf=False)
    enc_t.eval(); enc_f.eval()
    x = torch.randn(4, 200)
    with torch.no_grad():
        z_t = enc_t(x)
        z_f = enc_f(x)
    assert not torch.allclose(z_t, z_f, atol=1e-3), \
        "apply_tfidf flag had no effect on output — gating may be broken"


def test_chromvar_compatible_instantiation():
    """End-to-end: 700-dim chromVAR encoder with attn_dim=64 fusion-compatible."""
    enc = PeakLevelATACEncoder(
        n_peaks=700,
        svd_dim=50,
        hidden_dim=128,
        attn_dim=64,
        apply_tfidf=False,
    )
    enc.eval()
    x = torch.randn(16, 700)
    with torch.no_grad():
        z = enc(x)
    assert z.shape == (16, 64)
    assert enc.n_peaks == 700
    assert enc.apply_tfidf is False
