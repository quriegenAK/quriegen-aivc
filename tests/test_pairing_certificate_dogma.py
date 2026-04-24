"""Tests for PairingCertificate.make_dogma_seq() factory."""
from aivc.data.pairing_certificate import PairingCertificate, PairingType


def test_make_dogma_seq_3_physical_pairs():
    cert = PairingCertificate.make_dogma_seq()
    assert len(cert.pairs) == 3
    for pair in cert.pairs:
        assert pair.pairing_type == PairingType.PHYSICAL
        assert pair.paired_fraction == 1.0
        assert pair.barcode_col == "barcode"


def test_make_dogma_seq_includes_all_pairs():
    cert = PairingCertificate.make_dogma_seq()
    modality_pairs = {
        tuple(sorted([p.modality_a, p.modality_b]))
        for p in cert.pairs
    }
    assert modality_pairs == {
        ("atac", "rna"),
        ("protein", "rna"),
        ("atac", "protein"),
    }


def test_make_dogma_seq_validates_clean():
    cert = PairingCertificate.make_dogma_seq()
    result = cert.validate()
    assert result["valid"], f"Validation errors: {result['errors']}"


def test_make_dogma_seq_experiment_id_stable():
    cert = PairingCertificate.make_dogma_seq()
    assert cert.experiment_id == "mimitou2021_dogma_pbmc_gse156478"
    assert "DOGMA-seq" in cert.platform


def test_make_dogma_seq_json_roundtrip(tmp_path):
    cert = PairingCertificate.make_dogma_seq()
    p = tmp_path / "dogma_cert.json"
    cert.to_json(str(p))
    loaded = PairingCertificate.from_json(str(p))
    assert len(loaded.pairs) == 3
    assert loaded.experiment_id == cert.experiment_id
