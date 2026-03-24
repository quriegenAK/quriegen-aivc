"""
Tests for pairing certificate schema and validator.
All tests use mock data. No real files. Under 15 seconds on CPU.

Run: pytest tests/test_pairing_certificate.py -v
"""
import json
import os
import tempfile

import pytest

from aivc.data.pairing_certificate import (
    ModalityPair,
    PairingCertificate,
    PairingType,
)
from aivc.data.pairing_validator import PairingValidator


class TestModalityPair:

    def test_physical_pair_contrastive_valid(self):
        """PHYSICAL pair -> contrastive_loss_valid() == True."""
        pair = ModalityPair("rna", "protein", PairingType.PHYSICAL)
        assert pair.contrastive_loss_valid() is True

    def test_computational_pair_contrastive_invalid(self):
        """COMPUTATIONAL pair -> contrastive_loss_valid() == False."""
        pair = ModalityPair("rna", "protein", PairingType.COMPUTATIONAL, paired_fraction=0.0)
        assert pair.contrastive_loss_valid() is False

    def test_unknown_pair_contrastive_invalid(self):
        """UNKNOWN pair -> contrastive_loss_valid() == False."""
        pair = ModalityPair("rna", "protein", PairingType.UNKNOWN, paired_fraction=0.0)
        assert pair.contrastive_loss_valid() is False

    def test_partial_pair_requires_ot(self):
        """PARTIAL pair -> requires_ot_alignment() == True."""
        pair = ModalityPair("rna", "protein", PairingType.PARTIAL, paired_fraction=0.5)
        assert pair.requires_ot_alignment() is True
        assert pair.contrastive_loss_valid() is True  # also valid for paired subset


class TestCertificateValidation:

    def test_certificate_validate_physical_needs_barcode(self):
        """PHYSICAL pair with barcode_col=None -> validate() returns error."""
        cert = PairingCertificate(
            experiment_id="test", platform="test",
            created_by="test", creation_date="2026-01-01",
            pairs=[
                ModalityPair("rna", "protein", PairingType.PHYSICAL,
                             barcode_col=None, validated_by="someone"),
            ],
        )
        result = cert.validate()
        assert not result["valid"]
        assert any("barcode_col" in e for e in result["errors"])

    def test_certificate_validate_duplicate_pair(self):
        """Two entries for same pair -> validate() returns error."""
        cert = PairingCertificate(
            experiment_id="test", platform="test",
            created_by="test", creation_date="2026-01-01",
            pairs=[
                ModalityPair("rna", "protein", PairingType.PHYSICAL,
                             barcode_col="bc", validated_by="x"),
                ModalityPair("rna", "protein", PairingType.COMPUTATIONAL,
                             paired_fraction=0.0),
            ],
        )
        result = cert.validate()
        assert not result["valid"]
        assert any("Duplicate" in e for e in result["errors"])


class TestCertificateLookup:

    def test_certificate_get_pair_order_independent(self):
        """get_pair('rna', 'protein') == get_pair('protein', 'rna')."""
        cert = PairingCertificate(
            experiment_id="test", platform="test",
            created_by="test", creation_date="2026-01-01",
            pairs=[
                ModalityPair("rna", "protein", PairingType.PHYSICAL,
                             barcode_col="bc", validated_by="x"),
            ],
        )
        p1 = cert.get_pair("rna", "protein")
        p2 = cert.get_pair("protein", "rna")
        assert p1 is not None
        assert p2 is not None
        assert p1.modality_a == p2.modality_a

    def test_certificate_get_pair_returns_none_for_missing(self):
        """get_pair for undefined pair returns None."""
        cert = PairingCertificate(
            experiment_id="test", platform="test",
            created_by="test", creation_date="2026-01-01",
            pairs=[],
        )
        assert cert.get_pair("rna", "atac") is None


class TestCertificateSerialisation:

    def test_certificate_serialisation_roundtrip(self):
        """to_json() -> from_json() -> all fields preserved."""
        cert = PairingCertificate(
            experiment_id="test_exp",
            platform="QuRIE-seq",
            created_by="Tester",
            creation_date="2026-03-24",
            pairs=[
                ModalityPair("rna", "protein", PairingType.PHYSICAL,
                             paired_fraction=1.0, barcode_col="bc",
                             validated_by="Tester",
                             notes="test note"),
                ModalityPair("rna", "atac", PairingType.UNKNOWN,
                             paired_fraction=0.0, barcode_col=None),
            ],
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cert.to_json(path)
            loaded = PairingCertificate.from_json(path)
            assert loaded.experiment_id == "test_exp"
            assert loaded.platform == "QuRIE-seq"
            assert len(loaded.pairs) == 2
            assert loaded.pairs[0].pairing_type == PairingType.PHYSICAL
            assert loaded.pairs[0].notes == "test note"
            assert loaded.pairs[1].pairing_type == PairingType.UNKNOWN
        finally:
            os.unlink(path)


class TestPairingValidator:

    def test_validator_blocks_unknown_pairs(self):
        """Certificate with UNKNOWN pairs -> contrastive disabled."""
        cert = PairingCertificate.make_quriegen_pending()
        validator = PairingValidator()
        result = validator.validate_certificate_for_training(
            cert, modalities_present=["rna", "protein"]
        )
        assert result["can_use_contrastive"] is False
        assert len(result["blocked_by_unknown"]) > 0

    def test_validator_enables_contrastive_for_physical(self):
        """Certificate with PHYSICAL pairs -> contrastive enabled."""
        cert = PairingCertificate(
            experiment_id="test", platform="test",
            created_by="test", creation_date="2026-01-01",
            pairs=[
                ModalityPair("rna", "protein", PairingType.PHYSICAL,
                             barcode_col="bc", validated_by="x"),
            ],
        )
        validator = PairingValidator()
        result = validator.validate_certificate_for_training(
            cert, modalities_present=["rna", "protein"]
        )
        assert result["can_use_contrastive"] is True
        assert ("protein", "rna") in result["contrastive_pairs"] or \
               ("rna", "protein") in result["contrastive_pairs"]


class TestPrebuiltCertificates:

    def test_kang2018_certificate_is_computational(self):
        """make_kang2018() -> rna+protein pair is COMPUTATIONAL."""
        cert = PairingCertificate.make_kang2018()
        pair = cert.get_pair("rna", "protein")
        assert pair is not None
        assert pair.pairing_type == PairingType.COMPUTATIONAL
        assert pair.contrastive_loss_valid() is False

    def test_quriegen_pending_all_unknown(self):
        """make_quriegen_pending() -> all pairs UNKNOWN."""
        cert = PairingCertificate.make_quriegen_pending()
        for pair in cert.pairs:
            assert pair.pairing_type == PairingType.UNKNOWN
        assert len(cert.unknown_pairs()) == 3
