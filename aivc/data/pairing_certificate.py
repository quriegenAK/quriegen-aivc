"""
QuRIE-seq modality pairing certificate schema.

Determines which modalities can use contrastive learning (physically paired)
vs which require OT-based alignment (computationally paired).

PHYSICAL:      Same cell, same lysis. Barcode links both. Contrastive: YES.
COMPUTATIONAL: Different cells, statistically matched. Contrastive: NO.
PARTIAL:       Subset physically paired; rest computational. Contrastive: paired subset only.
UNKNOWN:       Pending wet-lab confirmation. Contrastive: NO (conservative).
"""
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PairingType(Enum):
    PHYSICAL = "physical"
    COMPUTATIONAL = "computational"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


@dataclass
class ModalityPair:
    """Pairing relationship between two modalities."""

    modality_a: str
    modality_b: str
    pairing_type: PairingType
    paired_fraction: float = 1.0
    barcode_col: Optional[str] = "barcode"
    protocol_ref: Optional[str] = None
    validated_by: Optional[str] = None
    notes: Optional[str] = None

    def contrastive_loss_valid(self) -> bool:
        """True if contrastive loss can be applied (PHYSICAL or PARTIAL only)."""
        return self.pairing_type in (PairingType.PHYSICAL, PairingType.PARTIAL)

    def requires_ot_alignment(self) -> bool:
        """True if OT-based alignment is needed (COMPUTATIONAL or PARTIAL)."""
        return self.pairing_type in (PairingType.COMPUTATIONAL, PairingType.PARTIAL)

    def to_dict(self) -> dict:
        return {
            "modality_a": self.modality_a,
            "modality_b": self.modality_b,
            "pairing_type": self.pairing_type.value,
            "paired_fraction": self.paired_fraction,
            "barcode_col": self.barcode_col,
            "protocol_ref": self.protocol_ref,
            "validated_by": self.validated_by,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModalityPair":
        d = d.copy()
        d["pairing_type"] = PairingType(d["pairing_type"])
        return cls(**d)


@dataclass
class PairingCertificate:
    """
    Full pairing certificate for one experiment / dataset.
    Describes the pairing status for every modality pair.
    Created ONCE per experiment by wet lab + comp bio team,
    loaded at training time to configure the loss function.
    """

    experiment_id: str
    platform: str
    created_by: str
    creation_date: str
    pairs: list = field(default_factory=list)

    def get_pair(self, modality_a: str, modality_b: str) -> Optional[ModalityPair]:
        """
        Look up pairing between two modalities (order-independent).
        Returns None if pair not defined.
        """
        for p in self.pairs:
            if (p.modality_a == modality_a and p.modality_b == modality_b) or \
               (p.modality_a == modality_b and p.modality_b == modality_a):
                return p
        return None

    def contrastive_pairs(self) -> list:
        """Return all pairs where contrastive loss is valid."""
        return [p for p in self.pairs if p.contrastive_loss_valid()]

    def unknown_pairs(self) -> list:
        """Return pairs with UNKNOWN status."""
        return [p for p in self.pairs if p.pairing_type == PairingType.UNKNOWN]

    def validate(self) -> dict:
        """
        Validate certificate for internal consistency.
        Returns dict with 'valid' (bool), 'errors' (list), 'warnings' (list).
        Never raises exceptions.
        """
        errors = []
        warnings = []
        seen_pairs = set()

        for p in self.pairs:
            key = tuple(sorted([p.modality_a, p.modality_b]))
            if key in seen_pairs:
                errors.append(
                    f"Duplicate modality pair: {p.modality_a} -- {p.modality_b}"
                )
            seen_pairs.add(key)

            if not (0.0 <= p.paired_fraction <= 1.0):
                errors.append(
                    f"{p.modality_a}+{p.modality_b}: paired_fraction "
                    f"{p.paired_fraction} out of range [0, 1]"
                )

            if p.pairing_type == PairingType.PHYSICAL:
                if not p.barcode_col:
                    errors.append(
                        f"{p.modality_a}+{p.modality_b}: PHYSICAL pair "
                        f"must specify barcode_col"
                    )
                if not p.validated_by:
                    warnings.append(
                        f"{p.modality_a}+{p.modality_b}: PHYSICAL pair "
                        f"has no validated_by"
                    )
                if abs(p.paired_fraction - 1.0) > 0.01:
                    warnings.append(
                        f"{p.modality_a}+{p.modality_b}: PHYSICAL pair "
                        f"has paired_fraction={p.paired_fraction:.2f} (expected 1.0)"
                    )

            if p.pairing_type == PairingType.COMPUTATIONAL:
                if p.paired_fraction > 0.01:
                    warnings.append(
                        f"{p.modality_a}+{p.modality_b}: COMPUTATIONAL pair "
                        f"has paired_fraction={p.paired_fraction:.2f} (expected 0.0)"
                    )

            if p.pairing_type == PairingType.UNKNOWN:
                warnings.append(
                    f"{p.modality_a}+{p.modality_b}: UNKNOWN pairing -- "
                    f"contrastive loss DISABLED for this pair."
                )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def to_json(self, path: str) -> None:
        """Serialise certificate to JSON file."""
        data = {
            "experiment_id": self.experiment_id,
            "platform": self.platform,
            "created_by": self.created_by,
            "creation_date": self.creation_date,
            "pairs": [p.to_dict() for p in self.pairs],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "PairingCertificate":
        """Load certificate from JSON file."""
        with open(path) as f:
            data = json.load(f)
        pairs = [ModalityPair.from_dict(p) for p in data.pop("pairs")]
        return cls(**data, pairs=pairs)

    @classmethod
    def make_kang2018(cls) -> "PairingCertificate":
        """Pre-built certificate for Kang 2018 (RNA only, no multi-modal)."""
        return cls(
            experiment_id="kang_2018_pbmc_ifnb",
            platform="10x Chromium v2",
            created_by="Kang et al. 2018 (GSE96583)",
            creation_date="2026-03-24",
            pairs=[
                ModalityPair(
                    modality_a="rna",
                    modality_b="protein",
                    pairing_type=PairingType.COMPUTATIONAL,
                    paired_fraction=0.0,
                    barcode_col=None,
                    protocol_ref="GSE96583 -- scRNA-seq only. No protein measured.",
                    validated_by="Ash Khan",
                    notes="Kang 2018 is a single-modality scRNA-seq dataset. "
                          "No protein or phospho modalities. ATAC not measured.",
                ),
            ],
        )

    @classmethod
    def make_quriegen_pending(cls) -> "PairingCertificate":
        """
        Placeholder certificate for QuRIE-seq (pending wet-lab confirmation).
        All pairs UNKNOWN until Thiago Patente confirms the protocol.
        """
        return cls(
            experiment_id="quriegen_pbmc_batch1_pending",
            platform="QuRIE-seq",
            created_by="PENDING -- Thiago Patente",
            creation_date="2026-03-24",
            pairs=[
                ModalityPair(
                    modality_a="rna",
                    modality_b="protein",
                    pairing_type=PairingType.UNKNOWN,
                    paired_fraction=0.0,
                    barcode_col=None,
                    protocol_ref=None,
                    validated_by=None,
                    notes="PENDING WET LAB CONFIRMATION. "
                          "QuRIE-seq protocol may physically co-measure "
                          "RNA and surface protein via antibody-oligo conjugates. "
                          "Thiago Patente must confirm: "
                          "(1) Are RNA and protein captured in same droplet? "
                          "(2) Is the barcode column shared across modalities? "
                          "(3) Protocol document reference?",
                ),
                ModalityPair(
                    modality_a="rna",
                    modality_b="phospho",
                    pairing_type=PairingType.UNKNOWN,
                    paired_fraction=0.0,
                    barcode_col=None,
                    protocol_ref=None,
                    validated_by=None,
                    notes="PENDING WET LAB CONFIRMATION. "
                          "Phospho-proteomics in QuRIE-seq: is this from "
                          "same cell lysis as RNA? Most phospho protocols "
                          "require separate sample preparation.",
                ),
                ModalityPair(
                    modality_a="rna",
                    modality_b="atac",
                    pairing_type=PairingType.UNKNOWN,
                    paired_fraction=0.0,
                    barcode_col=None,
                    protocol_ref=None,
                    validated_by=None,
                    notes="PENDING. ATAC modality not yet measured in QuRIE-seq. "
                          "Will require 10x Multiome or equivalent. "
                          "When available: confirm whether barcode links "
                          "RNA and ATAC from same nucleus.",
                ),
            ],
        )
