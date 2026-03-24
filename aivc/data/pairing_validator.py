"""
Pairing validator — gate-check before training starts.

Validates that:
  1. A pairing certificate exists for the experiment
  2. All PHYSICAL pairs have matching barcodes in the data
  3. The contrastive loss configuration matches the certificate
  4. No UNKNOWN pairs are silently ignored
"""
from itertools import combinations
from typing import Optional

from aivc.data.pairing_certificate import (
    ModalityPair,
    PairingCertificate,
    PairingType,
)


class PairingValidator:
    """Validates pairing certificates against actual data at load time."""

    def validate_physical_barcodes(
        self,
        adata_a,
        adata_b,
        pair: ModalityPair,
    ) -> dict:
        """
        For PHYSICAL pairs: verify barcode overlap between modalities.
        A PHYSICAL pair with < 90% overlap is suspicious.

        Returns dict with n_barcodes_a, n_barcodes_b, n_overlap,
        overlap_frac, valid, message.
        """
        bc_a = set(adata_a.obs_names.tolist())
        bc_b = set(adata_b.obs_names.tolist())
        overlap = bc_a & bc_b
        frac_a = len(overlap) / max(len(bc_a), 1)
        frac_b = len(overlap) / max(len(bc_b), 1)
        min_frac = min(frac_a, frac_b)
        valid = min_frac >= 0.90

        message = (
            f"{pair.modality_a}+{pair.modality_b} barcode overlap: "
            f"{len(overlap):,} / min({len(bc_a):,}, {len(bc_b):,}) = {min_frac:.1%}. "
            f"{'PASS' if valid else 'FAIL -- expected >= 90% for PHYSICAL pair.'}"
        )

        return {
            "n_barcodes_a": len(bc_a),
            "n_barcodes_b": len(bc_b),
            "n_overlap": len(overlap),
            "overlap_frac": min_frac,
            "valid": valid,
            "message": message,
        }

    def validate_certificate_for_training(
        self,
        certificate: PairingCertificate,
        modalities_present: list,
    ) -> dict:
        """
        Gate-check before training starts.

        Rules:
          1. Certificate must be valid (no schema errors)
          2. Every modality pair present in data must have an entry
          3. UNKNOWN pairs block contrastive loss
          4. Returns which loss terms are enabled

        Returns dict with can_train, can_use_contrastive, contrastive_pairs,
        ot_pairs, blocked_by_unknown, errors, warnings, loss_config.
        """
        errors = []
        warnings = []

        cert_check = certificate.validate()
        errors.extend(cert_check["errors"])
        warnings.extend(cert_check["warnings"])

        contrastive_pairs = []
        ot_pairs = []
        blocked_by_unknown = []

        for mod_a, mod_b in combinations(sorted(modalities_present), 2):
            pair = certificate.get_pair(mod_a, mod_b)
            if pair is None:
                warnings.append(
                    f"No certificate entry for {mod_a}+{mod_b}. "
                    f"Treating as COMPUTATIONAL (conservative default)."
                )
                ot_pairs.append((mod_a, mod_b))
            elif pair.pairing_type == PairingType.PHYSICAL:
                contrastive_pairs.append((mod_a, mod_b))
            elif pair.pairing_type == PairingType.PARTIAL:
                contrastive_pairs.append((mod_a, mod_b))
                ot_pairs.append((mod_a, mod_b))
            elif pair.pairing_type == PairingType.COMPUTATIONAL:
                ot_pairs.append((mod_a, mod_b))
            elif pair.pairing_type == PairingType.UNKNOWN:
                blocked_by_unknown.append((mod_a, mod_b))
                warnings.append(
                    f"UNKNOWN pairing for {mod_a}+{mod_b}. "
                    f"Contrastive loss DISABLED for this pair. "
                    f"Get wet-lab confirmation before enabling."
                )

        can_use_contrastive = len(contrastive_pairs) > 0
        can_train = len(errors) == 0

        loss_config = {
            "mse_loss": True,
            "lfc_loss": True,
            "cosine_loss": True,
            "l1_neumann": True,
            "contrastive_loss": can_use_contrastive,
            "cross_modal_pred": can_use_contrastive,
            "contrastive_pairs": contrastive_pairs,
            "ot_required_pairs": ot_pairs,
        }

        return {
            "can_train": can_train,
            "can_use_contrastive": can_use_contrastive,
            "contrastive_pairs": contrastive_pairs,
            "ot_pairs": ot_pairs,
            "blocked_by_unknown": blocked_by_unknown,
            "errors": errors,
            "warnings": warnings,
            "loss_config": loss_config,
        }
