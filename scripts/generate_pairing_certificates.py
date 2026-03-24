"""
Generate default pairing certificate files.

Run once to create the certificate files.
For QuRIE-seq: edit quriegen_pending.json after Thiago confirms
the pairing protocol. Change pairing_type from "unknown" to
"physical" or "computational" for each modality pair.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aivc.data.pairing_certificate import PairingCertificate

os.makedirs("data/pairing_certificates", exist_ok=True)

# Kang 2018 — RNA only, confirmed
cert_kang = PairingCertificate.make_kang2018()
cert_kang.to_json("data/pairing_certificates/kang2018.json")
print("Created: data/pairing_certificates/kang2018.json")

# QuRIE-seq — pending wet-lab confirmation
cert_quriegen = PairingCertificate.make_quriegen_pending()
cert_quriegen.to_json("data/pairing_certificates/quriegen_pending.json")
print("Created: data/pairing_certificates/quriegen_pending.json")

print("\n*** ACTION REQUIRED ***")
print("Send data/pairing_certificates/quriegen_pending.json to Thiago.")
print("Ask him to update pairing_type for each modality pair:")
print("  - rna + protein:  physical / computational / partial ?")
print("  - rna + phospho:  physical / computational / partial ?")
print("  - rna + atac:     physical / computational / partial ?")
print("Once updated, rename to: data/pairing_certificates/quriegen_v1.json")
print("Then re-run training -- contrastive loss activates automatically.")
