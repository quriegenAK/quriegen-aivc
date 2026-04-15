#!/usr/bin/env bash
# Download the 10x Genomics PBMC Multiome demo dataset (~10k cells).
#
# Source: 10x Genomics public datasets portal
#   https://www.10xgenomics.com/datasets/pbmc-from-a-healthy-donor-granulocytes-removed-through-cell-sorting-10-k-1-standard-2-0-0
#
# Files fetched into data/raw/pbmc10k_multiome/:
#   - atac_fragments.tsv.gz         (ATAC fragments — primary input for
#                                    scripts/harmonize_peaks.py)
#   - atac_fragments.tsv.gz.tbi     (tabix index)
#   - filtered_feature_bc_matrix.h5 (RNA + ATAC counts; used by
#                                    MultiomeLoader downstream)
#
# Idempotent: if every target file already exists AND its SHA-256 matches
# the constant recorded below, the script exits 0 without redownloading.
# On mismatch, it refuses to overwrite and exits nonzero — fix the
# constant or delete the file before rerunning.

set -euo pipefail

# ---- Source URLs ------------------------------------------------------
BASE_URL="https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k"
FRAG_URL="${BASE_URL}/pbmc_granulocyte_sorted_10k_atac_fragments.tsv.gz"
TBI_URL="${BASE_URL}/pbmc_granulocyte_sorted_10k_atac_fragments.tsv.gz.tbi"
H5_URL="${BASE_URL}/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5"

# ---- Expected SHA-256 hashes -----------------------------------------
# PENDING: populate on first successful run from a MACS2-equipped host.
# Leave the sentinel string intact until then — the verification step
# below will refuse to proceed while sentinels are present, preventing
# an un-verified artifact from being consumed by harmonize_peaks.py.
EXPECTED_FRAG_SHA256="<PENDING: execution on MACS2-equipped host>"
EXPECTED_TBI_SHA256="<PENDING: execution on MACS2-equipped host>"
EXPECTED_H5_SHA256="<PENDING: execution on MACS2-equipped host>"

# ---- Destination ------------------------------------------------------
DEST_DIR="${DEST_DIR:-data/raw/pbmc10k_multiome}"
mkdir -p "${DEST_DIR}"

fetch_if_missing() {
  local url="$1" dest="$2"
  if [[ -f "${dest}" ]]; then
    echo "[download] already present: ${dest}"
    return 0
  fi
  echo "[download] fetching ${url} -> ${dest}"
  curl --fail --location --show-error --retry 3 --output "${dest}" "${url}"
}

verify_sha256() {
  local dest="$1" expected="$2" label="$3"
  if [[ "${expected}" == "<PENDING:"* ]]; then
    echo "ERROR: expected SHA-256 for ${label} is still a PENDING sentinel." >&2
    echo "       Populate EXPECTED_*_SHA256 in this script after the first" >&2
    echo "       verified download, then rerun." >&2
    return 2
  fi
  local actual
  actual="$(shasum -a 256 "${dest}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "ERROR: SHA-256 mismatch for ${label}" >&2
    echo "  file:     ${dest}" >&2
    echo "  expected: ${expected}" >&2
    echo "  actual:   ${actual}" >&2
    return 2
  fi
  echo "[download] verified ${label}: ${actual}"
}

FRAG_PATH="${DEST_DIR}/atac_fragments.tsv.gz"
TBI_PATH="${DEST_DIR}/atac_fragments.tsv.gz.tbi"
H5_PATH="${DEST_DIR}/filtered_feature_bc_matrix.h5"

fetch_if_missing "${FRAG_URL}" "${FRAG_PATH}"
fetch_if_missing "${TBI_URL}" "${TBI_PATH}"
fetch_if_missing "${H5_URL}" "${H5_PATH}"

verify_sha256 "${FRAG_PATH}" "${EXPECTED_FRAG_SHA256}" "atac_fragments.tsv.gz"
verify_sha256 "${TBI_PATH}" "${EXPECTED_TBI_SHA256}" "atac_fragments.tsv.gz.tbi"
verify_sha256 "${H5_PATH}" "${EXPECTED_H5_SHA256}" "filtered_feature_bc_matrix.h5"

echo "[download] PBMC 10k Multiome demo staged at ${DEST_DIR}"
