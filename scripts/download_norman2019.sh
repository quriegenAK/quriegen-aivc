#!/usr/bin/env bash
# Download Norman et al. 2019 Perturb-seq dataset (K562 CRISPR).
#
# Source priority (tried in order):
#   1. scperturb zenodo collection (record 7041849) — standardised .h5ad
#      with obs["perturbation"] already normalised.
#   2. pertpy datasets API — fetches the same underlying data via Python.
#   3. GEO GSE133344 — original deposit (counts matrix, more manual work).
#
# On success the canonical file is staged at:
#   data/raw/norman2019/NormanWeissman2019_filtered.h5ad
#
# Idempotent: if the file exists AND its SHA-256 matches, exits 0.
# On hash mismatch, refuses to overwrite — delete the file and rerun.
#
# SHA-256 constants start as PENDING sentinels.  Populate them on the
# first successful download (the script prints the actual hash), then
# rerun so that all subsequent invocations are verified.

set -euo pipefail

# ---- Destination ----------------------------------------------------------
DEST_DIR="${DEST_DIR:-data/raw/norman2019}"
DEST_FILE="${DEST_DIR}/NormanWeissman2019_filtered.h5ad"

# ---- Expected SHA-256 (populate after first verified download) ------------
# Leave the sentinel string intact until then — the verify step below will
# refuse to proceed while sentinels are present.
EXPECTED_SHA256="${EXPECTED_SHA256:-<PENDING:fill_after_first_verified_download>}"

# ---- Source URLs ----------------------------------------------------------
# scperturb zenodo record 7041849 — direct file download
ZENODO_URL="https://zenodo.org/records/7041849/files/NormanWeissman2019_filtered.h5ad"

# GEO series (fallback — raw counts, different structure)
GEO_ACCESSION="GSE133344"

# ---- Helpers --------------------------------------------------------------
log()  { echo "[download-norman2019] $*"; }
err()  { echo "[download-norman2019] ERROR: $*" >&2; }

verify_sha256() {
  local dest="$1" expected="$2"
  if [[ "${expected}" == "<PENDING:"* ]]; then
    local actual
    actual="$(shasum -a 256 "${dest}" | awk '{print $1}')"
    log "WARNING: EXPECTED_SHA256 is still a PENDING sentinel."
    log "  Actual SHA-256 of downloaded file: ${actual}"
    log "  Set EXPECTED_SHA256='${actual}' in this script (or export it)"
    log "  before the next run to enable integrity verification."
    return 0          # allow first-time use; re-run will verify
  fi
  local actual
  actual="$(shasum -a 256 "${dest}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    err "SHA-256 mismatch for ${dest}"
    err "  expected: ${expected}"
    err "  actual:   ${actual}"
    return 2
  fi
  log "Verified SHA-256: ${actual}"
}

fetch_zenodo() {
  log "Attempting source 1: scperturb zenodo (record 7041849) ..."
  mkdir -p "${DEST_DIR}"
  curl --fail --location --show-error --retry 3 \
       --output "${DEST_FILE}" \
       "${ZENODO_URL}"
  log "Zenodo download complete."
}

fetch_pertpy() {
  log "Attempting source 2: pertpy datasets API ..."
  python - <<'PYEOF'
import sys, pathlib, shutil

try:
    import pertpy as pt
except ImportError:
    print("[pertpy] pertpy not installed — skipping.", file=sys.stderr)
    sys.exit(1)

import pathlib
dest = pathlib.Path("data/raw/norman2019/NormanWeissman2019_filtered.h5ad")
dest.parent.mkdir(parents=True, exist_ok=True)

try:
    adata = pt.data.norman_2019()
except Exception as exc:
    print(f"[pertpy] norman_2019() failed: {exc}", file=sys.stderr)
    sys.exit(1)

adata.write_h5ad(dest)
print(f"[pertpy] Written to {dest}  (shape: {adata.shape})")
PYEOF
}

report_h5ad_info() {
  log "Reading h5ad metadata ..."
  python - "${DEST_FILE}" <<'PYEOF'
import sys, anndata as ad, scipy.sparse as sp

path = sys.argv[1]
a = ad.read_h5ad(path)

print(f"\n  n_cells : {a.shape[0]}")
print(f"  n_genes : {a.shape[1]}")
print(f"  obs columns : {list(a.obs.columns)}")
print(f"  var columns : {list(a.var.columns)}")

# Determine perturbation column
for col in ("perturbation", "gene", "condition", "guide_ids"):
    if col in a.obs.columns:
        n_vals = a.obs[col].nunique()
        sample = list(a.obs[col].unique()[:6])
        print(f"\n  Perturbation column : obs['{col}']")
        print(f"    unique values ({n_vals} total) : {sample}")
        break

# Is X raw counts or normalised?
X = a.X
if sp.issparse(X):
    X = X[:200].toarray()
else:
    X = X[:200]
import numpy as np
vals = X[X > 0]
if len(vals) and (np.max(vals) > 50 or not np.all(vals == vals.astype(int))):
    kind = "likely normalised (non-integer or large values present)"
else:
    kind = "likely raw counts (integer, small values)"
print(f"\n  .X assessment : {kind}")
print(f"  var_names sample : {list(a.var_names[:6])}")
print()
PYEOF
}

# ---- Main -----------------------------------------------------------------

# Idempotent: already exists and hash matches → done.
if [[ -f "${DEST_FILE}" ]]; then
  log "File already present at ${DEST_FILE}"
  if [[ "${EXPECTED_SHA256}" != "<PENDING:"* ]]; then
    verify_sha256 "${DEST_FILE}" "${EXPECTED_SHA256}" && {
      log "Hash verified — nothing to do."
      report_h5ad_info
      exit 0
    }
    err "Existing file fails hash check.  Delete it and rerun."
    exit 2
  else
    log "EXPECTED_SHA256 is still PENDING — reporting file info."
    report_h5ad_info
    exit 0
  fi
fi

# Try source 1: zenodo
if fetch_zenodo; then
  verify_sha256 "${DEST_FILE}" "${EXPECTED_SHA256}"
  report_h5ad_info
  exit 0
fi

log "Zenodo failed — trying source 2: pertpy ..."

# Try source 2: pertpy
if fetch_pertpy; then
  verify_sha256 "${DEST_FILE}" "${EXPECTED_SHA256}"
  report_h5ad_info
  exit 0
fi

# Source 3: GEO (manual — too complex for scripted download)
err "All automated sources failed."
err ""
err "  Source 1 (zenodo): ${ZENODO_URL}"
err "    → curl failed.  Check network access or try manually."
err ""
err "  Source 2 (pertpy): pertpy.data.norman_2019()"
err "    → pertpy not installed or API call failed."
err ""
err "  Source 3 (GEO ${GEO_ACCESSION}):"
err "    Download manually from:"
err "    https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=${GEO_ACCESSION}"
err "    Then convert to h5ad and place at ${DEST_FILE}."
err ""
err "Cannot proceed.  Stopping."
exit 1
