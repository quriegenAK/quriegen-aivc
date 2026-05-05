#!/usr/bin/env bash
# Stage 3 Part 1 Report 2 — Mimitou CD4 CRISPR data download.
#
# Pulls the processed Mimitou 2021 CD4 CRISPR ASAP-seq files from the
# caleblareau/asap_reproducibility GitHub repo into a local staging dir,
# then optionally rsync's to BSC scratch.
#
# Source: https://github.com/caleblareau/asap_reproducibility/tree/master/CD4_CRISPR_asapseq
#
# WHY THIS REPO INSTEAD OF GEO?
#   GEO GSE156478 supplementary contains only ATAC FRAGMENTS (raw),
#   not the processed Cell Ranger filtered_peak_bc_matrix.h5. Reprocessing
#   would require ~50 GB intermediate disk + cellranger-atac. The
#   asap_reproducibility repo bundles the post-processed h5 (~87 MB) plus
#   the kite-format ADT + HTO matrices that match the paper's Figure 5
#   analyses — exactly what we need.
#
# Files staged (~110 MB total):
#   filtered_peak_bc_matrix.h5          (87.3 MB) — Cell Ranger ATAC h5
#   kallisto/adt/featurecounts.mtx      (14.5 MB) — protein counts
#   kallisto/adt/featurecounts.barcodes.txt
#   kallisto/adt/featurecounts.genes.txt
#   kallisto/hto/featurecounts.mtx      (3.7 MB) — HTO counts (encodes sgRNA target)
#   kallisto/hto/featurecounts.barcodes.txt
#   kallisto/hto/featurecounts.genes.txt
#   hashtag_list.txt                              — HTO → sgRNA target map
#   sgRNA_list.txt                                — guide name list
#   ADT_Gene_corresp_list.txt                     — ADT → gene name map
#   singlecell.csv                      (31.6 MB) — Cell Ranger per-cell QC
#
# Usage:
#   bash scripts/download_mimitou_crispr.sh \
#       --target_dir ~/data/mimitou_crispr_raw \
#       [--bsc]   # rsync to BSC scratch when done
#
# Prereqs: curl, optional rsync+ssh access to BSC.

set -euo pipefail

# --- Defaults ---
TARGET_DIR=""
SYNC_BSC=0
BSC_USER="quri020505"
BSC_LOGIN_HOST="alogin1.bsc.es"
BSC_PATH="/gpfs/scratch/ehpc748/quri020505/data/mimitou_crispr_raw"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target_dir) TARGET_DIR="$2"; shift 2 ;;
        --bsc) SYNC_BSC=1; shift ;;
        --bsc_path) BSC_PATH="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,40p' "$0"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 2 ;;
    esac
done

[[ -n "$TARGET_DIR" ]] || { echo "ABORT: --target_dir required"; exit 1; }

# --- File manifest ---
REPO_URL="https://raw.githubusercontent.com/caleblareau/asap_reproducibility/master/CD4_CRISPR_asapseq/data"

# Format: relative_path_in_repo|local_subpath
FILES=(
    "filtered_peak_bc_matrix.h5|filtered_peak_bc_matrix.h5"
    "singlecell.csv|singlecell.csv"
    "sgRNA_list.txt|sgRNA_list.txt"
    "ADT_Gene_corresp_list.txt|ADT_Gene_corresp_list.txt"
    "kallisto/hashtag_list.txt|hashtag_list.txt"
    "kallisto/adt/featurecounts.mtx|adt/featurecounts.mtx"
    "kallisto/adt/featurecounts.barcodes.txt|adt/featurecounts.barcodes.txt"
    "kallisto/adt/featurecounts.genes.txt|adt/featurecounts.genes.txt"
    "kallisto/hto/featurecounts.mtx|hto/featurecounts.mtx"
    "kallisto/hto/featurecounts.barcodes.txt|hto/featurecounts.barcodes.txt"
    "kallisto/hto/featurecounts.genes.txt|hto/featurecounts.genes.txt"
)

mkdir -p "$TARGET_DIR" "$TARGET_DIR/adt" "$TARGET_DIR/hto"

echo "=== Mimitou CD4 CRISPR download ==="
echo "  source: $REPO_URL"
echo "  target: $TARGET_DIR"
echo "  files:  ${#FILES[@]}"
echo

# --- Download ---
TOTAL_BYTES=0
for entry in "${FILES[@]}"; do
    src="${entry%|*}"
    dst="${entry#*|}"
    full_src="$REPO_URL/$src"
    full_dst="$TARGET_DIR/$dst"

    if [[ -f "$full_dst" ]] && [[ -s "$full_dst" ]]; then
        sz=$(stat -f%z "$full_dst" 2>/dev/null || stat -c%s "$full_dst")
        printf "  [skip] %s (already present, %d bytes)\n" "$dst" "$sz"
        TOTAL_BYTES=$((TOTAL_BYTES + sz))
        continue
    fi

    printf "  [pull] %s ... " "$dst"
    # GitHub raw redirects to raw.githubusercontent.com; -L follows.
    # --fail returns nonzero on HTTP 4xx/5xx.
    if curl -sSL --fail --connect-timeout 30 --max-time 600 \
            -o "$full_dst" "$full_src"; then
        sz=$(stat -f%z "$full_dst" 2>/dev/null || stat -c%s "$full_dst")
        echo "OK ($sz bytes)"
        TOTAL_BYTES=$((TOTAL_BYTES + sz))
    else
        echo "FAILED"
        echo "  ABORT: download failed for $full_src"
        exit 3
    fi
done

echo
echo "=== Download complete: $TOTAL_BYTES bytes ==="
echo

# --- Sanity-check: confirm filtered_peak_bc_matrix.h5 is a valid HDF5 ---
echo "=== Sanity checks ==="
H5_PATH="$TARGET_DIR/filtered_peak_bc_matrix.h5"
H5_HEAD=$(head -c 8 "$H5_PATH" | od -An -c | tr -s ' ' | head -1)
if [[ "$H5_HEAD" == *"H D F"* ]] || [[ "$H5_HEAD" == *"\\211   H   D   F"* ]]; then
    echo "  [ok] filtered_peak_bc_matrix.h5 has HDF5 magic"
else
    echo "  [warn] filtered_peak_bc_matrix.h5 magic bytes unexpected: $H5_HEAD"
    echo "         (file may be a Git LFS pointer rather than the actual h5)"
    echo "         First 200 bytes:"
    head -c 200 "$H5_PATH"; echo
fi

# Confirm hashtag_list.txt has expected sgRNA targets
HT_TARGETS=$(awk -F'\t' 'NR>1 && $3 ~ /sg/ {print $3}' "$TARGET_DIR/hashtag_list.txt" | sort -u | tr '\n' ' ')
echo "  [ok] hashtag targets: $HT_TARGETS"
EXPECTED_TARGETS="sgCD3E sgCD4 sgGuide1 sgGuide2 sgNFKB2 sgNTC sgZAP70"
if [[ "$HT_TARGETS" == "$EXPECTED_TARGETS " ]]; then
    echo "  [ok] target set matches expected panel"
else
    echo "  [warn] target set differs from expected"
    echo "         expected: $EXPECTED_TARGETS"
fi

# Confirm protein .mtx is non-trivial
ADT_LINES=$(wc -l < "$TARGET_DIR/adt/featurecounts.mtx")
HTO_LINES=$(wc -l < "$TARGET_DIR/hto/featurecounts.mtx")
echo "  [ok] adt/featurecounts.mtx lines: $ADT_LINES"
echo "  [ok] hto/featurecounts.mtx lines: $HTO_LINES"
if [[ "$ADT_LINES" -lt 1000 ]] || [[ "$HTO_LINES" -lt 1000 ]]; then
    echo "  [warn] kite mtx files seem small — may be Git LFS pointers."
fi

# --- Optional: rsync to BSC ---
if [[ "$SYNC_BSC" -eq 1 ]]; then
    echo
    echo "=== rsync to BSC ==="
    echo "  destination: ${BSC_USER}@${BSC_LOGIN_HOST}:${BSC_PATH}/"
    # --mkpath creates missing parent dirs on the remote (rsync 3.2.3+).
    # Without it, rsync fails if e.g. ${BSC_PATH%/*}/ doesn't exist yet.
    rsync -avz --progress --mkpath "$TARGET_DIR/" \
        "${BSC_USER}@${BSC_LOGIN_HOST}:${BSC_PATH}/"
    echo
    echo "  Done. On BSC, contents at: $BSC_PATH"
fi

echo
echo "=== READY ==="
echo "  Local stage:  $TARGET_DIR"
if [[ "$SYNC_BSC" -eq 1 ]]; then
    echo "  BSC stage:    ${BSC_USER}@${BSC_LOGIN_HOST}:${BSC_PATH}"
    echo
    echo "  Launch Report 2 on BSC with:"
    echo "    ssh ${BSC_USER}@${BSC_LOGIN_HOST}"
    echo "    cd /gpfs/scratch/ehpc748/quri020505/aivc_genelink"
    echo "    sbatch scripts/submit_mimitou_perturbation_probe.slurm \\"
    echo "        /gpfs/scratch/ehpc748/quri020505/checkpoints/pretrain/pretrain_encoders.pt \\"
    echo "        ${BSC_PATH}"
else
    echo
    echo "  To rsync to BSC, re-run with --bsc:"
    echo "    bash scripts/download_mimitou_crispr.sh --target_dir $TARGET_DIR --bsc"
fi
