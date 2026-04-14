#!/usr/bin/env bash
# ============================================================================
# stage_data.sh — Transfer AIVC training data from S3 to MareNostrum5 GPFS.
#
# CONTEXT: MareNostrum5 compute nodes have NO internet. Login nodes DO have
# limited internet. Run this script on the login node ONCE to pre-stage all
# training inputs onto /gpfs/scratch/ before submitting SLURM array jobs.
#
# SOURCE OF TRUTH: s3://quriegen-aivc-data (eu-west-1), created by
# setup_s3_bucket.sh. Wout's CellRanger outputs live in the same bucket under
# a different prefix; adjust DATA_PREFIX if you reorganize.
#
# PREREQUISITES (on MN5 login node):
#   1. AWS credentials for the quriegen-aivc-data bucket (read-only OK)
#      -> `aws configure` or `~/.aws/credentials`
#   2. `aws` CLI available — check with `module avail aws` or `which aws`
#      If unavailable, install via: pip install --user awscli
#   3. Allocation space: 2000 GB on scratch. Current data is ~few GB so fine.
#
# USAGE:
#   chmod +x scripts/stage_data.sh
#   ./scripts/stage_data.sh              # default: stage everything
#   ./scripts/stage_data.sh --dry-run    # show what would happen
#   ./scripts/stage_data.sh --data-only  # skip model checkpoints
#
# IDEMPOTENT: safe to re-run. `aws s3 sync` only transfers diffs.
# ============================================================================

set -euo pipefail

BUCKET="${BUCKET:-quriegen-aivc-data}"
REGION="${REGION:-eu-west-1}"

# MareNostrum5 layout (override via env if needed)
SCRATCH_ROOT="${SCRATCH_ROOT:-/gpfs/scratch/ehpc323/aivc}"
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/ehpc323/aivc}"
DATA_DIR="${SCRATCH_ROOT}/data"
CHECKPOINT_DIR="${SCRATCH_ROOT}/checkpoints"
MODEL_DIR="${PROJECT_ROOT}/models"

DRY_RUN=""
STAGE_MODELS=1
STAGE_DATA=1
for arg in "$@"; do
    case "$arg" in
        --dry-run)    DRY_RUN="--dryrun" ;;
        --data-only)  STAGE_MODELS=0 ;;
        --models-only) STAGE_DATA=0 ;;
        -h|--help)
            grep '^# ' "$0" | sed 's/^# //'
            exit 0
            ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  AIVC data staging (login node -> GPFS)"
echo "  Source:     s3://${BUCKET}/  (region: ${REGION})"
echo "  Data dest:  ${DATA_DIR}"
echo "  Model dest: ${MODEL_DIR}"
echo "  Mode:       ${DRY_RUN:-live transfer}"
echo "============================================================"

# ─── Preflight ──────────────────────────────────────────────────────────────
if ! command -v aws >/dev/null 2>&1; then
    echo "[FATAL] aws CLI not found. Try: pip install --user awscli"
    exit 2
fi
if ! aws sts get-caller-identity >/dev/null 2>&1; then
    echo "[FATAL] AWS credentials not configured or expired."
    echo "        Run: aws configure"
    exit 3
fi

# ─── Layout ─────────────────────────────────────────────────────────────────
mkdir -p "${DATA_DIR}" "${CHECKPOINT_DIR}" "${MODEL_DIR}"

# ─── Stage training data ────────────────────────────────────────────────────
if [ "${STAGE_DATA}" = "1" ]; then
    echo ""
    echo "[1/2] Syncing training data..."
    aws s3 sync \
        "s3://${BUCKET}/data/" "${DATA_DIR}/" \
        --region "${REGION}" \
        --exclude "*" \
        --include "*.h5ad" \
        --include "*.h5mu" \
        --include "*.npy" \
        --include "*.csv" \
        --include "*.txt" \
        ${DRY_RUN}
    echo ""
    echo "  Data staged. Contents:"
    du -h -d 1 "${DATA_DIR}" 2>/dev/null | tail -20 || true
fi

# ─── Stage pre-existing model checkpoints (so we can resume v1.0 eval) ─────
if [ "${STAGE_MODELS}" = "1" ]; then
    echo ""
    echo "[2/2] Syncing model checkpoints..."
    aws s3 sync \
        "s3://${BUCKET}/models/" "${MODEL_DIR}/" \
        --region "${REGION}" \
        ${DRY_RUN}
    echo ""
    echo "  Models staged:"
    ls -lh "${MODEL_DIR}"/*.pt 2>/dev/null | tail -20 || echo "  (none yet)"
fi

# ─── Post-staging integrity checks ──────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Integrity check"
echo "============================================================"
REQUIRED_FILES=(
    "kang2018_pbmc_fixed.h5ad"
    "X_ctrl_ot.npy"
    "X_stim_ot.npy"
    "cell_type_ot.npy"
    "donor_ot.npy"
    "edge_list_fixed.csv"
)
missing=0
for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "${DATA_DIR}/${f}" ]; then
        size=$(stat -c%s "${DATA_DIR}/${f}" 2>/dev/null || stat -f%z "${DATA_DIR}/${f}")
        printf "  [OK]   %-32s  %s bytes\n" "${f}" "${size}"
    else
        printf "  [MISS] %-32s  NOT FOUND\n" "${f}"
        missing=$((missing + 1))
    fi
done

if [ "${missing}" != "0" ]; then
    echo ""
    echo "[FATAL] ${missing} required file(s) missing — training will fail."
    exit 4
fi

echo ""
echo "  Disk usage after staging:"
quota -s 2>/dev/null || df -h "${SCRATCH_ROOT}" 2>/dev/null || true

echo ""
echo "============================================================"
echo "  Staging complete."
echo "  Next: submit the sweep"
echo "    sbatch scripts/train_v11.sbatch"
echo "============================================================"
