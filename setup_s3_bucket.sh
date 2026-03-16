#!/usr/bin/env bash
# ============================================================================
# setup_s3_bucket.sh — Create and configure the AIVC data bucket on AWS S3
#
# Bucket:  quriegen-aivc-data
# Region:  eu-west-1
#
# PREREQUISITES:
#   1. AWS CLI installed:        brew install awscli
#   2. Credentials configured:   aws configure
#   3. IAM permissions:          s3:CreateBucket, s3:PutBucketVersioning,
#                                s3:PutBucketEncryption, s3:PutObject
#
# USAGE:
#   chmod +x setup_s3_bucket.sh
#   ./setup_s3_bucket.sh
#
# This script is IDEMPOTENT — safe to run multiple times.
# ============================================================================

set -euo pipefail

BUCKET="quriegen-aivc-data"
REGION="eu-west-1"

echo "============================================"
echo "  AIVC S3 Bucket Setup"
echo "  Bucket: s3://${BUCKET}"
echo "  Region: ${REGION}"
echo "============================================"

# 1. Create bucket
echo ""
echo "[1/4] Creating bucket..."
aws s3api create-bucket \
    --bucket "${BUCKET}" \
    --region "${REGION}" \
    --create-bucket-configuration LocationConstraint="${REGION}"
echo "  Bucket created: s3://${BUCKET}"

# 2. Enable versioning
echo ""
echo "[2/4] Enabling versioning..."
aws s3api put-bucket-versioning \
    --bucket "${BUCKET}" \
    --versioning-configuration Status=Enabled
echo "  Versioning enabled"

# 3. Enable server-side encryption (AES-256)
echo ""
echo "[3/4] Enabling server-side encryption..."
aws s3api put-bucket-encryption \
    --bucket "${BUCKET}" \
    --server-side-encryption-configuration '{
        "Rules": [{
            "ApplyServerSideEncryptionByDefault": {
                "SSEAlgorithm": "AES256"
            },
            "BucketKeyEnabled": true
        }]
    }'
echo "  SSE-S3 (AES-256) encryption enabled"

# 4. Upload data files
echo ""
echo "[4/4] Uploading data files..."

DATA_DIR="data"
FILES_TO_UPLOAD=(
    "kang2018_pbmc.h5ad"
    "kang2018_pbmc_fixed.h5ad"
    "X_ctrl_ot.npy"
    "X_stim_ot.npy"
    "cell_type_ot.npy"
    "donor_ot.npy"
    "X_ctrl_paired.npy"
    "X_stim_paired.npy"
    "edge_list.csv"
    "edge_list_fixed.csv"
    "ot_pairing_manifest.csv"
    "pairing_manifest.csv"
    "ot_pairing_log.txt"
)

for f in "${FILES_TO_UPLOAD[@]}"; do
    if [ -f "${DATA_DIR}/${f}" ]; then
        echo "  Uploading ${f}..."
        aws s3 cp "${DATA_DIR}/${f}" "s3://${BUCKET}/data/${f}"
    else
        echo "  SKIP: ${DATA_DIR}/${f} not found"
    fi
done

# Upload model checkpoints from project root
MODEL_FILES=(
    "model_week3.pt"
    "model_week3_best.pt"
    "model_perturbation.pt"
    "model_fixed.pt"
)

for f in "${MODEL_FILES[@]}"; do
    if [ -f "${f}" ]; then
        echo "  Uploading ${f}..."
        aws s3 cp "${f}" "s3://${BUCKET}/models/${f}"
    else
        echo "  SKIP: ${f} not found"
    fi
done

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Bucket: s3://${BUCKET}"
echo "  Data:   s3://${BUCKET}/data/"
echo "  Models: s3://${BUCKET}/models/"
echo "============================================"
echo ""
echo "To download data on a new machine:"
echo "  aws s3 sync s3://${BUCKET}/data/ data/"
echo "  aws s3 sync s3://${BUCKET}/models/ ./"
