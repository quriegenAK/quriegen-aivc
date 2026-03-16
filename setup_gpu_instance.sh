#!/usr/bin/env bash
# ============================================================================
# setup_gpu_instance.sh — Bootstrap a GPU instance for AIVC training
#
# Tested on:  Ubuntu 22.04 (Deep Learning AMI recommended)
# GPU:        NVIDIA A10G / T4 / V100 (any CUDA 11.8+ compatible)
# Instance:   g5.xlarge (A10G, 24 GB VRAM) or g4dn.xlarge (T4, 16 GB VRAM)
#
# PREREQUISITES:
#   1. SSH access to the instance
#   2. AWS CLI configured (for S3 data download)
#   3. CUDA drivers installed (pre-installed on Deep Learning AMI)
#
# USAGE:
#   scp setup_gpu_instance.sh ubuntu@<instance-ip>:~/
#   ssh ubuntu@<instance-ip>
#   chmod +x setup_gpu_instance.sh
#   ./setup_gpu_instance.sh
# ============================================================================

set -euo pipefail

BUCKET="quriegen-aivc-data"
REGION="eu-west-1"
REPO="https://github.com/quriegenAK/quriegen-aivc.git"

echo "============================================"
echo "  AIVC GPU Instance Setup"
echo "============================================"

# 1. System packages
echo ""
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv git

# 2. Clone repository
echo ""
echo "[2/6] Cloning repository..."
if [ ! -d "quriegen-aivc" ]; then
    git clone "${REPO}"
fi
cd quriegen-aivc

# 3. Create virtual environment
echo ""
echo "[3/6] Creating virtual environment..."
python3 -m venv aivc_env
source aivc_env/bin/activate

# 4. Install dependencies (GPU-enabled PyTorch)
echo ""
echo "[4/6] Installing Python dependencies..."

# Pin numpy first for torch compatibility
pip install --quiet "numpy<2"

# Install PyTorch with CUDA support
pip install --quiet torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118

# Install torch-geometric (will use CUDA automatically)
pip install --quiet torch-geometric==2.7.0

# Install remaining dependencies
pip install --quiet -r requirements.txt

# Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 5. Download data from S3
echo ""
echo "[5/6] Downloading data from S3..."
mkdir -p data
aws s3 sync "s3://${BUCKET}/data/" data/ --region "${REGION}"
aws s3 sync "s3://${BUCKET}/models/" ./ --region "${REGION}"

echo ""
echo "  Downloaded files:"
ls -lh data/*.h5ad data/*.npy 2>/dev/null || echo "  (no data files yet — run setup_s3_bucket.sh first)"
ls -lh *.pt 2>/dev/null || echo "  (no model files yet)"

# 6. Verify setup
echo ""
echo "[6/6] Verifying setup..."
python -c "
import torch
import torch_geometric
import scanpy
import anndata
import ot
import numpy

print(f'torch:           {torch.__version__}')
print(f'CUDA:            {torch.cuda.is_available()}')
print(f'torch_geometric: {torch_geometric.__version__}')
print(f'numpy:           {numpy.__version__}')
print()
print('All imports OK — ready to train.')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  To train:"
echo "    source aivc_env/bin/activate"
echo "    python train_week3.py"
echo ""
echo "  To evaluate:"
echo "    python evaluate_week3.py"
echo "============================================"
