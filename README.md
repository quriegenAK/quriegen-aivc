# AIVC GeneLink

Graph neural network for predicting cellular perturbation responses from single-cell RNA-seq data.

## Overview

AIVC GeneLink uses a Graph Attention Network (GAT) to predict how gene expression changes when cells are perturbed (e.g., IFN-beta stimulation). The model operates on a gene regulatory network derived from STRING protein-protein interactions and learns to predict stimulated expression profiles from control expression.

**Key result:** Pearson r = 0.873 on held-out test donors (Kang 2018 PBMC dataset).

## Architecture

- **Graph**: 3,010 genes connected by 13,878 STRING PPI edges (score >= 700)
- **Model**: 2-layer GAT with perturbation and cell-type embeddings
- **Training**: Residual learning (predict delta, output = ctrl + delta)
- **Loss**: MSE + log-fold-change + cosine similarity (phased warm-up)
- **Data**: Optimal Transport cell pairing with stochastic mini-bulk aggregation
- **Split**: Donor-based (4 train, 1 val, 3 test donors)

## Project Structure

```
.
├── train_week3.py              # Main training script (Week 3 — current best)
├── perturbation_model.py       # PerturbationResponseModel (GAT + decoder)
├── losses.py                   # Combined loss: MSE + LFC + cosine
├── build_ot_pairs.py           # Optimal Transport cell pairing
├── build_edge_list.py          # STRING PPI edge list construction
├── evaluate_week3.py           # Held-out test evaluation
├── extract_attention_week3.py  # GAT attention weight analysis
├── demo_aivc_week3.ipynb       # Demo notebook with visualizations
├── test_model.py               # Regression test suite
├── requirements.txt            # Python dependencies
├── setup_s3_bucket.sh          # AWS S3 bucket setup (data storage)
├── setup_gpu_instance.sh       # GPU instance bootstrap script
└── data/                       # Data directory (large files on S3)
    ├── edge_list.csv           # STRING PPI edges
    ├── Expression.csv          # GeneLink reference expression
    ├── BL--network.csv         # GeneLink reference network
    └── ...
```

## Quick Start

### Local (CPU)

```bash
# Clone
git clone https://github.com/quriegenAK/quriegen-aivc.git
cd quriegen-aivc

# Environment
python3 -m venv aivc_env
source aivc_env/bin/activate
pip install numpy==1.26.4
pip install -r requirements.txt

# Download data from S3 (requires AWS credentials)
aws s3 sync s3://quriegen-aivc-data/data/ data/
aws s3 sync s3://quriegen-aivc-data/models/ ./

# Train
python train_week3.py

# Evaluate
python evaluate_week3.py

# Test suite
python test_model.py
```

### GPU (AWS)

```bash
# On a fresh GPU instance (g5.xlarge recommended):
chmod +x setup_gpu_instance.sh
./setup_gpu_instance.sh
```

## Data

Large files (`.h5ad`, `.npy`, `.pt`) are stored on S3 (`s3://quriegen-aivc-data/`) and excluded from git. To set up the S3 bucket:

```bash
chmod +x setup_s3_bucket.sh
./setup_s3_bucket.sh
```

**Dataset:** Kang 2018 PBMC — 24,673 cells, 8 cell types, IFN-beta stimulation.

## Results

| Metric | Value |
|--------|-------|
| Test Pearson r | 0.873 +/- 0.064 |
| Test MSE | 0.013 |
| JAK-STAT pathway recovery | 7/15 genes within 3x |
| Cell-type spread | 0.198 (range of per-type r) |

## Requirements

- Python 3.11+
- PyTorch 2.2+
- PyTorch Geometric 2.7+
- See `requirements.txt` for full list

## License

Proprietary. Copyright Quriegen.
