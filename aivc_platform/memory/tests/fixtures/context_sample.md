### Project: aivc_genelink

Last updated: 2026-04-10

### Mission
Multi-omics AI Virtual Cell platform that predicts perturbation response in primary immune cells from paired QuRIE-seq RNA + Protein + Phospho + ATAC single-cell measurements.

### Stack snapshot
Python, PyTorch 2.2.2, torch-geometric 2.7.0, torchvision 0.17.2, numpy 1.26.4, scipy, scanpy 1.11.5, anndata, POT (Python Optimal Transport), pandas, matplotlib, seaborn. API layer: FastAPI, uvicorn[standard], pydantic v2, requests (for AIVCClient SDK). Storage: TileDB-SOMA (gated behind `USE_SOMA_STORE` env flag, default False). Experiment tracking: MLflow (via `aivc/memory/mlflow_backend.py`). Python package dir `aivc/` (subpackages: critics, data, edge_cases, memory, orchestration, preprocessing, qc, skills).

### Architecture pattern
Modular ML research monorepo + FastAPI inference service. Top-level training/eval scripts co-exist with a packaged `aivc/` library of orchestration, QC, critics, preprocessing, and skill modules. The inference API (`api/server.py`) loads the trained checkpoint once at lifespan start and exposes REST endpoints. Trained artifacts live under `models/v1.0/` and `models/week4/`. Containerisation scaffolding is present (`containers/`). Pairing governance is enforced via certificate files (`data/pairing_certificates/quriegen_pending.json`) rather than runtime flags.

### Key files / entry points
- `train_week3.py` — v1.0 GAT training loop, produced the live r=0.873 checkpoint.
- `train_v11.py` — v1.1 Neumann-cascade sweep (coded, awaiting GPU; W scale fixed to [0, 0.1]).
- `perturbation_model.py` — GAT (2-layer, 4-head) RNA encoder + ResponseDecoder architecture.
- `losses.py` — MSE + log-fold-change + cosine + L1 + causal-ordering loss.
- `api/server.py` — FastAPI inference service, endpoints `/health`, `/model/info`, `/predict`, `/intervene`, `/model/top_edges`, `/model/jakstat`.

### Active APIs
FastAPI routes exposed from `api/server.py`:
- `GET /health` — liveness.
- `GET /model/info` — version, benchmark r, capability flags.
- `POST /predict` — perturbation response prediction over 3,010 genes.
- `POST /intervene` — SCM do(X) causal intervention (temporal-causal mask, not Pearl SCM).
- `GET /model/top_edges` — top-N edges from Neumann W matrix.
- `GET /model/jakstat` — JAK-STAT pathway edge weights (`JAKSTAT_GENES` constant).

Client SDK: `api/client.py` (requests-based `AIVCClient`).

### Models / ML components
- RNA encoder — 2-layer, 4-head GAT on STRING PPI, 3,010 genes → 128-dim. Live. Kang 2018 checkpoint r=0.873 held-out donors.
- Protein encoder — MLP + cross-attention, 200 antibodies → 128-dim. Coded, untrained, awaiting QuRIE-seq data (May 2026).
- Phospho encoder — MLP + kinase-substrate graph → 64-dim. Architecture defined, no implementation class yet.
- ATAC encoder — chromVAR TF scores → 64-dim (`aivc/skills/atac_encoder.py`). Coded, untrained, awaiting 10x Multiome.
- Temporal cross-modal fusion — 384-dim, causal mask enforced (lower-triangular, upper-triangle weight = 0.0). Ordering ATAC(t=0) → Phospho(t=1) → RNA(t=2) → Protein(t=3). Not multi-modal trained.
- Neumann propagation cascade — K=3, sparsity enforced, W scale fixed to [0, 0.1] in v1.2. Sweep pending re-run.
- Contrastive InfoNCE loss — gated on pairing certificate; disabled until QuRIE-seq pairing confirmed.
- 3-critic validation gate — `aivc/critics/{biological,statistical,methodological}.py`, includes mechanistic direction checks.
- Perturbation curriculum orchestrator — `aivc/orchestration/perturbation_curriculum.py`, 4-stage curriculum with synthetic gate.

Benchmarks: Pearson r=0.873 vs CPA r=0.856, scGEN r=0.820 on identical Kang 2018 test split. Zero-shot Norman 2019 K562 aggregate r=0.9908 but delta-r=0.0 — model predicts ctrl profile unchanged across all 17 CRISPRi perturbations. Documented as "MEMORISED" in README; expected for v1.0 single-perturbation training.

### Data sources
- Kang 2018 (GSE96583) — response training + benchmark. IFN-β stimulation of PBMCs.
- PBMC IFN-γ (OneK1K / Dixit 2016) — Stage 2 response training.
- Frangieh 2021 (SCP1064) — W-pretraining only, `USE_FOR_W_ONLY=True` enforced. Blocked from response training: melanoma JAK-STAT biology (BCR-ABL1, constitutive JAK2) is biologically wrong for PBMC IFN response.
- ImmPort SDY702 — W-pretraining only (cytokine GRN).
- Replogle 2022 (K562) — W-pretraining only, 267 housekeeping safe genes. 72 genes explicitly blocked (JAK-STAT, BCR-ABL1, oncogenes, immune receptors) because K562 constitutive JAK2/STAT5 would contaminate PBMC W matrix.
- Norman 2019 K562 — zero-shot eval only (memorised baseline).
- QuRIE-seq paired RNA+Protein+Phospho — pending wet-lab (May 2026). Pairing certificate at `data/pairing_certificates/quriegen_pending.json`, all modalities marked `pairing_type: "unknown"`.
- Local `data/geo_cache/` for downloaded datasets. STRING PPI edge list under `build_edge_list.py` output.
- On-disk anndata (`.h5ad`) is the primary runtime data format. Optional TileDB-SOMA bridge in `aivc/data/soma_training_bridge.py`.

### Open questions
- [ ] Neumann cascade K=3 W-scale sweep currently fixed at [0, 0.1] in v1.2 — GPU availability confirmed for sweep? Timeline for v1.3 benchmark run?
- [ ] Norman 2019 delta=0 (model memorised ctrl, zero perturbation signal) — fix decision needed: retrain with ctrl-subtracted targets, or architectural change to enforce non-zero Δ at loss level?
- [ ] Protein/Phospho/ATAC encoders present in codebase but untrained — proceed with synthetic pairs, or block on QuRIE-seq pairing certificate (quriegen_pending.json) before initialising encoder training?

### Do not touch
- inference_engine/core/ — Kang 2018 checkpoint at Pearson r=0.873. Do not retrain or modify without a full benchmark run against Norman 2019 and the Replogle 267-gene safe set.

### GitHub
Repo path: /Users/ashkhan/Projects/aivc_genelink
Active branch: [leave blank — Ash to fill]
Open PRs: [leave blank — Ash to fill]
