# Reference Repositories — Fitness Review for AIVC

**Author:** Ash Khan
**Date:** 2026-04-10
**Scope:** `nf-cellranger-multi`, `nf-star-align`, `bioinf-containers`, `aivc-process-10x-atac-rna`
**Evidence base:** Wout Megchelenbrink's Confluence pages (PM space) + grep of `aivc/preprocessing/` in this repo + git log of `/Users/ashkhan/Projects/aivc_genelink`

---

## TL;DR

| Repo | Owner | Verdict | One-line reasoning |
|---|---|---|---|
| `nf-cellranger-multi` | Wout / AWS Batch | **KEEP (external dependency)** | Upstream producer of the exact directory layout `aivc/preprocessing/atac_rna_pipeline.py` consumes; only path to QuRIE-seq data (Task 7 of Wout's training roadmap). |
| `nf-star-align` | Wout / AWS Batch | **ARCHIVE** | Bulk RNA-seq STAR aligner — Wout's own page says *"we will probably not use it much in the near future"*. AIVC consumes 10x Multiome, not bulk STAR counts. |
| `bioinf-containers` | Wout / ECR | **ARCHIVE** (for AIVC) | Interactive Rstudio/VScode containers on AWS ECR. Orthogonal to `containers/aivc-train.def` (HPC/SLURM training container we just built). |
| `aivc-process-10x-atac-rna` | Wout / R | **DELETE-after-parity** | `aivc/preprocessing/__init__.py` already states it *"Replaces Wout Megchelenbrink's R-based Seurat/Signac pipeline"*. Superseded in-tree. Retain only as a short-lived parity oracle, then drop. |

**Net:** Out of 4 reference repos, only 1 (`nf-cellranger-multi`) is a live dependency. It is a data-producer we consume, not code we reuse. The other 3 contribute zero lines to AIVC.

---

## Ground Truth from the AIVC Codebase

Grepping `aivc/preprocessing/` shows the intent was always to absorb, not depend on, the R pipeline:

- `aivc/preprocessing/__init__.py` — *"Replaces Wout Megchelenbrink's R-based Seurat/Signac pipeline with a production-grade Python implementation using muon/scanpy."*
- `aivc/preprocessing/atac_rna_pipeline.py` — *"Absorbs and extends Wout Megchelenbrink's R-based pipeline"*, expects `cellranger_output_dir/filtered_feature_bc_matrix.h5` and `cellranger_output_dir/atac_fragments.tsv.gz`.
- `aivc/preprocessing/tf_motif_scanner.py` — comment *"Wout's R pipeline had a stub here"*.
- `aivc/preprocessing/peak_gene_linker.py` — comment *"Wout flagged 100kb as a testing shortcut"*; default `window_kb=300`.
- `git log --author` shows only Ash Khan on AIVC. Wout has zero commits to this repo — his code lives in the 4 reference repos on AWS Batch / ECR, not here.

**Implication:** AIVC depends on the *output contract* (CellRanger directory layout) of `nf-cellranger-multi`, not on any source code from any of the 4 reference repos.

---

## Per-Repo Breakdown

### 1. `nf-cellranger-multi` — KEEP (as an external dependency, not as code to import)

**What it is.** Wout's NextFlow + AWS Batch wrapper around 10x Genomics CellRanger `multi` workflow. Confluence page `213385272` (Mar 27 2026) confirms it supports gex / vdj-t / vdj-b / antibody (surface, intracellular, phosphorylation) / cell hashing / γδ-TCR libraries. **Critically: it supports QuRIE-seq libraries.** Runs on AWS Batch via the `nextflow-ec2-manager` EC2 launch template, pulls FASTQs from `s3://quriegen-nextflow/input-data/<run>`, and writes CellRanger output directories back to S3.

**Relevance.** This is the only pipeline on Wout's roadmap (Confluence `90767783`, Task 7: *Unified multi-modal + QuRIE-seq model training*) that produces the exact input AIVC's training loop expects. The v1.2 AIVC Python preprocessing layer (`atac_rna_pipeline.py`) reads `filtered_feature_bc_matrix.h5` + `atac_fragments.tsv.gz`, which is exactly what CellRanger Multi emits. No other pipeline in the QurieGen ecosystem produces that layout.

**Practical usage.** Zero code is copied from this repo into AIVC, and that is the correct posture — it lives on AWS Batch, its source is at `github.com/quriegen/nf-cellranger-multi` (private), and we consume its outputs by URI. The only "integration" is the directory-layout contract.

**Future value.** High and time-critical. Wet-lab QuRIE-seq data lands end-May / mid-June per the roadmap. Without `nf-cellranger-multi` we have no path from FASTQ → `MuData` → AIVC training tensors.

**Jira alignment.** Aligns with QUGO-1596 (pipeline documentation) and with the Wet-Lab → AIVC handoff implied by QGRD wet-lab tickets. No Jira ticket asks us to re-implement it; we are consumers.

**Risk of keeping.** Minimal — it is owned by Wout on AWS, not by the AIVC team. Operational cost is effectively zero to us.

**Risk of archiving / deleting.** Very high — we lose the only supported path to ingest QuRIE-seq and 10x Multiome FASTQs into AIVC.

**Recommendation.** **KEEP as an external dependency.** Do not import, do not fork, do not shadow. Treat it as a black-box upstream service. What we *should* add on our side:
1. A locked data contract doc: "for each CellRanger output directory we expect these files with these shapes."
2. A one-shot validator (`aivc/preprocessing/validate_cellranger_output.py`) that fails loudly before training if the directory is malformed.
3. An AWS → GPFS sync leg in `stage_data.sh` once real QuRIE-seq outputs land, so H100 training on MN5 can consume them without Wout needing to touch SLURM.

---

### 2. `nf-star-align` — ARCHIVE

**What it is.** NextFlow + AWS Batch bulk RNA-seq pipeline using the STAR aligner, driven by a `samplesheet.tsv` with `library_id / read1 / read2` rows. Confluence page `213450846` (Apr 06 2026, Wout).

**Relevance — Wout's own words.** From Confluence `213385291`: *"A good starting point is to run the nf-star-align pipeline. … Although we will probably not use it much in the near future, the pipeline is conceptually simpler than the nf-cellranger pipeline."* This is explicitly positioned as a **training-wheels / tutorial pipeline** for new team members to learn the AWS-Batch-NextFlow workflow, not a production path.

**Practical usage.** Zero in AIVC. AIVC's data model is single-cell 10x Multiome + QuRIE-seq, not bulk RNA-seq from STAR. The output format (per-sample gene count matrix) does not match anything `aivc/preprocessing/` can consume, and there is no shim either.

**Future value.** Low unless we pivot to a bulk RNA-seq use case (e.g., reanalysing public TCGA / GEO bulk cohorts for GRN reconstruction). No Jira ticket currently plans this.

**Jira alignment.** None.

**Recommendation.** **ARCHIVE.** Do not delete — Wout uses it as onboarding / didactics material and it is paid for by his S3 budget, not ours. But do not track it on the AIVC roadmap, do not reference it in AIVC docs, do not wire it into `stage_data.sh`. If a bulk RNA-seq use case ever appears in Task 1 of Wout's roadmap (`90767783`), un-archive in one afternoon.

---

### 3. `bioinf-containers` — ARCHIVE (for AIVC purposes)

**What it is.** Confluence page `203653175` (Apr 02 2026, Wout): GitHub repo with Dockerfile recipes for **interactive** bioinformatics containers hosted on AWS ECR. Two flavours — Rstudio Server and VScode / coder server (browser IDE or local Remote-Containers). Integrated with EC2 launch templates for interactive data analysis.

**Relevance.** Orthogonal axis. `bioinf-containers` targets **interactive exploration on an EC2 instance**; `containers/aivc-train.def` (the Apptainer definition we shipped this week) targets **batch training on a MareNostrum5 H100 SLURM node**. The two sets of containers share zero base images, zero packages that matter (Rstudio Server vs. CUDA 12.2 + PyTorch 2.4.1 + PyG 2.7.0), and zero runtime constraints (ECR/Docker/internet vs. Apptainer/offline/GPFS).

**Practical usage.** Zero. We cannot lift anything from a Docker recipe targeting Rstudio into an Apptainer `.def` targeting H100 training.

**Future value.** Low for AIVC training; **moderate for AIVC interactive work** if we ever want to give Wout or a bench scientist a Rstudio-in-a-browser that talks to the same `MuData` artefacts AIVC produces. That is a UX concern, not a training concern, and should not be on the v1.1 critical path.

**Jira alignment.** None on the AIVC training side. Potentially relevant to a future "scientist-facing interactive workspace" ticket that does not currently exist.

**Recommendation.** **ARCHIVE for AIVC.** Leave it in Wout's control, do not fork it into AIVC, do not try to reconcile its Docker recipes with our Apptainer `.def`. Revisit only if/when an interactive-IDE requirement is filed.

---

### 4. `aivc-process-10x-atac-rna` — DELETE (after a short parity window)

**What it is.** Wout's R-based Seurat / Signac pipeline for 10x Multiome ATAC+RNA preprocessing. Not directly documented in the 5 Wout pages I just re-read (those were all NextFlow / ECR / AWS Batch pages), but its role is fixed by the AIVC in-tree docstrings quoted above.

**Relevance.** Superseded in-tree. `aivc/preprocessing/__init__.py` literally says *"Replaces Wout Megchelenbrink's R-based Seurat/Signac pipeline"*. We are not reusing code from this repo — we are re-implementing its behaviour in Python on top of `muon` / `scanpy`.

**Practical usage.** Zero lines of R executed anywhere in AIVC today. Every module that would have called into it (`peak_gene_linker.py`, `tf_motif_scanner.py`, `atac_rna_pipeline.py`) has a native Python implementation or an explicit stub. **Known gap:** `compute_chromvar_scores()` in `tf_motif_scanner.py` is still a stub; replacing it does *not* require running the R repo, it requires porting chromVAR semantics to Python (see Key Risks).

**Future value.** Only one: **parity validation**. Until the Python pipeline runs end-to-end on real 10x Multiome output and the numbers are cross-checked against Wout's R outputs on a tiny reference dataset, the R repo has residual value as a semantic oracle. Post-parity it is dead weight.

**Jira alignment.** Wet-lab QGRD tickets feed into QUGO pipeline tickets that feed into AIVC preprocessing — but the handoff is at the CellRanger output directory, not at Seurat objects. The R pipeline is outside that handoff.

**Recommendation.** **DELETE-after-parity.**
1. Keep the R repo accessible for 4-6 weeks as a parity oracle.
2. Port `compute_chromvar_scores()` to Python (separate task, not blocked on the R repo).
3. Run both pipelines on a tiny 10x Multiome reference (~500 cells) side-by-side, produce a one-page delta report (% peak overlap, correlation of per-cell gene-accessibility scores).
4. Once the delta report is green, delete the repo from the AIVC reference set and remove all `"Replaces Wout's R pipeline"` comments (they will have served their purpose).

---

## Cross-Cutting Observations

**1. All 4 reference repos target AWS Batch / ECR. None targets SLURM / HPC.** Our new v1.1 training stack (`scripts/train_hpc.py` + `scripts/train_v11.sbatch` + `containers/aivc-train.def` + `scripts/stage_data.sh`) is HPC-native and shares zero infrastructure with the reference set. This is fine — we should not try to force-port AWS Batch idioms onto MareNostrum5 — but it means "reusing patterns from the reference repos" is a false economy.

**2. The AWS → HPC data bridge is the only real integration surface.** Wout's `nf-cellranger-multi` writes outputs to S3. `scripts/stage_data.sh` already knows how to pull from `s3://quriegen-aivc-data`. What's missing is a **bucket contract** — Wout writes CellRanger outputs to a known prefix (suggest `s3://quriegen-aivc-data/cellranger/<run_id>/`), and `stage_data.sh` gains a `--cellranger-run <run_id>` flag that syncs that prefix into GPFS.

**3. Wout's "AIVC roadmap" page (`216825860`, Mar 31 2026) is empty.** Body content is the single sentence *"Short roadmap overview"* and nothing else. The substantive roadmap is `90767783` (*The AIVC model training roadmap*, Mar 17 2026), which is detailed but predates our v1.1 work. Worth nudging Wout to either fill in the empty page or delete it so there's one source of truth.

**4. No Jira ticket currently exists for any of: (a) deprecating `aivc-process-10x-atac-rna`, (b) integrating `nf-cellranger-multi` output with `stage_data.sh`, (c) porting `compute_chromvar_scores()`.** These are the three real follow-ups implied by this review.

---

## Final Recommendation Matrix

| Repo | Verdict | Action | Owner | Urgency |
|---|---|---|---|---|
| `nf-cellranger-multi` | KEEP | Lock data contract; add bucket-prefix sync to `stage_data.sh`; write validator | AIVC team (consumer side) | Medium — before QuRIE-seq data lands end-May |
| `nf-star-align` | ARCHIVE | Remove from AIVC's reference set; leave in Wout's control | N/A | Low |
| `bioinf-containers` | ARCHIVE | Remove from AIVC's reference set | N/A | Low |
| `aivc-process-10x-atac-rna` | DELETE-after-parity | Run parity test vs Python pipeline; delete on green | AIVC team | Medium — 4-6 weeks post parity test |

---

## Key Risks

1. **QuRIE-seq data blocker.** If `nf-cellranger-multi` output prefix is not wired into `stage_data.sh` before Wout's wet-lab run lands (end-May / mid-June per roadmap `90767783`), AIVC cannot train on real QuRIE-seq data on time.
2. **Silent schema drift.** `aivc/preprocessing/atac_rna_pipeline.py` hard-codes file names (`filtered_feature_bc_matrix.h5`, `atac_fragments.tsv.gz`). If 10x Genomics or Wout changes the CellRanger Multi output layout, preprocessing silently breaks. A validator is a one-day task; not writing one is the risk.
3. **Parity debt on the R pipeline.** Deleting `aivc-process-10x-atac-rna` before the parity delta report is signed off means we lose the only reference we have for verifying the Python rewrite is numerically correct.
4. **`compute_chromvar_scores()` stub.** Still a stub. Unrelated to which repos we archive, but it blocks the AIVC preprocessing layer from producing complete outputs, so it needs its own ticket. Risk is strictly owned by AIVC, not by the reference repos.

## Recommended Next Step

Open 3 Jira tickets in QUGO (or sub-tasks under the existing AIVC preprocessing epic, whichever you prefer):

1. **QUGO — "Lock CellRanger output contract for AIVC"** — Spec the expected `s3://quriegen-aivc-data/cellranger/<run_id>/` prefix with Wout, write `aivc/preprocessing/validate_cellranger_output.py`, extend `scripts/stage_data.sh` with `--cellranger-run`.
2. **QUGO — "Parity test Python preprocessing vs R pipeline"** — Small 10x Multiome reference (~500 cells), run both, one-page delta report, then delete `aivc-process-10x-atac-rna` from the reference set.
3. **QUGO — "Port chromVAR to Python"** — Replace `compute_chromvar_scores()` stub; not blocked on any reference repo.

Nothing else on this review list blocks the MareNostrum5 v1.1 sweep — which is still blocked on `sinfo -s` / partition / account info from you, not on reference repo decisions.

## What I Need From You

1. Confirm you're happy with **KEEP / ARCHIVE / DELETE-after-parity** as the three categories, or tell me the intended third category from your original message so I can re-classify.
2. Confirm the target S3 prefix for CellRanger outputs (I'm proposing `s3://quriegen-aivc-data/cellranger/<run_id>/`) — or tell me Wout wants a different bucket / prefix. If you'd like, I can draft a short contract doc and you forward it to him.
3. `sinfo -s`, `module avail apptainer`, and `quota -s` output from the MN5 login node once your EuroHPC account activates, so I can un-comment the `#SBATCH --partition=` / `--account=` lines in `scripts/train_v11.sbatch` and we can actually launch the sweep.
