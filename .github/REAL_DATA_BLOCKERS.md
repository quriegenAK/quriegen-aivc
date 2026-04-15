# Real-data rerun blockers

Tracked deliverables that MUST be closed before Phase 6.5 can start.
This file replaces `PHASE6_BLOCKERS.md` — the blocker is no longer
Phase-6-specific, it gates every downstream ablation that claims a
biological (rather than methodology) signal.

Phase 5 shipped without it, Phase 6 shipped without it. If Phase 6.5
starts before these items close, the next ablation is also
methodology-only — the same footnote repeats a third time.

## Mock-data checkpoint of record

The current checkpoint at `checkpoints/pretrain/pretrain_encoders.pt`
was produced on the synthetic Multiome fallback in
`scripts/pretrain_multiome.py`.

- **SHA-256:**
  `10665544bd888dbeddd1c3fdd4b880515e7bb05c16f9a235b453c88b50c5a24b`
- **Config:** `n_genes=500`, `hidden_dim=256`, `latent_dim=128`,
  `schema_version=1`.
- **Use:** methodology validation only. The linear-probe numbers in
  `.github/PR_BODY_phase6.md` derive from this checkpoint.

The real-data rerun MUST produce a checkpoint with a SHA-256 different
from the hash above. Same hash = same checkpoint = mock data was
reused → reject.

## Deliverable: harmonize_peaks + real-data pretrain

- **Owner:** Ash Khan
- **Target date:** 2026-04-29 (two weeks from Phase 6 PR merge)
- **Status (2026-04-15):** DONE (2026-04-15) — Phase 6.7 code merged
  and Phase 6.7-exec completed on local MacBook Pro (MACS2 2.2.9.1).
  Peak-set artifact produced at
  `data/peak_sets/pbmc10k_hg38_20260415.tsv` (SHA-256
  `57b66a257e85287cd6829e38b35bd82aea795d4e96c2b074566981b298a9ef0c`,
  115720 peaks); see "Artifacts produced" below.
- **Definition of done** (all three):
  1. Peak-set artifact exists at `data/peak_sets/pbmc10k_hg38_{date}.tsv`
     (see `data/peak_sets/README.md` for naming), produced by
     `scripts/harmonize_peaks.py` on 10x PBMC Multiome fragments.
     Canonical path for Phase 6.7b consumption:
     `data/peak_sets/pbmc10k_hg38_20260415.tsv` (canonical name; actual
     artifact not committed — see "Artifacts produced" section).
  2. One successful `scripts/pretrain_multiome.py` run consuming that
     peak-set artifact (not the mock fallback), with loss monotone on
     moving average and `schema_version=1` validated on save.
  3. The resulting checkpoint is stored at
     `checkpoints/pretrain/pretrain_encoders.pt` with a SHA-256
     **different** from the mock-data hash recorded above, and the
     new hash is appended to the appendix at the bottom of this file.

## Phase 6.7-exec: execution steps outstanding

> **This checklist is run ONCE on a MACS2-equipped host. Each step
> corresponds to a separate commit so the audit trail remains clear.**
> Execute top-to-bottom; each step's outcome is verifiable from the
> commit diff, not from operator memory. Prerequisites: MACS2-equipped
> host with ≥20 GB free in `data/raw/`.

### Step 1 — Install MACS2

Follow `scripts/INSTALL_MACS2.md`. Verify:

```
macs2 --version
```

must print a version string and exit 0. No commit produced by this
step (local-env setup).

### Step 2 — First download attempt (expected to fail)

Run:

```
bash scripts/download_pbmc10k_multiome.sh
```

The verify step **will fail** — the `EXPECTED_*_SHA256` constants are
still `<PENDING:...>` sentinels, and the script refuses to verify
against a sentinel. This failure is intentional: it forces you to
record the observed hashes rather than trusting an un-checked download.
From the error output, copy the three observed SHA-256 values:
`atac_fragments.tsv.gz`, `atac_fragments.tsv.gz.tbi`,
`filtered_feature_bc_matrix.h5`.

### Step 3 — Populate the download-script sentinels

In `scripts/download_pbmc10k_multiome.sh`, replace:

- `EXPECTED_FRAG_SHA256="<PENDING:...>"`
- `EXPECTED_TBI_SHA256="<PENDING:...>"`
- `EXPECTED_H5_SHA256="<PENDING:...>"`

with the three observed values from Step 2.

Commit: **`Phase 6.7-exec: populate download hashes`**

### Step 4 — Re-run the download (must succeed)

```
bash scripts/download_pbmc10k_multiome.sh
```

Must now exit 0 with all three files verified. No commit from this
step (files are gitignored by design).

### Step 5 — Produce the peak-set artifact

```
python scripts/harmonize_peaks.py \
    --fragments data/raw/pbmc10k_multiome/atac_fragments.tsv.gz \
    --output    data/peak_sets/pbmc10k_hg38_20260415.tsv \
    --genome    hg38 \
    --min_cells 10
```

Capture these values from stdout and the sibling `.meta.json`:

- `input_sha256=...`
- `output_sha256=...`
- `n_peaks=...`
- `macs2_version` (from `.meta.json`)
- `timestamp_utc` (from `.meta.json`)

Do **NOT** re-derive these values on a different host — use the
stdout from this execution run verbatim.

No commit from this step (the artifact is gitignored).

### Step 6 — Fill execution artifacts

Replace every `<PENDING: execution on MACS2-equipped host>` marker:

- 5 markers in `.github/REAL_DATA_BLOCKERS.md` "Artifacts produced"
  (below).
- 10 markers in `.github/PR_BODY_phase6_7.md` "Execution
  outstanding" and "Reproducibility" sections.

Commit: **`Phase 6.7-exec: fill execution artifacts`**

### Step 7 — Close the deliverable

In this file, change the "Status" line in the "Deliverable:
harmonize_peaks + real-data pretrain" section from
`IN PROGRESS — ...` to `DONE (YYYY-MM-DD) — ...` with the execution
date.

Commit: **`Phase 6.7-exec: mark harmonize_peaks DONE`**

After Step 7, Phase 6.7b (pretraining rerun) and Phase 6.5 (linear
probe) become unblocked. Both remain OPEN items in this file until
their own deliverables land.

## Artifacts produced

### pbmc10k_hg38_20260415

- **Path**: `data/peak_sets/pbmc10k_hg38_20260415.tsv` (gitignored)
- **Produced (UTC)**: 2026-04-15T21:29:36Z
- **Producer**: Ash Khan, local MacBook Pro
- **Input fragments**: `data/raw/pbmc10k_multiome/atac_fragments.tsv.gz`
- **Input fragments SHA-256**: `5075e32a0e9c6dded35b060bf90d6144375b150e131ffb0be121a93e3b5e1e38`
- **Output peak_set SHA-256**: `57b66a257e85287cd6829e38b35bd82aea795d4e96c2b074566981b298a9ef0c`
- **n_peaks**: 115720
- **MACS2 version**: 2.2.9.1
- **Genome**: hg38
- **min_cells filter**: 10
- **Command**:
  ```
  python scripts/harmonize_peaks.py \
      --fragments data/raw/pbmc10k_multiome/atac_fragments.tsv.gz \
      --output    data/peak_sets/pbmc10k_hg38_20260415.tsv \
      --genome    hg38
  ```

### Command sketch

```
python scripts/harmonize_peaks.py \
    --fragments pbmc_multiome_fragments/*.tsv.gz \
    --out data/peak_sets/pbmc10k_v1.tsv

python scripts/pretrain_multiome.py \
    --multiome_h5ad data/pbmc_multiome.h5ad \
    --peak_set data/peak_sets/pbmc10k_v1.tsv \
    --steps 10000
```

## Real-data linear-probe rerun (activates the Phase 6 gate)

Once the real-data checkpoint exists:

- Run:
  ```
  python scripts/exp1_linear_probe_ablation.py \
      --ckpt_path checkpoints/pretrain/pretrain_encoders.pt \
      --dataset_name norman2019 --seed 17 --wandb
  ```
- Confirm the logged `ckpt_sha256` differs from the mock hash above.
- This rerun — not the numbers in `PR_BODY_phase6.md` — is what
  activates the Phase 6 interpretation gate (≥ +5% relative R² on
  top-50 DE → proceed to Phase 6.5).

## Escalation

If `harmonize_peaks` turns out to require a multi-day investment, this
stops being a PR-closeout item and becomes a named work item on its
own branch. Do not open Phase 6.5 work until the definition of done
above is fully satisfied.

## Appendix: real-data checkpoint hashes (append on rerun)

<!-- On each real-data rerun, append a line: YYYY-MM-DD <sha256> <notes> -->
