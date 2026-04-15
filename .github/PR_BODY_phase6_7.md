# Phase 6.7: harmonize_peaks — real-data peak set artifact (code only)

> ## ⚠ CODE ONLY — no execution artifacts in this PR
> This PR ships the code needed to produce the peak-set artifact on a
> MACS2-equipped host. It intentionally contains **19 `<PENDING:
> execution on MACS2-equipped host>` markers** (3 in the download
> script, 6 in `REAL_DATA_BLOCKERS.md` — 5 in "Artifacts produced" +
> 1 pattern-reference inside the Phase 6.7-exec Step 6 instruction,
> 10 in this PR body). Every marker is intentional — they are **not**
> oversights. Execution and marker-fill happens on a separate host
> per the Phase 6.7-exec checklist in `REAL_DATA_BLOCKERS.md`, across
> three follow-up commits. Do not treat a PENDING marker as a gap to
> fix in this PR.

## What
- Promoted `scripts/harmonize_peaks.py` from stub to executable
  MACS2-on-pooled-fragments pipeline (`--fragments`, `--output`,
  `--genome {hg38,mm10}`, `--min_cells`, `--force`). Idempotent:
  refuses to overwrite without `--force`; writes a sibling
  `.meta.json` with input/output SHA-256 and run metadata.
- Added `scripts/download_pbmc10k_multiome.sh` for the 10x PBMC
  Multiome 10k demo, with per-file SHA-256 verification. Expected
  hash constants are `<PENDING>` until first verified run.
- Added `scripts/INSTALL_MACS2.md` documenting the pip/conda install
  paths. No Python-only peak caller fallback is provided by design.
- Established `data/peak_sets/` with a committed `README.md`
  documenting schema, naming convention, and regeneration steps.
- Updated `.gitignore` to exclude `data/raw/` and
  `data/peak_sets/*.tsv`/`*.meta.json` while keeping the README.

## Why
Unblocks Phase 6.7b (real-data pretrain rerun) and Phase 6.5
(linear probe against real checkpoint). All Phase 6 numbers to
date are mock-data only; this PR is the code foundation for the
first real-data signal. Execution is intentionally deferred — see
"Execution outstanding" below.

## Scope

This file tracks both the original code-only PR (#8, merged
2026-04-15) and the subsequent execution on a MACS2-equipped
host. The peak-set artifact itself is NOT committed
(gitignored by design); its SHA-256 and metadata are the
reproducibility contract.

## Execution complete

Executed on local MacBook Pro with MACS2 2.2.9.1 installed:

- `data/peak_sets/pbmc10k_hg38_20260415.tsv` path: `data/peak_sets/pbmc10k_hg38_20260415.tsv`
- Peak count: 115720
- Output SHA-256: `57b66a257e85287cd6829e38b35bd82aea795d4e96c2b074566981b298a9ef0c`
- Input fragments SHA-256: `5075e32a0e9c6dded35b060bf90d6144375b150e131ffb0be121a93e3b5e1e38`
- MACS2 version: 2.2.9.1
- Exact command line: `python scripts/harmonize_peaks.py --fragments data/raw/pbmc10k_multiome/atac_fragments.tsv.gz --output data/peak_sets/pbmc10k_hg38_20260415.tsv --genome hg38`

Checklist the exec host must satisfy before the follow-up commit:
1. `macs2 --version` succeeds.
2. `bash scripts/download_pbmc10k_multiome.sh` exits 0 after
   populating the `EXPECTED_*_SHA256` constants.
3. `python scripts/harmonize_peaks.py ... --genome hg38` exits 0
   with ≥50,000 peaks in the output TSV.
4. Re-running with `--force` on the same fragments file produces a
   byte-identical output TSV (SHA-256 match across runs).
5. `input_sha256=`, `output_sha256=`, `n_peaks=` lines from stdout
   are copied verbatim into `.github/REAL_DATA_BLOCKERS.md`
   "Artifacts produced" and into the bullets above.

## Invariants preserved
- No model, training, encoder, loss, or test file modified.
- `train_week3.py` untouched.
- `aivc/data/multiome_loader.py` untouched; `peak_set_path` contract
  is unchanged.
- All existing tests still pass (static-only changes for Phase 6.7).

## Reproducibility
- Input fragments SHA-256: `5075e32a0e9c6dded35b060bf90d6144375b150e131ffb0be121a93e3b5e1e38`
- Output peak_set SHA-256: `57b66a257e85287cd6829e38b35bd82aea795d4e96c2b074566981b298a9ef0c`
- MACS2 version: 2.2.9.1
- Command line: `python scripts/harmonize_peaks.py --fragments data/raw/pbmc10k_multiome/atac_fragments.tsv.gz --output data/peak_sets/pbmc10k_hg38_20260415.tsv --genome hg38`
- Produced UTC: 2026-04-15T21:29:36Z

Note: none of these markers is to be replaced locally on a host
that re-derives the values independently. They must be copied from
the stdout of the authoritative exec-host run that also updates
`REAL_DATA_BLOCKERS.md`.

## Phase 6.7b tripwires
- `scripts/pretrain_multiome.py` must consume the peak_set via
  `MultiomeLoader`'s `peak_set_path` — no other load path.
- Resulting checkpoint SHA-256 MUST differ from the mock-data hash
  `10665544bd888dbeddd1c3fdd4b880515e7bb05c16f9a235b453c88b50c5a24b`
  recorded in `REAL_DATA_BLOCKERS.md`. Asserting this difference is
  a Phase 6.7b validation gate.

## Not in scope
- Pretraining rerun (Phase 6.7b).
- Linear probe rerun (Phase 6.5).
- Any model code changes.
- Producing or committing the peak-set artifact itself (separate
  follow-up commit on an exec host; see "Execution outstanding").

## Known issues
- **Test pollution in `tests/test_ckpt_loader.py`** —
  quriegenAK/quriegen-aivc#7. The file fails when run as part of a
  full-suite pytest invocation but passes in isolation. Reproduces
  on `main` at 0918c36 without this PR's changes, so it is
  pre-existing and explicitly out of scope for Phase 6.7 (which does
  not touch any model/training/test file). Filed as a separate issue
  to be tracked, not absorbed into this PR's summary.
