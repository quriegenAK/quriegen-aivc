# data/peak_sets/

Canonical location for harmonized ATAC peak-set artifacts produced by
`scripts/harmonize_peaks.py`. These artifacts are consumed downstream
by `aivc.data.multiome_loader.MultiomeLoader` via its `peak_set_path`
argument.

## Schema

Each peak-set is a tab-separated file with a header row and exactly
five columns:

| column            | type   | description                                     |
|-------------------|--------|-------------------------------------------------|
| `peak_id`         | string | stable identifier, format `peak_{:07d}`         |
| `chrom`           | string | reference chromosome (e.g. `chr1`)              |
| `start`           | int    | 0-based half-open BED start                     |
| `end`             | int    | 0-based half-open BED end                       |
| `n_cells_covered` | int    | distinct barcodes with ≥1 overlapping fragment  |

Rows are sorted by `(chrom, start, end)`. `peak_id` is assigned **after**
sorting so that the on-disk row order matches the ID order — downstream
loaders rely on this.

A sibling `<peak_set>.meta.json` records the input fragments SHA-256,
output SHA-256, MACS2 version, genome, min_cells, command line, and UTC
timestamp. Treat the `.meta.json` as the canonical reproducibility
record for the `.tsv` next to it.

## Naming convention

```
{dataset}_{genome}_{YYYYMMDD}.tsv
```

Examples:
- `pbmc10k_hg38_20260415.tsv`
- `pbmc10k_hg38_20260415.tsv.meta.json`

## Git policy

- `data/peak_sets/README.md` **is committed**.
- `data/peak_sets/*.tsv` and `*.meta.json` **are NOT committed** (see
  `.gitignore`). The authoritative reference is the SHA-256 recorded in
  `.github/REAL_DATA_BLOCKERS.md` under "Artifacts produced".
- To reproduce an artifact, use the `command_line` field of its
  `.meta.json` against the fragments file with the matching
  `input_fragments_sha256`.

## Regeneration

```bash
# 1. Stage raw fragments (idempotent; verifies SHA-256):
bash scripts/download_pbmc10k_multiome.sh

# 2. Produce the peak set:
python scripts/harmonize_peaks.py \
    --fragments data/raw/pbmc10k_multiome/atac_fragments.tsv.gz \
    --output    data/peak_sets/pbmc10k_hg38_20260415.tsv \
    --genome    hg38 \
    --min_cells 10

# 3. The script prints `input_sha256=...` / `output_sha256=...` /
#    `n_peaks=...` on stdout and writes a sibling .meta.json. Those
#    three values get copied into .github/REAL_DATA_BLOCKERS.md and
#    the Phase 6.7 PR body.
```

MACS2 must be on `PATH`; see `scripts/INSTALL_MACS2.md`.
