"""Prepare Calderon 2019 (GSE118189) bulk ATAC counts as AnnData.

External evaluation corpus for AIVC Phase 6.5g.2 representation transfer.
RAW PEAKS variant (consistent with DOGMA assembly post-PR #30 amendment;
chromVAR motif deviations path was abandoned).

GEO supplementary: GSE118189_ATAC_counts.txt.gz (~111 MB, single file).
Format: tab-separated. Header has N sample IDs only (no leading
peak-column label, per GEO supplementary convention). Data rows have
peak coordinate followed by N count values, so data row width = N+1.
Sample IDs encode {donor}-{cell_type}-{stim_state} (e.g. "1001-CD4_Teff-S"
for stimulated, "1002-Bulk_B-U" for unstimulated). Peak coordinates use
underscore separators (chr1_10045_10517) in real GEO data; the legacy
colon-dash form (chr1:10045-10517) is also accepted for forward-compat.

Usage:
    python scripts/prepare_calderon_2019.py \\
        --counts data/calderon2019/GSE118189_ATAC_counts.txt.gz \\
        --out data/calderon2019/calderon_atac.h5ad

Outputs an AnnData with:
    - .X = (n_samples, n_peaks) raw counts (CSR sparse, int32)
    - .obs = donor, cell_type, stim_state, sample_id
    - .var = chrom, start, end, peak_id (raw string from input)
    - .uns = {'source': 'GSE118189', 'assay': 'bulk_ATAC', 'variant': 'raw_peaks'}
"""
from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


# --- Sample ID parser ---------------------------------------------------------

SAMPLE_RE = re.compile(
    r"^(?P<donor>\d+)-(?P<cell_type>[A-Za-z0-9_]+?)-(?P<stim>[SU])$"
)


def parse_sample_ids_to_metadata(sample_ids: list[str]) -> pd.DataFrame:
    rows = []
    for sid in sample_ids:
        m = SAMPLE_RE.match(sid)
        if m is None:
            rows.append({
                "sample_id": sid,
                "donor": "unknown",
                "cell_type": "unknown",
                "stim_state": "unknown",
            })
            continue
        rows.append({
            "sample_id": sid,
            "donor": m.group("donor"),
            "cell_type": m.group("cell_type"),
            "stim_state": "stim" if m.group("stim") == "S" else "unstim",
        })
    df = pd.DataFrame(rows).set_index("sample_id")
    return df


# --- Counts file parser -------------------------------------------------------

# PR #37 fix: accept both `_` (real Calderon) and `:` (legacy/synthetic) chrom-pos
# separators, and both `_` and `-` (legacy) start-end separators. PEAK_RE remains
# strict about chr-prefix + integer coordinates.
PEAK_RE = re.compile(r"^(?P<chrom>chr[\w]+)[_:](?P<start>\d+)[_-](?P<end>\d+)$")


def _open_counts(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def parse_calderon_counts(counts_path: Path) -> tuple[sp.csr_matrix, pd.DataFrame, list[str]]:
    """Parse GSE118189_ATAC_counts.txt[.gz].

    Real format: header is N sample IDs (tab-separated, no leading label).
    Data rows are peak_id (col 0) + N count values (cols 1..N).
    """
    counts_path = Path(counts_path)
    if not counts_path.exists():
        raise FileNotFoundError(f"Calderon counts file not found: {counts_path}")

    with _open_counts(counts_path) as fh:
        # PR #37 fix: header has NO leading peak-column label per GEO convention.
        # Previous behavior of `header[1:]` stripped the first sample ID.
        header = fh.readline().rstrip("\n").split("\t")
        sample_ids = header
        if len(sample_ids) == 0:
            raise ValueError("Calderon header empty")

        peak_ids: list[str] = []
        chroms: list[str] = []
        starts: list[int] = []
        ends: list[int] = []
        rows: list[int] = []
        cols: list[int] = []
        vals: list[int] = []

        n_samples = len(sample_ids)
        for row_idx, line in enumerate(fh):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != n_samples + 1:
                raise ValueError(
                    f"Row {row_idx} has {len(parts)} cols, expected "
                    f"{n_samples + 1} (peak_id + {n_samples} samples): "
                    f"{parts[:3]!r}..."
                )
            peak_id = parts[0]
            m = PEAK_RE.match(peak_id)
            if m is None:
                raise ValueError(f"Row {row_idx}: cannot parse peak {peak_id!r}")
            peak_ids.append(peak_id)
            chroms.append(m.group("chrom"))
            starts.append(int(m.group("start")))
            ends.append(int(m.group("end")))

            for col_idx, v_str in enumerate(parts[1:]):
                if v_str == "0":
                    continue
                v = int(v_str)
                if v == 0:
                    continue
                rows.append(row_idx)
                cols.append(col_idx)
                vals.append(v)

    n_peaks = len(peak_ids)
    coo = sp.coo_matrix(
        (np.asarray(vals, dtype=np.int32),
         (np.asarray(rows, dtype=np.int32),
          np.asarray(cols, dtype=np.int32))),
        shape=(n_peaks, n_samples),
    )
    X = coo.T.tocsr()

    var_df = pd.DataFrame(
        {"chrom": chroms, "start": starts, "end": ends},
        index=pd.Index(peak_ids, name="peak_id"),
    )
    return X, var_df, sample_ids


# --- Builder ------------------------------------------------------------------

def build_anndata(counts_path: Path) -> ad.AnnData:
    X, var_df, sample_ids = parse_calderon_counts(counts_path)
    obs_df = parse_sample_ids_to_metadata(sample_ids)
    adata = ad.AnnData(
        X=X,
        obs=obs_df,
        var=var_df,
        uns={
            "source": "GSE118189",
            "assay": "bulk_ATAC",
            "variant": "raw_peaks",
            "n_samples": int(X.shape[0]),
            "n_peaks": int(X.shape[1]),
        },
    )
    return adata


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--counts", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    adata = build_anndata(args.counts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.out, compression="gzip")
    print(f"Wrote {args.out} -- shape {adata.shape}, nnz {adata.X.nnz}")


if __name__ == "__main__":
    main()
