"""
harmonize_peaks.py — produce a unified peak set from pooled ATAC fragments.

This is a stub scaffolding for Phase 4. It is intentionally NOT wired
into training. When invoked separately, it produces a peak_set artifact
that MultiomeLoader consumes via the `peak_set_path` parameter.

PROCEDURE (MACS2 on pooled fragments)
-------------------------------------
1. Pool per-sample fragment files into a single BED:

     cat sample1.fragments.tsv.gz sample2.fragments.tsv.gz ... \
         | gunzip -c \
         | awk 'BEGIN{OFS="\\t"} {print $1, $2, $3}' \
         > pooled.fragments.bed

2. Call peaks with MACS2 in BEDPE/shifted-BED mode:

     macs2 callpeak \
       --treatment pooled.fragments.bed \
       --format BED \
       --gsize hs \
       --nomodel \
       --shift -75 --extsize 150 \
       --keep-dup all \
       --call-summits \
       --outdir macs2_out \
       --name harmonized

3. Post-filter + merge overlapping peaks:

     bedtools sort -i macs2_out/harmonized_peaks.narrowPeak \
       | bedtools merge -i - -d 0 \
       > peak_set.bed

EXPECTED OUTPUT SCHEMA
----------------------
A tab-separated BED-like artifact (no header) with columns:

    chrom    start    end    peak_id

The peak_id is a stable 1-based identifier; `peak_id` is the column
used by MultiomeLoader's peak-matrix alignment step downstream.

NOTE
----
Running MACS2 end-to-end is explicitly deferred beyond Phase 4. This
script currently fails fast if invoked; it exists to document the
procedure and to serve as the single source of truth for peak-set
generation once the fragment files are staged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fragments",
        nargs="+",
        required=False,
        help="Per-sample fragments.tsv(.gz) files to pool.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=False,
        default=Path("peak_set.bed"),
        help="Output peak_set artifact path.",
    )
    parser.add_argument(
        "--genome-size",
        default="hs",
        help="MACS2 -g/--gsize (default 'hs').",
    )
    args = parser.parse_args(argv)

    print(
        "scripts/harmonize_peaks.py is a Phase-4 stub. "
        "MACS2 is not invoked here. Follow the procedure in the module "
        "docstring to generate the peak_set artifact externally.",
        file=sys.stderr,
    )
    if args.fragments:
        print(f"Would pool {len(args.fragments)} fragment file(s) -> {args.out}",
              file=sys.stderr)
    return 2  # nonzero: stub, not executed end-to-end.


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
