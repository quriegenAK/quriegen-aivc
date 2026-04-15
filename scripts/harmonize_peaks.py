"""
harmonize_peaks.py — produce a unified peak set from pooled ATAC fragments.

Phase 6.7: promoted from Phase-4 stub to an executable MACS2-on-pooled-
fragments pipeline. The produced TSV is consumed by
`aivc.data.multiome_loader.MultiomeLoader` via its `peak_set_path`
parameter.

PIPELINE
--------
1. Validate the 10x fragments.tsv.gz input (schema: chrom, start, end,
   barcode, count; comment lines starting with '#' are skipped).
2. Pool fragments across all barcodes into a single BED (chrom,start,end).
3. Run MACS2 callpeak with ATAC-seq-appropriate parameters:

     macs2 callpeak --treatment <pooled.bed> \
       --format BED --gsize {hs|mm} \
       --nomodel --shift -75 --extsize 150 \
       -q 0.01 --keep-dup all \
       --outdir <tmp> --name harmonized

4. Compute per-peak cell coverage by intersecting the narrowPeak output
   with the original per-barcode fragments (pure-Python sweep). Drop
   peaks with coverage < --min_cells.
5. Write the surviving peaks to a tab-separated TSV with columns:

     peak_id, chrom, start, end, n_cells_covered

   Rows are sorted by (chrom, start). `peak_id` is assigned *after*
   sorting so the ID order matches the on-disk order — downstream
   MultiomeLoader indexing assumes this.

REPRODUCIBILITY
---------------
The script writes a sibling `<output>.meta.json` recording:
  - input_fragments_sha256
  - output_peak_set_sha256
  - macs2_version
  - command_line
  - genome, min_cells, timestamp_utc

It also prints both hashes to stdout. Re-running with --force on the
same input MUST produce a byte-identical output (explicit sort before
write; tempdir contents are not part of the output).

SAFETY
------
- Refuses to overwrite an existing --output unless --force is passed.
- Refuses to run against synthetic/mock inputs: the fragments file
  must exist on disk and parse as real 10x fragments.tsv.gz.
- No fallback peak caller. If MACS2 is missing, the script exits
  nonzero and points at scripts/INSTALL_MACS2.md.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path


GENOME_TO_GSIZE = {"hg38": "hs", "mm10": "mm"}


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _require_macs2() -> str:
    """Return MACS2 version string; raise if not on PATH."""
    exe = shutil.which("macs2")
    if exe is None:
        print(
            "ERROR: macs2 not found on PATH. See scripts/INSTALL_MACS2.md "
            "for installation instructions. This script does NOT fall back "
            "to a Python-only peak caller.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    try:
        result = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: macs2 --version failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    return (result.stdout + result.stderr).strip()


def _open_fragments(path: Path):
    """Open a 10x fragments file (.tsv.gz or .tsv) for text reading."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def _validate_and_pool(fragments: Path, pooled_bed: Path) -> int:
    """Write pooled BED (chrom, start, end) and validate 10x schema.

    Returns the number of pooled fragment records written.
    """
    n = 0
    with _open_fragments(fragments) as fin, open(pooled_bed, "wt") as fout:
        for lineno, line in enumerate(fin, 1):
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                raise ValueError(
                    f"{fragments}:{lineno}: expected 5 tab-separated columns "
                    f"(chrom, start, end, barcode, count); got {len(parts)}."
                )
            chrom, start, end = parts[0], parts[1], parts[2]
            # Basic type check on start/end.
            try:
                int(start)
                int(end)
            except ValueError as exc:
                raise ValueError(
                    f"{fragments}:{lineno}: start/end must be integers."
                ) from exc
            fout.write(f"{chrom}\t{start}\t{end}\n")
            n += 1
    if n == 0:
        raise ValueError(f"{fragments}: no fragment records found.")
    return n


def _run_macs2(pooled_bed: Path, gsize: str, outdir: Path) -> Path:
    """Run MACS2 callpeak; return path to narrowPeak output."""
    cmd = [
        "macs2", "callpeak",
        "--treatment", str(pooled_bed),
        "--format", "BED",
        "--gsize", gsize,
        "--nomodel",
        "--shift", "-75",
        "--extsize", "150",
        "-q", "0.01",
        "--keep-dup", "all",
        "--outdir", str(outdir),
        "--name", "harmonized",
    ]
    print(f"[harmonize_peaks] running: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)
    narrowpeak = outdir / "harmonized_peaks.narrowPeak"
    if not narrowpeak.exists():
        raise RuntimeError(f"MACS2 produced no narrowPeak file at {narrowpeak}")
    return narrowpeak


def _load_peaks(narrowpeak: Path) -> list[tuple[str, int, int]]:
    """Load (chrom, start, end) from a MACS2 narrowPeak file."""
    peaks: list[tuple[str, int, int]] = []
    with open(narrowpeak, "rt") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            peaks.append((parts[0], int(parts[1]), int(parts[2])))
    return peaks


def _count_cells_per_peak(
    fragments: Path,
    peaks: list[tuple[str, int, int]],
) -> list[int]:
    """For each peak, count distinct barcodes with >=1 overlapping fragment.

    Uses a per-chromosome sorted-sweep: O((F+P) log P) per chrom.
    """
    # Group peaks by chrom, sorted by start; remember original index.
    by_chrom: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for idx, (chrom, start, end) in enumerate(peaks):
        by_chrom[chrom].append((start, end, idx))
    for chrom in by_chrom:
        by_chrom[chrom].sort()

    # For each peak idx, accumulate a set of barcodes.
    cells_per_peak: list[set[str]] = [set() for _ in peaks]

    with _open_fragments(fragments) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            chrom = parts[0]
            plist = by_chrom.get(chrom)
            if not plist:
                continue
            fstart = int(parts[1])
            fend = int(parts[2])
            barcode = parts[3]
            # Linear scan candidate peaks where start < fend.
            # Binary search for the first peak with start >= fend; everything
            # before that is a candidate (peak.start < fend). We then filter
            # by peak.end > fstart.
            lo, hi = 0, len(plist)
            while lo < hi:
                mid = (lo + hi) // 2
                if plist[mid][0] < fend:
                    lo = mid + 1
                else:
                    hi = mid
            # plist[:lo] have start < fend; check end > fstart.
            for i in range(lo - 1, -1, -1):
                pstart, pend, pidx = plist[i]
                if pend <= fstart:
                    # Because peaks are sorted by start (not end), we cannot
                    # break early safely in general — but MACS2 peaks do not
                    # overlap, so once a peak's end is <= fstart, all earlier
                    # peaks also end <= fstart.
                    break
                cells_per_peak[pidx].add(barcode)

    return [len(s) for s in cells_per_peak]


def _write_peak_set(
    output: Path,
    peaks: list[tuple[str, int, int]],
    cell_counts: list[int],
    min_cells: int,
) -> int:
    """Write filtered, sorted peak-set TSV. Returns row count."""
    rows: list[tuple[str, int, int, int]] = []
    for (chrom, start, end), n in zip(peaks, cell_counts):
        if n >= min_cells:
            rows.append((chrom, start, end, n))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))

    seen_ids: set[str] = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wt") as fh:
        fh.write("peak_id\tchrom\tstart\tend\tn_cells_covered\n")
        for i, (chrom, start, end, n) in enumerate(rows, 1):
            peak_id = f"peak_{i:07d}"
            if peak_id in seen_ids:
                raise RuntimeError(f"duplicate peak_id {peak_id} (bug)")
            seen_ids.add(peak_id)
            fh.write(f"{peak_id}\t{chrom}\t{start}\t{end}\t{n}\n")
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Produce a peak-set TSV from pooled ATAC fragments via MACS2.",
    )
    parser.add_argument("--fragments", type=Path, required=True,
                        help="10x fragments.tsv.gz file.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output peak-set TSV path.")
    parser.add_argument("--genome", choices=sorted(GENOME_TO_GSIZE.keys()),
                        required=True, help="Reference genome.")
    parser.add_argument("--min_cells", type=int, default=10,
                        help="Drop peaks covered by fewer than this many "
                             "distinct barcodes (default: 10).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite --output if it exists.")
    args = parser.parse_args(argv)

    fragments: Path = args.fragments
    output: Path = args.output

    if not fragments.exists():
        print(f"ERROR: fragments file not found: {fragments}", file=sys.stderr)
        return 2
    if output.exists() and not args.force:
        print(
            f"ERROR: {output} already exists. Re-run with --force to overwrite.",
            file=sys.stderr,
        )
        return 2

    macs2_version = _require_macs2()
    gsize = GENOME_TO_GSIZE[args.genome]

    input_sha = _sha256_file(fragments)
    print(f"[harmonize_peaks] input sha256: {input_sha}", file=sys.stderr)

    with tempfile.TemporaryDirectory(prefix="harmonize_peaks_") as tmp:
        tmpdir = Path(tmp)
        pooled_bed = tmpdir / "pooled.bed"
        print(f"[harmonize_peaks] pooling fragments -> {pooled_bed}",
              file=sys.stderr)
        n_frags = _validate_and_pool(fragments, pooled_bed)
        print(f"[harmonize_peaks] pooled {n_frags} fragment records",
              file=sys.stderr)

        macs_out = tmpdir / "macs2"
        macs_out.mkdir()
        narrowpeak = _run_macs2(pooled_bed, gsize, macs_out)
        peaks = _load_peaks(narrowpeak)
        print(f"[harmonize_peaks] MACS2 returned {len(peaks)} raw peaks",
              file=sys.stderr)

        cell_counts = _count_cells_per_peak(fragments, peaks)
        n_rows = _write_peak_set(output, peaks, cell_counts, args.min_cells)

    output_sha = _sha256_file(output)
    print(f"[harmonize_peaks] wrote {n_rows} peaks -> {output}",
          file=sys.stderr)
    print(f"[harmonize_peaks] output sha256: {output_sha}", file=sys.stderr)

    meta = {
        "input_fragments": str(fragments),
        "input_fragments_sha256": input_sha,
        "output_peak_set": str(output),
        "output_peak_set_sha256": output_sha,
        "n_peaks_written": n_rows,
        "macs2_version": macs2_version,
        "genome": args.genome,
        "min_cells": args.min_cells,
        "command_line": ["python", "scripts/harmonize_peaks.py", *(argv or sys.argv[1:])],
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    with open(meta_path, "wt") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"[harmonize_peaks] wrote meta -> {meta_path}", file=sys.stderr)

    # Also echo the key hashes on stdout for easy capture by CI wrappers.
    print(f"input_sha256={input_sha}")
    print(f"output_sha256={output_sha}")
    print(f"n_peaks={n_rows}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
