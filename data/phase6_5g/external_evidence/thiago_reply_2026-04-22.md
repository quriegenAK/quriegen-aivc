# External evidence — Thiago (wet-lab) reply, 2026-04-22

Pre-search outcome shared via DM before Thursday 2026-04-23 call:

- K562 multiome confirmed absent from wet-lab's private catalog + 10x public demos.
- PBMC-adjacent cell lines (THP1, BJAB, Jurkat) also absent at useful scale.
- 4 PBMC/BMMC candidate datasets surfaced as alternatives:
  - GSE158013 (TEA-seq, PBMC trimodal)
  - GSE194028 (10x multiome benchmarking, PBMC bimodal)
  - GSE194122 (NeurIPS 2021 Open Problems, BMMC — integration benchmark)
  - GSE156478 (Mimitou 2021 superseries — contains DOGMA-seq trimodal RNA+ATAC+Protein,
    210-antibody TotalSeq-A, with DIG + LLL permeabilization arms)

Thursday 2026-04-23 call outcome:
- Working corpus locked (provisional, pending n_cells): DOGMA-seq (GSE156478)
- Protein encoder input dim: P1 = 210-D TotalSeq-A (locked)
- DIG + LLL handled as batch covariates with pre-registered >5% divergence tripwire
- K562 as pretrain target: abandoned
