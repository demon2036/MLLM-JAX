"""MiniOneRec SID SFT (plugins-first).

This package implements a non-invasive, reproducible pipeline for:
- SID token extension from `.index.json`
- Multi-task SFT (SID next-item + SID↔title alignment + optional fusion tasks)
- Constrained decoding evaluation (valid SID outputs)
- HR@K / NDCG@K metric computation
"""

