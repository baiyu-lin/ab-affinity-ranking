#!/usr/bin/env python
"""One-click inference demo: score + rank the v3 mAbs test set.

Thin wrapper around training/infer_benchmark.py with the repo defaults
(3-seed deleak_final ensemble, Borda per-group ensemble, 1..N ranks).
Writes code/predictions/mAbs_ranking.csv with per-seed scores/ranks
and the ensemble pred_rank.

Worked example (CPU, ~5-10 min, first run extracts embeddings):
  python code/inference.py
  # -> code/predictions/mAbs_ranking.csv (40 rows, 2 groups x 1..20)

Extra args are forwarded to training/infer_benchmark.py, e.g.:
  python code/inference.py --ckpt-dirs final_all_s0 final_all_s1 final_all_s2
"""
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
OUT = CODE_DIR / "predictions" / "mAbs_ranking.csv"


def main():
    cmd = [sys.executable, str(CODE_DIR / "training" / "infer_benchmark.py"),
           "--out", str(OUT), *sys.argv[1:]]
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=CODE_DIR, check=True)
    print(f"inference.py: wrote {OUT}")


if __name__ == "__main__":
    main()
