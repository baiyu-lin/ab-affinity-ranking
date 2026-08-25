#!/usr/bin/env python
"""One-click training entry (v3 de-leakage recipe).

Runs the full retraining pipeline on the GPU instance:
  1. data_pipeline/08_deleak_v3.py  -- benchmark de-leakage scan
     (exact VH+VL match of the 40 v3 antibodies + same-cluster removal;
     writes data_pipeline/output/deleak_v3_exclude.json)
  2. training/train.py x 3 seeds    -- v5 recipe (ListMLE, frozen PLM towers,
     ~1.86M trainable params) with data.exclude_json applied

Prerequisites (see training/README.md):
  - data_pipeline 01-07 outputs under data_pipeline/output/
  - embedding caches under extraction/cache_antiberty + cache_esm2
    (extraction caches are reused as-is: de-leakage only removes rows)

Worked example (RTX 4090, ~8 min per seed):
  python code/train.py
  # -> [train.py] exclude_json ...: dropping 101/641454 rows
  # -> fold all__mon100: 579829 train rows, 3037 val rows
  # -> training/runs/deleak_final_s{0,1,2}/best.pt
"""
import json
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
EXCLUDE_JSON = "data_pipeline/output/deleak_v3_exclude.json"


def run(script, *args):
    cmd = [sys.executable, str(CODE_DIR / script), *map(str, args)]
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=CODE_DIR, check=True)


def main():
    run("data_pipeline/08_deleak_v3.py")
    for seed in (0, 1, 2):
        override = json.dumps({
            "seed": seed,
            "output_dir": f"training/runs/deleak_final_s{seed}",
            "data": {"exclude_json": EXCLUDE_JSON},
        })
        run("training/train.py", "--config", "training/configs/final_all.json",
            "--override", override)
    print("train.py: all 3 seeds done -> training/runs/deleak_final_s{0,1,2}/")


if __name__ == "__main__":
    main()
