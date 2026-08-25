#!/bin/bash
# struct SaProt arm: 3 seeds, sequential (cron-launched 2026-08-11)
cd /root/autodl-tmp/workspace
for s in 0 1 2; do
  echo "=== seed $s start $(date -Is) ==="
  /root/miniconda3/bin/python training/train.py     --config training/configs/struct_saprot.json     --override "{\"seed\":$s,\"output_dir\":\"training/runs/struct_saprot_s$s\"}"
  echo "=== seed $s end $(date -Is) rc=$? ==="
done
echo "ALL SEEDS DONE"
