#!/bin/bash
# struct SaProt gated_pre arm (per-residue tokens fused before adapter_ab):
# 3 seeds, sequential
cd /root/autodl-tmp/workspace
for s in 0 1 2; do
  echo "=== seed $s start $(date -Is) ==="
  /root/miniconda3/bin/python training/train.py \
    --config training/configs/struct_saprot_gated_pre.json \
    --override "{\"seed\":$s,\"output_dir\":\"training/runs/struct_saprot_gated_pre_s$s\"}"
  echo "=== seed $s end $(date -Is) rc=$? ==="
done
echo "ALL SEEDS DONE"
