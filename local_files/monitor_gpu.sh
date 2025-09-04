#!/usr/bin/env bash
out="gpu_memory_log.csv"
echo "timestamp,gpu_index,memory_used_MiB,memory_total_MiB" > "$out"

while true; do
  ts=$(date +'%Y-%m-%d %H:%M:%S')
  nvidia-smi \
    --query-gpu=index,memory.used,memory.total \
    --format=csv,noheader,nounits \
  | while IFS=',' read -r idx used total; do
      echo "$ts,$idx,$used,$total" >> "$out"
    done
  sleep 1
done
