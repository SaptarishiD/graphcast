#!/usr/bin/env bash
out="gpu_memory_log.csv"
echo "timestamp,gpu_index,memory_used_MiB,memory_total_MiB" > "$out"

while true; do
  ts=$(date +'%Y-%m-%d %H:%M:%S')

  # Capture GPU memory usage into an array
  mapfile -t gpu_data < <(nvidia-smi \
    --query-gpu=index,memory.used,memory.total \
    --format=csv,noheader,nounits)

  all_below=true
  for line in "${gpu_data[@]}"; do
    IFS=',' read -r idx used total <<< "$line"
    echo "$ts,$idx,$used,$total" >> "$out"

    if (( used >= 50 )); then
      all_below=false
    fi
  done

  # Stop if all GPUs are below 100 MiB
  if $all_below; then
    echo "All GPUs below 100 MiB. Stopping."
    break
  fi

  sleep 1
done
