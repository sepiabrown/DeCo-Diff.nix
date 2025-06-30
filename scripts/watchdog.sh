#!/usr/bin/env bash
#
# adaptive_run.sh: retry torchrun on OOM, showing all logs live.

batch_size=546

while true; do
  echo "🔁 Running with --global-batch-size=${batch_size}..."
  # Create a temp logfile
  log="$(mktemp)"
  # Run and tee both stdout/stderr to console and logfile
  torchrun \
    --standalone --nnodes=1 --nproc-per-node=2 \
    "$(which train_deco_diff)" \
    --input-json ~/datasets/PCB/Huang/PCB_DATASET/PCB_SELECTED_GRAY/train___250624.json \
    --global-batch-size="${batch_size}" \
    --num-datafile="${batch_size}" \
    --rep-datafile="${batch_size}"
    2>&1 | tee "$log"
  ret=${PIPESTATUS[0]}

  if (( ret == 0 )); then
    echo "✅ Completed successfully at batch size ${batch_size}."
    rm -f "$log"
    break
  fi

  if grep -qi "CUDA out of memory" "$log"; then
    echo "⚠️  OOM detected. Lowering batch size..."
    batch_size=$(( batch_size - 10 ))
    rm -f "$log"
    if (( batch_size <= 0 )); then
      echo "❌ Batch size would go <= 0; aborting."
      exit 1
    fi
    echo "🔄 Retrying with batch size ${batch_size}..."
  else
    echo "❌ Failed with non-OOM error (exit code $ret). See log:"
    tail -n +1 "$log"
    rm -f "$log"
    exit $ret
  fi
done
