#!/usr/bin/env bash
# Full pipeline in the background (nohup; survives ssh disconnects), logged to logs/full_<timestamp>.log.
# Resumable: re-running skips every stage whose artifacts/full/<stage>/_DONE.json exists.
#   bash scripts/run_full.sh                         # everything
#   bash scripts/run_full.sh --stage xenc --force    # extra args go to run_all.py
#   FOREGROUND=1 bash scripts/run_full.sh            # stay attached (e.g. inside tmux)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export HF_HOME="${HF_HOME:-$ROOT/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # models were downloaded into hf_cache/ by start_remote.sh; no hub calls during the run
export PYTHONUTF8=1
# pin to one GPU (index as shown by nvidia-smi; default 0 = the A100)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/full_${TS}.log"
CMD=(.venv/bin/python code/business_entity_resolution/src/run_all.py --profile full "$@")
echo "command: ${CMD[*]}"
echo "log:     $LOG"
if [[ "${FOREGROUND:-0}" == "1" ]]; then
  "${CMD[@]}" 2>&1 | tee "$LOG"
else
  nohup bash -c "$(printf '%q ' "${CMD[@]}") 2>&1 | tee '$LOG'" > /dev/null 2>&1 &
  echo "started in background (pid $!). Follow with:  tail -f $LOG"
fi
