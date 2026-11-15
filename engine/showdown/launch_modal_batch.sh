#!/usr/bin/env bash
# Launch the real-Showdown experiment batch on Modal (attached, one process per run, budget-capped per run).
# Usage: cd engine && bash showdown/launch_modal_batch.sh      (logs in modal_logs/)
set -u
cd "$(dirname "$0")/.."
L=modal_logs; mkdir -p "$L"   # outside showdown/ (uploaded into the image)
run() {  # tag a b seeds max_usd
  .venv/bin/modal run showdown/modal_showdown.py::main --streams 10 --cpu-s-per-game 61 --slowdown 1 --util 1 \
    --tag "$1" --a "$2" --b "$3" --seeds "$4" --max-usd "$5" > "$L/$1.log" 2>&1
  echo "EXIT $?" >> "$L/$1.log"
}
run E2   hedge       stock      1-400 2.0 &
run E2fs hedge_fs    stock      1-400 2.0 &
run AF   agentform   stock      1-300 1.6 &
run E4u  bandit_ucb1 rmplus_avg 1-300 0.8 &
run E4e  bandit_exp3 rmplus_avg 1-300 0.8 &
run E5   hedge       agentform  1-300 1.0 &
wait
echo "ALL DONE"
