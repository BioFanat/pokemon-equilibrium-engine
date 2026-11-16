#!/bin/bash
# Replication arms for prereg_mw.md Addendum A (W1-rep, L1-rep): seeds 201-400, run concurrently (shared load).
# Waits for run_prereg_arms.sh to finish so the arms do not compete with it for CPU.
set -u
while pgrep -f run_prereg_arms.sh >/dev/null; do sleep 60; done
WT=/Users/biofanat/Documents/classes/f26/cs535/final-integration
PY=~/sandbox/bot-venv/bin/python
B="$WT/engine/showdown/batch.py"
LOG=~/sandbox/ourbot/prereg_logs
COMMON=(--k 4 --search-time-ms 100 --streams 10 --workers 10)
run() { local tag=$1 work=$2 a=$3; mkdir -p "$work"
  OURBOT_WORK="$work" "$PY" "$B" --tag "$tag" --a "$a" --b stock --seeds 201-400 "${COMMON[@]}" > "$LOG/$tag.log" 2>&1
  echo "$tag EXIT $?" >> "$LOG/$tag.log"; }
run MWs2 ~/sandbox/ourbot/mw_MWs2 hedge_bayes_mw &
run BHs2 ~/sandbox/ourbot/mw_BHs2 hedge_bayes &
wait
echo REPLICATION DONE
