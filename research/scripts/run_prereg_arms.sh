#!/bin/bash
# Launch the pre-registered arms (research/prereg_style_switch.md, research/prereg_mw.md) locally, two arms at a time.
# Paired arms run concurrently so they share machine load. Each arm has its own OURBOT_WORK (no carried state reuse).
set -u
WT=/Users/biofanat/Documents/classes/f26/cs535/final-integration
PY=~/sandbox/bot-venv/bin/python
B="$WT/engine/showdown/batch.py"
LOG=~/sandbox/ourbot/prereg_logs
mkdir -p "$LOG"
COMMON=(--k 4 --search-time-ms 100 --streams 10 --workers 10)

arm() {  # arm <tag> <workdir> <a> <b> <seeds> [ENV=VAL ...]
  local tag=$1 work=$2 a=$3 b=$4 seeds=$5; shift 5
  mkdir -p "$work"
  if ls "$work"/hedge_state/"$tag"-* >/dev/null 2>&1 && [ ! -f "$work/.started_$tag" ]; then
    echo "refusing: stale hedge_state for $tag in $work"; return 1
  fi
  touch "$work/.started_$tag"
  env OURBOT_WORK="$work" "$@" "$PY" "$B" --tag "$tag" --a "$a" --b "$b" --seeds "$seeds" "${COMMON[@]}" \
    > "$LOG/$tag.log" 2>&1
  echo "$tag EXIT $?" >> "$LOG/$tag.log"
}

SW=~/sandbox/ourbot/sw
# round 1: MW vs its matched control (same seeds, same load)
arm MWs  ~/sandbox/ourbot/mw_MWs  hedge_bayes_mw stock 1-200 &
arm BHs  ~/sandbox/ourbot/mw_BHs  hedge_bayes    stock 1-200 &
wait
# round 2: switching arms
arm SW2  $SW hedge_bayes switch:style_safe@2:style_uniform 1-100 &
arm SW4  $SW hedge_bayes switch:style_safe@4:style_uniform 1-180 &
wait
# round 3
arm SW4p ~/sandbox/ourbot/sw_p hedge_bayes switch:style_safe@4:style_uniform 1-80 OURBOT_BAYES_LAMBDA=0.98 &
arm MWeq ~/sandbox/ourbot/mw_MWeq hedge_bayes_mw style_eq 1-100 &
wait
echo ALL DONE
