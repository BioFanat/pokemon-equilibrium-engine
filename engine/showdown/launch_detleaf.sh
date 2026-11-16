#!/usr/bin/env bash
# Deterministic-leaf control and leaf-exposure measurement (see theory/tightening/verify/leaf/README.txt).
#   AFdet: agentform vs stock on AF's seeds 1-300 with OURBOT_DET_LEAF=1 -> paired difference vs AF (0.314)
#   LSs / LSn: default leaf with OURBOT_LEAF_STATS=1 in a stock (2 CPU) and a non-stock (1 CPU) container class
#   (leaf_stats counters are process-wide: sum one side's records per game; side A only.)
set -u
cd "$(dirname "$0")/.."
L=modal_logs; mkdir -p "$L"
run() {  # tag a b seeds max_usd cpu_s_per_game [extra main flags...]
  local tag=$1 a=$2 b=$3 seeds=$4 usd=$5 cps=$6; shift 6
  .venv/bin/modal run showdown/modal_showdown.py::main --streams 10 --cpu-s-per-game "$cps" --slowdown 1 --util 1 \
    --tag "$tag" --a "$a" --b "$b" --seeds "$seeds" --max-usd "$usd" "$@" > "$L/$tag.log" 2>&1
  echo "EXIT $?" >> "$L/$tag.log"
}
echo "START $(date)"
run AFdet agentform stock 1-300 0.9 36 --det-leaf --leaf-stats &
run LSs   agentform stock 1-60  0.25 36 --leaf-stats &
run LSn   agentform pimc  1-60  0.25 59 --leaf-stats &
wait
echo "ALL DONE $(date)"
