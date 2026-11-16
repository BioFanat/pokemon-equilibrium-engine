#!/usr/bin/env bash
# Launch the open-item real-Showdown runs on Modal (attached, one process per run, budget-capped per run).
# Usage: cd engine && bash showdown/launch_open_items.sh      (logs in modal_logs/)
#   LOG: hedge_bayes vs stock with Abar logging (two-sided Lecture 7 identity)
#   X*:  model-based exploiters vs max-visit / share bandits and RM+
#   E5t/E5s: deterministic top and sampled agent form vs agent form (E5 controls)
#   E3a/E3f: switching opponent, stock for the first 8 seeds of each 16-seed stream, then exploit_top
#   SX*: bayes_safex eps sweep (OURBOT_SAFEX_EPS set in the containers via --safex-eps)
#   RQ*: hedge_bayes vs the style bots (realisability)
# --cpu-s-per-game from measured runs (E2 38, E4 40-45, E5 59, BH 36), so each estimate fits its --max-usd.
set -u
cd "$(dirname "$0")/.."
L=modal_logs; mkdir -p "$L"   # outside showdown/ (uploaded into the image)
run() {  # tag a b seeds max_usd cpu_s_per_game [extra main flags...]
  local tag=$1 a=$2 b=$3 seeds=$4 usd=$5 cps=$6; shift 6
  .venv/bin/modal run showdown/modal_showdown.py::main --streams 10 --cpu-s-per-game "$cps" --slowdown 1 --util 1 \
    --tag "$tag" --a "$a" --b "$b" --seeds "$seeds" --max-usd "$usd" "$@" > "$L/$tag.log" 2>&1
  echo "EXIT $?" >> "$L/$tag.log"
}
# Three waves of <= 60 streams (<= 30 per Modal function): queued streams would count against the budget meter.
echo "WAVE 1 $(date)"
run LOG    hedge_bayes            stock                        1-300 0.9 36 --log-abar &
run E3a    hedge                  switch:stock@8:exploit_top   1-160 0.5 40 &
run E3f    hedge_fs               switch:stock@8:exploit_top   1-160 0.5 40 &
run XUmv   exploit_bandit_ucb1_mv bandit_ucb1_mv               1-300 0.8 45 &
run XU     exploit_bandit_ucb1    bandit_ucb1                  1-300 0.8 45 &
run XR     exploit_rmplus_avg     rmplus_avg                   1-300 0.8 45 &
wait
echo "WAVE 2 $(date)"
run SX3    bayes_safex            stock                        1-200 0.6 36 --safex-eps 0.003 &
run SX10   bayes_safex            stock                        1-200 0.6 36 --safex-eps 0.01 &
run SX30   bayes_safex            stock                        1-200 0.6 36 --safex-eps 0.03 &
run E5t    top                    agentform                    1-300 0.8 59 &
run E5s    agentform              agentform                    1-300 0.8 59 &
run XEmv   exploit_bandit_exp3_mv bandit_exp3_mv               1-300 0.8 45 &
wait
echo "WAVE 3 $(date)"
run SX1    bayes_safex            stock                        1-200 0.6 36 --safex-eps 0.001 &
run RQeq   hedge_bayes            style_eq                     1-100 0.3 59 &
run RQsafe hedge_bayes            style_safe                   1-100 0.3 59 &
run RQunif hedge_bayes            style_uniform                1-100 0.3 59 &
wait
echo "ALL DONE $(date)"
