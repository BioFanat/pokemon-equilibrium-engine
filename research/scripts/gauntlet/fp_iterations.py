"""Foul-play MCTS iterations per sampled battle, by per-state time budget, from the G1 gauntlet's fp.log files
(agentform vs stock@<ms> pairings: only the stock side runs MCTS). Usage: fp_iterations.py [replays_headless dir]."""
import glob, os, re, statistics as st, sys
d = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/sandbox/ourbot/replays_headless")
print("| foul-play budget (ms per state) | searches | median iterations | p10 | p90 | mean |")
print("|---|---|---|---|---|---|")
for ms in (25, 100, 400, 1000):
    its = []
    for f in glob.glob(os.path.join(d, f"G1_agentform_stockat{ms}-s*-g*.fp.log")):
        its += [int(x) for x in re.findall(r"Iterations \d+: (\d+)", open(f).read())]
    q = sorted(its)
    print(f"| {ms} | {len(its)} | {st.median(its):.0f} | {q[len(q)//10]} | {q[9*len(q)//10]} | {st.mean(its):.0f} |")
