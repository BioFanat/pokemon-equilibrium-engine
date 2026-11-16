"""Stock foul-play MCTS load in MWs vs BHs: per game, iterations per decision (sum over the 4 'Iterations i:' lines
of each search), timing.B cpu/wall per decision, and game start/end time (|t:| in the .log). Paired by (seed, game)."""
import glob, json, os, re, math
import numpy as np
from scipy import stats

SB = os.path.expanduser("~/sandbox/ourbot")
D = {"MWs": f"{SB}/mw_MWs/replays_headless", "BHs": f"{SB}/mw_BHs/replays_headless"}
IT = re.compile(r"^Iterations (\d+): (\d+)")


def fp_iters(path):
    per, cur = [], None
    for line in open(path, errors="replace"):
        if line.startswith("Searching for a move using MCTS"):
            if cur:
                per.append(cur)
            cur = []
        m = IT.match(line)
        if m and cur is not None:
            cur.append(int(m[2]))
    if cur:
        per.append(cur)
    return per


def times(path):
    ts = [int(l.split("|")[2]) for l in open(path, errors="replace") if l.startswith("|t:|")]
    return (ts[0], ts[-1]) if ts else (None, None)


rows = {}
for t, d in D.items():
    for s in range(1, 201):
        for g in (1, 2):
            base = f"{d}/{t}-s{s}-g{g}"
            j = json.load(open(base + ".json"))
            per = fp_iters(base + ".fp.log")
            tot = [sum(x) for x in per]
            nw = [len(x) for x in per]
            t0, t1 = times(base + ".log")
            tb = j["timing"]["B"]; ta = j["timing"]["A"]
            rows[(t, s, g)] = dict(it=np.mean(tot) if tot else np.nan, it_med=np.median(tot) if tot else np.nan,
                                   n_search=len(per), nworkers=sorted(set(nw)),
                                   b_cpu=tb["cpu_s_per_decision"], b_wall=tb["wall_s_per_decision"],
                                   a_cpu=ta["cpu_s_per_decision"], a_wall=ta["wall_s_per_decision"],
                                   b_dec=tb["decisions"], t0=t0, t1=t1, win=1.0 if j["winner_role"] == "A" else 0.0,
                                   stream=j["stream"], turns=j["turns"])
json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in rows.items()},
          open(os.path.join(os.path.dirname(__file__), "load_rows.json"), "w"), default=float)

for t in D:
    v = [r for (tt, s, g), r in rows.items() if tt == t]
    print(f"{t}: stock iters/decision mean={np.nanmean([r['it'] for r in v]):.0f} median-of-game-means="
          f"{np.nanmedian([r['it'] for r in v]):.0f}; stock cpu/dec={np.mean([r['b_cpu'] for r in v]):.3f} "
          f"wall/dec={np.mean([r['b_wall'] for r in v]):.3f}; A cpu/dec={np.mean([r['a_cpu'] for r in v]):.3f} "
          f"A wall/dec={np.mean([r['a_wall'] for r in v]):.3f}; workers per search={set(sum([r['nworkers'] for r in v], []))}")
    t0 = min(r["t0"] for r in v); t1 = max(r["t1"] for r in v)
    print(f"   time window: {t0} .. {t1}  ({(t1-t0)/3600:.2f} h)")

# paired by (seed, game)
keys = [(s, g) for s in range(1, 201) for g in (1, 2)]
for f in ("it", "b_cpu", "b_wall", "a_cpu", "a_wall"):
    x = np.array([rows[("MWs", s, g)][f] - rows[("BHs", s, g)][f] for s, g in keys])
    rel = np.nanmean([rows[("MWs", s, g)][f] for s, g in keys]) / np.nanmean([rows[("BHs", s, g)][f] for s, g in keys]) - 1
    se = np.nanstd(x, ddof=1) / math.sqrt(np.sum(~np.isnan(x)))
    print(f"paired MWs-BHs {f}: mean={np.nanmean(x):+.4g} (rel {rel:+.2%}) CI=[{np.nanmean(x)-1.96*se:+.4g},{np.nanmean(x)+1.96*se:+.4g}]")

# time alignment: start offset of the same (seed, game) in the two arms
off = np.array([(rows[("MWs", s, g)]["t0"] - rows[("BHs", s, g)]["t0"]) / 60 for s, g in keys])
print(f"start-time offset MWs-BHs for same (seed,game), minutes: median={np.median(off):.1f} "
      f"p10={np.percentile(off,10):.1f} p90={np.percentile(off,90):.1f}")

# time-window comparison: bin by start hour of the joint window, compare stock iterations in each arm
allr = [(t, r) for (t, s, g), r in rows.items()]
T0 = min(r["t0"] for _, r in allr)
bins = defaultdict = {}
for t, r in allr:
    b = int((r["t0"] - T0) // 1800)
    bins.setdefault(b, {}).setdefault(t, []).append(r["it"])
print("30-min bins (start time): stock iters/decision MWs vs BHs (n games)")
for b in sorted(bins):
    m = bins[b].get("MWs", []); h = bins[b].get("BHs", [])
    print(f"  bin {b}: MWs {np.nanmean(m) if m else float('nan'):.0f} (n={len(m)})  BHs {np.nanmean(h) if h else float('nan'):.0f} (n={len(h)})")

# does stock's iteration count predict A's win (within arm)? logistic-ish: compare mean iters in A-wins vs A-losses
for t in D:
    v = [r for (tt, s, g), r in rows.items() if tt == t]
    w = [r["it"] for r in v if r["win"] == 1]; l = [r["it"] for r in v if r["win"] == 0]
    print(f"{t}: stock iters when A wins {np.nanmean(w):.0f} (n={len(w)}) vs A loses {np.nanmean(l):.0f} (n={len(l)}); "
          f"MWU p={stats.mannwhitneyu(w, l).pvalue:.3f}")
# pooled: A win ~ iters (both arms), point-biserial
v = list(rows.values())
print("pooled corr(win, stock iters)=%.3f" % np.corrcoef([r["win"] for r in v], [r["it"] for r in v])[0, 1])
