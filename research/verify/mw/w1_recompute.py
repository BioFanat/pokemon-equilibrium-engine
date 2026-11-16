"""Independent recompute of W1 (MWs - BHs paired seed score), pairing/team checks, stream clustering, bootstrap,
distribution of per-seed diffs, and overlap with the earlier Modal BH run. Read-only on the data."""
import glob, json, os, re, math, sys
import numpy as np
from collections import Counter, defaultdict

SB = os.path.expanduser("~/sandbox/ourbot")
ARMS = {"MWs": f"{SB}/mw_MWs/replays_headless", "BHs": f"{SB}/mw_BHs/replays_headless",
        "BH": "/Users/biofanat/Documents/classes/f26/cs535/final/engine/showdown/results/BH/replays_headless"}
RE = re.compile(r"^(?P<tag>[A-Za-z0-9]+)-s(?P<seed>\d+)-g(?P<g>[12])\.json$")


def load(tag, d):
    out = {}
    for p in glob.glob(f"{d}/{tag}-s*-g*.json"):
        m = RE.match(os.path.basename(p))
        if not m or m["tag"] != tag:
            continue
        j = json.load(open(p))
        out[(int(m["seed"]), int(m["g"]))] = j
    return out


def res(j):
    w = j.get("winner_role")
    return 1.0 if w == "A" else 0.0 if w == "B" else 0.5


G = {t: load(t, d) for t, d in ARMS.items()}
for t, g in G.items():
    errs = [k for k, j in g.items() if j.get("error")]
    wr = Counter(j.get("winner_role") for j in g.values())
    print(f"{t}: games={len(g)} errors={len(errs)} {errs[:5]} winner_role={dict(wr)}")
    # consistency: winner_role vs winner name vs score field
    bad = 0
    for k, j in g.items():
        if j.get("error"):
            continue
        sides = j["sides"]; nm = sides["names"]
        wname = j.get("winner")
        exp = "A" if wname == nm[sides["A"]] else "B" if wname == nm[sides["B"]] else None
        if exp != j.get("winner_role"):
            bad += 1
    print(f"   winner_role inconsistent with winner name: {bad}")

def seed_scores(g, seeds):
    s = {}
    for sd in seeds:
        if (sd, 1) in g and (sd, 2) in g and not g[(sd, 1)].get("error") and not g[(sd, 2)].get("error"):
            s[sd] = (res(g[(sd, 1)]) + res(g[(sd, 2)])) / 2
    return s

S = range(1, 201)
sm, sb = seed_scores(G["MWs"], S), seed_scores(G["BHs"], S)
common = sorted(set(sm) & set(sb))
d = np.array([sm[s] - sb[s] for s in common])
n = len(d)
se = d.std(ddof=1) / math.sqrt(n)
print(f"\nW1: n_pairs={n} MWs={np.mean([sm[s] for s in common]):.4f} BHs={np.mean([sb[s] for s in common]):.4f} "
      f"diff={d.mean():.4f} se={se:.4f} CI=[{d.mean()-1.96*se:.4f},{d.mean()+1.96*se:.4f}]")
from scipy import stats
print("  t-test p=%.4f  t-CI half=%.4f" % (stats.ttest_1samp(d, 0).pvalue, stats.t.ppf(.975, n-1)*se))
print("  Wilcoxon p=%.4f" % stats.wilcoxon(d).pvalue)
# sign-flip permutation (seed level)
rng = np.random.default_rng(0)
flips = rng.choice([-1, 1], size=(200000, n))
perm = (flips * d).mean(1)
print("  sign-flip perm two-sided p=%.4f" % np.mean(np.abs(perm) >= abs(d.mean()) - 1e-12))
print("  diff distribution:", dict(sorted(Counter(d.tolist()).items())))
# game-level win rates
for t in ("MWs", "BHs"):
    g = G[t]
    for gi in (1, 2):
        print(f"  {t} g{gi} A win rate: {np.mean([res(g[(s, gi)]) for s in S]):.4f}")

# pairing: same seed -> same battle_seed, team_seeds, team species, stream, A side
mism = Counter()
for s in S:
    for gi in (1, 2):
        a, b = G["MWs"][(s, gi)], G["BHs"][(s, gi)]
        for f in ("battle_seed", "team_seeds", "sim_seed", "stream", "seed", "game"):
            if a.get(f) != b.get(f):
                mism[f] += 1
        if a["sides"]["A"] != b["sides"]["A"]:
            mism["A_side"] += 1
        ta = {p: [m["species"] for m in a["teams"][p]] for p in ("p1", "p2")}
        tb = {p: [m["species"] for m in b["teams"][p]] for p in ("p1", "p2")}
        if ta != tb:
            mism["team_species"] += 1
        if json.dumps(a["teams"], sort_keys=True) != json.dumps(b["teams"], sort_keys=True):
            mism["team_full"] += 1
    # within arm: g1 and g2 swap seats with identical teams
    for t in ("MWs", "BHs"):
        g1, g2 = G[t][(s, 1)], G[t][(s, 2)]
        if json.dumps(g1["teams"], sort_keys=True) != json.dumps(g2["teams"], sort_keys=True):
            mism[f"{t}_g1g2_team"] += 1
        if g1["sides"]["A"] == g2["sides"]["A"]:
            mism[f"{t}_g1g2_noswap"] += 1
        if g1.get("battle_seed") != g2.get("battle_seed"):
            mism[f"{t}_g1g2_battleseed_differs"] += 1
print("\npairing mismatches (count over 400 game slots / 200 seeds):", dict(mism) or "none")

# first move of each game: input_log identical up to first divergence? compute first divergent line index
div = []
for s in S:
    for gi in (1, 2):
        la, lb = G["MWs"][(s, gi)]["input_log"], G["BHs"][(s, gi)]["input_log"]
        i = next((i for i, (x, y) in enumerate(zip(la, lb)) if x != y), min(len(la), len(lb)))
        div.append(i)
print("input_log first divergence index: median=%d, share with identical first 3 lines (start+players)=%.3f, "
      "identical whole logs=%d" % (np.median(div), np.mean(np.array(div) >= 3),
                                   sum(G['MWs'][(s, gi)]['input_log'] == G['BHs'][(s, gi)]['input_log'] for s in S for gi in (1, 2))))

# stream clustering
st = {s: G["MWs"][(s, 1)]["stream"] for s in common}
assert all(G["BHs"][(s, 1)]["stream"] == st[s] for s in common)
print("\nstream map: stream k = seeds[k::10]?", all(st[s] == (s - 1) % 10 for s in common))
streams = sorted(set(st.values()))
by = defaultdict(list)
for s, x in zip(common, d):
    by[st[s]].append(x)
for k in streams:
    v = np.array(by[k])
    print(f"  stream {k}: n={len(v)} mean diff={v.mean():+.4f}  MWs={np.mean([sm[s] for s in common if st[s]==k]):.3f} "
          f"BHs={np.mean([sb[s] for s in common if st[s]==k]):.3f}")
# CR1 cluster SE with G=10, t(9)
Gk = len(streams)
resid = {k: np.sum(np.array(by[k]) - d.mean()) for k in streams}
V = sum(r ** 2 for r in resid.values()) / n ** 2 * Gk / (Gk - 1)
cse = math.sqrt(V)
tq = stats.t.ppf(.975, Gk - 1)
print(f"  stream-clustered SE={cse:.4f} (iid SE {se:.4f}); t(9) CI=[{d.mean()-tq*cse:.4f},{d.mean()+tq*cse:.4f}]; "
      f"normal CI=[{d.mean()-1.96*cse:.4f},{d.mean()+1.96*cse:.4f}]")
sm_means = np.array([np.mean(by[k]) for k in streams])
print(f"  10 stream means: t-test on stream means p={stats.ttest_1samp(sm_means, 0).pvalue:.4f}; "
      f"positive streams={int((sm_means>0).sum())}/10 zero={int((sm_means==0).sum())}")
# ICC-ish: one-way ANOVA across streams
print("  one-way ANOVA across streams on diffs: p=%.3f" % stats.f_oneway(*[by[k] for k in streams]).pvalue)
# stream bootstrap
B = 20000
rng = np.random.default_rng(1)
arrs = [np.array(by[k]) for k in streams]
bm = []
for _ in range(B):
    idx = rng.integers(0, Gk, Gk)
    v = np.concatenate([arrs[i] for i in idx])
    bm.append(v.mean())
bm = np.array(bm)
print(f"  stream bootstrap percentile 95% CI=[{np.percentile(bm,2.5):.4f},{np.percentile(bm,97.5):.4f}] "
      f"P(boot<=0)={np.mean(bm<=0):.4f}")
# stream-level sign-flip permutation (exact, 2^10)
import itertools
tot = []
for sg in itertools.product([-1, 1], repeat=Gk):
    tot.append(sum(sgn * np.sum(a) for sgn, a in zip(sg, arrs)) / n)
tot = np.array(tot)
print(f"  exact stream sign-flip test (1024 flips) two-sided p={np.mean(np.abs(tot) >= abs(d.mean())-1e-12):.4f}")
# seed-level bootstrap for reference
bs = d[rng.integers(0, n, (B, n))].mean(1)
print(f"  seed bootstrap CI=[{np.percentile(bs,2.5):.4f},{np.percentile(bs,97.5):.4f}]")

# leave-one-out / few-seed sensitivity
order = np.argsort(-d)
print("\nper-seed diffs: #pos=%d #neg=%d #zero=%d; |diff|=1 seeds: +1:%d -1:%d" % (
    (d > 0).sum(), (d < 0).sum(), (d == 0).sum(), (d == 1).sum(), (d == -1).sum()))
for drop in (1, 2, 3, 5):
    # remove the most favourable `drop` seeds
    keep = np.ones(n, bool); keep[order[:drop]] = False
    v = d[keep]; s_ = v.std(ddof=1) / math.sqrt(len(v))
    print(f"  drop top {drop} seeds: mean={v.mean():.4f} CI=[{v.mean()-1.96*s_:.4f},{v.mean()+1.96*s_:.4f}]")
# needed sign changes: how many +0.5 seeds flipped to 0 make CI include 0
for half in ("first", "second"):
    sel = [i for i, s in enumerate(common) if (s <= 100) == (half == "first")]
    v = d[sel]; s_ = v.std(ddof=1) / math.sqrt(len(v))
    print(f"  seeds {'1-100' if half=='first' else '101-200'}: mean={v.mean():.4f} CI=[{v.mean()-1.96*s_:.4f},{v.mean()+1.96*s_:.4f}]")

# earlier Modal BH on overlapping seeds 1-200
sh = seed_scores(G["BH"], S)
c2 = sorted(set(sh) & set(sb) & set(sm))
print(f"\nBH (Modal) seeds 1-200: n={len(c2)} BH={np.mean([sh[s] for s in c2]):.4f} BHs={np.mean([sb[s] for s in c2]):.4f} "
      f"MWs={np.mean([sm[s] for s in c2]):.4f}")
sh_all = seed_scores(G["BH"], range(1, 401))
print(f"BH (Modal) seeds 1-400: n={len(sh_all)} mean={np.mean(list(sh_all.values())):.4f}; seeds 201-400: "
      f"{np.mean([v for s, v in sh_all.items() if s > 200]):.4f}")
for lab, other in (("BHs", sb), ("MWs", sm)):
    x = np.array([other[s] - sh[s] for s in c2]); s_ = x.std(ddof=1) / math.sqrt(len(x))
    r = np.corrcoef([other[s] for s in c2], [sh[s] for s in c2])[0, 1]
    print(f"  {lab} - BH: {x.mean():+.4f} CI=[{x.mean()-1.96*s_:.4f},{x.mean()+1.96*s_:.4f}]  corr(per-seed)={r:.3f}")
r = np.corrcoef([sm[s] for s in c2], [sb[s] for s in c2])[0, 1]
print(f"  corr(MWs, BHs per-seed)={r:.3f}")
# are BH and BHs teams identical on same seed?
same = sum(json.dumps(G["BH"][(s, 1)]["teams"], sort_keys=True) == json.dumps(G["BHs"][(s, 1)]["teams"], sort_keys=True)
           for s in c2 if (s, 1) in G["BH"])
print(f"  BH vs BHs identical teams on seed g1: {same}/{len(c2)}; battle_seed same: "
      f"{sum(G['BH'][(s,1)].get('battle_seed')==G['BHs'][(s,1)].get('battle_seed') for s in c2)}")
print("  BH stream field present:", G["BH"][(1, 1)].get("stream"), " k:", G["BH"][(1, 1)].get("k"),
      " methods:", G["BH"][(1, 1)].get("methods"))
json.dump(dict(common=common, d=d.tolist(), stream={str(k): v for k, v in st.items()}),
          open(os.path.join(os.path.dirname(__file__), "w1_diffs.json"), "w"))
