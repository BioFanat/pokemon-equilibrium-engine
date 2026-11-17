"""Follow-up on dynamics/b3.out: same 120 matrices (same RNG draw). (1) FS99 bound for constant-step Hedge averages:
gap(avg) <= eps_R + eps_C with eps = ln n/(eta T) + eta/8 (rewards in [0,1]); (2) pure-saddle fraction; (3) last-iterate
gap split by saddle / no saddle."""
import pickle, numpy as np
D = '/private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/brainstorm/data.pkl'
recs = pickle.load(open(D, 'rb'))
print('records', len(recs), 'games', len(set(r['seed'] for r in recs)))
mats = [np.asarray(r['M'], float) for r in recs if min(np.asarray(r['M']).shape) >= 2]
mats = [M for M in mats if np.all(np.isfinite(M)) and np.ptp(M) > 0]
print('two-sided nonconstant', len(mats))
rng = np.random.default_rng(0)
idx = rng.choice(len(mats), size=120, replace=False)
mats = [(mats[i] - mats[i].min()) / np.ptp(mats[i]) for i in idx]
def gap(A, x, y): return (A @ y).max() - (x @ A).min()
def hedge(A, N, opt, eta=0.1):
    n, m = A.shape; Lx = np.zeros(n); Ly = np.zeros(m); xs = np.zeros(n); ys = np.zeros(m)
    x = np.ones(n) / n; y = np.ones(m) / m
    for t in range(N):
        gx = A @ y; gy = x @ A; Lx += gx; Ly += gy
        px = Lx + (gx if opt else 0); py = Ly + (gy if opt else 0)
        x = np.exp(eta * (px - px.max())); x /= x.sum(); y = np.exp(-eta * (py - py.min())); y /= y.sum()
        xs += x; ys += y
    return xs / N, ys / N, x, y
viol = 0; ratios = []; saddle = []; last = {False: [], True: []}; avg = []
for A in mats:
    n, m = A.shape
    xa, ya, xl, yl = hedge(A, 1000, False)
    b = np.log(n) / 100 + 0.1 / 8 + np.log(m) / 100 + 0.1 / 8
    g = gap(A, xa, ya); avg.append(g); ratios.append(g / b); viol += g > b
    sad = A.min(1).max() >= A.max(0).min() - 1e-12
    saddle.append(sad)
    for opt in (False, True):
        _, _, xl2, yl2 = hedge(A, 1000, opt)
        last[opt].append((sad, gap(A, xl2, yl2)))
print('Hedge avg gap median %.4f; FS99 bound violations %d/120; max gap/bound %.2f; median bound %.4f' % (np.median(avg), viol, max(ratios), np.median([np.log(A.shape[0])/100+np.log(A.shape[1])/100+0.025 for A in mats])))
print('pure saddle fraction %.2f' % np.mean(saddle))
for opt in (False, True):
    s = np.array([g for sd, g in last[opt] if sd]); ns = np.array([g for sd, g in last[opt] if not sd])
    print(('OMWU' if opt else 'Hedge') + ' last iterate: saddle n=%d median %.4f | no saddle n=%d median %.4f p90 %.4f' % (len(s), np.median(s), len(ns), np.median(ns), np.quantile(ns, .9)))
# (4) average gap split by saddle, both algorithms
for opt in (False, True):
    sp = {True: [], False: []}
    for A in mats:
        xa, ya, _, _ = hedge(A, 1000, opt)
        sad = A.min(1).max() >= A.max(0).min() - 1e-12
        sp[sad].append(gap(A, xa, ya))
    print(('OMWU' if opt else 'Hedge') + ' average: saddle median %.4f | no saddle median %.4f' % (np.median(sp[True]), np.median(sp[False])))
