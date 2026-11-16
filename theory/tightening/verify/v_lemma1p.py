"""Independent check of Lemma 1' (bounds_epsv.md sec 1.2) and the 'naive cancellation' counterexample (claim 7).
Own LP (scipy linprog), not nash.lp. Random + adversarial (hill-climb maximising loss - budget), and
non-unique optima (x*, yhat chosen adversarially among optimal faces via vertex enumeration of LP with objective).
python v_lemma1p.py
"""
import numpy as np
from scipy.optimize import linprog

rng = np.random.default_rng(12345)


def maximin(A):
    """row maximin strategy and value of A (row maximises)."""
    n, m = A.shape
    c = np.zeros(n + 1); c[-1] = -1
    A_ub = np.hstack([-A.T, np.ones((m, 1))])  # v - x^T A_j <= 0
    A_eq = np.hstack([np.ones((1, n)), np.zeros((1, 1))])
    r = linprog(c, A_ub=A_ub, b_ub=np.zeros(m), A_eq=A_eq, b_eq=[1], bounds=[(0, None)] * n + [(None, None)], method="highs")
    x = np.clip(r.x[:n], 0, None); return x / x.sum(), r.x[-1]


def optimal_face_extreme(A, v, w):
    """a maximin strategy of A minimising w.x (extreme point of the optimal face)."""
    n, m = A.shape
    r = linprog(w, A_ub=-A.T, b_ub=-(v - 1e-9) * np.ones(m), A_eq=np.ones((1, n)), b_eq=[1], bounds=[(0, None)] * n, method="highs")
    x = np.clip(r.x, 0, None); return x / x.sum()


def check(Q, B, x, xs, yh):
    vQ = maximin(Q)[1]; vB = maximin(B)[1]
    eps_s = max(0.0, vB - (x @ B).min())
    loss = vQ - (x @ Q).min()
    b3 = eps_s + xs @ (Q - B) @ yh + (x @ (B - Q)).max()
    b2 = eps_s + 2 * np.abs(Q - B).max()
    return loss, b3, b2


def gap(Q, E, mode):
    B = Q + E
    xB, vB = maximin(B)
    if mode == "eps":  # an eps_s-suboptimal x: mix the optimum with a random pure row
        x = 0.7 * xB + 0.3 * np.eye(B.shape[0])[rng.integers(B.shape[0])]
    else:
        x = xB
    xs, vQ = maximin(Q)
    yh, _ = maximin(-B.T)
    l, b3, b2 = check(Q, B, x, xs, yh)
    return l - b3, b3 - b2


worst1 = -np.inf; worst2 = -np.inf; nviol = 0
for trial in range(3000):
    n, m = rng.integers(2, 6, size=2)
    Q = rng.uniform(-1, 1, (n, m)); E = rng.uniform(-1, 1, (n, m)) * rng.choice([0.05, 0.3, 1.0, 3.0])
    mode = "eps" if trial % 2 else "opt"
    g1, g2 = gap(Q, E, mode)
    # adversarial hill-climb on E (and Q) to maximise loss - bud3
    if trial % 10 == 0:
        for it in range(150):
            Q2 = Q + 0.1 * rng.normal(size=Q.shape); E2 = E + 0.1 * rng.normal(size=E.shape)
            h1, h2 = gap(Q2, E2, mode)
            if h1 > g1:
                Q, E, g1, g2 = Q2, E2, h1, h2
    worst1 = max(worst1, g1); worst2 = max(worst2, g2)
    nviol += (g1 > 1e-7) + (g2 > 1e-7)
print(f"random+hill-climb: violations {nviol}; max(loss-bud3) {worst1:.2e}; max(bud3-bud2) {worst2:.2e}")

# non-unique optima: degenerate games, choose x*, yhat as arbitrary extreme points of optimal faces
nv = 0; mx = -np.inf
for trial in range(1500):
    n, m = rng.integers(2, 5, size=2)
    Q = rng.integers(-2, 3, (n, m)).astype(float); B = Q + rng.integers(-2, 3, (n, m)).astype(float) * 0.5
    xs0, vQ = maximin(Q); _, vB = maximin(B)
    xs = optimal_face_extreme(Q, vQ, rng.normal(size=n))
    yh_neg = optimal_face_extreme(-B.T, -vB, rng.normal(size=m))
    x = optimal_face_extreme(B, vB, rng.normal(size=n))
    l, b3, b2 = check(Q, B, x, xs, yh_neg)
    mx = max(mx, l - b3); nv += (l > b3 + 1e-7) + (b3 > b2 + 1e-7)
print(f"degenerate integer games, arbitrary optimal-face vertices: violations {nv}; max(loss-bud3) {mx:.2e}")

# Claim 7: draft's counterexample  Qhat = I, Q = Qhat + 1 (0,-10)
Qh = np.eye(2); Q = Qh + np.array([[0, -10], [0, -10]])
vQ = maximin(Q)[1]; vh = maximin(Qh)[1]
for x in [np.array([.5, .5]), np.array([0., 1.])]:
    print(f"draft CE: x={x}, eps_s in Qhat={vh-(x@Qh).min():.3f}, loss in Q={vQ-(x@Q).min():.3f}; eps_v={np.abs(Q-Qh).max()}")
# E = 1(0,-10) has identical rows -> (x2-x1)^T E sigma = 0 for every sigma: the error cancels exactly.
# Correct counterexample: Qhat = 0 (every x exactly optimal, eps_s = 0), Q = eps*[[1,-1],[1,1]]
eps = 1.0; Qh = np.zeros((2, 2)); Q = eps * np.array([[1., -1.], [1., 1.]])
vQ = maximin(Q)[1]
for x in [np.array([1., 0.]), np.array([0., 1.])]:
    print(f"fixed CE: x={x}, eps_s in Qhat={maximin(Qh)[1]-(x@Qh).min():.3f}, loss in Q={vQ-(x@Q).min():.3f}; eps_v={np.abs(Q-Qh).max()}")
