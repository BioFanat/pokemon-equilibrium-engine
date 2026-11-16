"""Value of information toy: VOI = eta_NL (Theorem C's non-locality term), and A3 (world-only leaves) sets it to 0.

Game. Types theta in {1,2}, prior b = P(theta=1). Stage 1 (ours): 'probe' (pay c, observe a signal of accuracy alpha
about theta) or 'skip'. Stage 2 (ours): pick row r in {1,2}; payoff 1 if r == theta else 0 (passive opponent, so the
stage-2 value as a function of the belief is v(b') = max(b', 1-b'), convex).
True (belief-dependent) values:  V_skip(b) = v(b);  V_probe(b) = E[v(b')] - c  with b' the posterior after the signal.
A3 (world-only continuation):    V^{A3}(s', b') = sum_theta b'(theta) v(delta_theta) = 1 for every b',
so the A3 agent values skip at 1 and probe at 1 - c and never probes.
Checks: (i) VOI(alpha) := E[v(b')] - v(b) >= 0 (Jensen, v convex), = alpha - 1/2 at b = 1/2;
(ii) eta_NL at the skip node = sum_theta b(theta) v(delta_theta) - v(b) = VOI of a perfect reveal = 1 - v(b);
(iii) loss of the A3 agent relative to the optimal policy = max(0, VOI(alpha) - c).
"""
import itertools

import numpy as np

v = lambda b: max(b, 1 - b)


def posterior_after_signal(b, alpha):
    # signal s in {1,2}; P(s=theta)=alpha. Returns list of (prob of s, posterior P(theta=1 | s)).
    out = []
    for s in (1, 2):
        p_s = b * (alpha if s == 1 else 1 - alpha) + (1 - b) * (alpha if s == 2 else 1 - alpha)
        post = b * (alpha if s == 1 else 1 - alpha) / p_s
        out.append((p_s, post))
    return out


worst_jensen = 0.0
print(f"{'b':>5} {'alpha':>5} {'c':>5} {'V_skip':>7} {'V_probe':>8} {'VOI':>6} {'eta_NL':>7} {'A3 picks':>9} {'A3 loss':>8} {'optimal':>8}")
for b, alpha, c in itertools.product((0.5, 0.3, 0.1), (0.5, 0.7, 0.9, 1.0), (0.0, 0.1, 0.3, 0.6)):
    post = posterior_after_signal(b, alpha)
    V_skip = v(b)
    EV = sum(p * v(q) for p, q in post)
    V_probe = EV - c
    voi = EV - V_skip
    worst_jensen = min(worst_jensen, voi)
    eta_nl = b * v(1.0) + (1 - b) * v(0.0) - v(b)        # Theorem C's eta at the skip node (perfect-reveal VOI)
    a3_skip, a3_probe = 1.0, 1.0 - c                      # world-only continuation: 1 at every belief
    a3_choice = "skip" if a3_skip >= a3_probe else "probe"
    opt = "probe" if V_probe > V_skip else "skip"
    a3_loss = max(V_skip, V_probe) - (V_skip if a3_choice == "skip" else V_probe)
    assert abs(a3_loss - max(0.0, voi - c)) < 1e-12
    if b == 0.5:
        assert abs(voi - (alpha - 0.5)) < 1e-12
    assert abs(eta_nl - (1 - v(b))) < 1e-12
    if c in (0.0, 0.3) and alpha in (0.7, 1.0):
        print(f"{b:>5.2f} {alpha:>5.2f} {c:>5.2f} {V_skip:>7.3f} {V_probe:>8.3f} {voi:>6.3f} {eta_nl:>7.3f} {a3_choice:>9} {a3_loss:>8.3f} {opt:>8}")
print(f"\nJensen check: min VOI over grid = {worst_jensen:.3f} (>= 0 required).")
print("Identity checks passed: VOI(alpha) = alpha - 1/2 at b = 1/2; eta_NL(skip node) = perfect-reveal VOI = 1 - v(b);"
      " A3 agent's loss = max(0, VOI - c).")
# Convexity of v across beliefs (Lemma 7(a) of full_game.md in this toy): random chords.
rng = np.random.default_rng(0)
viol = 0
for _ in range(10000):
    x, y, lam = rng.random(3)
    viol += v(lam * x + (1 - lam) * y) > lam * v(x) + (1 - lam) * v(y) + 1e-12
print("convexity chord violations:", viol, "/ 10000")
