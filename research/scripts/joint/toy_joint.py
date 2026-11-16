"""Hand-enumerable example where the joint (set, style) posterior differs from the product of its marginals, for
(a) predicting the opponent's next action and (b) our best reply.

Opponent sets S1 = {A, B}, S2 = {A, C} (prior 1/2 each; A is in BOTH sets, so observing A does not identify the set).
Styles: greedy (plays its best column vs the public reference mixture x_ref with prob 0.9, the rest shares 0.1) and
uniform; prior 1/2 each. Payoff matrix (to us, opponent minimises; rows r1, r2) chosen by a small integer search so
that the best reply flips. The "product" tracker runs two independent Bayes filters (set belief updated with
P(obs|S) = Σ_σ P(σ) q_σ(obs|S); style belief with Σ_S P(S) q_σ(obs|S)) and predicts with their product, which is
a hypothetical product tracker (the real pipeline is not this: foul-play applies hard set filters only, behavior.Posterior tracks σ).
"""
import itertools

import numpy as np

sets = {"S1": ["A", "B"], "S2": ["A", "C"]}


def q(style, S, M, x_ref):
    cols = sets[S]
    if style == "unif":
        return {c: 1 / len(cols) for c in cols}
    val = {c: x_ref @ M[c] for c in cols}
    best = min(val, key=val.get)
    return {c: (0.9 if c == best else 0.1 / (len(cols) - 1)) for c in cols}


def posterior(obs, M, x_ref, joint=True):
    PS = {"S1": .5, "S2": .5}; Pst = {"greedy": .5, "unif": .5}
    W = {(S, s): .25 for S in sets for s in Pst}
    for o in obs:
        if joint:
            W = {k: w * q(k[1], k[0], M, x_ref).get(o, 0.0) for k, w in W.items()}
            Z = sum(W.values()); W = {k: w / Z for k, w in W.items()}
        else:
            lS = {S: sum(Pst[s] * q(s, S, M, x_ref).get(o, 0.0) for s in Pst) for S in sets}
            ls = {s: sum(PS[S] * q(s, S, M, x_ref).get(o, 0.0) for S in sets) for s in Pst}
            PS = {S: PS[S] * lS[S] for S in sets}; Z = sum(PS.values()); PS = {S: v / Z for S, v in PS.items()}
            Pst = {s: Pst[s] * ls[s] for s in Pst}; Z = sum(Pst.values()); Pst = {s: v / Z for s, v in Pst.items()}
            W = {(S, s): PS[S] * Pst[s] for S in sets for s in Pst}
    return W


def predict(W, M, x_ref):
    p = {}
    for (S, s), w in W.items():
        for c, pc in q(s, S, M, x_ref).items():
            p[c] = p.get(c, 0) + w * pc
    return p


def row_values(p, M):
    return sum(pc * M[c] for c, pc in p.items())


def show(M, x_ref, obs):
    Wj, Wi = posterior(obs, M, x_ref, True), posterior(obs, M, x_ref, False)
    pj, pi = predict(Wj, M, x_ref), predict(Wi, M, x_ref)
    vj, vi = row_values(pj, M), row_values(pi, M)
    print(f"M (cols A,B,C; rows r1,r2) = {np.array([M[c] for c in 'ABC']).T.tolist()}, x_ref = {x_ref.tolist()}, obs = {obs}")
    print(f"  joint   P(S,σ) = { {k: round(v, 3) for k, v in Wj.items()} }")
    print(f"  product P(S)P(σ) = { {k: round(v, 3) for k, v in Wi.items()} }")
    print(f"  next action: joint {dict((c, round(v, 3)) for c, v in sorted(pj.items()))} | product {dict((c, round(v, 3)) for c, v in sorted(pi.items()))}  "
          f"TV = {0.5 * sum(abs(pj.get(c, 0) - pi.get(c, 0)) for c in 'ABC'):.3f}")
    print(f"  our row values: joint {vj.round(3)} -> r{vj.argmax() + 1} | product {vi.round(3)} -> r{vi.argmax() + 1}; "
          f"loss of playing the product's reply under the joint belief: {vj.max() - vj[vi.argmax()]:.3f}")


if __name__ == "__main__":
    x_ref = np.array([0.5, 0.5])
    found = None
    for vals in itertools.product(range(-2, 3), repeat=6):
        M = {"A": np.array(vals[0:2], float), "B": np.array(vals[2:4], float), "C": np.array(vals[4:6], float)}
        for obs in (["A"], ["A", "A"]):
            Wj, Wi = posterior(obs, M, x_ref, True), posterior(obs, M, x_ref, False)
            vj, vi = row_values(predict(Wj, M, x_ref), M), row_values(predict(Wi, M, x_ref), M)
            if vj.argmax() != vi.argmax() and vj.max() - vj[vi.argmax()] > 0.05 and vi.max() - vi[vj.argmax()] > 0.05:
                found = (M, obs); break
        if found:
            break
    show(*found[:1], x_ref, found[1])
