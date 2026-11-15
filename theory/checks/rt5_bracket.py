"""Prop R5: lower confidence bounds on exploitability from fixed exploiters; selection (winner's curse)
and its removal by fresh seeds. Win-rate units, v* = 1/2 (seat-swapped pairs), pair score in [0, 1].
Run: engine/.venv/bin/python theory/checks/rt5_bracket.py"""
import numpy as np


def radius(n, m, delta=0.05):
    """Hoeffding + union bound over m exploiters: P(exists j: mean_j - E_j >= r) <= m exp(-2 n r^2)."""
    return np.sqrt(np.log(m / delta) / (2 * n))


if __name__ == "__main__":
    print("Hoeffding radius r(n, m) at delta = 0.05 (pair scores in [0,1])")
    print("   n  " + " ".join(f"m={m:<5d}" for m in (1, 2, 3, 5, 10)))
    for n in (10, 20, 60, 100, 400):
        print(f"{n:4d}  " + " ".join(f"{radius(n, m):.3f}  " for m in (1, 2, 3, 5, 10)))
    print("\nApplied to the measured exploiter scores (exploiter's score, pairs, exploiters tried on that target):")
    for name, s, n, m in [("LBR vs top move (E6q)", 1 - 0.317, 30, 1), ("wb2 vs top move", 0.825, 20, 3),
                          ("wb2 vs sampled", 0.85, 10, 2), ("LBR vs foul-play 2000 (E6f)", 1 - 0.275, 30, 1)]:
        print(f"  {name:28s} point LB {s - .5:+.3f}; Hoeffding LCB (m=1) {s - .5 - radius(n, 1):+.3f}; "
              f"with selection over m={m}: {s - .5 - radius(n, m):+.3f}")
    print("\nWinner's curse: m exploiters with NO edge (true score 1/2), n pairs each, report the max.")
    rng = np.random.default_rng(0)
    for n in (20, 60):
        for m in (1, 3, 10):
            # pair score = mean of two games, each win/loss w.p. 1/2 -> values {0, .5, 1}
            sc = rng.integers(0, 2, size=(20000, m, n, 2)).mean(axis=(2, 3))
            mx = sc.max(1)
            fresh = rng.integers(0, 2, size=(20000, n, 2)).mean(axis=(1, 2))
            print(f"  n={n:3d} m={m:2d}: E[max score] - 1/2 = {mx.mean() - .5:+.3f}; P(max - 1/2 > r(n,m)) = "
                  f"{(mx - .5 > radius(n, m)).mean():.4f}; fresh-seed re-test bias {fresh.mean() - .5:+.4f}")
