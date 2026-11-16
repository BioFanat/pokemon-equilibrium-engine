"""T3(vi): misspecified concentration coefficient under the mismatch proxy.
Truth: style s* on A' = A + sigma_m ptp(A) N(0,1); our styles on A. Reference style s° = KL-closest (over the 57
matrices x 20 draws). Reports c(s) = E_{b~sigma}[sqrt(q_s(b)/q_s°(b))] (mean over matrices = the i.i.d.-matrix
contraction factor per turn), its max over matrices (sup version), and the per-turn KL gaps."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b4_concentration_lib import mats, Q, style_dists, STYLES  # noqa: E402

rng = np.random.default_rng(11)
for sname in ("eq", "safe", "uniform", "br_unif"):
    s = STYLES.index(sname)
    for sigma_m in (0.0, 0.05, 0.1, 0.2, 0.4):
        kl = np.zeros(5); cs = []
        draws = []
        for i, A in enumerate(mats):
            for rep in range(20 if sigma_m > 0 else 1):
                A2 = A + sigma_m * float(np.ptp(A)) * rng.normal(size=A.shape) if sigma_m > 0 else A
                qt = style_dists(A2)[s]
                draws.append((i, qt))
                kl += np.array([(qt * np.log(qt / Q[i][b])).sum() for b in range(5)])
        kl /= len(draws); s0 = int(kl.argmin())
        c = np.array([[(qt * np.sqrt(Q[i][b] / Q[i][s0])).sum() for b in range(5)] for i, qt in draws])
        print(f"truth {sname:>7} sigma_m {sigma_m:.2f}: KL-closest s°={STYLES[s0]:>7}; KL gaps " +
              " ".join(f"{STYLES[b]}={kl[b]-kl[s0]:.3f}" for b in range(5) if b != s0) +
              " | c(s) mean " + " ".join(f"{STYLES[b]}={c[:, b].mean():.3f}" for b in range(5) if b != s0) +
              " | c(s) max " + " ".join(f"{STYLES[b]}={c[:, b].max():.2f}" for b in range(5) if b != s0) +
              f" | bound sum 1/(1-mean c) = {sum(1/(1-c[:, b].mean()) for b in range(5) if b != s0 and c[:, b].mean() < 1):.1f}")
