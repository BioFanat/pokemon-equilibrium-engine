"""Check the identity Regret_T(s) := sum_{t<=T}(l_hat_t - l_{s,t}) = ln k + ln P_T(s) on the BH logs (clean streams),
with P_T the POST-update posterior after turn T (logged in feedback.bayes.posterior, 4 dp)."""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import records, streams, STYLES
LN5 = math.log(5); worst = 0.0; n = 0; worst_abs_T = None
for k, order in streams("BH"):
    if k in (0, 6): continue
    cum_q = 0.0; cum_s = np.zeros(5)
    for s_, g, p in order:
        for rec in records(p):
            fb = rec.get("feedback")
            if not fb or fb.get("skipped") or "bayes" not in fb: continue
            b = fb["bayes"]
            cum_q += -math.log(max(b["p_qhat"], 1e-9)); cum_s += [-math.log(max(b["p_style"][s], 1e-9)) for s in STYLES]
            post = np.array([b["posterior"][s] for s in STYLES]); n += 1
            for i, s in enumerate(STYLES):
                if post[i] >= 0.01:      # 4-dp rounding makes ln P unreliable below 0.01
                    err = abs((cum_q - cum_s[i]) - (LN5 + math.log(post[i])))
                    if err > worst: worst, worst_abs_T = err, (k, n, s, post[i])
print(f"identity Regret_T(s) = ln5 + ln P_T(s): max abs error over {n} turns x styles with P_T>=0.01: {worst:.4f} nats at {worst_abs_T}")
print("Consequence: Regret_T(s) <= ln 5 for every s, T, with equality iff P_T(s) = 1; the regret curve is the log-posterior of the comparator.")
