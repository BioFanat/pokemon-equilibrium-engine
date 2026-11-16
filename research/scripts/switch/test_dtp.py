"""Acceptance test for the opt-in discount-then-predict flag (OURBOT_BAYES_DTP) of engine/showdown/behavior.py.

(a) env unset: Posterior().config() is the pre-change dict (state files byte-identical);
(b) default update: the logp trajectory on a synthetic 50-step likelihood sequence is bit-identical to the pre-change class;
(c) DTP with lam = 0.98: same per-step argmax as the shipped predict-then-discount run (D'_t = lam * D_t).
Run: ~/sandbox/bot-venv/bin/python research/scripts/switch/test_dtp.py
"""
import os
import sys

import numpy as np

os.environ.pop("OURBOT_BAYES_DTP", None)
os.environ.pop("OURBOT_BAYES_LAMBDA", None)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "engine"), os.path.join(ROOT, "engine", "showdown")]
import behavior  # noqa: E402
from behavior import STYLES, TAU_FRAC, MIX, LAM, Posterior  # noqa: E402


class OldPosterior:
    """Verbatim copy of behavior.Posterior before the DTP change (logp/config/update only)."""

    def __init__(self, lam=LAM, tau_frac=TAU_FRAC, mix=MIX):
        self.lam, self.tau_frac, self.mix = lam, tau_frac, mix
        self.logp = np.full(len(STYLES), -np.log(len(STYLES)))

    def config(self):
        return dict(styles=list(STYLES), lam=self.lam, tau_frac=self.tau_frac, mix=self.mix)

    def update(self, qs, j):
        lik = qs[:, j]
        z = self.lam * self.logp + np.log(lik)
        self.logp = z - (z.max() + np.log(np.exp(z - z.max()).sum()))
        return lik


def lik_sequence(n=50, cols=6, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        qs = rng.dirichlet(np.ones(cols), size=len(STYLES)) * 0.95 + 0.05 / cols
        out.append((qs, int(rng.integers(cols))))
    return out


def run(post, seq):
    traj = []
    for qs, j in seq:
        post.update(qs, j)
        traj.append(post.logp.copy())
    return traj


def main():
    assert behavior.DTP is False
    # (a)
    old_cfg = dict(styles=list(STYLES), lam=1.0, tau_frac=0.1, mix=0.05)
    assert Posterior().config() == old_cfg == OldPosterior().config(), Posterior().config()
    assert list(Posterior().config()) == list(old_cfg)                  # key order too (json bytes)
    assert Posterior(dtp=True).config() == dict(old_cfg, dtp=True)
    print("(a) config unchanged with env unset:", Posterior().config())
    # (b)
    seq = lik_sequence()
    for lam in (1.0, 0.98):
        new, old = run(Posterior(lam=lam), seq), run(OldPosterior(lam=lam), seq)
        assert all(np.array_equal(a, b) for a, b in zip(new, old)), lam
        print(f"(b) lam={lam}: logp trajectory bit-identical over {len(seq)} steps")
    # (c)
    ptd, dtp = run(Posterior(lam=0.98), seq), run(Posterior(lam=0.98, dtp=True), seq)
    a_ptd, a_dtp = [int(x.argmax()) for x in ptd], [int(x.argmax()) for x in dtp]
    assert a_ptd == a_dtp, (a_ptd, a_dtp)
    dev = max(float(np.abs((d - d[-1]) - 0.98 * (p - p[-1])).max()) for p, d in zip(ptd, dtp))
    print(f"(c) DTP lam=0.98 argmax == PTD argmax at all {len(seq)} steps ({len(set(a_ptd))} distinct MAPs);"
          f" max |D_dtp - lam*D_ptd| = {dev:.2e}")
    print("OK")


if __name__ == "__main__":
    main()
