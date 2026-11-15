"""Bayesian behavioural model of the opponent: a posterior over a finite set of opponent styles (the "behavioural
tendencies" half of the proposal's Bayesian memory).

Each style s maps (Abar, p) to a distribution q_s over the OPPONENT's columns of Abar (our perspective: rows = our
options, the opponent minimises). p is our reference mixture this turn (see run_ourbot: the mixture we would play
with the old br_unif-only safex predictor, so the prediction is not circular in q-hat).

  eq       the opponent's minimax strategy y on Abar (nash.lp)
  br_unif  softmax_j(-mean_i Abar[i, j] / tau)      best reply to a uniform guess about us ("MCTS-like")
  br_us    softmax_j(-(p . Abar)_j / tau)           best reply to our actual reference mixture p
  safe     softmax_j(-max_i Abar[i, j] / tau)       the opponent's maximin (minimises its worst case)
  uniform  1/n

tau = TAU_FRAC * ptp(Abar) (1 if Abar is constant), shared by all softmax styles; every style is mixed with MIX
uniform so no likelihood is 0. Posterior: P_t(s) ∝ P_{t-1}(s)^lam * q_s,t(b_t) with q_s,t computed on turn t's Abar and
p (stored at the decision); skipped turns leave it unchanged. Prediction: q-hat = Σ_s P(s) q_s. The posterior is
carried across games in a JSON state file next to the Hedge state (<tag>-stream<k>-<role>.bayes.json).
"""
import json
import os

import numpy as np

import nash

STYLES = ("eq", "br_unif", "br_us", "safe", "uniform")
TAU_FRAC = float(os.environ.get("OURBOT_BAYES_TAU", "0.1"))       # softmax temperature, in units of ptp(Abar)
MIX = float(os.environ.get("OURBOT_BAYES_MIX", "0.05"))           # uniform mixed into every style
LAM = float(os.environ.get("OURBOT_BAYES_LAMBDA", "1.0"))         # forgetting factor (1 = none)


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def style_dists(A, p, tau_frac=TAU_FRAC, mix=MIX):
    """(len(STYLES) x n) matrix: row s = q_s over the opponent's columns of A given our reference mixture p."""
    n = A.shape[1]
    u = np.full(n, 1.0 / n)
    if n == 1:
        return np.ones((len(STYLES), 1))
    tau = tau_frac * float(np.ptp(A)) or 1.0
    y = np.clip(nash.lp(A)[1], 0, None)
    y = y / y.sum() if y.sum() > 0 else u
    raw = np.stack([y, _softmax(-A.mean(axis=0) / tau), _softmax(-(p @ A) / tau), _softmax(-A.max(axis=0) / tau), u])
    return (1 - mix) * raw + mix * u


def old_predictor(A, mix=MIX):
    """The old safex predictor (one-hot on argmin_j mean_i A[i, j]), smoothed with the same MIX for log loss."""
    n = A.shape[1]
    q = np.zeros(n); q[int(A.mean(axis=0).argmin())] = 1
    return (1 - mix) * q + mix / n


class Posterior:
    """Posterior over STYLES in log form. update(qs, j): logP <- lam * logP + log qs[:, j], renormalised."""

    def __init__(self, lam=LAM, tau_frac=TAU_FRAC, mix=MIX):
        self.lam, self.tau_frac, self.mix = lam, tau_frac, mix
        self.logp = np.full(len(STYLES), -np.log(len(STYLES)))

    def config(self):
        return dict(styles=list(STYLES), lam=self.lam, tau_frac=self.tau_frac, mix=self.mix)

    def probs(self):
        w = np.exp(self.logp - self.logp.max())
        return w / w.sum()

    def predict(self, qs):
        return self.probs() @ qs

    def update(self, qs, j):
        """Bayes step on the observed column j of the turn whose style matrix is qs; returns the likelihoods."""
        lik = qs[:, j]
        z = self.lam * self.logp + np.log(lik)
        self.logp = z - (z.max() + np.log(np.exp(z - z.max()).sum()))
        return lik


def state_path_for(hedge_path):
    """Posterior state file next to the Hedge state file (same tag / stream / role)."""
    if not hedge_path:
        return None
    root = hedge_path[:-len(".json")] if hedge_path.endswith(".json") else hedge_path
    return root + ".bayes.json"


def load_posterior(path):
    """Posterior restored from `path` if present (written with another config: error). Returns (posterior, meta)."""
    post = Posterior()
    meta = dict(games=0, updates=0, skipped=0)
    if path and os.path.exists(path):
        d = json.load(open(path))
        if d["config"] != post.config():
            raise ValueError(f"posterior state {path} was written with {d['config']}")
        post.logp = np.array(d["logp"])
        meta = d["meta"]
    meta["games"] += 1
    return post, meta


def save_posterior(path, post, meta):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = dict(config=post.config(), logp=post.logp.tolist(),
             posterior=dict(zip(STYLES, post.probs().round(6).tolist())), meta=meta)
    tmp = f"{path}.{os.getpid()}.tmp"                      # per process: parallel seeds of one stream share `path`
    json.dump(d, open(tmp, "w"))
    os.replace(tmp, path)


def decide(A, p_ref, post, labels):
    """Style distributions and q-hat on this turn's Abar; returns (qs, qhat, log record)."""
    qs = style_dists(A, p_ref, post.tau_frac, post.mix)
    qhat = post.predict(qs)
    rec = dict(posterior=dict(zip(STYLES, post.probs().round(4).tolist())), labels=list(labels),
               qhat=qhat.round(4).tolist(), q={s: q.round(4).tolist() for s, q in zip(STYLES, qs)})
    return qs, qhat, rec


def score_and_update(post, pend, label, forced):
    """Prequential scoring of the observed opponent column `label` (probabilities BEFORE the update) under q-hat, each
    style and the old predictor; then the Bayes update. Returns the feedback log record."""
    j = pend["labels"].index(label)
    qs, qhat, old = pend["qs"], pend["qhat"], pend["old"]
    rec = dict(col=label, j=j, forced=forced, n_cols=len(qhat),
               p_qhat=round(float(qhat[j]), 6), p_style={s: round(float(q[j]), 6) for s, q in zip(STYLES, qs)},
               p_old=round(float(old[j]), 6), hit_qhat=bool(int(qhat.argmax()) == j),
               hit_style={s: bool(int(q.argmax()) == j) for s, q in zip(STYLES, qs)}, hit_old=bool(int(old.argmax()) == j))
    post.update(qs, j)
    rec["posterior"] = dict(zip(STYLES, post.probs().round(4).tolist()))
    return rec
