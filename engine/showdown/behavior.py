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
DTP = os.environ.get("OURBOT_BAYES_DTP") == "1"                   # discount-then-predict (opt-in; default predict-then-discount)


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
    """Posterior over STYLES in log form. update(qs, j): logP <- lam * logP + log qs[:, j], renormalised. With dtp
    (OURBOT_BAYES_DTP=1) the discount also covers the new likelihood, logP <- lam * (logP + log qs[:, j]), so the stored
    (and predicting) posterior is the discounted one."""

    def __init__(self, lam=LAM, tau_frac=TAU_FRAC, mix=MIX, dtp=DTP):
        self.lam, self.tau_frac, self.mix, self.dtp = lam, tau_frac, mix, dtp
        self.logp = np.full(len(STYLES), -np.log(len(STYLES)))

    def config(self):
        d = dict(styles=list(STYLES), lam=self.lam, tau_frac=self.tau_frac, mix=self.mix)
        if self.dtp:
            d["dtp"] = True
        return d

    def probs(self):
        w = np.exp(self.logp - self.logp.max())
        return w / w.sum()

    def predict(self, qs):
        return self.probs() @ qs

    def update(self, qs, j):
        """Bayes step on the observed column j of the turn whose style matrix is qs; returns the likelihoods."""
        lik = qs[:, j]
        z = self.lam * (self.logp + np.log(lik)) if self.dtp else self.lam * self.logp + np.log(lik)
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


# ---------------------------------------------------------------- membership-aware per-world model (*_mw methods)
# The functions above work on the world-averaged Abar, whose columns are the UNION of the K worlds' labels, so every
# style spreads mass over options the opponent may not have. Below, each style is evaluated on each world's own matrix
# M_k (columns = that world's options) and the worlds are mixed with their weights w_k:
#   q(label) = Σ_k w_k Σ_σ P(σ) q_σ(label | M_k)     (world k contributes nothing to a label it lacks)
#   P(b | σ) = Σ_{k ∋ b} w_k q_σ(b | M_k)             (the likelihood of the Bayes step; MIX keeps it > 0)
# Inputs are the decision's worlds, their column labels, w and our reference mixture p only (no post-action data).

def style_dists_worlds(Ms, p, tau_frac=TAU_FRAC, mix=MIX):
    """[style_dists(M_k, p)] over the row-aligned world matrices: each style on world k's own columns."""
    return [style_dists(M, p, tau_frac, mix) for M in Ms]


def predict_worlds(P, qsk, cols, w, labels):
    """Membership-aware prediction over the union `labels`: q(label) = Σ_k w_k q̂_k(label), q̂_k = P @ qsk[k] over
    world k's columns cols[k]. Returns (q, [q̂_k])."""
    qk = [P @ qs for qs in qsk]
    pos = {lab: i for i, lab in enumerate(labels)}
    q = np.zeros(len(labels))
    for wk, c, qh in zip(w, cols, qk):
        for j, lab in enumerate(c):
            q[pos[lab]] += wk * qh[j]
    return q, qk


def lik_worlds(qsk, jk, w):
    """Per-style likelihood of the observed action: Σ_{k in jk} w_k qsk[k][:, jk[k]]; jk = {world k: column of the
    action in world k} over the worlds that contain it (the others contribute 0)."""
    return sum(w[k] * qsk[k][:, j] for k, j in jk.items())


class WorldPosterior(Posterior):
    """Posterior updated with the per-world likelihood lik_worlds (same lam / dtp log-space step as Posterior.update)."""

    def config(self):
        # marker so a *_mw state file is never silently shared with (or resumed by) a plain Posterior method
        return dict(super().config(), likelihood="per_world")

    def update_worlds(self, qsk, jk, w):
        lik = lik_worlds(qsk, jk, w)
        z = self.lam * (self.logp + np.log(lik)) if self.dtp else self.lam * self.logp + np.log(lik)
        self.logp = z - (z.max() + np.log(np.exp(z - z.max()).sum()))
        return lik


def load_world_posterior(path):
    """load_posterior for the *_mw methods (a WorldPosterior; same state file format)."""
    post = WorldPosterior()
    meta = dict(games=0, updates=0, skipped=0)
    if path and os.path.exists(path):
        d = json.load(open(path))
        if d["config"] != post.config():
            raise ValueError(f"posterior state {path} was written with {d['config']}")
        post.logp = np.array(d["logp"])
        meta = d["meta"]
    meta["games"] += 1
    return post, meta


def decide_worlds(Ms, cols, w, labels, p_ref, post):
    """Per-world style distributions qsk, the membership-aware q over `labels` and the per-world q̂_k; returns
    (qsk, q, qk, log fields)."""
    qsk = style_dists_worlds(Ms, p_ref, post.tau_frac, post.mix)
    q, qk = predict_worlds(post.probs(), qsk, cols, w, labels)
    rec = dict(qhat_mw=q.round(4).tolist(), cols=[list(c) for c in cols], qhat_k=[x.round(4).tolist() for x in qk])
    return qsk, q, qk, rec


def score_and_update_worlds(post, pend, jk, label, forced):
    """*_mw feedback: prequential scores BEFORE the update of the observed action under the membership-aware q
    (p_qhat_mw = Σ_k w_k q̂_k(b)), the averaged-matrix q-hat of the same posterior (p_qhat, as score_and_update), each
    style per world (p_style_mw, the likelihood) and on Abar (p_style), and the old predictor; then the Bayes step with
    the per-world likelihood. `label` is the action's label in the union (map_column's), jk = {k: column in world k}."""
    j = pend["labels"].index(label)
    qs, qhat, old, qmw = pend["qs"], pend["qhat"], pend["old"], pend["qhat_mw"]
    p_mw = sum(pend["w"][k] * pend["qhat_k"][k][jj] for k, jj in jk.items())
    rec = dict(col=label, j=j, forced=forced, n_cols=len(qhat), worlds_with=sorted(jk),
               p_qhat=round(float(qhat[j]), 6), p_qhat_mw=round(float(p_mw), 6),
               p_style={s: round(float(q[j]), 6) for s, q in zip(STYLES, qs)},
               p_old=round(float(old[j]), 6), hit_qhat=bool(int(qhat.argmax()) == j),
               hit_qhat_mw=bool(int(qmw.argmax()) == j))
    lik = post.update_worlds(pend["qsk"], jk, pend["w"])
    rec["p_style_mw"] = {s: round(float(v), 6) for s, v in zip(STYLES, lik)}
    rec["posterior"] = dict(zip(STYLES, post.probs().round(4).tolist()))
    return rec
