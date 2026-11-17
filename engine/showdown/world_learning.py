"""Pure expert selection and action-conditional credit for the *_mw live methods.

No network, sampling, posterior updates, or persistence. All inputs belong to the
decision BEFORE the observed action. World rows must already be aligned.
"""
import numpy as np

from exp3_live import expert_rewards

SEMANTICS = "per_world_security_action_conditional_credit_v2"


def normalized_weights(weights):
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("World weights must be finite, nonnegative, with positive total")
    return w / w.sum()


def pure_security_values(matrices, weights):
    """Pure restriction of agent form: value[a] = sum_k w[k] min_b M[k][a,b]."""
    w = normalized_weights(weights)
    if len(matrices) != len(w):
        raise ValueError("One matrix is required per world weight")
    return sum(wk * M.min(axis=1) for wk, M in zip(w, matrices))


def condition_on_action(cols, weights, predictions, label):
    """P(k | b) proportional to w[k] qhat_k(b), with exact action identity.

    The predictive probability and conditional weights use the saved pre-action
    style mixture. An absent action has zero likelihood; tera never falls back
    to plain. Return None for an observation outside positive predictive support.
    This posterior is for credit on this turn, not a carried joint particle filter.
    """
    w = normalized_weights(weights)
    if len(cols) != len(w) or len(predictions) != len(w):
        raise ValueError("Columns, predictions and weights must describe the same worlds")
    indices = {k: c.index(label) for k, c in enumerate(cols) if label in c and w[k] > 0}
    mass = np.zeros(len(w))
    for k, j in indices.items():
        mass[k] = w[k] * predictions[k][j]
    probability = float(mass.sum())
    if not np.all(np.isfinite(mass)) or np.any(mass < 0):
        raise ValueError("Action likelihoods must be finite and nonnegative")
    if probability <= 0:
        return None
    return dict(indices=indices, weights=mass / probability, probability=probability,
                support_weight=float(sum(w[k] for k in indices)))


def conditional_expert_credit(matrices, Q, conditioning, scale=400.0):
    """E[existing clipped surrogate reward | observed action], one Hedge update.

    Normalize/clip within each world BEFORE averaging, preserving the original
    reward definition and exact K=1 reduction. Raw expected utilities are exposed
    separately: clipping can hide utility differences and remains a limitation.
    """
    reward = np.zeros(len(Q))
    utility = np.zeros(len(Q))
    for k, j in conditioning["indices"].items():
        wk = conditioning["weights"][k]
        reward += wk * expert_rewards(matrices[k], Q, j, scale)
        utility += wk * (Q @ matrices[k][:, j])
    return reward, utility
