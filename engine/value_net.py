"""Value net V(s) ~ E[z | s] (side one's outcome), trained by supervised regression on value_data.

V(s) = tanh(0.5 * (g(x) - g(x_mirror))) with g an MLP in -> 256 -> 256 -> 1 (ReLU) and
x_mirror = x[MIRROR_PERM] the side-swapped encoding, so V(mirror s) = -V(s) by construction
(dropout 0.3 in training only; inference has no dropout and is exactly antisymmetric).

  .venv/bin/python value_net.py [data.npz]  # train -> results/value/vnet.npz + vnet_metrics.json
Inference (`load`, `value_many`) is pure numpy so worker processes never import torch.
"""
from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

import features

OUT = Path(__file__).parent / "results" / "value"
HID, MAX_MIN, PATIENCE, EVAL_EVERY, LR, DROP = 256, 5.0, 5, 100, 1e-4, 0.3


# ---------------------------------------------------------------- numpy inference

@lru_cache(maxsize=None)
def load(path: str = str(OUT / "vnet.npz")) -> dict:
    return dict(np.load(path))


def _g(W: dict, X: np.ndarray) -> np.ndarray:
    h = np.maximum(X @ W["W1"] + W["b1"], 0)
    h = np.maximum(h @ W["W2"] + W["b2"], 0)
    return (h @ W["W3"] + W["b3"])[:, 0]


def value_x(X: np.ndarray) -> np.ndarray:
    W = load()
    return np.tanh(0.5 * (_g(W, X) - _g(W, X[:, features.MIRROR_PERM])))


def value_many(states) -> np.ndarray:
    """V for each state, side-one perspective, in (-1, 1)."""
    return value_x(features.encode_many(list(states)))


# ---------------------------------------------------------------- training

def train(data: str = str(OUT / "data.npz"), seed: int = 0):
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    d = np.load(data)
    X, z, h, gid = d["X"], d["z"].astype(np.float32), d["h"], d["game_id"]
    rng = np.random.default_rng(seed)
    games = np.unique(gid)
    val_games = rng.choice(games, size=max(1, len(games) // 10), replace=False)
    va = np.isin(gid, val_games)
    tr = ~va
    perm = torch.as_tensor(features.MIRROR_PERM)
    g = torch.nn.Sequential(torch.nn.Linear(X.shape[1], HID), torch.nn.ReLU(), torch.nn.Dropout(DROP),
                            torch.nn.Linear(HID, HID), torch.nn.ReLU(), torch.nn.Dropout(DROP),
                            torch.nn.Linear(HID, 1))
    V = lambda xb: torch.tanh(0.5 * (g(xb) - g(xb[:, perm])))[:, 0]
    opt = torch.optim.Adam(g.parameters(), lr=LR, weight_decay=1e-4)
    Xt, zt = torch.as_tensor(X[tr]), torch.as_tensor(z[tr])
    Xv, zv = torch.as_tensor(X[va]), torch.as_tensor(z[va])

    def val_loss():
        g.eval()
        with torch.no_grad():
            out = float(((V(Xv) - zv) ** 2).mean())
        g.train()
        return out

    # positions of a game share teams and label, so the net overfits within an epoch:
    # evaluate every EVAL_EVERY steps and keep the best snapshot (patience counted in evals)
    best, best_state, bad, t0, step, ep = float("inf"), None, 0, time.perf_counter(), 0, 0
    while time.perf_counter() - t0 < 60 * MAX_MIN and bad < PATIENCE:
        ep += 1
        order = torch.randperm(len(Xt))
        for k in range(0, len(Xt), 512):
            idx = order[k:k + 512]
            loss = ((V(Xt[idx]) - zt[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % EVAL_EVERY:
                continue
            vl = val_loss()
            print(f"step {step} (epoch {ep}): val MSE {vl:.4f} ({time.perf_counter() - t0:.0f}s)", flush=True)
            if vl < best - 1e-4:
                best, bad, best_step = vl, 0, step
                best_state = {k: v.clone() for k, v in g.state_dict().items()}
            else:
                bad += 1
            if bad >= PATIENCE or time.perf_counter() - t0 > 60 * MAX_MIN:
                break
    g.load_state_dict(best_state)
    sd = {k: v.numpy() for k, v in g.state_dict().items()}
    W = dict(W1=sd["0.weight"].T, b1=sd["0.bias"], W2=sd["3.weight"].T, b2=sd["3.bias"],
             W3=sd["6.weight"].T, b3=sd["6.bias"])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "vnet.npz", **W)
    load.cache_clear()

    # diagnostics on val: net vs the engine heuristic leaf (sign only), decisive games
    v = value_x(X[va])
    zva, hva = z[va], h[va]
    dec = zva != 0
    sub = rng.choice(np.flatnonzero(dec), size=min(2000, int(dec.sum())), replace=False)
    sign_acc = lambda p, idx: float((np.sign(p[idx]) == zva[idx]).mean())
    m = dict(data=data, train_positions=int(tr.sum()), val_positions=int(va.sum()), val_games=int(len(val_games)),
             epochs=ep, steps=step, best_step=best_step, train_s=round(time.perf_counter() - t0, 1),
             val_mse=float(((v - zva) ** 2).mean()), val_mse_const0=float((zva ** 2).mean()),
             net_sign_acc_decisive=sign_acc(v, np.flatnonzero(dec)),
             subsample_n=int(len(sub)), net_sign_acc_sub=sign_acc(v, sub),
             heuristic_sign_acc_sub=sign_acc(hva, sub),
             heuristic_zero_frac_sub=float((hva[sub] == 0).mean()))
    (OUT / "vnet_metrics.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))
    return m


if __name__ == "__main__":
    import sys
    train(*sys.argv[1:2])                     # optional: path to a data.npz
