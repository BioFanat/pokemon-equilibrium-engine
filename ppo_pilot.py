"""PPO self-play pilot on poke-engine: how many decisions until the policy stops improving?

* Policy: 2-layer MLP (numpy forward in rollout workers, PyTorch for updates).
* Observation: full-information encoding of both sides (both teams visible; a pilot, not randbats).
* Actions: 14 slots = 4 moves, 4 tera-moves, 6 switch targets (team index); illegal ones masked.
* Self-play: both sides sample from the current policy; reward +1 win / -1 loss / 0 draw (turn cap).
* Logs (results/ppo_pilot.jsonl): decisions trained on, wall time, win rate vs random and vs MCTS.

Run:  .venv/bin/python ppo_pilot.py --minutes 25 --workers 12
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import time

import numpy as np

import pkbattle as pb

SPECIES = sorted(pb.SETS)
SP = {s: i for i, s in enumerate(SPECIES)}
STATUS = ["none", "burn", "freeze", "paralyze", "poison", "sleep", "toxic"]
WEATHER = ["none", "sun", "rain", "sand", "hail", "snow", "harshsun", "heavyrain"]
TERRAIN = ["none", "grassyterrain", "electricterrain", "mistyterrain", "psychicterrain"]
N_ACT, MAX_TURNS = 14, 200
MON_F = 3 + len(STATUS) + len(SPECIES) + 1


def _idx(lst, v):
    v = str(v).lower()
    return lst.index(v) if v in lst else 0


def _active(side) -> int:
    return int(str(side.active_index)[-1])


# ---------------------------------------------------------------- encoding

def enc_side(side) -> np.ndarray:
    act = _active(side)
    mons = np.zeros((6, MON_F), np.float32)
    for i, p in enumerate(side.pokemon):
        mons[i, 0] = p.hp / p.maxhp if p.maxhp else 0.0
        mons[i, 1] = p.hp > 0
        mons[i, 2] = i == act
        mons[i, 3 + _idx(STATUS, p.status)] = 1
        if p.id in SP:
            mons[i, 3 + len(STATUS) + SP[p.id]] = 1
        mons[i, -1] = p.terastallized
    sc = side.side_conditions
    extra = np.array([
        side.attack_boost, side.defense_boost, side.special_attack_boost, side.special_defense_boost,
        side.speed_boost, side.accuracy_boost, side.evasion_boost,
    ], np.float32) / 6.0
    extra2 = np.array([
        sc.stealth_rock, sc.spikes / 3, sc.toxic_spikes / 2, sc.sticky_web, sc.reflect > 0,
        sc.light_screen > 0, sc.aurora_veil > 0, sc.tailwind > 0, side.force_switch,
        any(p.terastallized for p in side.pokemon),
    ], np.float32)
    return np.concatenate([mons.ravel(), extra, extra2])


def obs(state, side: int) -> np.ndarray:
    me, opp = (state.side_one, state.side_two) if side == 1 else (state.side_two, state.side_one)
    field = np.zeros(len(WEATHER) + len(TERRAIN) + 1, np.float32)
    field[_idx(WEATHER, state.weather)] = 1
    field[len(WEATHER) + _idx(TERRAIN, state.terrain)] = 1
    field[-1] = state.trick_room
    return np.concatenate([enc_side(me), enc_side(opp), field])


OBS_DIM = 2 * (6 * MON_F + 17) + len(WEATHER) + len(TERRAIN) + 1


def opt_slots(side, opts) -> dict[int, str]:
    act = _active(side)
    moves = [m.id.lower() for m in side.pokemon[act].moves]
    names = [p.id.lower() for p in side.pokemon]
    slots = {}
    for o in opts:
        ol = o.lower()
        if ol in names:
            slots[8 + names.index(ol)] = o
        elif ol.endswith("-tera") and ol[:-5] in moves:
            slots[4 + moves.index(ol[:-5])] = o
        elif ol in moves:
            slots[moves.index(ol)] = o
    return slots


# ---------------------------------------------------------------- policy (numpy)

def forward(W, x):
    h = np.maximum(x @ W[0] + W[1], 0)
    h = np.maximum(h @ W[2] + W[3], 0)
    return h @ W[4] + W[5], (h @ W[6] + W[7])[:, 0]


def make_opponent(name: str):
    if name == "random":
        return pb.random_policy
    if name.startswith("mcts"):
        return pb.mcts_policy(int(name[4:]))
    if name == "nash":
        import nash
        return nash.nash_policy("rm_plus", 50, 200)
    raise ValueError(name)


def run_game(W, seed: int, opponent=None, policy_side: int = 1):
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    state = pb.random_state(rng, "hand")   # obs one-hot assumes the 12-species pool
    traj = {1: [], 2: []}
    winner, turns = 0, MAX_TURNS
    for t in range(MAX_TURNS):
        o1, o2 = pb.options(state)
        chosen = {}
        for side, opts in ((1, o1), (2, o2)):
            if opponent is not None and side != policy_side:
                chosen[side] = opponent(state, opts, side, rng)
                continue
            sobj = state.side_one if side == 1 else state.side_two
            slots = opt_slots(sobj, opts)
            if len(slots) <= 1:
                chosen[side] = next(iter(slots.values())) if slots else opts[0]
                continue
            x = obs(state, side)
            mask = np.zeros(N_ACT, bool); mask[list(slots)] = True
            logits, v = forward(W, x[None])
            l = np.where(mask, logits[0], -1e9)
            p = np.exp(l - l.max()); p /= p.sum()
            a = int(nrng.choice(N_ACT, p=p))
            traj[side].append((x, mask, a, math.log(p[a] + 1e-12), float(v[0])))
            chosen[side] = slots[a]
        state = pb.step(state, chosen[1], chosen[2], rng)
        a1, a2 = pb.alive(state.side_one), pb.alive(state.side_two)
        if a1 == 0 or a2 == 0:
            winner, turns = (0 if a1 == a2 else (1 if a1 else 2)), t + 1
            break
    return traj, winner, turns


def worker_selfplay(args):
    W, seeds = args
    out = []
    for s in seeds:
        traj, winner, turns = run_game(W, s)
        for side in (1, 2):
            if traj[side]:
                r = 0.0 if winner == 0 else (1.0 if winner == side else -1.0)
                out.append((traj[side], r))
    return out


def worker_eval(args):
    W, seeds, opp_name = args
    opp = make_opponent(opp_name)
    score = 0.0
    for s in seeds:
        side = 1 + s % 2
        _, winner, _ = run_game(W, s, opponent=opp, policy_side=side)
        score += 0.5 if winner == 0 else float(winner == side)
    return score, len(seeds)


# ---------------------------------------------------------------- learner (torch)

def build_net(in_dim=OBS_DIM):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1, self.l2 = nn.Linear(in_dim, 256), nn.Linear(256, 256)
            self.pi, self.v = nn.Linear(256, N_ACT), nn.Linear(256, 1)

        def forward(self, x):
            import torch.nn.functional as F
            h = F.relu(self.l2(F.relu(self.l1(x))))
            return self.pi(h), self.v(h).squeeze(-1)

    return Net()


def export(net):
    ps = [net.l1.weight, net.l1.bias, net.l2.weight, net.l2.bias, net.pi.weight, net.pi.bias, net.v.weight, net.v.bias]
    return [p.detach().numpy().T.copy() if p.ndim == 2 else p.detach().numpy().copy() for p in ps]


def gae(episodes, gamma=0.995, lam=0.95):
    X, M, A, LP, ADV, RET = [], [], [], [], [], []
    for steps, r in episodes:
        vals = [s[4] for s in steps]
        T, adv, last = len(steps), np.zeros(len(steps), np.float32), 0.0
        for t in reversed(range(T)):
            nxt = vals[t + 1] if t + 1 < T else 0.0
            rew = r if t == T - 1 else 0.0
            delta = rew + gamma * nxt - vals[t]
            last = delta + gamma * lam * last
            adv[t] = last
        for t, s in enumerate(steps):
            X.append(s[0]); M.append(s[1]); A.append(s[2]); LP.append(s[3])
        ADV.append(adv); RET.append(adv + np.array(vals, np.float32))
    return (np.stack(X), np.stack(M), np.array(A), np.array(LP, np.float32),
            np.concatenate(ADV), np.concatenate(RET))


def ppo_update(net, opt, batch, epochs=4, mb=4096, clip=0.2, ent_coef=0.01, vf_coef=0.5):
    import torch
    X, M, A, LP, ADV, RET = (torch.as_tensor(b) for b in batch)
    ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
    n, stats = len(X), []
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, mb):
            idx = perm[i:i + mb]
            logits, v = net(X[idx])
            logits = logits.masked_fill(~M[idx], -1e9)
            logp_all = torch.log_softmax(logits, -1)
            logp = logp_all.gather(1, A[idx, None]).squeeze(1)
            ratio = torch.exp(logp - LP[idx])
            pg = -torch.min(ratio * ADV[idx], ratio.clamp(1 - clip, 1 + clip) * ADV[idx]).mean()
            p_all = logp_all.exp()
            ent = -(p_all * logp_all.masked_fill(~M[idx], 0)).sum(-1).mean()
            vl = ((v - RET[idx]) ** 2).mean()
            loss = pg + vf_coef * vl - ent_coef * ent
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            stats.append((pg.item(), vl.item(), ent.item()))
    return np.mean(stats, axis=0)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=25)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--games-per-worker", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-games", type=int, default=96)
    ap.add_argument("--out", default="results/ppo_pilot.jsonl")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(2)
    torch.manual_seed(0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    net = build_net()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    pool = mp.get_context("spawn").Pool(args.workers)
    log = open(args.out, "w")
    t0, decisions, games, it, seed = time.time(), 0, 0, 0, 10_000

    def evaluate(opp_names):
        W, res = export(net), {}
        for name in opp_names:
            per = max(1, args.eval_games // args.workers)
            jobs = [(W, list(range(10**8 + k * per, 10**8 + (k + 1) * per)), name) for k in range(args.workers)]
            sc = pool.map(worker_eval, jobs)
            res[name] = sum(s for s, _ in sc) / sum(n for _, n in sc)
        return res

    def record(extra):
        rec = {"iter": it, "decisions": decisions, "games": games, "wall_s": round(time.time() - t0, 1), **extra}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()

    record({"eval": evaluate(["random", "mcts200"])})
    while time.time() - t0 < args.minutes * 60:
        it += 1
        W = export(net)
        jobs = []
        for _ in range(args.workers):
            jobs.append((W, list(range(seed, seed + args.games_per_worker))))
            seed += args.games_per_worker
        tr = time.time()
        episodes = [e for chunk in pool.map(worker_selfplay, jobs) for e in chunk]
        t_roll = time.time() - tr
        batch = gae(episodes)
        n_dec = len(batch[0])
        tu = time.time()
        pg, vl, ent = ppo_update(net, opt, batch)
        decisions += n_dec; games += args.workers * args.games_per_worker
        extra = {"roll_s": round(t_roll, 1), "upd_s": round(time.time() - tu, 1),
                 "dec_per_s": round(n_dec / t_roll), "pg": round(float(pg), 4), "vloss": round(float(vl), 4),
                 "entropy": round(float(ent), 3)}
        if it % args.eval_every == 0:
            extra["eval"] = evaluate(["random", "mcts200"])
        record(extra)
    record({"final_eval": evaluate(["random", "mcts200", "mcts2000", "nash"])})
    torch.save(net.state_dict(), args.out.replace(".jsonl", ".pt"))
    pool.close()


if __name__ == "__main__":
    main()
