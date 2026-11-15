"""PSRO and RL best-response exploitability on poke-engine (full information, 12-species hand pool).

* Meta-game: the population's payoff matrix M[i, j] = P(i beats j) - 1/2 (draw = half a win), estimated
  from paired games; antisymmetric, so the meta-game is a symmetric zero-sum game with value 0.
* Meta-Nash: a mixture x over members with min_j (x^T M)_j = 0 (no member beats it on average);
  solved by nash.lp (exact) or nash.rm_plus.
* Best response (BR): PPO policy trained against opponents sampled per episode from x
  (approximate; warm-started from the previous BR or results/ppo_pilot.pt).
* Exploitability lower bound: win rate of an approximate BR against a target minus 1/2. Any policy's
  win rate is <= the true best response's, so this LOWER-bounds exploitability (in win-rate units;
  x2 in +-1 payoff units), as in Timbers et al. 2022 (approximate best responses).
* gray_box (rl_br config): the BR net's observation also gets the opponent's move distribution q for the
  turn (the exact q a FoulplayAgent then samples from), as 14 slot probabilities after ppo_pilot.obs.

Games use the same state generator and caps as the PPO pilot: pb.random_state(rng, "hand") (the obs
one-hot assumes the 12-species pool) and ppo_pilot.MAX_TURNS = 200 (a draw at the cap). The matrix bots
in results/experiments/ were run on randbats teams with a 300-turn cap; numbers here are not comparable.

Hidden information (out of scope): build hidden_agents Agent-form bots in make_agent and swap
ppo_pilot.obs for a belief observation (own view + posterior over opponent sets) in NetAgent.

  .venv/bin/python psro.py --config first --dry-run       # plan + cost, plays nothing
  .venv/bin/python psro.py --config tiny --local-workers 4    # fast local iteration, cheap bots only
  .venv/bin/python psro.py --config smoke --local-workers 4
  .venv/bin/python psro.py --mode rl_br --target hedge_experts --config first
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

import ppo_pilot as pp
import pkbattle as pb

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("PSRO_ROOT", HERE))         # relative checkpoint paths resolve here (Modal too)
MAX_LOCAL_WORKERS = 6
EXPLOIT_LABEL = ("lower bound on exploitability: approximate RL best-response win rate - 0.5 "
                 "(Timbers et al. 2022 style; the true best response can only do better)")
CELL_SEED, TRAIN_SEED, EXPLOIT_SEED = 10**9, 2 * 10**9, 3 * 10**9   # disjoint seed ranges

DEFAULT = dict(
    run="psro_first", bots=["nash:rm+:0:1000:argmax", "nash:rm+:0:1000", "safest:0", "mcts_it:2000"],
    iters=3, br_decisions=200_000, games_per_cell=60, games_per_update=64, init="results/ppo_pilot.pt",
    solver="lp", lr=3e-4, rl_br_targets=["nash:rm+:0:1000", "hedge_experts"], rl_br_decisions=250_000,
    exploit_games=200, seed=0, train_pool="all")
# train_pool: "all" = BR trains against the full meta-Nash mixture; "cheap" = matrix bots / hedge (≈74 ms per move)
# are dropped from the BR's training opponents (mixture renormalised over the rest; they stay in the meta
# matrix and in evaluation). "cheap" is ~10x cheaper but the BR is then a best response to a different mixture.
SLOW_KINDS = ("nash", "safest", "hedge_experts", "vnet", "lbr")
PRESETS = {
    # fast iteration (seconds to a few minutes, cents): cheap bots only, no matrix bots anywhere
    "tiny": dict(run="psro_tiny", bots=["random", "mcts_it:200", "mcts_it:2000"], iters=1, br_decisions=10_000,
                 games_per_cell=10, rl_br_targets=["mcts_it:2000"], rl_br_decisions=10_000, exploit_games=40),
    # matrix bots in the meta matrix, but BRs train on cheap opponents only; exercises every code path
    "quick": dict(run="psro_quick", iters=1, br_decisions=20_000, games_per_cell=10, train_pool="cheap",
                  rl_br_targets=["hedge_experts"], rl_br_decisions=20_000, exploit_games=40),
    "smoke": dict(run="psro_smoke", iters=1, br_decisions=60_000, games_per_cell=30, rl_br_targets=["hedge_experts"],
                  rl_br_decisions=60_000, exploit_games=60),
    "first": dict(),
    "full": dict(run="psro_full", iters=8, br_decisions=2_000_000, games_per_cell=200,
                 rl_br_decisions=2_000_000, exploit_games=1000),
}


class BudgetExceeded(RuntimeError):
    """Raised by a map_fn (modal_psro's budget guard) to stop a run cleanly; results so far are saved."""


def load_config(name_or_path: str | None) -> dict:
    if not name_or_path:
        return dict(DEFAULT)
    if name_or_path in PRESETS:
        return {**DEFAULT, **PRESETS[name_or_path]}
    return {**DEFAULT, **json.loads(Path(name_or_path).read_text())}


# ---------------------------------------------------------------- agents

class BotAgent:
    """A fixed bot from experiments.make_agent (matrix bots, MCTS, random)."""

    def __init__(self, spec):
        import experiments
        self.pol = experiments.make_agent(spec)

    def act(self, state, opts, side, rng):
        return self.pol(state, opts, side, rng)

    def observe(self, opp_move):
        pass


class FoulplayAgent:
    """experiments.make_agent's foulplay_it:N / foulplay_ms:N bot, split so a gray-box opponent can read q:
    prepare() runs the turn's MCTS and caches (own, q); act() samples from that cached q (same calls and
    rng draws as the experiments.py policy when nothing is prepared)."""

    def __init__(self, spec):
        kind, *a = spec.split(":")
        a = [x for x in a if x not in ("legacy", "argmax")]
        n = int(a[0])
        self.ms, self.it = (0, n) if kind == "foulplay_it" else (n, 0)
        self.cache = None

    def prepare(self, state, opts, side):
        import nash
        if len(opts) == 1:
            self.cache = (opts, np.ones(1))
        else:
            self.cache = nash.foulplay_strategy(state, side, self.ms, self.it)
        return self.cache

    def act(self, state, opts, side, rng):
        if len(opts) == 1:
            self.cache = None
            return opts[0]
        own, q = self.cache if self.cache is not None else self.prepare(state, opts, side)
        self.cache = None
        return rng.choices(own, weights=q)[0]

    def observe(self, opp_move):
        pass


class HedgeAgent:
    """exp3_live.Learner as a member: plays p = Σ_e w_e q_e over the 3 experts, then Hedge-updates on the
    opponent's realized move. Weights start uniform every game (a stationary member; cross-battle
    adaptation as in exp3_live.main is not modelled). Forced turns do not update, as in exp3_live."""

    def __init__(self):
        import exp3_live
        self.L, self.pending = exp3_live.Learner(), None

    def act(self, state, opts, side, rng):
        self.pending = None
        if len(opts) == 1:
            return opts[0]
        o1, o2 = pb.options(state)
        own, A, Q, p = self.L.plan(state, side)
        self.pending = (A, Q, o2 if side == 1 else o1)
        return rng.choices(own, weights=p)[0]

    def observe(self, opp_move):
        if self.pending:
            A, Q, opp = self.pending
            self.L.update(A, Q, opp.index(opp_move))


class NetAgent:
    """Samples from the PPO policy (numpy forward); records (x, mask, a, logp, v) per non-forced decision,
    the tuple format ppo_pilot.gae expects (same logic as ppo_pilot.run_game).
    Gray box (input width OBS_DIM + N_ACT): play_game sets opp_q = the opponent's prepared (opts, q) before
    act, and x = obs ++ q laid out on the opponent's 14 slots."""

    def __init__(self, W, seed):
        self.W, self.nrng, self.traj = W, np.random.default_rng(seed), []
        self.gray, self.opp_q = W[0].shape[0] == pp.OBS_DIM + pp.N_ACT, None

    def act(self, state, opts, side, rng):
        if self.gray and self.opp_q is None:
            raise RuntimeError("gray-box NetAgent.act called without opp_q (play_game must prepare the opponent)")
        try:
            return self._act(state, opts, side)
        finally:
            self.opp_q = None

    def _opp_qv(self, state, side) -> np.ndarray:
        opp_opts, q = self.opp_q
        qv = np.zeros(pp.N_ACT, np.float32)
        if list(opp_opts) == ["none"]:      # opponent has no move this turn (we switch after a KO/pivot): all-zero qv
            return qv
        for slot, o in pp.opt_slots(state.side_two if side == 1 else state.side_one, opp_opts).items():
            qv[slot] = q[opp_opts.index(o)]
        # An option opt_slots can't place (e.g. struggle; none seen in ~600 checked decisions) keeps its mass out of
        # qv (sum < 1) rather than crashing a long run; the black-box net can't represent such options either.
        return qv

    def _act(self, state, opts, side):
        slots = pp.opt_slots(state.side_one if side == 1 else state.side_two, opts)
        if len(slots) <= 1:
            return next(iter(slots.values())) if slots else opts[0]
        x = pp.obs(state, side)
        if self.gray:
            x = np.concatenate([x, self._opp_qv(state, side)])
        mask = np.zeros(pp.N_ACT, bool); mask[list(slots)] = True
        logits, v = pp.forward(self.W, x[None])
        l = np.where(mask, logits[0], -1e9)
        p = np.exp(l - l.max()); p /= p.sum()
        a = int(self.nrng.choice(pp.N_ACT, p=p))
        self.traj.append((x, mask, a, math.log(p[a] + 1e-12), float(v[0])))
        return slots[a]

    def observe(self, opp_move):
        pass


_W_CACHE: dict[str, list] = {}


def load_weights(path: str) -> list:
    """Numpy weights for a net member: <path>.npz if present (workers need no torch), else the torch .pt."""
    if path not in _W_CACHE:
        p = ROOT / path
        npz = p.with_suffix(".npz")
        if npz.exists():
            z = np.load(npz)
            _W_CACHE[path] = [z[f"arr_{k}"] for k in range(len(z.files))]
        else:
            import torch
            sd = torch.load(p, map_location="cpu")
            net = pp.build_net(sd["l1.weight"].shape[1]); net.load_state_dict(sd)
            _W_CACHE[path] = pp.export(net)
    return _W_CACHE[path]


def member_from_spec(spec: str) -> dict:
    """'net:<path>' -> net member; anything else (bot spec or 'hedge_experts') -> bot member."""
    return {"kind": "net", "path": spec[4:]} if spec.startswith("net:") else {"kind": "bot", "spec": spec}


def make_agent(member: dict, seed: int):
    if member["kind"] == "net":
        return NetAgent(load_weights(member["path"]), seed)
    if member["spec"].split(":")[0] in ("foulplay_it", "foulplay_ms"):
        return FoulplayAgent(member["spec"])
    return HedgeAgent() if member["spec"] == "hedge_experts" else BotAgent(member["spec"])


def name(member: dict) -> str:
    return member["spec"] if member["kind"] == "bot" else "net:" + member["path"]


# ---------------------------------------------------------------- games and parallel tasks

def play_game(a1, a2, seed: int) -> tuple[int, int]:
    """One game on the PPO pilot's state distribution. Teams come from random.Random(seed) before any
    agent draws, so both seats of a pair see the same teams. Returns (winner 1/2/0, turns)."""
    rng = random.Random(seed)
    state = pb.random_state(rng, "hand")
    for t in range(1, pp.MAX_TURNS + 1):
        o1, o2 = pb.options(state)
        for me, other, s in ((a1, a2, 1), (a2, a1, 2)):
            if getattr(me, "gray", False):                 # gray box: hand me the q the opponent will sample from
                if not hasattr(other, "prepare"):
                    raise TypeError(f"gray-box net needs an opponent with prepare(); got {type(other).__name__}")
                me.opp_q = other.prepare(state, o2 if s == 1 else o1, 3 - s)
        m1, m2 = a1.act(state, o1, 1, rng), a2.act(state, o2, 2, rng)
        a1.observe(m2); a2.observe(m1)
        state = pb.step(state, m1, m2, rng)
        n1, n2 = pb.alive(state.side_one), pb.alive(state.side_two)
        if n1 == 0 or n2 == 0:
            return (0 if n1 == n2 else (1 if n1 else 2)), t
    return 0, pp.MAX_TURNS


def _score(winner: int, side: int) -> float:
    return 0.5 if winner == 0 else float(winner == side)


def eval_task(jobs):
    """jobs = [(tag, member_a, member_b, seed)] or [(tag, member_a, member_b, seed, a_side)]: a 4-tuple is a
    pair (a on side one, then side two, same seed); a 5-tuple is that one game. Returns
    [(tag, mean score of a in [0, 1], turns)] per job."""
    out = []
    for tag, ma, mb, seed, *side in jobs:
        s, turns = 0.0, 0
        for a_side in (side or (1, 2)):
            a, b = make_agent(ma, 2 * seed + a_side), make_agent(mb, 2 * seed + 3 - a_side)
            w, t = play_game(a, b, seed) if a_side == 1 else play_game(b, a, seed)
            s += _score(w, a_side); turns += t
        out.append((tag, s / (len(side) or 2), turns))
    return out


def play_pairs(jobs, map_fn=map) -> list:
    """jobs = [(tag, a, b, seed)] pairs, played as one map input PER GAME (2 per pair) so the slowest map
    input is one game, not a chunk; returns [(tag, pair score, turns)] in job order (pairs regrouped)."""
    games = [(k, a, b, seed, side) for k, (_, a, b, seed) in enumerate(jobs) for side in (1, 2)]
    res = map_fn(eval_task, [[g] for g in games])
    acc = {}
    for (k, *_), r in zip(games, res):
        _, sc, t = r[0]
        s0, t0 = acc.get(k, (0.0, 0))
        acc[k] = (s0 + sc / 2, t0 + t)
    return [(jobs[k][0], *acc[k]) for k in range(len(jobs))]


def rollout_task(args):
    """args = (W, [(opponent member, seed)]): the learner plays side 1 + seed % 2. Returns its episodes
    [(trajectory, reward +1/-1/0)] for ppo_pilot.gae, plus on-policy score and game counts."""
    W, jobs = args
    episodes, score, turns = [], 0.0, 0
    for opp, seed in jobs:
        side = 1 + seed % 2
        me, other = NetAgent(W, seed), make_agent(opp, seed + 1)
        w, t = play_game(me, other, seed) if side == 1 else play_game(other, me, seed)
        if me.traj:
            episodes.append((me.traj, 0.0 if w == 0 else (1.0 if w == side else -1.0)))
        score += _score(w, side); turns += t
    return dict(episodes=episodes, score=score, games=len(jobs), turns=turns)


def _chunks(xs: list, n: int) -> list[list]:
    n = max(1, min(n, len(xs)))
    return [xs[k::n] for k in range(n)] if xs else []


# ---------------------------------------------------------------- population and meta-game

class Population:
    """Members ({"kind": "bot", "spec"} or {"kind": "net", "path"}), meta matrix M = win rate - 1/2
    (antisymmetric, M[j, i] = -M[i, j]) and per-cell game counts N."""

    def __init__(self, members=None):
        self.members, self.M, self.N = [], np.zeros((0, 0)), np.zeros((0, 0), int)
        for m in members or []:
            self.add(m)

    def __len__(self):
        return len(self.members)

    def add(self, member: dict) -> int:
        n = len(self.members)
        self.members.append(member)
        self.M = np.pad(self.M, ((0, 1), (0, 1)))
        self.N = np.pad(self.N, ((0, 1), (0, 1)))
        return n

    def record(self, i: int, j: int, winrate: float, games: int):
        self.M[i, j], self.M[j, i] = winrate - 0.5, 0.5 - winrate
        self.N[i, j] = self.N[j, i] = games

    def evaluate_pair(self, i: int, j: int, n_games: int, seed: int, map_fn=map, n_chunks: int = 1):
        """Win rate of member i vs j over n_games (n_games // 2 paired seeds, both seats)."""
        jobs = [(None, self.members[i], self.members[j], seed + k) for k in range(n_games // 2)]
        res = play_pairs(jobs, map_fn)
        return sum(r[1] for r in res) / len(res), 2 * len(res)

    def to_json(self) -> dict:
        return dict(members=self.members, names=[name(m) for m in self.members],
                    M=self.M.round(4).tolist(), N=self.N.tolist())


def cell_seed(seed: int, i: int, j: int) -> int:
    i, j = min(i, j), max(i, j)
    return CELL_SEED + seed * 10**8 + (i * 100 + j) * 10**4


def _slow(member: dict) -> bool:
    return member["kind"] == "bot" and member["spec"].split(":")[0] in SLOW_KINDS


def _cache_path() -> Path:
    return ROOT / "results/psro/meta_cache.json"


def fill_meta(pop: Population, idxs, n_games: int, map_fn=map, seed: int = 0, n_chunks: int = 4):
    """Fill every unplayed cell touching idxs in ONE parallel map. Bot-vs-bot cells are cached across runs in
    results/psro/meta_cache.json (key: both specs, seed, games), so iterating never replays the bot table."""
    cache = json.loads(_cache_path().read_text()) if _cache_path().exists() else {}
    key = lambda i, j: f"{name(pop.members[i])}|{name(pop.members[j])}|{seed}|{n_games}"
    cells, jobs = [], []
    for i in idxs:
        for j in range(len(pop)):
            if j == i or pop.N[i, j] or (j, i) in cells or (i, j) in cells:
                continue
            if pop.members[i]["kind"] == pop.members[j]["kind"] == "bot":
                if key(i, j) in cache:
                    pop.record(i, j, *cache[key(i, j)]); continue
                if key(j, i) in cache:
                    w, g = cache[key(j, i)]; pop.record(i, j, 1 - w, g); continue
            cells.append((i, j))
            jobs += [((i, j), pop.members[i], pop.members[j], cell_seed(seed, i, j) + k) for k in range(n_games // 2)]
    res = play_pairs(jobs, map_fn) if jobs else []
    for (i, j) in cells:
        sc = [r[1] for r in res if r[0] == (i, j)]
        pop.record(i, j, sum(sc) / len(sc), 2 * len(sc))
        if pop.members[i]["kind"] == pop.members[j]["kind"] == "bot":
            cache[key(i, j)] = [sum(sc) / len(sc), 2 * len(sc)]
    if cells:
        _write(_cache_path(), cache)


def extend_meta(pop: Population, new_idx: int, n_games: int, map_fn=map, seed: int = 0, n_chunks: int = 4):
    """Fill only row/column new_idx (cells already played are skipped), all cells in one parallel map."""
    fill_meta(pop, [new_idx], n_games, map_fn, seed, n_chunks)


def meta_solve(M: np.ndarray, method: str = "lp", iters: int = 10_000) -> np.ndarray:
    """Meta-Nash mixture of the symmetric zero-sum meta-game (nash.lp exact, or nash.rm_plus average)."""
    import nash
    if len(M) == 1:
        return np.ones(1)
    x = nash.lp(M)[0] if method == "lp" else nash.rm_plus(M, iters)[0]
    x = np.clip(x, 0, None)
    return x / x.sum()


def meta_nashconv(M: np.ndarray, x: np.ndarray) -> float:
    """Within-population NashConv of x: max_i (M x)_i - min_j (x^T M)_j (0 at the meta-Nash)."""
    import nash
    return nash.exploitability(M, x, x)


# ---------------------------------------------------------------- best response and exploitability

def train_best_response(pop: Population, mix, budget_decisions: int, init: str | None = None, map_fn=map,
                        n_chunks: int = 4, games_per_update: int = 64, lr: float = 3e-4, seed_base: int = 0,
                        out: str | None = None, log=None, train_pool: str = "all", games_per_input: int = 1,
                        eval_every=None, on_checkpoint=None, gray: bool = False) -> dict:
    """PPO best response to the mixture `mix` over pop.members: each episode's opponent is sampled from
    mix, rollouts go through map_fn, updates reuse ppo_pilot.gae / ppo_update. The checkpoint
    (<out>.pt + <out>.npz) is written even when a BudgetExceeded stops training early.
    games_per_input: games per map input (contiguous chunks, so episodes regroup in game order).
    eval_every: decisions between checkpoints (int) or a list of decision thresholds; at each one the net is
    saved to <out>_ckpt<k>.npz (a fresh path: workers cache weights by path) and
    on_checkpoint(path, decisions, games) is called.
    gray: input width OBS_DIM + N_ACT (opponent q appended); a plain-width init gets N_ACT zero input columns,
    so the initial policy equals the init net's."""
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(seed_base)
    net = pp.build_net(pp.OBS_DIM + pp.N_ACT if gray else pp.OBS_DIM)
    if init:
        sd = torch.load(ROOT / init, map_location="cpu")
        w = sd["l1.weight"].shape[1]
        if gray and w == pp.OBS_DIM:
            sd["l1.weight"] = torch.cat([sd["l1.weight"], sd["l1.weight"].new_zeros(sd["l1.weight"].shape[0], pp.N_ACT)], 1)
        elif not gray and w != pp.OBS_DIM:
            raise ValueError(f"init {init} has input width {w}; a non-gray BR needs {pp.OBS_DIM}")
        net.load_state_dict(sd)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    mix = np.asarray(mix, float) / np.sum(mix)
    if train_pool == "cheap":
        keep = np.array([not _slow(m) for m in pop.members], float)
        if (mix * keep).sum() > 0:
            mix = mix * keep / (mix * keep).sum()
        elif keep.sum() > 0:                      # meta-Nash is all slow bots: train on the cheap members uniformly
            mix = keep / keep.sum()
    rng, next_seed = np.random.default_rng(seed_base), TRAIN_SEED + seed_base * 10**7
    decisions = games = 0
    history, t0 = [], time.time()
    marks = (sorted(eval_every) if isinstance(eval_every, (list, tuple))
             else list(range(eval_every, budget_decisions, eval_every)) if eval_every else [])
    n_ckpt = 0
    try:
        while decisions < budget_decisions:
            W = pp.export(net)
            picks = rng.choice(len(mix), size=games_per_update, p=mix)
            jobs = [(pop.members[k], next_seed + g) for g, k in enumerate(picks)]
            next_seed += games_per_update
            gpi = max(1, games_per_input)                             # 1 = one game per input (no chunk stragglers)
            outs = map_fn(rollout_task, [(W, jobs[i:i + gpi]) for i in range(0, len(jobs), gpi)])
            episodes = [e for o in outs for e in o["episodes"]]
            g, sc = sum(o["games"] for o in outs), sum(o["score"] for o in outs)
            games += g
            if not episodes:
                continue
            batch = pp.gae(episodes)
            pg, vl, ent = pp.ppo_update(net, opt, batch)
            decisions += len(batch[0])
            rec = dict(decisions=decisions, games=games, winrate=round(sc / g, 4), wall_s=round(time.time() - t0, 1),
                       pg=round(float(pg), 4), vloss=round(float(vl), 4), entropy=round(float(ent), 3))
            history.append(rec)
            if log:
                log(rec)
            if marks and decisions >= marks[0] and decisions < budget_decisions and out and on_checkpoint:
                while marks and decisions >= marks[0]:
                    marks.pop(0)
                n_ckpt += 1
                ck = f"{out}_ckpt{n_ckpt}"
                (ROOT / ck).parent.mkdir(parents=True, exist_ok=True)
                np.savez(ROOT / f"{ck}.npz", *pp.export(net))
                on_checkpoint(f"{ck}.npz", decisions, games)
    finally:
        if out:
            (ROOT / out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), ROOT / f"{out}.pt")
            np.savez(ROOT / f"{out}.npz", *pp.export(net))
    return dict(path=f"{out}.pt" if out else None, decisions=decisions, games=games, history=history,
                train_mix=mix.round(4).tolist())


def exploitability_estimate(br_path: str, target, n_games: int, map_fn=map, n_chunks: int = 4,
                            seed: int = EXPLOIT_SEED) -> dict:
    """Win rate of the BR net at br_path vs target (a member dict, or a list of (member, weight) whose
    opponent is sampled per pair) on held-out seeds, minus 1/2: a LOWER bound on exploitability."""
    br = {"kind": "net", "path": br_path}
    pairs = n_games // 2
    if isinstance(target, dict):
        opps = [target] * pairs
    else:
        ms, w = zip(*target)
        idx = np.random.default_rng(seed).choice(len(ms), size=pairs, p=np.asarray(w) / np.sum(w))
        opps = [ms[k] for k in idx]
    jobs = [(None, br, o, seed + k) for k, o in enumerate(opps)]
    s = np.array([r[1] for r in play_pairs(jobs, map_fn)])
    h = 1.96 * s.std(ddof=1) / math.sqrt(len(s)) if len(s) > 1 else float("nan")
    return dict(winrate=float(s.mean()), exploitability_lb=float(s.mean() - 0.5),
                ci95=[float(s.mean() - 0.5 - h), float(s.mean() - 0.5 + h)], n_games=2 * len(s),
                label=EXPLOIT_LABEL)


# ---------------------------------------------------------------- drivers

def _write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))


def psro_loop(cfg: dict, map_fn=map, n_chunks: int = 4) -> Population:
    """Initial population = cfg['bots']; per iteration: meta-solve -> PPO BR to the meta-Nash mixture ->
    add it -> extend the meta matrix -> results/psro/<run>/iter_<k>.json. On BudgetExceeded the current
    population and matrix (incl. a BR added after its last iter file) go to stopped_population.json."""
    pop = Population([member_from_spec(s) for s in cfg["bots"]])
    try:
        return _psro_iters(cfg, pop, map_fn, n_chunks)
    except BudgetExceeded:
        _write(ROOT / "results/psro" / cfg["run"] / "stopped_population.json", pop.to_json())
        raise


def _psro_iters(cfg: dict, pop: Population, map_fn, n_chunks: int) -> Population:
    out, t0 = Path("results/psro") / cfg["run"], time.time()
    fill_meta(pop, range(len(pop)), cfg["games_per_cell"], map_fn, cfg["seed"], n_chunks)   # one batch
    _write(ROOT / out / "iter_0.json", dict(iter=0, wall_s=round(time.time() - t0, 1), **pop.to_json()))
    total_dec, init = 0, cfg["init"]
    for k in range(1, cfg["iters"] + 1):
        x = meta_solve(pop.M, cfg["solver"])
        br = train_best_response(pop, x, cfg["br_decisions"], init, map_fn, n_chunks, cfg["games_per_update"],
                                 cfg["lr"], seed_base=cfg["seed"] * 1000 + k, out=str(out / f"br_{k}"),
                                 log=lambda r: print(json.dumps(r), flush=True), train_pool=cfg["train_pool"])
        total_dec += br["decisions"]
        idx = pop.add({"kind": "net", "path": br["path"]})
        extend_meta(pop, idx, cfg["games_per_cell"], map_fn, cfg["seed"], n_chunks)
        br_vs_mix = 0.5 + float(pop.M[idx, :idx] @ x)
        rec = dict(iter=k, meta_nash=dict(zip([name(m) for m in pop.members[:idx]], x.round(4).tolist())),
                   meta_nashconv=meta_nashconv(pop.M[:idx, :idx], x), br_winrate_vs_mix=br_vs_mix,
                   br_gain=br_vs_mix - 0.5, nashconv_proxy=2 * (br_vs_mix - 0.5),
                   br_gain_note="BR win rate vs meta-Nash minus 1/2 = " + EXPLOIT_LABEL,
                   br_decisions=br["decisions"], br_games=br["games"], br_train_mix=br["train_mix"], total_decisions=total_dec,
                   wall_s=round(time.time() - t0, 1), br_history=br["history"], **pop.to_json())
        _write(ROOT / out / f"iter_{k}.json", rec)
        print(json.dumps({k_: rec[k_] for k_ in ("iter", "meta_nash", "br_winrate_vs_mix", "wall_s")}), flush=True)
        init = br["path"]
    return pop


def rl_br(target: str, cfg: dict, map_fn=map, n_chunks: int = 4) -> dict:
    """One PPO best response to a fixed target (bot spec, 'hedge_experts' or 'net:<path>') from cfg['init'],
    then its exploitability lower bound on held-out seeds -> results/psro/<run>/rl_br_<target>.json."""
    slug = target.replace(":", "_").replace("/", "_").replace("+", "p")
    out, member, t0 = Path("results/psro") / cfg["run"], member_from_spec(target), time.time()
    every, curve = cfg.get("eval_every_decisions"), []
    ev = lambda path: exploitability_estimate(path, member, cfg["exploit_games"], map_fn, n_chunks)

    def point(ex_, decisions, games):     # one learning-curve point (same held-out seeds); ci95 in win-rate units
        curve.append(dict(decisions=decisions, games_trained=games, winrate=ex_["winrate"],
                          ci95=[round(c + 0.5, 4) for c in ex_["ci95"]]))
        print(json.dumps(dict(curve=curve[-1])), flush=True)

    ex0 = None
    if every and cfg["init"]:                                # curve point 0 = the init net, evaluated up front
        ex0 = ev(cfg["init"]); point(ex0, 0, 0)
    br = train_best_response(Population([member]), [1.0], cfg["rl_br_decisions"], cfg["init"], map_fn, n_chunks,
                             cfg["games_per_update"], cfg["lr"], seed_base=cfg["seed"] * 1000 + 999,
                             out=str(out / f"rlbr_{slug}"), log=lambda r: print(json.dumps(r), flush=True),
                             games_per_input=cfg.get("games_per_input", 1), eval_every=every,
                             on_checkpoint=lambda p, d, g: point(ev(p), d, g), gray=bool(cfg.get("gray_box", False)))
    ex = exploitability_estimate(br["path"], member, cfg["exploit_games"], map_fn, n_chunks)
    if every:
        point(ex, br["decisions"], br["games"])
    if cfg.get("eval_init") and cfg["init"]:                 # same held-out seeds, untrained start net
        ex0 = ex0 or exploitability_estimate(cfg["init"], member, cfg["exploit_games"], map_fn, n_chunks)
        ex["init_winrate"], ex["init_ci95"] = ex0["winrate"], ex0["ci95"]
    rec = dict(target=target, br_path=br["path"], init=cfg["init"], gray_box=bool(cfg.get("gray_box", False)),
               decisions=br["decisions"], games=br["games"],
               wall_s=round(time.time() - t0, 1), **ex, history=br["history"], **({"curve": curve} if every else {}))
    _write(ROOT / out / f"rl_br_{slug}.json", rec)
    print(json.dumps({k: rec.get(k) for k in ("target", "decisions", "winrate", "exploitability_lb", "ci95", "init_winrate")}), flush=True)
    return rec


def run(cfg: dict, mode: str, map_fn=map, n_chunks: int = 4):
    """mode 'psro': PSRO loop, then cfg['rl_br_targets']; mode 'rl_br': only the rl_br targets.
    A BudgetExceeded from map_fn stops cleanly and writes results/psro/<run>/stopped.json."""
    try:
        if mode == "psro":
            psro_loop(cfg, map_fn, n_chunks)
        for t in cfg["rl_br_targets"]:
            rl_br(t, cfg, map_fn, n_chunks)
    except BudgetExceeded as e:
        _write(ROOT / "results/psro" / cfg["run"] / "stopped.json", dict(reason="budget", detail=str(e)))
        print(f"stopped: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="first", help="preset (tiny/quick/smoke/first/full) or JSON file overriding DEFAULT")
    ap.add_argument("--mode", choices=["psro", "rl_br"], default="psro")
    ap.add_argument("--target", default=None, help="rl_br target (overrides config rl_br_targets)")
    ap.add_argument("--local-workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--slowdown", type=float, default=None, help="Modal core vs local core (cost estimate)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.target:
        cfg["rl_br_targets"] = [a.target]
    workers = max(1, min(a.local_workers, MAX_LOCAL_WORKERS))
    if a.dry_run:
        import psro_cost
        kw = {"slowdown": a.slowdown} if a.slowdown else {}
        print(f"config: {json.dumps(cfg)}\nmode: {a.mode}")
        print(psro_cost.format_estimate(psro_cost.estimate(cfg, a.mode, **kw), local_workers=workers))
        return
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(workers) as pool:
        run(cfg, a.mode, pool.map, n_chunks=workers)


if __name__ == "__main__":
    main()
