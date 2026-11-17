"""No-regret variants of the native MCTS ladder bot, on the same worlds, budget and record schema as mcts_native.

mcts_native_sampled: mcts_native's weighted root-visit shares, but the move is sampled from them.
hedge_mcts: Hedge (exp3_live.Learner, carried across games in state_path) over the experts mcts_top (argmax of the
  shares), mcts_sampled (the shares) and lp_sampled (agent-form LP over the per-world full matrices); plays and
  credits exactly like run_ourbot's non-mw hedge path (opponent's realised column in its highest-weight world).
smmcts_rm: per world, outcome-sampling SM-MCTS with regret matching at every node (smmcts_rm.py); plays a sample of
  the world-weighted root average strategy mapped onto the server's legal actions.
World sampling (OURBOT_WORLD_SAMPLER), the exact budget split (OURBOT_MCTS_MS), the forced-action bypass and server
legality follow native_mcts.make_move_fn, which is left untouched.
"""
from copy import deepcopy
import json
import os
import sys
import time

import numpy as np
import poke_engine as pe

import hidden_agents as ha
import nash
import pkbattle as pb
from exp3_live import Learner
from native_mcts import aggregate_visits
import smmcts_rm

METHODS = ("mcts_native_sampled", "hedge_mcts", "smmcts_rm")
HEDGE_MCTS_EXPERTS = ["mcts_top", "mcts_sampled", "lp_sampled"]
HEDGE_MCTS_CONFIG = dict(eta=None, alpha=0.0, scale=400.0)
HEDGE_MCTS_SEMANTICS = "hedge_mcts_v1"
POLICIES = {"mcts_native_sampled": "weighted_root_visit_sample_v1", "hedge_mcts": "hedge_mcts_v1",
            "smmcts_rm": "smmcts_rm_avg_sample_v1"}


def _ro():
    """The loaded run_ourbot module (its RNG, finishers and protocol helpers), also when it runs as __main__."""
    mod = sys.modules.get("run_ourbot")
    if mod is None and hasattr(sys.modules.get("__main__"), "_FINISHERS"):
        mod = sys.modules["__main__"]
    if mod is None:
        import run_ourbot as mod
    return mod


def settings(K):
    """mcts_native's environment checks: (budget_ms, sampler)."""
    budget_ms = int(os.environ.get("OURBOT_MCTS_MS", "1000"))
    if K < 1 or not K <= budget_ms <= 10000:
        raise ValueError("Native MCTS budget must be between K and 10000 milliseconds")
    if os.environ.get("OURBOT_BELIEF_SOURCE", "foulplay_baseline") != "foulplay_baseline":
        raise ValueError("MCTS methods currently use the foulplay_baseline sampler")
    sampler = os.environ.get("OURBOT_WORLD_SAMPLER", "baseline")
    if sampler not in {"baseline", "item_stratified"}:
        raise ValueError("Unknown MCTS world sampler: " + sampler)
    return budget_ms, sampler


def prepare(battle, K, sampler, converter, request_options):
    """Worlds of one decision as in mcts_native: (battle copy, states, weights, legal, sampling metadata)."""
    battle = deepcopy(battle)
    if battle.team_preview:
        battle.user.active = battle.user.reserve.pop(0)
        battle.opponent.active = battle.opponent.reserve.pop(0)
    if sampler == "item_stratified":
        from stratified_worlds import prepare_worlds
        sampled, sampling = prepare_worlds(battle, K)
    else:
        sampled = list(battle.mode.prepare_battles(battle, K))
        sampling = dict(name="baseline")
    states, weights = zip(*[(converter(b), weight) for b, weight in sampled])
    allowed = request_options(battle)
    first_options, _ = pb.options(states[0])
    legal = [a for a in first_options if allowed is None or a in allowed]
    if not legal:
        raise ValueError("No engine options match the server's legal actions")
    return battle, list(states), list(weights), legal, allowed, sampling


def split_budget(budget_ms, n):
    return [budget_ms // n + (i < budget_ms % n) for i in range(n)]


def native_search(states, weights, legal, budget_ms):
    """mcts_native's budgeted search over the worlds: (shares, worlds, fallback, searches)."""
    results, searches = [], []
    for state, duration in zip(states, split_budget(budget_ms, len(states))):
        t0 = time.perf_counter()
        results.append(pe.monte_carlo_tree_search(state, duration_ms=duration, threads=1))
        searches.append(dict(budget_ms=duration, elapsed_secs=time.perf_counter()-t0))
    shares, worlds, fallback = aggregate_visits(results, weights, legal)
    return shares, worlds, fallback, searches


def sample(p, rng):
    return int(rng.choices(range(len(p)), weights=p)[0])


def mix_of(p, legal):
    top = np.argsort(-p)[:3]
    return {legal[i]: round(float(p[i]), 3) for i in top if p[i] >= 0.01}


# ---------------------------------------------------------------- hedge_mcts carried state
def load_learner(path):
    """Hedge over HEDGE_MCTS_EXPERTS restored from path (a file of another method, config, experts or semantics is an
    error, never silently reused). Returns (learner, meta) with meta['games'] counting this game."""
    L = Learner(experts=HEDGE_MCTS_EXPERTS, **HEDGE_MCTS_CONFIG)
    meta = dict(games=0, updates=0, skipped=0)
    if path and os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        if (d.get("method") != "hedge_mcts" or d.get("config") != HEDGE_MCTS_CONFIG
                or d.get("experts") != HEDGE_MCTS_EXPERTS or d.get("semantics") != HEDGE_MCTS_SEMANTICS):
            raise ValueError(f"hedge_mcts state {path} was written by {d.get('method')} {d.get('config')} "
                             f"{d.get('experts')} {d.get('semantics')}")
        L.logw, L.cum, L.t = np.array(d["logw"]), np.array(d["cum"]), int(d["t"])
        meta = d["meta"]
    meta["games"] += 1
    return L, meta


def save_learner(path, L, meta):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    d = dict(method="hedge_mcts", config=HEDGE_MCTS_CONFIG, experts=HEDGE_MCTS_EXPERTS,
             semantics=HEDGE_MCTS_SEMANTICS, logw=L.logw.tolist(), cum=L.cum.tolist(), t=L.t,
             weights=L.weights().round(6).tolist(), meta=meta)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, path)


# ---------------------------------------------------------------- move functions
def make_move_fn(method, K, log, converter, request_options, raw_action, state_path=None, seed=0):
    assert method in METHODS, method
    budget_ms, sampler = settings(K)
    depth = int(os.environ.get("OURBOT_SMRM_DEPTH", "3"))
    if depth < 1:
        raise ValueError("OURBOT_SMRM_DEPTH must be at least 1")
    ctx = dict(pending=None, n=0)
    learner = meta = None
    if method == "hedge_mcts":
        learner, meta = load_learner(state_path)
        save_learner(state_path, learner, meta)

    def write(rec):
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def feedback(lines, final=False):
        """run_ourbot's non-mw hedge feedback: credit the experts with the opponent's realised column."""
        ro = _ro()
        pend, ctx["pending"] = ctx["pending"], None
        if pend is None:
            return None
        labels = set().union(*pend["cols"])
        if len(labels) == 1:                                 # opponent had one option (waiting, recharge, locked)
            b, how = next(iter(labels)), "forced"
        else:
            b, how = ro.opp_action(lines if final else lines[pend["idx"]:], pend["opp"])
        fb = dict(for_turn=pend["turn"], b=b, how=how)
        hit = ro.map_column(b, pend["cols"], pend["w"]) if b is not None else None
        if hit is None:
            meta["skipped"] += 1
            fb.update(skipped=True, reason=how if b is None else "not_in_worlds")
        else:
            k, j = hit
            fb.update(skipped=False, world=k)
            before = learner.cum.copy()
            learner.update(pend["Ms"][k], pend["Q"], j)
            meta["updates"] += 1
            fb.update(r=(learner.cum - before).round(4).tolist(), weights=learner.weights().round(4).tolist())
        save_learner(state_path, learner, meta)
        return fb

    def finish(lines):
        fb = feedback(lines, final=True)
        if fb is not None:
            write(dict(method=method, final=True, feedback=fb))

    def move_fn(battle):
        started = time.perf_counter()
        ro = _ro()
        fb, lines = None, list(getattr(battle, "_ourlines", []))
        if method == "hedge_mcts":
            if ctx["pending"] is not None and ctx["pending"].get("tag") != battle.battle_tag:
                ctx["pending"] = None                        # a previous battle's decision never got its feedback
            fb = feedback(lines)
            ro._FINISHERS[battle.battle_tag] = finish
        battle, states, weights, legal, allowed, sampling = prepare(battle, K, sampler, converter, request_options)
        forced = len(legal) == 1
        extra, search_extra = {}, dict(matrices=[])
        mcts = dict(policy=POLICIES[method], budget_ms=budget_ms, threads=1, forced=forced, fallback=False,
                    visit_policy=[1.0] if forced else None, searches=[], worlds=[])
        if forced:
            played = np.ones(1)
        elif method == "smmcts_rm":
            played, mcts_extra = smrm_decision(states, weights, legal, budget_ms, depth, seed, battle.turn, ctx["n"])
            mcts.update(mcts_extra)
        else:
            shares, mcts["worlds"], mcts["fallback"], mcts["searches"] = native_search(states, weights, legal,
                                                                                         budget_ms)
            mcts["visit_policy"] = shares.tolist()
            if method == "mcts_native_sampled":
                played = shares
            else:
                played, extra, search_extra = hedge_decision(states, weights, legal, allowed, shares, learner)
                ctx["pending"] = dict(tag=battle.battle_tag, turn=battle.turn, idx=len(lines), opp=battle.opponent.name,
                                      Ms=extra.pop("_Ms"), cols=search_extra["cols"], w=extra.pop("_w"),
                                      Q=extra.pop("_Q"))
        chosen = 0 if forced else sample(played, ro.RNG)
        action = raw_action(legal[chosen], battle)
        record = dict(method=method, turn=battle.turn, battle_tag=battle.battle_tag, decision_index=ctx["n"],
                      worlds=len(states), legal=legal, move=action, mix=mix_of(played, legal),
                      pfull=np.asarray(played, dtype=float).tolist(), secs=round(time.perf_counter()-started, 4),
                      mcts=mcts, **extra)
        if method == "hedge_mcts":
            record["feedback"] = fb
        record["sampler"] = sampling
        if os.environ.get("OURBOT_LOG_STATES") == "1":
            record["search"] = dict(states=[s.to_string() for s in states], weights=list(weights), legal=legal,
                                    rows=search_extra.get("rows", [pb.options(s)[0] for s in states]),
                                    cols=search_extra.get("cols", [pb.options(s)[1] for s in states]),
                                    matrices=search_extra["matrices"])
        write(record)
        ctx["n"] += 1
        return action
    return move_fn


def hedge_decision(states, weights, legal, allowed, shares, learner):
    """Experts Q (rows: mcts_top, mcts_sampled, lp_sampled; columns aligned with legal) and the Hedge mixture."""
    Ms, rows, cols, raw = [], [], [], []
    for state in states:
        M, s1, s2 = nash.full_matrix(state, side_one_options=allowed)
        raw.append(M.tolist()); rows.append(s1); cols.append(s2)
        Ms.append(ha.align(M, s1, legal))
    w = np.asarray(weights, dtype=float) / np.sum(weights)
    x = ha.agent_form_lp(Ms, w)[0]
    Q = np.stack([np.eye(len(legal))[int(shares.argmax())], shares, x])
    wts = learner.weights()
    p = np.clip(wts @ Q, 0, None)
    p /= p.sum()
    extra = dict(weights=dict(zip(HEDGE_MCTS_EXPERTS, wts.round(4).tolist())),
                 Q=Q.round(4).tolist(),
                 experts={e: legal[int(q.argmax())] for e, q in zip(HEDGE_MCTS_EXPERTS, Q)},
                 _Ms=Ms, _w=w, _Q=Q)
    return p, extra, dict(rows=rows, cols=cols, matrices=raw)


def smrm_decision(states, weights, legal, budget_ms, depth, seed, turn, n):
    """SM-MCTS-RM per world on its share of the budget; Σ_k β_k σ̄_k mapped onto legal by label."""
    w = np.asarray(weights, dtype=float) / np.sum(weights)
    positions = {label: i for i, label in enumerate(legal)}
    policy, searches, worlds = np.zeros(len(legal)), [], []
    for k, (state, duration) in enumerate(zip(states, split_budget(budget_ms, len(states)))):
        rng = np.random.default_rng([seed or 0, turn or 0, n, k])
        labels, avg, stats = smmcts_rm.search(state, duration, depth, rng)
        for label, prob in zip(labels, avg):
            if label in positions:
                policy[positions[label]] += w[k] * prob
        searches.append(dict(budget_ms=duration, elapsed_secs=stats["elapsed_secs"],
                             iterations=stats["iterations"], depth=depth, nodes=stats["nodes"]))
        worlds.append(dict(weight=float(w[k]), root={l: round(float(p), 4) for l, p in zip(labels, avg)}))
    fallback = bool(policy.sum() <= 0)
    policy = np.full(len(legal), 1.0 / len(legal)) if fallback else policy / policy.sum()
    return policy, dict(fallback=fallback, visit_policy=policy.tolist(), searches=searches, worlds=worlds,
                        depth=depth, gamma=smmcts_rm.GAMMA)
