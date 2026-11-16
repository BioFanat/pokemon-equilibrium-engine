"""Run OUR agent (agent-form solve over sampled worlds) on a LOCAL Showdown server through foul-play's client.

foul-play (GPL-3.0, ~/sandbox/foul-play) does the websocket protocol, battle tracking and sampling of the opponent's
hidden sets. We replace only its move choice: each turn, take K sampled battles with weights from foul-play's
random-battle mode, convert each to a poke-engine State, build our full depth-2 payoff matrix per world
(nash.full_matrix), and solve ONE agent-form LP over the worlds (hidden_agents.agent_form_lp). Then sample.

Local-only: login skips play.pokemonshowdown.com (the local server runs with --no-security).

  ~/sandbox/bot-venv/bin/python ~/sandbox/ourbot/run_ourbot.py            # waits for your challenge as 'eqbot'
Then in the browser (http://127.0.0.1:8000): pick a name, find user 'eqbot', challenge to [Gen 9] Random Battle.
Per-turn decisions are logged to ~/sandbox/ourbot/decisions.jsonl.
"""
import json
import os
import random
import sys
import time
import zlib

HOME = os.path.expanduser("~")
FP = os.environ.get("FOULPLAY_DIR", os.path.join(HOME, "sandbox", "foul-play"))
ENGINE = os.environ.get("ENGINE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK = os.environ.get("OURBOT_WORK", os.path.join(HOME, "sandbox", "ourbot"))
LOG = os.path.join(WORK, "decisions.jsonl")
K_WORLDS = int(os.environ.get("OURBOT_K", "4"))
sys.path[:0] = [FP, ENGINE]

import numpy as np  # noqa: E402

import fp.battle.protocol as fp_protocol  # noqa: E402
import fp.modes.base as fp_base  # noqa: E402
import fp.run_battle as fp_run  # noqa: E402
from fp import websocket_client as fp_ws  # noqa: E402
from fp.battle.helpers import normalize_name  # noqa: E402
from battle_adapter import battle_to_poke_engine_state, request_options, raw_action  # noqa: E402

import hidden_agents as ha  # noqa: E402
import nash  # noqa: E402
import pkbattle as pb  # noqa: E402
from exp3_live import Learner  # noqa: E402

import behavior  # noqa: E402
from belief_worlds import MechanicsWorldProvider  # noqa: E402

RNG = random.Random()
ROOT_METHODS = ("agentform", "pimc")
HEDGE_METHODS = ("hedge", "hedge_fs", "hedge_bayes")       # PA1 (E2, E5): carried Hedge over stage experts
SAFEX_METHODS = ("bayes_safex",)                           # eps-safe exploit of the Bayesian style posterior's q-hat
BAYES_METHODS = ("hedge_bayes", "bayes_safex")             # carry a style posterior (behavior.py)
BANDIT_METHODS = ("bandit_ucb1", "bandit_exp3", "bandit_ucb1_mv", "bandit_exp3_mv", "rmplus_avg")   # *_mv: play the most-visited arm (MCTS decision rule)   # PA3 (E4): bandit root vs full-information RM+
EXPLOIT_VICTIMS = ("bandit_ucb1", "bandit_ucb1_mv", "bandit_exp3_mv", "rmplus_avg", "top", "agentform")
EXPLOIT_METHODS = tuple(f"exploit_{v}" for v in EXPLOIT_VICTIMS)   # model-based pure best reply to a known algorithm
STYLE_METHODS = ("style_eq", "style_safe", "style_uniform")       # realisability: play a behavior.py style
CONTROL_METHODS = ("top",)                                         # E5 control: argmax of the agent-form LP
METHODS = ROOT_METHODS + HEDGE_METHODS + SAFEX_METHODS + BANDIT_METHODS + EXPLOIT_METHODS + STYLE_METHODS + \
    CONTROL_METHODS
LEARN_METHODS = HEDGE_METHODS + SAFEX_METHODS              # read the opponent's realised action after each decision
HEDGE_EXPERTS = ["top", "sampled", "safest", "safex"]
HEDGE_CONFIG = {"hedge": dict(eta=None, alpha=0.0, scale=400.0),        # anytime FTRL, eta_t = sqrt(8 ln k / t)
                "hedge_fs": dict(eta=0.2, alpha=0.002, scale=400.0),    # Hedge + fixed share (X6 constants)
                "hedge_bayes": dict(eta=None, alpha=0.0, scale=400.0)}  # as hedge; its safex expert uses q-hat
SAFEX_EPS = 0.003                                          # safex expert: 0.3% of the payoff range
BAYES_SAFEX_EPS = float(os.environ.get("OURBOT_SAFEX_EPS", SAFEX_EPS))  # eps of the q-hat safex (BAYES_METHODS)
BANDIT_N, BANDIT_TAU, RMPLUS_ITERS = 2000, 160.0, 1000     # E4: root iterations, win-prob squash, RM+ iterations


# ---------------------------------------------------------------- what the opponent did (protocol lines)
# foul-play clears battle.msg_list after parsing it, and battle.opponent.last_used_move misreports the CHOSEN action
# (cant rewrites it with the previous move, a post-KO switch-in overwrites the move, tera is dropped). So keep every
# parsed protocol line on the battle (battle._ourlines) and read the opponent's choice from the lines themselves.
_orig_process = fp_protocol.process_battle_updates
_orig_finished = fp_run.battle_is_finished
_FINISHERS = {}                                            # battle_tag -> callback(final message lines) at game end


def _recording_process(battle):
    if not hasattr(battle, "_ourlines"):
        battle._ourlines = []
    battle._ourlines.extend(battle.msg_list)
    return _orig_process(battle)


def _finished_hook(battle_tag, msg):
    """foul-play never parses the final block (|win| is caught first); hand it to the move fn's finisher."""
    done = _orig_finished(battle_tag, msg)
    if done and battle_tag in _FINISHERS:
        _FINISHERS.pop(battle_tag)(msg.split("\n"))
    return done


fp_protocol.process_battle_updates = _recording_process    # update_battle looks both up as module globals
fp_run.battle_is_finished = _finished_hook


def opp_action(lines, opp):
    """The opponent's chosen action in the lines after a decision, up to the next |turn|: its first |move| (called
    moves with [from] are skipped; '-tera' if it terastallized first) or its first |switch| (canonical species id,
    as in poke-engine's option labels). Returns (label, how) or (None, reason): reason 'fainted' (KO'd before acting;
    the next switch is a forced replacement), 'cant' (sleep/paralysis/flinch: choice not revealed), 'no_action'
    (no opponent action before the turn ended or the game ended). |drag| is not a choice and is ignored."""
    pre, tera = f"{opp}a", False
    for ln in lines:
        p = ln.split("|")
        if len(p) < 3:
            continue
        if p[1] == "turn":
            return None, "no_action"
        if not p[2].startswith(pre):
            continue
        if p[1] == "-terastallize":
            tera = True
        elif p[1] == "move" and not any(s.startswith("[from]") for s in p[4:]):
            return normalize_name(p[3].strip().lower()) + ("-tera" if tera else ""), "move"
        elif p[1] == "switch":
            return normalize_name(p[3].split(",")[0].strip().lower()), "switch"
        elif p[1] in ("faint", "cant"):
            return None, "fainted" if p[1] == "faint" else "cant"
    return None, "no_action"


def map_column(b, cols, w):
    """(world k, column j) of label b in the highest-weight world whose opponent options contain it, else None.
    Columns are per world (each world samples its own hidden moves/species), so they are matched by label."""
    cands = [b, b[:-len("-tera")]] if b.endswith("-tera") else [b]
    for k in sorted(range(len(cols)), key=lambda k: -w[k]):
        for c in cands:
            if c in cols[k]:
                return k, cols[k].index(c)
    return None


def avg_matrix(Ms, cols, w):
    """Abar over the worlds' row-aligned matrices: columns matched by canonical label across worlds (union, in first-seen
    order); column j averages M_k[:, j] over the worlds k that contain label j, weights renormalised over those worlds
    (posterior given that the option exists). Returns (Abar, labels)."""
    labels = list(dict.fromkeys(c for cs in cols for c in cs))
    Abar = np.zeros((Ms[0].shape[0], len(labels)))
    for j, lab in enumerate(labels):
        ks = [k for k in range(len(Ms)) if lab in cols[k]]
        wk = np.array([w[k] for k in ks])
        Abar[:, j] = sum(wi * Ms[k][:, cols[k].index(lab)] for wi, k in zip(wk / wk.sum(), ks))
    return Abar, labels


def nash_gap(A, x, y):
    """max(A y) - min(xᵀA): 0 at an equilibrium of the zero-sum matrix A."""
    return float((A @ y).max() - (x @ A).min())


# ---------------------------------------------------------------- PA3: bandit self-play at the root
def bandit_selfplay(A01, kind, N, rng):
    """Decoupled bandit self-play on A01 in [0,1] (row maximises): each iteration both sides pick an arm, observe only
    r ~ Bernoulli(A01[i, j]) (the column player gets 1 - r) and update. ucb1: untried arms first, then mean +
    sqrt(2 ln t / n). exp3: anytime loss-form Exp3 with importance weights, eta_t = sqrt(ln k / (k t)), no forced
    uniform mixing. Returns the realised visit shares (x̄, ȳ)."""
    m, n = A01.shape
    cnt = [np.zeros(m), np.zeros(n)]
    u = rng.random((N, 3))
    if kind == "ucb1":
        rew = [np.zeros(m), np.zeros(n)]

        def pick(s, t):
            c = cnt[s]
            if (c == 0).any():
                return int(np.argmin(c > 0))
            return int(np.argmax(rew[s] / c + np.sqrt(2 * np.log(t) / c)))
        for t in range(1, N + 1):
            i, j = pick(0, t), pick(1, t)
            r = float(u[t - 1, 0] < A01[i, j])
            cnt[0][i] += 1; rew[0][i] += r
            cnt[1][j] += 1; rew[1][j] += 1 - r
    else:
        loss = [np.zeros(m), np.zeros(n)]
        for t in range(1, N + 1):
            ps = []
            for s, k in ((0, m), (1, n)):
                z = -np.sqrt(np.log(k) / (k * t)) * loss[s]
                p = np.exp(z - z.max())
                ps.append(p / p.sum())
            i = min(int(np.searchsorted(np.cumsum(ps[0]), u[t - 1, 1] * ps[0].sum())), m - 1)
            j = min(int(np.searchsorted(np.cumsum(ps[1]), u[t - 1, 2] * ps[1].sum())), n - 1)
            r = float(u[t - 1, 0] < A01[i, j])
            loss[0][i] += (1 - r) / ps[0][i]
            loss[1][j] += r / ps[1][j]
            cnt[0][i] += 1; cnt[1][j] += 1
    return cnt[0] / N, cnt[1] / N


# ---------------------------------------------------------------- open items: exploiter, style bots (new methods)
def column_agent_form(Ms, cols, labels, w):
    """The opponent's (column player's) agent-form equilibrium on OUR worlds: agent_form_lp over [-M_kᵀ] with each
    world's rows (its column labels) aligned on the union `labels` by ha.align (a label missing from world k gets that
    matrix's column-min, a dominated row, the same treatment our own rows get). Returns y over `labels`."""
    return ha.agent_form_lp([ha.align(-M.T, cols[k], labels) for k, M in enumerate(Ms)], w)[0]


def exploit_yhat(victim, Abar, Ms, cols, labels, w, rng):
    """exploit_<victim>: the predicted column strategy y-hat of a victim whose algorithm is known but whose worlds and
    RNG are not, computed on OUR view (Abar, our worlds; the victim's column side). bandit_ucb1: its column shares from
    bandit_selfplay; *_mv: uniform over the argmax of those shares (the victim's random tie-break, in expectation);
    rmplus_avg: RM+'s column average; top: one-hot at argmax of the column agent-form equilibrium; agentform: that
    equilibrium."""
    if victim in ("top", "agentform"):
        y = column_agent_form(Ms, cols, labels, w)
        if victim == "top":
            y = np.eye(len(labels))[int(y.argmax())]
        return y
    A01 = 0.5 * (1 + np.tanh(Abar / BANDIT_TAU))
    if victim == "rmplus_avg":
        y = nash.rm_plus(A01, RMPLUS_ITERS)[1]
    else:
        y = bandit_selfplay(A01, victim.split("_")[1], BANDIT_N, rng)[1]
    y = np.clip(y, 0, None) / np.clip(y, 0, None).sum()
    if victim.endswith("_mv"):
        y = (y >= y.max() - 1e-12).astype(float)
        y /= y.sum()
    return y


def style_strategy(style, Abar):
    """style_<s> bot: the distribution over OUR rows that behavior.style_dists assigns to style s for the column player
    of -Abarᵀ, i.e. our roles swapped into the opponent model's convention (rows = the opponent's labels, columns = our
    moves, the column player minimises). Same TAU_FRAC (tau = 0.1 ptp, invariant under -·ᵀ) and same MIX (5% uniform)
    as the hedge_bayes posterior: eq = our maximin strategy on Abar (nash.lp), safe = softmax of our per-move worst case,
    uniform = 1/m. The reference mixture p only enters br_us, which is not offered; it is passed as uniform over the
    opponent's labels. Expected mismatch: the posterior evaluates the same functional on ITS Abar (its sampled worlds
    of our hidden sets), not ours, so its q_s is not exactly this distribution."""
    A = -Abar.T
    q = behavior.style_dists(A, np.full(A.shape[0], 1.0 / A.shape[0]))[behavior.STYLES.index(style)]
    return q / q.sum()


# ---------------------------------------------------------------- PA1: carried Hedge state
def load_learner(method, path):
    """exp3_live.Learner over HEDGE_EXPERTS with the method's constants, restored from the JSON state file if present
    (a file written with other constants or experts is an error, never silently reused). Returns (learner, meta)."""
    cfg = HEDGE_CONFIG[method]
    L = Learner(experts=HEDGE_EXPERTS, **cfg)
    meta = dict(games=0, updates=0, skipped=0)
    if path and os.path.exists(path):
        d = json.load(open(path))
        if d["method"] != method or d["config"] != cfg or d["experts"] != HEDGE_EXPERTS:
            raise ValueError(f"hedge state {path} was written by {d['method']} {d['config']} {d['experts']}")
        L.logw, L.cum, L.t = np.array(d["logw"]), np.array(d["cum"]), int(d["t"])
        meta = d["meta"]
    meta["games"] += 1
    return L, meta


def save_learner(method, path, L, meta):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = dict(method=method, config=HEDGE_CONFIG[method], experts=HEDGE_EXPERTS, logw=L.logw.tolist(),
             cum=L.cum.tolist(), t=L.t, weights=L.weights().round(6).tolist(), meta=meta)
    tmp = path + ".tmp"
    json.dump(d, open(tmp, "w"))
    os.replace(tmp, path)


def solve(method, Ms, w):
    """Our mixed strategy over the aligned rows. agentform: one LP max_x Σ_k w_k min_j (M_kᵀ x)_j.
    pimc: LP each world, blend x̄ = Σ_k w_k x_k (the engine's hidden_agents pimc, with foul-play's world weights)."""
    if method == "agentform":
        return ha.agent_form_lp(Ms, w)[0]
    x = sum(wk * ha.lp_x(M)[0] for wk, M in zip(w, Ms))
    return x / x.sum()


# ---------------------------------------------------------------- belief logging (OURBOT_LOG_WORLDS=1; logging only)
LOG_WORLDS_EXTRA = int(os.environ.get("OURBOT_LOG_WORLDS_EXTRA", "32"))   # extra worlds sampled for scoring only


def _mon_dict(p, revealed):
    return dict(name=p.name, item=p.item, ability=p.ability, tera_type=p.tera_type, terastallized=bool(p.terastallized),
                moves=[m.name for m in p.moves], revealed=revealed, hp=p.hp, fainted=bool(p.fainted))


def _world_teams(worlds, revealed):
    return [dict(w=float(chance), opp=[_mon_dict(p, p.name in revealed)
                                       for p in [b.opponent.active] + list(b.opponent.reserve) if p is not None])
            for b, chance in worlds]


def belief_log(battle, worlds):
    """foul-play's belief about the opponent at this decision (scored by score_belief.py). `battle` is the real battle
    (deep copy, untouched by prepare_battles). Fields: known (what is public per revealed mon, before sampling),
    exact (count-weighted marginals over battle.mode.get_all_remaining_sets: the distribution worlds are drawn from),
    worlds (the K decision worlds), extra (OURBOT_LOG_WORLDS_EXTRA more worlds, for bench species). The global random
    state is saved and restored, so the game's RNG stream is the same as without logging."""
    from copy import deepcopy
    from collections import Counter
    st = random.getstate()
    try:
        opp = [p for p in [battle.opponent.active] + list(battle.opponent.reserve) if p is not None]
        revealed = {p.name for p in opp}
        known, exact = [], {}
        for p in opp:
            known.append(dict(name=p.name, moves=[m.name for m in p.moves],
                              item=None if p.item == "unknownitem" else p.item, removed_item=p.removed_item,
                              ability=p.ability, tera_type=p.tera_type if p.terastallized else None,
                              terastallized=bool(p.terastallized), fainted=bool(p.fainted)))
            sets = battle.mode.get_all_remaining_sets(deepcopy(p))
            tot = float(sum(s.pkmn_set.count for s in sets))
            if not tot:
                continue
            m = dict(item=Counter(), ability=Counter(), tera_type=Counter(), moves=Counter())
            for s in sets:
                c = s.pkmn_set.count / tot
                m["item"][s.pkmn_set.item] += c
                m["ability"][s.pkmn_set.ability] += c
                m["tera_type"][s.pkmn_set.tera_type] += c
                for mv in set(s.pkmn_moveset.moves):
                    m["moves"][mv] += c
            exact[p.name] = dict(n_sets=len(sets), **{k: {str(a): round(b, 6) for a, b in v.items()} for k, v in m.items()})
        extra = battle.mode.prepare_battles(battle, LOG_WORLDS_EXTRA) if LOG_WORLDS_EXTRA else []
        return dict(known=known, exact=exact, worlds=_world_teams(worlds, revealed),
                    extra=[[(q["name"], q["revealed"]) for q in w["opp"]] for w in _world_teams(extra, revealed)])
    except Exception as e:                                  # logging must never break a game
        return dict(error=repr(e)[:300])
    finally:
        random.setstate(st)


def make_move_fn(method: str, K: int, log=None, seed: int = 0, game: int = 0, state_path=None):
    """Move function over K of foul-play's sampled worlds (shared sampling, matrices and alignment), solved by
    `method` (one of METHODS); returns a poke-engine move string. Logs each decision to `log`, or to the
    module-level LOG (read at call time) when log is None.
    agentform / pimc: each record also carries the other method's solve on the same matrices (alt_*; not timed in
    secs, never consumes RNG) so the two arms can be compared per state.
    hedge / hedge_fs (PA1): Hedge (exp3_live.Learner) over the experts top / sampled (agent-form LP) / safest (pure
    maximin row of Abar) / safex(0.3%) on Abar; plays p = Σ_e w_e q_e. At the next call (or at game end) the opponent's
    realised action b is read from the protocol lines (opp_action) and mapped to a column (map_column); the learner is
    updated with that world's aligned matrix and the expert matrix Q saved at the decision, else the turn is skipped.
    Weights are carried across games in the JSON file state_path (loaded now, saved after every update).
    hedge_bayes / bayes_safex: a posterior over opponent styles (behavior.py, carried in <state_path>.bayes.json) gives
    q-hat; hedge_bayes = hedge with its safex expert maximising x.Abar.q-hat, bayes_safex plays that safex alone (eps
    OURBOT_SAFEX_EPS). The posterior is updated with the same mapped opponent action (skipped turns leave it unchanged).
    bandit_ucb1 / bandit_exp3 / rmplus_avg (PA3): bandit self-play (BANDIT_N iterations) or RM+ (RMPLUS_ITERS) on
    Abar squashed to [0,1] by (1 + tanh(Abar / BANDIT_TAU)) / 2; plays from our row shares; logs their Nash gap on it.
    top: deterministic argmax of the agent-form LP. exploit_<v>: pure best reply argmax_i (Abar y-hat)_i to victim v's
    algorithm run for the column side on our own view (exploit_yhat). style_<s>: samples behavior.py's style s with the
    roles swapped (style_strategy). OURBOT_LOG_ABAR=1 adds abar / labels / legal / pfull (+ Q, qs) to each record.
    seed/game seed the bandit RNG (with the turn and the decision index); exploit / style use the same seeding."""
    assert method in METHODS, method
    other = {"agentform": "pimc", "pimc": "agentform"}.get(method)
    learner = meta = post = bmeta = None
    if method in HEDGE_METHODS:
        learner, meta = load_learner(method, state_path)
        save_learner(method, state_path, learner, meta)
    bpath = behavior.state_path_for(state_path)
    if method in BAYES_METHODS:
        post, bmeta = behavior.load_posterior(bpath)
        behavior.save_posterior(bpath, post, bmeta)
    ctx = dict(pending=None, n=0)
    belief_source = os.environ.get("OURBOT_BELIEF_SOURCE", "foulplay_baseline")
    if belief_source not in {"foulplay_baseline", "reveals_v2", "mechanics_v2"}:
        raise ValueError(f"unknown OURBOT_BELIEF_SOURCE: {belief_source}")
    mechanics = (MechanicsWorldProvider(belief_source,
                 shadow=os.environ.get("OURBOT_BELIEF_SHADOW", "1") != "0",
                 rng=random.Random((seed or 0) * 1009 + (game or 0)))
                 if belief_source != "foulplay_baseline" else None)
    log_abar = os.environ.get("OURBOT_LOG_ABAR") == "1"    # open item 1: log Abar, labels, legal, full p (+ Q, qs)
    if nash.LEAF_STATS_ON:                                # OURBOT_LEAF_STATS / OURBOT_DET_LEAF: per-decision leaf counts
        ctx["leaf"] = leaf_stats_copy()

    def write(rec):
        with open(log or LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def feedback(lines, final=False):
        """Consume the pending decision (if any): credit the experts with the opponent's realised column. `lines` is the
        battle's whole parsed history (read from the decision's index on), or with final=True the unparsed last block."""
        pend, ctx["pending"] = ctx["pending"], None
        if pend is None:
            return None
        labels = set().union(*pend["cols"])
        if len(labels) == 1:                                 # opponent had one option (waiting, recharge, locked)
            b, how = next(iter(labels)), "forced"
        else:
            b, how = opp_action(lines if final else lines[pend["idx"]:], pend["opp"])
        fb = dict(for_turn=pend["turn"], b=b, how=how)
        hit = map_column(b, pend["cols"], pend["w"]) if b is not None else None
        if hit is None:
            for m in (meta, bmeta):
                if m is not None:
                    m["skipped"] += 1
            fb.update(skipped=True, reason=how if b is None else "not_in_worlds")
        else:
            k, j = hit
            fb.update(skipped=False, world=k)
            if learner is not None:
                before = learner.cum.copy()
                learner.update(pend["Ms"][k], pend["Q"], j)
                meta["updates"] += 1
                fb.update(r=(learner.cum - before).round(4).tolist(), weights=learner.weights().round(4).tolist())
            if post is not None:                             # prequential score of q-hat, then the Bayes step
                fb["bayes"] = behavior.score_and_update(post, pend, pend["cols"][k][j], how == "forced")
                bmeta["updates"] += 1
        if learner is not None:
            save_learner(method, state_path, learner, meta)
        if post is not None:
            behavior.save_posterior(bpath, post, bmeta)
        return fb

    def finish(lines):
        """Game end: the final block was never parsed by foul-play, so read the last decision's feedback from it."""
        fb = feedback(lines, final=True)
        if fb is not None:
            write(dict(method=method, final=True, feedback=fb))
        if mechanics is not None:
            mechanics.finish(lines)
            write(dict(method=method, final=True, mechanics=mechanics.summary()))

    def find_best_move(battle) -> str:
        from copy import deepcopy
        battle = deepcopy(battle)
        if battle.team_preview:                              # not used in random battles; mirror foul-play
            battle.user.active = battle.user.reserve.pop(0)
            battle.opponent.active = battle.opponent.reserve.pop(0)
        t0 = time.time()
        fb, lines = None, getattr(battle, "_ourlines", [])
        if method in LEARN_METHODS:
            fb = feedback(lines)
            _FINISHERS[battle.battle_tag] = finish
        # The v2 provider owns its prior and posterior independently of Foul
        # Play's inference. Never reweight prepare_battles with our likelihoods.
        worlds, states = None, None
        if mechanics is not None:
            mechanics.update(battle, battle_to_poke_engine_state)
            _FINISHERS[battle.battle_tag] = finish
            states = mechanics.sample_states(battle, battle_to_poke_engine_state, K)
        if states is None:
            worlds = battle.mode.prepare_battles(battle, K)
            states = ((battle_to_poke_engine_state(b), chance) for b, chance in worlds)
        Ms, ws, rows, cols, legal, raw = [], [], [], [], None, {}
        allowed = request_options(battle)
        snapshots = [] if os.environ.get("OURBOT_LOG_STATES") == "1" else None
        search_matrices = []
        for state, chance in states:
            M, s1, s2 = nash.full_matrix(state, side_one_options=allowed)
            if legal is None:
                legal = s1
                raw = {label: raw_action(label, battle) for label in s1}
            if snapshots is not None:
                snapshots.append(state.to_string())
                search_matrices.append(M.tolist())
            Ms.append(M); ws.append(chance); rows.append(s1); cols.append(s2)
            if len(legal) == 1:
                break
        if method in ROOT_METHODS:
            rec, move = root_decision(method, other, Ms, ws, rows, legal, raw, t0, cols=cols if log_abar else None)
            rec = dict(method=method, turn=battle.turn, **rec)
        elif len(legal) == 1:
            move = raw.get(legal[0], legal[0])
            rec = dict(method=method, turn=battle.turn, worlds=len(Ms), shape=list(Ms[0].shape), mix={legal[0]: 1.0},
                       move=move, secs=round(time.time() - t0, 2))
        else:
            Ms = [ha.align(M, s1, legal) for M, s1 in zip(Ms, rows)]
            w = np.asarray(ws) / np.sum(ws)
            Abar, labels = avg_matrix(Ms, cols, w)
            rec = dict(method=method, turn=battle.turn, worlds=len(Ms), shape=list(Ms[0].shape),
                       abar_shape=list(Abar.shape))
            if method in HEDGE_METHODS:
                x = ha.agent_form_lp(Ms, w)[0]
                sx = np.clip(nash.safex_strategy(Abar, SAFEX_EPS), 0, None)
                eye = np.eye(len(legal))
                Q = np.stack([eye[int(x.argmax())], x, eye[int(Abar.min(1).argmax())], sx / sx.sum()])
                wts = learner.weights()
                bay = {}
                if post is not None:                         # hedge_bayes: safex expert on q-hat; br_us sees the
                    p_ref = np.clip(wts @ Q, 0, None)        # mixture with the old safex (non-circular reference)
                    qs, qhat, rec["bayes"] = behavior.decide(Abar, p_ref / p_ref.sum(), post, labels)
                    bx = np.clip(nash.safex_strategy(Abar, BAYES_SAFEX_EPS, qhat), 0, None)
                    Q[3] = bx / bx.sum()
                    bay = dict(labels=labels, qs=qs, qhat=qhat, old=behavior.old_predictor(Abar, post.mix))
                p = np.clip(wts @ Q, 0, None)
                p /= p.sum()
                move = raw.get(choice := legal[RNG.choices(range(len(legal)), weights=p)[0]], choice)
                ctx["pending"] = dict(turn=battle.turn, idx=len(lines), opp=battle.opponent.name, Ms=Ms, cols=cols,
                                      w=w, Q=Q, **bay)
                rec.update(weights=dict(zip(HEDGE_EXPERTS, wts.round(4).tolist())),
                           experts={e: legal[int(q.argmax())] for e, q in zip(HEDGE_EXPERTS, Q)},
                           p={legal[i]: round(float(p[i]), 4) for i in np.argsort(-p) if p[i] >= 0.001})
            elif method in SAFEX_METHODS:                    # bayes_safex: eps-safe exploit of q-hat; br_us sees
                p_ref = np.clip(nash.safex_strategy(Abar, BAYES_SAFEX_EPS), 0, None)   # the old one-hot safex
                qs, qhat, rec["bayes"] = behavior.decide(Abar, p_ref / p_ref.sum(), post, labels)
                p = np.clip(nash.safex_strategy(Abar, BAYES_SAFEX_EPS, qhat), 0, None)
                p /= p.sum()
                move = raw.get(choice := legal[RNG.choices(range(len(legal)), weights=p)[0]], choice)
                ctx["pending"] = dict(turn=battle.turn, idx=len(lines), opp=battle.opponent.name, Ms=Ms, cols=cols,
                                      w=w, labels=labels, qs=qs, qhat=qhat, old=behavior.old_predictor(Abar, post.mix))
                rec.update(eps=BAYES_SAFEX_EPS,
                           p={legal[i]: round(float(p[i]), 4) for i in np.argsort(-p) if p[i] >= 0.001})
            elif method in CONTROL_METHODS:                  # top: argmax of the agent-form LP (Hedge's top expert)
                p = np.eye(len(legal))[int(ha.agent_form_lp(Ms, w)[0].argmax())]
                move = raw.get(choice := legal[int(p.argmax())], choice)
            elif method in EXPLOIT_METHODS:                  # pure best reply (lowest index) to the predicted y-hat
                rng = np.random.default_rng([seed or 0, game or 0, battle.turn, ctx["n"], zlib.crc32(method.encode())])
                victim = method[len("exploit_"):]
                yhat = exploit_yhat(victim, Abar, Ms, cols, labels, w, rng)
                br = int(np.argmax(Abar @ yhat))
                p = np.eye(len(legal))[br]
                move = raw.get(choice := legal[br], choice)
                rec.update(victim=victim, br=legal[br],
                           yhat={labels[j]: round(float(yhat[j]), 4) for j in np.argsort(-yhat)[:5] if yhat[j] > 0})
            elif method in STYLE_METHODS:                    # sample our move from the style's distribution
                rng = np.random.default_rng([seed or 0, game or 0, battle.turn, ctx["n"], zlib.crc32(method.encode())])
                p = style_strategy(method[len("style_"):], Abar)
                move = raw.get(choice := legal[int(rng.choice(len(legal), p=p))], choice)
                rec.update(style=method[len("style_"):])
            else:
                A01 = 0.5 * (1 + np.tanh(Abar / BANDIT_TAU))
                rng = np.random.default_rng([seed or 0, game or 0, battle.turn, ctx["n"], zlib.crc32(method.encode())])
                if method == "rmplus_avg":
                    x, y = nash.rm_plus(A01, RMPLUS_ITERS)
                else:
                    x, y = bandit_selfplay(A01, method.split("_")[1], BANDIT_N, rng)
                x = np.clip(x, 0, None) / np.clip(x, 0, None).sum()
                if method.endswith("_mv"):          # MCTS-style decision rule: most-visited arm, ties at random
                    best = np.flatnonzero(x >= x.max() - 1e-12)
                    move = raw.get(choice := legal[int(rng.choice(best))], choice)
                else:
                    move = raw.get(choice := legal[int(rng.choice(len(legal), p=x))], choice)
                rec.update(played_rule="max_visit" if method.endswith("_mv") else "shares", gap=round(nash_gap(A01, x, y), 5), iters=RMPLUS_ITERS if method == "rmplus_avg" else BANDIT_N)
                p = x
            if log_abar:                                     # logging only: no RNG, no change to p or the move
                pplay = p
                if method in BANDIT_METHODS and method.endswith("_mv"):   # played: uniform over the most-visited
                    pplay = (x >= x.max() - 1e-12).astype(float)
                    pplay /= pplay.sum()
                rec.update(abar=Abar.round(2).tolist(), labels=list(labels), legal=list(legal),
                           pfull=np.round(pplay, 4).tolist())
                if method in HEDGE_METHODS:
                    rec["Q"] = Q.round(4).tolist()
                if method in BAYES_METHODS:                  # q-hat is already in rec["bayes"]["qhat"]
                    rec["qs"] = np.round(qs, 4).tolist()
            top = np.argsort(-p)[:3]
            rec.update(mix={legal[i]: round(float(p[i]), 3) for i in top if p[i] >= 0.01}, move=move,
                       secs=round(time.time() - t0, 2))
        if method in LEARN_METHODS:
            rec["feedback"] = fb
        rec["battle_tag"] = battle.battle_tag
        rec["decision_index"] = ctx["n"]
        if snapshots is not None:
            # Matrices retain the exact payoffs even if timed native leaf
            # searches differ on a subsequent run. No account/request data.
            rec["search"] = dict(states=snapshots, weights=ws, rows=rows, cols=cols,
                                 legal=legal, matrices=search_matrices)
        if os.environ.get("OURBOT_LOG_WORLDS") == "1" and worlds is not None:
            rec["belief"] = belief_log(battle, worlds)
        if nash.LEAF_STATS_ON:
            cur = leaf_stats_copy()
            rec["leaf_stats"] = leaf_stats_delta(cur, ctx["leaf"])
            ctx["leaf"] = cur
        ctx["n"] += 1
        if mechanics is not None:
            rec["mechanics"] = mechanics.summary()
        write(rec)
        return move
    return find_best_move


def leaf_stats_copy():
    """Snapshot of nash.LEAF_STATS (depth_hist copied)."""
    return dict(nash.LEAF_STATS, depth_hist=dict(nash.LEAF_STATS["depth_hist"]))


def leaf_stats_delta(cur, prev):
    """cur - prev for two leaf_stats_copy snapshots (depth_hist: only the depths seen since prev)."""
    d = {kk: cur[kk] - prev[kk] for kk in ("calls", "non_d1", "retries", "fallback")}
    d["depth_hist"] = {kk: v - prev["depth_hist"].get(kk, 0) for kk, v in cur["depth_hist"].items()
                       if v != prev["depth_hist"].get(kk, 0)}
    return d


def root_decision(method, other, Ms, ws, rows, legal, raw, t0, cols=None):
    """agentform / pimc decision and its record fields (unchanged format: mix, move, secs and the alt_* shadow solve).
    cols (OURBOT_LOG_ABAR=1 only): also log abar / labels / legal / pfull (no RNG, after the timed solve)."""
    alt = None
    if len(legal) == 1:
        move = raw.get(legal[0], legal[0]); x = np.ones(1)
        secs = time.time() - t0
    else:
        Ms = [ha.align(M, s1, legal) for M, s1 in zip(Ms, rows)]
        w = np.asarray(ws) / np.sum(ws)
        x = solve(method, Ms, w)
        move = raw.get(choice := legal[RNG.choices(range(len(legal)), weights=x)[0]], choice)
        secs = time.time() - t0
        t1 = time.time()
        alt = solve(other, Ms, w)
        alt_secs = time.time() - t1
    top = np.argsort(-x)[:3]
    rec = dict(worlds=len(Ms), shape=[int(Ms[0].shape[0]), int(Ms[0].shape[1])],
               mix={legal[i]: round(float(x[i]), 3) for i in top if x[i] >= 0.01},
               move=move, secs=round(secs, 2))
    if alt is not None:
        atop = np.argsort(-alt)[:3]
        rec.update(alt_method=other, alt_mix={legal[i]: round(float(alt[i]), 3) for i in atop if alt[i] >= 0.01},
                   argmax_same=bool(np.argmax(x) == np.argmax(alt)), tv=round(float(0.5 * np.abs(x - alt).sum()), 4),
                   alt_secs=round(alt_secs, 3))
        if cols is not None:
            Abar, labels = avg_matrix(Ms, cols, w)
            rec.update(abar_shape=list(Abar.shape), abar=Abar.round(2).tolist(), labels=list(labels),
                       legal=list(legal), pfull=np.round(x, 4).tolist())
    return rec, move


our_find_best_move = make_move_fn(os.environ.get("OURBOT_METHOD", "agentform"), K_WORLDS)


async def local_login(self):
    """Local --no-security server: no assertion needed, so never contact play.pokemonshowdown.com."""
    await self.get_id_and_challstr()
    await self.send_message("", [f"/trn {self.username},0,"])
    import asyncio
    await asyncio.sleep(1)
    return self.username


REPLAYS = os.path.join(WORK, "replays")
REPLAY_HTML = """<!DOCTYPE html>
<meta charset="utf-8"><title>{tag}</title>
<div class="wrapper replay-wrapper" style="max-width:1180px;margin:0 auto">
<div class="battle"></div><div class="battle-log"></div><div class="replay-controls"></div><div class="replay-controls-2"></div>
<script type="text/plain" class="battle-log-data">{log}</script>
</div>
<script src="https://play.pokemonshowdown.com/js/replay-embed.js"></script>
"""
_buf: dict[str, list[str]] = {}
_orig_receive = fp_ws.PSWebsocketClient.receive_message


async def recording_receive(self):
    """Tee each battle room's public protocol lines; on |win| or |tie| write <tag>.log and a replay <tag>.html
    (the standard Showdown replay format, rendered by play.pokemonshowdown.com's replay-embed.js)."""
    msg = await _orig_receive(self)
    if os.environ.get("OURBOT_RAW"):
        open(os.path.join(WORK, "raw.txt"), "a").write(repr(msg[:300]) + "\n")
    if msg.startswith(">battle-"):
        tag, *lines = msg.split("\n")
        tag = tag[1:].strip()
        out, skip = _buf.setdefault(tag, []), False
        for ln in lines:
            if skip:                                        # |split|pN: the next line is the private copy, drop it
                skip = False
                continue
            if ln.startswith("|split|"):
                skip = True
                continue
            if ln.startswith(("|request|", "|init|", "|title|", "|j|", "|J|", "|l|", "|L|")) or not ln:
                continue
            out.append(ln)
        if any(ln.startswith("|win|") or ln == "|tie" or ln.startswith("|tie|") for ln in lines):
            os.makedirs(REPLAYS, exist_ok=True)
            log = "\n".join(out)
            open(os.path.join(REPLAYS, f"{tag}.log"), "w").write(log + "\n")
            open(os.path.join(REPLAYS, f"{tag}.html"), "w").write(REPLAY_HTML.format(tag=tag, log=log.replace("</", "<\\/")))
            _buf.pop(tag, None)
    return msg


fp_base.find_best_move = our_find_best_move
fp_ws.PSWebsocketClient.login = local_login
fp_ws.PSWebsocketClient.receive_message = recording_receive

if __name__ == "__main__":
    import asyncio
    from local_bot_safety import validate_local_bot_identity

    _create_client = fp_ws.PSWebsocketClient.create

    async def create_local_client(cls, username, password, address):
        validate_local_bot_identity(username, password, address)
        return await _create_client(username, password, address)

    fp_ws.PSWebsocketClient.create = classmethod(create_local_client)
    defaults = ["--websocket-uri", "ws://127.0.0.1:8000/showdown/websocket", "--ps-username", "eqbot",
                "--bot-mode", "accept_challenge", "--pokemon-format", "gen9randombattle", "--run-count", "100",
                "--log-level", "INFO"]
    sys.argv = [sys.argv[0]] + (sys.argv[1:] or defaults)
    os.chdir(FP)                                            # foul-play loads data relative to its root
    from fp.main import run_foul_play
    asyncio.run(run_foul_play())
