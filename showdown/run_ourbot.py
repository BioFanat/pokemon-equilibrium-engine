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
from fp.search.poke_engine_helpers import battle_to_poke_engine_state  # noqa: E402

import hidden_agents as ha  # noqa: E402
import nash  # noqa: E402
import pkbattle as pb  # noqa: E402
from exp3_live import Learner  # noqa: E402

RNG = random.Random()
ROOT_METHODS = ("agentform", "pimc")
HEDGE_METHODS = ("hedge", "hedge_fs")                      # PA1 (E2, E5): carried Hedge over stage experts
BANDIT_METHODS = ("bandit_ucb1", "bandit_exp3", "rmplus_avg")   # PA3 (E4): bandit root vs full-information RM+
METHODS = ROOT_METHODS + HEDGE_METHODS + BANDIT_METHODS
HEDGE_EXPERTS = ["top", "sampled", "safest", "safex"]
HEDGE_CONFIG = {"hedge": dict(eta=None, alpha=0.0, scale=400.0),        # anytime FTRL, eta_t = sqrt(8 ln k / t)
                "hedge_fs": dict(eta=0.2, alpha=0.002, scale=400.0)}    # Hedge + fixed share (X6 constants)
SAFEX_EPS = 0.003                                          # safex expert: 0.3% of the payoff range
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
    bandit_ucb1 / bandit_exp3 / rmplus_avg (PA3): bandit self-play (BANDIT_N iterations) or RM+ (RMPLUS_ITERS) on
    Abar squashed to [0,1] by (1 + tanh(Abar / BANDIT_TAU)) / 2; plays from our row shares; logs their Nash gap on it.
    seed/game seed the bandit RNG (with the turn and the decision index)."""
    assert method in METHODS, method
    other = {"agentform": "pimc", "pimc": "agentform"}.get(method)
    learner = meta = None
    if method in HEDGE_METHODS:
        learner, meta = load_learner(method, state_path)
        save_learner(method, state_path, learner, meta)
    ctx = dict(pending=None, n=0)

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
            meta["skipped"] += 1
            fb.update(skipped=True, reason=how if b is None else "not_in_worlds")
        else:
            k, j = hit
            before = learner.cum.copy()
            learner.update(pend["Ms"][k], pend["Q"], j)
            meta["updates"] += 1
            fb.update(skipped=False, world=k, r=(learner.cum - before).round(4).tolist(),
                      weights=learner.weights().round(4).tolist())
        save_learner(method, state_path, learner, meta)
        return fb

    def finish(lines):
        """Game end: the final block was never parsed by foul-play, so read the last decision's feedback from it."""
        fb = feedback(lines, final=True)
        if fb is not None:
            write(dict(method=method, final=True, feedback=fb))

    def find_best_move(battle) -> str:
        from copy import deepcopy
        battle = deepcopy(battle)
        if battle.team_preview:                              # not used in random battles; mirror foul-play
            battle.user.active = battle.user.reserve.pop(0)
            battle.opponent.active = battle.opponent.reserve.pop(0)
        t0 = time.time()
        fb, lines = None, getattr(battle, "_ourlines", [])
        if method in HEDGE_METHODS:
            fb = feedback(lines)
            _FINISHERS[battle.battle_tag] = finish
        worlds = battle.mode.prepare_battles(battle, K)
        Ms, ws, rows, cols, legal, raw = [], [], [], [], None, {}
        for b, chance in worlds:
            state = battle_to_poke_engine_state(b)
            r = nash.pe.id(state, 0)                         # raw engine option strings for side one (us)
            M, s1, s2 = nash.full_matrix(state)
            if legal is None:
                legal = s1
                raw = {pb._canon(o): o for o in r.s1}
            Ms.append(M); ws.append(chance); rows.append(s1); cols.append(s2)
            if len(legal) == 1:
                break
        if method in ROOT_METHODS:
            rec, move = root_decision(method, other, Ms, ws, rows, legal, raw, t0)
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
                p = np.clip(wts @ Q, 0, None)
                p /= p.sum()
                move = raw.get(choice := legal[RNG.choices(range(len(legal)), weights=p)[0]], choice)
                ctx["pending"] = dict(turn=battle.turn, idx=len(lines), opp=battle.opponent.name, Ms=Ms, cols=cols,
                                      w=w, Q=Q)
                rec.update(weights=dict(zip(HEDGE_EXPERTS, wts.round(4).tolist())),
                           experts={e: legal[int(q.argmax())] for e, q in zip(HEDGE_EXPERTS, Q)},
                           p={legal[i]: round(float(p[i]), 4) for i in np.argsort(-p) if p[i] >= 0.001})
            else:
                A01 = 0.5 * (1 + np.tanh(Abar / BANDIT_TAU))
                rng = np.random.default_rng([seed or 0, game or 0, battle.turn, ctx["n"], zlib.crc32(method.encode())])
                if method == "rmplus_avg":
                    x, y = nash.rm_plus(A01, RMPLUS_ITERS)
                else:
                    x, y = bandit_selfplay(A01, method.split("_")[1], BANDIT_N, rng)
                x = np.clip(x, 0, None) / np.clip(x, 0, None).sum()
                move = raw.get(choice := legal[int(rng.choice(len(legal), p=x))], choice)
                rec.update(gap=round(nash_gap(A01, x, y), 5), iters=RMPLUS_ITERS if method == "rmplus_avg" else BANDIT_N)
                p = x
            top = np.argsort(-p)[:3]
            rec.update(mix={legal[i]: round(float(p[i]), 3) for i in top if p[i] >= 0.01}, move=move,
                       secs=round(time.time() - t0, 2))
        if method in HEDGE_METHODS:
            rec["feedback"] = fb
        ctx["n"] += 1
        write(rec)
        return move
    return find_best_move


def root_decision(method, other, Ms, ws, rows, legal, raw, t0):
    """agentform / pimc decision and its record fields (unchanged format: mix, move, secs and the alt_* shadow solve)."""
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
    return rec, move


our_find_best_move = make_move_fn("agentform", K_WORLDS)


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
    defaults = ["--websocket-uri", "ws://127.0.0.1:8000/showdown/websocket", "--ps-username", "eqbot",
                "--bot-mode", "accept_challenge", "--pokemon-format", "gen9randombattle", "--run-count", "100",
                "--log-level", "INFO"]
    sys.argv = [sys.argv[0]] + (sys.argv[1:] or defaults)
    os.chdir(FP)                                            # foul-play loads data relative to its root
    from fp.main import run_foul_play
    asyncio.run(run_foul_play())
