"""Byte identity of the existing Bayes paths: run make_move_fn of the PRE-EDIT modules and of the current ones on the
same stubbed decision sequence (K = 4 synthetic worlds per turn, seeded run_ourbot.RNG, opponent actions fed back
through protocol lines, incl. forced and not_in_worlds turns) and compare every record (minus `secs`), every move and
the final Hedge / posterior state files. Methods: hedge_bayes, bayes_safex, with OURBOT_LOG_ABAR off and on. Also runs
the new *_mw methods through the same harness and checks their log fields.
  ~/sandbox/bot-venv/bin/python research/scripts/joint/tests/test_byte_identity.py <snapshot dir>
<snapshot dir>/engine/{hidden_agents.py, showdown/behavior.py, showdown/run_ourbot.py} are the pre-edit copies."""
import _path  # noqa: F401
import importlib.util
import json
import os
import random
import sys
import tempfile
from types import SimpleNamespace as NS

import numpy as np

import run_ourbot as new                          # imported first: sys.modules keeps the current engine modules
import behavior as new_behavior
import hidden_agents as new_ha
import nash

SNAP = os.path.abspath(sys.argv[1])


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


old_behavior = load("old_behavior", os.path.join(SNAP, "engine", "showdown", "behavior.py"))
old_ha = load("old_hidden_agents", os.path.join(SNAP, "engine", "hidden_agents.py"))
old = load("old_run_ourbot", os.path.join(SNAP, "engine", "showdown", "run_ourbot.py"))
old.behavior, old.ha = old_behavior, old_ha
assert old.behavior is not new.behavior and old.ha is not new.ha

POOL = ["tackle", "ember", "surf", "thunderbolt", "icebeam", "pikachu", "eevee", "snorlax", "flareblitz"]
ROWS = ["tackle", "growl", "switch bulbasaur", "switch onix", "quickattack-tera"]
NAMES = {"pikachu": "Pikachu", "eevee": "Eevee", "snorlax": "Snorlax"}


class Mode:
    def __init__(self, worlds):
        self.worlds = worlds

    def prepare_battles(self, battle, K):
        return self.worlds[:K]


def script(seed, T=40):
    """Per turn: the K worlds (M, s1, s2) and the opponent's protocol line after our decision."""
    rng = np.random.default_rng(seed)
    turns = []
    for t in range(T):
        forced = rng.random() < 0.1
        cols = [["snorlax"]] * 4 if forced else \
            [list(rng.choice(POOL, int(rng.integers(2, 7)), replace=False)) for _ in range(4)]
        m = int(rng.integers(2, len(ROWS) + 1))
        worlds = [(NS(M=np.round(rng.normal(0, 100, (m, len(c))), 2), s1=ROWS[:m], s2=c), 0.25) for c in cols]
        b = "snorlax" if forced else str(rng.choice(POOL))         # sometimes in no world: not_in_worlds skip
        tera = (not forced) and rng.random() < 0.15
        if b in NAMES:
            line = [f"|switch|p2a: {NAMES[b]}|{NAMES[b]}, L80|100/100"]
        else:
            line = (["|-terastallize|p2a: Foo|Fire"] if tera else []) + [f"|move|p2a: Foo|{b}|p1a: Bar"]
        turns.append((worlds, line + [f"|turn|{t + 2}"]))
    return turns


def run(mod, method, turns, work, log_abar):
    os.environ.pop("OURBOT_LOG_ABAR", None)
    if log_abar:
        os.environ["OURBOT_LOG_ABAR"] = "1"
    mod.battle_to_poke_engine_state = lambda b: b
    nash.full_matrix = lambda st: (st.M, st.s1, st.s2)
    nash.pe = NS(id=lambda st, side: NS(s1=st.s1))
    mod.RNG = random.Random(1234)
    log, state = os.path.join(work, "dec.jsonl"), os.path.join(work, "hedge_state", "x.json")
    fn = mod.make_move_fn(method, 4, log=log, seed=1, game=1, state_path=state)
    lines, moves = [], []
    for t, (worlds, opp) in enumerate(turns):
        battle = NS(team_preview=False, turn=t + 1, _ourlines=list(lines), opponent=NS(name="p2"),
                    battle_tag="battle-x", mode=Mode(worlds))
        moves.append(fn(battle))
        lines += opp
    recs = [json.loads(ln) for ln in open(log)]
    for r in recs:
        r.pop("secs", None)
    states = {f: open(os.path.join(work, "hedge_state", f)).read() for f in sorted(os.listdir(os.path.join(work, "hedge_state")))}
    return moves, recs, states


real = (nash.full_matrix, nash.pe)
try:
    for seed in (1, 2, 3):
        turns = script(seed)
        for method in ("hedge_bayes", "bayes_safex"):
            for log_abar in (False, True):
                with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
                    ro, rn = run(old, method, turns, a, log_abar), run(new, method, turns, b, log_abar)
                assert ro[0] == rn[0], "moves differ"
                assert json.dumps(ro[1]) == json.dumps(rn[1]), "records differ"
                assert ro[2] == rn[2], "state files differ"
                fbs = [r["feedback"] for r in rn[1] if r.get("feedback")]
                print(f"seed {seed} {method:11s} log_abar={int(log_abar)}: {len(rn[0])} moves, {len(rn[1])} records, "
                      f"{len(rn[2])} state files identical; feedback scored {sum(not f['skipped'] for f in fbs)}, "
                      f"skipped {sum(f['skipped'] for f in fbs)}")
        for method in ("hedge_bayes_mw", "bayes_safex_mw"):
            with tempfile.TemporaryDirectory() as b:
                moves, recs, states = run(new, method, turns, b, True)
            dec = [r for r in recs if "bayes" in r]
            assert all({"qhat", "qhat_mw", "qhat_k", "cols", "posterior"} <= set(r["bayes"]) for r in dec)
            sc = [r["feedback"]["bayes"] for r in recs if r.get("feedback") and not r["feedback"]["skipped"]]
            assert sc and all({"p_qhat", "p_qhat_mw", "p_style_mw", "posterior"} <= set(f) for f in sc)
            ll = lambda key: -np.mean(np.log([f[key] for f in sc]))
            print(f"seed {seed} {method:14s}: {len(moves)} moves, {len(dec)} Bayes decisions, {len(sc)} scored; "
                  f"log loss averaged {ll('p_qhat'):.3f} vs membership-aware {ll('p_qhat_mw'):.3f} (synthetic)")
finally:
    nash.full_matrix, nash.pe = real
print("PASS")
