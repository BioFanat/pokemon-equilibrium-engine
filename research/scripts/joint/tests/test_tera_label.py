"""Tera labels: the opponent terastallizes and uses a move (b = "flareblitz-tera") while some worlds list only
"flareblitz" and others only "flareblitz-tera". Runs make_move_fn (hedge_bayes_mw and bayes_safex_mw) through the real
run_ourbot feedback path on stubbed worlds and checks that both logged predictive probabilities refer to the same
exact label: p_qhat = q-hat(col) on the averaged matrix and p_qhat_mw = q_mw(col) of the same decision, and
that only the worlds containing exactly that label are in worlds_with.
  ~/sandbox/bot-venv/bin/python research/scripts/joint/tests/test_tera_label.py"""
import _path  # noqa: F401
import json
import os
import random
import tempfile
from types import SimpleNamespace as NS

import numpy as np

import run_ourbot as ro
import nash

ROWS = ["tackle", "growl", "switch onix"]
CASES = {   # world column sets -> exact observed label and containing worlds, independent of world order
    "plain first": ([["flareblitz", "surf"], ["flareblitz", "ember"], ["flareblitz-tera", "surf"],
                     ["flareblitz-tera", "flareblitz", "ember"]], "flareblitz-tera", [2, 3]),
    "tera first": ([["flareblitz-tera", "flareblitz", "surf"], ["flareblitz", "ember"], ["flareblitz-tera", "surf"],
                    ["ember", "surf"]], "flareblitz-tera", [0, 2]),
}


class Mode:
    def __init__(self, worlds):
        self.worlds = worlds

    def prepare_battles(self, battle, K):
        return self.worlds[:K]


real = (nash.full_matrix, nash.pe)
ro.battle_to_poke_engine_state = lambda b: b
nash.full_matrix = lambda st, side_one_options=None: (st.M, st.s1, st.s2)
nash.pe = NS(id=lambda st, side: NS(s1=st.s1))
rng = np.random.default_rng(5)
n = 0
try:
    for method in ("hedge_bayes_mw", "bayes_safex_mw"):
        for name, (cols, expect, ks) in CASES.items():
            with tempfile.TemporaryDirectory() as work:
                log = os.path.join(work, "dec.jsonl")
                fn = ro.make_move_fn(method, 4, log=log, seed=1, game=1, state_path=os.path.join(work, "hs", "x.json"))
                lines = []
                for t in range(3):
                    worlds = [(NS(M=rng.normal(0, 100, (3, len(c))), s1=ROWS, s2=c), 0.25) for c in cols]
                    fn(NS(team_preview=False, turn=t + 1, _ourlines=list(lines), opponent=NS(name="p2"),
                          request_json=None, user=NS(reserve=[]),
                          battle_tag="battle-x", mode=Mode(worlds)))
                    lines += ["|-terastallize|p2a: Foo|Fire", "|move|p2a: Foo|Flare Blitz|p1a: Bar", f"|turn|{t + 2}"]
                recs = [json.loads(ln) for ln in open(log)]
            for prev, cur in zip(recs, recs[1:]):
                fb = cur["feedback"]
                assert fb["b"] == "flareblitz-tera" and not fb["skipped"], fb
                bf = fb["bayes"]
                assert bf["col"] == expect and bf["worlds_with"] == ks, (name, bf["col"], bf["worlds_with"])
                labels = prev["bayes"]["labels"]
                j = labels.index(bf["col"])
                assert bf["j"] == j
                assert abs(bf["p_qhat"] - prev["bayes"]["qhat"][j]) < 1e-4, "p_qhat is not q-hat(col)"
                assert abs(bf["p_qhat_mw"] - prev["bayes"]["qhat_mw"][j]) < 1e-4, "p_qhat_mw is not q_mw(col)"
                n += 1
            print(f"{method:14s} {name:11s}: col {expect!r}, worlds_with {ks}; p_qhat and p_qhat_mw both score it")
finally:
    nash.full_matrix, nash.pe = real
print(f"{n} feedback records checked")
print("PASS")
