"""Offline regressions for live bot decisions; native tests use the local venv.

python -m unittest discover -s engine/tests -p test_battle_decisions.py -v
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "engine"), str(ROOT / "engine/showdown")]
from battle_adapter import request_options, raw_action

NATIVE = importlib.util.find_spec("poke_engine") is not None
FP = Path(os.environ.get("FOULPLAY_DIR", ROOT / ".local-play/foul-play"))


class RequestOptionsTests(unittest.TestCase):
    def battle(self, **active):
        return NS(request_json={"active": [active]}, team_preview=False,
                  user=NS(active=NS(reviving=False), reserve=[NS(name="heatran", hp=100), NS(name="toxapex", hp=0)]))

    def test_server_disabled_moves_pp_and_tera(self):
        b = self.battle(moves=[dict(id="gigatonhammer", disabled=True, pp=5),
                               dict(id="knockoff", pp=2), dict(id="drainpunch", pp=0)], canTerastallize=False)
        self.assertEqual(request_options(b), ["heatran", "knockoff"])

    def test_uncertain_disable_does_not_prohibit_and_tera_is_identity_based(self):
        b = self.battle(moves=[dict(id="knockoff", pp=2, maybeDisabled=True)], canTerastallize="Dark")
        self.assertEqual(request_options(b), ["heatran", "knockoff", "knockoff-tera"])

    def test_forced_switch_trapping_and_struggle(self):
        b = self.battle(trapped=True, moves=[dict(id="toxic", pp=0)])
        self.assertEqual(request_options(b), ["struggle"])
        b.request_json["forceSwitch"] = [True]
        self.assertEqual(request_options(b), ["heatran"])
        b.user.active.reviving = True
        self.assertEqual(request_options(b), ["toxapex"])

    def test_action_mapping_does_not_use_reordered_engine_indices(self):
        b = self.battle()
        for label in ("knockoff", "drainpunch", "knockoff-tera"):
            self.assertEqual(raw_action(label, b), label)
        self.assertEqual(raw_action("heatran", b), "switch heatran")


@unittest.skipUnless(NATIVE, "poke-engine unavailable; use .local-play/venv/bin/python")
class MechanicsRegressions(unittest.TestCase):
    def setUp(self):
        import poke_engine as pe
        import hidden as hd
        import nash
        self.pe, self.hd, self.nash = pe, hd, nash

    def mon(self, name, types, moves, **kw):
        return self.pe.Pokemon(id=name, types=types, base_types=types, hp=300, maxhp=300,
                               attack=200, defense=150, special_attack=200, special_defense=150,
                               speed=100, terastallized=False, moves=[self.pe.Move(id=m) for m in moves], **kw)

    def state(self, p, q, **side):
        return self.pe.State(side_one=self.pe.Side(pokemon=[p], **side), side_two=self.pe.Side(pokemon=[q]))

    def test_magic_bounce_and_existing_rocks(self):
        # Hippowdon -> Hatterene, replay 2715005923 turns 4/5.
        p = self.mon("hippowdon", ("ground", "typeless"), ["stealthrock", "earthquake"])
        q = self.mon("hatterene", ("psychic", "fairy"), ["splash"], ability="magicbounce")
        s = self.state(p, q)
        c = s.apply_instructions(self.pe.generate_instructions(s, "stealthrock", "splash")[0])
        self.assertEqual(c.side_one.side_conditions.stealth_rock, 1)
        self.assertEqual(c.side_two.side_conditions.stealth_rock, 0)
        d = c.apply_instructions(self.pe.generate_instructions(c, "stealthrock", "splash")[0])
        self.assertEqual(d.side_one.side_conditions.stealth_rock, 1)

    def test_toxic_steel_and_ground_flying_are_immune(self):
        # Dragalge -> Probopass, replay 2715021909 turns 15/16.
        p = self.mon("dragalge", ("poison", "dragon"), ["toxic", "earthquake"])
        q = self.mon("probopass", ("rock", "steel"), ["splash"])
        s = self.state(p, q)
        for branch in self.pe.generate_instructions(s, "toxic", "splash"):
            self.assertEqual(str(s.apply_instructions(branch).side_two.pokemon[0].status).lower(), "none")
        q = self.mon("pelipper", ("water", "flying"), ["splash"])
        s = self.state(p, q)
        for branch in self.pe.generate_instructions(s, "earthquake", "splash"):
            self.assertEqual(s.apply_instructions(branch).side_two.pokemon[0].hp, 300)

    def test_setup_at_both_caps_has_no_boost(self):
        p = self.mon("latias", ("dragon", "psychic"), ["calmmind", "psychic"])
        q = self.mon("gothitelle", ("psychic", "typeless"), ["splash"])
        s = self.state(p, q, special_attack_boost=6, special_defense_boost=6)
        for branch in self.pe.generate_instructions(s, "calmmind", "splash"):
            c = s.apply_instructions(branch)
            self.assertEqual((c.side_one.special_attack_boost, c.side_one.special_defense_boost), (6, 6))

    def test_punch_immunity_but_knock_off_damages_and_removes_item(self):
        p = self.mon("gurdurr", ("fighting", "typeless"), ["machpunch", "drainpunch", "knockoff"])
        q = self.mon("sableye", ("ghost", "dark"), ["splash"], item="leftovers")
        s = self.state(p, q)
        for move in ("machpunch", "drainpunch"):
            for branch in self.pe.generate_instructions(s, move, "splash"):
                self.assertEqual(s.apply_instructions(branch).side_two.pokemon[0].hp, 300)
        for branch in self.pe.generate_instructions(s, "knockoff", "splash"):
            c = s.apply_instructions(branch)
            self.assertLess(c.side_two.pokemon[0].hp, 300)
            self.assertEqual(str(c.side_two.pokemon[0].item).lower(), "none")

    def test_terminal_win_outranks_healing_and_is_antisymmetric(self):
        # Before the fix a completed 1v1 KO scored ~110, below healing (~112).
        p = self.mon("hippowdon", ("ground", "typeless"), ["earthquake", "slackoff"])
        q = self.mon("hatterene", ("psychic", "fairy"), ["psychic"], ability="magicbounce")
        p = self.hd.copy_pokemon(p, hp=220, maxhp=311, speed=120)
        q = self.hd.copy_pokemon(q, hp=30, maxhp=250, speed=60)
        s = self.state(p, q)
        M, rows, cols = self.nash.full_matrix(s)
        self.assertGreater(M[rows.index("earthquake"), cols.index("psychic")],
                           M[rows.index("slackoff"), cols.index("psychic")])
        c = s.apply_instructions(self.pe.generate_instructions(s, "earthquake", "psychic")[0])
        self.assertEqual(self.nash._leaf(c, "mid"), 10000)
        self.assertEqual(self.nash._leaf(self.nash._mirror(c), "mid"), -10000)
        dead = self.hd.copy_pokemon(p, hp=0)
        self.assertEqual(self.nash._leaf(self.state(dead, self.hd.copy_pokemon(q, hp=0)), "mid"), 0)

    def test_request_filters_matrix_rows_before_solver(self):
        p = self.mon("gurdurr", ("fighting", "typeless"), ["machpunch", "knockoff"])
        q = self.mon("sableye", ("ghost", "dark"), ["splash"])
        _, rows, _ = self.nash.full_matrix(self.state(p, q), side_one_options=["knockoff"])
        self.assertEqual(rows, ["knockoff"])

    def test_information_planner_uses_exact_terminal_rewards(self):
        import info_planner as ip
        import numpy as np
        p = self.mon("hippowdon", ("ground", "typeless"), ["earthquake"])
        q = self.hd.copy_pokemon(self.mon("hatterene", ("psychic", "fairy"), ["splash"]), hp=1)
        s = self.state(p, q)
        child = s.apply_instructions(self.pe.generate_instructions(s, "earthquake", "splash")[0])
        for side, expected in ((1, 10000), (2, -10000)):
            M, rows, cols = ip.own_child_matrix(child, side)
            np.testing.assert_array_equal(M, [[expected]])
            self.assertEqual((rows, cols), (["none"], ["none"]))
        draw = self.state(self.hd.copy_pokemon(p, hp=0), self.hd.copy_pokemon(q, hp=0))
        self.assertEqual(ip.own_child_matrix(draw, 1)[0][0, 0], 0)
        M, rows, _ = ip.full_matrix_lpleaf(s)
        np.testing.assert_array_equal(M[rows.index("earthquake")], 10000)
        result = ip.plan([s], 1, ["earthquake"], {0})
        for arm in ("A1_rep", "A2_rep", "A1_pw", "A2_pw", "A0L"):
            self.assertEqual(result[arm]["v"], 10000)


class RevivalIntegrationTests(unittest.TestCase):
    def test_revival_and_trapped_recharge_bypass_search(self):
        from revival_guard import guard
        search = lambda battle: self.fail("forced action must bypass search")
        battle = NS(request_json=dict(forceSwitch=[True], side=dict(pokemon=[dict(reviving=True)])),
                    user=NS(reserve=[NS(name="heatran", hp=20), NS(name="toxapex", hp=0)]))
        self.assertEqual(guard(search)(battle), "switch toxapex")
        battle.request_json = dict(active=[dict(trapped=True, moves=[dict(id="recharge")])])
        self.assertEqual(guard(search)(battle), "recharge")


@unittest.skipUnless(NATIVE, "poke-engine unavailable; use .local-play/venv/bin/python")
class OfflineReplayTests(unittest.TestCase):
    def test_offline_replay_aligns_each_world_by_action_name(self):
        from replay_decision import replay
        record = dict(method="agentform", turn=3, move="knockoff", search=dict(
            legal=["knockoff", "drainpunch"], weights=[0.25, 0.75],
            rows=[["knockoff", "drainpunch"], ["drainpunch", "knockoff"]],
            matrices=[[[10], [0]], [[0], [10]]]))
        for method in ("agentform", "pimc", "top"):
            record["method"] = method
            self.assertEqual(replay(record)["mix"], {"knockoff": 1.0, "drainpunch": 0.0})
        with self.assertRaisesRegex(ValueError, "no search snapshot"):
            replay(dict(method="agentform", turn=3, move="knockoff"))


@unittest.skipUnless(NATIVE and (FP / "fp").is_dir(), "local Foul Play environment unavailable")
class AdapterIntegrationTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(FP))
        from fp.config import FoulPlayConfig
        config = patch.object(FoulPlayConfig, "pokemon_format", "gen9randombattle")
        config.start()
        self.addCleanup(config.stop)

    def test_conversion_preserves_stages_hazards_ability_and_revealed_move(self):
        sys.path.insert(0, str(FP))
        from fp.battle.state import Battle, Pokemon
        from battle_adapter import battle_to_poke_engine_state
        b = Battle("offline-regression")
        b.user.active = Pokemon("hippowdon", 82)
        b.opponent.active = Pokemon("hatterene", 82)
        for side in (b.user, b.opponent):
            side.active.ability = "magicbounce" if side is b.opponent else "sandstream"
            side.active.item = "leftovers"
            side.active.add_move("hurricane")
            side.active.boosts.update({"accuracy": -2, "evasion": 1, "special-attack": 6})
            side.side_conditions["stealthrock"] = 1
        s = battle_to_poke_engine_state(b)
        for side in (s.side_one, s.side_two):
            self.assertEqual((side.accuracy_boost, side.evasion_boost, side.special_attack_boost), (-2, 1, 6))
            self.assertEqual(side.side_conditions.stealth_rock, 1)
            self.assertEqual(side.pokemon[0].moves[0].id, "hurricane")
        self.assertEqual(s.side_two.pokemon[0].ability, "magicbounce")

    def test_live_decision_snapshot_replays_offline(self):
        sys.path.insert(0, str(FP))
        from fp.battle.state import Battle, Pokemon
        import run_ourbot
        from replay_decision import replay
        b = Battle("offline-snapshot")
        b.turn = 7
        for side, name, move in ((b.user, "gurdurr", "knockoff"), (b.opponent, "sableye", "splash")):
            side.active = Pokemon(name, 82)
            side.active.ability = "guts" if side is b.user else "prankster"
            side.active.item = "leftovers"
            side.active.add_move(move)
        b.mode = NS(prepare_battles=lambda battle, k: [(battle, 1.0)])
        b.request_json = {"active": [dict(moves=[dict(id="knockoff", pp=10)])]}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, OURBOT_LOG_STATES="1"):
            log = Path(directory) / "decisions.jsonl"
            move = run_ourbot.make_move_fn("agentform", 1, log=str(log))(b)
            record = json.loads(log.read_text())
        self.assertEqual(move, "knockoff")
        self.assertEqual(replay(record)["mix"], {"knockoff": 1.0})
        self.assertEqual(record["battle_tag"], "offline-snapshot")

    def test_concurrent_bayesian_runtime_isolates_five_checkpoint_streams(self):
        from contextlib import ExitStack
        from fp.battle.state import Battle, Pokemon
        import fp.modes.base as fp_base
        import fp.run_battle as fp_run
        import fp.websocket_client as fp_ws
        import ladder
        import run_ourbot

        def battle(tag):
            b = Battle(tag)
            b.turn, b._ourlines = 1, []
            b.user.name, b.opponent.name = "p1", "p2"
            for side, name, moves in ((b.user, "gurdurr", ["knockoff", "drainpunch"]),
                                      (b.opponent, "sableye", ["splash"])):
                side.active = Pokemon(name, 82)
                side.active.ability = "guts" if side is b.user else "prankster"
                side.active.item = "leftovers"
                for move in moves:
                    side.active.add_move(move)
            b.mode = NS(prepare_battles=lambda value, k: [(value, 1.0)])
            b.request_json = {"active": [dict(moves=[dict(id=m, pp=10) for m in ["knockoff", "drainpunch"]])]}
            return b

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            for obj, attr in ((fp_base, "find_best_move"), (fp_run, "battle_mode"),
                              (fp_ws.PSWebsocketClient, "login"), (fp_ws.PSWebsocketClient, "receive_message"),
                              (ladder, "MOVE_POOL"), (run_ourbot, "LOG"), (run_ourbot, "REPLAYS")):
                stack.enter_context(patch.object(obj, attr, getattr(obj, attr)))
            stack.enter_context(patch.object(ladder, "FP", str(FP)))
            stack.enter_context(patch.dict(os.environ, OURBOT_WORK=directory, OURBOT_BELIEF_SOURCE="foulplay_baseline"))
            source = Path(directory) / "hedge_bayes_mw_state.json"
            run_ourbot.make_move_fn("hedge_bayes_mw", 1, state_path=str(source))
            ladder.setup_runtime(dict(out=directory, method="hedge_bayes_mw", k=1, concurrency=5, local=True))
            battles = [battle(f"offline-pool-{i}") for i in range(5)]
            for i, b in enumerate(battles):
                fp_base.find_best_move(b)
                self.assertEqual(ladder.MOVE_POOL.stream_for(b.battle_tag), i)
            files = [Path(directory) / "learning" / f"hedge_bayes_mw-stream{i}.bayes.json" for i in range(5)]
            before = [p.read_text() for p in files]
            self.assertTrue(all(json.loads(text)["meta"]["updates"] == 0 for text in before))
            battles[0].turn = 2
            battles[0]._ourlines = ["|move|p2a: Sableye|Splash|p1a: Gurdurr", "|turn|2"]
            fp_base.find_best_move(battles[0])
            self.assertEqual(json.loads(files[0].read_text())["meta"]["updates"], 1)
            self.assertEqual([p.read_text() for p in files[1:]], before[1:])
            run_ourbot._FINISHERS.pop(battles[0].battle_tag)(["|move|p2a: Sableye|Splash|p1a: Gurdurr", "|win|Bot"])
            ladder.MOVE_POOL.release(battles[0].battle_tag)
            b = battle("offline-pool-reused")
            fp_base.find_best_move(b)
            self.assertEqual(ladder.MOVE_POOL.stream_for(b.battle_tag), 0)
            self.assertEqual(json.loads(files[0].read_text())["meta"]["updates"], 2)
            self.assertEqual([p.read_text() for p in files[1:]], before[1:])
            for tag in list(run_ourbot._FINISHERS):
                if tag.startswith("offline-pool-"):
                    run_ourbot._FINISHERS.pop(tag)
