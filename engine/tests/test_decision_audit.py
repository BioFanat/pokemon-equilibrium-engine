import json
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "showdown"))
from audit_decisions import audit_directory, audit_game, modeled_values, snapshot_context
from run_public_evaluation import plan, records


def replay(own="Hippowdon", opponent="Hatterene", body=""):
    return (f"|player|p1|bot_bonp\n|player|p2|Opponent\n"
            f"|switch|p1a: {own}|{own}, L80|300/300\n"
            f"|switch|p2a: {opponent}|{opponent}, L80|100/100\n" + body).splitlines()


def decision(move, turn=1, legal=None):
    return dict(turn=turn, move=move, search=dict(legal=legal or [move, "earthquake"]), decision_index=turn - 1)


class AuditTests(unittest.TestCase):
    def test_first_cohort_stays_fixed_when_more_games_are_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "replays").mkdir()
            (out / "decisions").mkdir()
            rows = []
            for i, result in enumerate(["win", "loss"]):
                tag = f"battle-{i}"
                rows.append(dict(battle_tag=tag, result=result))
                (out / "replays" / (tag + ".log")).write_text("\n".join(replay(body="|turn|1\n")))
                (out / "decisions" / (tag + ".jsonl")).write_text("")
            (out / "ladder_games.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            self.assertEqual(audit_directory(out, "bot_bonp", limit=1)["results"], {"win": 1})
            self.assertEqual(audit_directory(out, "bot_bonp")["completed_games"], 2)

    def test_magic_bounce_is_replay_backed(self):
        lines = replay(body="|turn|1\n|move|p1a: Hippowdon|Stealth Rock|p2a: Hatterene\n"
                       "|move|p2a: Hatterene|Stealth Rock|p1a: Hippowdon|[from] ability: Magic Bounce\n"
                       "|-sidestart|p1: bot_bonp|move: Stealth Rock\n")
        result = audit_game(lines, [decision("stealthrock")], "bot_bonp")
        self.assertIn("rocks_reflected", result["findings"][0]["categories"])
        self.assertEqual(result["matched"], 1)

    def test_flying_switch_in_is_distinct_from_active_immunity(self):
        lines = replay("Toedscruel", "Mienshao", "|turn|1\n|switch|p2a: Pelipper|Pelipper, L86|40/100\n"
                       "|move|p1a: Toedscruel|Earth Power|p2a: Pelipper\n|-immune|p2a: Pelipper\n")
        result = audit_game(lines, [decision("earthpower")], "bot_bonp")
        f = result["findings"][0]
        self.assertEqual(f["categories"], ["ground_into_flying"])
        self.assertTrue(f["opponent_switched"])
        self.assertEqual(f["planned_opponent"], "mienshao")

    def test_setup_cap_separates_forced_action(self):
        lines = replay("Latias", "Gothitelle", "|-setboost|p1a: Latias|spa|6\n|-setboost|p1a: Latias|spd|6\n"
                       "|turn|1\n|move|p1a: Latias|Calm Mind|p1a: Latias\n"
                       "|-boost|p1a: Latias|spa|0\n|-boost|p1a: Latias|spd|0\n")
        result = audit_game(lines, [decision("calmmind", legal=["calmmind"])], "bot_bonp")
        self.assertIn("setup_at_cap", result["findings"][0]["categories"])
        self.assertEqual(result["findings"][0]["classification"], "only_one_legal_action")

    def test_hurricane_must_have_been_previously_revealed(self):
        body = ("|turn|1\n|move|p2a: Pelipper|Hurricane|p1a: Swalot\n|turn|2\n"
                "|switch|p1a: Urshifu|Urshifu-Rapid-Strike, L75|274/274\n"
                "|move|p2a: Pelipper|Hurricane|p1a: Urshifu\n|-damage|p1a: Urshifu|0 fnt\n|faint|p1a: Urshifu\n")
        result = audit_game(replay("Swalot", "Pelipper", body),
                            [decision("switch urshifurapidstrike", 2, ["urshifurapidstrike", "protect"])], "bot_bonp")
        self.assertIn("switch_into_revealed_hurricane", result["findings"][0]["categories"])
        self.assertEqual(result["findings"][0]["actor"], "urshifurapidstrike")
        self.assertIn("risky_hurricane_switch", result["findings"][0]["categories"])

    def test_resisted_hurricane_switch_is_not_labeled_risky(self):
        body = ("|turn|1\n|move|p2a: Noivern|Hurricane|p1a: Houndoom\n|turn|2\n"
                "|switch|p1a: Iron Thorns|Iron Thorns, L80|295/295\n"
                "|move|p2a: Noivern|Hurricane|p1a: Iron Thorns\n"
                "|-resisted|p1a: Iron Thorns\n|-damage|p1a: Iron Thorns|249/295\n")
        result = audit_game(replay("Houndoom", "Noivern", body),
                            [decision("switch ironthorns", 2, ["ironthorns", "darkpulse"])], "bot_bonp")
        f = result["findings"][0]
        self.assertEqual(f["categories"], ["switch_into_revealed_hurricane"])
        self.assertEqual(f["hurricane_effectiveness"], 0.25)
        self.assertAlmostEqual(f["hurricane_hp_loss"], 46 / 295)

    def test_unexecuted_decisions_are_not_reported_as_immune_attacks(self):
        result = audit_game(replay(body="|turn|1\n|cant|p1a: Hippowdon|par\n"), [decision("toxic")], "bot_bonp")
        self.assertEqual(result["findings"], [])
        self.assertEqual(result["unexecuted_or_unmatched"], 1)

    def test_modeled_values_align_each_world_by_name(self):
        record = dict(search=dict(matrices=[[[10], [0]], [[0], [20]]],
                                  rows=[["knockoff", "drainpunch"], ["drainpunch", "knockoff"]], weights=[1, 1]),
                      bayes=dict(qhat_k=[[1], [1]]))
        self.assertEqual(modeled_values(record, ["knockoff"])["knockoff"]["expected_vs_logged_prediction"], 15)

    @unittest.skipUnless(importlib.util.find_spec("poke_engine"), "poke-engine unavailable")
    def test_native_species_ids_are_normalized_before_comparing_to_replays(self):
        import poke_engine as pe
        state = pe.State(side_one=pe.Side(pokemon=[pe.Pokemon(id="drifblim")]),
                         side_two=pe.Side(pokemon=[pe.Pokemon(id="clawitzer")]))
        context = snapshot_context(dict(search=dict(legal=["calmmind"], states=[state.to_string()])))
        self.assertEqual((context["own"], context["opponent"]), ("drifblim", "clawitzer"))


class EvaluationBudgetTests(unittest.TestCase):
    def test_new_batch_is_additional_100_games_with_five_streams(self):
        _, config, completed, remaining = plan(NS(games=100, method="hedge_bayes_mw", out=None))
        self.assertEqual((config["concurrency"], completed, remaining), (5, 0, 100))

    def test_resume_counts_unique_completed_results_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _, config, _, _ = plan(NS(games=100, method="hedge_bayes_mw", out=None))
            (out / "evaluation.json").write_text(json.dumps(config))
            row = dict(battle_tag="battle-1", result="win")
            (out / "ladder_games.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            self.assertEqual(len(records(out)), 1)
            self.assertEqual(plan(NS(games=100, method="hedge_bayes_mw", out=out))[-1], 99)
            with self.assertRaisesRegex(ValueError, "different method"):
                plan(NS(games=100, method="agentform", out=out))

    def test_explicit_concurrency_change_keeps_same_game_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _, config, _, _ = plan(NS(games=100, method="hedge_bayes_mw", out=None, concurrency=1))
            (out / "evaluation.json").write_text(json.dumps(config))
            (out / "ladder_games.jsonl").write_text(json.dumps(dict(battle_tag="game-1", result="win")) + "\n")
            _, resumed, completed, remaining = plan(NS(games=100, method="hedge_bayes_mw", out=out, concurrency=5))
            self.assertEqual((resumed["concurrency"], completed, remaining), (5, 1, 99))


if __name__ == "__main__":
    unittest.main()
