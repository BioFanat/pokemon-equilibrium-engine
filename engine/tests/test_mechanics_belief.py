"""Run with python -m unittest discover -s engine/tests -v (no third-party deps)."""
import copy
import json
import math
import random
import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from belief_catalog import Catalog, Candidate, CatalogEntry, enumerate_species
from belief_engine import parse_rolls
from belief_events import DamageEvidence, HPObservation, Reveal, Skip, SpeedEvidence
from belief_likelihoods import DamagePMF, Unsupported, effective_speed, filter_hp, incoming_likelihood
from belief_tracker import BeliefSession, PriorSupportError
from belief_worlds import MechanicsWorldProvider
from showdown.belief_observer import ProtocolBeliefObserver, own_snapshot


def catalog(items=None, abilities=None):
    role = dict(moves=["Thunderbolt", "Ice Beam", "Surf", "Earthquake"],
                abilities=abilities or ["Levitate"], items=items or ["Leftovers", "Choice Scarf"],
                teraTypes=["Electric"])
    data = {"Test": dict(level=50, abilities=["Levitate"], roles={"one": role})}
    dex = {"test": {"baseStats": dict.fromkeys(("hp", "atk", "def", "spa", "spd", "spe"), 100)}}
    return Catalog(data, dex)


def reveal(session, entity="p2:test"):
    session.observe(Reveal("initial", entity, "Test"))


class Oracle:
    def distribution(self, candidate, evidence, current_item=None, current_ability=None):
        if candidate.item == "choicescarf":
            return DamagePMF.from_uniform_rolls([20, 20, 21, 22])
        return DamagePMF.from_uniform_rolls([20, 21, 22, 23])


class CatalogTests(unittest.TestCase):
    def test_full_checked_in_catalog(self):
        cat = Catalog()
        self.assertEqual(len(cat.data), 509)
        for species in cat.data:
            entries = cat.entries(species)
            self.assertAlmostEqual(sum(e.prior for e in entries), 1)
            self.assertEqual(len({e.candidate for e in entries}), len(entries))
            self.assertTrue(all(e.prior > 0 for e in entries))

    def test_duplicate_paths_and_list_frequencies(self):
        cat = catalog(items=["Leftovers", "Leftovers", "Choice Scarf"])
        cat.data["Test"]["roles"]["two"] = copy.deepcopy(cat.data["Test"]["roles"]["one"])
        entries = cat.entries("test")
        self.assertEqual(len(entries), 2)
        self.assertAlmostEqual(next(e.prior for e in entries if e.candidate.item == "leftovers"), 2 / 3)

    def test_move_pool_conditioning(self):
        cat = catalog()
        cat.data["Test"]["roles"]["two"] = copy.deepcopy(cat.data["Test"]["roles"]["one"])
        cat.data["Test"]["roles"]["two"]["moves"] += ["Psychic"]
        session = BeliefSession(cat)
        reveal(session)
        self.assertAlmostEqual(session.marginal("p2:test", "move")["psychic"], 0.4)
        session.observe(Reveal("psychic", "p2:test", "Test", "move", "Psychic"))
        self.assertAlmostEqual(session.marginal("p2:test", "move")["psychic"], 1)


class PosteriorTests(unittest.TestCase):
    def setUp(self):
        self.session = BeliefSession(catalog(), Oracle())
        reveal(self.session)

    def test_damage_bayes_and_exactly_once(self):
        event = DamageEvidence("damage", "p2:test", "thunderbolt", HPObservation(100, 100, True),
                               HPObservation(80, 100, True), supported=True)
        self.assertTrue(self.session.observe(event))
        # P(20 | scarf)=1/2, P(20 | leftovers)=1/4, equal priors.
        self.assertAlmostEqual(self.session.marginal("p2:test", "item")["choicescarf"], 2 / 3)
        self.assertFalse(self.session.observe(event))
        self.assertAlmostEqual(self.session.marginal("p2:test", "item")["choicescarf"], 2 / 3)
        self.assertEqual(self.session.diagnostics["duplicate"], 1)

    def test_speed_removes_slow_candidate(self):
        self.session.observe(SpeedEvidence("order", "p2:test", 160, True, supported=True))
        self.assertEqual(self.session.marginal("p2:test", "item")["choicescarf"], 1)
        self.assertTrue(all(self.session.sample("p2:test", random.Random(i)).item == "choicescarf" for i in range(20)))

    def test_speed_tie_is_not_a_hard_elimination(self):
        base = self.session.catalog.stats(self.session.entities["p2:test"].entries[0].candidate)["spe"]
        self.session.observe(SpeedEvidence("tie", "p2:test", base, False, supported=True))
        self.assertEqual(self.session.marginal("p2:test", "item")["leftovers"], 1)

    def test_persistence_across_observations(self):
        self.session.observe(SpeedEvidence("order", "p2:test", 160, True, supported=True))
        self.session.observe(Reveal("move", "p2:test", "Test", "move", "Surf"))
        self.assertEqual(self.session.marginal("p2:test", "item")["choicescarf"], 1)

    def test_event_time_item_not_future_removed_item(self):
        self.session.observe(Reveal("gone", "p2:test", "Test", "item", "None", False))
        self.session.observe(SpeedEvidence("earlier_order", "p2:test", 160, True, supported=True,
                                           attributes_at_event=True, item_at_event=None))
        self.assertEqual(self.session.marginal("p2:test", "item")["choicescarf"], 1)

    def test_unsupported_candidate_skips_whole_factor(self):
        session = BeliefSession(catalog(abilities=["Levitate", "Swift Swim"]))
        reveal(session)
        before = session.entities["p2:test"].probabilities()
        self.assertFalse(session.observe(SpeedEvidence("order", "p2:test", 160, True, supported=True)))
        self.assertEqual(before, session.entities["p2:test"].probabilities())
        self.assertEqual(session.diagnostics["skip:speed_ability:swiftswim"], 1)

    def test_impossible_damage_rolls_back(self):
        before = self.session.entities["p2:test"].probabilities()
        event = DamageEvidence("bad", "p2:test", "thunderbolt", HPObservation(100, 100, True),
                               HPObservation(1, 100, True), supported=True)
        self.assertFalse(self.session.observe(event))
        self.assertEqual(before, self.session.entities["p2:test"].probabilities())
        self.assertEqual(self.session.diagnostics["model_mismatch"], 1)

    def test_explicit_reveals_and_current_item_separate(self):
        self.session.observe(Reveal("item", "p2:test", "Test", "item", "Choice Scarf"))
        self.session.observe(Reveal("gone", "p2:test", "Test", "item", "None", False))
        self.assertEqual(self.session.marginal("p2:test", "item")["choicescarf"], 1)
        self.assertEqual(self.session.entities["p2:test"].item, "none")

    def test_out_of_prior_direct_reveal_never_silently_resets(self):
        with self.assertRaises(PriorSupportError):
            self.session.observe(Reveal("impossible", "p2:test", "Test", "item", "Life Orb"))
        with self.assertRaises(PriorSupportError):
            self.session.sample("p2:test", random.Random())

    def test_reveal_only_control(self):
        session = BeliefSession(catalog(), mechanics=False)
        reveal(session)
        session.observe(SpeedEvidence("order", "p2:test", 160, True, supported=True))
        self.assertAlmostEqual(session.marginal("p2:test", "item")["choicescarf"], .5)

    def test_numerical_stability(self):
        for i in range(1500):
            self.session.observe(DamageEvidence(str(i), "p2:test", "thunderbolt", HPObservation(100, 100, True),
                                                HPObservation(80, 100, True), supported=True))
        self.assertAlmostEqual(sum(self.session.entities["p2:test"].probabilities()), 1)


class LikelihoodTests(unittest.TestCase):
    def test_damage_duplicates_and_censored_ko(self):
        pmf = DamagePMF.from_uniform_rolls([20, 20, 30, 40])
        self.assertEqual(dict(pmf.outcomes)[20], .5)
        self.assertEqual(incoming_likelihood(pmf, HPObservation(25, 100, True), HPObservation(0, 100, True)), .5)

    def test_reject_non_pmf(self):
        with self.assertRaises(ValueError):
            DamagePMF(((20, .1),))

    def test_cli_contract_requires_rolls_not_maxima(self):
        text = "Damage Rolls: " + ",".join(str(i) for i in range(20, 36))
        self.assertEqual(len(parse_rolls(text + "\n" + text, True).outcomes), 16)
        with self.assertRaises(Unsupported):
            parse_rolls("Damage Rolls: 100,150\nDamage Rolls: 40,60", True)

    def test_speed_modifiers_are_conservative(self):
        self.assertEqual(effective_speed(131, "levitate", "choicescarf"), 196)
        self.assertEqual(effective_speed(131, "levitate", "leftovers", stage=1), 196)
        with self.assertRaises(Unsupported):
            effective_speed(131, "levitate", "quickclaw")
        with self.assertRaises(Unsupported):
            effective_speed(131, "levitate", "choicescarf", paralyzed=True)

    def test_hidden_hp_sequential_filter_matches_enumeration(self):
        prior = {100: .25, 101: .75}
        pmf = DamagePMF.from_uniform_rolls([5, 6, 6])
        obs1, obs2 = HPObservation(95, 100), HPObservation(89, 100)
        p1, h1 = filter_hp(prior, pmf, obs1, 101)
        p2, h2 = filter_hp(h1, pmf, obs2, 101)
        brute = {}
        for h, ph in prior.items():
            for d1, pd1 in pmf.outcomes:
                h_mid = max(0, h - d1)
                if obs1.compatible(h_mid, 101):
                    for d2, pd2 in pmf.outcomes:
                        h_last = max(0, h_mid - d2)
                        if obs2.compatible(h_last, 101):
                            brute[h_last] = brute.get(h_last, 0) + ph * pd1 * pd2
        self.assertAlmostEqual(p1 * p2, sum(brute.values()))
        self.assertEqual(h2, {h: p / sum(brute.values()) for h, p in brute.items()})


def own():
    return dict(species="own", level=50, hp=300, maxhp=300, speed=160, attack=100, defense=100,
                special_attack=100, special_defense=100, types=["normal", "typeless"],
                ability="levitate", item="leftovers", status="none", moves=["surf"],
                weight_kg=50, tera_type="normal")


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.session = BeliefSession(catalog(), Oracle())
        self.observer = ProtocolBeliefObserver(self.session, "p1")
        self.observer.feed(["|switch|p1a: Me|Own, L50|300/300", "|switch|p2a: Them|Test, L50|100/100", "|turn|1"])
        self.observer.set_own(own())

    def test_clean_hit_and_action_order_update(self):
        self.observer.feed(["|move|p2a: Them|Thunderbolt|p1a: Me", "|-damage|p1a: Me|280/300",
                            "|move|p1a: Me|Surf|p2a: Them", "|-damage|p2a: Them|90/100", "|turn|2"])
        self.assertEqual(self.session.diagnostics["damage"], 1)
        self.assertEqual(self.session.diagnostics["speed"], 1)
        self.assertEqual(self.session.marginal("p2:test", "item")["choicescarf"], 1)

    def test_hidden_unexecuted_move_is_not_revealed(self):
        self.observer.feed(["|cant|p2a: Them|par", "|turn|2"])
        self.assertEqual(self.session.entities["p2:test"].moves, set())

    def test_called_move_not_original_set_evidence(self):
        self.observer.feed(["|move|p2a: Them|Psychic|p1a: Me|[from] move: Sleep Talk", "|turn|2"])
        self.assertNotIn("psychic", self.session.entities["p2:test"].moves)
        self.assertFalse(self.session.entities["p2:test"].degraded)

    def test_critical_multihit_and_residual_are_not_clean_damage(self):
        self.observer.feed(["|move|p2a: Them|Thunderbolt|p1a: Me", "|-crit|p1a: Me",
                            "|-damage|p1a: Me|280/300", "|-damage|p1a: Me|270/300|[from] psn", "|turn|2"])
        self.assertEqual(self.session.diagnostics["damage"], 0)
        self.assertEqual(self.session.diagnostics["skip:complex_move_block"], 1)

    def test_weather_is_recognized_without_an_actor(self):
        self.observer.feed(["|-weather|RainDance"])
        self.assertTrue(self.observer.field_unsafe)

    def test_transferred_item_does_not_rewrite_original_item(self):
        self.observer.feed(["|-item|p2a: Them|Life Orb|[from] move: Trick",
                            "|-enditem|p2a: Them|Life Orb"])
        self.assertFalse(self.session.entities["p2:test"].degraded)
        self.assertEqual(self.session.entities["p2:test"].item, "none")
        self.assertAlmostEqual(self.session.marginal("p2:test", "item")["choicescarf"], .5)

    def test_no_opponent_private_requests_or_raw_split_lines(self):
        with self.assertRaises(ValueError):
            self.observer.feed(['|request|{"side":{"id":"p2"}}'])
        with self.assertRaises(ValueError):
            self.observer.feed(["|split|p2"])

    def test_provider_history_idempotent(self):
        provider = MechanicsWorldProvider(session=self.session)
        provider.observer.own_side = "p1"
        lines = ["|switch|p2a: Them|Test, L50|100/100"]
        provider.ingest(lines)
        before = self.session.summary()
        provider.ingest(lines)
        self.assertEqual(before, self.session.summary())
        provider.ingest(["|turn|99"])
        self.assertEqual(provider.failure, "protocol_history_changed")

    def test_posterior_drives_world_sampling_without_reading_opponent_side(self):
        self.session.observe(SpeedEvidence("order", "p2:test", 160, True, supported=True))
        provider = MechanicsWorldProvider(shadow=False, session=self.session, rng=random.Random(7))
        provider.observer = self.observer
        provider.own_side = "p1"
        own_side = types.SimpleNamespace(force_switch=False)
        class PrivateState:
            side_one = own_side
            @property
            def side_two(self):
                raise AssertionError("must not import Foul Play's inferred opponent state")
        def pokeset(*args):
            return types.SimpleNamespace(species=args[0], ability=args[3], item=args[4], tera=args[5], moves=args[6])
        def pokemon(s):
            return types.SimpleNamespace(maxhp=100, hp=100, ability=s.ability, item=s.item,
                                         tera_type=s.tera, moves=s.moves)
        fake_rb = types.SimpleNamespace(PokeSet=pokeset, to_pokemon=pokemon,
                                        sample_team=lambda rng, n, exclude: [pokeset("Bench", "", 50, "levitate", "none", "normal", ())] * n)
        fake_pe = types.SimpleNamespace(Side=lambda **kw: types.SimpleNamespace(**kw),
                                        State=lambda **kw: types.SimpleNamespace(**kw),
                                        Move=lambda **kw: types.SimpleNamespace(**kw))
        fake_hd = types.SimpleNamespace(base_of=lambda x: x.lower(), CHOICE_ITEMS={"choicescarf"},
                                        copy_side=copy.deepcopy,
                                        copy_pokemon=lambda p, **kw: types.SimpleNamespace(**{**vars(p), **kw}))
        with patch.dict(sys.modules, {"poke_engine": fake_pe, "randbats": fake_rb, "hidden": fake_hd}):
            worlds = provider.sample_states(None, lambda b: PrivateState(), 4)
        self.assertEqual(len(worlds), 4)
        self.assertTrue(all(state.side_two.pokemon[0].item == "choicescarf" for state, _ in worlds))
        self.assertEqual([w for _, w in worlds], [.25] * 4)

    def test_same_public_transcript_same_belief(self):
        # Hidden teams/commands are deliberately absent from the adapter interface.
        transcript = ["|move|p2a: Them|Surf|p1a: Me", "|cant|p1a: Me|slp", "|turn|2"]
        a = BeliefSession(catalog()); b = BeliefSession(catalog())
        for session in [a, b]:
            obs = ProtocolBeliefObserver(session, "p1")
            obs.feed(["|switch|p2a: Them|Test, L50|100/100"] + transcript)
        self.assertEqual(a.summary(), b.summary())


if __name__ == "__main__":
    unittest.main()
