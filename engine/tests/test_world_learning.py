"""Analytic and live-path regressions for per-world expert selection and credit."""
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FP = Path(os.environ.get("FOULPLAY_DIR", ROOT / ".local-play/foul-play"))
sys.path[:0] = [str(ROOT / "engine"), str(ROOT / "engine/showdown"), str(FP)]
from exp3_live import Learner, expert_rewards
from world_learning import (SEMANTICS, condition_on_action, conditional_expert_credit,
                            pure_security_values)


class WorldLearningTests(unittest.TestCase):
    def setUp(self):
        self.Ms = [np.array([[100., 0.], [0., 0.]]), np.array([[0., 0.], [200., 0.]])]
        self.cols = [["attack", "wait"], ["attack", "wait"]]
        self.w = np.array([.8, .2])
        self.qk = [np.array([.1, .9]), np.array([.9, .1])]
        self.Q = np.eye(2)

    def test_pure_security_matches_agent_form_restricted_to_pure_actions(self):
        # Averaging the matrices first wrongly gives row 0 security 50; its
        # world-contingent opponent can instead secure 0 in either world.
        Ms = [np.array([[100., 0.], [40., 40.]]), np.array([[0., 100.], [40., 40.]])]
        self.assertEqual(int(((Ms[0] + Ms[1]) / 2).min(1).argmax()), 0)
        np.testing.assert_array_equal(pure_security_values(Ms, [1, 1]), [0, 40])
        self.assertEqual(int(pure_security_values(Ms, [1, 1]).argmax()), 1)

    def test_choice_handles_world_specific_columns_and_zero_weight_worlds(self):
        Ms = [np.array([[2.], [1.]]), np.array([[-100., 100.], [0., 0.]])]
        np.testing.assert_array_equal(pure_security_values(Ms, [1, 0]), [2, 1])
        np.testing.assert_array_equal(pure_security_values(Ms, [1, 1]), [-49, .5])

    def test_action_posterior_can_reverse_prior_world_ranking(self):
        cond = condition_on_action(self.cols, self.w, self.qk, "attack")
        self.assertAlmostEqual(cond["probability"], .26)
        np.testing.assert_allclose(cond["weights"], [4/13, 9/13])
        reward, utility = conditional_expert_credit(self.Ms, self.Q, cond)
        np.testing.assert_allclose(reward, [1/13, 4.5/13])
        np.testing.assert_allclose(utility, [400/13, 1800/13])
        # The previous highest-prior-world credit preferred the opposite expert.
        self.assertEqual(expert_rewards(self.Ms[0], self.Q, 0).argmax(), 0)
        self.assertEqual(reward.argmax(), 1)

    def test_unavailable_and_zero_probability_actions_do_not_get_credit(self):
        self.assertIsNone(condition_on_action(self.cols, self.w, self.qk, "attack-tera"))
        self.assertIsNone(condition_on_action([["x"], ["attack"]], [1, 0], [[1], [1]], "attack"))
        self.assertIsNone(condition_on_action(self.cols, self.w, [[0, 1], [0, 1]], "attack"))

    def test_exact_tera_identity_beats_higher_weight_plain_move(self):
        cond = condition_on_action([["attack"], ["attack-tera"]], [.9, .1], [[1], [1]], "attack-tera")
        self.assertEqual(cond["indices"], {1: 0})
        np.testing.assert_array_equal(cond["weights"], [0, 1])
        self.assertAlmostEqual(cond["support_weight"], .1)

    def test_total_expectation_over_observed_actions_recovers_prior_reward(self):
        result = np.zeros(2)
        direct = np.zeros(2)
        for j, label in enumerate(self.cols[0]):
            cond = condition_on_action(self.cols, self.w, self.qk, label)
            result += cond["probability"] * conditional_expert_credit(self.Ms, self.Q, cond)[0]
            for k in range(2):
                direct += self.w[k] * self.qk[k][j] * expert_rewards(self.Ms[k], self.Q, j)
        np.testing.assert_allclose(result, direct)

    def test_world_order_and_splitting_identical_particles_do_not_change_credit(self):
        base = condition_on_action(self.cols, self.w, self.qk, "attack")
        r, _ = conditional_expert_credit(self.Ms, self.Q, base)
        for ix, w in (([1, 0], [.2, .8]), ([0, 1, 0], [.4, .2, .4])):
            Ms, cols, qs = ([items[i] for i in ix] for items in (self.Ms, self.cols, self.qk))
            cond = condition_on_action(cols, w, qs, "attack")
            np.testing.assert_allclose(conditional_expert_credit(Ms, self.Q, cond)[0], r)
            np.testing.assert_allclose(pure_security_values(Ms, w), pure_security_values(self.Ms, self.w))

    def test_column_permutation_does_not_change_credit(self):
        base = condition_on_action(self.cols, self.w, self.qk, "attack")
        perm = condition_on_action([c[::-1] for c in self.cols], self.w,
                                   [q[::-1] for q in self.qk], "attack")
        np.testing.assert_allclose(conditional_expert_credit([M[:, ::-1] for M in self.Ms], self.Q, perm)[0],
                                   conditional_expert_credit(self.Ms, self.Q, base)[0])

    def test_one_world_reduces_bitwise_to_original_learner_for_all_step_rules(self):
        rng = np.random.default_rng(71)
        for cfg in ({}, dict(eta=.2, alpha=.002), dict(scale=None)):
            old, new = Learner(experts=range(4), **cfg), Learner(experts=range(4), **cfg)
            for _ in range(30):
                M = rng.normal(0, 500, (5, 3))
                Q = rng.dirichlet(np.ones(5), size=4)
                j = int(rng.integers(3))
                cond = condition_on_action([["a", "b", "c"]], [1], [[.2, .3, .5]], "abc"[j])
                r, _ = conditional_expert_credit([M], Q, cond, new.scale)
                old.update(M, Q, j)
                new.update_rewards(r)
                np.testing.assert_array_equal(new.logw, old.logw)
                np.testing.assert_array_equal(new.cum, old.cum)
                np.testing.assert_array_equal(pure_security_values([M], [1]), M.min(1))

    def test_clipping_is_inside_expectation_and_raw_utility_is_separate(self):
        Ms = [np.array([[1000.], [0.]]), np.array([[0.], [100.]])]
        cond = condition_on_action([["x"], ["x"]], [1, 1], [[1], [1]], "x")
        r, u = conditional_expert_credit(Ms, self.Q, cond)
        np.testing.assert_allclose(r, [.5, .125])
        np.testing.assert_allclose(u, [500, 50])

    def test_forced_action_keeps_prior_world_weights(self):
        cond = condition_on_action([["wait"], ["wait"]], self.w, [[1], [1]], "wait")
        np.testing.assert_array_equal(cond["weights"], self.w)

    def test_credit_world_posterior_and_style_update_are_marginals_of_same_joint(self):
        import behavior
        rng = np.random.default_rng(431)
        for _ in range(30):
            post = behavior.WorldPosterior()
            post.logp = np.log(rng.dirichlet(np.ones(len(behavior.STYLES))))
            P = post.probs()
            qs = [rng.dirichlet([1., 1.], size=len(P)) for _ in range(2)]
            qk = [P @ q for q in qs]
            cond = condition_on_action(self.cols, self.w, qk, "attack")
            joint = np.stack([self.w[k] * P * qs[k][:, 0] for k in range(2)])
            joint /= joint.sum()
            np.testing.assert_allclose(cond["weights"], joint.sum(1))
            post.update_worlds(qs, {0: 0, 1: 0}, self.w)
            np.testing.assert_allclose(post.probs(), joint.sum(0))

    def test_invalid_weights_and_mismatched_world_counts_fail(self):
        for weights in ([0, 0], [-1, 2], [float("nan"), 1]):
            with self.assertRaises(ValueError):
                pure_security_values(self.Ms, weights)
        with self.assertRaises(ValueError):
            condition_on_action(self.cols[:1], self.w, self.qk, "attack")


@unittest.skipUnless(importlib.util.find_spec("poke_engine"), "poke-engine unavailable")
class LiveWorldLearningTests(unittest.TestCase):
    def test_feedback_uses_saved_predictions_once_including_final_callback(self):
        import run_ourbot as ro
        import nash
        import behavior
        Ms = [np.array([[100., 0.], [0., 0.]]), np.array([[0., 0.], [200., 0.]])]
        cols = [["attack", "wait"], ["attack-tera", "wait"]]
        worlds = [(NS(M=M, s1=["tackle", "growl"], s2=c), w)
                  for M, c, w in zip(Ms, cols, [.8, .2])]
        b = NS(team_preview=False, turn=1, _ourlines=[], opponent=NS(name="p2"),
               request_json=None, user=NS(reserve=[]), battle_tag="test-credit",
               mode=NS(prepare_battles=lambda battle, k: worlds))
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            stack.enter_context(patch.object(ro, "battle_to_poke_engine_state", lambda value: value))
            stack.enter_context(patch.object(nash, "full_matrix", lambda st, **kw: (st.M, st.s1, st.s2)))
            stack.enter_context(patch.object(nash, "pe", NS(id=lambda st, side: NS(s1=st.s1))))
            stack.enter_context(patch.dict(os.environ, OURBOT_LOG_ABAR="1", OURBOT_LOG_STATES="0"))
            log, state = Path(tmp)/"decisions.jsonl", Path(tmp)/"states/new.json"
            fn = ro.make_move_fn("hedge_bayes_mw", 2, log=str(log), state_path=str(state))
            fn(b)
            first = json.loads(log.read_text().splitlines()[0])
            finish = ro._FINISHERS.pop(b.battle_tag)
            finish(["|-terastallize|p2a: Foo|Fire", "|move|p2a: Foo|Attack|p1a: Bar", "|win|Winner"])
            fb = json.loads(log.read_text().splitlines()[-1])["feedback"]
            self.assertEqual(fb["credit"]["world_weights"], [0, 1])
            self.assertEqual(fb["bayes"]["worlds_with"], [1])
            self.assertEqual(fb["bayes"]["col"], "attack-tera")
            self.assertAlmostEqual(fb["credit"]["predictive_probability"],
                                   .2 * first["bayes"]["qhat_k"][1][0], places=4)
            persisted = json.loads(state.read_text())
            self.assertEqual(persisted["t"], 1)
            self.assertEqual(persisted["semantics"], SEMANTICS)
            finish(["|win|Winner"])
            self.assertEqual(json.loads(state.read_text())["t"], 1)
            # A missing tera label must be skipped, not credited as a plain move.
            worlds[:] = [(NS(M=Ms[0], s1=["tackle", "growl"], s2=cols[0]), 1)]
            fn(b)
            ro._FINISHERS.pop(b.battle_tag)(["|-terastallize|p2a: Foo|Fire", "|move|p2a: Foo|Attack|p1a: Bar"])
            last = json.loads(log.read_text().splitlines()[-1])["feedback"]
            self.assertTrue(last["skipped"])
            self.assertEqual(last["reason"], "not_in_worlds")
            self.assertEqual(json.loads(state.read_text())["t"], 1)

    def test_checkpoint_semantics_prevent_silent_reuse_of_old_rewards(self):
        import run_ourbot as ro
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/"state.json")
            L, meta = ro.load_learner("hedge_bayes_mw", None)
            ro.save_learner("hedge_bayes_mw", path, L, meta)
            self.assertEqual(ro.load_learner("hedge_bayes_mw", path)[0].t, 0)
            data = json.loads(Path(path).read_text())
            data.pop("semantics")
            Path(path).write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "fresh state path"):
                ro.load_learner("hedge_bayes_mw", path)
            # Historical non-mw methods retain their checkpoint contract.
            data["method"] = "hedge_bayes"
            Path(path).write_text(json.dumps(data))
            self.assertEqual(ro.load_learner("hedge_bayes", path)[0].t, 0)

    def test_live_policy_uses_the_corrected_security_objective(self):
        import run_ourbot as ro
        from replay_world_policy import replay_policy
        record = dict(method="hedge_bayes_mw", turn=1,
            weights=dict.fromkeys(ro.HEDGE_EXPERTS, .25),
            bayes=dict(posterior=dict.fromkeys(ro.behavior.STYLES, .2)),
            search=dict(legal=["tackle", "growl"], rows=[["tackle", "growl"]]*2,
                cols=[["attack", "wait"]]*2, weights=[.5, .5],
                matrices=[[[100, 0], [40, 40]], [[0, 100], [40, 40]]]))
        result = replay_policy(ro, record)
        self.assertEqual(result["experts"]["safest"], "growl")
        self.assertEqual(result["Q"][2], [0, 1])
        self.assertEqual(result["pure_security"], [0, 40])

    def test_concurrent_preflight_rejects_legacy_state_before_connecting(self):
        import run_ourbot as ro
        import ladder
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"learning/hedge_bayes_mw-stream0.json"
            L, meta = ro.load_learner("hedge_bayes_mw", None)
            ro.save_learner("hedge_bayes_mw", str(path), L, meta)
            data = json.loads(path.read_text()); data.pop("semantics")
            path.write_text(json.dumps(data))
            before = path.read_text()
            with patch.object(ladder, "FP", str(FP)), patch.object(ro, "LOG", ro.LOG), patch.object(ro, "REPLAYS", ro.REPLAYS):
                with self.assertRaisesRegex(ValueError, "fresh state path"):
                    ladder.setup_runtime(dict(out=tmp, method="hedge_bayes_mw", concurrency=5, k=2, local=True))
            self.assertEqual(before, path.read_text())


if __name__ == "__main__":
    unittest.main()
