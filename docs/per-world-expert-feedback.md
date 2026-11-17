# Per-world safest expert and learning-credit correction

Implemented November 17, 2026. The correction removes two inconsistent treatments of sampled worlds in `hedge_bayes_mw`. It also corrects exact action matching in both `*_mw` methods. It does **not** establish improved win rate or eliminate the setup mistakes.

## Design

For row-aligned payoff matrices `M_k` and normalized world weights `w_k`, the pure safest expert now selects

```
argmax_a sum_k w_k min_b M_k[a, b].
```

This is the pure-action restriction of the existing agent-form objective: the opponent can respond differently in each hidden world. Previously this expert minimized a column of the averaged union matrix, which couples responses across worlds and conditions different columns on different membership sets. The agent-form sampled expert and per-world safe-exploitation expert were already using per-world objectives.

Hedge previously assigned all credit using the highest-prior-weight world containing the observed action. Equal-weight particles therefore made this update depend on world ordering. Meanwhile, the style posterior already used a likelihood marginalized across worlds.

The corrected update conditions world weights on the exact observed action `b`:

```
alpha_k = w_k qhat_k(b) / sum_l w_l qhat_l(b).
r_e = sum_k alpha_k clip((Q_e M_k[:, b] - min(M_k)) / 400, 0, 1).
```

An absent action contributes zero likelihood. `qhat_k` and the expert strategies `Q` are saved at decision time, before observing the action or updating either learner. There is **one** Hedge update per observation, not one per world. The existing per-world bounded reward is preserved, with clipping before averaging, giving an exact one-world reduction.

Under the existing factorized world/style model, `alpha_k` and the style posterior update are the two marginals of the same temporary joint posterior:

```
P(k, style | b) proportional to w_k P(style) q_style,k(b).
```

This conditions credit for one turn; it does not introduce a carried joint set × style particle filter or change the world sampler. With forgetting enabled, style memory additionally applies its existing discount rule.

Exact action matching is important: the old mapper could select a plain move in an earlier/higher-weight world even when another world contained the observed tera move. The `*_mw` paths now use exact labels across all worlds, and skip unsupported observations. The legacy methods retain their previous mapping and credit rules.

## First-100 replay findings

These are separate, frozen-input ablations over the completed 52–48 public batch. They hold logged expert weights/strategies/predictions fixed to isolate each correction. Full-precision payoff matrices are available; logged expert distributions, weights, and predictions have four-decimal precision and are renormalized for this analysis.

| Measurement | Result |
| --- | ---: |
| Decisions with expert snapshots | 2,951 |
| Safest label changes | 496 |
| Strict improvement in pure per-world security, tolerance 0.001 | **495 (16.8%)** |
| Scored feedback updates | 2,384 |
| Updates supported by multiple worlds | 2,299 |
| Credit changes by more than 0.00001 for at least one expert | **1,444 (60.6%)** |
| Strictly disjoint best-credit expert sets, excluding ties | **77 (3.2%)** |
| Mean maximum absolute expert reward change per supported update | 0.0172 |
| Largest maximum reward change | 0.9184 |

There are 2,950 paired feedback records. Of these, 214 were outside sampled action support, 201 were censored by an opponent KO, 66 by `cant`, and 85 had no observed action. One final pending decision has no paired feedback record. The new exact-label rule does not drop any additional previously scored observation in these 100 games. The tera-mapping defect is demonstrated by targeted integration tests, rather than an occurrence in this cohort.

For example, battle `2715117197`, turn 11, observed Knock Off: the old world-0 credit preferred `top`/`safest` (0.844) over `sampled` (0.684). Conditioning across all four worlds, with weights approximately `[.220, .261, .249, .270]`, instead credits `sampled` at 0.915 versus 0.829 for `top`/`safest`. This is a change in modeled counterfactual credit, not an observed battle reward.

Correcting the pure expert does **not** guarantee that the final Hedge mixture becomes safer in every position. Replacing only that expert at the recorded weights improves the mixture's per-world worst-case value in 311 decisions and worsens it in 182, with a mean change of +0.357 evaluation units over all 2,951 decisions. Mixture security depends on how all experts' strategies complement each other. The existing final Hedge mixture has no overall agent-form safety constraint; the fourth expert's constraint alone does not provide one for the full mixture.

## Replaying the actual live policy

Seven flagged positions with multiple legal actions were also passed through the pre-change and corrected **production move functions**, using saved matrices and reconstructed, frozen learning memory. This includes recomputing the reference mixture, style predictions, and safe-exploitation expert. The pre-change reconstruction differs from logged action probabilities by at most 0.0001.

- **Rhyperior Earthquake into active Drifblim**, `2715109969`, turn 4: the safest expert changes to Stone Edge; Earthquake probability falls **41.41% → 18.74%**. Other experts still assign it probability.
- **Lucario Nasty Plot into Weavile**, the same battle, turn 14: probability remains **20.05%**.
- Glaceon Calm Mind and Iron Leaves Swords Dance against Sinistcha remain **85.39%** and **96.55%**. Blaziken and Delphox's roughly 1.5% setup selections also remain.

The implementation repairs the identified inconsistency. It does not, by itself, resolve evaluator preferences or all harmful mixture selections. These replays do not simulate subsequent learning or changed battle trajectories.

## Additional feedback limitation: reward saturation

In **95/2,384** supported updates, at least one expert's reward clips in a supported world. In 72, every expert saturates at one in every supported world; 24 updates give all experts identical conditional rewards despite different expected raw utilities.

The fixed scale of 400 can therefore erase distinctions, particularly around large terminal-value ranges. Clipping also means the highest expected raw-utility expert need not receive the highest bounded reward. This was deliberately left unchanged to isolate uncertainty handling. Auditing the reward scale/objective is a separate follow-up; more training cannot recover distinctions removed by this transformation. Credit remains based on model values, not learned full-game win probabilities, and censored/unsupported actions remain excluded.

## Validation and operation

- **103 local unit tests pass**, including analytic credit examples, total expectation, joint-posterior marginal consistency, world/column permutation, duplicated-particle invariance, zero support, exact tera labels, production selection, final feedback exactly once, and checkpoint preflight.
- Existing `hedge_bayes` and `bayes_safex` retain identical moves, log records excluding timing, and state files on **12 controlled 40-decision sequences** against the preserved baseline.
- Imported per-world checks pass: **2,616 support checks, 2,000 one-world prediction/update checks, 600 one-world safe-exploitation reductions**, plus eight exact-tera feedback checks through both live methods.
- Two local headless games against `agentform`, seed 4 with identical teams and swapped seats, finish without simulator errors: corrected method wins both, in 35 and 43 turns. This is an integration smoke test, **not a strength estimate**. Its 88 decisions produce 75 credited updates, 74 involving multiple worlds. Credit weights sum to one within `2.3e-16`; their predictive probability agrees with the style-feedback log within its six-decimal precision.

The shared bounded-reward calculation is in `engine/exp3_live.py`; pure selection and conditional credit are in `engine/showdown/world_learning.py`; the live integration is in `run_ourbot.py`. Decision/feedback logs now record the semantic version, pure security values, conditional weights, predictive mass, and raw utilities/rewards.

Corrected Hedge checkpoints carry `per_world_security_action_conditional_credit_v2`. Legacy `hedge_bayes_mw` checkpoints are rejected rather than silently mixing rewards from different expert definitions. **Use a fresh output directory for a v2 run.** Concurrent ladder startup validates checkpoints before connecting, so this rejection cannot first occur during an active game. The first-100 artifacts/checkpoints were preserved. The queued continuation of that old batch still awaits its Terminal login and cannot reuse its legacy checkpoints with this revision; no new public batch was started for this change.

The compact results are in [the checked-in JSON](../research/per_world_credit_v2_results.json). Full audit, baseline source copies, seven policy replays, and headless logs are under `.local-play/work/per-world-credit-v2/`.

Reproduce the frozen-input audit from the repository root:

```sh
OUT=.local-play/work/ladder/decision_audit_hedge_bayes_mw_20261116T233404Z
.local-play/venv/bin/python engine/showdown/audit_world_learning.py "$OUT" \
  --limit 100 --write .local-play/work/per-world-credit-v2/audit_first100.json
.local-play/venv/bin/python engine/showdown/replay_world_policy.py "$OUT" \
  --baseline .local-play/work/per-world-credit-v2/baseline/engine/showdown/run_ourbot.py \
  --write .local-play/work/per-world-credit-v2/policy_replay_first100.json
PYTHONWARNINGS=ignore::ResourceWarning .local-play/venv/bin/python -m unittest discover -s engine/tests -q
```
