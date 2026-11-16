# Figure vocabulary (one term per concept, in figures, legends and RESULTS.md)

| Term | Meaning | Code spec |
|---|---|---|
| Safest move | pure maximin on the turn's payoff matrix | `safest:*` |
| Equilibrium, sampled | play a Nash equilibrium of the turn's matrix by sampling its mix | `nash:rm+:*` |
| Equilibrium, top move | play the equilibrium's most probable action | `nash:...:argmax` |
| MCTS | poke-engine Monte Carlo tree search, 2,000 iterations, most-visited move | `mcts_it:2000` |
| Full matrix | unpruned depth-2 payoff matrix (`nash.full_matrix`) | default |
| Pruned matrix | poke-engine iterative-deepening matrix with pruned cells set to 0 (bug) | `:legacy` |
| Agent-form | Bayesian agent-form matrix game over 4 sampled worlds, one LP | `agentform:4:*` |
| PIMC blend | solve each of 4 sampled worlds separately, average the strategies | `pimc:4:*` |
| Most-likely set | solve one world with each revealed Pokémon's most probable set | `mostlikely:*` |
| Oracle | full-information equilibrium on the true state | `oracle:*` |
| Sampled play / Top-move play | agents sample their strategy / play its top action | (none) / `:argmax` |
| Blend loss | agent-form value minus the agent-form objective of the PIMC blend strategy | E5 logs |
| Bound 4·δ_H | 4 × Hausdorff distance between the worlds' payoff-column sets | E5 logs |
| Score | wins + ½ draws, per game, averaged over game pairs | |
| Self-play RL | PPO self-play pilot (12-species pool, full information) | `ppo_pilot.py` |
| Safe exploit | ε-safe exploit: best reply to the predicted opponent strategy among our strategies that guarantee at least the turn's equilibrium value minus ε times the turn's payoff range | `bayes_safex` (`engine/showdown/run_ourbot.py`) |
| Equilibrium | opponent style (and case-study label): plays a Nash equilibrium of the turn's matrix | (case-study figures) |
| Best reply to uniform | opponent style: best-responds to a uniformly random guess about our move | (case-study figures) |
| Best reply to us | opponent style: soft best reply to our reference mixture (our strategy before the exploit step) | `br_us` (`engine/showdown/behavior.py`) |
| Cautious (maximin) | opponent style: plays the maximin (safest) move | (case-study figures) |
| Uniform random | opponent style: picks uniformly among its options | (case-study figures) |
| Ours | our agent (the Hedge/equilibrium player) in game results | (none) |

## Proposed (fig_gauntlet, fig_switch; not yet accepted)

| Term | Meaning | Code spec |
|---|---|---|
| Agent-form, top move | play the agent-form LP's most probable action | `top` (`engine/showdown/run_ourbot.py`) |
| Max damage | max-damage baseline bot (highest base power times type effectiveness) | `maxpower` (`engine/showdown/simple_bots.py`) |
| Uniform random | (bot) picks uniformly among its options; same term as the opponent style | `random` (`engine/showdown/simple_bots.py`) |
| foul-play, N ms | stock Foul Play bot with N ms of search per state | `stock@N` (`engine/showdown/headless.py`) |
| Rating (Elo) | Bradley-Terry rating on the Elo scale (400 log10), agent-form = 0 | `gauntlet.py` |
| Realised time | updates after the switch until the posterior's top style is uniform random for 20 straight updates | `tau_map` |
| Predicted time | re-concentration time predicted from the log-posterior gap at the switch | `tau_hat` |
| λ | posterior forgetting factor (λ = 1: plain Bayes; λ = 0.98: forgetting) | `lam` |
