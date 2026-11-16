# Figure vocabulary (one term per concept, in figures, legends and RESULTS.md)

| Term | Meaning | Code spec |
|---|---|---|
| Pure maximin move | plays the pure maximin (security) row of the turn's payoff matrix | `safest:*` |
| Equilibrium, sampled | play a Nash equilibrium of the turn's matrix by sampling its mix | `nash:rm+:*` |
| Equilibrium, top move | play the equilibrium's most probable action | `nash:...:argmax` |
| MCTS | poke-engine Monte Carlo tree search, 2,000 iterations, most-visited move | `mcts_it:2000` |
| Full matrix | unpruned depth-2 payoff matrix (`nash.full_matrix`) | default |
| Pruned matrix | poke-engine iterative-deepening matrix with pruned cells set to 0 (bug) | `:legacy` |
| Bayesian LP | Bayesian matrix game over 4 sampled worlds solved by its maximin LP (the one-sided case of Ponssard and Sorin's LP); the Bayesian-LP agent. Old term: Agent-form | `agentform:4:*` |
| PIMC averaging | solve each of 4 sampled worlds separately, average the strategies | `pimc:4:*` |
| Most-likely set | solve one world with each revealed Pokémon's most probable set | `mostlikely:*` |
| Oracle | full-information equilibrium on the true state | `oracle:*` |
| Sampled play / Top-move play | agents sample their strategy / play its top action | (none) / `:argmax` |
| PIMC-averaging loss | Bayesian LP value minus the Bayesian-matrix-game objective of the PIMC averaging strategy | E5 logs |
| Bound 4·δ_H | 4 × Hausdorff distance between the worlds' payoff-column sets | E5 logs |
| Score | wins + ½ draws, per game, averaged over game pairs | |
| Self-play RL | PPO self-play pilot (12-species pool, full information) | `ppo_pilot.py` |
| ε-safe best response | best reply (Ganzfried and Sandholm 2015) to the predicted opponent strategy among our strategies that guarantee at least the turn's equilibrium value minus ε times the turn's payoff range | `bayes_safex` (`engine/showdown/run_ourbot.py`) |
| Stage equilibrium | opponent style (and case-study label): plays a Nash equilibrium of the turn's matrix | (case-study figures) |
| Best response to uniform | opponent style: best-responds to a uniformly random guess about our move | (case-study figures) |
| Logit best response to our mixture | opponent style: logit best response to our reference mixture (our strategy before the exploit step) | `br_us` (`engine/showdown/behavior.py`) |
| Cautious (maximin) | opponent style: plays its pure maximin (security) move | (case-study figures) |
| Uniform random | opponent style: picks uniformly among its options | (case-study figures) |
| Ours | our agent (the Hedge/equilibrium player) in game results | (none) |

## Proposed (fig_gauntlet, fig_switch; not yet accepted)

| Term | Meaning | Code spec |
|---|---|---|
| Bayesian LP, top move | play the Bayesian LP's most probable action | `top` (`engine/showdown/run_ourbot.py`) |
| Max damage | max-damage baseline bot (highest base power times same-type bonus (STAB) times type effectiveness) | `maxpower` (`engine/showdown/simple_bots.py`) |
| Uniform random | (bot) picks uniformly among its options; same term as the opponent style | `random` (`engine/showdown/simple_bots.py`) |
| Foul Play, N ms | Foul Play bot with N ms of search per sampled battle | `stock@N` (`engine/showdown/headless.py`) |
| Rating (Elo) | Bradley-Terry rating on the Elo scale (400 log10), Bayesian LP = 0 | `gauntlet.py` |
| Realised time | updates after the switch until the posterior's top style is uniform random for 20 straight updates | `tau_map` |
| Predicted time | detection delay predicted from the log-posterior gap at the switch | `tau_hat` |
| λ | posterior forgetting factor (λ = 1: plain Bayes; λ = 0.98: forgetting) | `lam` |
