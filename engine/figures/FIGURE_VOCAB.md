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
