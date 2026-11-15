# PSRO / RL best response: cost estimate on Modal (CPU only)

Made 2026-11-14 from `psro_cost.py` (`.venv/bin/python psro_cost.py`, or `psro.py --config <preset> --dry-run`).
No games were played to make this estimate. Every number comes from logs that already existed.

## Pricing (fetched from https://modal.com/pricing on 2026-11-14)
- CPU: $0.0000131 per physical core per second, which is $0.0472 per core-hour. One physical core counts as 2 vCPU. Each container has a minimum of 0.125 cores.
- Memory: $0.00000222 per GiB per second, which is $0.0080 per GiB-hour.
- Billing is for requested resources over the whole time a container is alive, including idle time.
- The Starter plan includes $30 of compute per month.
- Containers: workers get 1 core and 1 GiB each, and there are 32 of them. The driver (PPO learner) gets 2 cores and 4 GiB.

## Throughput (measured on the local Apple M4 Max)
| Quantity | Value | Source |
|---|---|---|
| PPO rollout | 2,470 decisions/s per worker | `results/ppo_pilot.jsonl` `dec_per_s`, median 29,644 with 12 workers |
| Decisions per game | 87.6 for both sides, so 43.8 for the BR side | same file, last record: `decisions` / `games` = 29,851,761 / 340,800 |
| PPO update | 8.2 µs per decision | same file, sum of `upd_s` (244.5 s) / decisions |
| Equilibrium, sampled (`nash:rm+:0:1000`) | 74.3 ms per move | `results/experiments/E1fix.jsonl` `ms_per_move_A` |
| Equilibrium, top move (`...:argmax`) | 72.4 ms per move | same |
| Safest move (`safest:0`) | 67.3 ms per move | same |
| MCTS 2,000 iterations | 2.7 ms per move | same, `ms_per_move_B` |
| `hedge_experts` | 74.3 ms per move (inferred) | no timing was logged. It makes the same calls as `nash:rm+:0:1000`: one `full_matrix` plus `rm_plus(1000)` per turn |
| Turns per game | 60 (assumed) | E1fix mean `turns` is 52–70, on randbats teams with a 300-turn cap |

Sanity check: in E1fix, nash vs safest has a mean `wall_s` of 8.9 s. The model gives 60 × (74 + 69) ms ≈ 8.6 s.

**Assumptions (calibrate them with the smoke run):**
- A Modal core is 2.0× slower than an M4 Max core.
- Container utilization is 0.6, because containers stay billed while the driver waits for straggler games.
- During BR training, all the meta-Nash weight sits on the slowest bot. This is the worst case.

## Configurations (32 containers)
| Config | PSRO iters × BR decisions | Games per cell | rl_br targets × decisions | Games | Decisions trained | Local core-h | Modal core-h | $ | Wall |
|---|---|---|---|---|---|---|---|---|---|
| (a) smoke | 1 × 60k | 30 | hedge × 60k (60 eval games) | 3,100 | 120k | 3.9 | 13.1 | **$0.77** | 0.4 h |
| (b) first real run | 3 × 200k | 60 | nash sampled + hedge, 250k each (200 eval games) | 26,774 | 1.1M | 33.1 | 110 | **$6.52** | 3.4 h |
| (c) fuller | 8 × 2M | 200 | both targets, 2M each (1,000 eval games) | 471,820 | 20M | 579 | 1,930 | **$114** | 60 h |

Sensitivity for (b):
- Slowdown 1.5 / 2 / 3 at utilization 0.6 costs $4.89 / $6.52 / $9.77.
- At slowdown 2 with utilization 0.4, it costs $9.77.
- Running only the two rl_br targets costs $2.90.
- The `--max-usd` guard stops the run cleanly if these assumptions turn out too optimistic.

## Dominant cost
The dominant cost is **BR training against the matrix bots**, not the meta matrix. Each PPO training game against a full-matrix bot costs about 60 × 74 ms ≈ 4.5 s of that bot's thinking. For comparison, the net side costs about 25 ms per game. In (b), BR training plus rl_br training is 31 of the 33 local core-hours. The meta matrix (initial cells plus the new rows) is only 1.3 core-hours.

Mitigations:
1. Use fewer PPO decisions per BR. Warm starts from the pilot or the previous BR make this workable.
2. Exclude the matrix bots from the BR training opponents. This is cheap, but the result is then no longer a best response to the meta-Nash.
3. Count on the mixture shifting. As BR nets enter the population and take meta-Nash weight, training games get cheaper (a net opponent costs about 0 ms per move). The estimate assumes the worst case.
4. Use a one-entry `full_matrix` cache per turn. This halves the cost of bot-vs-bot cells only, which are a small share.
5. Run 2 processes per physical core. Each core is 2 vCPU, so this may give about 1.2–1.3× throughput (not measured).
6. Use fewer games per cell. 60 games gives a CI of about ±0.13 on a cell.
