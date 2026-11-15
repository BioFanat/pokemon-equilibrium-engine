# The game: Pokémon Showdown Gen 9 Random Battles (singles) as a formal object

Draft 2026-11-15. Preliminaries, part 1 (the game). Part 2 is the no-regret preliminaries (Lectures 6/7,
`theory_map.md` §1.2). Notation follows `theory_map.md` §0.1; assumptions A1–A9 are `theory_map.md` §2 and are
reused by number; A10–A12 are new here (§5).

**Sources and citation keys.** Every mechanic claim carries a `path:line` pointer. Claims I could not source are
tagged **[unverified]** (grep for it; the full list is §8).

| Key | Source |
|---|---|
| `SD` | `~/sandbox/pokemon-showdown`, commit `7e74376` |
| `PE` | poke-engine 0.0.48 Rust source, `/private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/probe_corr/pe_src` (this copy carries probe counters in `src/mcts.rs:12–14,182–186`; they touch no kernel). Our build is `--features poke-engine/terastallization --no-default-features` (`EN showdown/requirements-bot.txt:2–3`), i.e. `gen9` + Tera |
| `PY` | the Python binding inside `PE`: `poke-engine-py/src/lib.rs` |
| `FP` | `~/sandbox/foul-play`, commit `8f541f8` |
| `EN` | `/Users/biofanat/Documents/classes/f26/cs535/final/engine` |
| `B34` | `.../scratchpad/bridges/b34/final_out.txt` (bridge probe, 21 games, 519 damage events) |

Status words: **verified** = read in the source at the pointer; **derived** = my arithmetic from verified facts;
**probe** = a small measurement I ran (script path given); **[unverified]** = stated without a source.

---

## 0. Notation

Symbols from `theory_map.md` §0.1 keep their meaning. New symbols are marked †.

| Symbol | Meaning |
|---|---|
| $G$ | the finite-horizon two-player zero-sum stochastic game with simultaneous moves and chance (M2 object) |
| $G_{\rm sd}$ †, $G_{\rm pe}$ † | the game as implemented by Showdown (M4) and by poke-engine + our loop (M2) |
| $s\in S$; $Z\subset S$ | state; terminal states |
| $a\in A(s)$, $b\in B(s)$ | our action (row, maximiser) and the opponent's (column). A bare $b$ is always an action |
| $h_t$ | public history $(s_0,a_0,b_0,\dots,s_t)$ (full-information form) |
| $t$; $H$; $\tau$ | **decision-node index** (one engine `step`, §1.4); horizon in decision nodes; realised length |
| $\kappa$ † | Showdown's turn counter (`battle.turn`); $\kappa\le t$ (§1.4) |
| $P(\cdot\mid s,a,b)$ | transition kernel; $P_{\rm pe}$, $P_{\rm sd}$ for the two simulators |
| $u$ | terminal payoff in $\{-1,0,+1\}$ (§1.6) |
| $\Theta$ †, $\theta$ † | type space (a full team of 6 sets) and the opponent's type; $\theta_1,\theta_2$ both teams |
| $b_0$; $b^{\rm hard}_h$; $p_h$ | prior over $\theta$; hard-evidence belief (`full_game.md` (6.2)); true posterior (`full_game.md` (6.1)). Beliefs always carry a subscript |
| $o^i_t$ † | player $i$'s observation at node $t$ (§2.1) |
| $\mathrm{pct}(h,M)$ † | displayed HP percentage of a Pokémon with $h$ of $M$ HP (§2.1) |
| $V^*_t,\ Q^*_t$ | game value and true stage matrix of $G$ |
| $\hat Q_t(s)$ | the stage matrix `nash.full_matrix` builds (§4) |
| $\omega$ † | a chance branch (one entry of `generate_instructions`' list) with probability $p_\omega$ (§4). `full_game.md` writes $\beta$ for the opponent's action; here $\beta$ is not used |
| $\ell(c)$ † | the leaf value of a child state $c$ (§4) |
| $\mathrm{ev}(s)$ † | poke-engine's heuristic evaluation (`PE src/genx/evaluate.rs:157–269`), engine units |
| $\Delta$; $\mathcal P(X)$ | payoff range of a matrix; probability simplex |
| $\varepsilon_s,\ \varepsilon_v,\ \varepsilon_{\rm sim}$ | solver gap; matrix error (A7); simulator mismatch (A5) |
| $\bar H$ | expected length (decision nodes) against a best response |
| $K$ | number of sampled worlds |

---

## 1. The game as a formal object

### 1.1 Definition

**Definition 1 (Random Battle game).** $G=(\Theta, b_0, S, Z, A, B, P, u, H)$ with

- types $\theta=(\theta_1,\theta_2)\in\Theta\times\Theta$ drawn once at the start from the team generator, independently
  per side (§1.7);
- a finite state set $S$ (§1.2) whose initial state $s_0(\theta)$ has both teams at full HP with slot 0 active;
- action menus $A(s)$, $B(s)$ (§1.3), either of which may be a singleton (§1.4);
- a kernel $P(s'\mid s,a,b)$ (§1.5);
- terminal states $Z$ with payoff $u:Z\to\{-1,0,1\}$, absorbing (§1.6);
- a horizon $H$ in decision nodes, with payoff $u_H\equiv0$ for a game live at $H$ (A6, §1.6).

Player 1 (us) maximises $\mathbb E[u]$, player 2 minimises it. Each player observes its own team exactly and a public
signal about the other (§2): $G$ is an imperfect-information game with private types. Its full-information
restriction (both $\theta$ public) is a finite-horizon zero-sum stochastic game in the sense of `full_game.md` §1,
**except at pivot nodes** (§1.4, A11).

### 1.2 State space

A state is the content of poke-engine's `State` (`PE src/state.rs:1321–1330`), plus the turn counter that Showdown
keeps and poke-engine does not.

| Component | Range (poke-engine field) | Source |
|---|---|---|
| per Pokémon (6 per side) | species `id`, `level`, `types`, `base_types`, `hp ≤ maxhp`, `ability`, `base_ability`, `item`, `nature`, `evs`, 5 stats, `status` (7 values incl. none), `rest_turns`, `sleep_turns`, `weight_kg`, `terastallized`, `tera_type`, 4 moves each with `pp` and `disabled` | `PE src/state.rs:852–876`, `541–546` |
| per side | `active_index`, `side_conditions` (19 counters: screens, hazards, tailwind, toxic count, …), `volatile_statuses` (a bitset over 104 enum values incl. `NONE`), `volatile_status_durations` (6 counters), 7 boost stages in $\{-6,\dots,6\}$, `substitute_health`, `wish`, `future_sight`, `force_switch`, `force_trapped`, `slow_uturn_move`, `baton_passing`, `shed_tailing`, `last_used_move`, `switch_out_move_second_saved_move` | `PE src/state.rs:1056–1079`, `601–621`, `787–794`; `PE src/genx/state.rs:162–267` |
| field | weather (8 values) and turns left; terrain (5 values) and turns left; trick room flag and turns left | `PE src/state.rs:733–775`; `PE src/genx/state.rs:270–295` |
| turn counter | Showdown only: `battle.turn`, incremented once per full turn | `SD sim/battle.ts:1640–1641` |

Boosts and volatiles live on the side, not the Pokémon, in poke-engine (they apply to the active slot). Showdown keeps
volatiles per Pokémon but clears them on switch-out; the two layouts carry the same information **[unverified]**
(Baton Pass and Shed Tail are the exceptions both engines special-case: `PE src/state.rs:1058–1059`).

**Proposition 1 (size, derived).** For a fixed type pair $\theta$, the number of representable live states is of order
$10^{150}$, of which about $10^{60}$ is move PP; over types, $|\Theta|$ is of order $10^{18}$ per side under
foul-play's empirical set support.

*Derivation (an order-of-magnitude upper bound on representable states, not on reachable states).* Per side,
$\log_{10}$ of the number of values: HP $6\log_{10}272\approx15$ (median max HP over the 509 species is 271, max 572,
Blissey; probe `scratchpad/game_model/sizes.py`); status with sleep/rest counters $6\log_{10}28\approx8.7$; PP
$24\log_{10}17\approx29.5$ (engine default 16 PP, `PY lib.rs:846`); boosts $7\log_{10}13\approx7.8$; active slot
$\log_{10}6$; item present/removed $6\log_{10}2$; Tera $\log_{10}7$; side conditions $\approx5$ (spikes 4 × toxic spikes
3 × rocks 2 × web 2 × three screens 6 each × tailwind 5); about 10 volatile flags in practice, $\approx3$. Two sides give
$\approx2\times74$; the field adds $\log_{10}(8\cdot8\cdot5\cdot8\cdot2\cdot5)\approx4.4$. Total $\approx152$; without PP
$\approx93$. Types: $\binom{509}{6}\approx10^{13.4}$ species sextuples times $\approx10^{5.1}$ (6 × mean $\log_{10}$ of the
distinct full sets per species, 5,802 sets over 509 species in `FP fp/data/pkmn_sets_cache/gen9randombattle.json`).
$\square$

*Remark.* The bitset over 104 volatiles makes the representable count far larger ($2^{103}$ per side); I count only
the flags that occur, so Proposition 1 is an estimate, not a bound on `State`.

### 1.3 Action sets

**Showdown.** A singles choice is `move i` ($i\le4$), `move i terastallize`, or `switch j`
(`SD sim/SIM-PROTOCOL.md:637–700`). Tera is available once per side: after one Pokémon terastallizes, every
teammate's `canTerastallize` is cleared (`SD sim/battle-actions.ts:2001–2004`; `canTerastallize` returns the Tera
type in Gen 9, `:1982–1987`). Restrictions: a hard-locked move (Outrage-type, recharge) leaves only that move and
traps (`SD sim/pokemon.ts:1105–1111`, `985–1004`); a Pokémon with no usable move gets the single choice Struggle
(`SD sim/pokemon.ts:1124–1126`); trapping hides switches (`:1136–1155`). A side that has nothing to decide receives
`{wait: true}` (`SD sim/battle.ts:1469–1476`); `default`/`undo` are meta-commands, not actions.

**poke-engine.** `MoveChoice` is `Move(i)`, `MoveTera(i)`, `Switch(j)` or `None`. For a normal node each usable move
(not disabled, PP > 0, not blocked by Encore/Taunt/Assault Vest) contributes `Move(i)` and, while the side has not
terastallized, `MoveTera(i)`; every living benched Pokémon contributes `Switch(j)` unless trapped
(`PE src/genx/state.rs:483–526`, `1047–1056`, `1070–1080`, `1272–1300`). Hence

$$|A(s)|\le 4+4+5=13,$$

attained (6.5% of decision nodes in random-vs-random play, probe `scratchpad/game_model/branching.py`). Special
cases, each with its source:

| Case | Showdown | poke-engine |
|---|---|---|
| replacement after a faint | `switch j` request with `forceSwitch` (`SD sim/battle.ts:1437–1446`) | menu = switches only; the other side gets `[None]` (`PE src/genx/state.rs:1253–1270`) |
| both actives fainted | both sides get a `forceSwitch` request, simultaneous | both menus = switches (`PE src/genx/state.rs:1256–1259`) |
| U-turn / Volt Switch / Flip Turn pivot | mid-turn `forceSwitch` request to the pivoting side (`SD sim/battle-actions.ts:1370–1372`; `sim/battle.ts:2937–2969`) | pivoting side: switches; other side: `[saved move]` or `[None]` (`PE src/genx/state.rs:1227–1251`) |
| must recharge | locked choice `recharge` (`SD sim/pokemon.ts:988`) | `[None]` (`PE src/genx/state.rs:1272–1276`) |
| charging two-turn move | locked to the move | `[Move(i)]` (`PE src/genx/state.rs:1277–1278`) |
| Choice item lock | other moves disabled | `DisableMove` instructions on the other slots (seen in `B34` debug output, `scratchpad/bridges/b34/debug_out.txt`) |
| Outrage-type lock | locked 2–3 turns, trapped | `LOCKEDMOVE` volatile traps (`PE src/genx/state.rs:1082–1090`); see §1.5 for duration |
| no usable move | Struggle | **no Struggle**: no `STRUGGLE` reference in `PE src/genx/*.rs`; an empty menu becomes `[None]` (`PE src/genx/state.rs:1210–1215`) or switches only |
| nothing to decide | `{wait: true}` | `[None]`, an explicit action |

### 1.4 Decision structure: where the game is simultaneous and where it is not

A Showdown **turn** is a sequence of choice requests. At the start of a turn both players commit simultaneously;
the queue is sorted and run (`SD sim/battle.ts:3057–3085`, `2997–3016`). Two events interrupt it with a request to a
subset of players:

1. **End-of-turn replacement.** In Gen 9, fainted actives are flagged for replacement only when the queue is empty,
   i.e. after residuals (`SD sim/battle.ts:2902–2911`, `2540–2549`). If one side fainted, only it chooses (the other
   receives `wait`); if both fainted, both choose simultaneously (`SD sim/battle.ts:1437–1446`, `1469–1476`).
2. **Mid-turn pivot.** A successful self-switching move sets `switchFlag` (`SD sim/battle-actions.ts:1370–1372`);
   the battle stops and requests a switch from that side (`SD sim/battle.ts:2937–2969`). The rest of the turn,
   including the opponent's **already committed** action if it has not run yet, is kept and appended after the
   switch (`SD sim/battle.ts:3064–3078`, comment at `:3060–3063`).

The turn counter increments once, after all of this (`SD sim/battle.ts:1640–1641` in `endTurn`, called when the
queue is exhausted, `:3008–3014`).

poke-engine flattens every request into one call of `generate_instructions_from_move_pair(s, a, b)`, which we count
as one **decision node**. At a replacement or pivot node the idle side's menu is a singleton (§1.3), and the
end-of-turn block is skipped for that call (`PE src/genx/generate_instructions.rs:4224–4233`;
`add_end_of_turn_instructions` also returns early while a `force_switch` flag is set, `:2818–2820`). In the next call
the switch clears the flag (`generate_instructions_from_switch`, `:248–260`) before the end-of-turn check, so the
turn's residuals are applied once, in the call that completes the turn. So $t$ counts decision nodes and $\kappa\le t$.

**Definition 2 (node types).** A decision node $s$ is

- **simultaneous** if $|A(s)|\ge2$ and $|B(s)|\ge2$;
- **one-sided** if exactly one menu is a singleton (replacement after a one-sided faint, a pivot, recharge, or a
  forced single move);
- **trivial** if both menus are singletons (`nash_policy` returns the only option without solving,
  `EN nash.py:178–180`).

In random-vs-random engine play, 19% of decision nodes have at least one singleton menu (probe, 4,188 nodes,
`scratchpad/game_model/branching.py`; random policies, not representative of our agents).

**Proposition 2 (what the stage matrix covers).** (a) At simultaneous and one-sided nodes of $G_{\rm pe}$, the
one-node game is a matrix game, $|A(s)|\times|B(s)|$, with $m=1$ or $n=1$ at one-sided nodes; Lemma 1 and Theorem A
apply with no change (a $1\times m$ or $n\times1$ matrix game has a pure optimal strategy for the side with the
menu). (b) At a **pivot node where the pivoting Pokémon moved first**, $G_{\rm pe}$ and $G_{\rm sd}$ have different
information structures. In $G_{\rm pe}$ the opponent's pending move is the state field
`switch_out_move_second_saved_move` (`PE src/genx/generate_instructions.rs:3670–3681`) and is the single column of
the matrix (`PE src/genx/state.rs:1227–1238`): the switching side sees it. In $G_{\rm sd}$ that move was committed at
the start of the turn and is not revealed until it runs (it sits in the saved queue, `SD sim/battle.ts:3064–3078`;
no protocol message carries it). The Showdown node is an information set of the pivoting player spanning the
opponent's possible committed actions, weighted by the opponent's turn-start mixture. (c) At a pivot node where the
pivoting Pokémon moved second, nothing is pending (the saved move is set to none,
`PE src/genx/generate_instructions.rs:3683–3693`) and the node is a one-sided node in both games.

*Proof.* (a) is the definition of a matrix game with one singleton factor. (b) and (c) are read off the cited lines.
$\square$

*Consequence.* "The game is a matrix game at every node" (theory_map A1(i), the per-node form of Theorem A) is true
of $G_{\rm pe}$ as written, with one-sided nodes as degenerate matrices. It is **not** true of $G_{\rm sd}$ at
case-(b) pivot nodes: there the true object is a two-step extensive form inside the turn (both commit; one side
later picks a switch-in without seeing the other's commitment). The engine's version grants the pivoting side
information it never has on Showdown (A11). In the hidden-information model the pending move is resampled from the
determinized set (`EN hidden.py:337–340`), which is the correct M3 treatment for the opponent's pending move.

### 1.5 Transition kernel

**Order within a turn (Showdown).** Actions are sorted by order bracket (switch 103, terastallize 106, moves 200,
residual 300; `SD sim/battle-queue.ts:174–198`), then priority, then speed, then sub-order
(`SD sim/battle.ts:400–416`); exact ties are shuffled by the battle PRNG (`SD sim/battle.ts:434–464`, shuffle at
`:461`). Trick Room replaces speed by $10000-\text{speed}$ (`SD sim/pokemon.ts:662–670`). In Gen ≥ 8 speeds are
recomputed and the queue re-sorted before each move (`SD sim/battle.ts:2978–2985`).

**Order (poke-engine).** Switches run first, then Tera, then moves (`PE src/genx/generate_instructions.rs:4162–4197`;
Pursuit is special-cased, `:4128–4160`). The move order is fixed once per call by `moves_first`: a switch precedes a
move; otherwise higher priority first; Custap Berry at $<\tfrac14$ HP; equal speed gives `SpeedTie`; Trick Room
reverses the comparison (`PE src/genx/generate_instructions.rs:2582–2670`). A speed tie is two branches of
probability $\tfrac12$ (`:4259–4261`).

**Chance events.** poke-engine does not draw random numbers in instruction generation: it enumerates outcomes with
probabilities (`StateInstructions.percentage`), and our loop samples one branch (`EN pkbattle.py:90–93`). Showdown
draws from its PRNG at each event.

| Event | Showdown | poke-engine, Python `generate_instructions` (`branch_on_damage = true`, `PY lib.rs:1067`) | Branches (engine) |
|---|---|---|---|
| speed tie | uniform shuffle (`SD sim/battle.ts:456–461`) | two branches × ½ (`PE …/generate_instructions.rs:4259–4261`) | 2 |
| accuracy | `randomChance(acc, 100)` after accuracy/evasion boosts (`SD sim/battle-actions.ts:778–804`) | hit/miss with $p=\min(1,\text{acc}/100\cdot m(\text{acc boost}))$ (`PE …/generate_instructions.rs:1194–1213`); target evasion handling **[unverified]** | ≤ 2 |
| critical hit | `randomChance(1, critMult[critRatio])`, `critMult = [0,24,8,2,1]` (`SD sim/battle-actions.ts:1681–1700`); ×1.5 (`:1808–1810`) | base $1/24$ (`PE …/generate_instructions.rs:51–57`), $1/8$ for high-crit moves, 1 for guaranteed (`:2391–2401`), ×1.5 (`PE src/genx/damage_calc.rs:63–70`). Stacked crit stages (e.g. Super Luck + high-crit move) **[unverified]** | ≤ 2 |
| damage roll | 16 equiprobable rolls $\lfloor\lfloor B(100-r)\rfloor/100\rfloor$, $r\in\{0,\dots,15\}$, applied **before** STAB and type (`SD sim/battle.ts:2407–2410`; `sim/battle-actions.ts:1812–1813`) | **no roll branching.** With $D_{\max}$ the max roll: if $0.85D_{\max}\ge$ HP, damage $\lfloor0.925D_{\max}\rfloor$ (a KO); if $D_{\max}<$ HP, crit/non-crit branches at $\lfloor0.925\cdot\rfloor$; if the grid $\{(0.85+0.01i)D_{\max}\}_{i<16}$ straddles HP, a KO branch of probability $(1-c)k/16+c$ ($k$ = grid points strictly above HP) and a non-KO branch at the mean of the grid points strictly below HP (`PE …/generate_instructions.rs:2374–2430`, `1016–1036`) | 1–2 per hit |
| secondary effect | `random(100) < chance` per secondary (`SD sim/battle-actions.ts:1400–1409`) | hit/miss branch per secondary (`PE …/generate_instructions.rs:1038–1065`) | ≤ 2 each |
| 2–5-hit moves | 2,3,4,5 hits w.p. .35,.35,.15,.15; Loaded Dice 4–5 (`SD sim/battle-actions.ts:930–936`) | fixed 3 hits (5 Skill Link, 4 Loaded Dice) (`PE …/generate_instructions.rs:2350–2358`) | 1 |
| full paralysis | ¼ (`SD data/conditions.ts:41`) | 0.25 (`PE …/generate_instructions.rs:81`) | 2 |
| sleep length | $\text{startTime}\in\{2,3,4\}$ at onset, i.e. 1–3 skipped turns uniform (`SD data/conditions.ts:58–60`, `66–75`) | wake chance $1/(4-k)$ after $k\ge1$ turns asleep (`PE …/generate_instructions.rs:68,163–169`): the same uniform law on 1–3 (derived: $\tfrac13,\ \tfrac23\cdot\tfrac12,\ \tfrac13\cdot1$) | 2 |
| thaw | 1/5 (`SD data/conditions.ts:99`) | 0.20 (`PE …/generate_instructions.rs:74`) | 2 |
| confusion self-hit | 33/100 (`SD data/conditions.ts:187`) | 1/3 (`PE …/generate_instructions.rs:91`) | 2 |
| Outrage-type lock length | 2 or 3 turns (`SD data/conditions.ts:265`) | always 3 (`PE …/generate_instructions.rs:3249–3256`) | 1 |

Branching per joint action in random-vs-random engine play: mean 2.28, median 2, 90th percentile 4, max 57 (probe,
`scratchpad/game_model/branching.py`, 60 games). Inside poke-engine's own search the damage-branch rule is switched
off: expectiminimax (`pe.id`) always uses `branch_on_damage = false` (`PE src/search.rs:55–56`), and MCTS uses it only
at the root and its children (`PE src/mcts.rs:148`). With it off, a damaging move takes the single value
$\lfloor0.925D_{\max}\rfloor$ and **crits are not branched at all** (the final `else` branch,
`PE …/generate_instructions.rs:2436–2438`).

**PRNG and determinism (Showdown).** The battle owns one PRNG, seeded at construction
(`SD sim/battle.ts:225`). Seeds starting `sodium,` use a ChaCha20 stream (`SD sim/prng.ts:176–232`); numeric seeds
use the Gen 5 LCG $x_{n+1}=(\texttt{0x5D588B656C078965}\,x_n+\texttt{0x269EC3})\bmod2^{64}$, returning the top 32 bits
(`SD sim/prng.ts:235–296`). Absent a seed, a fresh `sodium` seed is drawn from `crypto.getRandomValues`
(`SD sim/prng.ts:160–162`, `210–228`). Teams are generated by a separate generator with per-player seeds
(`SD sim/battle.ts:3229–3236`). Our headless harness fixes the battle seed and both team seeds (Gen 5 form,
`EN showdown/headless.py:38–47`).

**Proposition 3 (determinism, by reading).** Given the battle seed, both team seeds and the sequence of choices, a
Showdown battle is a deterministic function: every chance event above draws from the battle PRNG
(`random`, `randomChance`, `sample`, `shuffle` in `SD sim/prng.ts:91–150`), and no other randomness source appears in
the cited mechanics. **[unverified]** that no mechanic anywhere in `SD data/` uses `Math.random`; I checked only the
mechanics cited here.

*What pairing buys.* Two games with equal seeds and swapped seats start from identical teams and an identical PRNG
state. The PRNG is a single stream, so the draws coincide only while both games consume the same sequence of draws;
once the action sequences differ (a different move, a move that misses instead of a status move), the number of
draws per turn differs and later outcomes are no longer common random numbers. Pairing therefore removes team
variance exactly and chance variance only up to the first divergence. The engine's paired games (`EN
hidden_experiments.py:41–76`) behave the same way: one `random.Random` drives teams, agents and chance draws.

### 1.6 Payoff and end conditions

| Event | Showdown $G_{\rm sd}$ | poke-engine search | Our engine loop $G_{\rm pe}$ |
|---|---|---|---|
| one side has no Pokémon left | that side loses (`SD sim/battle.ts:2660–2670`) | `battle_is_over` = ±1 (`PE src/state.rs:1359–1370`) | win/loss (`EN pkbattle.py:106–108`) |
| both sides out at once | **Gen ≥ 5: the side of the last Pokémon to faint wins** (`this.win(faintData.target.side)`, `SD sim/battle.ts:2661–2663`); a tie only if no faint data | $-1$ (side one checked first, `PE src/state.rs:1363–1365`) | draw, $u=0$ (`EN pkbattle.py:108`, `hidden_experiments.py:74–75`) |
| turn limit | tie when $\kappa>1000$, checked only for $\kappa>100$ (`SD sim/battle.ts:1855–1859`) | none (`State` has no turn field, `PE src/state.rs:1321–1330`) | draw at $t=300$ decision nodes (`EN pkbattle.py:96–109`; `hidden_experiments.py:41`) |
| Endless Battle Clause | **not in this format's ruleset** (`SD config/formats.ts:29–35`); the check returns early without the rule (`SD sim/battle.ts:1871`) | — | — |
| forfeit | the forfeiting side loses (`lose` → `win(foe)`, `SD sim/battle.ts:1560–1566`) | — | — |
| timer tiebreak | most Pokémon left, then HP %, then HP (`SD sim/battle.ts:1480–1521`) | — | — |

So $u\in\{-1,0,1\}$ in all three, but the **draw events differ**: Showdown draws only at the 1000-turn limit (or by a
tiebreak/forced tie); our loop draws on a simultaneous wipe-out and at 300 decision nodes. Our Showdown harness has no
300 cap (no turn limit in `EN showdown/headless.py`; the game ends on `|win|`/`|tie|`, `EN showdown/run_ourbot.py:150`),
so on Showdown $H$ is 1000 **turns**. A6 holds in both games with $u_H\equiv0$; only the value of $H$ and its unit
differ.

### 1.7 Team generation: the initial chance move

The format is `[Gen 9] Random Battle`: `team: 'random'`, ruleset `PotD, Obtainable, Species Clause, HP Percentage
Mod, Cancel Mod, Sleep Clause Mod, Illusion Level Mod` (`SD config/formats.ts:29–35`). There is no Team Preview, so
the generator's slot 0 leads.

**Showdown's generator** (`SD data/random-battles/gen9/teams.ts`), per side:

1. Draw a base species without replacement, uniformly from the pool (`:1790`), then a forme uniformly (`:1797`).
   Reject it if the base species is already on the team (Species Clause enforced by the generator, `:1800–1801`; the
   rule itself is a validator rule, `SD data/rulesets.ts:789–806`, and `Battle.getTeam` returns the generator's team
   without calling a validator, `SD sim/battle.ts:3225–3240`), if it is
   Zoroark in the last slot (`:1806–1807`), or if it breaks a team limit: ≤ 2 per type, ≤ 3 weak to a type, ≤ 1 double
   weakness, ≤ 4 weak to Freeze-Dry, ≤ 1 level-100 Pokémon, and pairwise compatibility (`:1821–1870`). Leads exclude
   `NO_LEAD_POKEMON` (`:1882–1893`). PotD, if configured, replaces the second Pokémon (`:1878`).
2. For the species, draw one **role set** uniformly from its role sets in `sets.json` (`:1504–1524`), and a Tera type
   uniformly from the role's list (`:1531`). A role set is a named role with its allowed abilities, items, Tera types
   and a move pool (1–10 moves, mean 4.9). The 873 role sets over 509 species fall into 10 roles: Bulky Attacker 134,
   Fast Attacker 133, Setup Sweeper 128, Bulky Setup 115, Bulky Support 108, Wallbreaker 83, Fast Support 56, Fast
   Bulky Setup 43, Tera Blast user 38, AV Pivot 35 (probe on `EN data/gen9randombattle.json`, the published copy of
   these sets, `EN data/SOURCES.md`). Then build the **moveset by rules** (`randomMoveset`, `:1543`), and pick
   ability (`:1547`) and item (`:1551–1558`) by rules that depend on moves, team details (weather, hazards) and lead
   status. Level is fixed per species (`:1442–1449`, `:1561`); EVs 85 and IVs 31 everywhere, with HP-EV and
   Attack/Speed adjustments (`:1536–1609`).

**Our engine's generator** (`EN randbats.py`): species keys uniform with base-name dedupe (`:95–104`); role uniform,
4 moves a uniform subset of the role's pool, ability/item/Tera uniform over the listed values (`:58–70`); EVs 84
default. 84 and 85 give the same stats, since stats use $\lfloor\text{EV}/4\rfloor=21$ (`EN randbats.py:73–75`). It
applies none of the team limits or move rules; its docstring says so (`EN randbats.py:7–10`).

**foul-play's prior** (`FP fp/data/sets/randbats.py:24–67`): count-weighted full sets from
`pkmn.github.io/randbats/data/full` — 5,802 distinct full sets with total count 600,000 over 509 species (probe).
Unrevealed species are drawn uniformly over species with soft team-limit rejection for 10 tries
(`FP fp/search/random_battles.py:66–109`).

So there are **three different priors** $b_0$: Showdown's (the truth on M4), `randbats.py`'s (truth and belief in
the engine), foul-play's (belief on M4). This is the content of A4.

---

## 2. Information structure

### 2.1 What each player observes

**Own side.** Exact: each `|request|` carries the full own team, with exact HP, moves, PP, item, ability, Tera type
(`SD sim/SIM-PROTOCOL.md:741–760`; sent per player via `sideupdate`, `SD sim/SIMULATOR.md:147–155`).

**Opponent, public.** Through the public stream:

- **species, level, gender, Tera** on switch-in, in `DETAILS` (`SD sim/SIM-PROTOCOL.md:205–209`; built in
  `SD sim/pokemon.ts:558–573`). Levels are fixed per species in this format (`SD data/random-battles/gen9/teams.ts:1449`),
  so level carries no information beyond species, except under Illusion (§2.5).
- **HP as a percentage.** Under HP Percentage Mod (`SD data/rulesets.ts:1335–1343`), the shared HP is
  $$\mathrm{pct}(h,M)=\begin{cases}0 & h=0\\ 99 & \lceil100h/M\rceil=100,\ h<M\\ \lceil100h/M\rceil & \text{otherwise}\end{cases}$$
  out of 100 (`SD sim/pokemon.ts:2097–2103`). Precision: one percentage point, i.e. a cell of about $M/100$ HP
  (2.7 HP at the median max HP 271, 5.7 at 572); 100 means exactly full. `EN hidden.py:146–150` implements the same
  formula.
- **status**, **boosts**, most volatiles (`-status`, `-boost`, `-start`, `SD sim/SIM-PROTOCOL.md:341–360` and
  following); **moves** when used (`|move|`, `:239–251`); **items and abilities** when they act or are revealed
  (`-item`, `-enditem`, `-ability`, `[from] item:` tags); **Tera** (`-terastallize`, `SD sim/battle-actions.ts:2001`).

**Opponent, private.** Unrevealed team slots; unrevealed moves, item, ability and Tera type; exact HP; PP; the
opponent's chosen action until it executes (including a committed action pending across a pivot, §1.4).

### 2.2 Public stream versus private information

The simulator writes one log. A line containing private detail is written as
`|split|PLAYERID` followed by a SECRET line (for that player) and a PUBLIC line (for everyone else)
(`SD sim/SIMULATOR.md:137–145`; `battle.add` builds both from a `getHealth`-style function, `SD sim/battle.ts:3152–3173`).
HP is the main split field: secret `h/M`, shared `pct/100` (`SD sim/pokemon.ts:2082–2120`). Our harness resolves
splits per player (`EN showdown/headless.py:50–63`). Choice requests are private (`sideupdate`).

Let $o^i_t$ be everything player $i$ has received up to node $t$: its own requests and the public stream with its own
secret lines. The **public history** is the spectator's view (all PUBLIC lines); $o^i_t$ = public history + player
$i$'s private team.

### 2.3 Types, beliefs and the public belief state

Player $i$'s private type is its team $\theta_i$ (six full sets: species, role, moves, ability, item, Tera, level,
stats). Types are drawn independently from $b_0$ (§1.7) and never change, except by revealed in-game events (item
loss, Transform). Dynamic quantities (HP, status, boosts) are public up to the HP rounding of §2.1.

- **Hard-evidence belief** $b^{\rm hard}_h(\theta_2)\propto b_0(\theta_2)\,\mathbf 1[\text{consistent with }h]\cdot
  \prod_t P_{\theta}(s_{t+1}\mid s_t,a_t,b_t)$ (`full_game.md` (6.2)): legality evidence (a revealed move must be in
  the set) and chance likelihoods (e.g. damage dealt depends on the hidden item and ability).
- **True posterior** $p_h$ adds the likelihood of the opponent's choices under its strategy $\sigma_\theta$
  (`full_game.md` (6.1)). $p_h=b^{\rm hard}_h$ under A2.
- **Public belief state** at $h$: the pair (public state, joint belief over both types). Both players' types are
  private, so the PBS carries a belief over $\theta_1$ (what the opponent can infer about us) and over $\theta_2$.

**Our implementation is a coarsening of $b^{\rm hard}$.** `hidden.Belief` conditions on revealed moves, revealed
Tera type and "item gone" only (`EN hidden.py:6–17`, `191–212`, `224–279`); it uses no chance likelihood (damage
observations do not update item or ability), no behavioural evidence, and the species of unrevealed slots is drawn
uniformly over keys with an unused base (`:270–279`). foul-play has a separate inference module
(`FP fp/battle/inference.py`), not characterised here.

### 2.4 Why the game has imperfect information, and how we approximate it

Each player's payoff-relevant private type is drawn by chance at the start and observed only by its owner, and the
public stream reveals it gradually and partially; together with simultaneous moves this makes $G$ an
imperfect-information game with private types (a Bayesian stochastic game) rather than a stochastic game. Its value
need not be computable node by node: the value at a public node depends on the belief, and beliefs depend on the
strategies at other nodes (non-locality, `full_game.md` §6.2).

Our M3 agents replace it with $K$ **sampled worlds**: `determinize` draws a full opponent team from `Belief`
(`EN hidden.py:322–342`), and each world is solved as a full-information stage game. PIMC solves each world and
averages the strategies; **agent form** solves one LP that maximises the belief-weighted worst case over worlds with
a single strategy (`EN hidden_agents.py:36–49`). Theorem C (`theory_map.md` §4.3) bounds the loss of agent form by
five per-stage terms; three of them need A2 (non-revealing opponent: TV term 0), A3 (world-only leaves:
$\eta_{\rm NL}=0$) and A4 (exact prior). The concealment of our own team is added once (two-sided remark,
`full_game.md:188`).

### 2.5 Exceptions to "revealed is true"

- **Illusion (Zoroark, Zoroark-Hisui).** The Pokémon enters disguised as the last unfainted party member
  (`SD data/abilities.ts:2057–2066`); under Illusion Level Mod it also shows that member's level
  (`SD data/rulesets.ts:2904–2912`, `SD sim/pokemon.ts:567–571`). A damaging hit ends it with `|replace|`
  (`SD data/abilities.ts:2072–2082`; "everything you thought you knew about the previous Pokémon is now wrong",
  `SD sim/SIM-PROTOCOL.md:284–291`). The generator never puts Zoroark last (`SD …/gen9/teams.ts:1806–1807`).
  foul-play detects it from moves outside the shown species' pool or impossible immunities
  (`FP fp/modes/random_battle.py:98–206`). poke-engine has the enum value `ILLUSION` but no handler
  (`PE src/genx/abilities.rs:99`; no `Abilities::ILLUSION` match in `PE src/genx/`).
- **Imposter/Transform (Ditto).** The species shown is Ditto, but it copies the opponent. poke-engine defines
  `TRANSFORM` with no effect (`PE src/choices.rs:17990–17999`) and has no `Abilities::IMPOSTER` handler
  (`PE src/genx/abilities.rs:275`): engine Ditto (only move: Transform, `EN data/gen9randombattle.json`) acts as a
  Pokémon whose move does nothing.
- **Forme changes** (`|detailschange|`, `-formechange`, `SD sim/SIM-PROTOCOL.md:274–282`): Palafin, Mimikyu
  (Disguise), Ogerpon/Terapagos on Tera. `EN hidden.py:315–318` re-derives stats for a public forme change.
- **Called moves.** Moves called by Sleep Talk, Metronome and similar appear in `|move|` with a `[from]` tag and are
  not in the user's set; `B34` skipped 10 such events. foul-play's Zoroark check excludes `from` lines
  (`FP fp/modes/random_battle.py:108`). Our `Tracker` reveals the move *chosen*, not the move executed, so a move
  chosen but not executed (full paralysis, flinch) still counts as revealed (`EN hidden.py:10–13`): a deviation of our
  model from the true observation, in the direction of revealing more.

---

## 3. Our models of the game

### 3.1 The hierarchy (aligned with `theory_map.md` §1.1)

- **M2: $G_{\rm pe}$, poke-engine with full information.** States are `pe.State`; menus from `pe.id`'s root options
  (`EN pkbattle.py:72–83`); kernel $P_{\rm pe}$ = the branch list of `pe.generate_instructions` with probabilities
  (`EN pkbattle.py:90–93`); teams from `randbats.py`; draw at a simultaneous wipe-out and at $H=300$ decision nodes.
  Both teams are known to both players.
- **M3: sampled worlds in $G_{\rm pe}$.** As M2, but each agent sees only its `View` (`EN hidden.py:153–179`) and
  solves $K$ determinized worlds (§2.4). The true team is drawn from `randbats.py`, the belief from `hidden.Belief`.
- **M4: $G_{\rm sd}$, Showdown.** Real kernel, real teams, real information; our agent sees foul-play's battle
  model, converts each of foul-play's $K$ sampled worlds into a `pe.State` (`FP fp/search/poke_engine_helpers.py:299`)
  and builds `nash.full_matrix` per world (`EN showdown/run_ourbot.py:50–70`). So the agent at M4 still plans in
  $G_{\rm pe}$.

### 3.2 poke-engine versus Showdown: differences

Classes. **(K)** kernel difference: same state and action, different distribution over next states. It enters
$\varepsilon_{\rm sim}$ (A5). **(S)** structural difference: menus, information or terminal payoffs differ. It is not a
kernel mismatch, so it is **not covered by A5 or Proposition SIM**, which assume the same action sets and terminal
payoffs (`theory_map.md` §7); I state it as A10–A12 (§5). **(V)** a difference between the engine's lookahead model
and $G_{\rm pe}$ itself; it enters $\varepsilon_v$ (A7) even at M2.

| # | Difference | Showdown | poke-engine | Class |
|---|---|---|---|---|
| 1 | damage rolls | 16 uniform rolls, mid-chain (`SD sim/battle.ts:2407–2410`) | one value $\lfloor0.925D_{\max}\rfloor$ per branch, KO/non-KO split only when the roll grid straddles HP (`PE …/generate_instructions.rs:2374–2430`). Exact-state TV ≥ 0.89 per damaging move; KO-indicator TV 0.012 mean over 516 events (`B34` lines 10–11) | K |
| 2 | roll grid and rounding | $\lfloor B(100-r)/100\rfloor$ before STAB/type | grid $(0.85+0.01i)D_{\max}$ after all modifiers; a grid point equal to HP counts as neither KO nor non-KO (`PE …/generate_instructions.rs:1016–1036`). 98.8% of observed damage lies within ±2 HP of a 16-roll grid that the `B34` scorer rebuilds from poke-engine's max roll (`B34` line 3; `score_b34.py:8–9`), not of the engine's own kernel | K |
| 3 | drain rounding | `Math.round(dmg·num/den)` in Gen ≥ 5 (`SD sim/battle.ts:2187–2190`) | truncation (`PE …/generate_instructions.rs:1527–1530`) | K (±1 HP) |
| 4 | multi-hit 2–5 | 35/35/15/15 (`SD sim/battle-actions.ts:933`) | fixed 3 (`PE …/generate_instructions.rs:2357`) | K |
| 5 | Outrage-type lock | 2–3 turns (`SD data/conditions.ts:265`) | always 3 (`PE …/generate_instructions.rs:3249–3256`) | K |
| 6 | crit stacking | stages 0–4 → 1/24, 1/8, 1/2, 1 (`SD sim/battle-actions.ts:1691`) | 1/24, 1/8 or 1 only (`PE …/generate_instructions.rs:2391–2401`); stacking **[unverified]**; `B34` found one crit/Unaware mismatch ($n=1$, Quagsire event in `debug_out.txt`) | K |
| 7 | mid-turn speed re-sort | Gen ≥ 8 re-sorts before each move (`SD sim/battle.ts:2978–2985`) | order fixed once per call (`PE …/generate_instructions.rs:4207–4212`) | K; with one move per side I found no singles case where it changes the order **[unverified]** |
| 8 | Illusion | disguise, level mod, `|replace|` (§2.5) | no handler (`PE src/genx/abilities.rs:99`) | K (and information) |
| 9 | Imposter / Transform | copies the target | no effect (`PE src/choices.rs:17990–17999`) | K |
| 10 | unknown items | — | Light Clay, Scope Lens, Leppa Berry become `UNKNOWNITEM` with no effect (`EN randbats.py:12–14`, project note, not re-checked) | K |
| 11 | Struggle | forced when no move is usable (`SD sim/pokemon.ts:1124–1126`) | absent; menu `[None]` or switches only (`PE src/genx/state.rs:1210–1215`) | S (menu) |
| 12 | PP | real PP with 3 PP Ups, e.g. 48 for a 30-PP move (`SD sim/pokemon.ts:382–388`) | every move 16 PP in our constructors (`PY lib.rs:846`; `EN randbats.py:91`, `pkbattle.py:56` pass no `pp`); foul-play's converter sets PP from its battle model **[unverified]** | S (menus differ once PP runs out) |
| 13 | pending move at a pivot | hidden until executed (§1.4) | a public state field, the single column (`PE src/genx/state.rs:1227–1238`) | S (information) |
| 14 | "none" / wait | no action (`{wait: true}`) | explicit `None` action | none: a singleton menu in both |
| 15 | simultaneous wipe-out | last-fainted side wins (`SD sim/battle.ts:2661–2663`) | search: side one loses (`PE src/state.rs:1363`); our loop: draw (`EN pkbattle.py:108`) | S (terminal payoff) |
| 16 | horizon | tie at $\kappa>1000$ turns (`SD sim/battle.ts:1856–1858`) | none in the engine; our loop draws at 300 decision nodes | S (horizon) |
| 17 | forced-switch timing | replacement at end of turn, same turn number (`SD sim/battle.ts:2902–2911`, `1640–1641`) | a separate call with end-of-turn skipped (`PE …/generate_instructions.rs:4224–4233`) | none for the kernel (same order of events, by reading); changes only the counting of $t$ |
| 18 | crit and roll branching inside the lookahead | — | `pe.id` (used for our leaf) never branches on crits or KO rolls (`PE src/search.rs:56`) | V |
| 19 | leaf | — | heuristic $\mathrm{ev}$, terminal children scored by $\mathrm{ev}$ too (§4) | V |
| 20 | team prior | Showdown generator (§1.7) | `randbats.py` uniform role/move/item model; foul-play's empirical prior at M4 | A4, not a kernel term |

### 3.3 Which error term

- Rows 1–10 are kernel mismatches and belong to $\varepsilon_{{\rm sim},t}$ of A5. Rows 1–2 dominate at exact-state
  resolution; at KO resolution the measured mismatch is about 0.01 per move (`B34`; `theory_map.md` §3.1).
- Rows 11–16 make $G_{\rm pe}$ and $G_{\rm sd}$ different games: different menus (11, 12), a different information
  partition (13), different terminal payoffs (15) and a different horizon (16). Proposition SIM does not cover them;
  A10–A12 name them. Rows 15–16 can be absorbed into a kernel term by redefining terminal states as absorbing
  transitions, but then the TV at those transitions is 1 on the affected events, not small.
- Rows 18–19 are in $\varepsilon_v$: they are errors of our estimate of $Q^*_{\rm pe}$, present even at M2.
- Row 20 is A4.

---

## 4. The stage matrix, precisely

### 4.1 Definition

For a live state $s$ with menus $A(s)=\{a_1,\dots,a_n\}$, $B(s)=\{b_1,\dots,b_m\}$ (from `pe.id(s,0)`,
`EN pkbattle.py:72–76`), `nash.full_matrix` computes (`EN nash.py:122–133`)

$$\hat Q_t(s)[a,b]=\sum_{\omega\in\Omega(s,a,b)}p_\omega\,\ell(s\oplus\omega),$$

where $\Omega(s,a,b)$ is the branch list of `pe.generate_instructions(s,a,b)` (with `branch_on_damage = true`,
`PY lib.rs:1044–1073`), $p_\omega$ its probability, and $s\oplus\omega$ the child. The leaf is

$$\ell(c)=\tfrac12\big(\underline v(c)-\underline v(\mathrm{mirror}(c))\big),\qquad
\underline v(c)=\max_{a'}\min_{b'}M_c[a',b'],$$

with $M_c[a',b']=\sum_{\gamma}p_\gamma\,\mathrm{ev}(c\oplus\gamma)$ over the branches of
`generate_instructions_from_move_pair(c,a',b',\text{false})` (`PE src/search.rs:55–62`). `nash._leaf` computes
$\underline v$ as `pe.id(c, 0)`'s depth-1 maximin (`EN nash.py:96–103`) and $\mathrm{mirror}(c)$ by swapping the two
sides and copying the field (`EN nash.py:106–119`). If poke-engine's kernel and $\mathrm{ev}$ were exactly antisymmetric
under the side swap, $-\underline v(\mathrm{mirror}(c))$ would equal the pure minimax
$\overline v(c)=\min_{b'}\max_{a'}M_c[a',b']$ and $\ell$ would be the midpoint of the pure maximin and minimax, which
bracket the mixed value of $M_c$. They are only nearly antisymmetric (`EN nash.py:115–116`): `moves_first` checks side
one's Custap Berry first (`PE …/generate_instructions.rs:2618–2640`) and `battle_is_over` checks side one's wipe-out
first (`PE src/state.rs:1363–1368`). So the midpoint reading is an approximation, and the definition above is the one
the code computes. (`pe.id` prunes cells with alpha-beta; the maximin is unaffected,
`EN nash.py:97–99`.) "Depth 2" means: ply 1 with crit/KO branching, ply 2 without, then $\mathrm{ev}$.

*Remark (the returned depth).* `pe.id(c, 0)` runs depth 1 synchronously, then starts depth 2 on a thread and stops
after a 0 ms sleep, returning the last completed depth (`PE src/search.rs:161–234`). Depth 1 is returned unless the
thread finishes depth 2 before the main thread sets the stop flag; `_child_value` does not check the depth
**[unverified: race never observed or measured]**.

### 4.2 Range

$\mathrm{ev}$ is a weighted sum of material: per living Pokémon $30+\max(0,100h/M+\text{status}+10\cdot[\text{item}])$,
plus boosts of the active (weights 15–30 per stage, multipliers up to 3.3), hazards, screens, Tera use ($-75$),
substitute, confusion, leech seed; side two's terms enter with a minus sign (`PE src/genx/evaluate.rs:7–52`,
`127–155`, `157–269`). It is bounded; a crude bound from these constants is $|\mathrm{ev}|\lesssim3\times10^3$
engine units (derived; a full-HP Pokémon with an item is worth 140). Hence $\hat Q_t(s)\in[-L,L]$ with
$L\approx3\times10^3$, not $[-1,1]$. The map from engine units to the payoff scale is unknown and is part of
$\varepsilon_v$ (A7, `theory_map.md` §0.2).

**Terminal children are not exact.** If $c$ is terminal, `expectiminimax_search` returns
$100\cdot\text{depth}\cdot\texttt{battle\_is\_over}+\mathrm{ev}(c)$ with depth already decremented to 0
(`PE src/search.rs:22,34–40`), so $\ell(c)=\mathrm{ev}(c)$: a won position is worth the winner's remaining material,
not a fixed win value. `squash_matrix` (`EN nash.py:136–151`) and `vnet_matrix` (`:196–216`) score terminal children
exactly as $+1/-1/0$ and squash the rest into $(-1,1)$; their entries lie in $[-1,1]$.

### 4.3 When the matrix is exact

Relative to $G_{\rm pe}$ (M2), $\hat Q_t(s)=Q^*_{{\rm pe},t}(s)$ requires $\ell(c)=V^*_{{\rm pe},t+1}(c)$ for every child.
With the default leaf this never holds by construction (heuristic units, no exact terminals). With `squash_matrix`
it holds at states all of whose children are terminal, **with the engine loop's payoff convention** (a simultaneous
wipe-out scored 0, row 15) and the engine kernel $P_{\rm pe}$. Relative to $G_{\rm sd}$ it is never exact: rows 1–10 of
§3.2 change the ply-1 kernel itself. Everywhere else $\hat Q$ is an estimate and its error is $\varepsilon_v(t,s)$ of
A7 (measured only on exact endgames, `theory_map.md` §3, B01).

### 4.4 Connection to Lecture 7

Lecture 7's setting is one fixed matrix game with payoffs in $[0,1]$. Our stage game is the matrix
$\hat Q_t(s)\in\mathbb R^{n\times m}$, $n,m\le13$, row player maximising. The affine map
$A_t=(\hat Q_t(s)-c\mathbf 1\mathbf 1^\top)/\bar\Delta$ with a fixed $c$ and normaliser $\bar\Delta$ (A9; engine:
$\bar\Delta=400$) and clipping places it in $[0,1]$. For computing one stage equilibrium any positive affine map is
harmless: RM+ and the LP return the same strategies (Lemma R6a, `theory_map.md` §4.1). For a learner that compares
rewards across turns the map must be fixed (A9). Lecture 7's guarantees are then statements about $A_t$; the passage
from $A_t$ to the game uses Lemma 1 / Theorem A with $\varepsilon_v$ (A7) and, at M4, A5 and A10–A12.

---

## 5. Assumptions

A1–A9 are as in `theory_map.md` §2 and are not restated. Comments on how each meets the true game:

- **A1(i)** (simultaneous moves at each live node) holds at simultaneous and one-sided nodes of $G_{\rm pe}$ and
  $G_{\rm sd}$ (Proposition 2(a)). At case-(b) pivot nodes of $G_{\rm sd}$ the opponent's relevant action was drawn
  earlier in the turn; A1(i) still holds for the turn-start draw, but the stage at that node is not the engine's
  $n\times1$ matrix (A11).
- **A4** (exact prior) is *approximate* even in the engine (§6, item 2).
- **A5** must be read at a coarse resolution (`theory_map.md` §3.1); rows 1–2 of §3.2 are why.
- **A6** holds with $H=300$ decision nodes in $G_{\rm pe}$ and $H=1000$ turns in $G_{\rm sd}$ (§1.6).

New assumptions.

**A10 (menu and payoff alignment).** For every state $s$ reachable in both games,
$A_{\rm pe}(s)=A_{\rm sd}(s)$, $B_{\rm pe}(s)=B_{\rm sd}(s)$, and the terminal payoffs agree. *Status:* violated on
measure-small sets: Struggle states (row 11), states where real PP and the engine's 16 PP differ in which moves are
usable (row 12), simultaneous wipe-outs (row 15). Proposition SIM is stated under A10.

**A11 (information at pivot nodes).** At a pivot node where the pivoting Pokémon moved first, the switching player
does not observe the opponent's committed action. *Status:* true in $G_{\rm sd}$ (§1.4); false in $G_{\rm pe}$, where the
action is a state field. Our M3 code samples it from the determinized set for the opponent (`EN hidden.py:337–340`);
the oracle and every M2 result give the pivoting side this information.

**A12 (horizon unit).** $H$ counts decision nodes (engine `step` calls). Showdown's turn counter $\kappa$ counts full
turns, and replacement and pivot nodes do not advance it. Results stated with $\bar H$ in decision nodes transfer to
Showdown with $\bar H$ measured in decision nodes, not turns.

---

## 6. Conflicts with `theory_map.md`

1. **"$H$ = 300 in the engine and on Showdown"** (`theory_map.md` §0.1; A6; `theory_section.md` "300-turn cap").
   Showdown's limit is a tie at turn 1000 (`SD sim/battle.ts:1856–1858`), Endless Battle Clause is not in this
   format, and our Showdown harness has no cap. In the engine, 300 counts decision nodes including replacement and
   pivot nodes, not turns. A6's form (cap payoff 0) survives; the number and the unit do not.
2. **A4 "exact by construction" in the engine.** The set posterior within a species is exact
   (`EN hidden.py:191–212`), but the species of unrevealed slots is not the exact conditional of
   `randbats.sample_team`: sequential sampling with base dedupe is order-dependent, and `Belief.sample_unrevealed`
   draws uniformly over unused keys. Toy check: keys $\{A_1,A_2,B,C\}$, team of 2, slot 1 $=B$: true conditional for
   slot 0 is $(3/8,3/8,1/4)$, the sampler gives $(1/3,1/3,1/3)$ (probe `scratchpad/game_model/a4_toy.py`). The effect
   is confined to multi-forme bases and is small at 509 keys **[size unmeasured]**. Also, `Belief` is not
   $b^{\rm hard}$: it uses no chance likelihoods (§2.3), so Theorem C's TV term is not zero even under A2.
3. **Pivot nodes versus "a matrix game at every node"** (A1(i), Theorem A's node-by-node use). True in $G_{\rm pe}$
   with degenerate matrices; false in $G_{\rm sd}$ at case-(b) pivot nodes (Proposition 2(b), A11).
4. **Proposition SIM's preconditions** ("the same action sets and terminal payoffs in both kernels") fail on rows
   11–16 of §3.2; SIM is valid only together with A10–A12, or with those events moved into $\varepsilon_{\rm sim}$ at TV
   1.
5. **Draw convention.** `theory_section.md` and A6 treat a draw as the 300-turn cap; our loop also draws on a
   simultaneous wipe-out, which Showdown awards to the last side to faint, and poke-engine's own search scores as a
   loss for side one.
6. **"Exact terminals" in the default matrix.** `full_matrix` scores terminal children with $\mathrm{ev}$
   (§4.2); results that assume exact terminals (Theorem L5, Lemma R6b(3), Theorem L6's terminal term) apply to
   `squash_matrix`/`vnet_matrix`, not to `full_matrix`.
7. **"poke-engine 16-roll damage model"** (`theory_map.md` §3.1, B34 bullet). poke-engine has no 16-roll model in its
   kernel; the 16-roll grid in `B34` is the scorer's reconstruction (`scratchpad/bridges/b34/score_b34.py:8–9`) from
   poke-engine's max roll. The measured numbers stand; the wording should say "a 16-roll grid built from poke-engine's
   max roll".

---

## 7. Summary for the paper

**The game.** A Gen 9 Random Battle is a two-player zero-sum game with chance, simultaneous moves and private
types. Each player's type is its team of six Pokémon, drawn independently by Showdown's random-team generator: a
base species uniformly without replacement subject to Species Clause and team limits (≤ 2 per type, ≤ 3 weak to a
type, ≤ 1 level-100 Pokémon), a role set uniformly per species, and a moveset, ability and item chosen by rules
(`teams.ts`). A state holds, per side, six Pokémon (species, level, stats, HP, status, item, ability, four moves with
PP, Tera type and flag), the active slot's boosts and volatile conditions, side conditions (hazards, screens), and
the field (weather, terrain, Trick Room). For fixed teams there are of order $10^{150}$ representable states; the
type space has of order $10^{18}$ teams per side.

**Moves and nodes.** At a normal decision node each side chooses one of up to four moves, each also with
Terastallization while the side's single Tera is unused, or one of up to five switches: at most $13$ actions.
Choices are simultaneous. Replacements after a faint and switches after U-turn-type moves are nodes where only one
side chooses (both, if both actives fainted); we treat them as degenerate $n\times1$ stage games. The one place where
the matrix view is not faithful is a pivot after which the opponent's move is still pending: on Showdown it is
hidden, in the engine it is part of the state.

**Chance.** Turn order is by priority bracket, then speed (Trick Room reverses it), with exact ties broken uniformly.
Accuracy, critical hits (base $1/24$), one of 16 equiprobable damage rolls, secondary effects, multi-hit counts and
status durations are chance events. Showdown draws them from a seeded PRNG, so a battle is a deterministic function
of the seeds and the choices; we use this to play paired games, which share teams exactly and share chance draws
until the action sequences diverge.

**Information.** Each player sees its own team exactly. Of the opponent it sees species, level and Tera on reveal,
HP as $\lceil100h/M\rceil$ percent (1-point precision, 99 when not full), status, boosts, and moves, items and
abilities as they are used or revealed. Showdown's protocol carries private details on `|split|` lines. Illusion
(Zoroark) breaks "revealed species is true species" until it is hit. Beliefs over the opponent's team start from
the generator prior and are updated on hard evidence; our agents sample $K$ worlds from the belief and solve either
each world (PIMC) or one agent-form LP across worlds.

**Payoff.** $+1$ win, $-1$ loss, $0$ draw. Showdown ends a battle when one side has no Pokémon left (in Gen 9, if both
run out at once, the side whose Pokémon fainted last wins), on forfeit, or with a tie at turn 1000; Endless Battle
Clause is not in this format. Our engine loop scores a simultaneous wipe-out as a draw and caps the game at 300
decision nodes with payoff 0.

**Our models.** M2 is the full-information game under poke-engine 0.0.48; M3 adds hidden teams and solves $K$
sampled worlds; M4 is Showdown. poke-engine enumerates chance outcomes exactly except damage: it uses one damage
value per branch (the mean roll), splitting off a KO branch only when the roll range straddles the target's HP, so
its kernel matches Showdown's KO events closely (mean KO-probability gap 0.012 per move over 516 events) but matches
the exact HP outcome on only 10.6% of single-value damage events ($n=264$) (bridge probe, 21 games). Smaller kernel differences (drain rounding, fixed 3-hit multi-hit moves,
fixed 3-turn Outrage, no Illusion or Imposter) enter the simulator term $\varepsilon_{\rm sim}$. Differences in menus
(no Struggle, 16 PP per move), information (the pending move at pivots) and terminal payoffs are not kernel errors;
they are stated as separate assumptions.

**The stage matrix.** At each live node we solve
$\hat Q_t(s)[a,b]=\sum_\omega p_\omega\,\ell(s\oplus\omega)$: one simulated turn with poke-engine's chance branches,
then a leaf $\ell$ equal to the average of the two sides' pure maximin values of the child's one-turn matrix scored
by poke-engine's heuristic evaluation (approximately the midpoint of its pure maximin and minimax). Entries are in heuristic engine units (bounded, about $\pm3\times10^3$), and
terminal children are scored by the heuristic, not by $\pm1$. The matrix is therefore an estimate of $Q^*$ with error
$\varepsilon_v$ (A7); after a fixed affine rescaling to $[0,1]$ it is the payoff matrix of Lecture 7's setting.

---

## 8. Unverified claims (the list behind every [unverified] tag)

1. Showdown's per-Pokémon volatiles and poke-engine's per-side volatiles carry the same information (§1.2).
2. poke-engine's accuracy check ignores or handles target evasion (§1.5, accuracy row).
3. poke-engine's handling of stacked crit stages (Super Luck, Scope Lens + high-crit move) (§1.5, §3.2 row 6).
4. No mechanic in `SD data/` uses a randomness source other than the battle PRNG (Proposition 3).
5. Gen 8+ mid-turn speed re-sorting never changes the order in singles (§3.2 row 7).
6. foul-play's state converter sets real PP (§3.2 row 12).
7. The `pe.id(c, 0)` depth race (§4.1 remark).
8. The size of the A4 species-conditional error at 509 keys (§6 item 2).
9. Light Clay, Scope Lens and Leppa Berry are unknown to poke-engine (§3.2 row 10; a project note, not re-checked).

Probe scripts (scratch, not project files):
`/private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/game_model/branching.py`,
`.../game_model/sizes.py`, `.../game_model/a4_toy.py`.
