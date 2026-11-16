# Offline bot decision review — November 16, 2026

Resumed the unfinished review from **Assess proposal progress** using the local
checkout, installed Foul Play/poke-engine, saved spectator replays, decision
mixtures, and sampler logs. No login or public games were started.

## Confirmed fixes

1. **Completed battles used material scores.** `pe.id` on a state with no
   surviving opponent returns a material heuristic. In a constructed 1v1,
   immediately winning could score below healing while the opponent survived.
   `nash._leaf` now returns +10000 for a win, -10000 for a loss, and 0 for a
   simultaneous defeat before searching. This applies to `mid` and `maximin`
   leaves. It does not prove that every individual KO in a 6v6 is optimal.
2. **Accuracy and evasion stages were discarded.** The pinned Foul Play
   converter hardcodes both stages to zero. Our adapter retains them for both
   sides while preserving boosts, hazards, abilities, revealed moves, and field
   state. Installed upstream sources were left unchanged.
3. **Our root actions trusted engine legality alone.** The root now intersects
   engine options with the current server request before matrix construction.
   This enforces explicit disabled/PP flags and tera availability, including
   Foul Play's special re-enabling of Gigaton Hammer/Blood Moon. Uncertain
   `maybeDisabled` flags do not prohibit moves. Choice/Encore/pivot constraints
   remain enforced by the engine; forced replacements and Struggle remain
   available. Matrix rows map to action IDs, not search-reordered indices.
4. **Historical decision logs were insufficient for exact replay.** Optional
   `OURBOT_LOG_STATES=1` snapshots store the actual sampled states, weights,
   rows, columns, legal rows, and full-precision payoff matrices. Every new
   decision also has a battle tag and decision index, distinguishing replacement
   decisions sharing a turn number. `replay_decision.py` re-solves root methods
   offline using the exact logged payoffs, independently of timed native search.

## Review of the reported decisions

Sources below are `.log` replays and matching `decisions/*.jsonl` under
`.local-play/work/ladder/`. The long private replay suffixes are omitted here;
search by the numeric battle ID. These are observed outcomes, not reconstructed
original payoff matrices: the historical logs did not save those matrices.

| Reported behavior | Saved evidence | Finding |
| --- | --- | --- |
| Stealth Rock into Magic Bounce | **2715005923**, turns 4–5, Hippowdon vs Hatterene | Both attempts were reflected. Sampler logs already assign Hatterene `magicbounce`; the recorded policy gave rocks 16.3% and 28.3%. A mechanics regression confirms reflection, so this is not evidence of missing ability tracking. Choosing a move that could work on a predicted switch remains a search/model issue. |
| Rocks already present | **2715025576**, turn 34, Hippowdon vs Umbreon | Rocks were present on the opponent's side and the attempt failed. The converter preserves that hazard, and the native simulator does not stack rocks. Repeated-hazard policy weakness remains; removal followed by replacement within a turn can sometimes make hazards valid. |
| Toxic into Steel | **2715021909**, turns 15–16, Dragalge vs Probopass | Both attempts were immune against an already active Steel opponent. Native simulation correctly rejects Toxic in a matching mechanics regression. Historical logs lack the original matrices needed to identify the exact payoff defect or switch prediction. |
| Ground into Flying | **2715027616**, turn 16, Toedscruel Earth Power into Pelipper | Pelipper switched into the move that turn; Mienshao had been active when Toedscruel chose. This particular replay is a failed prediction, not proof that the bot ignored an already active Flying type. The native Ground/Flying immunity regression passes. |
| Setup before being KO'd | **2715019705**, turn 20, Espathra Calm Mind before Noctowl Hyper Voice KO | Confirmed outcome. Calm Mind does raise SpD; this alone does not establish that another move survives or wins. The completed-battle scoring defect above is independently reproduced and fixed, but cannot be assigned as this turn's cause from mixture-only logs. |
| Setup at maximum stages | **2715023327**, Latias turns 39–43; **2715035259**, Gurdurr turns 55/57 | Latias was Tricked a Choice Scarf on turn 33 and trapped; turns 34–43 have one legal root action. Gurdurr was Encored on turn 55. Turn 57 is a genuine freely selected capped Bulk Up after Encore ended, and remains a policy weakness. Never globally ban setup at the cap without accounting for forced moves or an earlier boost reset. |
| Switching into revealed Hurricane | **2715027616**, turn 11, Munkidori; **2715037008**, turn 5, Urshifu-Rapid-Strike | Pelipper's Hurricane had already been revealed. Urshifu was immediately KO'd in rain. The converter keeps revealed moves, and sampling is conditioned on known moves; no conversion omission was established. Future snapshots are needed to inspect the switch's exact modeled payoffs. |
| Mach/Drain Punch instead of Knock Off | **2714982743**, turn 9, Gurdurr into Sableye; **2714980346**, turn 7, Scrafty into Brambleghast | Drain Punch was immune in both cases; Knock Off had already been used in each game. Mechanics tests confirm Fighting immunity and Knock Off damage/item removal. No action-ID permutation was found. Short search and sampled switch responses can still produce these poor choices. |

## Limits and next experiment

The baseline remains a depth-2 heuristic search over four sampled worlds and
an equilibrium against adversarial responses in those worlds. Correct mechanics
do not make its payoff estimates accurate or its switch predictions calibrated.
Avoid blanket bans on immune attacks: opponent switches, terastallization,
Roost, and ability bypasses can change their effect.

No ladder improvement or win-rate gain is claimed. Stored research outputs and
paper figures were not regenerated; the terminal reward correction changes the
current evaluator, so future comparisons should identify the revised code.

For a targeted follow-up experiment, capture local/headless snapshots for the
unresolved hazard, Toxic, freely capped setup, and Hurricane positions. Compare
deeper or terminal-aware continuation search on those same sampled states and
then run paired offline games. The historical mixtures cannot establish those
counterfactual improvements on their own.

## Validation

```bash
.local-play/venv/bin/python -m unittest discover -s engine/tests -v
.local-play/venv/bin/python engine/test_matrix.py
```

The regression suite covers conversion, request legality, action mapping,
terminal win/loss/draw values, Magic Bounce, existing rocks, type immunities,
capped boosts, Knock Off, and offline snapshot replay. The matrix suite checks
finite payoffs and legal actions on random positions, solver gaps, mirrored
positions, and whole games. All checks ran without a server connection.
