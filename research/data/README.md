# Frozen final-bot catalogue

`gen9randombattle_20261117.json` is an unmodified copy of the public Random Battle set-frequency catalogue cached by the managed Foul Play installation for the November 17, 2026 ladder runs.

- Original data endpoint: `https://pkmn.github.io/randbats/data/full/gen9randombattle.json`.
- Runtime location: `.local-play/foul-play/fp/data/pkmn_sets_cache/gen9randombattle.json`.
- Size: 461321 bytes.
- SHA-256: `4264b4d70fd86c7ea8ae362bc6aa2c027d2524a73b6df3f286dc0dffbabe5d3f`.

The endpoint changes over time. This snapshot allows another checkout to restore the same catalogue without substituting later set frequencies. It contains public set data, not battle logs, account credentials, opponent private teams, or learned checkpoints.

After managed environment setup, copy this file to the runtime location above. See [the final bot handoff](../../docs/final-ladder-bot-handoff.md) for the full configuration and run commands.
