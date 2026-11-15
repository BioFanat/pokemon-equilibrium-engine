# Data sources (fetched 2026-11-13)
- gen9randombattle.json: https://pkmn.github.io/randbats/data/gen9randombattle.json (Showdown Random Battle sets, per species: level, roles -> abilities/items/teraTypes/moves/evs/ivs)
- dex_gen9randbats.json: types, baseStats, weightkg for those species, sliced from https://play.pokemonshowdown.com/data/pokedex.json
- moves_gen9.json: type, basePower, category per move id, sliced from https://play.pokemonshowdown.com/data/moves.json (fetched 2026-11-14; @pkmn/dex@0.11.0 on jsdelivr ships only bundled JS, no plain JSON). Covers all 348 moves in gen9randombattle.json.
