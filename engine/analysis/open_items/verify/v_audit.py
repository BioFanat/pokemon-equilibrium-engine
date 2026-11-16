"""Independent audit: file counts, error games, summary pair means, hedge_state game counts vs files."""
import glob, json, os, re, collections
RES = "/Users/biofanat/Documents/classes/f26/cs535/final/engine/showdown/results"
TAGS = {"LOG": 300, "E3a": 160, "E3f": 160, "XUmv": 300, "XU": 300, "XR": 300, "XEmv": 300, "E5t": 300, "E5s": 300,
        "SX1": 200, "SX3": 200, "SX10": 200, "SX30": 200, "RQeq": 100, "RQsafe": 100, "RQunif": 100}
FN = re.compile(r"^(.+)-s(\d+)-g([12])\.json$")
tot_err = 0; tags_err = 0
for tag, ns in TAGS.items():
    d = os.path.join(RES, tag, "replays_headless")
    games = {}
    for p in glob.glob(os.path.join(d, f"{tag}-s*-g*.json")):
        m = FN.match(os.path.basename(p))
        if not m: continue
        g = json.load(open(p))
        games[(int(m[2]), int(m[3]))] = (g.get("winner_role"), g.get("error"), g.get("stream"), g.get("k"),
                                         json.dumps(g.get("methods")), json.dumps(g.get("team_seeds")))
    exp = {(s, k) for s in range(1, ns + 1) for k in (1, 2)}
    missing = exp - set(games)
    errs = sorted(k for k, v in games.items() if v[1])
    tot_err += len(errs); tags_err += bool(errs)
    # stream rule
    bad_stream = sum(1 for (s, k), v in games.items() if v[2] != (s - 1) % 10)
    ks = collections.Counter(v[3] for v in games.values()); meths = collections.Counter(v[4] for v in games.values())
    # pair means: seeds with both games non-error
    per_seed = {}
    for s in range(1, ns + 1):
        a, b = games.get((s, 1)), games.get((s, 2))
        if a and b and not a[1] and not b[1]:
            per_seed[s] = ((a[0] == "A") + (b[0] == "A")) / 2
    draws = sum(1 for v in games.values() if not v[1] and v[0] not in ("A", "B"))
    pm = sum(per_seed.values()) / len(per_seed)
    S = json.load(open(os.path.join(RES, tag, f"{tag}_summary.json")))
    sps = {int(k): v for k, v in S["pair_score_A"]["per_seed"].items()}
    match_seed = sps == per_seed
    # hedge state
    st = []
    for p in sorted(glob.glob(os.path.join(RES, tag, "hedge_state", "*.json"))):
        j = json.load(open(p)); mm = re.search(r"stream(\d+)", p)
        sk = int(mm[1]); nf = sum(1 for (s, k) in games if (s - 1) % 10 == sk)
        g = j.get("meta", {}).get("games")
        if g is not None and g != nf:
            st.append((os.path.basename(p), g, nf))
    print(tag, "files", len(games), "missing", len(missing), "err", errs, "summary_err", S.get("error_games"),
          "badstream", bad_stream, "draws", draws, "k", dict(ks), "methods", len(meths),
          "pair", round(pm, 4), S["pair_score_A"]["mean"], len(per_seed), S["n_pairs"], "seedmatch", match_seed)
    if st: print("   state>files:", st)
print("TOTAL error games", tot_err, "tags with errors", tags_err)
