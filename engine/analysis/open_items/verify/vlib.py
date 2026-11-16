import glob, json, os, re, math
import numpy as np
RES = "/Users/biofanat/Documents/classes/f26/cs535/final/engine/showdown/results"
FN = re.compile(r"^(.+)-s(\d+)-g([12])\.json$")


def games(tag):
    out = {}
    for p in glob.glob(os.path.join(RES, tag, "replays_headless", f"{tag}-s*-g*.json")):
        m = FN.match(os.path.basename(p))
        if not m: continue
        d = json.load(open(p))
        out[(int(m[2]), int(m[3]))] = dict(win=d.get("winner_role"), err=d.get("error"), stream=d.get("stream"),
                                           active=d.get("active"), methods=d.get("methods"), turns=d.get("turns"))
    return out


def pairs(tag, role="A"):
    g = games(tag)
    out = {}
    for (s, k) in g:
        if k != 1: continue
        a, b = g.get((s, 1)), g.get((s, 2))
        if a and b and not a["err"] and not b["err"]:
            out[s] = ((a["win"] == role) + (b["win"] == role)) / 2
    return out


def decs(tag, s, k, role="A"):
    p = os.path.join(RES, tag, "replays_headless", f"{tag}-s{s}-g{k}.{role}.decisions.jsonl")
    if not os.path.exists(p): return None
    return [json.loads(l) for l in open(p) if l.strip()]


def ci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean(); se = x.std(ddof=1) / math.sqrt(n)
    return round(m, 4), round(m - 1.96 * se, 3), round(m + 1.96 * se, 3), n


def cl_ci(x, g):
    from scipy import stats
    x = np.asarray(x, float); g = np.asarray(g); ug = np.unique(g); G = len(ug); m = x.mean()
    s = np.array([(x[g == u] - m).sum() for u in ug])
    se = math.sqrt(G / (G - 1) * (s ** 2).sum()) / len(x)
    h = stats.t.ppf(0.975, G - 1) * se
    return round(m, 4), round(m - h, 3), round(m + h, 3), G
