"""B1 aggregate: reads out_B1_log_games.json -> out_B1_summary.json"""
import json, collections
import numpy as np
from oi_common import *
d = json.load(open(os.path.join(OUT, "out_B1_log_games.json")))
games = d["games"]
def stats(sel, name):
    out = {}
    for mp in ("logit", "tanh"):
        Dall, gid, stm = [], [], []
        sums = collections.defaultdict(list)
        per_game = dict(sumD=[], sumabsD=[], sumR=[], sumS=[], opp=[], our=[], lump=[], nknown=[], ndec=[], z=[], zphi=[], unk=[], sumabsD_scaled=[])
        for g in sel:
            m = g[mp]
            Ds = [x for x in m["D"] if x is not None]
            Dall += Ds; gid += [(g["seed"], g["game"])] * len(Ds); stm += [stream_of(g["seed"], 1)] * len(Ds)
            per_game["sumD"].append(m["sumD_known"]); per_game["sumabsD"].append(m["sumabsD_known"]); per_game["sumR"].append(m["sumR_known"])
            per_game["sumS"].append(m["sumS_known"]); per_game["opp"].append(m["sum_opp_gap"]); per_game["our"].append(m["sum_our_gap"]); per_game["lump"].append(m["lump_total"]); per_game["nknown"].append(m["n_known"])
            per_game["ndec"].append(g["n_dec"]); per_game["z"].append(g["z"]); per_game["zphi"].append(m["z_minus_phi_v1"]); per_game["unk"].append(m["sum_lump_unknown"])
            per_game["sumabsD_scaled"].append(m["sumabsD_known"] * g["n_dec"] / max(m["n_known"], 1))
        Dall = np.array(Dall)
        n = len(Dall)
        # SE: decision-level iid, game-cluster, stream-cluster
        gcodes = {k: i for i, k in enumerate(sorted(set(gid)))}
        gg = np.array([gcodes[k] for k in gid])
        dm = Dall.mean()
        out[mp] = dict(n_games=len(sel), n_dec_known=n, mean_D_per_dec=float(dm),
                       se_iid=float(Dall.std(ddof=1) / np.sqrt(n)), se_game_cluster=float(cluster_ci(Dall, gg)[1]),
                       se_stream_cluster=float(cluster_ci(Dall, np.array(stm))[1]),
                       mean_abs_D=float(np.abs(Dall).mean()),
                       per_game={k: dict(mean=float(np.mean(v)), se=float(np.std(v, ddof=1) / np.sqrt(len(v)))) for k, v in per_game.items()},
                       drift_channel_minus_sumD_per_game=float(-np.mean(per_game["sumD"])))
    return out
res = dict(source="out_B1_log_games.json (from b1_log_identity.py)")
res["all"] = stats(games, "all")
for seat in ("p1", "p2"):
    res[seat] = stats([g for g in games if g["side"] == seat], seat)
res["A_win"] = stats([g for g in games if g["z"] > 0], "win")
res["A_loss"] = stats([g for g in games if g["z"] < 0], "loss")
res["closure"] = dict(max_dec=max(max(g["logit"]["closure_dec"], g["tanh"]["closure_dec"]) for g in games),
                      max_game=max(max(g["logit"]["closure_game"], g["tanh"]["closure_game"]) for g in games))
res["coverage"] = dict(n_games=len(games), n_dec=sum(g["n_dec"] for g in games), n_known=sum(g["n_j_known"] for g in games),
                       why=dict(collections.Counter(w for g in games for w in g["whys"])), i_unmatched=sum(g["i_unmatched"] for g in games),
                       forced_decisions_not_in_identity="records without abar (single legal option) are excluded; D spans them")
res["E_phi_v1"] = {mp: dict(mean=float(np.mean([g[mp]["phi_v1"] for g in games])), se=float(np.std([g[mp]["phi_v1"] for g in games], ddof=1) / np.sqrt(len(games)))) for mp in ("logit", "tanh")}
res["E_z"] = dict(mean=float(np.mean([g["z"] for g in games])), n=len(games))
dump("out_B1_summary.json", res)
for k in ("all", "p1", "p2", "A_win", "A_loss"):
    for mp in ("logit", "tanh"):
        s = res[k][mp]; pg = s["per_game"]
        print(k, mp, "n_g", s["n_games"], "n_dec", s["n_dec_known"], "D/dec %.4f se_iid %.4f se_g %.4f se_s %.4f |D| %.3f" % (s["mean_D_per_dec"], s["se_iid"], s["se_game_cluster"], s["se_stream_cluster"], s["mean_abs_D"]),
              "sum|D|/g %.2f (scaled %.2f) sumD/g %.2f sumR %.2f sumS %.2f oppgap %.2f ourgap %.2f n/g %.1f" % (pg["sumabsD"]["mean"], pg["sumabsD_scaled"]["mean"], pg["sumD"]["mean"], pg["sumR"]["mean"], pg["sumS"]["mean"], pg["opp"]["mean"], pg["our"]["mean"], pg["nknown"]["mean"]))
print(res["closure"], res["coverage"], res["E_phi_v1"], res["E_z"])
