"""Verifier: the 4 two-sided subgames (seeds 5500-5503) with OURBOT_DET_LEAF=1 (leaf evaluated twice, must agree),
plus the pre-fix shared-stage-0-block variant (toy_two_sided.seqform2_shared) for V_real and sum_phi q V_conc."""
import json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts")); sys.path.insert(0, HERE)
assert os.environ.get("OURBOT_DET_LEAF") == "1"
import tera_subgame as ts  # noqa: E402
import tera_two_sided as t2  # noqa: E402
import toy_two_sided as tt  # noqa: E402
MIS = [0]


def st2(fn, key, tries=12, need=3):
    a = fn(); b = fn()
    if key(a) != key(b): MIS[0] += 1
    return a


ts._stable = st2
STASH = {}
_ob = t2.build2


def b2(*a):
    r = _ob(*a); STASH["r"] = r; return r


t2.build2 = b2


def run(seed):
    out = t2.worker(seed)
    if "error" in out:
        return out
    A0, B0, info = STASH["r"]
    th, ph = out["thetas"], out["phis"]
    bu = {t: 1 / len(th) for t in th}; qu = {p: 1 / len(ph) for p in ph}
    out["V_real_bug"] = tt.seqform2_shared(info, A0, B0, th, ph, bu, qu)
    out["V_conc_bug"] = {p: tt.seqform2_shared(info, A0, B0, th, ph, bu, {pp: float(pp == p) for pp in ph}) for p in ph}
    out["det_mismatch"] = MIS[0]
    return out


if __name__ == "__main__":
    from multiprocessing import get_context
    with get_context("spawn").Pool(2) as pool, open(sys.argv[1], "a") as f:
        for out in pool.imap_unordered(run, [5500, 5501, 5502, 5503]):
            f.write(json.dumps(out) + "\n"); f.flush()
            print({k: out.get(k) for k in ("seed", "own", "opp", "Delta", "V_real", "V_conc", "conceal_value", "voi_opp_two_sided", "voi_opp_one_sided_avg", "V_real_bug", "V_conc_bug", "det_mismatch", "error")}, flush=True)
