"""Case-study figures: a real Showdown battle position next to what our agent computed there.

Run: engine/.venv/bin/python report/make_case_studies.py [--shots DIR] [--refresh-shots]
Writes report/figures/case_study_<n>.{png,pdf}, report/figures/case_studies.json (every number shown, with its
source file) and report/figures/case_studies_captions.md (caption drafts whose numbers come from that JSON).

Source: the LOG run (hedge_bayes vs foul-play stock, OURBOT_LOG_ABAR=1), engine/showdown/results/LOG/replays_headless/
  <game>.json            the game record (teams, sim input log, winner, turns)
  <game>.A.decisions.jsonl  our per-turn records: abar/labels/legal (averaged stage matrix, engine units),
                         Q (the four Hedge experts' strategies; Q[1] = the agent-form equilibrium), pfull (played
                         mixture), weights, bayes (style posterior and q-hat), feedback (the opponent's actual move).
Screenshot: the result folders keep no replay HTML, so the spectator log is rebuilt by feeding the record's
input_log to `node pokemon-showdown simulate-battle` (deterministic; checked against the record's winner and turn
count), wrapped in the same replay page as engine/showdown/run_ourbot.REPLAY_HTML, and captured with headless Chrome
(the browser used by the demo recorder report/demo/combined/record_replay.py) after the viewer's own
Replays.battle.seekTurn(t), i.e. the position at the start of turn t, when the agent decided. Our side is at the
bottom. The viewer script loads from play.pokemonshowdown.com, so capturing needs the network; captures are cached in
--shots (default: $TMPDIR/cs535_case_study_shots), outside the repo.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report/figures"
RES = ROOT / "engine/showdown/results"
SHOWDOWN = Path.home() / "sandbox/pokemon-showdown"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
sys.path.insert(0, "/Users/biofanat/.claude/skills/figure-style")
sys.path.insert(0, str(ROOT / "report"))
from figcheck import check_figure  # noqa: E402
from make_figures import COL  # noqa: E402  (one colour per method, shared with every figure)

plt.style.use("/Users/biofanat/.claude/skills/figure-style/figure.mplstyle")

INK, GRAY, LIGHT = "#222222", "#888888", "#EDEDED"
OURS = COL["eq_top"]                    # our agent (accent)
OPP = COL["mcts"]                       # MCTS / foul-play
DIVERGE = LinearSegmentedColormap.from_list("div", [OPP, "#F4F4F4", OURS])   # bad for us -> 0 -> good for us
SEQ = LinearSegmentedColormap.from_list("seq", ["#F4F4F4", OURS])
W_IN, H_IN = 7.09, 2.6                  # 180 mm wide
SHOT_W_IN = 3.4                         # screenshot width; its height keeps the viewer's 16:9
SHOT_FRAC = SHOT_W_IN / W_IN            # share of the width taken by the screenshot
METHOD_NAME = {"stock": "MCTS"}         # replay display name per method; every other method is ours ("Ours")

# Plain names (proposed; not yet in engine/figures/FIGURE_VOCAB.md except the three equilibrium/safest terms).
EXPERT = {"top": "Equilibrium, top move", "sampled": "Equilibrium, sampled", "safest": "Safest move",
          "safex": "Safe exploit"}
STYLE = {"eq": "Equilibrium", "br_unif": "Best reply to uniform", "br_us": "Best reply to us",
         "safe": "Cautious (maximin)", "uniform": "Uniform random"}
# Emphasis: the highest-belief style in the opponent-model colour; the others in grays, by belief rank, each with its
# own line style so they stay separable.
STYLE_RANKS = [(COL["bayes"], "-"), ("#444444", (0, (4, 1.5))), ("#6E6E6E", (0, (1, 1.2))),
               ("#999999", (0, (4, 1.5, 1, 1.5))), ("#BBBBBB", "-")]

CASES = [
    dict(n=1, kind="mixed", tag="LOG", game="LOG-s15-g2", turn=2),
    dict(n=2, kind="bayes", tag="LOG", game="LOG-s2-g1", turn=15),
    dict(n=3, kind="hedge", tag="LOG", game="LOG-s272-g2", turn=18),
]


# ---------------------------------------------------------------- data
def player(rec, role):
    """Display name of role A/B, written into the replay instead of the run's bot names."""
    return METHOD_NAME.get(rec["methods"][role], "Ours")


def load(tag, game):
    d = RES / tag / "replays_headless"
    rec = json.loads((d / f"{game}.json").read_text())
    dec = [json.loads(ln) for ln in open(d / f"{game}.A.decisions.jsonl")]
    return rec, dec, d


def turn_record(dec, turn):
    """The start-of-turn decision (the position seekTurn shows), not a mid-turn forced switch."""
    return next(r for r in dec if r.get("turn") == turn and "abar" in r and len(r["labels"]) > 1)


def feedback_for(dec, turn):
    return next(r["feedback"] for r in dec if r.get("feedback") and r["feedback"]["for_turn"] == turn)


def names_for(ids):
    """Display names from Showdown's Dex: species or move id (with optional -tera) -> name."""
    base = sorted({i.removesuffix("-tera") for i in ids})
    js = ("const {Dex}=require('./dist/sim');const ids=JSON.parse(process.argv[1]);const o={};"
          "for(const i of ids){const s=Dex.species.get(i),m=Dex.moves.get(i);"
          "o[i]=s.exists?{n:s.name,k:'switch'}:m.exists?{n:m.name,k:'move'}:{n:i,k:'?'}}"
          "console.log(JSON.stringify(o))")
    o = json.loads(subprocess.run(["node", "-e", js, json.dumps(base)], cwd=SHOWDOWN, capture_output=True,
                                  text=True, check=True).stdout)
    out = {}
    for i in ids:
        b = o[i.removesuffix("-tera")]
        nm = b["n"] + (" (Tera)" if i.endswith("-tera") else "")
        out[i] = dict(name=nm, kind=b["k"], label=("→ " + nm) if b["k"] == "switch" else nm)
    return out


# ---------------------------------------------------------------- screenshot
def spectator_log(rec):
    """Re-run the battle from its input log; return the public (spectator) log, checked against the record."""
    raw = subprocess.run(["node", "pokemon-showdown", "simulate-battle"], cwd=SHOWDOWN, capture_output=True,
                         text=True, input="\n".join(rec["input_log"]) + "\n", timeout=300, check=True).stdout
    lines = []
    for block in raw.split("\n\n"):
        b = block.strip("\n").split("\n")
        if b[0] == "update":
            lines += b[1:]
    pub, i = [], 0
    while i < len(lines):                      # |split|pN: private copy, then public copy (headless.public_lines)
        if lines[i].startswith("|split|"):
            pub.append(lines[i + 2])
            i += 3
        else:
            pub.append(lines[i])
            i += 1
    pub = [ln for ln in pub if ln and not ln.startswith("|t:|")]
    turns = [int(ln.split("|")[2]) for ln in pub if ln.startswith("|turn|")]
    win = [ln.split("|")[2] for ln in pub if ln.startswith("|win|")]
    assert turns[-1] == rec["turns"] and win == [rec["winner"]], (rec["tag"], turns[-1], win)
    log = "\n".join(pub)
    for role, side in (("A", rec["sides"]["A"]), ("B", rec["sides"]["B"])):
        log = log.replace(rec["sides"]["names"][side], player(rec, role))
    return log


PAGE = """<!DOCTYPE html><meta charset="utf-8"><title>{tag}</title>
<style>body{{margin:0!important;padding:0!important;overflow:hidden}}
.battle-log,.replay-controls,.replay-controls-2{{display:none!important}}.wrapper{{margin:0!important;max-width:none!important}}</style>
<div class="wrapper replay-wrapper"><div class="battle"></div><div class="battle-log"></div>
<div class="replay-controls"></div><div class="replay-controls-2"></div>
<script type="text/plain" class="battle-log-data">{log}</script></div>
<script src="https://play.pokemonshowdown.com/js/replay-embed.js"></script>
<script>window.addEventListener('load',function(){{setTimeout(function(){{var b=Replays.battle;
{flip}b.seekTurn({turn});setTimeout(function(){{b.pause();}},500);}},{delay});}});</script>"""


def screenshot(rec, turn, path, refresh=False):
    if path.exists() and not refresh:
        return
    log = spectator_log(rec)
    flip = "b.switchViewpoint();" if rec["sides"]["A"] == "p2" else ""     # our side at the bottom
    for budget, delay in ((30000, 6000), (60000, 12000)):
        work = Path(tempfile.mkdtemp(prefix="cs_shot_"))
        page = work / "page.html"
        page.write_text(PAGE.format(tag=rec["tag"], log=log.replace("</", "<\\/"), flip=flip, turn=turn, delay=delay))
        raw = work / "shot.png"
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--mute-audio",
                        f"--user-data-dir={work}/profile", "--window-size=640,370", "--force-device-scale-factor=2",
                        f"--virtual-time-budget={budget}", f"--screenshot={raw}", page.as_uri()],
                       capture_output=True, timeout=180)
        if raw.exists():
            im = Image.open(raw).convert("RGB").crop((0, 20, 1280, 740))   # drop the viewer's 10 px top strip
            if np.asarray(im).std() > 20:                                     # not a blank page
                path.parent.mkdir(parents=True, exist_ok=True)
                im.save(path)
                return
    raise RuntimeError(f"screenshot failed for {rec['tag']} turn {turn}")


# ---------------------------------------------------------------- figure helpers
def frame(shot):
    fig = plt.figure(figsize=(W_IN, H_IN))
    im = np.asarray(Image.open(shot))
    h = SHOT_W_IN * im.shape[0] / im.shape[1] / H_IN
    ax = fig.add_axes([0.0, (1 - h) / 2, SHOT_FRAC, h])
    ax.imshow(im, interpolation="lanczos")
    ax.set_axis_off()
    return fig


def corner(ax, top, left):
    """Header for a labelled grid: what the columns are (top) and what the rows are (left)."""
    ax.text(-0.6, -0.6, f"{top} →\n{left} ↓", ha="right", va="bottom", color=GRAY, transform=ax.transData,
            linespacing=1.3)


def wrap(s):
    """Two-line tick label for a long name."""
    return s.replace(" ", "\n", 1) if len(s) > 9 and " " in s else s


def finish(fig, name):
    issues = check_figure(fig, min_pt=8, max_words=12)
    print(f"[{name}] figcheck:", issues if issues else "clean")
    with plt.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(OUT / f"{name}.pdf")
        fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)
    kb = (OUT / f"{name}.png").stat().st_size / 1024
    print(f"[{name}] png {kb:.0f} KB")
    assert kb < 1500, f"{name}.png is {kb:.0f} KB"
    return issues


def signed(v):
    return f"{v:+.0f}".replace("-", "\u2212")


def pct(v):
    return f"{100 * v:.0f}%"


def outcome(rec, game):
    return dict(game=game, source_json=str((RES / rec["run_tag"] / "replays_headless" / f"{game}.json").relative_to(ROOT)),
                source_decisions=str((RES / rec["run_tag"] / "replays_headless" / f"{game}.A.decisions.jsonl")
                                     .relative_to(ROOT)),
                our_side=rec["sides"]["A"], winner=player(rec, rec["winner_role"]), total_turns=rec["turns"])


# ---------------------------------------------------------------- case 1: a mixed equilibrium
def case_mixed(c, rec, dec, shot):
    r = turn_record(dec, c["turn"])
    fb = feedback_for(dec, c["turn"])
    A, L, R = np.array(r["abar"]), r["labels"], r["legal"]
    x, p, q = np.array(r["Q"][1]), np.array(r["pfull"]), np.array(r["bayes"]["qhat"])
    rows = [int(i) for i in np.argsort(-x)[:3]]
    cols = [int(j) for j in np.argsort(-q)[:4]]
    jb = L.index(fb["b"])
    forced = jb not in cols                    # the opponent's actual reply is always shown
    if forced:
        cols[-1] = jb
    played = r["move"].removeprefix("switch ")
    nm = names_for([R[i] for i in rows] + [L[j] for j in cols] + [played])
    sub = A[np.ix_(rows, cols)]

    # drawn transposed: columns = our options (equilibrium share), rows = the opponent's replies (q-hat)
    fig = frame(shot)
    ax = fig.add_axes([SHOT_FRAC + 0.20, 0.10, 0.29, 0.55])
    lim = np.abs(sub).max()
    ax.imshow(sub.T, cmap=DIVERGE, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for a in range(len(rows)):
        for b in range(len(cols)):
            v = sub[a, b]
            ax.text(a, b, f"{v:+.0f}".replace("-", "−"), ha="center", va="center",
                    color="white" if abs(v) > 0.6 * lim else INK)
    ax.xaxis.tick_top()
    ax.set_xticks(range(len(rows)), [f"{nm[R[i]]['label']}\n{pct(x[i])}" for i in rows])
    ax.set_yticks(range(len(cols)), [f"{nm[L[j]]['label']}  {pct(q[j])}" for j in cols])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    a_pl = rows.index(R.index(played))
    ax.add_patch(Rectangle((a_pl - 0.5, -0.5), 1, len(cols), fill=False, ec=INK, lw=1.4))
    b_op = cols.index(jb)
    ax.add_patch(Rectangle((-0.5, b_op - 0.5), len(rows), 1, fill=False, ec=INK, lw=1.4, ls=(0, (2, 1.5))))
    corner(ax, "Our move", "Opponent reply")
    fig.text(SHOT_FRAC + 0.02, 0.97, "Our equilibrium mixes two plans", fontsize=10, ha="left", va="top")
    fig.text(SHOT_FRAC + 0.02, 0.885, f"Played {nm[played]['label']}; opponent used {nm[L[jb]]['label']}",
             ha="left", va="top", color=INK)
    fig.text(SHOT_FRAC + 0.02, 0.02, "Payoff to us (engine units)", ha="left", va="bottom", color=GRAY)
    issues = finish(fig, f"case_study_{c['n']}")
    return dict(
        **outcome(rec, c["game"]), turn=c["turn"], figure=f"case_study_{c['n']}",
        cell_units="averaged stage matrix Abar, heuristic engine units (positive = good for us)",
        rows=[dict(id=R[i], name=nm[R[i]]["name"], kind=nm[R[i]]["kind"], equilibrium=round(float(x[i]), 4),
                   played_prob=round(float(p[i]), 4)) for i in rows],
        cols=[dict(id=L[j], name=nm[L[j]]["name"], kind=nm[L[j]]["kind"], qhat=round(float(q[j]), 4)) for j in cols],
        payoff=sub.round(2).tolist(), payoff_layout="payoff[row][col]; figure draws it transposed",
        cols_rule="opponent's top replies by q-hat" + ("; last replaced by its actual reply" if forced else ""),
        tv_equilibrium_vs_played=round(float(0.5 * np.abs(x - p).sum()), 4),
        n_our_options=len(R), n_opp_options=len(L),
        played=dict(id=played, name=nm[played]["name"], raw=r["move"]),
        opponent=dict(id=fb["b"], name=nm[L[jb]]["name"], how=fb["how"], qhat=round(float(q[jb]), 4)),
        payoff_played_vs_actual=round(float(A[R.index(played), jb]), 2),
        payoff_other_vs_actual={R[i]: round(float(A[i, jb]), 2) for i in rows if R[i] != played},
        figcheck=issues or "clean")


# ---------------------------------------------------------------- case 2: the opponent model
def case_bayes(c, rec, dec, shot):
    r = turn_record(dec, c["turn"])
    fb = feedback_for(dec, c["turn"])
    L, q = r["bayes"]["labels"], np.array(r["bayes"]["qhat"])
    post = r["bayes"]["posterior"]
    traj, seen = [], set()                     # posterior at each start-of-turn decision, turns 1..t
    for d in dec:
        t = d.get("turn")
        if d.get("bayes") and t not in seen and t <= c["turn"] and len(d["bayes"]["labels"]) > 1:
            seen.add(t)
            traj.append((t, d["bayes"]["posterior"]))
    top = [int(j) for j in np.argsort(-q)[:5]]
    jb = L.index(fb["b"])
    if jb not in top:
        top[-1] = jb
    nm = names_for([L[j] for j in top] + [r["move"].removeprefix("switch ")])

    fig = frame(shot)
    ax1 = fig.add_axes([SHOT_FRAC + 0.07, 0.60, 0.20, 0.28])
    T = [t for t, _ in traj]
    order = sorted(STYLE, key=lambda s: -post[s])          # by belief at this turn; order[0] = the highest
    sty = dict(zip(order, STYLE_RANKS))
    for s in order[::-1]:                                   # highest-belief style drawn last, on top
        y = [pp[s] for _, pp in traj]
        ax1.plot(T, y, color=sty[s][0], ls=sty[s][1], lw=1.6 if s == order[0] else 1.1)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax1.set_xlim(1, c["turn"])
    ticks = [1] + list(range(5, c["turn"] + 1, 5))
    ax1.set_xticks(ticks, ["Turn 1"] + [str(t) for t in ticks[1:]])
    ax1.set_ylabel("Belief")
    ax1.set_title("Opponent style belief", fontsize=10, loc="left", pad=4)
    ys = np.linspace(0.97, 0.03, len(order))
    for s, yy in zip(order, ys):           # direct labels in ink, with a line swatch, ordered by belief
        ax1.plot([1.03, 1.11], [yy, yy], transform=ax1.transAxes, clip_on=False, color=sty[s][0], ls=sty[s][1],
                 lw=1.6 if s == order[0] else 1.1)
        ax1.annotate(f"{STYLE[s]} {post[s]:.2f}", xy=(1.14, yy), xycoords="axes fraction",
                     color=INK, va="center", ha="left")

    ax2 = fig.add_axes([SHOT_FRAC + 0.18, 0.08, 0.30, 0.28])
    ypos = np.arange(len(top))[::-1]
    ax2.barh(ypos, [q[j] for j in top], color=[OPP if j == jb else LIGHT for j in top], height=0.7)
    u = 1 / len(L)
    ax2.plot([u] * 2, [-0.45, ypos[0] + 0.45], color=GRAY, lw=0.8)
    ax2.text(u, -0.55, "uniform", color=GRAY, ha="center", va="top")
    for y, j in zip(ypos, top):            # value labels start right of the reference line, so they never touch it
        ax2.text(max(q[j], u) + 0.012, y, pct(q[j]) + (", used" if j == jb else ""), va="center", ha="left")
    ax2.set_yticks(ypos, [nm[L[j]]["label"] for j in top])
    ax2.tick_params(axis="y", length=0)
    ax2.set_xlim(0, max(q[top]) * 1.3)
    ax2.set_ylim(-1.3, len(top) - 0.5)
    ax2.set_xticks([])
    ax2.spines["bottom"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.set_title("Predicted opponent move", fontsize=10, loc="left", pad=4)
    issues = finish(fig, f"case_study_{c['n']}")
    played = r["move"].removeprefix("switch ")
    return dict(
        **outcome(rec, c["game"]), turn=c["turn"], figure=f"case_study_{c['n']}",
        styles={s: STYLE[s] for s in STYLE},
        posterior_first=traj[0][1], posterior_at_turn=post, posterior_after_update=fb["bayes"]["posterior"],
        posterior_trajectory=[dict(turn=t, posterior=pp) for t, pp in traj],
        n_opp_options=len(L), uniform_prob=round(1 / len(L), 4),
        qhat_shown=[dict(id=L[j], name=nm[L[j]]["name"], kind=nm[L[j]]["kind"], qhat=round(float(q[j]), 4))
                    for j in top],
        opponent=dict(id=fb["b"], name=nm[L[jb]]["name"], how=fb["how"], qhat=round(float(q[jb]), 4),
                      qhat_rank=int((q > q[jb]).sum()) + 1, hit_qhat=fb["bayes"]["hit_qhat"]),
        played=dict(id=played, name=nm[played]["name"], raw=r["move"]),
        figcheck=issues or "clean")


# ---------------------------------------------------------------- case 3: Hedge over experts
def case_hedge(c, rec, dec, shot):
    r = turn_record(dec, c["turn"])
    fb = feedback_for(dec, c["turn"])
    R, Q, p, w = r["legal"], np.array(r["Q"]), np.array(r["pfull"]), r["weights"]
    cols = [int(i) for i in np.argsort(-p)[:4]]
    played = r["move"].removeprefix("switch ")
    opp_names = names_for([fb["b"]])
    nm = names_for([R[i] for i in cols] + [played] + list(r["experts"].values()))
    ex = list(EXPERT)
    M = np.vstack([Q[[ex.index(e) for e in ex]][:, cols], p[cols]])

    fig = frame(shot)
    ax = fig.add_axes([SHOT_FRAC + 0.15, 0.04, 0.30, 0.62])
    ax.imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for a in range(M.shape[0]):
        for b in range(M.shape[1]):
            v = M[a, b]
            if v >= 0.005:
                ax.text(b, a, pct(v), ha="center", va="center", color="white" if v > 0.6 else INK)
    ax.axhline(len(ex) - 0.5, color="white", lw=3)
    ax.xaxis.tick_top()
    ax.set_xticks(range(len(cols)), [wrap(nm[R[i]]["label"]) for i in cols])
    ax.set_yticks(range(len(ex) + 1), [wrap(EXPERT[e]) for e in ex] + ["Played\nmixture"])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    b_pl = cols.index(R.index(played))
    ax.add_patch(Rectangle((b_pl - 0.5, len(ex) - 0.5), 1, 1, fill=False, ec=INK, lw=1.4))
    for a, e in enumerate(ex):                 # Hedge weights, right of the grid
        ax.text(len(cols) - 0.35, a, f"{w[e]:.2f}", ha="left", va="center")
    ax.text(len(cols) - 0.35, -0.6, "Weight", ha="left", va="bottom", color=GRAY)
    corner(ax, "Our move", "Expert")
    fig.text(SHOT_FRAC + 0.02, 0.97, "Hedge blends four experts", fontsize=10, ha="left", va="top")
    fig.text(SHOT_FRAC + 0.02, 0.885, f"Played {nm[played]['label']}; opponent used "
             f"{opp_names[fb['b']]['label']}", ha="left", va="top")
    issues = finish(fig, f"case_study_{c['n']}")
    return dict(
        **outcome(rec, c["game"]), turn=c["turn"], figure=f"case_study_{c['n']}",
        experts={e: EXPERT[e] for e in ex}, weights=w,
        expert_top_move={e: dict(id=r["experts"][e], name=nm[r["experts"][e]]["name"])
                         for e in ex},
        moves_shown=[dict(id=R[i], name=nm[R[i]]["name"], kind=nm[R[i]]["kind"],
                          expert_probs={e: round(float(Q[ex.index(e), i]), 4) for e in ex},
                          played_prob=round(float(p[i]), 4)) for i in cols],
        played=dict(id=played, name=nm[played]["name"], raw=r["move"], prob=round(float(p[R.index(played)]), 4)),
        opponent=dict(id=fb["b"], name=opp_names[fb["b"]]["name"], how=fb["how"]),
        n_our_options=len(R), weights_note="carried across the games of the run's stream (not reset per game)",
        figcheck=issues or "clean")


# ---------------------------------------------------------------- captions
def captions(cs):
    c1, c2, c3 = cs
    r1 = {r["id"]: r for r in c1["rows"]}
    a, b = c1["rows"][0], c1["rows"][1]
    sw = lambda r: ("a switch to " if r["kind"] == "switch" else "") + r["name"]
    t1 = (f"**Case study 1.** Our equilibrium at one turn splits between {sw(a)} and {sw(b)}; the table shows payoffs "
          f"to us against the opponent's likely replies (game {c1['game']}, turn {c1['turn']}). The opponent is the "
          f"MCTS bot (stock foul-play); our side won in {c1['total_turns']} turns. The equilibrium puts "
          f"{pct(a['equilibrium'])} on {a['name']} and {pct(b['equilibrium'])} on {b['name']}; against the opponent's "
          f"{len(c1['cols'])} "
          + ("most likely replies" if "replaced" not in c1["cols_rule"] else "most likely replies and its actual one")
          + f" our top {len(c1['rows'])} options each win against different replies, so neither dominates, and the "
          f"played Hedge mixture is within {c1['tv_equilibrium_vs_played']:.3f} (total variation) of the equilibrium. "
          f"The agent played {c1['played']['name']} and the opponent used {c1['opponent']['name']}, worth "
          f"{signed(c1['payoff_played_vs_actual'])} to us, against "
          + " and ".join(f"{signed(v)} for {r1[k]['name']}" for k, v in c1["payoff_other_vs_actual"].items())
          + ". Solid box: our move; dashed box: the opponent's reply. → marks a switch. Payoffs are in engine units, "
          "averaged over the four sampled opponent teams.")
    po, pf = c2["posterior_at_turn"], c2["posterior_first"]
    lead = max(po, key=po.get)
    t2 = (f"**Case study 2.** The belief over five opponent styles up to turn {c2['turn']}, and the move the believed "
          f"style predicts (game {c2['game']}). Our side won in {c2['total_turns']} turns. From an even start "
          f"({pf[lead]:.2f} per style) the belief moves to {c2['styles'][lead]} (the opponent best-responds to a "
          f"uniformly random guess about our move), {po[lead]:.2f} at turn {c2['turn']}. The prediction gives "
          f"{c2['opponent']['name']} {pct(c2['opponent']['qhat'])} among {c2['n_opp_options']} opponent options "
          f"(uniform: {pct(c2['uniform_prob'])}), and the opponent then used {c2['opponent']['name']}. "
          "→ marks a switch.")
    w = c3["weights"]
    pl = c3["played"]
    tm = c3["expert_top_move"]
    t3 = (f"**Case study 3.** Four experts' strategies and the Hedge-weighted mixture we played at one decision "
          f"(game {c3['game']}, turn {c3['turn']}). Our side won in {c3['total_turns']} turns. The experts disagree: "
          f"{c3['experts']['top']} and {c3['experts']['sampled']} favour {tm['top']['name']}; "
          f"{c3['experts']['safest']} picks {tm['safest']['name']}; {c3['experts']['safex']} picks "
          f"{tm['safex']['name']}. With Hedge weights " + ", ".join(f"{w[e]:.2f}" for e in c3["experts"])
          + f" (carried over from earlier games), the played mixture puts {pct(pl['prob'])} on {pl['name']}, which "
          f"the agent played; the opponent used {c3['opponent']['name']}. Rows are the four experts and, at the "
          f"bottom, the played mixture; columns are our {len(c3['moves_shown'])} most-played options (of "
          f"{c3['n_our_options']}). → marks a switch.")
    return "# Case-study caption drafts\n\nEvery number below is read from `case_studies.json`.\n\n" + \
        "\n\n".join([t1, t2, t3]) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default=os.path.join(tempfile.gettempdir(), "cs535_case_study_shots"))
    ap.add_argument("--refresh-shots", action="store_true")
    args = ap.parse_args()
    out = []
    for c in CASES:
        rec, dec, _ = load(c["tag"], c["game"])
        shot = Path(args.shots) / f"{c['game']}-t{c['turn']}.png"
        screenshot(rec, c["turn"], shot, args.refresh_shots)
        fn = {"mixed": case_mixed, "bayes": case_bayes, "hedge": case_hedge}[c["kind"]]
        out.append(dict(case=c["n"], kind=c["kind"], **fn(c, rec, dec, shot)))
    (OUT / "case_studies.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    (OUT / "case_studies_captions.md").write_text(captions(out))
    print("wrote", OUT / "case_studies.json", "and", OUT / "case_studies_captions.md")


if __name__ == "__main__":
    main()
