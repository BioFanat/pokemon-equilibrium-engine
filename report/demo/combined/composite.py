"""Combined demo: the real Showdown replay (top, 1180x380) over our agent's decision for the turn on screen
(bottom, 1180x420), synced by the replay viewer's own turn counter.

  python composite.py <game>.log <game>.A.decisions.jsonl <recdir> <out.mp4> [--gif out.gif --gif-start S --gif-len L]

<recdir> holds the recorder's frames.json ([jpg, t]) and turns.json ([t, Replays.battle.turn, ended]). For output
frame k (t = k/30) we take the latest screencast frame <= t and the latest turn sample <= t, and draw the panel of
that turn: a 4x4 slice of the payoff matrix of sampled world k* (our 4 most likely rows x the opponent's 4 most
likely columns), cells = estimated win chance 1/2(1 + tanh(M/160)), our mix as row bars, the column player's
equilibrium on M_k* above the columns, and the played row/column highlighted. Our side = p1.
"""
import bisect
import json
import math
import os
import re
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

W, TOP_H, BOT_H, FPS = 1180, 380, 420, 30
H = TOP_H + BOT_H
BG, PANEL = (24, 31, 44), (33, 42, 58)
INK, MUTED, ACCENT, GOOD, BAD = (232, 236, 242), (140, 152, 170), (240, 150, 110), (47, 102, 144), (181, 70, 47)
HF = "/System/Library/Fonts/Helvetica.ttc"
f = lambda s, b=False: ImageFont.truetype(HF, s, index=1 if b else 0)
F15, F16, F17, F18, F20, F22B, F26B, F40B = f(15), f(16), f(17), f(18), f(20), f(22, True), f(26, True), f(40, True)
MOVES = json.load(open(os.path.expanduser("~/sandbox/foul-play/fp/data/moves.json")))
ME, OPP = "p1", "p2"


def ident(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def win(m):
    return 0.5 * (1 + math.tanh(m / 160))


def mix(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def cell_color(p):          # diverging: orange (bad for us) -> grey (50%) -> blue (good for us)
    mid = (70, 78, 94)
    return mix(mid, GOOD, min(1, (p - 0.5) * 2.2)) if p >= 0.5 else mix(mid, BAD, min(1, (0.5 - p) * 2.2))


def short(s, n):
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------- spectator log: per-turn actions and display names ----------

def parse_log(path):
    """turn N -> {side: canonical action of that side's first action in the |turn|N block}; display names."""
    acts, names, turn, tera, fainted = {}, {}, 0, set(), set()
    for ln in open(path):
        p = ln.rstrip("\n").split("|")
        if len(p) < 2:
            continue
        kind = p[1]
        if kind == "turn":
            turn, tera, fainted = int(p[2]), set(), set()
            continue
        if kind == "faint":
            fainted.add(p[2][:2])
            continue
        if kind in ("switch", "drag", "move", "-terastallize", "cant") and len(p) > 2:
            side = p[2][:2]
            if kind == "switch":
                names[ident(p[3].split(",")[0])] = p[3].split(",")[0]
            if kind == "-terastallize":
                tera.add(side)
                continue
            if turn == 0 or side in acts.setdefault(turn, {}):
                continue
            if kind == "switch" and (side in fainted or "[from]" in ln):
                continue                      # faint replacement / U-turn switch: not the turn's chosen action
            if kind == "move":
                a = ident(p[3]) + ("-tera" if side in tera else "")
                acts[turn][side] = a
            elif kind == "switch":
                acts[turn][side] = ident(p[3].split(",")[0])
            elif kind == "cant":
                acts[turn][side] = None
    return acts, names


def label(canon, raw_label, names):
    """'stoneedge-tera' -> 'Stone Edge (tera)'; switch labels -> 'switch: Hippowdon'."""
    if raw_label.startswith("switch: "):
        sp = raw_label[len("switch: "):]
        return "switch: " + names.get(ident(sp), sp.title())
    tera = canon.endswith("-tera")
    base = canon[:-5] if tera else canon
    nm = MOVES.get(base, {}).get("name", base)
    return nm + (" (tera)" if tera else "")


# ---------- decisions ----------

def load_decisions(path):
    """turn -> (main record, [later records on the same turn])."""
    by = {}
    for ln in open(path):
        r = json.loads(ln)
        by.setdefault(r["turn"], []).append(r)
    return {t: (rs[0], rs[1:]) for t, rs in by.items()}


def canon_move(m):
    return m[len("switch "):] if m.startswith("switch ") else m


# ---------- panel ----------

def fit(d, text, sizes, bold, maxw):
    for s in sizes:
        fo = f(s, bold)
        if d.textlength(text, font=fo) <= maxw:
            return fo
    return f(sizes[-1], bold)


def wrap(d, text, font, maxw, sep=" "):
    """Greedy wrap on sep; returns lines."""
    lines, cur = [], ""
    for part in text.split(sep):
        cand = part if not cur else cur + sep + part
        if cur and d.textlength(cand, font=font) > maxw:
            lines.append(cur)
            cur = part
        else:
            cur = cand
    return lines + [cur]


def caption(d, head, body):
    """One line (bold head + body) if it fits at >= 16 px; otherwise two lines split at ' · '."""
    maxw = W - 48
    for s in (20, 19, 18, 17, 16):
        hb, fb = f(s, True), f(s)
        wh = d.textlength(head, font=hb)
        if wh + d.textlength(body, font=fb) <= maxw:
            d.text((24, 18), head, font=hb, fill=INK)
            d.text((24 + wh, 18), body, font=fb, fill=INK)
            return
    for s in (18, 17, 16, 15):
        hb, fb = f(s, True), f(s)
        wh = d.textlength(head, font=hb)
        parts = body.split(" · ")
        l1 = parts[0]
        k = 1
        while k < len(parts) and wh + d.textlength(l1 + " · " + parts[k], font=fb) <= maxw:
            l1 += " · " + parts[k]
            k += 1
        l2 = " · ".join(parts[k:])
        if d.textlength(l2, font=fb) <= maxw or s == 15:
            d.text((24, 8), head, font=hb, fill=INK)
            d.text((24 + wh, 8), l1, font=fb, fill=INK)
            d.text((24, 31), l2, font=fb, fill=INK)
            return


def panel_base():
    im = Image.new("RGB", (W, BOT_H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 2], fill=(60, 72, 94))
    return im, d


def message_panel(title, sub=""):
    im, d = panel_base()
    d.text((W // 2, BOT_H // 2 - 20), title, font=F40B, fill=ACCENT, anchor="mm")
    if sub:
        d.text((W // 2, BOT_H // 2 + 30), sub, font=F20, fill=MUTED, anchor="mm")
    return im


def decision_panel(turn, rec, extra, opp_act, our_act, names):
    im, d = panel_base()
    rows_raw, x = rec["rows_raw"], rec["x"]
    rl = [label(c, r, names) for c, r in zip(rows_raw, rec["rows"])]
    played_i = rows_raw.index(canon_move(rec["move"])) if canon_move(rec["move"]) in rows_raw else None
    we = rl[played_i] if played_i is not None else canon_move(rec["move"])
    if our_act != canon_move(rec["move"]):
        we += " (did not get to move)"
    they = "nothing (did not move)" if opp_act is None else label(
        opp_act, "switch: " + opp_act if opp_act in names else opp_act, names)
    then = ""
    for e in extra:                             # a later decision on the same turn (U-turn / faint replacement)
        c = canon_move(e["move"])
        if c in e["rows_raw"]:
            then += f" (then {label(c, e['rows'][e['rows_raw'].index(c)], names)})"
    if "M" not in rec:
        caption(d, f"Turn {turn} · ", f"only one legal option · we played {we}{then} · foul-play played {they}")
        d.text((W // 2, 210), "Forced move: nothing to decide this turn", font=F26B, fill=MUTED, anchor="mm")
        return im
    order = sorted(range(len(x)), key=lambda i: -x[i])
    mixs = " / ".join(f"{rl[i]} {round(100 * x[i])}%" for i in order[:3] if x[i] >= 0.01)
    caption(d, f"Turn {turn} · ",
            f"agent form over {rec['K']} sampled opponent teams · mix: {mixs} · we played {we}{then}"
            f" · foul-play played {they}")
    M, y, cols_raw = rec["M"], rec["y"], rec["cols_raw"]
    cl = [label(c, r, names) for c, r in zip(cols_raw, rec["cols"])]
    rows = order[:4]
    cols = sorted(range(len(y)), key=lambda j: -y[j])[:4]
    played_j = cols_raw.index(opp_act) if opp_act in cols_raw else None

    X0, cx, cy, cw, ch = 24, 330, 150, 140, 62
    d.text((cx, 58), "opponent's options (its equilibrium probability in sampled team #1)", font=F15, fill=MUTED)
    for k, j in enumerate(cols):
        hl = j == played_j
        nm = cl[j]
        tera = nm.endswith(" (tera)")
        nm = nm[:-7] if tera else nm
        xc = cx + k * cw + cw // 2
        d.text((xc, cy - 58), short(nm, 15), font=F17, fill=ACCENT if hl else INK, anchor="mm")
        d.text((xc, cy - 38), "tera" if tera else "", font=F15, fill=ACCENT, anchor="mm")
        d.text((xc, cy - 18), f"{y[j]:.2f}", font=F16, fill=MUTED, anchor="mm")
    d.text((X0, cy - 26), "our options (agent-form LP prob.)", font=F15, fill=MUTED)
    for kr, i in enumerate(rows):
        yy = cy + kr * ch
        hl = i == played_i
        d.text((X0, yy + 8), short(rl[i], 26), font=F18, fill=ACCENT if hl else INK)
        d.rounded_rectangle([X0, yy + 36, X0 + 220, yy + 46], 5, fill=(20, 26, 36))
        if x[i] > 0.005:
            d.rounded_rectangle([X0, yy + 36, X0 + max(6, int(220 * x[i])), yy + 46], 5, fill=ACCENT)
        d.text((X0 + 230, yy + 31), f"{x[i]:.2f}", font=F16, fill=MUTED)
        for kc, j in enumerate(cols):
            p = win(M[i][j])
            box = [cx + kc * cw + 3, yy + 3, cx + (kc + 1) * cw - 3, yy + ch - 3]
            d.rounded_rectangle(box, 8, fill=cell_color(p))
            d.text(((box[0] + box[2]) // 2, (box[1] + box[3]) // 2), f"{round(100 * p)}%", font=F20, fill=INK,
                   anchor="mm")
            if i == played_i and j == played_j:
                d.rounded_rectangle(box, 8, outline=(255, 255, 255), width=4)
    # right column: legend
    RX = cx + 4 * cw + 24
    d.text((RX, 70), "cell = our estimated", font=F15, fill=MUTED)
    d.text((RX, 90), "win chance if both", font=F15, fill=MUTED)
    d.text((RX, 110), "options are played", font=F15, fill=MUTED)
    for k, (p, t) in enumerate(((0.15, "15%"), (0.5, "50%"), (0.85, "85%"))):
        bx = RX + k * 62
        d.rounded_rectangle([bx, 138, bx + 56, 166], 6, fill=cell_color(p))
        d.text((bx + 28, 152), t, font=F15, fill=INK, anchor="mm")
    d.text((RX, 190), f"shown: {len(rl)} x {len(cl)} matrix,", font=F15, fill=MUTED)
    d.text((RX, 210), "top 4 x top 4 slice", font=F15, fill=MUTED)
    notes = []
    if played_i is not None and played_i not in rows:
        notes.append(f"we played {we} (outside the slice)")
    if played_j is None:
        notes.append(f"foul-play played {they} (not an option in sampled team #1)" if opp_act is not None
                     and opp_act not in cols_raw else f"foul-play played {they}")
    elif played_j not in cols:
        notes.append(f"foul-play played {they} (outside the slice)")
    ny = 250
    for t in notes:
        for ln in wrap(d, t, F15, W - RX - 16):
            d.text((RX, ny), ln, font=F15, fill=ACCENT)
            ny += 20
        ny += 8
    return im


def grey(im):
    return Image.blend(im, Image.new("RGB", im.size, BG), 0.6)


def main():
    logp, decp, recdir, out = sys.argv[1:5]
    opts = dict(zip(sys.argv[5::2], sys.argv[6::2]))
    acts, names = parse_log(logp)
    dec = load_decisions(decp)
    nturns = max(acts)
    frames = json.load(open(os.path.join(recdir, "frames.json")))
    turns = json.load(open(os.path.join(recdir, "turns.json")))
    ft, tt = [t for _, t in frames], [t for t, _, _ in turns]
    cache = {}

    def panel(turn, ended):
        key = (turn, ended)
        if key in cache:
            return cache[key]
        if ended:
            im = message_panel(f"Our agent won the battle in {nturns} turns",
                               "agent-form equilibrium over sampled opponent teams  vs  foul-play (MCTS)")
        elif turn == 0:
            im = message_panel("Our agent (near side, p1)  vs  foul-play (far side, p2)",
                               "each turn: solve the payoff matrix over sampled opponent teams, sample a move")
        elif turn in dec:
            rec, extra = dec[turn]
            im = decision_panel(turn, rec, extra, acts.get(turn, {}).get(OPP), acts.get(turn, {}).get(ME), names)
        else:                                   # no decision on our side: previous panel, greyed
            prev = max((t for t in dec if t < turn), default=None)
            im = grey(panel(prev, False)) if prev is not None else message_panel(f"Turn {turn}")
        cache[key] = im
        return im

    total = ft[-1] + 1.0
    nout = int(total * FPS)
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                          f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf",
                          "20", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    timeline, last_jpg, top = [], None, None
    for k in range(nout):
        t = k / FPS
        fi = max(0, bisect.bisect_right(ft, t) - 1)
        ti = bisect.bisect_right(tt, t) - 1
        turn, ended = (turns[ti][1], turns[ti][2]) if ti >= 0 else (0, False)
        if frames[fi][0] != last_jpg:
            last_jpg = frames[fi][0]
            top = Image.open(last_jpg).convert("RGB").resize((W, TOP_H))
        im = Image.new("RGB", (W, H), BG)
        im.paste(top, (0, 0))
        im.paste(panel(turn, ended), (0, TOP_H))
        p.stdin.write(im.tobytes())
        timeline.append((round(t, 3), turn, ended))
    p.stdin.close()
    p.wait()
    json.dump(timeline, open(os.path.splitext(out)[0] + ".timeline.json", "w"))
    print(f"{nout} frames, {nout / FPS:.1f}s -> {out}")
    if "--gif" in opts:
        s0, ln = float(opts.get("--gif-start", 0)), float(opts.get("--gif-len", 28))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(s0), "-t", str(ln), "-i", out, "-vf",
                        "fps=12,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
                        "[b][p]paletteuse=dither=bayer:bayer_scale=4", opts["--gif"]], check=True)
        print(f"gif {opts['--gif']}: {os.path.getsize(opts['--gif']) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
