"""Visual demo: battle scene (left) + 4x4 slice of the real payoff matrix (right), one battle from battle.json.

Cells show estimated win chance ½(1 + tanh(M/160)) from the heuristic leaf. Rows = our 4 highest-probability
moves under RM+, columns = the opponent's 4 highest under its equilibrium strategy. Outputs viz.mp4 and viz.gif.
"""
import json
import math
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

D = os.path.dirname(os.path.abspath(__file__))
recs = json.load(open(os.path.join(D, "battle.json")))
W, H, FPS = 1280, 720, 15
BG, PANEL, ARENA = (24, 31, 44), (33, 42, 58), (44, 56, 78)
INK, MUTED, ACCENT, GOOD, BAD = (232, 236, 242), (140, 152, 170), (240, 150, 110), (47, 102, 144), (181, 70, 47)
HF = "/System/Library/Fonts/Helvetica.ttc"
f = lambda s, b=False: ImageFont.truetype(HF, s, index=1 if b else 0)
F14, F16, F18, F20, F22B, F26B, F30B = f(14), f(16), f(18), f(20), f(22, True), f(26, True), f(30, True)
SPR = {}


def sprite(name, back):
    key = (name, back)
    if key not in SPR:
        p = os.path.join(D, "sprites", f"{'gen5-back' if back else 'gen5'}_{name}.png")
        im = Image.open(p).convert("RGBA") if os.path.exists(p) else Image.new("RGBA", (96, 96), (0, 0, 0, 0))
        SPR[key] = im.resize((96 * 2 if not back else int(96 * 2.4),) * 2, Image.NEAREST)
    return SPR[key]


def win(m):
    return 0.5 * (1 + math.tanh(m / 160))


def mix(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def cell_color(p):          # diverging: orange (bad for us) -> panel grey (50%) -> blue (good for us)
    mid = (70, 78, 94)
    return mix(mid, GOOD, min(1, (p - 0.5) * 2.2)) if p >= 0.5 else mix(mid, BAD, min(1, (0.5 - p) * 2.2))


def hp_color(h):
    return (98, 197, 84) if h > 0.5 else (245, 191, 79) if h > 0.2 else (237, 106, 94)


def nice(s):
    """'switch → x' -> 'switch: x'; 'move-tera' -> ('move', True)."""
    s = s.replace("switch → ", "switch: ")
    return (s[:-5], True) if s.endswith("-tera") else (s, False)


def short(s, n=13):
    return s if len(s) <= n else s[: n - 1] + "…"


def hp_card(d, x, y, name, hp, alive, team_hps):
    d.rounded_rectangle([x, y, x + 260, y + 74], 10, fill=PANEL)
    d.text((x + 14, y + 10), name, font=F20, fill=INK)
    d.rounded_rectangle([x + 14, y + 40, x + 196, y + 52], 6, fill=(20, 26, 36))
    if hp > 0:
        d.rounded_rectangle([x + 14, y + 40, x + 14 + max(8, int(182 * hp)), y + 52], 6, fill=hp_color(hp))
    d.text((x + 204, y + 36), f"{round(100 * hp)}%", font=F16, fill=MUTED)
    for i, h in enumerate(team_hps):                       # team pips
        c = (x + 20 + i * 16, y + 62)
        d.ellipse([c[0] - 4, c[1] - 4, c[0] + 4, c[1] + 4], fill=INK if h > 0 else (70, 78, 94))


def scene(im, d, us, them, shake_us=0, shake_them=0):
    d.rounded_rectangle([24, 24, 640, 600], 18, fill=ARENA)
    d.ellipse([360, 250, 600, 300], fill=(56, 70, 96))      # opponent platform
    d.ellipse([60, 500, 340, 560], fill=(56, 70, 96))       # our platform
    if them["hp"] > 0 or them.get("show", True):
        s = sprite(them["species"], False)
        im.alpha_composite(s, (480 - s.width // 2 + shake_them, 285 - s.height + 20))
    s = sprite(us["species"], True)
    im.alpha_composite(s, (200 - s.width // 2 + shake_us, 540 - s.height + 40))
    hp_card(d, 44, 44, them["species"], them["hp"], them["alive"], them["hps"])
    hp_card(d, 360, 500, us["species"], us["hp"], us["alive"], us["hps"])
    d.text((44, 560 + 6), "", font=F14, fill=MUTED)


def matrix(d, r, highlight):
    X0, Y0 = 670, 24
    d.rounded_rectangle([X0, Y0, W - 24, 600], 18, fill=PANEL)
    d.text((X0 + 24, Y0 + 18), "Payoff matrix slice", font=F26B, fill=INK)
    d.text((X0 + 24, Y0 + 54), "cell = our estimated win chance if both moves are played", font=F16, fill=MUTED)
    if "M" not in r:
        d.text((X0 + 24, Y0 + 260), "Forced move: nothing to decide this turn", font=F22B, fill=MUTED)
        return
    x, y, M, s1, s2 = r["x"], r["y"], r["M"], r["s1"], r["s2"]
    rows = sorted(range(len(s1)), key=lambda i: -x[i])[:4]
    cols = sorted(range(len(s2)), key=lambda j: -y[j])[:4]
    cx, cy, cw, ch = X0 + 210, Y0 + 150, 88, 86
    d.text((cx, Y0 + 80), "opponent’s moves (equilibrium prob.)", font=F14, fill=MUTED)
    for k, j in enumerate(cols):
        played = highlight and s2[j] == r["m2"]
        nm, tera = nice(s2[j])
        d.text((cx + k * cw + cw // 2, cy - 46), short(nm, 12), font=F14, fill=ACCENT if played else INK, anchor="mm")
        if tera:
            d.text((cx + k * cw + cw // 2, cy - 29), "tera", font=F14, fill=ACCENT, anchor="mm")
        d.text((cx + k * cw + cw // 2, cy - 12), f"{y[j]:.2f}", font=F14, fill=MUTED, anchor="mm")
    d.text((X0 + 24, cy - 22), "our moves (RM+ prob.)", font=F14, fill=MUTED)
    for kr, i in enumerate(rows):
        yy = cy + kr * ch
        played = highlight and s1[i] == r["m1"]
        nm, tera = nice(s1[i])
        d.text((X0 + 24, yy + 22), short(nm + (" (tera)" if tera else ""), 17), font=F18, fill=ACCENT if played else INK)
        d.rounded_rectangle([X0 + 24, yy + 50, X0 + 24 + 130, yy + 60], 5, fill=(20, 26, 36))
        if x[i] > 0.005:
            d.rounded_rectangle([X0 + 24, yy + 50, X0 + 24 + max(6, int(130 * x[i])), yy + 60], 5, fill=ACCENT)
        d.text((X0 + 162, yy + 45), f"{x[i]:.2f}", font=F14, fill=MUTED)
        for kc, j in enumerate(cols):
            p = win(M[i][j])
            box = [cx + kc * cw + 3, yy + 3, cx + (kc + 1) * cw - 3, yy + ch - 3]
            d.rounded_rectangle(box, 8, fill=cell_color(p))
            d.text(((box[0] + box[2]) // 2, (box[1] + box[3]) // 2), f"{round(100 * p)}%", font=F20, fill=INK, anchor="mm")
            if highlight and s1[i] == r["m1"] and s2[j] == r["m2"]:
                d.rounded_rectangle(box, 8, outline=(255, 255, 255), width=4)
    if highlight and r["m2"] not in [s2[j] for j in cols]:
        d.text((X0 + 24, 560), f"opponent played {txt(r['m2'])} (outside the slice)", font=F14, fill=MUTED)
    if highlight and r["m1"] not in [s1[i] for i in rows]:
        d.text((X0 + 24, 578), f"we played {txt(r['m1'])} (outside the slice)", font=F14, fill=MUTED)


def caption(d, text, sub=""):
    d.rounded_rectangle([24, 620, W - 24, H - 24], 14, fill=PANEL)
    d.text((48, 640), text, font=F22B, fill=INK)
    if sub:
        d.text((48, 672), sub, font=F16, fill=MUTED)


def base():
    im = Image.new("RGBA", (W, H), BG)
    return im, ImageDraw.Draw(im)


def txt(m):
    nm, tera = nice(m)
    return nm + (" (tera)" if tera else "")


def turn_text(r):
    if "M" not in r:
        return f"Turn {r['turn']}: forced move", f"we play {txt(r['m1'])}  ·  MCTS plays {txt(r['m2'])}"
    top = sorted(range(len(r["s1"])), key=lambda i: -r["x"][i])
    mixs = " / ".join(f"{txt(r['s1'][i])} {round(100 * r['x'][i])}%" for i in top[:2] if r["x"][i] >= 0.01)
    return f"Turn {r['turn']}: RM+ mixes {mixs}", f"we play {txt(r['m1'])}  ·  MCTS plays {txt(r['m2'])}"


frames = []
def emit(im, secs):
    frames.extend([im.convert("RGB")] * max(1, round(secs * FPS)))


# intro
im, d = base()
scene(im, d, recs[0]["us"], recs[0]["them"])
d.rounded_rectangle([670, 24, W - 24, 600], 18, fill=PANEL)
d.text((700, 220), "Each turn:", font=F30B, fill=INK)
for k, t in enumerate(["1. build the full payoff matrix", "2. solve it with RM+ (1,000 iterations)", "3. sample our move from the mix"]):
    d.text((700, 280 + 44 * k), t, font=F22B if False else F20, fill=INK)
caption(d, "Equilibrium agent (us, bottom)  vs  MCTS, 2,000 iterations (top)", "one battle, played turn by turn")
emit(im, 2.5)

for k, r in enumerate(recs[:-1]):
    nxt = recs[k + 1]
    t1, t2 = turn_text(r)
    for hl, secs in ((False, 0.35), (True, 0.7)):
        im, d = base(); scene(im, d, r["us"], r["them"]); matrix(d, r, hl)
        caption(d, t1, t2 if hl else "solving…")
        emit(im, secs)
    # HP animation: current actives' HP moves to their value after the turn
    iu, it = r["us"]["team"].index(r["us"]["species"]), r["them"]["team"].index(r["them"]["species"])
    hu1, ht1 = nxt["us"]["hps"][iu], nxt["them"]["hps"][it]
    hu0, ht0 = r["us"]["hp"], r["them"]["hp"]
    n = 5
    for s in range(1, n + 1):
        a = s / n
        us = dict(r["us"], hp=hu0 + (hu1 - hu0) * a); th = dict(r["them"], hp=ht0 + (ht1 - ht0) * a)
        sh = 6 if s % 2 else -6
        im, d = base()
        scene(im, d, us, th, shake_us=sh if hu1 < hu0 - 0.01 else 0, shake_them=sh if ht1 < ht0 - 0.01 else 0)
        matrix(d, r, True); caption(d, t1, t2)
        emit(im, 1 / FPS)

last = recs[-1]
im, d = base(); scene(im, d, last["us"], dict(last["them"], show=False))
d.rounded_rectangle([670, 24, W - 24, 600], 18, fill=PANEL)
res = {"win": "We win", "loss": "MCTS wins", "draw": "Draw"}[last["result"]]
d.text((700, 260), res, font=f(64, True), fill=ACCENT)
d.text((700, 350), f"in {last['turn'] - 1} turns", font=F26B, fill=INK)
caption(d, f"{res} in {last['turn'] - 1} turns", "one illustrative battle; win rates in the paper come from the full experiments")
emit(im, 3.0)

print(len(frames), "frames,", round(len(frames) / FPS, 1), "s")
p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                      "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                      "-movflags", "+faststart", os.path.join(D, "viz.mp4")], stdin=subprocess.PIPE)
for fr in frames:
    p.stdin.write(fr.tobytes())
p.stdin.close(); p.wait()
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "viz.mp4", "-vf",
                "fps=10,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse",
                "viz.gif"], cwd=D, check=True)
frames[int(len(frames) * 0.55)].save(os.path.join(D, "still.png"))
