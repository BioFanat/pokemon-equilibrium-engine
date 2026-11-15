"""Render demo.txt as a terminal-style reveal: frames/*.png, demo.gif, and (via ffmpeg) demo.mp4."""
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

D = os.path.dirname(os.path.abspath(__file__))
lines = open(os.path.join(D, "demo.txt")).read().rstrip("\n").split("\n")
W, H, PAD, SIZE, LH = 1280, 720, 36, 22, 30
BAR = 44
BG, BARBG, FG = (27, 36, 51), (20, 27, 39), (214, 221, 230)
MUTED, ACCENT, BLUE, GREEN = (110, 122, 140), (233, 150, 127), (125, 178, 222), (150, 214, 160)
font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", SIZE, index=0)
bold = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", SIZE, index=1)
ROWS = (H - BAR - 2 * PAD) // LH
STEP, HOLD, FPS = 0.11, 3.0, 30


def style(s):
    if s.startswith("──") or s.startswith("────"):
        return MUTED, font
    if "mixed strategy" in s:
        return ACCENT, bold
    if "payoff matrix" in s:
        return BLUE, font
    if s.startswith("Result"):
        return GREEN, bold
    if s.startswith("Equilibrium agent"):
        return FG, bold
    return FG, font


def frame(n):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, BAR], fill=BARBG)
    for i, c in enumerate([(237, 106, 94), (245, 191, 79), (98, 197, 84)]):
        d.ellipse([18 + i * 24, 15, 32 + i * 24, 29], fill=c)
    d.text((W // 2, BAR // 2), "demo_battle.py — equilibrium agent vs MCTS", font=font, fill=MUTED, anchor="mm")
    shown = lines[:n][-ROWS:]
    for i, s in enumerate(shown):
        col, f = style(s)
        d.text((PAD, BAR + PAD + i * LH), s, font=f, fill=col)
    return im


frames = [frame(n) for n in range(1, len(lines) + 1)]
fd = os.path.join(D, "frames"); os.makedirs(fd, exist_ok=True)
for f in os.listdir(fd):
    os.remove(os.path.join(fd, f))
for i, im in enumerate(frames):
    im.save(os.path.join(fd, f"f{i:04d}.png"))
durs = [int(STEP * 1000)] * (len(frames) - 1) + [int(HOLD * 1000)]
frames[0].save(os.path.join(D, "demo.gif"), save_all=True, append_images=frames[1:], duration=durs, loop=0, optimize=True)
# mp4: concat demuxer with per-frame durations, constant 30 fps output
with open(os.path.join(D, "list.txt"), "w") as fh:
    for i, du in enumerate(durs):
        fh.write(f"file 'frames/f{i:04d}.png'\nduration {du / 1000}\n")
    fh.write(f"file 'frames/f{len(frames) - 1:04d}.png'\n")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", "list.txt", "-vf",
                f"fps={FPS},format=yuv420p", "-c:v", "libx264", "-crf", "20", "-movflags", "+faststart", "demo.mp4"],
               cwd=D, check=True)
print(len(frames), "frames;", round(sum(durs) / 1000, 1), "s")
