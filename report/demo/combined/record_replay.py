"""Record a saved Showdown replay (replays/<tag>.html) to MP4 with headless Chrome + the DevTools screencast.

  ~/sandbox/bot-venv/bin/python ~/sandbox/ourbot/record_replay.py replays/<tag>.html [seconds] [speed]
speed: Hyperfast | Fast | Normal (default Fast). Writes replays/<tag>.mp4. Uses a throwaway Chrome profile;
needs ffmpeg. The replay page loads Showdown's viewer script from play.pokemonshowdown.com.
Demo extension: with a 4th argument OUTDIR, frames are kept in OUTDIR with frames.json ([path, t] per frame) and
turns.json ([t, turn, ended] sampled every ~200 ms from the viewer's global Replays.battle), and recording stops
3 s after the battle has ended (seconds is then only a cap).
"""
import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import websockets

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 9333
TURN_JS = "(window.Replays && Replays.battle) ? [Replays.battle.turn, !!Replays.battle.ended] : null"
AUTOPLAY = """<script>
window.addEventListener('load',function(){setTimeout(function(){
 var q=function(re){return [].slice.call(document.querySelectorAll('button')).find(function(x){return re.test(x.textContent.trim())})};
 var b=q(/sound off/i); if(b)b.click();
 setTimeout(function(){var f=q(new RegExp('^%s$')); if(f)f.click();},300);
},1500);});
</script>"""


async def record(url, seconds, outdir):
    ver = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/version"))
    tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
    page = next(t for t in tabs if t["type"] == "page")
    frames, turns, polls = [], [], {}
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=50_000_000) as ws:
        n = 0

        async def send(method, **params):
            nonlocal n
            n += 1
            await ws.send(json.dumps(dict(id=n, method=method, params=params)))

        await send("Emulation.setDeviceMetricsOverride", width=1180, height=380, deviceScaleFactor=1, mobile=False)
        await send("Page.enable")
        await send("Page.navigate", url=url)
        await send("Page.startScreencast", format="jpeg", quality=85, everyNthFrame=1)
        t0 = time.time()
        last_poll, ended_at = 0.0, None
        while time.time() - t0 < seconds:
            now = time.time()
            if now - last_poll >= 0.2:                     # poll the viewer's turn counter
                last_poll = now
                await send("Runtime.evaluate", returnByValue=True, expression=TURN_JS)
                polls[n] = now - t0
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.05))
            except asyncio.TimeoutError:
                continue
            if msg.get("id") in polls:
                val = msg.get("result", {}).get("result", {}).get("value")
                if val:
                    turns.append([polls.pop(msg["id"]), val[0], val[1]])
                    if val[1] and ended_at is None:
                        ended_at = time.time()
                if ended_at is not None and time.time() - ended_at > 3.0:
                    break
                continue
            if msg.get("method") == "Page.screencastFrame":
                p = msg["params"]
                path = os.path.join(outdir, f"f{len(frames):05d}.jpg")
                open(path, "wb").write(base64.b64decode(p["data"]))
                frames.append((path, time.time() - t0))
                await send("Page.screencastFrameAck", sessionId=p["sessionId"])
    return frames, turns


def main():
    src = os.path.abspath(sys.argv[1])
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 150
    speed = sys.argv[3] if len(sys.argv) > 3 else "Fast"
    work = tempfile.mkdtemp(prefix="replayrec_")
    page = os.path.join(work, "page.html")
    open(page, "w").write(open(src).read() + AUTOPLAY % speed)
    chrome = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", f"--remote-debugging-port={PORT}",
                               f"--user-data-dir={work}/profile", "--window-size=1180,380", "about:blank"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/version"); break
            except Exception:
                time.sleep(0.2)
        frames, turns = asyncio.run(record("file://" + page, seconds, work))
    finally:
        chrome.terminate()
    if len(sys.argv) > 4:                                  # demo: keep frames + turn timing for the compositor
        outdir = os.path.abspath(sys.argv[4])
        os.makedirs(outdir, exist_ok=True)
        kept = []
        for path, t in frames:
            dst = os.path.join(outdir, os.path.basename(path))
            shutil.move(path, dst)
            kept.append([dst, round(t, 3)])
        json.dump(kept, open(os.path.join(outdir, "frames.json"), "w"))
        json.dump(turns, open(os.path.join(outdir, "turns.json"), "w"))
        shutil.rmtree(work, ignore_errors=True)
        print(f"{len(frames)} frames over {frames[-1][1]:.0f}s, {len(turns)} turn samples, last turn "
              f"{turns[-1] if turns else None} -> {outdir}")
        return
    # concat demuxer with each frame's real duration -> constant 30 fps MP4 (crop to even size)
    lst = os.path.join(work, "list.txt")
    with open(lst, "w") as f:
        for k, (path, t) in enumerate(frames):
            nxt = frames[k + 1][1] if k + 1 < len(frames) else t + 2.0
            f.write(f"file '{path}'\nduration {max(nxt - t, 0.001):.3f}\n")
        f.write(f"file '{frames[-1][0]}'\n")
    out = os.path.splitext(src)[0] + ".mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-vf",
                    "fps=30,crop=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p", "-c:v", "libx264", "-crf", "20",
                    "-movflags", "+faststart", out], check=True)
    shutil.rmtree(work, ignore_errors=True)
    print(f"{len(frames)} frames over {frames[-1][1]:.0f}s -> {out}")


if __name__ == "__main__":
    main()
