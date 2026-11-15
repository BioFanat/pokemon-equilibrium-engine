"""Record a saved Showdown replay (replays/<tag>.html) to MP4 with headless Chrome + the DevTools screencast.

  ~/sandbox/bot-venv/bin/python ~/sandbox/ourbot/record_replay.py replays/<tag>.html [seconds] [speed]
speed: Hyperfast | Fast | Normal (default Fast). Writes replays/<tag>.mp4. Uses a throwaway Chrome profile;
needs ffmpeg. The replay page loads Showdown's viewer script from play.pokemonshowdown.com.
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
    frames = []
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
        while time.time() - t0 < seconds:
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.0))
            except asyncio.TimeoutError:
                continue
            if msg.get("method") == "Page.screencastFrame":
                p = msg["params"]
                path = os.path.join(outdir, f"f{len(frames):05d}.jpg")
                open(path, "wb").write(base64.b64decode(p["data"]))
                frames.append((path, time.time() - t0))
                await send("Page.screencastFrameAck", sessionId=p["sessionId"])
    return frames


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
        frames = asyncio.run(record("file://" + page, seconds, work))
    finally:
        chrome.terminate()
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
