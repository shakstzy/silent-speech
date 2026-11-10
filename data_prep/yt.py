"""Serial, volume-safe yt-dlp for this workspace (rules: .claude/skills/youtube/references/reads.md).
Library: run(args) -> CompletedProcess; raises Blocked after a bot check or 429. guard() raises Blocked while data_prep/.blocked
is under 60 min old. CLI (used by find_videos.sh): `yt.py --guard` or `yt.py <yt-dlp args>`; exit 75 = blocked, stop the run.
Try order per call: viewer cookies (private copy), anonymous once, one cookie refresh per process and one more try with the jar.
Sleeps come from ~/.config/yt-dlp/config; never add your own."""
import fcntl, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILL = HERE.parents[2] / ".claude/skills/youtube"
JAR = SKILL / "data/viewer-cookies.txt"
BLOCKED = HERE / ".blocked"
COOLDOWN = 3600
BOT = ("not a bot", "Sign in to confirm")
RATE = ("HTTP Error 429", "Too Many Requests", "rate-limited")
GONE = ("Video unavailable", "video is not available", "Private video", "has been removed")
_refreshed = False


class Blocked(Exception):
    pass


def guard():
    if BLOCKED.exists():
        left = COOLDOWN - (time.time() - BLOCKED.stat().st_mtime)
        if left > 0:
            raise Blocked(f"blocked, wait {left/60:.0f} more min ({BLOCKED.read_text().strip()})")
        BLOCKED.unlink()


def _block(why):
    BLOCKED.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')} {why}\n")
    raise Blocked(f"YouTube {why}; wrote {BLOCKED}, no lookups for 60 min")


def _kind(r):
    t = (r.stderr or "") + (r.stdout or "")
    if r.returncode == 0:
        return "ok"
    if any(w in t for w in RATE):
        return "rate"
    if any(w in t for w in BOT):
        return "bot"
    if any(w in t for w in GONE):
        return "gone"
    return "other"


def _once(args, jar):
    tmp = None
    cmd = ["yt-dlp"]
    if jar and JAR.exists():
        fd, tmp = tempfile.mkstemp(suffix=".txt"); os.close(fd)
        shutil.copyfile(JAR, tmp)           # yt-dlp rewrites the jar it reads
        cmd += ["--cookies", tmp]
    try:
        return subprocess.run(cmd + list(args), capture_output=True, text=True)
    finally:
        if tmp:
            os.unlink(tmp)


def _refresh():
    global _refreshed
    _refreshed = True
    lock = SKILL / "data/viewer-cookies.lock"
    with open(lock, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        subprocess.run(["node", str(SKILL / "scripts/yt_cookies.js")], capture_output=True, text=True, timeout=180)


def run(args):
    guard()
    seen_bot = False
    r = None
    for step in ("cookies", "anonymous", "refreshed"):
        if step == "refreshed":
            if _refreshed:
                break
            _refresh()
        r = _once(args, jar=step != "anonymous")
        k = _kind(r)
        if k == "rate":
            _block("rate limit (429)")
        if k in ("ok", "other"):
            return r
        seen_bot = seen_bot or k == "bot"
        if step == "refreshed" and k == "bot":
            _block("bot check after cookies, anonymous and a refresh")
    if seen_bot:
        _block("bot check after cookies, anonymous and a refresh")
    return r  # unavailable on every try: a gap for this video, not a block


if __name__ == "__main__":
    try:
        if sys.argv[1:] == ["--guard"]:
            guard(); sys.exit(0)
        res = run(sys.argv[1:])
    except Blocked as e:
        print(e, file=sys.stderr); sys.exit(75)
    sys.stdout.write(res.stdout); sys.stderr.write(res.stderr); sys.exit(res.returncode)
