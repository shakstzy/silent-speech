"""build.py <url-list-file>: public talking-head videos -> data/pretrain/{manifest.jsonl,crops/<id>.npy}.
Run with vendor/venv/bin/python. Scratch (downloads, clips, transcripts) in $SS_SCRATCH (default /tmp/silent-speech-prep).
Resumes: a video with data/pretrain/state/<vid>.json is skipped, downloads and transcripts are cached, manifest ids are not duplicated.
yt-dlp runs serially through yt.py; a bot check or 429 stops the run (exit 2) and writes data_prep/.blocked (no lookups for 60 min). Contract: manifest {id,text,src,dur,mode:"spoken"},
crops uint8 [T,96,96] @25fps (crop.video_to_crops path)."""
import sys, os, re, json, subprocess, tempfile, hashlib, collections, time
from pathlib import Path
import numpy as np
import yt

HERE = Path(__file__).resolve().parent
WS = HERE.parent
PRE = WS / "data/pretrain"
SCR = Path(os.environ.get("SS_SCRATCH", Path(tempfile.gettempdir()) / "silent-speech-prep"))
WHISPER_PY = WS.parents[1] / ".claude/skills/whisper-at/data/.venv/bin/python"

MIN_S, MAX_S = 2.0, 8.0
PAUSE = 0.35            # gap that ends a unit
MIN_MEAN_P, MIN_WORD_P = 0.85, 0.4
MIN_DET, MAX_MULTI = 0.95, 0.02   # share of frames with a face / with >=2 faces
MAX_YAW, MIN_FACE_H = 0.12, 90.0  # |nose offset| fraction of jaw width; face box height in px (480p source)
YT = re.compile(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})")


def vid_of(url):
    m = YT.search(url)
    return m.group(1) if m else "u" + hashlib.md5(url.encode()).hexdigest()[:10]


def download(url):
    v = vid_of(url)
    d = SCR / "dl"; d.mkdir(parents=True, exist_ok=True)
    got = [p for p in d.glob(v + ".*") if p.suffix in (".mkv", ".mp4", ".webm")]
    if got:
        return v, got[0]
    r = yt.run(["-f", "bv*[height<=480]+ba/b[height<=480]", "--merge-output-format", "mkv",
                        "--no-playlist", "-q", "--no-warnings", "-o", str(d / (v + ".%(ext)s")), url])
    got = [p for p in d.glob(v + ".*") if p.suffix in (".mkv", ".mp4", ".webm")]
    if not got:
        print("DOWNLOAD FAIL", url, r.stderr.strip()[-200:], file=sys.stderr)
        return v, None
    return v, got[0]


def duration(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                                capture_output=True, text=True).stdout.strip() or 0)


def clean(words):
    t = " ".join(w["word"].strip() for w in words).upper().replace("-", " ")
    if re.search(r"[0-9]", t):
        return None
    t = re.sub(r"[^A-Z' ]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t or None


def split_long(ws):
    if ws[-1]["end"] - ws[0]["start"] <= MAX_S or len(ws) < 2:
        return [ws]
    k = max(range(1, len(ws)), key=lambda i: ws[i]["start"] - ws[i - 1]["end"])
    return split_long(ws[:k]) + split_long(ws[k:])


def segments(words):
    units, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1]["start"] - w["end"] if i + 1 < len(words) else 9
        if re.search(r"[.?!]['\"]?$", w["word"].strip()) or nxt >= PAUSE:
            units.append(cur); cur = []
    if cur: units.append(cur)
    units = [u for x in units for u in split_long(x)]
    out, cur = [], []
    def flush():
        nonlocal cur
        if cur: out.append(cur); cur = []
    for u in units:
        if cur and (u[-1]["end"] - cur[0]["start"] > MAX_S or u[0]["start"] - cur[-1]["end"] > 1.0):
            flush()
        cur += u
        if cur[-1]["end"] - cur[0]["start"] >= 4 and re.search(r"[.?!]['\"]?$", cur[-1]["word"].strip()):
            flush()
    flush()
    return out


def process_video(v, path, words, crop):
    st = collections.Counter(); kept = []
    dur_total = duration(path)
    for seg in segments(words):
        s, e = seg[0]["start"], seg[-1]["end"]
        d = e - s
        st["segments"] += 1
        if d < MIN_S: st["drop_too_short"] += 1; continue
        ps = [w["probability"] for w in seg]
        if np.mean(ps) < MIN_MEAN_P or min(ps) < MIN_WORD_P: st["drop_low_whisper_conf"] += 1; continue
        text = clean(seg)
        if not text: st["drop_digits_or_empty"] += 1; continue
        t0 = max(0.0, s - 0.1); t1 = min(dur_total, e + 0.1)
        clip = SCR / "clips" / f"{v}_{int(s*1000):07d}.mp4"; clip.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.3f}", "-t", f"{t1-t0:.3f}", "-i", str(path), "-an",
                        "-r", "25", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16", str(clip)], check=True)
        try:
            a = crop.analyze(str(clip))
            n = len(a["nfaces"])
            if n < 25 * MIN_S: st["drop_clip_short"] += 1; continue
            det = (a["nfaces"] >= 1).mean()
            if det < MIN_DET: st["drop_face_not_detected"] += 1; continue
            if (a["nfaces"] >= 2).mean() > MAX_MULTI: st["drop_multiple_faces"] += 1; continue
            ok = a["nfaces"] >= 1
            if np.median(np.abs(a["yaw"][ok])) > MAX_YAW: st["drop_not_frontal"] += 1; continue
            if np.median(a["bbox_h"][ok]) < MIN_FACE_H: st["drop_face_small"] += 1; continue
            c = crop.crops_from_analysis(a)
        except Exception as ex:
            st["drop_error_" + type(ex).__name__] += 1; continue
        finally:
            clip.unlink(missing_ok=True)
        sid = f"{v}_{int(s*1000):07d}"
        np.save(PRE / "crops" / f"{sid}.npy", c)
        kept.append(dict(id=sid, text=text, src=f"https://www.youtube.com/watch?v={v}&t={int(s)}", dur=round(len(c) / 25, 2), mode="spoken"))
        st["kept"] += 1; st["kept_s"] += len(c) / 25
    return kept, dict(st), dur_total


def main():
    urls = [u.strip() for u in open(sys.argv[1]) if u.strip() and not u.startswith("#")]
    (PRE / "crops").mkdir(parents=True, exist_ok=True); (PRE / "state").mkdir(exist_ok=True)
    yt.guard()
    seen = {json.loads(l)["id"] for l in open(PRE / "manifest.jsonl")} if (PRE / "manifest.jsonl").exists() else set()
    todo = [u for u in urls if not (PRE / "state" / f"{vid_of(u)}.json").exists()]
    print(f"{len(urls)} urls, {len(todo)} to do", flush=True)
    t0 = time.time()
    dl, blocked = {}, None
    for u in todo:
        try:
            v, p = download(u)
        except yt.Blocked as e:
            blocked = e; print("STOP:", e, file=sys.stderr); break
        if p: dl[v] = p
    print(f"downloaded {len(dl)}/{len(todo)} in {time.time()-t0:.0f}s", flush=True)
    t1 = time.time()
    tr = SCR / "tr"
    need = [str(p) for v, p in dl.items() if not (tr / f"{v}.json").exists()]
    if need:
        subprocess.run([str(WHISPER_PY), str(HERE / "transcribe.py"), str(tr)] + need, check=False)
    print(f"transcribed in {time.time()-t1:.0f}s", flush=True)
    sys.path.insert(0, str(HERE)); import crop
    t2 = time.time()
    for v, p in dl.items():
        if not (tr / f"{v}.json").exists():
            print("no transcript", v); continue
        words = json.load(open(tr / f"{v}.json"))
        kept, st, dtot = process_video(v, p, words, crop)
        with open(PRE / "manifest.jsonl", "a") as f:
            for k in kept:
                if k["id"] not in seen: f.write(json.dumps(k) + "\n")
        st["downloaded_s"] = dtot
        (PRE / "state" / f"{v}.json").write_text(json.dumps(st))
        print(v, f"{dtot/60:.1f}min", {k: (round(x, 1) if isinstance(x, float) else x) for k, x in st.items()}, flush=True)
    print(f"cropped in {time.time()-t2:.0f}s")
    if blocked: print("run stopped early; rerun after the cooldown to resume"); sys.exit(2)
    # stats over every video in the list
    tot = collections.Counter()
    for u in urls:
        f = PRE / "state" / f"{vid_of(u)}.json"
        if f.exists():
            for k, x in json.load(open(f)).items(): tot[k] += x
    print("\n=== YIELD ===")
    print(f"videos done {sum((PRE/'state'/(vid_of(u)+'.json')).exists() for u in urls)}/{len(urls)}  "
          f"downloaded {tot['downloaded_s']/3600:.2f} h  kept {tot['kept_s']/3600:.2f} h ({100*tot['kept_s']/max(tot['downloaded_s'],1):.1f}%)  "
          f"segments kept {tot['kept']}/{tot['segments']}")
    for k, x in sorted(tot.items()):
        if k.startswith("drop"): print(f"  {k}: {x}")


if __name__ == "__main__":
    main()
