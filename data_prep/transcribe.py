"""Word-level Whisper-AT transcription. Run with the whisper-at skill's venv:
   .claude/skills/whisper-at/data/.venv/bin/python transcribe.py OUTDIR file1 file2 ...
Writes OUTDIR/<stem>.json = list of {word,start,end,probability}. Model loads once for all files."""
import sys, json, os, subprocess, tempfile
from pathlib import Path
SKILL = Path(__file__).resolve().parents[3] / ".claude/skills/whisper-at/data"
os.environ["XDG_CACHE_HOME"] = str(SKILL / "state/cache")
import whisper_at

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
model = None
for f in sys.argv[2:]:
    dst = out / (Path(f).stem + ".json")
    if dst.exists():
        continue
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "a.wav")
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f, "-vn", "-ac", "1", "-ar", "16000", wav])
        if r.returncode:
            print("ffmpeg fail", f, file=sys.stderr); continue
        if model is None:
            model = whisper_at.load_model("medium", device="cuda", download_root=str(SKILL / "state/cache"))
        res = model.transcribe(wav, word_timestamps=True, language="en", condition_on_previous_text=False)
    words = [dict(word=w["word"], start=w["start"], end=w["end"], probability=w["probability"])
             for s in res["segments"] for w in s.get("words", [])]
    dst.write_text(json.dumps(words))
    print("transcribed", f, len(words), flush=True)
