#!/usr/bin/env python
"""Eval harness: manifest.jsonl + clips dir -> per-clip pred vs truth, WER/CER, split by mode.

Run with the vendor venv:
  vendor/venv/bin/python -I eval/eval.py --manifest M.jsonl --clips DIR [--checkpoint model.pth] [--tag NAME]
Manifest lines: {"id": "...", "text": "truth", "mode": "mouthed"|"spoken"}; clip is DIR/<id>.(mp4|webm|mov|mkv).
"""
import argparse, json, os, re, statistics, subprocess, sys, tempfile, time
from datetime import datetime
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
CHAPLIN = WS / "vendor" / "chaplin"
EXTS = (".mp4", ".webm", ".mov", ".mkv")


def norm(s):
    s = s.upper().replace("\u2019", "").replace("'", "")  # THAT'S -> THATS
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def find_clip(clips, cid):
    for e in EXTS:
        p = clips / (cid + e)
        if p.is_file():
            return p
    sys.exit(f"no clip for id {cid!r} in {clips}")


def to_25fps_mp4(src, dst, trim=0.0):
    """Constant 25 fps mp4 (webcam webm is variable frame rate); optionally drop `trim` s at both ends."""
    pre = []
    if trim:
        pts = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "packet=pts_time", "-select_streams", "v:0",
                              "-of", "csv=p=0", str(src)], capture_output=True, text=True).stdout.split()
        dur = max(float(x) for x in pts if x != "N/A")
        pre = ["-ss", str(trim), "-t", str(max(dur - 2 * trim, 0.5))]
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", *pre, "-i", str(src), "-an", "-vf", "fps=25", "-fps_mode", "cfr",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(dst)],
                       capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"ffmpeg failed on {src}: {r.stderr[-400:]}")


def needs_convert(p):
    if p.suffix.lower() != ".mp4":
        return True
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=r_frame_rate", "-of", "csv=p=0", str(p)], capture_output=True, text=True)
    return r.stdout.strip() != "25/1"


def make_config(base_cfg, checkpoint, tmpdir):
    """Absolute-path config; optionally swap the VSR checkpoint (model_path)."""
    from configparser import ConfigParser
    c = ConfigParser(); c.read(base_cfg)
    for k in ("model_path", "model_conf", "rnnlm", "rnnlm_conf"):
        c["model"][k] = str((CHAPLIN / c["model"][k]).resolve())
    if checkpoint:
        import torch
        sd = torch.load(checkpoint, map_location="cpu")
        base = torch.load(c["model"]["model_path"], map_location="cpu")
        if set(sd) != set(base):  # partial (changed-tensors-only) file from train/finetune.py: merge onto baseline
            unknown = set(sd) - set(base)
            if unknown:
                sys.exit(f"checkpoint has keys the model lacks: {sorted(unknown)[:3]}")
            base.update(sd); sd = base
        merged = Path(tmpdir) / "model.pth"
        torch.save(sd, merged)
        c["model"]["model_path"] = str(merged)
    out = Path(tmpdir) / "eval.ini"
    with open(out, "w") as f:
        c.write(f)
    return str(out)


def score(jiwer, refs, hyps):
    # jiwer rejects empty references; clips with empty truth are skipped upstream
    return {"wer": jiwer.wer(refs, hyps), "cer": jiwer.cer(refs, hyps), "n": len(refs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--checkpoint", help="VSR model.pth; default is the config's LRS3_V_WER19.1 baseline")
    ap.add_argument("--config", default=str(CHAPLIN / "configs/LRS3_V_WER19.1.ini"))
    ap.add_argument("--tag", help="model tag for the results filename (default: checkpoint dir/name or 'baseline')")
    ap.add_argument("--trim", type=float, default=0.0, help="seconds to cut from the start and end of every clip")
    ap.add_argument("--out-dir", default=str(WS / "runs"))
    a = ap.parse_args()

    import torch
    torch.zeros(1, device="cuda")  # init CUDA first: loading a checkpoint before it OOMs on GB10
    import jiwer
    rows = [json.loads(l) for l in open(a.manifest) if l.strip()]
    clips = Path(a.clips).resolve()
    tag = a.tag or (re.sub(r"\W+", "-", Path(a.checkpoint).resolve().parent.name + "-" + Path(a.checkpoint).stem)
                    if a.checkpoint else "baseline")
    if a.checkpoint and not Path(a.checkpoint).is_file():
        sys.exit(f"checkpoint not found: {a.checkpoint}")

    tmp = tempfile.TemporaryDirectory(prefix="ss_eval_")
    paths = {}
    for r in rows:  # convert up front so ffmpeg time is not counted as compute
        src = find_clip(clips, r["id"])
        if a.trim or needs_convert(src):
            dst = Path(tmp.name) / (r["id"] + ".mp4"); to_25fps_mp4(src, dst, a.trim); src = dst
        paths[r["id"]] = str(src.resolve())
    cfg = make_config(a.config, a.checkpoint, tmp.name)

    # --- load model once (same setup as runs/run_vsr.py) ---
    import torch
    torch.zeros(1, device="cuda")  # init CUDA before heavy imports (else OOM on GB10)
    sys.path.insert(0, str(CHAPLIN)); os.chdir(CHAPLIN)
    import av, numpy as np, torchvision.io
    def _rv(fn, pts_unit="sec"):  # torchvision >=0.22 dropped read_video
        with av.open(fn) as c:
            fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        return (torch.from_numpy(np.stack(fr)), None, None)
    torchvision.io.read_video = _rv
    from pipelines.pipeline import InferencePipeline
    t = time.time()
    pipe = InferencePipeline(cfg, device=torch.device("cuda:0"), detector="retinaface", face_track=True)
    load_s = time.time() - t
    print(f"model loaded in {load_s:.1f}s (tag={tag})", flush=True)

    results = []
    for i, r in enumerate(rows):
        torch.cuda.synchronize(); t = time.time()
        try:
            pred = pipe(paths[r["id"]]); err = None
        except Exception as e:  # a clip with no detectable face must not kill the run
            pred, err = "", repr(e)
        torch.cuda.synchronize(); dt = time.time() - t
        ref_n, hyp_n = norm(r["text"]), norm(pred)
        results.append({"id": r["id"], "mode": r.get("mode", "unknown"), "truth": r["text"], "pred": pred,
                        "truth_norm": ref_n, "pred_norm": hyp_n, "seconds": round(dt, 2), "error": err,
                        "wer": jiwer.wer(ref_n, hyp_n) if ref_n else None})
        print(f"[{i+1}/{len(rows)}] {r['id']} ({r.get('mode')}) {dt:.1f}s{' ERROR '+err if err else ''}\n"
              f"  truth: {ref_n}\n  pred:  {hyp_n}", flush=True)

    ok = [x for x in results if x["truth_norm"]]
    summary = {"overall": score(jiwer, [x["truth_norm"] for x in ok], [x["pred_norm"] for x in ok]) if ok else None,
               "by_mode": {}, "median_seconds_per_clip": statistics.median(x["seconds"] for x in results) if results else None,
               "errors": sum(1 for x in results if x["error"]), "model_load_seconds": round(load_s, 1)}
    for m in sorted({x["mode"] for x in ok}):
        g = [x for x in ok if x["mode"] == m]
        summary["by_mode"][m] = score(jiwer, [x["truth_norm"] for x in g], [x["pred_norm"] for x in g])

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(a.out_dir) / f"eval-{ts}-{tag}.json"
    out.write_text(json.dumps({"tag": tag, "checkpoint": a.checkpoint, "config": a.config, "manifest": str(Path(a.manifest).resolve()),
                               "timestamp": ts, "summary": summary, "clips": results}, indent=2))
    pct = lambda d: f"WER {d['wer']*100:.1f}%  CER {d['cer']*100:.1f}%  (n={d['n']})"
    print("\n== OVERALL ==", pct(summary["overall"]) if summary["overall"] else "no scorable clips")
    for m, d in summary["by_mode"].items():
        print(f"   {m}: {pct(d)}")
    print(f"median compute per clip: {summary['median_seconds_per_clip']}s   errors: {summary['errors']}")
    print("results:", out)


if __name__ == "__main__":
    main()
