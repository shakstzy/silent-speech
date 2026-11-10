"""Mouth cropping identical to the model's inference path (chaplin / Auto-AVSR, retinaface + FAN).

    from crop import video_to_crops
    crops = video_to_crops("clip.mp4")   # uint8 [T,96,96] grayscale, 25 fps

Run with vendor/venv/bin/python. Same detector, same landmark smoothing, same affine
alignment + 96x96 cut as InferencePipeline (detector="retinaface", face_track=True).
The model's own transform (center crop 88, normalise) is applied later, at train/infer time.
"""
import os, subprocess, tempfile
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
CHAPLIN = os.path.abspath(os.path.join(HERE, "..", "vendor", "chaplin"))
FPS = 25
_det = None
_vp = None


def _load():
    global _det, _vp
    if _det is None:
        torch.zeros(1, device="cuda")  # init CUDA before heavy imports (GB10 OOM otherwise)
        import sys
        if CHAPLIN not in sys.path:
            sys.path.insert(0, CHAPLIN)
        from pipelines.detectors.retinaface.detector import LandmarksDetector
        from pipelines.detectors.retinaface.video_process import VideoProcess
        _det = LandmarksDetector(device="cuda:0")
        _vp = VideoProcess(convert_gray=True)
    return _det, _vp


def read_frames(path):
    """RGB uint8 [T,H,W,3] at 25 fps (resampled with ffmpeg if the source is not 25 fps)."""
    import av
    with av.open(path) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate or 0)
    tmp = None
    if abs(fps - FPS) > 0.01:
        tmp = tempfile.mktemp(suffix=".mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-an", "-r", str(FPS),
                        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16", tmp], check=True)
        path = tmp
    try:
        with av.open(path) as c:
            return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])
    finally:
        if tmp:
            os.remove(tmp)


def analyze(path):
    """Per-frame detection. Returns dict(frames, landmarks, nfaces, bbox_h, yaw). Face choice = largest box,
    exactly as the inference detector does."""
    det, _ = _load()
    frames = read_frames(path)
    lms, nfaces, bbh, yaw = [], [], [], []
    for fr in frames:
        faces = det.face_detector(fr, rgb=False)
        pts, _ = det.landmark_detector(fr, faces, rgb=True)
        nfaces.append(len(faces))
        if len(faces) == 0:
            lms.append(None); bbh.append(0.0); yaw.append(np.nan); continue
        k = max(range(len(faces)), key=lambda i: (faces[i][2] - faces[i][0]) + (faces[i][3] - faces[i][1]))
        p = pts[k]
        lms.append(p)
        bbh.append(float(faces[k][3] - faces[k][1]))
        yaw.append(((p[30, 0] - p[0, 0]) / max(p[16, 0] - p[0, 0], 1e-3)) - 0.5)  # 0 = frontal
    return dict(frames=frames, landmarks=lms, nfaces=np.array(nfaces), bbox_h=np.array(bbh), yaw=np.array(yaw))


def crops_from_analysis(a):
    _, vp = _load()
    lms = list(a["landmarks"])
    out = vp(a["frames"], lms)
    if out is None:
        raise ValueError("no face landmarks")
    return np.asarray(out, dtype=np.uint8)


def video_to_crops(path):
    return crops_from_analysis(analyze(path))
