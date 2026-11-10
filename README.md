# silent-speech

Experiment: type by mouthing words in front of a webcam, with no sound, using open-source lip reading. Work in progress, parked. Nothing here is a finished product.

## Where it got to

- Baseline: the Auto-AVSR visual-only model (via [Chaplin](https://github.com/amanvirparhar/chaplin)) runs on an NVIDIA GB10 (aarch64, CUDA) at about real time, 2.3 GB GPU memory. On a clean studio talking-head clip it gets about 15% of words wrong.
- On 15 to 16 clips of one person mouthing everyday sentences silently into a laptop webcam, it gets **about 51 to 53% of words wrong**. Two clips were exact matches, many were near misses, some were unrelated sentences.
- An LLM cleanup pass (Claude Haiku, one sentence at a time, and with the other sentences as context) did not help: 51.2% to 52.0%. When the lip model loses the content words, there is nothing to repair.
- Capture was ruled out as the cause: face detected on every frame, steady 25 fps, crops centred, and trimming clip ends made it worse. The likely causes are the domain gap (the model learned from people speaking aloud, and silent mouthing moves the lips much less) and a small, soft mouth crop from a laptop camera.
- Not done: fine-tuning on real data (the script is written and smoke tested on 2 clips only), the public-video data pipeline (written, never run end to end because YouTube bot-checked the machine), and any shrink to a size that runs on a laptop.

Honest read: open-vocabulary silent dictation you can trust without proofreading is not realistic with today's open models. A closed set of commands or phrases is a much easier target.

## What is in here

- `recorder/`: single-page webcam recorder (video only, no microphone) with 60 prompt sentences, plus a small receiver that stores each clip and a manifest line. Camera access needs HTTPS (for example `tailscale serve`).
- `eval/eval.py`: runs the model over a manifest of clips, prints per-clip prediction vs truth, WER and CER (jiwer), optional `--checkpoint` and `--trim`.
- `train/finetune.py`: partial fine-tune (front end, embedding, first encoder layer, as in Lipflow), bf16, early stop on a held-out set of videos.
- `data_prep/`: turns public talking-head videos into (96x96 mouth crops, Whisper transcript) pairs. Written for the author's own environment: `yt.py` and `build.py` point at local skills (yt-dlp cookies, Whisper-AT venv) you will need to replace.
- `runs/run_vsr.py`: minimal inference script for one clip.

## Setup (baseline, verified on aarch64 + CUDA 12.9)

```sh
mkdir vendor && cd vendor
git clone https://github.com/amanvirparhar/chaplin
(cd chaplin && bash setup.sh)        # downloads LRS3_V_WER19.1 (1 GB) and a subword LM from Hugging Face
uv venv --python 3.12 venv
uv pip install --python venv/bin/python torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
uv pip install --python venv/bin/python hydra-core opencv-python scipy scikit-image av six sentencepiece editdistance jiwer
# face detector (mediapipe has no aarch64 wheel with mp.solutions): ibug retinaface
mkdir ibug_src
git clone https://github.com/hhj1897/face_detection ibug_src/fd
git clone https://github.com/hhj1897/face_alignment ibug_src/fa
curl -sL -o ibug_src/fd/ibug/face_detection/retina_face/weights/Resnet50_Final.pth \
  https://github.com/hhj1897/face_detection/raw/master/ibug/face_detection/retina_face/weights/Resnet50_Final.pth
uv pip install --python venv/bin/python -e ibug_src/fd -e ibug_src/fa
cd ..
vendor/venv/bin/python -I runs/run_vsr.py $PWD/vendor/chaplin configs/LRS3_V_WER19.1.ini clip.mp4
```

Gotchas handled in `run_vsr.py`: initialise CUDA before importing the model (else out-of-memory on load), torchvision no longer has `io.read_video` (shimmed with PyAV), Chaplin needs its own directory as cwd.

## Licences and credit

Code here is MIT. It builds on [Auto-AVSR](https://github.com/mpc001/auto_avsr) (Apache-2.0), [Chaplin](https://github.com/amanvirparhar/chaplin) and [Lipflow](https://github.com/amywork777/lipflow) (MIT). The pretrained weights come from LRS3-trained models, so treat them as research and non-commercial use only. No recordings of any person are included.
