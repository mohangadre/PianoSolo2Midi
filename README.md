---
title: Piano Solo to MIDI
emoji: 🎹
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
---

# Piano Solo → MIDI

Streamlit app that transcribes piano audio to MIDI using **Onsets-and-Frames** (ByteDance `piano_transcription_inference`) with optional **CQT** fallback, optional **DeepFilterNet3** denoising, and optional **reference MIDI** evaluation via **mir_eval**.

[Hugging Face Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker) uses the **`Dockerfile`** in this repo (Python 3.10, **ffmpeg**, Streamlit on port **7860**).

## Requirements

- Python 3.9+ (see `requirements.txt`)
- **ffmpeg** on `PATH` (for MP3/M4A via pydub). The Docker image installs it via `apt`.

## Install (local)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional denoise (not in the default `requirements.txt` — larger install): `pip install deepfilternet`.

## Run (local)

```bash
streamlit run appv2.py
```

Open the URL shown in the terminal (e.g. `http://localhost:8501`).

## Docker (local test)

```bash
docker build -t piano-solo2midi .
docker run --rm -p 7860:7860 piano-solo2midi
```

Then open `http://localhost:7860`.

## Features

- Upload **WAV / MP3 / M4A** (max size per file is set in-app).
- Sidebar: transcription profile, optional polyphony cap, optional DeepFilterNet3.
- Download **JSON** metadata and **MIDI**.
- **Optional ground-truth MIDI**: upload `.mid` / `.midi` to compare transcription quality with **mir_eval** (precision / recall / F1).

## License

See project root for license if applicable.
