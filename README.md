# Piano Solo → MIDI

Streamlit app that transcribes piano audio to MIDI using **Onsets-and-Frames** (ByteDance `piano_transcription_inference`) with optional **CQT** fallback, optional **DeepFilterNet3** denoising, and optional **reference MIDI** evaluation via **mir_eval**.

## Requirements

- Python 3.9+ (see `requirements.txt`)
- **ffmpeg** on `PATH` (for MP3/M4A via pydub)

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional denoise: `pip install deepfilternet` (already listed in requirements).

## Run

```bash
streamlit run appv2.py
```

Open the URL shown in the terminal (e.g. `http://localhost:8501`).

## Features

- Upload **WAV / MP3 / M4A** (max size per file is set in-app).
- Sidebar: transcription profile, optional polyphony cap, optional DeepFilterNet3.
- Download **JSON** metadata and **MIDI**.
- **Optional ground-truth MIDI**: upload ` .mid` / `.midi` to compare transcription quality with **mir_eval** (precision / recall / F1).

## License

See project root for license if applicable.
