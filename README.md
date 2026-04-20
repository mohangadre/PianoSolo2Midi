---
title: PianoSolo2MIDI
emoji: 🐢
colorFrom: purple
colorTo: red
sdk: docker
pinned: false
license: mit
short_description: This allows you to see what notes to play for any piano solo
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Faster inference (GPU)

- **Hardware:** In Space **Settings → Hardware**, pick a **GPU** option (e.g. ZeroGPU or paid GPU) so `torch.cuda.is_available()` is true. CPU-only is much slower than real time.
- **Code:** `detect_notes_onsets_frames` uses `@spaces.GPU` (package **`spaces`**) so Hugging Face can allocate a GPU for ByteDance O&F when supported.
- **Note:** [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu) is documented mainly for **Gradio** hosting; Docker/Streamlit Spaces still benefit from **GPU** tiers and CUDA-enabled PyTorch in the image if you add it.

