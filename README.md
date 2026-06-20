# Attention-Aware Toy

An experimental attention-perception prototype for Human-Robot Interaction (HRI). It combines computer vision with locally-run multimodal models to explore how a social robot could perceive visual engagement and decide when to start an interaction.

This is a *toy project* for experimentation and learning — not a production system. It performs no identity recognition and should not be used to infer intent, emotion, or mental state.

## Overview

A webcam feed runs through a face-landmark pipeline that scores attention every frame, classifies it into a discrete state, and tracks how long that state holds. When attention is sustained, the system asynchronously: captures the frame, describes the scene with a local vision-language model, and generates a short reply conditioned on both the scene and the current attention state — while the live overlay keeps rendering uninterrupted.

## How it works

**Face and landmark detection** — MediaPipe FaceLandmarker tracks up to 4 faces per frame; the largest is used as the active subject.

**Attention scoring** — a weighted score combines head pose (40%), iris direction (30%), nose position (15%), eye symmetry (10%), and face position (5%), derived from landmarks and the 3D facial transformation matrix.

**Attention state machine** — the score is classified each frame into `NO_FACE`, `FACE_DETECTED` (≤0.30), `LOOKING_BRIEFLY` (≤0.50), `DISTRACTED` (≤0.70), or `ATTENDING` (>0.70).

**Session gating** — sustained attention triggers one interaction per "session"; a new one is only allowed after attention drops below threshold and stays there for a release window, preventing repeated firing during a single gaze.

**Asynchronous pipeline** — perception and generation run on a background thread, so the webcam loop and overlay never block while a response is pending.

**Scene description** — a local vision-language model (MiniCPM-V, via Ollama) returns a constrained, structured description (person present, count, objects, gesture, scene) from a downscaled frame, explicitly barred from inferring identity, emotion, or intention.

**Contextual response generation** — a local language model (Qwen2.5, via Ollama) turns that description into a short reply, conditioned on the current attention state and gaze duration.

**Modular VLM/LLM boundary** — vision description and response generation are independent components, each with its own Ollama client, model, and timeout (`OLLAMA_VISION_MODEL` / `OLLAMA_MODEL`). This is intentional: the perception layer (landmarks, scoring, state machine, gating) has no dependency on either model, so the VLM or LLM can be swapped for a more specialized one without touching perception logic.

**Live overlay** — face box, eye contours, iris position, and 3D head-pose axes are drawn in real time, alongside state, score, pitch/yaw/roll, and gaze duration.

**Graceful degradation** — if the vision or language model fails, each stage falls back independently to a safe default instead of crashing. A GPU-verification failure is the one error that still halts execution.

**GPU correctness guard** — if an NVIDIA GPU is visible but Ollama reports the model loaded with zero VRAM, the request is rejected instead of silently falling back to CPU.

## Requirements

- Python 3.11+
- a webcam accessible via OpenCV (if you have multiple webcams, you may need to check which one to use and change the default index inside the code)
- Ollama running, with `openbmb/minicpm-v4.6` (vision) and `qwen2.5:1.5b` (text) pulled
> `openbmb/minicpm-v4.6` and `qwen2.5:1.5b` was chosen because it worked well in 4GB of RAM in a NVIDIA 3050ti. You can select other models if you have a better graphics card (or worse).

On startup the program checks the Ollama API and required models, and exits with a fix-it message before opening the webcam if anything is missing.

## Installation

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

ollama pull openbmb/minicpm-v4.6
ollama pull qwen2.5:1.5b
ollama serve
```

## Running

```bash
python -m src.text_app   # text-only path
python -m src.app        # full webcam pipeline
```

Press `q` or `Esc` to quit (`Ctrl+C` also works). On exit, the app asks Ollama to unload its models immediately, freeing standby RAM/VRAM.

## Ollama configuration

| Variable | Default | Description |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local API address |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Text response model |
| `OLLAMA_VISION_MODEL` | `openbmb/minicpm-v4.6` | Scene description model |
| `OLLAMA_TIMEOUT_SECONDS` | `30` | Text response timeout |
| `OLLAMA_VISION_TIMEOUT_SECONDS` | `60` | Scene description timeout |
| `ATTENTION_LOG_DIR` | `logs/` | Telemetry log directory |

Attention thresholds, gaze duration, release window, cooldown, camera index, and window name are tunable constants at the top of `src/app.py`.

## Testing

```bash
python -m unittest discover -s tests -v
```

Covers attention-state classification, the landmark detector, the Ollama client and preflight check, response generation, scene description, and the profiling layer.

## Profiling and telemetry

The pipeline logs high-precision, per-step measurements to `logs/performance-<date>-<pid>.jsonl` (one JSON object per line) without affecting functional decisions — wall/CPU time, tokens, VRAM, and an aggregated `interaction_report` per interaction (latency, CPU, RAM, GPU). CPU/RAM telemetry uses `psutil`; GPU telemetry (utilization, VRAM, power, temperature) uses `pynvml` and degrades to `null` fields on machines without an NVIDIA GPU. Resources are sampled every 200ms during inference so long model waits don't hide resource spikes. New functions can be instrumented with the `@profile_step("name")` decorator.

## Limitations and responsible use

- The attention index is a heuristic, not an objective measure of attention.
- Lighting, camera quality, occlusions, glasses, and individual traits affect estimates.
- Looking at the camera does not imply interest, consent, or intent to interact.
- Local model responses can contain errors.
- No per-user calibration or scientific evaluation yet.

Any evolution toward experiments with people should include consent, privacy safeguards, ethical review, and bias evaluation.

## Project status

Exploratory development — interfaces, thresholds, models, and architecture may change as experiments progress.
