# Attention-Aware Toy

Experimental Human-Robot Interaction prototype driven by webcam attention,
apparent facial-expression cues, mocked interaction context, and a WIP Moktak
audio feedback module.

This is not a production system. It does not identify people and must not be
used to infer a person's real emotion, intent, consent, or mental state.
Facial-expression output is only an apparent-expression heuristic from visible
MediaPipe blendshapes.

## Current Pipeline

The main app runs camera capture, attention detection, and event processing in
separate threads. Expression detection and response generation are event-driven:

```text
camera frame
  -> continuous attention scoring
  -> sustained attention gate
  -> snapshot/frame
  -> apparent facial-expression detection
  -> mock mudra detection
  -> mock gesture description
  -> mock LLM response
  -> WIP Moktak decision
  -> terminal/debug output
```

The main app does not classify expression on every camera frame. It calls
`EmotionDetector.detect()` only after sustained attention triggers an
interaction event.

## Moktak Module (WIP)

`src/audio` contains the in-progress Moktak feedback path. The goal is to map
interaction state into a parameterized Moktak sound without coupling the policy
to OpenCV, MediaPipe, or the main app loop.

Current pieces:

- `MoktakParameters`: immutable playback intent model.
- `decide_moktak()`: pure policy that maps apparent expression, optional mudra,
  optional LLM response, and attention state to Moktak parameters.
- `render_moktak()`: renders `src/audio/moktak.wav` into an in-memory audio
  buffer using gain, intensity, duration, fades, and frequency scaling.
- `MoktakPlayer`: optional `sounddevice` adapter for looping rendered audio.
- `MoktakDisplay`: lightweight adapter used by the main app today; it logs the
  Moktak decision instead of owning real playback.
- `render_moktak_visualizer()`: OpenCV debug panel for the manual Moktak test.

Current policy behavior:

- `negative_expression` lowers the Moktak frequency.
- `positive_expression` raises the Moktak frequency.
- `neutral_expression` or missing expression keeps the base frequency.
- `NO_FACE` disables Moktak output.
- calm mudra markers can deepen the negative-expression frequency mapping, but
  mudra detection is still mocked.

WIP status:

- The main app currently uses `MoktakDisplay`, so it prints Moktak parameters
  rather than playing audio.
- The manual Moktak test uses `MoktakPlayer` and can play real audio when
  `sounddevice` and the host audio device are available.
- Mudra and LLM signals are still mocks in the webcam pipeline.
- The Moktak mapping is experimental and should be treated as a debug feedback
  mechanism, not a validated affective response model.

> [!IMPORTANT]
> To run/debug/demonstrate the moktak logic, use `python -m tests.test_moktak`

## Structure

```text
src/
  app.py                  threaded webcam app
  text_app.py             terminal-only Ollama text path
  profiling.py            JSONL profiling helpers
  attention/
    detector.py           MediaPipe attention score
    tracker.py            attention state and gaze duration
  emotions/
    detector.py           FACS AU-inspired blendshape heuristic
    types.py              EmotionState model
  audio/
    display.py            Moktak decision logger
    models.py             MoktakParameters
    player.py             optional real-audio adapter
    policy.py             pure Moktak policy
    renderer.py           WAV renderer/frequency shifter
    visualizer.py         OpenCV audio debug panel
    moktak.wav            source Moktak asset
  debug/
    windows.py            OpenCV overlays and debug windows
  llm/
    lifecycle.py          Ollama readiness/unload helpers
    mocks.py              webcam-pipeline mocks
    ollama_client.py      Ollama HTTP client
    response_generator.py text response prompt path
    scene_describer.py    vision prompt path
```

## Main Modules

**attention**: MediaPipe face-landmark attention score, attention state
classification, and sustained-gaze tracking.

**emotions**: MediaPipe Face Landmarker blendshape detection. The classifier is
a FACS Action Units (AUs)-inspired heuristic over visible blendshape cues and
returns `positive_expression`, `negative_expression`, or `neutral_expression`.

**audio**: WIP Moktak policy, parameter model, WAV renderer, optional real
player, terminal display adapter, and OpenCV visualizer.

**llm**: Ollama client and lifecycle helpers. The current webcam app uses mocks
for mudra detection, gesture description, and final response generation; the
Ollama-backed paths remain available separately.

**debug**: OpenCV overlay and window helpers. Detectors do not own rendering.

## Apparent Expression Heuristic

`src/emotions/detector.py` approximates FACS Action Units using MediaPipe Face
Landmarker blendshapes:

- positive cues: smile, cheek/eye activation, and surprise-like upper-face cues;
- negative cues: supported sadness, anger/tension, disgust/aversion, and
  anxiety/tension patterns;
- neutral fallback: returned when no candidate passes its threshold.

Negative expression detection requires supporting upper- and lower-face cues to
reduce false positives from isolated blendshape activation. The labels describe
visible facial cues only, not internal emotional state.

## Requirements

- Python 3.11+
- webcam accessible through OpenCV
- MediaPipe model files in `models/`
- `src/audio/moktak.wav`
- audio output device only for the manual Moktak audio test
- Ollama only for the optional `src.text_app` path

Install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running

Run the main attention-triggered webcam app:

```bash
python -m src.app
```

The main app opens the camera and emotion snapshot windows. It uses mocked
mudra/LLM components and logs the current Moktak decision to the terminal.

Run the optional terminal-only Ollama path:

```bash
ollama pull qwen2.5:1.5b
ollama serve
python -m src.text_app
```

Press `q` or `Esc` to quit OpenCV windows.

## Manual Checks

Run the isolated emotion webcam check:

```bash
python tests/test_emotions.py
# or
python -m tests.test_emotions
```

This opens `Emotion Test` and `Emotion Debug`, then prints the current
apparent-expression label and confidence.

Run the WIP Moktak webcam/audio check:

```bash
python -m tests.test_moktak
```

This opens `Moktak Camera Test` and `Moktak Audio Visualizer`, runs attention
and expression detection continuously, and uses `MoktakPlayer` for real audio
when available. If audio playback fails, the visualizer still shows the rendered
decision state.

Run the end-to-end main app check:

```bash
python -m src.app
```

This verifies the threaded webcam pipeline, sustained-attention trigger,
event-driven expression snapshot, mocked mudra/LLM path, and WIP Moktak decision
logging.

## Configuration

Webcam and interaction settings are constants near the top of `src/app.py`:

| Setting | Default | Purpose |
| --- | ---: | --- |
| `CAMERA_INDEX` | `0` | OpenCV camera device |
| `ATTENTION_THRESHOLD` | `0.7` | Threshold reference kept near app settings; state cutoffs live in `AttentionState.classify()` |
| `ATTENTION_DURATION_SECONDS` | `1.0` | Required sustained-attention time |
| `COOLDOWN_SECONDS` | `5.0` | Minimum delay between interaction events |
| `SHOW_CAMERA_WINDOW` | `True` | Toggle the live camera window |
| `SHOW_EMOTION_SNAPSHOT_WINDOW` | `True` | Toggle the latest event snapshot window |

Ollama-backed modules accept these environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API address |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Text model |
| `OLLAMA_VISION_MODEL` | `openbmb/minicpm-v4.6` | Vision model |
| `OLLAMA_TIMEOUT_SECONDS` | `30` | Text request timeout |
| `OLLAMA_VISION_TIMEOUT_SECONDS` | `60` | Vision request timeout |
| `ATTENTION_LOG_DIR` | `logs/` | Profiling JSONL output directory |

## Windows

Main app:

| Window | Purpose |
| --- | --- |
| `Camera` | Live camera frame with attention overlay |
| `Emotion Snapshot` | Latest frame that triggered event-driven expression detection |

Emotion check:

| Window | Purpose |
| --- | --- |
| `Emotion Test` | Live webcam frame with apparent-expression overlay |
| `Emotion Debug` | Apparent-expression metrics and top blendshape scores |

Moktak check:

| Window | Purpose |
| --- | --- |
| `Moktak Camera Test` | Live webcam frame with attention and expression overlays |
| `Moktak Audio Visualizer` | Current Moktak parameters, waveform, and output level |

## Limitations

- The attention score is a heuristic, not an objective measure of attention.
- The emotion module detects apparent facial expression, not real emotion.
- Moktak feedback is WIP and not validated as an affective audio model.
- Lighting, camera quality, occlusions, glasses, and individual differences
  affect webcam output.
- Looking at the camera does not imply consent, interest, or intent.
- Mudra detection, gesture description, and main-app response generation are
  currently mocks.
- The main app logs Moktak parameters; real audio playback is currently isolated
  to the manual Moktak check.
- Thresholds and mappings are experimental and have not been scientifically
  validated.
