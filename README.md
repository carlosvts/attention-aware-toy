# Attention-Aware Toy

Experimental Human-Robot Interaction prototype driven by webcam attention,
apparent facial-expression cues, mocked interaction context, and a Moktak audio
feedback module.

This is not a production system. It does not identify people and must not be
used to infer a person's real emotion, intent, consent, or mental state.
Facial-expression output is only an apparent-expression heuristic from visible
MediaPipe blendshapes.

## Current Pipeline

The main app runs camera capture, attention detection, and event processing in
separate threads. The Moktak audio loop now has its own persistent thread and is
controlled by events rather than by the snapshot/LLM/TTS path:

```text
camera frame
  -> continuous attention scoring
  -> sustained attention gate
  -> Moktak start/stop events
  -> snapshot/frame
  -> apparent facial-expression detection
  -> positive-expression hold tracker
  -> Moktak positive-effect event
  -> mock mudra detection
  -> mock gesture description
  -> mock LLM response
  -> terminal/debug output
```

The response-generation path remains event-driven. While the Moktak is playing,
the event worker also polls the latest frame for apparent expression so
`positive_expression` can trigger the final effect without waiting for a new
snapshot/LLM event.

## Moktak Module

`src/audio` contains the Moktak feedback path. The audio lifecycle is decoupled
from OpenCV, MediaPipe, LLM, and TTS. The app sends commands to a persistent
Moktak thread, and that thread owns beat scheduling.

Current pieces:

- `MoktakParameters`: immutable playback intent model.
- `decide_moktak()`: pure policy that maps apparent expression, optional mudra,
  optional LLM response, and attention state to Moktak parameters.
- `render_moktak()`: renders `src/audio/moktak.wav` into an in-memory audio
  buffer using gain, intensity, duration, and fades while preserving the source
  hit's natural pitch.
- `render_moktak_hit()`: extracts a single natural-speed Moktak hit for the
  event-driven beat scheduler.
- `MoktakPlayer`: optional `sounddevice` adapter for one-shot beat playback.
- `MoktakSession`: persistent audio-thread controller with `STOPPED`,
  `PLAYING`, `POSITIVE_EFFECT`, and `SHUTDOWN` states.
- `AttentionActivationTracker`: emits one start event for each new sustained
  attention activation.
- `PositiveEmotionHoldTracker`: emits one positive-effect event after
  `positive_expression` remains active for the configured hold duration.
- `render_moktak_visualizer()`: OpenCV audio-buffer debug helper.

Lifecycle behavior:

- The Moktak starts in `STOPPED`.
- A new sustained-attention activation sends `start_session()` and moves to
  `PLAYING`.
- Intermediate attention states such as `DISTRACTED`, `LOOKING_BRIEFLY`, closed
  eyes, and gaze drops do not stop playback.
- `NO_FACE` sends `stop_session()` and stops playback immediately.
- While `PLAYING`, `positive_expression` held for
  `MOKTAK_POSITIVE_HOLD_SECONDS` seconds sends `trigger_positive_effect()`.
- The positive effect accelerates, decelerates with fade-out, then stops in
  `STOPPED`.
- After the effect finishes, the Moktak does not restart until attention leaves
  the activation state and a new sustained-attention activation occurs.

Positive-effect profile:

- The source WAV is a single Moktak hit; BPM is produced by repeating that hit
  at changing intervals.
- Acceleration uses a normalized `sin^2(pi * t / 2Ta)` curve mapped from
  `normal_bpm` to `peak_bpm`.
- Deceleration uses `0.5 * (1 + cos(pi * t / Td))` mapped from `peak_bpm` to a
  finite `final_bpm`.
- Gain fades during deceleration using the same cosine energy envelope and
  `fade_power`.
- The thread waits with command timeouts, so `NO_FACE` and shutdown can
  interrupt the effect.

> [!IMPORTANT]
> To run/debug/demonstrate the Moktak logic without Ollama or TTS, use
> `python -m tests.test_moktak`.

## Organograms

Mermaid workflow diagrams live in `organograms/`:

- [Main app workflow](organograms/main-app-workflow.md)
- [Moktak session lifecycle](organograms/moktak-session-lifecycle.md)
- [Positive-effect profile](organograms/moktak-positive-effect.md)
- [Manual Moktak debug workflow](organograms/moktak-debug-workflow.md)

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
    models.py             MoktakParameters, MoktakEffectProfile, MoktakState
    player.py             optional real-audio adapter
    policy.py             pure Moktak policy
    renderer.py           WAV renderer and single-hit extractor
    session.py            persistent Moktak thread and event trackers
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

**audio**: Moktak policy, parameter/effect models, WAV renderer, one-shot
player, persistent session thread, event trackers, terminal display adapter,
and OpenCV visualizer.

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

The main app opens attention, snapshot, mudra, and Moktak debug windows. It uses
mocked mudra/LLM components and runs Moktak playback in a dedicated audio
thread.

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

Run the Moktak webcam/audio check:

```bash
python -m tests.test_moktak
```

This opens `Moktak Camera Test`, `Moktak Emotion Debug`, and
`Moktak Logic Debug`, runs attention and expression detection continuously, and
uses `MoktakSession`/`MoktakPlayer` for real audio when available. It does not
use Ollama, LLM, or TTS.

Run the end-to-end main app check:

```bash
python -m src.app
```

This verifies the threaded webcam pipeline, sustained-attention trigger,
event-driven expression snapshot, mocked mudra/LLM path, and Moktak audio
session lifecycle.

## Configuration

Copy `.env-example` to `.env` and adjust values for your local machine:

```bash
cp .env-example .env
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

Webcam and interaction settings are loaded from `.env`:

| Variable | Default | Purpose |
| --- | ---: | --- |
| `CAMERA_INDEX` | `0` | OpenCV camera device |
| `ATTENTION_THRESHOLD` | `0.7` | Threshold reference kept near app settings; state cutoffs live in `AttentionState.classify()` |
| `ATTENTION_DURATION_SECONDS` | `1.0` | Required sustained-attention time |
| `COOLDOWN_SECONDS` | `5.0` | Minimum delay between interaction events |
| `EMOTION_POLL_SECONDS` | `0.15` | Live apparent-expression poll interval while Moktak is playing |
| `SHOW_ATTENTION_WINDOW` | `True` | Toggle the continuous attention camera window |
| `SHOW_EMOTION_SNAPSHOT_WINDOW` | `True` | Toggle the event emotion snapshot window |
| `SHOW_MUDRA_SNAPSHOT_WINDOW` | `True` | Toggle the event mudra snapshot window |
| `SHOW_MOKTAK_DEBUG_WINDOW` | `True` | Toggle the Moktak lifecycle debug window |

Moktak effect settings are loaded from `.env`:

| Variable | Default | Purpose |
| --- | ---: | --- |
| `MOKTAK_NORMAL_BPM` | `60.0` | Normal beat rate while playing |
| `MOKTAK_PEAK_BPM` | `180.0` | Peak rate reached during positive effect |
| `MOKTAK_POSITIVE_HOLD_SECONDS` | `3.0` | Required continuous `positive_expression` duration |
| `MOKTAK_ACCELERATION_STEPS` | `5` | Number of beat steps in the acceleration phase |
| `MOKTAK_DECELERATION_STEPS` | `8` | Number of beat steps in the deceleration/fade phase |
| `MOKTAK_NORMAL_GAIN` | `0.40` | Normal Moktak gain |
| `MOKTAK_MINIMUM_GAIN` | `0.08` | Final fade-out gain |
| `MOKTAK_FADE_POWER` | `1.25` | Fade curve exponent during deceleration |
| `MOKTAK_FINAL_INTERVAL_MULTIPLIER` | `1.8` | Final interval multiplier relative to normal interval |

Ollama-backed modules accept these variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API address |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Text model |
| `OLLAMA_VISION_MODEL` | `openbmb/minicpm-v4.6` | Vision model |
| `OLLAMA_TIMEOUT_SECONDS` | `30` | Text request timeout |
| `OLLAMA_VISION_TIMEOUT_SECONDS` | `60` | Vision request timeout |
| `ATTENTION_LOG_DIR` | `logs/` | Profiling JSONL output directory |

TTS accepts these variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `TTS_PROVIDER` | `generic` | `generic` prints speech only; `openai` enables API TTS |
| `OPENAI_API_KEY` | empty | Required only when `TTS_PROVIDER=openai` |
| `OPENAI_TTS_MODEL` | `tts-1` | Optional OpenAI text-to-speech model |
| `OPENAI_TTS_VOICE` | `alloy` | Optional OpenAI built-in TTS voice |
| `OPENAI_TTS_INSTRUCTIONS` | calm guide prompt | Optional OpenAI voice style instructions |

The default `generic` provider does not call any external TTS API. It keeps the
LLM response in the terminal. To re-enable OpenAI TTS later, install the OpenAI
SDK and set `TTS_PROVIDER=openai` with a valid `OPENAI_API_KEY`.

## Windows

Main app:

| Window | Purpose |
| --- | --- |
| `Attention` | Continuous frame processed by the attention detector with attention overlay |
| `Emotion Snapshot` | Event frame with apparent-expression overlay |
| `Mudra Snapshot` | Event frame with mudra overlay |
| `Moktak Debug` | Latest apparent expression, attention duration, and Moktak lifecycle state |

Emotion check:

| Window | Purpose |
| --- | --- |
| `Emotion Test` | Live webcam frame with apparent-expression overlay |
| `Emotion Debug` | Apparent-expression metrics and top blendshape scores |

Moktak check:

| Window | Purpose |
| --- | --- |
| `Moktak Camera Test` | Live webcam frame with attention and expression overlays |
| `Moktak Emotion Debug` | Apparent-expression metrics and top blendshape scores |
| `Moktak Logic Debug` | Moktak lifecycle state, attention/positive-expression timers, and trigger progress |

## Limitations

- The attention score is a heuristic, not an objective measure of attention.
- The emotion module detects apparent facial expression, not real emotion.
- Moktak feedback is experimental and not validated as an affective audio model.
- Lighting, camera quality, occlusions, glasses, and individual differences
  affect webcam output.
- Looking at the camera does not imply consent, interest, or intent.
- Mudra detection, gesture description, and main-app response generation are
  currently mocks.
- Thresholds and mappings are experimental and have not been scientifically
  validated.
