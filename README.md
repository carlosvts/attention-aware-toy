# Attention-Aware Toy

Toy project for an attention-triggered Human-Robot Interaction pipeline.

This is an experimental prototype, not a production system. It does not identify people and must not be used to infer a person's real emotion, intent, or mental state. Facial-expression output is only an apparent-expression heuristic from visible blendshapes.

## Current Pipeline

The main app runs camera capture, attention detection, and event processing in separate threads. Emotion detection is event-driven:

```text
attention detected
  -> screenshot/frame
  -> apparent facial-expression detection
  -> mock mudra detection
  -> mock gesture description
  -> mock LLM response
  -> terminal output
```

The main app does not run emotion detection on every frame. It only calls `EmotionDetector.detect()` after sustained attention triggers an interaction event.

## Structure

```text
src/
  app.py
  __init__.py
  attention/
    __init__.py
    detector.py
    tracker.py
  emotions/
    __init__.py
    detector.py
    types.py
  llm/
    __init__.py
    lifecycle.py
    mocks.py
    ollama_client.py
    response_generator.py
    scene_describer.py
  debug/
    __init__.py
    windows.py
  profiling.py
  text_app.py
```

There is no final `src/state` or `src/perception` module. Attention, emotion, debug windows, and LLM/VLM-related code are separated by responsibility.

## Modules

**attention**: MediaPipe face-landmark attention score, gaze duration, and sustained-attention gating.

**emotions**: MediaPipe Face Landmarker blendshape heuristics. `EmotionDetector` loads `models/face_landmarker.task` with `output_face_blendshapes=True`, receives OpenCV BGR frames, converts to RGB, and returns `EmotionState | None`.

**llm**: Ollama VLM/LLM clients plus local mocks used by the current webcam pipeline for mudra detection, gesture description, and response generation.

**debug**: OpenCV drawing and `cv2.imshow` helpers. Detector logic does not own window rendering.

## Apparent Expression Heuristic

`src/emotions/detector.py` maps blendshapes conservatively. Paired
blendshapes are averaged before the candidate scores are calculated:

- `positive_expression`: strongest valid smile or surprise pattern;
- `negative_expression`: strongest coherent tension, nasal-negative, or sadness/displeasure pattern, threshold `0.16`;
- `neutral_expression`: returned when the strongest candidate does not reach its threshold.

Terminal and overlay text use cautious labels such as `apparent_expression=positive_expression` and `apparent_expression=neutral_expression`. These labels describe visible facial cues, not a person's internal emotional state.

## Requirements

- Python 3.11+
- a webcam accessible through OpenCV
- the MediaPipe model files included in `models/`
- Ollama only when running the optional text-only LLM entry point

## Running

Install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the attention-triggered app:

```bash
python -m src.app
```

The webcam app currently uses local mocks after expression detection, so it does not require Ollama.

Run the optional text-only Ollama path:

```bash
ollama pull qwen2.5:1.5b
ollama serve
python -m src.text_app
```

Run the isolated emotion webcam debug script:

```bash
python tests/test_emotions.py
# or
python -m tests.test_emotions
```

Press `q` or `Esc` to quit OpenCV windows.

## Configuration

Webcam and interaction settings are constants near the top of `src/app.py`:

| Setting | Default | Purpose |
| --- | ---: | --- |
| `CAMERA_INDEX` | `0` | OpenCV camera device |
| `ATTENTION_THRESHOLD` | `0.7` | Minimum attention score |
| `ATTENTION_DURATION_SECONDS` | `1.0` | Required sustained-attention time |
| `COOLDOWN_SECONDS` | `5.0` | Minimum delay between events |

The Ollama-backed modules accept these environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API address |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Text model |
| `OLLAMA_VISION_MODEL` | `qwen3-vl:2b` | Vision model |
| `OLLAMA_TIMEOUT_SECONDS` | `30` | Text request timeout |
| `OLLAMA_VISION_TIMEOUT_SECONDS` | `60` | Vision request timeout |
| `ATTENTION_LOG_DIR` | `logs/` | Telemetry output directory |

## Windows

Main app:

| Window | Purpose |
| --- | --- |
| `Camera` | Live camera frame with attention overlay |
| `Emotion Snapshot` | Captured frame used for event-driven emotion detection |

Emotion test:

| Window | Purpose |
| --- | --- |
| `Emotion Test` | Live webcam frame with apparent-expression overlay |
| `Emotion Debug` | Apparent-expression metrics and top blendshape scores |

## Tests

Unit tests:

```bash
python -m unittest discover -s tests -v
```

The automated suite covers attention classification and gating, expression heuristics, overlays, Ollama preflight and client behavior, prompts, scene description, and profiling.

Manual webcam emotion test:

```bash
python tests/test_emotions.py
```

## Limitations

- The attention score is a heuristic, not an objective measure of attention.
- The emotion module detects apparent facial expression, not real emotion.
- Lighting, camera quality, occlusions, glasses, and individual differences affect output.
- Looking at the camera does not imply consent, interest, or intent.
- Mudra detection and gesture description are currently mocks.
- The webcam pipeline's final response generator is currently a mock; the Ollama-backed modules remain available separately.
- Thresholds are experimental and have not been scientifically validated.
