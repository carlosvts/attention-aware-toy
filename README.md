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

**llm**: current Ollama/VLM/LLM code plus local mocks for unavailable mudra and gesture-description modules.

**debug**: OpenCV drawing and `cv2.imshow` helpers. Detector logic does not own window rendering.

## Apparent Expression Heuristic

`src/emotions/detector.py` maps blendshapes conservatively:

- `mouthSmileLeft + mouthSmileRight` -> `smiling_expression`
- `mouthFrownLeft + mouthFrownRight` -> `frowning_expression`
- `browDownLeft + browDownRight` -> `focused_expression`
- `eyeWideLeft + eyeWideRight + jawOpen` -> `surprised_expression`
- otherwise -> `neutral_expression`

Terminal and overlay text use cautious labels such as `apparent_expression=smiling_expression` and `apparent_expression=neutral_expression`.

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

Run the isolated emotion webcam debug script:

```bash
python tests/test_emotions.py
# or
python -m tests.test_emotions
```

Press `q` or `Esc` to quit OpenCV windows.

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
