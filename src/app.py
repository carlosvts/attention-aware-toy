"""Threaded webcam app for the attention-triggered expression pipeline."""

from dataclasses import dataclass, replace
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
import time

import cv2
import numpy as np
from numpy.typing import NDArray

from src.attention import (
    AttentionDetector,
    AttentionResult,
    AttentionState,
    GazeDurationTracker,
)
from src.audio import MoktakParameters, MoktakPlayer, create_tts, decide_moktak
from src.config import env_bool, env_float, env_int
from src.debug import draw_attention_overlay, draw_emotion_overlay, show_or_close
from src.emotions import EmotionDetector, EmotionState
from src.llm import generate_response
from src.llm.mocks import MudraDetectorMock, MudraState

CAMERA_INDEX = env_int("CAMERA_INDEX", 0)
ATTENTION_THRESHOLD = env_float("ATTENTION_THRESHOLD", 0.7)
ATTENTION_DURATION_SECONDS = env_float("ATTENTION_DURATION_SECONDS", 1.0)
COOLDOWN_SECONDS = env_float("COOLDOWN_SECONDS", 5.0)
MIN_DUCKING_SECONDS = env_float("MIN_DUCKING_SECONDS", 1.2)
SPEECH_WORDS_PER_SECOND = env_float("SPEECH_WORDS_PER_SECOND", 2.6)

SHOW_ATTENTION_WINDOW = env_bool("SHOW_ATTENTION_WINDOW", True)
SHOW_EMOTION_SNAPSHOT_WINDOW = env_bool("SHOW_EMOTION_SNAPSHOT_WINDOW", True)
SHOW_MUDRA_SNAPSHOT_WINDOW = env_bool("SHOW_MUDRA_SNAPSHOT_WINDOW", True)
ATTENTION_WINDOW_NAME = "Attention"
EMOTION_SNAPSHOT_WINDOW_NAME = "Emotion Snapshot"
MUDRA_SNAPSHOT_WINDOW_NAME = "Mudra Snapshot"


@dataclass(frozen=True)
class FramePacket:
    frame: NDArray[np.uint8]
    timestamp: float


@dataclass(frozen=True)
class AttentionPacket:
    frame: NDArray[np.uint8]
    result: AttentionResult
    state: AttentionState
    gaze_duration: float


@dataclass(frozen=True)
class EmotionSnapshot:
    frame: NDArray[np.uint8]
    emotion: EmotionState | None
    mudra: MudraState | None = None


class LatestValue:
    """Thread-safe single-slot storage for live video state."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._value = None

    def set(self, value) -> None:
        with self._lock:
            self._value = value

    def get(self):
        with self._lock:
            return self._value


def _put_latest(queue: Queue[FramePacket], packet: FramePacket) -> None:
    if queue.full():
        try:
            queue.get_nowait()
        except Empty:
            pass
    try:
        queue.put_nowait(packet)
    except Full:
        pass


def _format_meditation_context(
    mudra_state: MudraState,
    emotion_state: EmotionState | None,
) -> str:
    emotion_label = emotion_state.label if emotion_state else "none"
    emotion_confidence = emotion_state.confidence if emotion_state else 0.0
    return (
        "meditation_context:\n"
        "  attention: sustained\n"
        f"  mudra: {mudra_state.label}\n"
        f"  mudra_confidence: {mudra_state.confidence:.3f}\n"
        f"  apparent_expression: {emotion_label}\n"
        f"  apparent_expression_confidence: {emotion_confidence:.3f}\n"
        "  role: buddhist meditation guide\n"
        "  instruction: guide the person through one calm breath or posture cue\n"
        "  language: English"
    )


def _ducked_parameters(parameters: MoktakParameters) -> MoktakParameters:
    return replace(parameters, gain=parameters.gain * parameters.ducking_gain)


def _estimated_speech_seconds(response: str) -> float:
    word_count = len(response.split())
    return max(MIN_DUCKING_SECONDS, word_count / SPEECH_WORDS_PER_SECOND)


def _draw_mudra_overlay(
    frame: NDArray[np.uint8],
    mudra: MudraState | None,
    origin: tuple[int, int] = (20, 135),
) -> None:
    text = (
        "mudra=none"
        if mudra is None
        else f"mudra={mudra.label} confidence={mudra.confidence:.2f}"
    )
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (180, 255, 160),
        2,
        cv2.LINE_AA,
    )


def _draw_snapshot_title(
    frame: NDArray[np.uint8],
    title: str,
    color: tuple[int, int, int],
) -> None:
    cv2.putText(
        frame,
        title,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        color,
        2,
        cv2.LINE_AA,
    )


def _camera_worker(
    stop_event: Event,
    latest_frame: LatestValue,
    attention_frames: Queue[FramePacket],
) -> None:
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print(f"Não foi possível abrir a webcam no índice {CAMERA_INDEX}.")
        stop_event.set()
        camera.release()
        return

    try:
        while not stop_event.is_set():
            ok, frame = camera.read()
            if not ok:
                print("Não foi possível capturar um frame da webcam.")
                stop_event.set()
                break

            packet = FramePacket(frame=frame.copy(), timestamp=time.monotonic())
            latest_frame.set(packet)
            _put_latest(
                attention_frames,
                FramePacket(frame=frame.copy(), timestamp=packet.timestamp),
            )
    finally:
        camera.release()


def _attention_worker(
    stop_event: Event,
    attention_frames: Queue[FramePacket],
    latest_attention: LatestValue,
    emotion_events: Queue[FramePacket],
) -> None:
    gaze_tracker = GazeDurationTracker()
    last_trigger_at = float("-inf")

    try:
        with AttentionDetector() as detector:
            while not stop_event.is_set():
                try:
                    packet = attention_frames.get(timeout=0.05)
                except Empty:
                    continue

                result = detector.process(packet.frame)
                state = AttentionState.classify(
                    result.score,
                    face_detected=result.face_box is not None,
                )
                now = time.monotonic()
                gaze_duration = gaze_tracker.update(state, now)
                sustained_attention = (
                    state is AttentionState.ATTENDING
                    and gaze_duration >= ATTENTION_DURATION_SECONDS
                )

                latest_attention.set(
                    AttentionPacket(
                        frame=packet.frame.copy(),
                        result=result,
                        state=state,
                        gaze_duration=gaze_duration,
                    )
                )

                if (
                    sustained_attention
                    and now - last_trigger_at >= COOLDOWN_SECONDS
                    and not emotion_events.full()
                ):
                    _put_latest(
                        emotion_events,
                        FramePacket(frame=packet.frame.copy(), timestamp=now),
                    )
                    last_trigger_at = now
    except Exception as error:
        print(f"Erro na thread de atenção: {error}")
        stop_event.set()


def _event_worker(
    stop_event: Event,
    emotion_events: Queue[FramePacket],
    latest_snapshot: LatestValue,
    latest_attention: LatestValue,
) -> None:
    mudra_detector = MudraDetectorMock()
    tts = create_tts()
    number_responses = 0
    active_moktak_parameters = MoktakParameters.disabled()
    try:
        with EmotionDetector() as emotion_detector, MoktakPlayer() as moktak_player:
            while not stop_event.is_set():
                try:
                    packet = emotion_events.get(timeout=0.05)
                except Empty:
                    attention = latest_attention.get()
                    if (
                        active_moktak_parameters.enabled
                        and attention is not None
                        and attention.state is AttentionState.NO_FACE
                    ):
                        active_moktak_parameters = MoktakParameters.disabled()
                        moktak_player.play(active_moktak_parameters)
                    continue

                attention = latest_attention.get()
                attention_state = (
                    attention.state
                    if attention is not None
                    else AttentionState.ATTENDING
                )
                gaze_duration = attention.gaze_duration if attention is not None else 0.0

                initial_moktak_parameters = decide_moktak(
                    emotion_state=None,
                    attention_state=attention_state,
                )
                moktak_player.play(initial_moktak_parameters)
                active_moktak_parameters = initial_moktak_parameters

                emotion = emotion_detector.detect(packet.frame)
                mudra = mudra_detector.detect(packet.frame)
                expression = emotion.label if emotion else "none"
                confidence = emotion.confidence if emotion else 0.0

                latest_snapshot.set(
                    EmotionSnapshot(
                        frame=packet.frame.copy(),
                        emotion=emotion,
                        mudra=mudra,
                    )
                )

                meditation_context = _format_meditation_context(mudra, emotion)
                moktak_parameters = decide_moktak(
                    emotion_state=emotion,
                    mudra_state=mudra,
                    attention_state=attention_state,
                )
                moktak_player.play(moktak_parameters)
                active_moktak_parameters = moktak_parameters

                response = generate_response(
                    meditation_context,
                    attention_state=attention_state,
                    gaze_duration=gaze_duration,
                    emotion_state=emotion,
                )
                speech_audio = tts.synthesize(response)
                
                ###################################################
                # Debug prints
                print()
                print("="*50)
                print("Interaction Number: ", number_responses)
                print("attention detected", flush=True)
                print(
                    f"apparent_expression={expression} confidence={confidence:.3f}",
                    flush=True,
                )
                print(
                    f"mock_mudra={mudra.label} confidence={mudra.confidence:.3f}",
                    flush=True,
                )
                print(f"meditation_context={meditation_context}", flush=True)
                print(f"llm_response={response}", flush=True)
                print("moktak_ducking=on", flush=True)
                print("="*50)
                print()
                if speech_audio is not None:
                    print("tts_audio=on", flush=True)
                    moktak_player.play_ducked_speech(
                        moktak_parameters,
                        speech_audio.samples,
                        speech_audio.sample_rate,
                    )
                else:
                    if tts.last_error:
                        print(f"tts_audio=off reason={tts.last_error}", flush=True)
                    ducked_parameters = _ducked_parameters(moktak_parameters)
                    moktak_player.play(ducked_parameters)
                    time.sleep(_estimated_speech_seconds(response))
                moktak_player.play(moktak_parameters)
                active_moktak_parameters = moktak_parameters
                print("moktak_ducking=off", flush=True)

                number_responses += 1
                # End debug prints
                ###################################################

                latest_snapshot.set(
                    EmotionSnapshot(
                        frame=packet.frame.copy(),
                        emotion=emotion,
                        mudra=mudra,
                    )
                )
    except Exception as error:
        print(f"Erro na thread de evento: {error}")
        stop_event.set()


def run() -> None:
    """Run the threaded webcam attention loop."""
    stop_event = Event()
    latest_frame = LatestValue()
    latest_attention = LatestValue()
    latest_snapshot = LatestValue()
    attention_frames: Queue[FramePacket] = Queue(maxsize=1)
    emotion_events: Queue[FramePacket] = Queue(maxsize=1)

    workers = (
        Thread(
            target=_camera_worker,
            args=(stop_event, latest_frame, attention_frames),
            name="camera-worker",
            daemon=True,
        ),
        Thread(
            target=_attention_worker,
            args=(stop_event, attention_frames, latest_attention, emotion_events),
            name="attention-worker",
            daemon=True,
        ),
        Thread(
            target=_event_worker,
            args=(stop_event, emotion_events, latest_snapshot, latest_attention),
            name="event-worker",
            daemon=True,
        ),
    )

    for worker in workers:
        worker.start()

    try:
        cv2.namedWindow(ATTENTION_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.namedWindow(EMOTION_SNAPSHOT_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.namedWindow(MUDRA_SNAPSHOT_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.moveWindow(ATTENTION_WINDOW_NAME, 40, 80)
        cv2.moveWindow(EMOTION_SNAPSHOT_WINDOW_NAME, 760, 80)
        cv2.moveWindow(MUDRA_SNAPSHOT_WINDOW_NAME, 760, 560)

        while not stop_event.is_set():
            frame_packet = latest_frame.get()
            attention_packet = latest_attention.get()
            snapshot_packet = latest_snapshot.get()

            if attention_packet is not None:
                attention_view = attention_packet.frame.copy()
                draw_attention_overlay(
                    attention_view,
                    attention_packet.result,
                    attention_packet.state,
                    attention_packet.gaze_duration,
                )
                show_or_close(
                    ATTENTION_WINDOW_NAME,
                    SHOW_ATTENTION_WINDOW,
                    attention_view,
                )
            elif frame_packet is not None:
                show_or_close(
                    ATTENTION_WINDOW_NAME,
                    SHOW_ATTENTION_WINDOW,
                    frame_packet.frame,
                )

            if snapshot_packet is not None:
                emotion_view = snapshot_packet.frame.copy()
                _draw_snapshot_title(
                    emotion_view,
                    "Emotion Snapshot",
                    (120, 220, 255),
                )
                draw_emotion_overlay(
                    emotion_view,
                    snapshot_packet.emotion,
                    origin=(20, 78),
                )
                show_or_close(
                    EMOTION_SNAPSHOT_WINDOW_NAME,
                    SHOW_EMOTION_SNAPSHOT_WINDOW,
                    emotion_view,
                )

                mudra_view = snapshot_packet.frame.copy()
                _draw_snapshot_title(
                    mudra_view,
                    "Mudra Snapshot",
                    (180, 255, 160),
                )
                _draw_mudra_overlay(
                    mudra_view,
                    snapshot_packet.mudra,
                    origin=(20, 78),
                )
                show_or_close(
                    MUDRA_SNAPSHOT_WINDOW_NAME,
                    SHOW_MUDRA_SNAPSHOT_WINDOW,
                    mudra_view,
                )

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                stop_event.set()
                break

            time.sleep(0.001)
    finally:
        stop_event.set()
        for worker in workers:
            worker.join(timeout=2.0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
