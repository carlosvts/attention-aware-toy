"""Threaded webcam app for the attention-triggered expression pipeline."""

from dataclasses import dataclass
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
from src.debug import draw_attention_overlay, draw_emotion_overlay, show_or_close
from src.emotions import EmotionDetector, EmotionState
from src.llm.mocks import GestureDescriptionMock, LLMResponseMock, MudraDetectorMock

CAMERA_INDEX = 0
ATTENTION_THRESHOLD = 0.7
ATTENTION_DURATION_SECONDS = 1.0
COOLDOWN_SECONDS = 5.0

SHOW_CAMERA_WINDOW = True
SHOW_EMOTION_SNAPSHOT_WINDOW = True
CAMERA_WINDOW_NAME = "Camera"
EMOTION_SNAPSHOT_WINDOW_NAME = "Emotion Snapshot"


@dataclass(frozen=True)
class FramePacket:
    frame: NDArray[np.uint8]
    timestamp: float


@dataclass(frozen=True)
class AttentionPacket:
    result: AttentionResult
    state: AttentionState
    gaze_duration: float


@dataclass(frozen=True)
class EmotionSnapshot:
    frame: NDArray[np.uint8]
    emotion: EmotionState | None


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
) -> None:
    mudra_detector = MudraDetectorMock()
    gesture_description = GestureDescriptionMock()
    llm_response = LLMResponseMock()
    number_responses = 0
    try:
        with EmotionDetector() as emotion_detector:
            while not stop_event.is_set():
                try:
                    packet = emotion_events.get(timeout=0.05)
                except Empty:
                    continue

                emotion = emotion_detector.detect(packet.frame)
                mudra = mudra_detector.detect(packet.frame)
                description = gesture_description.describe(mudra, emotion)
                response = llm_response.generate(description)
                expression = emotion.label if emotion else "unknown"
                confidence = emotion.confidence if emotion else 0.0
                
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
                print(f"mock_gesture_description={description}", flush=True)
                print(f"mock_llm_response={response}", flush=True)
                print("="*50)
                print()
                number_responses += 1
                # End debug prints
                ###################################################

                snapshot = packet.frame.copy()
                draw_emotion_overlay(snapshot, emotion)
                latest_snapshot.set(EmotionSnapshot(frame=snapshot, emotion=emotion))
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
            args=(stop_event, emotion_events, latest_snapshot),
            name="event-worker",
            daemon=True,
        ),
    )

    for worker in workers:
        worker.start()

    try:
        cv2.namedWindow(CAMERA_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.namedWindow(EMOTION_SNAPSHOT_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.moveWindow(CAMERA_WINDOW_NAME, 40, 80)
        cv2.moveWindow(EMOTION_SNAPSHOT_WINDOW_NAME, 760, 80)

        while not stop_event.is_set():
            frame_packet = latest_frame.get()
            attention_packet = latest_attention.get()
            snapshot_packet = latest_snapshot.get()

            if frame_packet is not None:
                frame = frame_packet.frame.copy()
                if attention_packet is not None:
                    draw_attention_overlay(
                        frame,
                        attention_packet.result,
                        attention_packet.state,
                        attention_packet.gaze_duration,
                    )
                show_or_close(CAMERA_WINDOW_NAME, SHOW_CAMERA_WINDOW, frame)

            if snapshot_packet is not None:
                show_or_close(
                    EMOTION_SNAPSHOT_WINDOW_NAME,
                    SHOW_EMOTION_SNAPSHOT_WINDOW,
                    snapshot_packet.frame,
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
