"""Persistent, event-driven moktak session controller."""

from __future__ import annotations

from dataclasses import replace
from enum import Enum
from queue import Empty, Queue
from threading import Lock, Thread
import time
from typing import Callable, Protocol

from .models import BeatStep, MoktakEffectProfile, MoktakState
from .player import MoktakPlayer


Clock = Callable[[], float]


class BeatPlayer(Protocol):
    """Minimum player surface required by the moktak audio thread."""

    def play_beat(self, gain: float = 1.0) -> bool:
        """Play one moktak hit."""

    def stop(self) -> None:
        """Stop any currently sounding moktak hit."""


class _Command(Enum):
    START = "start"
    STOP = "stop"
    POSITIVE_EFFECT = "positive_effect"
    SHUTDOWN = "shutdown"


class MoktakSession:
    """Own a persistent moktak thread and drive it from event commands."""

    def __init__(
        self,
        *,
        beat_player: BeatPlayer | None = None,
        effect_profile: MoktakEffectProfile | None = None,
        clock: Clock = time.monotonic,
        thread_factory: Callable[..., Thread] = Thread,
    ) -> None:
        self.beat_player = beat_player or MoktakPlayer()
        self.effect_profile = effect_profile or MoktakEffectProfile()
        self.clock = clock
        self._commands: Queue[_Command] = Queue()
        self._state = MoktakState.STOPPED
        self._state_lock = Lock()
        self._thread = thread_factory(
            target=self._run,
            name="moktak-audio",
            daemon=True,
        )
        self._started = False

    @property
    def state(self) -> MoktakState:
        with self._state_lock:
            return self._state

    @property
    def positive_hold_seconds(self) -> float:
        return self.effect_profile.positive_hold_seconds

    @property
    def thread_alive(self) -> bool:
        return self._thread.is_alive()

    def start(self) -> None:
        """Start the persistent audio thread once."""
        if self._started:
            return
        self._started = True
        self._thread.start()

    def start_session(self) -> None:
        self._send(_Command.START)

    def trigger_positive_effect(self) -> None:
        self._send(_Command.POSITIVE_EFFECT)

    def stop_session(self) -> None:
        self._send(_Command.STOP)

    def shutdown(self, timeout: float | None = 2.0) -> None:
        self._send(_Command.SHUTDOWN)
        if self._started:
            self._thread.join(timeout=timeout)

    def _send(self, command: _Command) -> None:
        if self.state is MoktakState.SHUTDOWN:
            return
        self._commands.put(command)

    def _set_state(self, state: MoktakState) -> None:
        with self._state_lock:
            self._state = state

    def _run(self) -> None:
        next_beat_at = self.clock()
        while True:
            state = self.state
            if state is MoktakState.STOPPED:
                command = self._commands.get()
                if self._apply_stopped_command(command):
                    return
                next_beat_at = self.clock()
                continue

            if state is MoktakState.PLAYING:
                now = self.clock()
                if now >= next_beat_at:
                    self.beat_player.play_beat(self.effect_profile.normal_gain)
                    next_beat_at = now + self.effect_profile.normal_interval_seconds

                timeout = max(0.0, next_beat_at - self.clock())
                try:
                    command = self._commands.get(timeout=timeout)
                except Empty:
                    continue

                if self._apply_playing_command(command):
                    return
                if self.state is MoktakState.POSITIVE_EFFECT:
                    if self._run_positive_effect():
                        return
                    next_beat_at = self.clock()

    def _apply_stopped_command(self, command: _Command) -> bool:
        if command is _Command.SHUTDOWN:
            self.beat_player.stop()
            self._set_state(MoktakState.SHUTDOWN)
            return True
        if command is _Command.START:
            self._set_state(MoktakState.PLAYING)
        elif command is _Command.STOP:
            self.beat_player.stop()
        return False

    def _apply_playing_command(self, command: _Command) -> bool:
        if command is _Command.SHUTDOWN:
            self.beat_player.stop()
            self._set_state(MoktakState.SHUTDOWN)
            return True
        if command is _Command.STOP:
            self.beat_player.stop()
            self._set_state(MoktakState.STOPPED)
        elif command is _Command.POSITIVE_EFFECT:
            self._set_state(MoktakState.POSITIVE_EFFECT)
        return False

    def _run_positive_effect(self) -> bool:
        for step in self.effect_profile.positive_steps():
            self.beat_player.play_beat(step.gain)
            command = self._wait_during_step(step)
            if command is None:
                continue
            if command is _Command.SHUTDOWN:
                self.beat_player.stop()
                self._set_state(MoktakState.SHUTDOWN)
                return True
            if command is _Command.STOP:
                self.beat_player.stop()
                self._set_state(MoktakState.STOPPED)
                return False

        self.beat_player.stop()
        self._set_state(MoktakState.STOPPED)
        return False

    def _wait_during_step(self, step: BeatStep) -> _Command | None:
        deadline = self.clock() + step.interval_seconds
        while True:
            timeout = max(0.0, deadline - self.clock())
            if timeout <= 0.0:
                return None
            try:
                command = self._commands.get(timeout=timeout)
            except Empty:
                return None
            if command not in {_Command.START, _Command.POSITIVE_EFFECT}:
                return command

    def __enter__(self) -> "MoktakSession":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()


class PositiveEmotionHoldTracker:
    """Convert existing positive emotion state into a single session event."""

    def __init__(
        self,
        hold_seconds: float,
        *,
        clock: Clock = time.monotonic,
    ) -> None:
        self.hold_seconds = hold_seconds
        self.clock = clock
        self._positive_started_at: float | None = None
        self._fired = False

    def reset(self) -> None:
        self._positive_started_at = None
        self._fired = False

    def update(
        self,
        emotion_state: object | None,
        timestamp: float | None = None,
    ) -> bool:
        if self._fired:
            return False

        label = getattr(emotion_state, "label", "")
        if label != "positive_expression":
            self._positive_started_at = None
            return False

        now = self.clock() if timestamp is None else timestamp
        if self._positive_started_at is None:
            self._positive_started_at = now
            return False

        if now - self._positive_started_at < self.hold_seconds:
            return False

        self._fired = True
        return True


class AttentionActivationTracker:
    """Emit one event for each new sustained-attention activation."""

    def __init__(self) -> None:
        self._sustained_was_active = False

    def update(self, *, sustained_attention: bool, no_face: bool = False) -> bool:
        if no_face:
            self._sustained_was_active = False
            return False
        if not sustained_attention:
            self._sustained_was_active = False
            return False
        if self._sustained_was_active:
            return False
        self._sustained_was_active = True
        return True


def build_effect_profile_from_env(
    *,
    env_float: Callable[[str, float], float],
    env_int: Callable[[str, int], int],
) -> MoktakEffectProfile:
    """Build moktak effect settings from the existing env config helpers."""
    profile = MoktakEffectProfile()
    return replace(
        profile,
        normal_bpm=env_float("MOKTAK_NORMAL_BPM", profile.normal_bpm),
        peak_bpm=env_float("MOKTAK_PEAK_BPM", profile.peak_bpm),
        positive_hold_seconds=env_float(
            "MOKTAK_POSITIVE_HOLD_SECONDS",
            profile.positive_hold_seconds,
        ),
        acceleration_steps=env_int(
            "MOKTAK_ACCELERATION_STEPS",
            profile.acceleration_steps,
        ),
        deceleration_steps=env_int(
            "MOKTAK_DECELERATION_STEPS",
            profile.deceleration_steps,
        ),
        normal_gain=env_float("MOKTAK_NORMAL_GAIN", profile.normal_gain),
        minimum_gain=env_float("MOKTAK_MINIMUM_GAIN", profile.minimum_gain),
        fade_power=env_float("MOKTAK_FADE_POWER", profile.fade_power),
        final_interval_multiplier=env_float(
            "MOKTAK_FINAL_INTERVAL_MULTIPLIER",
            profile.final_interval_multiplier,
        ),
    )
