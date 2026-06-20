"""Low-overhead profiling and resource telemetry for the experimental pipeline."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from functools import wraps
import os
import threading
import time
from typing import Any, ParamSpec, TypeVar

try:
    import psutil
except ImportError:  # pragma: no cover - exercised only without optional telemetry
    psutil = None  # type: ignore[assignment]

try:
    import pynvml
except ImportError:  # pragma: no cover - exercised only without optional telemetry
    pynvml = None  # type: ignore[assignment]


P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True)
class ResourceSnapshot:
    """Process, host, and NVIDIA GPU resource usage at one instant."""

    process_cpu_percent: float
    per_core_cpu_percent: tuple[float, ...]
    rss_bytes: int
    system_memory_used_bytes: int
    system_memory_total_bytes: int
    gpu_util_percent: float | None = None
    vram_used_bytes: int | None = None
    vram_free_bytes: int | None = None
    gpu_power_watts: float | None = None
    gpu_temperature_c: float | None = None


@dataclass
class InteractionMetrics:
    """Measurements collected while one multimodal interaction is running."""

    started_at: float
    cpu_started_at: float
    snapshots: list[ResourceSnapshot] = field(default_factory=list)
    snapshot_lock: threading.Lock = field(default_factory=threading.Lock)

    def add_snapshot(self, snapshot: ResourceSnapshot) -> None:
        with self.snapshot_lock:
            self.snapshots.append(snapshot)

    def snapshot_copy(self) -> list[ResourceSnapshot]:
        with self.snapshot_lock:
            return list(self.snapshots)


class _NvidiaMonitor:
    """Best-effort aggregate NVML monitor for every visible NVIDIA GPU."""

    def __init__(self) -> None:
        self._handles: tuple[Any, ...] = ()
        self._error: str | None = None
        if pynvml is None:
            self._error = "pynvml não instalado"
            return
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            self._handles = tuple(
                pynvml.nvmlDeviceGetHandleByIndex(index) for index in range(count)
            )
            if not self._handles:
                self._error = "nenhuma GPU NVIDIA detectada"
        except Exception as error:  # NVML errors depend on driver/runtime versions.
            self._error = str(error)

    @property
    def availability_error(self) -> str | None:
        return self._error

    def snapshot(self) -> dict[str, float | int | None] | None:
        if not self._handles or pynvml is None:
            return None

        utilizations: list[float] = []
        used_bytes = 0
        free_bytes = 0
        powers: list[float] = []
        temperatures: list[float] = []
        for handle in self._handles:
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilizations.append(float(utilization.gpu))
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_bytes += int(memory.used)
                free_bytes += int(memory.free)
            except Exception:
                continue
            try:
                powers.append(float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0)
            except Exception:
                pass
            try:
                temperatures.append(
                    float(
                        pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                    )
                )
            except Exception:
                pass

        if not utilizations:
            return None
        return {
            "util": sum(utilizations) / len(utilizations),
            "used": used_bytes,
            "free": free_bytes,
            "power": sum(powers) if powers else None,
            "temperature": max(temperatures) if temperatures else None,
        }


_print_lock = threading.Lock()
_current_interaction: ContextVar[InteractionMetrics | None] = ContextVar(
    "current_interaction", default=None
)
_process = psutil.Process(os.getpid()) if psutil is not None else None
_nvidia = _NvidiaMonitor()
_availability_reported = False


def _megabytes(value: int) -> float:
    return value / (1024 * 1024)


def _format_duration(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds:.3f} s"
    return f"{seconds * 1000:.1f} ms"


def _resource_snapshot(process_cpu_percent: float) -> ResourceSnapshot:
    if psutil is not None and _process is not None:
        per_core = tuple(float(value) for value in psutil.cpu_percent(percpu=True))
        rss_bytes = int(_process.memory_info().rss)
        system_memory = psutil.virtual_memory()
        memory_used = int(system_memory.used)
        memory_total = int(system_memory.total)
    else:
        per_core = ()
        rss_bytes = 0
        memory_used = 0
        memory_total = 0

    gpu = _nvidia.snapshot()
    return ResourceSnapshot(
        process_cpu_percent=process_cpu_percent,
        per_core_cpu_percent=per_core,
        rss_bytes=rss_bytes,
        system_memory_used_bytes=memory_used,
        system_memory_total_bytes=memory_total,
        gpu_util_percent=float(gpu["util"]) if gpu else None,
        vram_used_bytes=int(gpu["used"]) if gpu else None,
        vram_free_bytes=int(gpu["free"]) if gpu else None,
        gpu_power_watts=(
            float(gpu["power"])
            if gpu and gpu["power"] is not None
            else None
        ),
        gpu_temperature_c=(
            float(gpu["temperature"])
            if gpu and gpu["temperature"] is not None
            else None
        ),
    )


def _emit_availability_once() -> None:
    global _availability_reported
    if _availability_reported:
        return
    _availability_reported = True
    messages = []
    if psutil is None:
        messages.append("CPU/RAM indisponíveis: psutil não instalado")
    if _nvidia.availability_error:
        messages.append(f"GPU NVIDIA indisponível: {_nvidia.availability_error}")
    if messages:
        with _print_lock:
            print(f"[TELEMETRY] {'; '.join(messages)}")


def record_step(name: str, wall_seconds: float, cpu_seconds: float = 0.0) -> None:
    """Emit one standardized timing and resource sample."""
    process_cpu = (
        max(0.0, cpu_seconds) / wall_seconds * 100.0 if wall_seconds > 0 else 0.0
    )
    snapshot = _resource_snapshot(process_cpu)
    interaction = _current_interaction.get()
    if interaction is not None:
        interaction.add_snapshot(snapshot)

    cores = ",".join(f"{value:.0f}" for value in snapshot.per_core_cpu_percent)
    gpu_fields = "gpu=n/a"
    if snapshot.gpu_util_percent is not None:
        power = (
            f"{snapshot.gpu_power_watts:.1f}W"
            if snapshot.gpu_power_watts is not None
            else "n/a"
        )
        temperature = (
            f"{snapshot.gpu_temperature_c:.0f}C"
            if snapshot.gpu_temperature_c is not None
            else "n/a"
        )
        gpu_fields = (
            f"gpu={snapshot.gpu_util_percent:.1f}% "
            f"vram_used={_megabytes(snapshot.vram_used_bytes or 0):.1f}MB "
            f"vram_free={_megabytes(snapshot.vram_free_bytes or 0):.1f}MB "
            f"power={power} temp={temperature}"
        )

    with _print_lock:
        print(f"[PERF] {name:.<28} {_format_duration(wall_seconds)}")
        print(
            f"[TELEMETRY] step={name} cpu_process={process_cpu:.1f}% "
            f"cpu_time={_format_duration(cpu_seconds)} cores=[{cores}] "
            f"rss={_megabytes(snapshot.rss_bytes):.1f}MB "
            f"ram_used={_megabytes(snapshot.system_memory_used_bytes):.1f}/"
            f"{_megabytes(snapshot.system_memory_total_bytes):.1f}MB "
            f"{gpu_fields}"
        )
    _emit_availability_once()


@contextmanager
def profile_block(name: str) -> Generator[None, None, None]:
    """Measure a named block with high-resolution wall and process CPU clocks."""
    started_at = time.perf_counter()
    cpu_started_at = time.process_time()
    try:
        yield
    finally:
        record_step(
            name,
            time.perf_counter() - started_at,
            time.process_time() - cpu_started_at,
        )


def profile_step(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a synchronous pipeline function with standardized profiling."""
    def decorator(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            with profile_block(name):
                return function(*args, **kwargs)

        return wrapped

    return decorator


def record_elapsed(
    name: str,
    started_at: float,
    cpu_started_at: float,
) -> None:
    """Record a milestone measured from caller-owned start clocks."""
    record_step(
        name,
        time.perf_counter() - started_at,
        time.process_time() - cpu_started_at,
    )


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _print_interaction_report(metrics: InteractionMetrics) -> None:
    latency = time.perf_counter() - metrics.started_at
    cpu_seconds = time.process_time() - metrics.cpu_started_at
    cpu_average = cpu_seconds / latency * 100.0 if latency > 0 else 0.0
    snapshots = metrics.snapshot_copy() or [_resource_snapshot(cpu_average)]
    final = snapshots[-1]
    peak_rss = max(snapshot.rss_bytes for snapshot in snapshots)
    gpu_average = _average(
        [
            snapshot.gpu_util_percent
            for snapshot in snapshots
            if snapshot.gpu_util_percent is not None
        ]
    )
    vram_used = final.vram_used_bytes

    with _print_lock:
        print("===== INTERACTION REPORT =====")
        print(f"Total latency: {latency:.2f} s")
        print(f"CPU avg: {cpu_average:.1f}%")
        print(f"RAM: {_megabytes(final.rss_bytes):.1f} MB RSS")
        print(f"RAM peak: {_megabytes(peak_rss):.1f} MB RSS")
        print(
            f"System RAM used: {_megabytes(final.system_memory_used_bytes):.1f} / "
            f"{_megabytes(final.system_memory_total_bytes):.1f} MB"
        )
        print(
            f"GPU util avg: {gpu_average:.1f}%"
            if gpu_average is not None
            else "GPU util avg: n/a"
        )
        print(
            f"VRAM used: {_megabytes(vram_used):.1f} MB"
            if vram_used is not None
            else "VRAM used: n/a"
        )
        print("==============================")


@contextmanager
def profile_interaction() -> Generator[None, None, None]:
    """Collect nested samples and emit a report for one complete interaction."""
    metrics = InteractionMetrics(time.perf_counter(), time.process_time())
    token: Token[InteractionMetrics | None] = _current_interaction.set(metrics)
    stop_sampling = threading.Event()
    sampler = threading.Thread(
        target=_sample_interaction,
        args=(metrics, stop_sampling),
        name="interaction-telemetry",
        daemon=True,
    )
    sampler.start()
    try:
        yield
    finally:
        stop_sampling.set()
        sampler.join()
        record_elapsed("interaction_total", metrics.started_at, metrics.cpu_started_at)
        _print_interaction_report(metrics)
        _current_interaction.reset(token)


def _sample_interaction(
    metrics: InteractionMetrics,
    stop_sampling: threading.Event,
) -> None:
    """Sample resources during long inference waits for meaningful averages."""
    previous_wall = time.perf_counter()
    previous_cpu = time.process_time()
    metrics.add_snapshot(_resource_snapshot(0.0))
    while not stop_sampling.wait(0.2):
        current_wall = time.perf_counter()
        current_cpu = time.process_time()
        wall_delta = current_wall - previous_wall
        cpu_delta = current_cpu - previous_cpu
        process_cpu = cpu_delta / wall_delta * 100.0 if wall_delta > 0 else 0.0
        metrics.add_snapshot(_resource_snapshot(process_cpu))
        previous_wall = current_wall
        previous_cpu = current_cpu
