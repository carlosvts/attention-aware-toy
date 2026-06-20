"""Low-overhead profiling and resource telemetry for the experimental pipeline."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, ParamSpec, TypeVar
from uuid import uuid4

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

    @property
    def available(self) -> bool:
        return bool(self._handles)

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


_log_lock = threading.Lock()
_current_interaction: ContextVar[InteractionMetrics | None] = ContextVar(
    "current_interaction", default=None
)
_process = psutil.Process(os.getpid()) if psutil is not None else None
_nvidia = _NvidiaMonitor()
_availability_reported = False
_run_id = str(uuid4())
_project_root = Path(__file__).resolve().parents[1]
_log_directory = Path(os.getenv("ATTENTION_LOG_DIR", _project_root / "logs"))
_log_started_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
_log_path = _log_directory / f"performance-{_log_started_at}-{os.getpid()}.jsonl"


def get_performance_log_path() -> Path:
    """Return the JSONL file used by this process."""
    return _log_path


def nvidia_gpu_available() -> bool:
    """Return whether NVML can see at least one local NVIDIA GPU."""
    return _nvidia.available


def _component_for_step(name: str) -> str:
    if name.startswith("qwen_vlm"):
        return "vlm"
    if name.startswith("qwen_llm"):
        return "llm"
    return "pipeline"


def _write_event(
    event: str,
    component: str,
    data: dict[str, Any],
) -> None:
    """Append one self-contained, machine-readable JSON object to the run log."""
    payload = {
        "schema_version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": _run_id,
        "event": event,
        "component": component,
        **data,
    }
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with _log_lock:
        _log_directory.mkdir(parents=True, exist_ok=True)
        with _log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{line}\n")


def record_model_metrics(
    component: str,
    model: str,
    metrics: dict[str, Any],
) -> None:
    """Record Ollama-native metrics separately for the VLM or LLM."""
    _write_event(
        "model_metrics",
        component,
        {"model": model, "metrics": metrics},
    )


def record_ollama_lifecycle(
    message: str,
    component: str,
    model: str,
    resident_models: list[dict[str, Any]] | None,
    **details: Any,
) -> None:
    """Record explicit Ollama load, reuse, and unload lifecycle evidence."""
    _write_event(
        "ollama_lifecycle",
        component,
        {
            "message": message,
            "model": model,
            "resident_models": resident_models,
            "details": details,
        },
    )


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
        _write_event(
            "telemetry_availability",
            "system",
            {"available": False, "messages": messages},
        )


def record_step(name: str, wall_seconds: float, cpu_seconds: float = 0.0) -> None:
    """Emit one standardized timing and resource sample."""
    process_cpu = (
        max(0.0, cpu_seconds) / wall_seconds * 100.0 if wall_seconds > 0 else 0.0
    )
    snapshot = _resource_snapshot(process_cpu)
    interaction = _current_interaction.get()
    if interaction is not None:
        interaction.add_snapshot(snapshot)

    _write_event(
        "step_metric",
        _component_for_step(name),
        {
            "step": name,
            "metrics": {
                "wall_seconds": wall_seconds,
                "cpu_seconds": cpu_seconds,
                "process_cpu_percent": process_cpu,
                "per_core_cpu_percent": snapshot.per_core_cpu_percent,
                "rss_bytes": snapshot.rss_bytes,
                "system_memory_used_bytes": snapshot.system_memory_used_bytes,
                "system_memory_total_bytes": snapshot.system_memory_total_bytes,
                "gpu_util_percent": snapshot.gpu_util_percent,
                "vram_used_bytes": snapshot.vram_used_bytes,
                "vram_free_bytes": snapshot.vram_free_bytes,
                "gpu_power_watts": snapshot.gpu_power_watts,
                "gpu_temperature_c": snapshot.gpu_temperature_c,
            },
        },
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


def _record_interaction_report(metrics: InteractionMetrics) -> None:
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
    _write_event(
        "interaction_report",
        "interaction",
        {
            "metrics": {
                "total_latency_seconds": latency,
                "cpu_average_percent": cpu_average,
                "rss_bytes": final.rss_bytes,
                "peak_rss_bytes": peak_rss,
                "system_memory_used_bytes": final.system_memory_used_bytes,
                "system_memory_total_bytes": final.system_memory_total_bytes,
                "gpu_util_average_percent": gpu_average,
                "vram_used_bytes": final.vram_used_bytes,
                "sample_count": len(snapshots),
            }
        },
    )


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
        _record_interaction_report(metrics)
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
