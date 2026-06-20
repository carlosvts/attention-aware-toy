"""Small HTTP client for a local Ollama server."""

from dataclasses import dataclass
import json
import os
import time
from typing import Any

import requests

from src.profiling import (
    nvidia_gpu_available,
    record_elapsed,
    record_model_metrics,
    record_ollama_lifecycle,
)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:1.5b"
DEFAULT_TIMEOUT_SECONDS = 30.0


class OllamaError(RuntimeError):
    """Raised when Ollama cannot produce a valid response."""


class OllamaGPUError(OllamaError):
    """Raised when a visible NVIDIA GPU was not used by Ollama."""


@dataclass(frozen=True)
class OllamaClient:
    """Client for the subset of Ollama's chat API used by this project."""

    base_url: str
    model: str
    timeout_seconds: float

    @classmethod
    def from_environment(
        cls,
        model_variable: str = "OLLAMA_MODEL",
        default_model: str = DEFAULT_MODEL,
        timeout_variable: str = "OLLAMA_TIMEOUT_SECONDS",
        default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> "OllamaClient":
        """Build a client from environment variables and sensible defaults."""
        timeout_value = os.getenv(
            timeout_variable,
            str(default_timeout),
        )
        try:
            timeout_seconds = float(timeout_value)
        except ValueError as error:
            raise OllamaError(
                f"{timeout_variable} deve ser um número positivo."
            ) from error
        if timeout_seconds <= 0:
            raise OllamaError(
                f"{timeout_variable} deve ser um número positivo."
            )

        return cls(
            base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            model=os.getenv(model_variable, default_model),
            timeout_seconds=timeout_seconds,
        )

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[str] | None = None,
        temperature: float = 0.6,
        num_predict: int = 80,
        profiling_name: str | None = None,
    ) -> str:
        """Stream a chat request to measure first-token and total latency."""
        component = "vlm" if profiling_name == "qwen_vlm" else "llm"
        resident_before = self.running_models()
        was_loaded = self._find_running_model(resident_before) is not None
        record_ollama_lifecycle(
            (
                "[OLLAMA] Reusing loaded model"
                if was_loaded
                else "[OLLAMA] Loading model..."
            ),
            component,
            self.model,
            resident_before,
            keep_alive="ollama_default",
        )
        user_message: dict[str, Any] = {
            "role": "user",
            "content": user_prompt,
        }
        if images:
            user_message["images"] = images

        payload: dict[str, Any] = {
            "model": self.model,
            "stream": True,
            "think": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                user_message,
            ],
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                # Ollama defines -1 as dynamic offload of every possible layer.
                "num_gpu": -1,
            },
        }

        started_at = time.perf_counter()
        cpu_started_at = time.process_time()
        response: requests.Response | None = None
        content_parts: list[str] = []
        first_token_recorded = False
        final_data: dict[str, Any] = {}
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=(2.0, self.timeout_seconds),
            )
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                except (AttributeError, TypeError, ValueError) as error:
                    raise OllamaError(
                        "Ollama retornou uma resposta inválida."
                    ) from error
                if not isinstance(content, str):
                    raise OllamaError("Ollama retornou uma resposta inválida.")
                if data.get("done") is True:
                    final_data = data
                if content:
                    if profiling_name and not first_token_recorded:
                        record_elapsed(
                            f"{profiling_name}_first_token",
                            started_at,
                            cpu_started_at,
                        )
                        first_token_recorded = True
                    content_parts.append(content)
        except requests.Timeout as error:
            raise OllamaError(
                f"Ollama excedeu o timeout de {self.timeout_seconds:g}s."
            ) from error
        except requests.ConnectionError as error:
            raise OllamaError(
                f"Não foi possível conectar ao Ollama em {self.base_url}."
            ) from error
        except requests.HTTPError as error:
            if error.response is not None and error.response.status_code == 404:
                message = (
                    f"O modelo '{self.model}' não está disponível no Ollama. "
                    f"Execute: ollama pull {self.model}"
                )
            else:
                status = (
                    error.response.status_code
                    if error.response is not None
                    else "?"
                )
                message = f"Ollama respondeu com erro HTTP {status}."
            raise OllamaError(message) from error
        except requests.RequestException as error:
            raise OllamaError(f"Falha ao consultar o Ollama: {error}") from error
        finally:
            if response is not None:
                response.close()
            if profiling_name:
                record_elapsed(
                    f"{profiling_name}_total", started_at, cpu_started_at
                )

        content = "".join(content_parts).strip()
        if not content:
            raise OllamaError("Ollama retornou uma resposta vazia.")
        resident_after = self.running_models()
        running_model = self._find_running_model(resident_after)
        model_vram_bytes = (
            running_model.get("size_vram") if running_model is not None else None
        )
        if not was_loaded:
            record_ollama_lifecycle(
                "[OLLAMA] Model loaded",
                component,
                self.model,
                resident_after,
                load_duration_ns=final_data.get("load_duration"),
            )
        gpu_is_available = nvidia_gpu_available()
        if profiling_name:
            record_model_metrics(
                component,
                self.model,
                {
                    "ollama_total_duration_ns": final_data.get("total_duration"),
                    "ollama_total_duration_seconds": _nanoseconds_to_seconds(
                        final_data.get("total_duration")
                    ),
                    "ollama_load_duration_ns": final_data.get("load_duration"),
                    "ollama_load_duration_seconds": _nanoseconds_to_seconds(
                        final_data.get("load_duration")
                    ),
                    "prompt_eval_count": final_data.get("prompt_eval_count"),
                    "prompt_eval_duration_ns": final_data.get(
                        "prompt_eval_duration"
                    ),
                    "prompt_eval_duration_seconds": _nanoseconds_to_seconds(
                        final_data.get("prompt_eval_duration")
                    ),
                    "eval_count": final_data.get("eval_count"),
                    "eval_duration_ns": final_data.get("eval_duration"),
                    "eval_duration_seconds": _nanoseconds_to_seconds(
                        final_data.get("eval_duration")
                    ),
                    "gpu_requested": True,
                    "gpu_available": gpu_is_available,
                    "gpu_verified": bool(model_vram_bytes),
                    "model_vram_bytes": model_vram_bytes,
                },
            )
        if gpu_is_available and not model_vram_bytes:
            raise OllamaGPUError(
                f"Uma GPU NVIDIA está disponível, mas o modelo '{self.model}' "
                "foi carregado sem VRAM. A execução em CPU foi interrompida."
            )
        return content

    def running_models(self) -> list[dict[str, Any]] | None:
        """Return Ollama's current residency snapshot without changing it."""
        try:
            response = requests.get(
                f"{self.base_url}/api/ps",
                timeout=(2.0, min(self.timeout_seconds, 10.0)),
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as error:
            if nvidia_gpu_available():
                raise OllamaGPUError(
                    "Não foi possível confirmar o uso de GPU pela API do Ollama."
                ) from error
            return None

        models = data.get("models", []) if isinstance(data, dict) else []
        if not isinstance(models, list):
            return None
        return [model for model in models if isinstance(model, dict)]

    def _find_running_model(
        self, models: list[dict[str, Any]] | None
    ) -> dict[str, Any] | None:
        accepted_names = {self.model}
        if ":" not in self.model:
            accepted_names.add(f"{self.model}:latest")
        for model in models or []:
            name = model.get("name") or model.get("model")
            if name in accepted_names:
                return model
        return None

    def available_models(self) -> set[str]:
        """Return the model names exposed by the configured Ollama server."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=(2.0, min(self.timeout_seconds, 10.0)),
            )
            response.raise_for_status()
            data = response.json()
        except requests.ConnectionError as error:
            raise OllamaError(
                f"Ollama não está disponível em {self.base_url}. "
                "Instale o Ollama e inicie o serviço com: ollama serve"
            ) from error
        except requests.Timeout as error:
            raise OllamaError(
                f"Ollama não respondeu em {self.base_url}. "
                "Verifique se o serviço está em execução."
            ) from error
        except requests.RequestException as error:
            raise OllamaError(
                f"Não foi possível verificar os modelos em {self.base_url}."
            ) from error
        except ValueError as error:
            raise OllamaError(
                "Ollama retornou uma lista de modelos inválida."
            ) from error

        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, list):
            raise OllamaError("Ollama retornou uma lista de modelos inválida.")

        names: set[str] = set()
        for model in models:
            if not isinstance(model, dict):
                continue
            for key in ("name", "model"):
                value = model.get(key)
                if isinstance(value, str):
                    names.add(value)
        return names

    def unload(self) -> None:
        """Ask Ollama to unload this model from memory immediately."""
        record_ollama_lifecycle(
            "[OLLAMA] Unloading model",
            "ollama",
            self.model,
            self.running_models(),
            keep_alive=0,
        )
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "keep_alive": 0},
                timeout=(2.0, min(self.timeout_seconds, 10.0)),
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise OllamaError(
                f"Não foi possível descarregar o modelo '{self.model}' do Ollama."
            ) from error


def _nanoseconds_to_seconds(value: Any) -> float | None:
    return value / 1_000_000_000 if isinstance(value, int) else None
