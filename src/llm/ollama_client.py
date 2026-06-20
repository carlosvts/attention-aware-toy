"""Small HTTP client for a local Ollama server."""

from dataclasses import dataclass
import json
import os
import time
from typing import Any

import requests

from src.profiling import record_elapsed

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_TIMEOUT_SECONDS = 30.0


class OllamaError(RuntimeError):
    """Raised when Ollama cannot produce a valid response."""


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
            },
        }

        started_at = time.perf_counter()
        cpu_started_at = time.process_time()
        response: requests.Response | None = None
        content_parts: list[str] = []
        first_token_recorded = False
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
        return content

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
