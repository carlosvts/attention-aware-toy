"""Lifecycle helpers for the Ollama models owned by the application."""

import sys

from .ollama_client import DEFAULT_MODEL, OllamaClient, OllamaError


def unload_models(include_vision: bool = True) -> None:
    """Unload the application's models without hiding an earlier shutdown error."""
    configurations = [
        ("OLLAMA_MODEL", DEFAULT_MODEL, "OLLAMA_TIMEOUT_SECONDS", 30.0),
    ]
    if include_vision:
        # Kept here to avoid importing OpenCV through src.vision during shutdown.
        configurations.append(
            (
                "OLLAMA_VISION_MODEL",
                "qwen3-vl:2b-instruct",
                "OLLAMA_VISION_TIMEOUT_SECONDS",
                60.0,
            )
        )

    unloaded: set[tuple[str, str]] = set()
    for configuration in configurations:
        model_variable, default_model, timeout_variable, default_timeout = configuration
        try:
            client = OllamaClient.from_environment(
                model_variable=model_variable,
                default_model=default_model,
                timeout_variable=timeout_variable,
                default_timeout=default_timeout,
            )
            model_key = (client.base_url, client.model)
            if model_key in unloaded:
                continue
            client.unload()
            unloaded.add(model_key)
        except OllamaError as error:
            print(f"Aviso durante o encerramento: {error}", file=sys.stderr)
