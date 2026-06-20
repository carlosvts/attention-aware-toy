"""Lifecycle helpers for the Ollama models owned by the application."""

import sys

from .ollama_client import DEFAULT_MODEL, OllamaClient, OllamaError

DEFAULT_VISION_MODEL = "qwen3-vl:2b-instruct"


def _configured_clients(include_vision: bool) -> list[OllamaClient]:
    configurations = [
        ("OLLAMA_MODEL", DEFAULT_MODEL, "OLLAMA_TIMEOUT_SECONDS", 30.0),
    ]
    if include_vision:
        configurations.append(
            (
                "OLLAMA_VISION_MODEL",
                DEFAULT_VISION_MODEL,
                "OLLAMA_VISION_TIMEOUT_SECONDS",
                60.0,
            )
        )
    return [
        OllamaClient.from_environment(
            model_variable=model_variable,
            default_model=default_model,
            timeout_variable=timeout_variable,
            default_timeout=default_timeout,
        )
        for model_variable, default_model, timeout_variable, default_timeout
        in configurations
    ]


def ensure_ollama_ready(include_vision: bool = True) -> None:
    """Exit early unless Ollama and every configured model are available."""
    try:
        clients = _configured_clients(include_vision)
        available_models = clients[0].available_models()
    except OllamaError as error:
        raise SystemExit(f"Erro: {error}") from error

    missing_models = []
    for client in clients:
        accepted_names = {client.model}
        if ":" not in client.model:
            accepted_names.add(f"{client.model}:latest")
        if available_models.isdisjoint(accepted_names):
            missing_models.append(client.model)

    if missing_models:
        commands = "\n".join(f"  ollama pull {model}" for model in missing_models)
        names = ", ".join(missing_models)
        raise SystemExit(
            f"Erro: modelos obrigatórios não encontrados no Ollama: {names}\n"
            f"Instale-os antes de executar o programa:\n{commands}"
        )


def unload_models(include_vision: bool = True) -> None:
    """Unload the application's models without hiding an earlier shutdown error."""
    unloaded: set[tuple[str, str]] = set()
    try:
        clients = _configured_clients(include_vision)
    except OllamaError as error:
        print(f"Aviso durante o encerramento: {error}", file=sys.stderr)
        return

    for client in clients:
        try:
            model_key = (client.base_url, client.model)
            if model_key in unloaded:
                continue
            client.unload()
            unloaded.add(model_key)
        except OllamaError as error:
            print(f"Aviso durante o encerramento: {error}", file=sys.stderr)
