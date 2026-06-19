"""Generate an HRI response with Ollama and a safe local fallback."""

import sys

from src.attention import AttentionState

from .ollama_client import OllamaClient, OllamaError

FALLBACK_RESPONSE = (
    "[FAALBACK] Oi, percebi que você está olhando para mim. Posso ajudar em algo?"
)
SYSTEM_PROMPT = "Você é um robô social assistivo."


def generate_response(
    scene_description: str,
    attention_state: AttentionState = AttentionState.ATTENDING,
    gaze_duration: float = 0.0,
) -> str:
    """Generate a short response from a scene description using local Ollama."""
    if not scene_description.strip():
        raise ValueError("Scene description cannot be empty")

    user_prompt = f"""Estado de atenção do usuário: {attention_state.name}
Duração do olhar: {max(0.0, gaze_duration):.1f}s
Cena observada: {scene_description.strip()}

Responda de forma curta, amigável e contextual.
Não mencione detalhes incertos da imagem."""
    try:
        return OllamaClient.from_environment().chat(SYSTEM_PROMPT, user_prompt)
    except OllamaError as error:
        print(f"Aviso: {error} Usando resposta de fallback.", file=sys.stderr)
        return FALLBACK_RESPONSE
