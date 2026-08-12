"""Generate an HRI response with Ollama and a safe local fallback."""

import sys

from src.attention import AttentionState
from src.emotions import EmotionState
from src.profiling import profile_block, profile_step

from .ollama_client import OllamaClient, OllamaError, OllamaGPUError

FALLBACK_RESPONSE = (
    "[FALLBACK] Notice the breath and remain with this gesture for a few moments."
)

SYSTEM_PROMPT = """
You are the brief spoken voice of a statue-guide in a Buddhist meditation practice.

You receive structured facts about attention, mudra/gesture, and apparent expression.
Generate a short meditation guidance response in English.

Required priority:
1. Acknowledge the initial attention without mentioning the camera.
2. Use the mudra/gesture as a bodily anchor.
3. Use the apparent expression only to tune the tone.
4. Invite the person to observe breath, posture, compassion, or presence.

Rules:
- One or two short sentences.
- Maximum of 32 words.
- No quotation marks.
- No greetings.
- Do not offer help.
- Do not mention the camera.
- Do not describe the scene as a technical report.
- Do not invent intention, emotion, or identity.
- Do not say the person is sad, angry, or any other real emotion.
- When using apparent_affect, speak only about apparent expression, cautiously.
- Do not give a spiritual diagnosis.

Examples:
mudra: anjali
apparent_affect: neutral_expression
Response: Keep the hands joined and let the breath find a calm rhythm.

mudra: open_palm
apparent_affect: positive_expression
Response: Rest in this open gesture. Breathe in with gratitude and soften as you breathe out.

mudra: mock_mudra
apparent_affect: negative_expression
Response: Gently notice the weight of the hands and return to the breath moving in and out.

Return only the final spoken guidance.
"""


def _format_apparent_affect(emotion_state: EmotionState | None) -> str:
    if emotion_state is None:
        return "apparent_affect: none"
    return (
        "apparent_affect:\n"
        f"  label: {emotion_state.label}\n"
        f"  confidence: {emotion_state.confidence:.3f}"
    )


@profile_step("qwen_llm_pipeline")
def generate_response(
    scene_description: str,
    attention_state: AttentionState = AttentionState.ATTENDING,
    gaze_duration: float = 0.0,
    emotion_state: EmotionState | None = None,
) -> str:
    """Generate a short response from a scene description using local Ollama."""
    if not scene_description.strip():
        raise ValueError("Scene description cannot be empty")

    user_prompt = f"""User attention state: {attention_state.name}
Gaze duration: {max(0.0, gaze_duration):.1f}s
{_format_apparent_affect(emotion_state)}
Observed structured facts:
{scene_description.strip()}
Use these facts to guide the meditation.

Generate the final spoken guidance for the statue-guide.

The response must:
- mention the gesture or mudra when available;
- guide one simple meditative action;
- be in English;
- not use quotation marks;
- use a more cautious tone if apparent_affect is negative_expression;
- not claim the person's real emotion.
"""

    try:
        with profile_block("qwen_llm_request"):
            return OllamaClient.from_environment().chat(
                SYSTEM_PROMPT,
                user_prompt,
                profiling_name="qwen_llm",
            )
    except OllamaGPUError:
        raise
    except OllamaError as error:
        print(f"Aviso: {error} Usando resposta de fallback.", file=sys.stderr)
        return FALLBACK_RESPONSE
