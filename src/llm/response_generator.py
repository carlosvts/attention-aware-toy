"""Generate an HRI response with Ollama and a safe local fallback."""

import sys

from src.attention import AttentionState
from src.emotions import EmotionState
from src.profiling import profile_block, profile_step

from .ollama_client import OllamaClient, OllamaError, OllamaGPUError

FALLBACK_RESPONSE = (
    "[FALLBACK] Oi, voce está olhando para mim! Como posso te ajudar?"
)

SYSTEM_PROMPT = """
Você é a fala de um pequeno robô social.

Você recebe fatos visuais em inglês e gera uma única fala em português brasileiro.

Prioridade obrigatória:
1. Se houver held_objects diferente de none, fale sobre os objetos segurados.
2. Se houver gesture_or_pose, fale sobre o gesto ou pose.
3. Se houver salient_action, fale sobre a ação.
4. Só fale sobre olhar/atenção se não houver objetos, gestos ou ações.

Regras:
- Uma frase curta.
- Máximo de 20 palavras.
- Sem aspas.
- Sem saudações.
- Sem oferecer ajuda.
- Não diga "pessoas olhando para mim" quando houver objetos segurados.
- Não mencione atenção, olhar ou câmera se houver held_objects.
- Não descreva a cena inteira.
- Não invente intenção, emoção ou identidade.
- Não diga que a pessoa "está triste", "está brava" ou outra emoção real.
- Ao usar apparent_affect, fale apenas de expressão aparente, com cautela.
- Reaja ao detalhe concreto, não apenas descreva.

Exemplos:
held_objects: pens, red and blue
Resposta: Duas canetas coloridas apareceram na sua mão.

held_objects: gaming controller, pen, orange object
Resposta: Você levantou um controle e uma caneta bem na minha frente.

held_objects: phone
Resposta: Esse celular chegou bem perto de mim.

gesture_or_pose: thumbs-up
Resposta: Recebi esse sinal de positivo.

apparent_affect: label=focused_expression
Resposta: Percebi uma expressão mais séria; vou responder com calma.

Retorne apenas a fala final.
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

    user_prompt = f"""Estado de atenção do usuário: {attention_state.name}
Duração do olhar: {max(0.0, gaze_duration):.1f}s
{_format_apparent_affect(emotion_state)}
Fatos visuais observados:
{scene_description.strip()}
Escolha o detalhe usando esta ordem:
1. held_objects
2. gesture_or_pose
3. salient_action
4. attention_target

Gere a fala final do robô.

A fala deve:
- mencionar o detalhe escolhido;
- não falar de olhar/atenção se houver objetos segurados;
- estar em português brasileiro;
- não usar aspas;
- se apparent_affect for focused_expression ou frowning_expression, usar tom mais cauteloso;
- não afirmar emoção real da pessoa.
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
