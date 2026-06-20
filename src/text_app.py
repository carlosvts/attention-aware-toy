"""Terminal-only entry point for testing the local language model."""

from src.llm import ensure_ollama_ready, generate_response, unload_models


def run() -> None:
    """Read a scene description and print the generated robot response."""
    ensure_ollama_ready(include_vision=False)
    try:
        scene_description = input("Descrição da cena: ").strip()
        if not scene_description:
            print("A descrição da cena não pode ser vazia.")
            return
        print(f"Resposta: {generate_response(scene_description)}")
    finally:
        unload_models(include_vision=False)


if __name__ == "__main__":
    run()
