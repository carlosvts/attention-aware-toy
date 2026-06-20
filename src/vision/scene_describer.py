"""Describe webcam frames with a local Qwen vision-language model."""

import base64
import sys

import cv2
import numpy as np
from numpy.typing import NDArray

from src.llm.ollama_client import OllamaClient, OllamaError, OllamaGPUError
from src.profiling import profile_block, profile_step

MAX_DIMENSION = 512
DEFAULT_VISION_MODEL = "openbmb/minicpm-v4.6"
DEFAULT_VISION_TIMEOUT_SECONDS = 60.0
FALLBACK_DESCRIPTION = "A person is in front of the camera"
SYSTEM_PROMPT = (
    "You are the vision module of a social robot. "
    "Describe only visible facts from the image. "
    "Do not infer identity, emotion, intention, or relationship. "
    "Return only the requested fields in English."
    "person: yes/no"
    "people_count: number"
    "objects: comma-separated visible objects or none"
    "gesture_or_pose: short phrase"
    "scene: short phrase"
)

def _encode_frame(frame: NDArray[np.uint8]) -> str:
    """Resize and encode a BGR frame as base64 JPEG for Ollama."""
    with profile_block("vl_image_preprocess"):
        height, width = frame.shape[:2]
        largest_dimension = max(height, width)
        if largest_dimension > MAX_DIMENSION:
            scale = MAX_DIMENSION / largest_dimension
            frame = cv2.resize(
                frame,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )

    with profile_block("image_encode"):
        encoded, jpeg = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )
    if not encoded:
        raise ValueError("Não foi possível codificar o frame da webcam.")
    with profile_block("image_base64"):
        return base64.b64encode(jpeg.tobytes()).decode("ascii")


@profile_step("qwen_vlm_pipeline")
def describe_scene(frame: NDArray[np.uint8]) -> str:
    """Describe a captured BGR frame using Qwen VLM with a safe fallback."""
    if frame.size == 0:
        raise ValueError("Cannot describe an empty frame")

    try:
        image = _encode_frame(frame)
        client = OllamaClient.from_environment(
            model_variable="OLLAMA_VISION_MODEL",
            default_model=DEFAULT_VISION_MODEL,
            timeout_variable="OLLAMA_VISION_TIMEOUT_SECONDS",
            default_timeout=DEFAULT_VISION_TIMEOUT_SECONDS,
        )
        with profile_block("qwen_vlm_request"):
            return client.chat(
                system_prompt=SYSTEM_PROMPT,
                user_prompt="Descreva a cena atual para orientar o robô social.",
                images=[image],
                temperature=0.2,
                num_predict=100,
                profiling_name="qwen_vlm",
            )
    except OllamaGPUError:
        raise
    except (OllamaError, ValueError) as error:
        print(
            f"Aviso: falha ao descrever a cena: {error} Usando fallback.",
            file=sys.stderr,
        )
        return FALLBACK_DESCRIPTION
