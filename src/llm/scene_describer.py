"""Describe webcam frames with a local Qwen vision-language model."""

import base64
import sys

import cv2
import numpy as np
from numpy.typing import NDArray

from src.llm.ollama_client import OllamaClient, OllamaError, OllamaGPUError
from src.profiling import profile_block, profile_step

MAX_DIMENSION = 512
DEFAULT_VISION_MODEL = "qwen3-vl:2b"
DEFAULT_VISION_TIMEOUT_SECONDS = 60.0
FALLBACK_DESCRIPTION = "A person is in front of the camera"
SYSTEM_PROMPT = """
    You are the perception module of a social robot.

    Your task is NOT to write image captions.

    Your task is to extract concrete visual facts that can be used by another language model to generate a short spoken reaction.

    Rules:

    * Describe only directly visible information.
    * Do not speculate.
    * If information is unclear, write "uncertain".
    * If information is absent, write "none".
    * Focus on people, actions, gestures, posture, objects being used, and attention toward the camera.
    * Prefer concrete observations over scene descriptions.
    * Keep all fields concise.
    * Return ONLY the fields below.
    * Do not add explanations, comments, markdown, or extra text.

    Output format:

    person: yes/no
    people_count: number
    attention_target: camera_or_robot / near_camera / away / unclear / none
    main_subject: short phrase
    salient_action: short phrase
    held_objects: comma-separated objects or none
    visible_objects: comma-separated important objects or none
    gesture_or_pose: short phrase
    facial_cues: visible facial cues only or none
    setting: short phrase
    robot_reaction_hint: one concrete visual detail that would make a good robot response

    Guidelines:

    * attention_target should be camera_or_robot when a person appears to be looking directly at the camera.
    * held_objects should contain only objects actively held or used.
    * visible_objects should contain only the most relevant objects.
    * robot_reaction_hint should identify the most salient observable detail for the robot to mention.

    Example:

    person: yes
    people_count: 2
    attention_target: camera_or_robot
    main_subject: two people near the camera
    salient_action: posing together
    held_objects: none
    visible_objects: headphones, bed, wall fan
    gesture_or_pose: one person waving
    setting: indoor room
    robot_reaction_hint: one person is waving while wearing headphones
"""

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
                user_prompt="""
                    Analyze the current webcam frame for a social robot.
                    Return only the structured fields requested by the system prompt.
                    Focus on concrete visual details that can be safely mentioned by the robot.
                """,
                images=[image],
                temperature=0.1,
                num_predict=250,
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
