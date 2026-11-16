# SPDX-License-Identifier: Apache-2.0
"""Cosmos3.5 instruction-editing request contract.

Mirrors the cosmos-framework inference API: ``image2image`` (an image input and
a single output frame) is always an image edit, and ``video2video`` becomes a
video edit when ``editing_instruction`` is set. The instruction replaces the
prompt. Defaults come from ``defaults/image2image`` and
``defaults/video_editing``; the output size follows the source.
"""

import json
from typing import Any

import msgspec

EDIT_MODE_IMAGE = "image"
EDIT_MODE_VIDEO = "video"

# Verbatim from cosmos-framework (``qwen3_vl._SYSTEM_PROMPT_IMAGE_EDITING`` and
# ``preprocessing/editing.SYSTEM_PROMPT_VIDEO_EDITING``).
COSMOS3_IMAGE_EDIT_SYSTEM_PROMPT = (
    "You are a helpful assistant who will edit images based on the user's instructions."
)
COSMOS3_VIDEO_EDIT_SYSTEM_PROMPT = (
    "You are a helpful assistant who will edit videos based on the user's instructions."
)


class Cosmos3EditDefaults(msgspec.Struct, frozen=True):
    num_inference_steps: int
    guidance_scale: float
    flow_shift: float
    fps: int
    # Video edits: a cap; the output length is the largest 4N+1 the source covers.
    num_frames: int
    max_pixels: int
    padding_constant: int


# cosmos-framework ``defaults/image2image/sample_args.json``.
COSMOS3_IMAGE_EDIT_DEFAULTS = Cosmos3EditDefaults(
    num_inference_steps=50,
    guidance_scale=4.0,
    flow_shift=3.0,
    fps=24,
    num_frames=1,
    max_pixels=1024 * 1024,
    padding_constant=32,
)
# cosmos-framework ``defaults/video_editing/sample_args.json``.
COSMOS3_VIDEO_EDIT_DEFAULTS = Cosmos3EditDefaults(
    num_inference_steps=35,
    guidance_scale=7.0,
    flow_shift=5.0,
    fps=16,
    num_frames=81,
    max_pixels=512 * 512,
    padding_constant=16,
)

COSMOS3_EDIT_DEFAULTS = {
    EDIT_MODE_IMAGE: COSMOS3_IMAGE_EDIT_DEFAULTS,
    EDIT_MODE_VIDEO: COSMOS3_VIDEO_EDIT_DEFAULTS,
}

# Request fields that only an edit request may set.
COSMOS3_EDIT_ONLY_FIELDS = (
    "editing_instruction",
    "edited_first_frame_path",
    "system_prompt",
    "max_pixels",
    "padding_constant",
)

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def normalize_edit_fields(params: Any) -> None:
    """Coerce transport-decoded values to the field types.

    Multipart forms JSON-decode their values, so an instruction that is itself a
    JSON object arrives as a dict and a number as an int.
    """
    for name in ("editing_instruction", "edited_first_frame_path", "system_prompt"):
        value = getattr(params, name)
        if value is not None and not isinstance(value, str):
            setattr(params, name, json.dumps(value))
    for name in ("max_pixels", "padding_constant"):
        value = getattr(params, name)
        if isinstance(value, str):
            setattr(params, name, int(value) if value.strip() else None)


def resolve_edit_mode(
    *,
    is_image_task: bool,
    has_image: bool,
    has_video: bool,
    editing_instruction: str | None,
) -> str | None:
    """The edit mode a Cosmos3 request selects, or ``None`` for generation."""
    if is_image_task and has_image and not has_video:
        return EDIT_MODE_IMAGE
    if editing_instruction is not None and has_video:
        return EDIT_MODE_VIDEO
    return None


def resolve_edit_prompt(prompt: Any, editing_instruction: str | None) -> str:
    """The caption of an edit request: the stripped instruction, else the prompt.

    As in cosmos-framework, ``editing_instruction`` must be non-empty and is
    exclusive with a different prompt. An empty prompt is accepted alongside it
    because the HTTP video endpoint always carries a ``prompt`` field.
    """
    if isinstance(prompt, (list, tuple)):
        raise ValueError(
            "Cosmos3 editing takes a single instruction, not a prompt list"
        )
    if editing_instruction is None:
        if prompt is None or not str(prompt).strip():
            raise ValueError("Cosmos3 editing requires a non-empty instruction")
        return str(prompt)
    instruction = editing_instruction.strip()
    if not instruction:
        raise ValueError("editing_instruction must be non-empty")
    if prompt not in (None, "", editing_instruction, instruction):
        raise ValueError("editing_instruction is mutually exclusive with prompt")
    return instruction


def validate_edit_request(
    *,
    edit_mode: str | None,
    params: Any,
    control_paths: list[str],
    size_multiple: int,
) -> None:
    """Reject inputs the cosmos-framework editing routes do not accept."""
    if edit_mode is None:
        given = [
            name
            for name in COSMOS3_EDIT_ONLY_FIELDS
            if getattr(params, name) is not None
        ]
        if given:
            raise ValueError(
                f"{', '.join(given)} require(s) an edit request: an input image with "
                "num_frames=1 (image editing) or video_path with editing_instruction "
                "(video editing)"
            )
        return
    unsupported = []
    if control_paths:
        unsupported.append("control_path")
    if params.action_mode is not None:
        unsupported.append("action_mode")
    if float(params.sound_duration or 0.0) > 0.0:
        unsupported.append("sound generation")
    if params.control_guidance not in (None, 1.0):
        unsupported.append("control_guidance")
    if params.condition_frame_indexes:
        unsupported.append("condition_frame_indexes")
    if edit_mode == EDIT_MODE_IMAGE:
        if params.edited_first_frame_path is not None:
            unsupported.append("edited_first_frame_path")
        image_path = params.image_path
        if isinstance(image_path, (list, tuple)) and len(image_path) != 1:
            raise ValueError("Cosmos3 image editing takes exactly one source image")
    else:
        if params.image_path:
            unsupported.append("image input")
        if params.system_prompt is not None:
            # cosmos-framework always uses the video-edit system prompt.
            unsupported.append("system_prompt")
        first = params.edited_first_frame_path
        if first is not None and not str(first).split("?", 1)[0].lower().endswith(
            _IMAGE_EXTENSIONS
        ):
            raise ValueError("edited_first_frame_path must be an image")
    if unsupported:
        kind = "image" if edit_mode == EDIT_MODE_IMAGE else "video"
        raise ValueError(
            f"Cosmos3 {kind} editing does not support: {', '.join(unsupported)}"
        )
    for name in ("max_pixels", "padding_constant"):
        value = getattr(params, name)
        if value is not None and int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if params.padding_constant is not None and params.padding_constant % size_multiple:
        # The source-derived size is a multiple of padding_constant and must sit
        # on the latent grid; cosmos-framework fails at its VAE stride check.
        raise ValueError(f"padding_constant must be a multiple of {size_multiple}")
    if params.max_pixels is not None and params.padding_constant is not None:
        if params.max_pixels < params.padding_constant**2:
            raise ValueError("max_pixels must be at least padding_constant squared")
    if (params.width is None) != (params.height is None):
        raise ValueError(
            "Cosmos3 editing takes either both width and height or neither "
            "(the output then follows the source size)"
        )
