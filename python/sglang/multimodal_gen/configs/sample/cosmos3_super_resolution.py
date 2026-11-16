# SPDX-License-Identifier: Apache-2.0
"""Cosmos3.5 LR-to-HR super-resolution request contract.

A super-resolution checkpoint declares LR-to-HR routes under the root
``config.json`` ``model.config.high_resolution`` block. A request supplies LR
media (``lr_vision_path``) and asks for an HR canvas (``width`` x ``height``)
that one declared route produces. The LR is packed as a clean vision item ahead
of the generated HR item (``sequence_lr_conditioning``).

Mirrors cosmos-framework ``configs/base/defaults/high_resolution.py``,
``inference/config/hf.py`` (``_published_high_resolution``) and
``inference/preprocessing/omni_requests.py`` (``resolve_high_resolution_capability``,
``_validate_lr_to_high_resolution_request``). No torch here: the sampling
params resolve requests in the HTTP server process.
"""

import functools
import json
import math
import os
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlparse

import msgspec

SR_WORKFLOW_NONE = "none"
SR_WORKFLOW_SEQUENCE_LR = "sequence_lr_conditioning"
SR_WORKFLOWS = frozenset({SR_WORKFLOW_NONE, SR_WORKFLOW_SEQUENCE_LR})

SR_IMAGE_MODALITIES = ("text2image", "image2image")
SR_VIDEO_MODALITIES = ("text2video", "image2video", "video2video")
SR_MODALITIES = frozenset(SR_IMAGE_MODALITIES + SR_VIDEO_MODALITIES)

# cosmos-framework inference/preprocessing/requests.py IMAGE_EXTENSIONS / VIDEO_EXTENSIONS.
SR_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
SR_VIDEO_EXTENSIONS = frozenset({".mp4"})

# Request fields that only an LR-to-HR request may carry.
SR_REQUEST_FIELDS = (
    "lr_vision_path",
    "lr_resolution",
    "sr_modality",
    "resolution_adapter_policy",
    "resolution_adapter_output_size",
)

VAE_SPATIAL_COMPRESSION = 16
VAE_TEMPORAL_COMPRESSION = 4
DIT_PATCH_SPATIAL = 2
# Text split of an empty caption: EOS + start-of-generation; the Qwen tokenizer has no BOS.
EMPTY_TEXT_SPLIT_TOKENS = 2
# cosmos-framework MoTPipelineConfig default when a checkpoint does not publish its budget.
DEFAULT_MAX_NUM_TOKENS_AFTER_PACKING = 13312

# cosmos-framework inference/preprocessing/resolution_presets.json, (width, height).
VIDEO_RES_SIZE_INFO: dict[str, dict[str, tuple[int, int]]] = {
    "256": {
        "1,1": (256, 256),
        "4,3": (320, 256),
        "3,4": (256, 320),
        "16,9": (320, 192),
        "9,16": (192, 320),
    },
    "272": {
        "1,1": (352, 352),
        "4,3": (416, 304),
        "3,4": (304, 416),
        "16,9": (480, 272),
        "9,16": (272, 480),
    },
    "544": {
        "1,1": (720, 720),
        "4,3": (832, 624),
        "3,4": (624, 832),
        "16,9": (960, 544),
        "9,16": (544, 960),
    },
    "480": {
        "1,1": (640, 640),
        "4,3": (736, 544),
        "3,4": (544, 736),
        "16,9": (832, 480),
        "9,16": (480, 832),
        "2,3": (512, 768),
    },
    "704": {
        "1,1": (960, 960),
        "4,3": (1088, 832),
        "3,4": (832, 1088),
        "16,9": (1280, 704),
        "9,16": (704, 1280),
    },
    "720": {
        "1,1": (960, 960),
        "4,3": (1104, 832),
        "3,4": (832, 1104),
        "16,9": (1280, 720),
        "9,16": (720, 1280),
    },
    "768": {
        "1,1": (1024, 1024),
        "4,3": (1184, 880),
        "3,4": (880, 1184),
        "16,9": (1360, 768),
        "9,16": (768, 1360),
    },
    "1080": {
        "1,1": (1440, 1440),
        "4,3": (1664, 1248),
        "3,4": (1248, 1664),
        "16,9": (1920, 1088),
        "9,16": (1088, 1920),
    },
    "1280": {
        "1,1": (1712, 1712),
        "4,3": (1968, 1472),
        "3,4": (1472, 1968),
        "16,9": (2272, 1280),
        "9,16": (1280, 2272),
    },
    "2048": {
        "1,1": (2728, 2728),
        "4,3": (3160, 2368),
        "3,4": (2368, 3160),
        "16,9": (3640, 2048),
        "9,16": (2048, 3640),
    },
    "gt_2048": {
        "1,1": (5464, 5464),
        "4,3": (6304, 4728),
        "3,4": (4728, 6304),
        "16,9": (7280, 4096),
        "9,16": (4096, 7280),
    },
    "1440": {
        "1,1": (1920, 1920),
        "4,3": (2208, 1664),
        "3,4": (1664, 2208),
        "16,9": (2560, 1440),
        "9,16": (1440, 2560),
    },
}
# The image table drops the 480 "2,3" video bucket and adds 1024_shortside.
IMAGE_RES_SIZE_INFO: dict[str, dict[str, tuple[int, int]]] = {
    **{
        tier: {ar: size for ar, size in sizes.items() if (tier, ar) != ("480", "2,3")}
        for tier, sizes in VIDEO_RES_SIZE_INFO.items()
    },
    "1024_shortside": {
        "1,1": (1024, 1024),
        "4,3": (1024, 768),
        "3,4": (768, 1024),
        "16,9": (1024, 576),
        "9,16": (576, 1024),
    },
}

# Aspect ratio a request without an explicit HR size uses, per the mode's
# cosmos-framework defaults JSON (text2image "1,1"; others "16,9" or unset).
_DEFAULT_ASPECT_RATIO = {"text2image": "1,1"}


class Cosmos3SRCapability(msgspec.Struct, frozen=True):
    """One declared LR-to-HR route (cosmos-framework ``ResolutionCapability``)."""

    modality: str
    hr_resolution: str
    num_frames: int
    tokenizer_resolution: str
    lr_resolution: str


class Cosmos3SuperResolutionConfig(msgspec.Struct, frozen=True):
    """Checkpoint-declared SR support plus the metadata SR requests are checked against."""

    enabled: bool = False
    inference_capabilities_enabled: bool = False
    sr_workflow: str = SR_WORKFLOW_NONE
    capabilities: tuple[Cosmos3SRCapability, ...] = ()
    # Keys of model.config.tokenizer.encode_chunk_frames; every route must name one.
    tokenizer_routes: frozenset[str] = frozenset()
    # Negative disables the packed-token preflight, as in cosmos-framework.
    max_num_tokens_after_packing: int = -1
    default_resolution: str | None = None
    # The checkpoint's inference shift (scheduler flow_shift); SR requests are bound to it.
    inference_flow_shift: float = 1.0
    video_temporal_causal: bool = False


class Cosmos3SRPlan(msgspec.Struct, frozen=True):
    """A request resolved onto one declared route; sizes are (width, height)."""

    modality: str
    hr_resolution: str
    lr_resolution: str
    aspect_ratio: str
    tokenizer_resolution: str
    num_frames: int
    hr_size: tuple[int, int]
    lr_size: tuple[int, int]

    @property
    def spatial_position_scale(self) -> tuple[float, float]:
        """``(scale_h, scale_w)`` placing LR patches in HR patch coordinates."""
        return (
            self.hr_size[1] / self.lr_size[1],
            self.hr_size[0] / self.lr_size[0],
        )


def _int_field(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Published {name} must be an integer, got {value!r}.")
    return value


def _str_field(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"Published {name} must be a non-empty string, got {value!r}.")
    return value


def _parse_capability(block: Any) -> Cosmos3SRCapability:
    if not isinstance(block, Mapping):
        raise TypeError("Each published high-resolution capability must be a mapping.")
    values = {str(key): value for key, value in block.items() if key != "_type"}
    expected = set(Cosmos3SRCapability.__struct_fields__)
    if set(values) != expected:
        raise ValueError(
            f"Published high-resolution capability keys {sorted(values)} must be {sorted(expected)}."
        )
    capability = Cosmos3SRCapability(
        modality=_str_field(values["modality"], "capability modality"),
        hr_resolution=_str_field(values["hr_resolution"], "capability hr_resolution"),
        num_frames=_int_field(values["num_frames"], "capability num_frames"),
        tokenizer_resolution=_str_field(
            values["tokenizer_resolution"], "capability tokenizer_resolution"
        ),
        lr_resolution=_str_field(values["lr_resolution"], "capability lr_resolution"),
    )
    if capability.modality not in SR_MODALITIES:
        raise ValueError(
            f"Published capability modality {capability.modality!r} must be one of "
            f"{sorted(SR_MODALITIES)}."
        )
    if capability.num_frames < 1:
        raise ValueError("Published capability num_frames must be at least 1.")
    return capability


def parse_super_resolution_config(
    root_config: Mapping[str, Any], scheduler_config: Mapping[str, Any]
) -> Cosmos3SuperResolutionConfig:
    """Read ``model.config.high_resolution`` with the cosmos-framework schema checks.

    An absent block is the disabled default, as for checkpoints published
    before SR existed.
    """
    model = root_config.get("model")
    settings = model.get("config") if isinstance(model, Mapping) else None
    if not isinstance(settings, Mapping):
        settings = {}
    flow_shift = float(scheduler_config.get("flow_shift", 1.0))
    tokenizer = settings.get("tokenizer")
    chunk_frames = (
        tokenizer.get("encode_chunk_frames") if isinstance(tokenizer, Mapping) else None
    )
    common = dict(
        tokenizer_routes=frozenset(
            str(key)
            for key in (chunk_frames if isinstance(chunk_frames, Mapping) else {})
        ),
        max_num_tokens_after_packing=_int_field(
            settings.get(
                "max_num_tokens_after_packing", DEFAULT_MAX_NUM_TOKENS_AFTER_PACKING
            ),
            "max_num_tokens_after_packing",
        ),
        default_resolution=(
            str(settings["resolution"])
            if settings.get("resolution") is not None
            else None
        ),
        inference_flow_shift=flow_shift,
        video_temporal_causal=bool(settings.get("video_temporal_causal", False)),
    )
    block = settings.get("high_resolution")
    if block is None:
        return Cosmos3SuperResolutionConfig(**common)
    if not isinstance(block, Mapping):
        raise TypeError("Published high_resolution must be a mapping.")
    values = {str(key): value for key, value in block.items() if key != "_type"}
    published = values.pop("capabilities", [])
    if isinstance(published, (str, bytes)) or not isinstance(published, (list, tuple)):
        raise TypeError("Published high_resolution.capabilities must be a list.")
    unknown = set(values) - {"enabled", "inference_capabilities_enabled", "sr_workflow"}
    if unknown:
        raise ValueError(f"Unknown published high_resolution keys: {sorted(unknown)}.")
    enabled = values.get("enabled", False)
    inference_enabled = values.get("inference_capabilities_enabled", False)
    if not isinstance(enabled, bool) or not isinstance(inference_enabled, bool):
        raise TypeError("Published high_resolution flags must be booleans.")
    sr_workflow = values.get("sr_workflow", SR_WORKFLOW_NONE)
    if sr_workflow not in SR_WORKFLOWS:
        raise ValueError(
            f"Published sr_workflow {sr_workflow!r} must be one of {sorted(SR_WORKFLOWS)}."
        )
    capabilities = tuple(_parse_capability(item) for item in published)
    if enabled != (sr_workflow != SR_WORKFLOW_NONE):
        raise ValueError(
            "Enabled high-resolution support requires an SR workflow, and only then."
        )
    if sr_workflow == SR_WORKFLOW_NONE and capabilities:
        raise ValueError("Capabilities apply only to an LR-to-HR workflow.")
    if sr_workflow != SR_WORKFLOW_NONE and not capabilities:
        raise ValueError("An LR-to-HR workflow must declare at least one capability.")
    keys = [
        (c.modality, c.lr_resolution, c.hr_resolution, c.num_frames)
        for c in capabilities
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("High-resolution checkpoint capabilities must be unique.")
    return Cosmos3SuperResolutionConfig(
        enabled=enabled,
        inference_capabilities_enabled=inference_enabled,
        sr_workflow=sr_workflow,
        capabilities=capabilities,
        **common,
    )


@functools.lru_cache(maxsize=None)
def load_super_resolution_config(model_path: str) -> Cosmos3SuperResolutionConfig:
    """Parse the SR config of a local checkpoint directory (disabled when absent)."""
    root_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(root_path):
        return Cosmos3SuperResolutionConfig()
    with open(root_path, encoding="utf-8") as file:
        root_config = json.load(file)
    scheduler_path = os.path.join(model_path, "scheduler", "scheduler_config.json")
    scheduler_config: dict[str, Any] = {}
    if os.path.isfile(scheduler_path):
        with open(scheduler_path, encoding="utf-8") as file:
            scheduler_config = json.load(file)
    return parse_super_resolution_config(root_config, scheduler_config)


def resolution_sizes(num_frames: int) -> dict[str, dict[str, tuple[int, int]]]:
    return IMAGE_RES_SIZE_INFO if num_frames == 1 else VIDEO_RES_SIZE_INFO


def find_canvas_tier(
    size: tuple[int, int], *, tiers: frozenset[str] | set[str], num_frames: int
) -> tuple[str, str] | None:
    """Return the ``(tier, aspect_ratio)`` among ``tiers`` whose bucket is ``size``."""
    sizes = resolution_sizes(num_frames)
    matches = sorted(
        (tier, aspect_ratio)
        for tier in tiers
        for aspect_ratio, bucket in sizes.get(tier, {}).items()
        if tuple(bucket) == tuple(size)
    )
    return matches[0] if matches else None


def declared_hr_tiers(config: Cosmos3SuperResolutionConfig) -> frozenset[str]:
    return frozenset(c.hr_resolution for c in config.capabilities)


def default_sr_modality(*, num_frames: int, has_image_input: bool) -> str:
    """Route an SR request names when it does not pick one: the LR media's own mode."""
    if num_frames == 1:
        return "image2image"
    return "image2video" if has_image_input else "video2video"


def resolve_sr_capability(
    config: Cosmos3SuperResolutionConfig,
    *,
    modality: str,
    resolution: str,
    aspect_ratio: str,
    num_frames: int,
    lr_resolution: str | None,
) -> Cosmos3SRCapability | None:
    """cosmos-framework ``resolve_high_resolution_capability`` for an SR checkpoint.

    ``None`` means ``resolution`` is not a declared HR tier.
    """
    if not config.enabled:
        return None
    candidates = [c for c in config.capabilities if c.hr_resolution == resolution]
    if not candidates:
        return None
    if not config.inference_capabilities_enabled:
        raise ValueError(
            f"Resolution {resolution!r} is configured for training and architecture "
            "validation only."
        )
    is_image_mode = modality in SR_IMAGE_MODALITIES
    if is_image_mode and num_frames != 1:
        raise ValueError(f"Image output requires one frame, got {num_frames}.")
    if not is_image_mode and num_frames == 1:
        raise ValueError("Video output requires more than one frame.")
    matching = [
        c for c in candidates if c.modality == modality and c.num_frames == num_frames
    ]
    if lr_resolution is not None:
        matching = [c for c in matching if c.lr_resolution == lr_resolution]
    elif len(matching) > 1:
        declared = sorted(c.lr_resolution for c in matching)
        raise ValueError(
            f"LR input resolution is ambiguous for {modality}@{resolution}: {declared}."
        )
    if len(matching) != 1:
        supported = sorted(
            (c.modality, c.lr_resolution, c.num_frames) for c in candidates
        )
        raise ValueError(
            f"Checkpoint does not declare {modality}@{resolution} with {num_frames} "
            f"frame(s). Declared tuples for this resolution: {supported}."
        )
    capability = matching[0]
    sizes = resolution_sizes(num_frames)
    vision = "image" if is_image_mode else "video"
    if resolution not in sizes:
        raise ValueError(
            f"Resolution {resolution!r} is absent from the {vision} resolution definitions."
        )
    if aspect_ratio not in sizes[resolution]:
        raise ValueError(
            f"Resolution {resolution!r} does not support aspect ratio {aspect_ratio!r} for "
            f"{vision} output; supported values are {sorted(sizes[resolution])}."
        )
    if capability.tokenizer_resolution not in config.tokenizer_routes:
        raise ValueError(
            f"Tokenizer route {capability.tokenizer_resolution!r} is not configured in "
            "encode_chunk_frames."
        )
    if aspect_ratio not in sizes.get(capability.lr_resolution, {}):
        raise ValueError(
            f"LR input {capability.lr_resolution!r} with aspect ratio {aspect_ratio!r} is "
            "absent from the resolution definitions."
        )
    if capability.lr_resolution not in config.tokenizer_routes:
        raise ValueError(
            f"LR tokenizer route {capability.lr_resolution!r} is not configured in "
            "encode_chunk_frames."
        )
    return capability


def hr_vision_token_count(size: tuple[int, int], *, num_frames: int) -> int:
    """GEN tokens of the HR item (cosmos-framework ``compute_high_resolution_vision_tokens``)."""
    width, height = size
    latent_frames = 1 + (num_frames - 1) // VAE_TEMPORAL_COMPRESSION
    return (
        latent_frames
        * math.ceil(height // VAE_SPATIAL_COMPRESSION / DIT_PATCH_SPATIAL)
        * math.ceil(width // VAE_SPATIAL_COMPRESSION / DIT_PATCH_SPATIAL)
    )


def require_vae_aligned_size(size: tuple[int, int], *, name: str) -> None:
    width, height = size
    if width % VAE_SPATIAL_COMPRESSION or height % VAE_SPATIAL_COMPRESSION:
        raise ValueError(
            f"{name} size {width}x{height} must be divisible by the VAE spatial "
            f"compression factor {VAE_SPATIAL_COMPRESSION}."
        )


def resolve_sr_plan(
    config: Cosmos3SuperResolutionConfig,
    *,
    modality: str,
    hr_size: tuple[int, int] | None,
    num_frames: int,
    lr_resolution: str | None,
) -> Cosmos3SRPlan:
    """Resolve an SR request onto exactly one declared route, or raise.

    ``hr_size`` is the requested HR canvas; ``None`` takes the checkpoint's
    default resolution at the mode's default aspect ratio.
    """
    if not config.enabled:
        raise ValueError(
            "LR-to-HR input fields require an exact LR-to-HR checkpoint capability; "
            "this checkpoint declares no high_resolution capabilities."
        )
    if config.sr_workflow != SR_WORKFLOW_SEQUENCE_LR:
        raise ValueError(f"Unsupported SR workflow {config.sr_workflow!r}.")
    if modality not in SR_MODALITIES:
        raise ValueError(
            f"LR-to-HR generation supports {sorted(SR_MODALITIES)}, not {modality!r}."
        )
    if hr_size is None:
        if config.default_resolution is None:
            raise ValueError("SR requests must set width and height (the HR canvas).")
        resolution = config.default_resolution
        aspect_ratio = _DEFAULT_ASPECT_RATIO.get(modality, "16,9")
    else:
        tier = find_canvas_tier(
            hr_size, tiers=declared_hr_tiers(config), num_frames=num_frames
        )
        if tier is None:
            canvases = sorted(
                {
                    f"{w}x{h}"
                    for c in config.capabilities
                    for w, h in resolution_sizes(num_frames)
                    .get(c.hr_resolution, {})
                    .values()
                }
            )
            raise ValueError(
                f"HR size {hr_size[0]}x{hr_size[1]} is not a declared SR canvas; "
                f"declared canvases: {canvases}."
            )
        resolution, aspect_ratio = tier
    capability = resolve_sr_capability(
        config,
        modality=modality,
        resolution=resolution,
        aspect_ratio=aspect_ratio,
        num_frames=num_frames,
        lr_resolution=lr_resolution,
    )
    if capability is None:
        raise ValueError(
            "LR-to-HR input fields require an exact LR-to-HR checkpoint capability; "
            f"resolution {resolution!r} is not declared."
        )
    sizes = resolution_sizes(num_frames)
    plan = Cosmos3SRPlan(
        modality=modality,
        hr_resolution=resolution,
        lr_resolution=capability.lr_resolution,
        aspect_ratio=aspect_ratio,
        tokenizer_resolution=capability.tokenizer_resolution,
        num_frames=num_frames,
        hr_size=tuple(sizes[resolution][aspect_ratio]),
        lr_size=tuple(sizes[capability.lr_resolution][aspect_ratio]),
    )
    require_vae_aligned_size(plan.hr_size, name="HR output")
    require_vae_aligned_size(plan.lr_size, name="LR input")
    minimum_tokens = (
        hr_vision_token_count(plan.hr_size, num_frames=num_frames)
        + EMPTY_TEXT_SPLIT_TOKENS
    )
    budget = config.max_num_tokens_after_packing
    if 0 <= budget < minimum_tokens:
        raise ValueError(
            f"High-resolution output requires at least {minimum_tokens} packed tokens, "
            f"which exceeds max_num_tokens_after_packing={budget}."
        )
    return plan


def media_extension(path: str) -> str:
    return os.path.splitext(urlparse(path).path if "://" in path else path)[1].lower()


def validate_lr_media_extension(path: str, *, modality: str) -> None:
    extension = media_extension(path)
    if modality in SR_IMAGE_MODALITIES:
        if extension not in SR_IMAGE_EXTENSIONS:
            raise ValueError(f"LR source for {modality} must be image media.")
    elif extension not in SR_VIDEO_EXTENSIONS:
        raise ValueError(f"LR source for {modality} must be video media.")


def fit_super_resolution_content_size(
    source_size: tuple[int, int], bucket_size: tuple[int, int]
) -> tuple[int, int]:
    """Aspect-preserving (width, height) fitting ``source_size`` in ``bucket_size``.

    Rounds like the training ``ResizeLargestSideAspectPreserving`` stage.
    """
    source_width, source_height = source_size
    bucket_width, bucket_height = bucket_size
    scale = min(bucket_width / source_width, bucket_height / source_height)
    return int(scale * source_width + 0.5), int(scale * source_height + 0.5)


def scale_super_resolution_content_size(
    lr_content_size: tuple[int, int],
    lr_bucket_size: tuple[int, int],
    hr_bucket_size: tuple[int, int],
) -> tuple[int, int]:
    """HR region generated from ``lr_content_size``, scaled bucket to bucket."""
    return (
        round(lr_content_size[0] * hr_bucket_size[0] / lr_bucket_size[0]),
        round(lr_content_size[1] * hr_bucket_size[1] / lr_bucket_size[1]),
    )


def aligned_frame_indices(
    *,
    average_fps: float,
    num_source_frames: int,
    target_fps: float,
    num_frames: int,
) -> list[int]:
    """Nearest source frames of a ``target_fps`` timeline (cosmos-framework editing.py)."""
    if target_fps <= 0 or average_fps <= 0:
        raise ValueError("FPS values must be positive")
    indices = [round(i * average_fps / target_fps) for i in range(num_frames)]
    return [min(max(index, 0), num_source_frames - 1) for index in indices]


def lr_video_frame_indices(
    *,
    average_fps: float,
    num_source_frames: int,
    target_fps: float,
    num_frames: int,
) -> list[int]:
    """LR frames for ``num_frames`` outputs at ``target_fps``; never repeats a frame."""
    if average_fps <= 0:
        raise ValueError(f"LR source FPS must be positive, got {average_fps}.")
    if num_frames <= 1:
        raise ValueError(
            f"Video LR source requires more than one output frame, got {num_frames}."
        )
    if round((num_frames - 1) * average_fps / target_fps) >= num_source_frames:
        raise ValueError(
            f"LR source cannot provide {num_frames} frames at {target_fps} FPS without "
            "repeating frames."
        )
    indices = aligned_frame_indices(
        average_fps=average_fps,
        num_source_frames=num_source_frames,
        target_fps=target_fps,
        num_frames=num_frames,
    )
    if len(set(indices)) != len(indices):
        raise ValueError(
            f"LR source cannot provide {num_frames} distinct frames at {target_fps} FPS "
            "without repeating frames."
        )
    return indices


def _unsupported_sr_features(
    *,
    control_paths: list[str],
    sound_duration: float,
    action_mode: str | None,
    rollout: bool,
    video_temporal_causal: bool,
) -> list[str]:
    """Features that do not compose with SR (cosmos-framework ``_unsupported_high_resolution_features``)."""
    unsupported = []
    if control_paths:
        unsupported.append("transfer")
    if sound_duration > 0.0:
        unsupported.append("sound")
    if action_mode is not None:
        unsupported.append("action")
    if video_temporal_causal:
        unsupported.append("causal autoregressive generation")
    if rollout:
        unsupported.append("rollout")
    return unsupported


def _validate_sr_source_inputs(
    *,
    modality: str,
    image_path: Any,
    video_path: Any,
    condition_frame_indexes: list[int] | None,
) -> None:
    """Source-media rules of ``_validate_high_resolution_conditioning``.

    image2video carries its source image, which the HR pass does not use; every
    other route takes the LR media alone.
    """
    if video_path:
        raise ValueError(f"LR-to-HR {modality} does not accept video_path.")
    if modality == "image2video":
        if not image_path or isinstance(image_path, (list, tuple)):
            raise ValueError(
                "LR-to-HR image2video requires one source image in image_path."
            )
        if isinstance(image_path, str) and "://" not in image_path:
            extension = media_extension(image_path)
            if extension and extension not in SR_IMAGE_EXTENSIONS:
                raise ValueError(
                    "LR-to-HR image2video requires an image in image_path."
                )
        if condition_frame_indexes not in (None, [0]):
            raise ValueError(
                "LR-to-HR image2video requires conditioning on latent frame 0 only."
            )
        return
    if image_path:
        raise ValueError(f"LR-to-HR {modality} does not accept image_path.")
    if condition_frame_indexes:
        raise ValueError(
            f"LR-to-HR {modality} does not accept conditioning frame indexes."
        )


def resolve_super_resolution_request(
    config: Cosmos3SuperResolutionConfig | None,
    *,
    lr_vision_path: str | None,
    lr_resolution: str | None,
    sr_modality: str | None,
    hr_size: tuple[int, int] | None,
    num_frames: int,
    num_frames_explicit: bool,
    image_path: Any,
    video_path: Any,
    condition_frame_indexes: list[int] | None,
    control_paths: list[str],
    sound_duration: float,
    action_mode: str | None,
    rollout: bool,
    resolution_adapter_policy: str | None = None,
    resolution_adapter_output_size: tuple[int, int] | None = None,
    probe_source_size: Callable[[str], tuple[int, int]] | None = None,
) -> "tuple[Cosmos3SRPlan, Cosmos3SRAdapterPlan | None] | None":
    """Validate a request against the checkpoint's SR routes.

    Returns the plan of an LR-to-HR request (and its resolution-adapter plan, if
    the request names a policy), ``None`` for any other request. ``hr_size`` is
    ``None`` when the request sets no explicit width/height;
    ``probe_source_size`` reads the LR media's (width, height) for adapter planning.
    """
    config = config or Cosmos3SuperResolutionConfig()
    if resolution_adapter_output_size is not None and resolution_adapter_policy is None:
        raise ValueError(
            "resolution_adapter_output_size requires an explicit resolution_adapter_policy."
        )
    if lr_vision_path is None:
        if resolution_adapter_policy is not None:
            raise ValueError(
                "Resolution adaptation requires a supplied lr_vision_path."
            )
        if lr_resolution is not None or sr_modality is not None:
            raise ValueError(
                "lr_resolution and sr_modality apply only to an LR-to-HR request "
                "with lr_vision_path."
            )
        if (
            config.enabled
            and hr_size is not None
            and find_canvas_tier(
                hr_size, tiers=declared_hr_tiers(config), num_frames=num_frames
            )
            is not None
        ):
            raise ValueError(
                f"{hr_size[0]}x{hr_size[1]} is a declared super-resolution canvas; "
                "this request requires lr_vision_path."
            )
        return None

    if sr_modality is None:
        sr_modality = (
            "image2image"
            if media_extension(lr_vision_path) in SR_IMAGE_EXTENSIONS
            else default_sr_modality(
                num_frames=num_frames, has_image_input=bool(image_path)
            )
        )
    if sr_modality not in SR_MODALITIES:
        raise ValueError(
            f"sr_modality must be one of {sorted(SR_MODALITIES)}, got {sr_modality!r}."
        )
    if sr_modality in SR_IMAGE_MODALITIES and not num_frames_explicit:
        num_frames = 1
    unsupported = _unsupported_sr_features(
        control_paths=control_paths,
        sound_duration=sound_duration,
        action_mode=action_mode,
        rollout=rollout,
        video_temporal_causal=config.video_temporal_causal,
    )
    if unsupported:
        raise ValueError(
            f"LR-to-HR generation does not support: {', '.join(unsupported)}."
        )
    adapter = None
    if resolution_adapter_policy is not None:
        if probe_source_size is None:
            raise ValueError("Resolution adaptation needs the LR source size.")
        adapter = plan_resolution_adapter(
            config,
            policy=resolution_adapter_policy,
            modality=sr_modality,
            hr_size=hr_size,
            num_frames=num_frames,
            requested_input_size=probe_source_size(lr_vision_path),
            requested_output_size=resolution_adapter_output_size,
        )
        if (
            lr_resolution is not None
            and lr_resolution != adapter.capability_lr_resolution
        ):
            raise ValueError(
                f"Declared lr_resolution {lr_resolution!r} conflicts with the selected "
                f"resolution-adapter input {adapter.capability_lr_resolution!r}."
            )
        hr_size = adapter.model_output_size
        lr_resolution = adapter.capability_lr_resolution
    plan = resolve_sr_plan(
        config,
        modality=sr_modality,
        hr_size=hr_size,
        num_frames=num_frames,
        lr_resolution=lr_resolution,
    )
    _validate_sr_source_inputs(
        modality=sr_modality,
        image_path=image_path,
        video_path=video_path,
        condition_frame_indexes=condition_frame_indexes,
    )
    validate_lr_media_extension(lr_vision_path, modality=sr_modality)
    return plan, adapter


def bind_sr_flow_shift(
    config: Cosmos3SuperResolutionConfig,
    *,
    requested_flow_shift: float | None,
    flow_shift_explicit: bool,
) -> float:
    """SR samples at the checkpoint's inference shift; a different explicit shift is an error."""
    required = config.inference_flow_shift
    if flow_shift_explicit and requested_flow_shift is not None:
        if not math.isclose(
            float(requested_flow_shift), required, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                f"Requested shift {requested_flow_shift} does not match model inference "
                f"shift {required}."
            )
    return required


# ===== Resolution adapter (cosmos-framework data/generator/conditioning/high_resolution_adapter.py)

RESOLUTION_ADAPTER_POLICIES = ("cost_first", "quality_first")


class Cosmos3SRAdapterPlan(msgspec.Struct, frozen=True):
    """One declared pair plus the down-resizes done outside the model; sizes are (width, height).

    The model canvases are the declared LR/HR buckets; the content sizes are the
    top-left regions the request occupies in them.
    """

    policy: str
    route: str
    requested_input_size: tuple[int, int]
    requested_output_size: tuple[int, int]
    capability_lr_resolution: str
    capability_hr_resolution: str
    capability_aspect_ratio: str
    model_input_size: tuple[int, int]
    model_output_size: tuple[int, int]
    model_input_content_size: tuple[int, int]
    model_output_content_size: tuple[int, int]

    @property
    def resizes_output(self) -> bool:
        return self.model_output_content_size != self.requested_output_size


def _orientation(size: tuple[int, int]) -> str:
    width, height = size
    if width > height:
        return "landscape"
    if width < height:
        return "portrait"
    return "square"


def _same_aspect_ratio(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] * right[1] == right[0] * left[1]


def _validate_size(size: Any, *, name: str) -> tuple[int, int]:
    if (
        not isinstance(size, (list, tuple))
        or len(size) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in size)
    ):
        raise ValueError(
            f"{name} must contain two positive integer dimensions, got {size!r}."
        )
    return (int(size[0]), int(size[1]))


def build_resolution_adapter_plan(
    *,
    policy: str,
    capabilities: tuple[Cosmos3SRCapability, ...],
    modality: str,
    num_frames: int,
    aspect_ratio: str,
    requested_input_size: tuple[int, int],
    requested_output_size: tuple[int, int],
) -> Cosmos3SRAdapterPlan:
    """Map a boundary request onto one declared LR-to-HR pair, never upscaling outside the model."""
    if policy not in RESOLUTION_ADAPTER_POLICIES:
        raise ValueError(f"Unsupported resolution-adapter policy {policy!r}.")
    requested_input_size = _validate_size(
        requested_input_size, name="requested_input_size"
    )
    requested_output_size = _validate_size(
        requested_output_size, name="requested_output_size"
    )
    if num_frames > 1 and any(dimension % 2 for dimension in requested_output_size):
        raise ValueError(
            "Video resolution-adapter output dimensions must both be even so the encoded "
            f"file preserves the requested size exactly, got {requested_output_size}."
        )
    if _orientation(requested_input_size) != _orientation(requested_output_size):
        raise ValueError(
            "Resolution adaptation does not rotate or transpose media: input is "
            f"{_orientation(requested_input_size)} and output is "
            f"{_orientation(requested_output_size)}."
        )
    if not _same_aspect_ratio(requested_input_size, requested_output_size):
        raise ValueError(
            "Resolution adaptation requires matching input and output aspect ratios: "
            f"got {requested_input_size} and {requested_output_size}."
        )
    sizes = resolution_sizes(num_frames)
    candidates = []
    for capability in capabilities:
        if capability.modality != modality or capability.num_frames != num_frames:
            continue
        lr_sizes = sizes.get(capability.lr_resolution)
        hr_sizes = sizes.get(capability.hr_resolution)
        if (
            lr_sizes is None
            or hr_sizes is None
            or aspect_ratio not in lr_sizes
            or aspect_ratio not in hr_sizes
        ):
            continue
        model_input_size = tuple(lr_sizes[aspect_ratio])
        model_output_size = tuple(hr_sizes[aspect_ratio])
        input_content = fit_super_resolution_content_size(
            requested_input_size, model_input_size
        )
        output_content = scale_super_resolution_content_size(
            input_content, model_input_size, model_output_size
        )
        if not _same_aspect_ratio(input_content, requested_input_size):
            continue
        if not _same_aspect_ratio(output_content, requested_output_size):
            continue
        # Postprocessing may only keep or reduce detail, never enlarge the model output.
        if (
            output_content[0] < requested_output_size[0]
            or output_content[1] < requested_output_size[1]
        ):
            continue
        candidates.append(
            (
                capability,
                model_input_size,
                model_output_size,
                input_content,
                output_content,
            )
        )
    exact = [
        c
        for c in candidates
        if c[3] == requested_input_size and c[4] == requested_output_size
    ]
    if exact:
        eligible, route = exact, "exact"
    elif policy == "quality_first":
        # Keep every source pixel; take the smallest HR content that down-resizes to the request.
        eligible = [c for c in candidates if c[3] == requested_input_size]
        route = "quality_first"
    else:
        # Cost-first may down-resize the source, never upscale it.
        eligible = [
            c
            for c in candidates
            if c[3][0] <= requested_input_size[0] and c[3][1] <= requested_input_size[1]
        ]
        route = "cost_first"
    if not eligible:
        declared = sorted(
            (c.lr_resolution, c.hr_resolution, input_content, output_content)
            for c, _, _, input_content, output_content in candidates
        )
        raise ValueError(
            f"No {policy!r} resolution-adapter route can map {requested_input_size} to "
            f"{requested_output_size} for {modality} with {num_frames} frame(s), aspect ratio "
            f"{aspect_ratio!r}. Eligible declared pairs before policy filtering: {declared}."
        )

    def preference(candidate):
        capability, _, _, input_content, output_content = candidate
        area = output_content[0] * output_content[1]
        names = (capability.hr_resolution, capability.lr_resolution or "")
        if exact or policy == "quality_first":
            return (area, *names)
        return (area, input_content[0] * input_content[1], *names)

    capability, model_input_size, model_output_size, input_content, output_content = (
        min(eligible, key=preference)
    )
    return Cosmos3SRAdapterPlan(
        policy=policy,
        route=route,
        requested_input_size=requested_input_size,
        requested_output_size=requested_output_size,
        capability_lr_resolution=capability.lr_resolution,
        capability_hr_resolution=capability.hr_resolution,
        capability_aspect_ratio=aspect_ratio,
        model_input_size=model_input_size,
        model_output_size=model_output_size,
        model_input_content_size=input_content,
        model_output_content_size=output_content,
    )


def plan_resolution_adapter(
    config: Cosmos3SuperResolutionConfig,
    *,
    policy: str,
    modality: str,
    hr_size: tuple[int, int] | None,
    num_frames: int,
    requested_input_size: tuple[int, int],
    requested_output_size: tuple[int, int] | None,
) -> Cosmos3SRAdapterPlan:
    """Adapter plan of a request; the requested HR canvas only sets the default output size."""
    if not config.enabled:
        raise ValueError(
            "Resolution adaptation requires an enabled high-resolution checkpoint capability."
        )
    sizes = resolution_sizes(num_frames)
    if hr_size is None:
        if config.default_resolution is None:
            raise ValueError("Resolution adaptation requires an output resolution.")
        aspect_ratio = _DEFAULT_ASPECT_RATIO.get(modality, "16,9")
        canvas = sizes[config.default_resolution][aspect_ratio]
    else:
        tier = find_canvas_tier(hr_size, tiers=frozenset(sizes), num_frames=num_frames)
        if tier is None:
            raise ValueError(
                f"HR size {hr_size[0]}x{hr_size[1]} is not a resolution preset bucket."
            )
        aspect_ratio = tier[1]
        canvas = tuple(hr_size)
    # By default, the source's aspect ratio fitted inside the requested HR bucket.
    output_size = requested_output_size or fit_super_resolution_content_size(
        requested_input_size, canvas
    )
    return build_resolution_adapter_plan(
        policy=policy,
        capabilities=config.capabilities,
        modality=modality,
        num_frames=num_frames,
        aspect_ratio=aspect_ratio,
        requested_input_size=requested_input_size,
        requested_output_size=output_size,
    )


def parse_size(value: Any) -> tuple[int, int] | None:
    """``"WxH"``, ``"W,H"`` or ``[W, H]`` -> ``(W, H)``."""
    if value is None:
        return None
    if isinstance(value, str):
        parts = value.lower().replace(",", "x").split("x")
        if len(parts) != 2:
            raise ValueError(f"Size must be 'WIDTHxHEIGHT', got {value!r}.")
        value = [part.strip() for part in parts]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Size must have two dimensions, got {value!r}.")
    return _validate_size(tuple(int(v) for v in value), name="size")
