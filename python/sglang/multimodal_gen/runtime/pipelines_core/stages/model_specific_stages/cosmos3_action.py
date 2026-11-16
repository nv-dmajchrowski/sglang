# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 action modality helpers: domain mapping, mode constants, the
structured JSON caption, dataset-derived action (de)normalization, and the
inverse-dynamics contract (Image2Action layout, plain captions, bucket input
fitting, unified_v1 slots) of the reference action framework.
"""

import json
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

ACTION_MODE_POLICY = "policy"
ACTION_MODE_FORWARD_DYNAMICS = "forward_dynamics"
ACTION_MODE_INVERSE_DYNAMICS = "inverse_dynamics"
ACTION_MODES = {
    ACTION_MODE_POLICY,
    ACTION_MODE_FORWARD_DYNAMICS,
    ACTION_MODE_INVERSE_DYNAMICS,
}

EMBODIMENT_TO_DOMAIN_ID: dict[str, int] = {
    "no_action": 0,
    "av": 1,
    "camera_pose": 2,
    "hand_pose": 3,
    # WebHumanAction hands omit the camera block of hand_pose's layout, so they
    # get their own projector slot instead of reusing domain 3.
    "webhumanaction_hand": 31,
    "pusht": 4,
    "libero": 5,
    "umi": 6,
    "bridge_orig_lerobot": 7,
    "droid_lerobot": 8,
    "robomind-franka": 8,
    "galbot": 9,
    "robomind-franka-dual": 12,
    "robomind-ur": 13,
    "agibotworld": 15,
    "agibot_gear_gripper": 15,
    "agibot_gear_gripper_ext": 15,
    "xdof_yam": 16,
    "molmoact2_yam": 16,
    "abc_yam": 16,
    "robotwin": 17,
    "robodojo": 17,
    "fractal": 20,
    "drawanything": 21,
    "behavior1k_lerobot": 22,
    "maniparena": 23,
    "webhumanaction_body": 24,
    "so101-molmo-midtrain-15hz": 25,
    "so100-molmo-midtrain-15hz": 26,
    "so101-bimanual-midtrain-conditional": 27,
    "geniesim3_g2a": 15,
    "geniesim3_g2a_joint": 28,
    "maniparena_mobile": 29,
    "robocasa": 30,
    "galbot_20260828": 32,
}

# Embodiment -> real (unpadded) action channel count. Channels beyond this are
# zero-padding up to the model's action_dim. libero, hand_pose and robocasa are
# absent on purpose: their width is set per dataset, so requests must pass
# raw_action_dim.
EMBODIMENT_TO_RAW_ACTION_DIM: dict[str, int] = {
    "av": 9,
    "camera_pose": 9,
    "pusht": 2,
    "umi": 10,
    "bridge_orig_lerobot": 10,
    "droid_lerobot": 10,
    "robomind-franka": 10,
    "robomind-franka-dual": 20,
    "robomind-ur": 10,
    "galbot": 30,
    "agibotworld": 29,
    "agibot_gear_gripper": 29,
    "agibot_gear_gripper_ext": 29,
    # head(9) + [R_wrist(9) + R_fingertips(15)] + [L_wrist(9) + L_fingertips(15)]
    "webhumanaction_body": 57,
    # [R_wrist(9) + R_fingertips(15)] + [L_wrist(9) + L_fingertips(15)]
    "webhumanaction_hand": 48,
    "xdof_yam": 20,
    "molmoact2_yam": 20,
    "abc_yam": 20,
    "maniparena": 20,
    "maniparena_mobile": 29,
    "robotwin": 14,
    "robodojo": 14,
    "fractal": 10,
    "drawanything": 3,
    "behavior1k_lerobot": 23,
    "so101-molmo-midtrain-15hz": 10,
    "so100-molmo-midtrain-15hz": 10,
    "so101-bimanual-midtrain-conditional": 20,
    "geniesim3_g2a": 29,
    "geniesim3_g2a_joint": 16,
    "galbot_20260828": 50,
}

# Canonical (width, height) targets per resolution tier and aspect ratio, used
# to render the aspect_ratio field of the action caption.
VIDEO_RES_SIZE_INFO: dict[str, dict[str, tuple[int, int]]] = {
    "256": {
        "1,1": (256, 256),
        "4,3": (320, 256),
        "3,4": (256, 320),
        "16,9": (320, 192),
        "9,16": (192, 320),
    },
    "480": {
        "1,1": (640, 640),
        "4,3": (736, 544),
        "3,4": (544, 736),
        "16,9": (832, 480),
        "9,16": (480, 832),
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
}

# Every native canvas size (all resolution tiers, native order): the action
# caption labels a canvas by its table aspect key, so sizes whose reduced ratio
# differs from the key (e.g. 960x544 -> "16,9", not "30,17") need the entry.
ASPECT_RATIO_SIZE_INFO: dict[str, dict[str, tuple[int, int]]] = {
    "256": VIDEO_RES_SIZE_INFO["256"],
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
    "480": {**VIDEO_RES_SIZE_INFO["480"], "2,3": (512, 768)},
    "704": VIDEO_RES_SIZE_INFO["704"],
    "720": VIDEO_RES_SIZE_INFO["720"],
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

VIEWPOINT_TEMPLATES: dict[str, str] = {
    "ego_view": "This video is captured from a first-person perspective looking at the scene.",
    "third_person_view": "This video is captured from a third-person perspective looking towards the agent from the front.",
    "wrist_view": "This video is captured from a wrist-mounted camera.",
    "concat_view": "This video contains concatenated views from multiple camera perspectives.",
}

_STAT_KEYS = {"mean", "std", "min", "max", "q01", "q99"}


def get_raw_action_dim(embodiment: str) -> int:
    key = embodiment.lower().strip()
    if key not in EMBODIMENT_TO_RAW_ACTION_DIM:
        raise ValueError(
            f"No raw action dim for Cosmos3 embodiment {embodiment!r}. Expected one "
            f"of {sorted(EMBODIMENT_TO_RAW_ACTION_DIM)}."
        )
    return EMBODIMENT_TO_RAW_ACTION_DIM[key]


def canonical_aspect_ratio(width: int, height: int) -> str:
    """Canonical ``"W,H"`` aspect string for the action caption."""
    for sizes in VIDEO_RES_SIZE_INFO.values():
        for aspect, (cand_w, cand_h) in sizes.items():
            if width == cand_w and height == cand_h:
                return aspect
    divisor = math.gcd(width, height)
    if divisor == 0:
        raise ValueError(
            f"width and height must be non-zero, got width={width}, height={height}."
        )
    return f"{width // divisor},{height // divisor}"


def build_action_prompt(
    description: str | list[str],
    view_point: str,
    num_frames: int,
    fps: float,
    height: int,
    width: int,
) -> str | list[str]:
    """Render the structured JSON action caption the action checkpoints expect."""
    duration_seconds = num_frames / fps
    minutes, secs = divmod(round(duration_seconds), 60)
    if isinstance(description, (list, tuple)):
        descriptions = [str(d) for d in description]
    else:
        descriptions = [description]

    prompts = []
    for desc in descriptions:
        if desc and desc[-1] not in ".!?":
            desc = desc + "."
        prompt = {
            "cinematography": {
                "framing": VIEWPOINT_TEMPLATES.get(
                    view_point, VIEWPOINT_TEMPLATES["ego_view"]
                )
            },
            "actions": [{"time": f"0:00-{minutes}:{secs:02d}", "description": desc}],
            "duration": f"{int(duration_seconds)}s",
            "fps": float(fps),
            "resolution": {"H": int(height), "W": int(width)},
            "aspect_ratio": canonical_aspect_ratio(int(width), int(height)),
        }
        prompts.append(json.dumps(prompt))

    if isinstance(description, (list, tuple)):
        return prompts
    return prompts[0]


def resolve_raw_action_dim(
    domain_name: str | None, raw_action_dim: int | None
) -> int | None:
    """Real action width for a domain; ``None`` if the domain has no fixed width
    and none was given. A fixed-width domain rejects a different explicit width."""
    explicit = None if raw_action_dim is None else int(raw_action_dim)
    if not domain_name:
        return explicit
    table_dim = EMBODIMENT_TO_RAW_ACTION_DIM.get(domain_name.strip().lower())
    if table_dim is None:
        return explicit
    if explicit is not None and explicit != table_dim:
        raise ValueError(
            f"raw_action_dim ({explicit}) does not match domain_name="
            f"{domain_name!r} dimension ({table_dim})"
        )
    return table_dim


# Modes whose action stream may start with the predicted frame-0 initial-state
# row a0 (Image2Action); forward dynamics and policy never carry it.
INITIAL_STATE_ACTION_MODES = frozenset({ACTION_MODE_INVERSE_DYNAMICS})
# Action modes whose conditioning input can be bucket-fitted (action_resolution).
ACTION_INPUT_FIT_MODES = frozenset(ACTION_MODES)


class ActionLayout(NamedTuple):
    """Action rows for one clip: count, clean (given) rows, and row 0's video frame."""

    num_rows: int
    condition_rows: tuple[int, ...]
    start_frame_offset: int


def action_layout(
    mode: str, num_frames: int, predict_initial_state: bool = False
) -> ActionLayout:
    """Lay out the action stream of a ``num_frames`` clip without history rows.

    By default there is one row per frame transition and row ``r`` drives frame
    ``r + 1``. With ``predict_initial_state`` (Image2Action) row 0 is the
    predicted frame-0 pose a0, row ``k`` sits on frame ``k``, and no row is given.
    """
    if mode not in ACTION_MODES:
        raise ValueError(f"Unsupported action mode {mode!r}.")
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}.")
    if predict_initial_state:
        if mode not in INITIAL_STATE_ACTION_MODES:
            raise ValueError(
                "predict_initial_state is only defined for "
                f"{sorted(INITIAL_STATE_ACTION_MODES)}, got action_mode={mode!r}."
            )
        return ActionLayout(
            num_rows=num_frames, condition_rows=(), start_frame_offset=0
        )
    if num_frames > 1:
        num_rows, start_frame_offset = num_frames - 1, 1
    else:
        # A single-frame request keeps one action row on frame 0.
        num_rows, start_frame_offset = 1, 0
    condition_rows = (
        tuple(range(num_rows)) if mode == ACTION_MODE_FORWARD_DYNAMICS else ()
    )
    return ActionLayout(
        num_rows=num_rows,
        condition_rows=condition_rows,
        start_frame_offset=start_frame_offset,
    )


# unified_v1 is the 59D cross-embodiment action layout:
# [ego_pose 0:9, right_wrist 9:18, right_fingertips 18:33, right_openness 33,
#  left_wrist 34:43, left_fingertips 43:58, left_openness 58]. A domain fills
# only its own slots; the rest stay zero and are excluded by the valid mask.
UNIFIED_ACTION_SCHEMA = "unified_v1"
ACTION_SCHEMAS = ("native", UNIFIED_ACTION_SCHEMA)
UNIFIED_ACTION_DIM = 59
_EGO, _R_WRIST, _R_TIPS, _R_OPEN = (0, 9), (9, 18), (18, 33), (33, 34)
_L_WRIST, _L_TIPS, _L_OPEN = (34, 43), (43, 58), (58, 59)
_SINGLE_ARM = (_R_WRIST, _R_OPEN)
_DUAL_ARM = (_R_WRIST, _R_OPEN, _L_WRIST, _L_OPEN)
_HEAD_DUAL_ARM = (_EGO, *_DUAL_ARM)
_HAND_POSE = (_EGO, _R_WRIST, _R_TIPS, _L_WRIST, _L_TIPS)
# Domain -> unified_v1 slots its datasets' mappings fill (single/dual-arm,
# humanoid/head-dual-arm, hand-pose and ego mappings of the action datasets).
# Joint-space contracts (pusht, drawanything, geniesim3_g2a_joint, galbot) have
# no unified_v1 mapping. "umi" is the single-arm layout of its 10D table width;
# bimanual UMI shares domain 6 with the dual-arm slots.
UNIFIED_V1_DOMAIN_SLOTS: dict[str, tuple[tuple[int, int], ...]] = {
    "av": (_EGO,),
    "camera_pose": (_EGO,),
    "hand_pose": _HAND_POSE,
    "webhumanaction_hand": (_R_WRIST, _R_TIPS, _L_WRIST, _L_TIPS),
    "webhumanaction_body": _HAND_POSE,
    "libero": _SINGLE_ARM,
    "umi": _SINGLE_ARM,
    "bridge_orig_lerobot": _SINGLE_ARM,
    "droid_lerobot": _SINGLE_ARM,
    "robomind-franka": _SINGLE_ARM,
    "robomind-franka-dual": _DUAL_ARM,
    "robomind-ur": _SINGLE_ARM,
    "agibotworld": _HEAD_DUAL_ARM,
    "agibot_gear_gripper": _HEAD_DUAL_ARM,
    "agibot_gear_gripper_ext": _HEAD_DUAL_ARM,
    "geniesim3_g2a": _HEAD_DUAL_ARM,
    "xdof_yam": _DUAL_ARM,
    "molmoact2_yam": _DUAL_ARM,
    "abc_yam": _DUAL_ARM,
    "robotwin": _DUAL_ARM,
    "robodojo": _DUAL_ARM,
    "fractal": _SINGLE_ARM,
    "behavior1k_lerobot": _HEAD_DUAL_ARM,
    "maniparena": _DUAL_ARM,
    "maniparena_mobile": _HEAD_DUAL_ARM,
    "so101-molmo-midtrain-15hz": _SINGLE_ARM,
    "so100-molmo-midtrain-15hz": _SINGLE_ARM,
    "so101-bimanual-midtrain-conditional": _DUAL_ARM,
    "galbot_20260828": _HEAD_DUAL_ARM,
}
_HAND_POSE_57D_TO_UNIFIED = (
    (0, 9, 0),
    (9, 18, 9),
    (18, 33, 18),
    (33, 42, 34),
    (42, 57, 43),
)
# Human domain -> (native width, ((native_start, native_stop, unified_start), ...)).
UNIFIED_V1_HUMAN_NATIVE_LAYOUTS: dict[
    str, tuple[int, tuple[tuple[int, int, int], ...]]
] = {
    "webhumanaction_hand": (48, ((0, 9, 9), (9, 24, 18), (24, 33, 34), (33, 48, 43))),
    "webhumanaction_body": (57, _HAND_POSE_57D_TO_UNIFIED),
    "hand_pose": (57, _HAND_POSE_57D_TO_UNIFIED),
}


def unified_action_valid_mask(domain_name: str | None) -> torch.Tensor:
    """``[59]`` bool mask of the unified_v1 slots ``domain_name`` fills."""
    key = (domain_name or "").strip().lower()
    if key not in UNIFIED_V1_DOMAIN_SLOTS:
        raise ValueError(
            f"No unified_v1 slots for domain_name={domain_name!r}; "
            f"supported domains: {sorted(UNIFIED_V1_DOMAIN_SLOTS)}."
        )
    mask = torch.zeros(UNIFIED_ACTION_DIM, dtype=torch.bool)
    for start, stop in UNIFIED_V1_DOMAIN_SLOTS[key]:
        mask[start:stop] = True
    return mask


def unified_to_native_action(
    action: torch.Tensor, domain_name: str | None
) -> torch.Tensor:
    """Gather a human domain's native ``[..., D_native]`` action from unified_v1 ``[..., 59]``."""
    key = (domain_name or "").strip().lower()
    if key not in UNIFIED_V1_HUMAN_NATIVE_LAYOUTS:
        raise ValueError(
            f"No native unified_v1 layout for domain_name={domain_name!r}."
        )
    native_dim, components = UNIFIED_V1_HUMAN_NATIVE_LAYOUTS[key]
    native = action.new_zeros(*action.shape[:-1], native_dim)
    for start, stop, target in components:
        native[..., start:stop] = action[..., target : target + stop - start]
    return native


# Plain-caption semantics of the Lance action recipes, as postfixes in their
# canonical order (domain, embodiment, view composition, caption subject).
# hand_pose is the Mecka / VITRA egocentric protocol.
_REAL_DOMAIN_POSTFIX = "The video is captured from a real-world environment."
_HUMAN_EMBODIMENT_POSTFIX = "The video shows a human actor."
_STATIC_SINGLE_VIEW_POSTFIX = (
    "This video is captured from a static perspective looking towards the actor."
)
ACTION_CAPTION_SEMANTICS: dict[str, tuple[str, str, str, str]] = {
    "webhumanaction_hand": (
        _REAL_DOMAIN_POSTFIX,
        _HUMAN_EMBODIMENT_POSTFIX,
        _STATIC_SINGLE_VIEW_POSTFIX,
        "The action performed by the human actor is defined as the wrist and "
        "fingertip motion of the actor's hands, mapped to the right and left "
        "wrist-pose and fingertip components.",
    ),
    "webhumanaction_body": (
        _REAL_DOMAIN_POSTFIX,
        _HUMAN_EMBODIMENT_POSTFIX,
        _STATIC_SINGLE_VIEW_POSTFIX,
        "The action performed by the human actor is defined as the ego pose of "
        "the actor's head and the wrist motion of the actor's hands, mapped to "
        "the ego-pose and right and left wrist-pose components.",
    ),
    "hand_pose": (
        _REAL_DOMAIN_POSTFIX,
        _HUMAN_EMBODIMENT_POSTFIX,
        "This video is captured from a head-mounted camera showing an egocentric "
        "view of the scene.",
        "The action performed by the human actor is defined as the ego motion of "
        "the actor's head and the wrist and fingertip motion of the actor's "
        "hands, mapped to the ego-pose and right and left wrist-pose and "
        "fingertip components.",
    ),
}
ACTION_RESOLUTION_SENTENCE = "This video is of {height}x{width} resolution."
ACTION_MODE_DESCRIPTIONS: dict[str, str] = {
    ACTION_MODE_POLICY: "Predict future action given the image.",
    ACTION_MODE_FORWARD_DYNAMICS: "Predict the future video given the image and action.",
    ACTION_MODE_INVERSE_DYNAMICS: "Predict the underlying action given the video.",
}


def add_action_mode_metadata(caption: str, mode: str) -> str:
    """Plain-caption mode description; inverse dynamics drops the input caption,
    which describes the action it must predict."""
    description = ACTION_MODE_DESCRIPTIONS.get(mode)
    if description is None:
        return caption
    if mode == ACTION_MODE_INVERSE_DYNAMICS:
        return description
    task = caption.strip()
    if task and not task.endswith((".", "!", "?")):
        task += "."
    return " ".join(segment for segment in (task, description) if segment)


def build_plain_action_caption(
    caption: str,
    *,
    domain_name: str | None,
    height: int,
    width: int,
    append_semantics: bool,
    mode: str | None = None,
) -> str:
    """Plain action caption: the optional mode description (``mode``), then with
    ``append_semantics`` the domain's semantic postfixes and the resolution
    sentence, in the order the Lance action recipes build it."""
    if mode is not None:
        caption = add_action_mode_metadata(caption, mode)
    if not append_semantics or not caption:
        return caption
    key = (domain_name or "").strip().lower()
    postfixes = ACTION_CAPTION_SEMANTICS.get(key)
    if postfixes is None:
        raise ValueError(
            f"No action caption semantics for domain_name={domain_name!r}; "
            f"known domains: {sorted(ACTION_CAPTION_SEMANTICS)}."
        )
    caption = caption.rstrip()
    separator = " " if caption.endswith((".", "!", "?")) else ". "
    caption = f"{caption}{separator}{' '.join(postfixes)}"
    separator = " " if caption.rstrip().endswith(".") else ". "
    return (
        caption
        + separator
        + ACTION_RESOLUTION_SENTENCE.format(height=height, width=width)
    )


def render_action_prompt(
    description: str | list[str],
    *,
    format_prompt_as_json: bool,
    append_semantics: bool,
    mode_prompt: str | None,
    domain_name: str | None,
    view_point: str,
    num_frames: int,
    fps: float,
    height: int,
    width: int,
) -> str | list[str]:
    """The action caption handed to the tokenizer: structured JSON, or a plain
    caption (verbatim unless ``mode_prompt`` or ``append_semantics``)."""
    if format_prompt_as_json:
        if mode_prompt is not None:
            raise ValueError("Mode-specific action prompts require a plain caption.")
        return build_action_prompt(
            description, view_point, num_frames, fps, height, width
        )
    descriptions = (
        [str(d) for d in description]
        if isinstance(description, (list, tuple))
        else [description]
    )
    captions = [
        build_plain_action_caption(
            d,
            domain_name=domain_name,
            height=height,
            width=width,
            append_semantics=append_semantics,
            mode=mode_prompt,
        )
        for d in descriptions
    ]
    return captions if isinstance(description, (list, tuple)) else captions[0]


class ActionInputFit(NamedTuple):
    """Geometry of one bucket-fitted action input, in pixels unless named latent."""

    bucket_hw: tuple[int, int]
    content_hw: tuple[int, int]
    latent_hw: tuple[int, int]
    spatial_compression: int

    @property
    def generation_hw(self) -> tuple[int, int]:
        """Pixel size of the generated content region (whole latent cells)."""
        return (
            self.latent_hw[0] * self.spatial_compression,
            self.latent_hw[1] * self.spatial_compression,
        )


def action_bucket_size(
    height: int, width: int, resolution: str | int
) -> tuple[int, int]:
    """``(target_w, target_h)`` of the tier's bucket whose ``h / w`` is closest."""
    resolution = str(resolution)
    if resolution not in ASPECT_RATIO_SIZE_INFO:
        raise ValueError(
            f"Unknown action resolution tier {resolution!r}; expected one of "
            f"{list(ASPECT_RATIO_SIZE_INFO)}."
        )
    input_ratio = height / width
    best: tuple[int, int] | None = None
    best_diff = float("inf")
    for cand_w, cand_h in ASPECT_RATIO_SIZE_INFO[resolution].values():
        diff = abs(input_ratio - cand_h / cand_w)
        if diff < best_diff:
            best_diff = diff
            best = (cand_w, cand_h)
    assert best is not None
    return best


def fit_action_video(
    frames: torch.Tensor,
    *,
    resolution: str | int,
    num_frames: int,
    spatial_compression: int = 16,
) -> tuple[torch.Tensor, ActionInputFit]:
    """Fit a ``uint8 [C, T, H, W]`` clip onto its resolution bucket like the
    reference action inference: repeat the last frame up to ``num_frames``,
    resize aspect-preserving without upscaling (bicubic, antialiased, on uint8),
    then pad bottom/right by reflection (edge when the pad is not smaller than
    the content). Returns float32 ``[C, num_frames, H_bucket, W_bucket]`` in
    ``[-1, 1]``; the encoded latent must be cropped to ``fit.latent_hw``.
    """
    import torchvision.transforms.functional as TF

    if frames.ndim != 4 or frames.dtype != torch.uint8:
        raise ValueError(
            "Action input frames must be uint8 [C, T, H, W], got "
            f"{frames.dtype} {tuple(frames.shape)}"
        )
    if frames.shape[1] < num_frames:
        pad = frames[:, -1:].repeat(1, num_frames - frames.shape[1], 1, 1)
        frames = torch.cat([frames, pad], dim=1)
    elif frames.shape[1] > num_frames:
        frames = frames[:, :num_frames]

    orig_h, orig_w = int(frames.shape[-2]), int(frames.shape[-1])
    target_w, target_h = action_bucket_size(orig_h, orig_w, resolution)
    scaling_ratio = min(target_w / orig_w, target_h / orig_h, 1.0)
    content_h = int(scaling_ratio * orig_h + 0.5)
    content_w = int(scaling_ratio * orig_w + 0.5)
    if (content_h, content_w) != (orig_h, orig_w):
        frames = TF.resize(
            frames,
            size=[content_h, content_w],
            interpolation=TF.InterpolationMode.BICUBIC,
            antialias=True,
        )
    if (content_h, content_w) != (target_h, target_w):
        pad_right, pad_bottom = target_w - content_w, target_h - content_h
        padding_mode = (
            "edge" if pad_right >= content_w or pad_bottom >= content_h else "reflect"
        )
        frames = TF.pad(
            frames, [0, 0, pad_right, pad_bottom], padding_mode=padding_mode
        )
    pixels = frames.to(torch.float32) / 127.5 - 1.0
    fit = ActionInputFit(
        bucket_hw=(target_h, target_w),
        content_hw=(content_h, content_w),
        latent_hw=(
            max(content_h // spatial_compression, 1),
            max(content_w // spatial_compression, 1),
        ),
        spatial_compression=spatial_compression,
    )
    return pixels, fit


ACTION_NORMALIZATION_METHODS = (
    "quantile",
    "quantile_rot",
    "quantile_rot_scale_floor",
    "asinh_rot",
    "piecewise_asinh_rot",
    "global_asinh",
    "meanstd",
    "minmax",
)
_ASINH_METHODS = frozenset({"asinh_rot", "global_asinh"})
# These methods read the rotation-preserving "global_raw" stats block.
_RAW_ROTATION_STATS_METHODS = frozenset(
    {"quantile_rot", "quantile_rot_scale_floor", "asinh_rot", "piecewise_asinh_rot"}
)
# Forward normalization clamps these affine range methods to [-1, 1], which is
# the reference's apply_forward_clamp=True; asinh variants and meanstd never clamp.
_FORWARD_CLAMP_METHODS = frozenset(
    {"quantile", "quantile_rot", "quantile_rot_scale_floor", "minmax"}
)
_QUANTILE_ROT_SCALE_FLOOR_TARGET_MAX = 5.0


def default_action_stats_key(method: str) -> str:
    """Stats block a normalization method reads by default."""
    return "global_raw" if method in _RAW_ROTATION_STATS_METHODS else "global"


def load_action_stats(
    stats_path: str, stats_key: str = "global"
) -> dict[str, torch.Tensor]:
    """Load per-channel action normalization stats from a JSON file.

    A missing ``stats_key`` falls back to top-level stats; a missing
    non-``global`` block without top-level stats is an error.
    """
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Action normalization stats not found at {stats_path}."
        )
    raw = json.loads(path.read_text())
    if stats_key in raw:
        raw = raw[stats_key]
        if not isinstance(raw, dict):
            raise TypeError(
                f"Action stats block {stats_key!r} in {stats_path} must be a dict."
            )
    elif stats_key != "global" and not any(key in raw for key in _STAT_KEYS):
        raise KeyError(f"Action stats block {stats_key!r} not found in {stats_path}.")
    return {
        k: torch.as_tensor(np.array(v, dtype=np.float32))
        for k, v in raw.items()
        if k in _STAT_KEYS
    }


class ActionNormalizer(NamedTuple):
    """Resolved per-channel normalizer: ``z = (x - offset) / scale`` followed by
    the method's transform (clamp, asinh or piecewise asinh)."""

    method: str
    offset: torch.Tensor
    scale: torch.Tensor

    def normalize(self, action: torch.Tensor) -> torch.Tensor:
        offset = self.offset.to(device=action.device, dtype=action.dtype)
        scale = self.scale.to(device=action.device, dtype=action.dtype)
        z = (action - offset) / scale
        if self.method in _ASINH_METHODS:
            unit = torch.as_tensor(
                math.asinh(1.0), device=action.device, dtype=action.dtype
            )
            return torch.asinh(z) / unit
        if self.method == "piecewise_asinh_rot":
            beta = torch.as_tensor(1.0, device=action.device, dtype=action.dtype)
            abs_z = z.abs()
            tail = 1.0 + torch.asinh(beta * (abs_z - 1.0)) / beta
            return torch.where(abs_z <= 1.0, z, z.sign() * tail)
        if self.method in _FORWARD_CLAMP_METHODS:
            return z.clamp(-1.0, 1.0)
        return z

    def denormalize(self, action: torch.Tensor) -> torch.Tensor:
        offset = self.offset.to(device=action.device, dtype=action.dtype)
        scale = self.scale.to(device=action.device, dtype=action.dtype)
        if self.method in _ASINH_METHODS:
            unit = torch.as_tensor(
                math.asinh(1.0), device=action.device, dtype=action.dtype
            )
            return torch.sinh(action * unit) * scale + offset
        if self.method == "piecewise_asinh_rot":
            beta = torch.as_tensor(1.0, device=action.device, dtype=action.dtype)
            abs_y = action.abs()
            tail = 1.0 + torch.sinh(beta * (abs_y - 1.0)) / beta
            z = torch.where(abs_y <= 1.0, action, action.sign() * tail)
            return z * scale + offset
        return action * scale + offset


def resolve_action_normalizer(
    method: str, stats: dict[str, torch.Tensor]
) -> ActionNormalizer:
    if method not in ACTION_NORMALIZATION_METHODS:
        raise ValueError(
            f"Unknown action normalization method {method!r}; expected one of "
            f"{list(ACTION_NORMALIZATION_METHODS)}."
        )
    if method == "global_asinh":
        for name in ("q01", "q99"):
            if tuple(stats[name].shape) != (UNIFIED_ACTION_DIM,):
                raise ValueError(
                    f"global_asinh stats {name!r} must have shape "
                    f"[{UNIFIED_ACTION_DIM}], got {tuple(stats[name].shape)}."
                )
    if method == "meanstd":
        return ActionNormalizer(method, stats["mean"], stats["std"].clamp(min=1e-8))
    if method == "minmax":
        lo, hi = stats["min"], stats["max"]
    else:
        lo, hi = stats["q01"], stats["q99"]
    offset = (hi + lo) / 2.0
    scale = (hi - lo).clamp(min=1e-8) / 2.0
    if method == "quantile_rot_scale_floor":
        min_dev = (stats["min"] - offset).abs()
        max_dev = (stats["max"] - offset).abs()
        scale_floor = (
            torch.maximum(min_dev, max_dev) / _QUANTILE_ROT_SCALE_FLOOR_TARGET_MAX
        )
        scale = torch.maximum(scale, scale_floor.clamp(min=1e-8))
    return ActionNormalizer(method, offset, scale)


def load_action_normalizer(method: str, stats_path: str) -> ActionNormalizer:
    return resolve_action_normalizer(
        method, load_action_stats(stats_path, default_action_stats_key(method))
    )


def normalize_action(
    action: torch.Tensor, method: str, stats: dict[str, torch.Tensor]
) -> torch.Tensor:
    return resolve_action_normalizer(method, stats).normalize(action)


def denormalize_action(
    action: torch.Tensor, method: str, stats: dict[str, torch.Tensor]
) -> torch.Tensor:
    return resolve_action_normalizer(method, stats).denormalize(action)


def denormalize_action_rows(
    action: torch.Tensor,
    *,
    method: str,
    stats_path: str | None,
    initial_state_stats_path: str | None,
    has_initial_state: bool,
) -> torch.Tensor:
    """Map predicted ``[..., T, D]`` actions back to raw units. Row 0 of an
    Image2Action stream (a0) uses its own stats; without stats the actions stay
    in model space."""
    if stats_path is None:
        return action
    base = load_action_normalizer(method, stats_path)
    if not has_initial_state:
        return base.denormalize(action)
    if initial_state_stats_path is None:
        raise ValueError(
            "predict_initial_state with action stats also needs the server's "
            "action_initial_state_stats_path for the a0 row."
        )
    initial_state = load_action_normalizer(method, initial_state_stats_path)
    row0 = initial_state.denormalize(action[..., :1, :])
    rest = base.denormalize(action[..., 1:, :])
    return torch.cat([row0, rest], dim=-2)


def select_action_stats_path(
    stats_path: str | dict[str, str] | None, domain_name: str | None
) -> str | None:
    """Server-side stats file for one request; a mapping is keyed by domain name."""
    if stats_path is None or isinstance(stats_path, str):
        return stats_path
    by_domain = {str(k).strip().lower(): v for k, v in stats_path.items()}
    key = (domain_name or "").strip().lower()
    if key not in by_domain:
        raise ValueError(
            f"No action stats configured for domain_name={domain_name!r}; "
            f"configured domains: {sorted(by_domain)}."
        )
    return by_domain[key]
