# SPDX-License-Identifier: Apache-2.0
"""Cosmos3-Nano-Transfer-Auto (Multiview-AV) pipeline configuration.

``transformer/config.json`` marks the checkpoint with
``backbone_type: cosmos3_multiview`` and a ``multiview`` object that carries the
deployment contract vLLM-Omni's ``multiview_config.py`` defines (HF da7c96b,
Oct 7 2026): the camera list, the past-only cross-view window, the rig view
embedding table, the request defaults, and for joint checkpoints the LiDAR
tokenizer metadata and the LiDAR latent patch. Everything older exports spelled
out is implied by the architecture: masked block-sparse attention with the
kernel left to the runtime, decomposed scope, controls reading their own view's
targets, LiDAR reading every caption, one caption per camera, camera subsets.
This config reuses ``Cosmos3Config`` (Wan VAE, Qwen2 tokenizer, FlowUniPC) and
swaps in the transformer subclass that installs the multiview attention.

Cosmos3.5 Ultra multiview checkpoints (``backbone_type:
cosmos3_5_qwen_3_vl_moe``) carry the same ``multiview`` block under i4's
attention contract instead (``COSMOS3_MULTIVIEW_I4_CONTRACT_FIELDS``): the
general cross-view window ``(start, end)`` with optional first-frame
visibility, an optional rig view embedding (the Ultra recipes train none), any
camera keys, and request defaults that fall back to i4's
``defaults/multiview/sample_args.json``. i4 has no exporter for that block yet;
``multiview_block_from_i4_model_config`` derives it from the i4 model config a
checkpoint's root ``config.json`` embeds (``model.config``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import msgspec

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import (
    Cosmos3Config,
    _transformer_config,
)
from sglang.multimodal_gen.configs.task_type import ModelTaskType
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

COSMOS3_MULTIVIEW_BACKBONE_TYPE = "cosmos3_multiview"
# The Cosmos3.5 Ultra (Qwen3-VL-MoE) backbone; its multiview checkpoints follow
# the i4 attention contract below rather than the vLLM-Omni Oct-7 one.
COSMOS3_5_ULTRA_BACKBONE_TYPE = "cosmos3_5_qwen_3_vl_moe"
COSMOS3_MULTIVIEW_BACKBONE_TYPES = (
    COSMOS3_MULTIVIEW_BACKBONE_TYPE,
    COSMOS3_5_ULTRA_BACKBONE_TYPE,
)
# Kernels behind the one attention pattern this model has: FlexAttention's Triton
# kernels or FlashAttention-4 block-sparse CuTe. Both count every permitted key
# once; the choice is a speed knob.
COSMOS3_MULTIVIEW_ATTENTION_BACKENDS = ("triton", "fa4")
# Left to the device: FA4 where its kernels run (Hopper, Blackwell), Triton elsewhere.
COSMOS3_MULTIVIEW_AUTO_BACKEND = "auto"
MULTIVIEW_BACKEND_ENV_VAR = "SGLANG_DIFFUSION_COSMOS3_MULTIVIEW_ATTENTION_BACKEND"
# The deployment contract, field for field what vLLM-Omni's multiview_config.py
# reads. Anything else fails loudly: a new field means a new architecture.
COSMOS3_MULTIVIEW_CONTRACT_FIELDS = frozenset(
    {
        "cameras",
        "cross_view_past_window_seconds",
        "rig_view_embedding",
        "lidar_latent_patch_size_hw",
        "lidar",
        "inference_defaults",
    }
)

# The i4 attention contract of the Ultra multiview recipes
# (``model.config.multiview_attention``), spelled as deployment fields:
# ``cross_view_temporal_window_bounds`` is ``mask.cross_view.temporal_window_bounds``,
# ``cross_view_attends_first_frame`` ``mask.cross_view.attends_first_frame``,
# ``attention_scope`` / ``lidar_attends_captions`` the mask fields of that name.
COSMOS3_MULTIVIEW_I4_CONTRACT_FIELDS = frozenset(
    {
        "cameras",
        "cross_view_past_window_seconds",
        "cross_view_temporal_window_bounds",
        "cross_view_attends_first_frame",
        "attention_scope",
        "rig_view_embedding",
        "lidar_latent_patch_size_hw",
        "lidar",
        "lidar_attends_captions",
        "inference_defaults",
    }
)
COSMOS3_MULTIVIEW_ATTENTION_SCOPES = ("all_views", "same_view", "decomposed")
# i4 ``defaults/multiview/sample_args.json`` (the standalone route's multiview
# defaults), which an i4-contract block's ``inference_defaults`` overrides.
COSMOS3_MULTIVIEW_I4_INFERENCE_DEFAULTS: dict[str, Any] = {
    "num_frames": 93,
    "aspect_ratio": "16,9",
    "resolution": "720",
    "fps": 10.0,
    "num_steps": 35,
    "guidance": 6.0,
    "shift": 10.0,
    "control_guidance": 1.0,
    "emphasize_control_in_prompt": True,
    "guidance_interval": None,
    "control_guidance_interval": None,
    "normalize_cfg": False,
}
# The multiview deployment schemas: the vLLM-Omni Oct-7 contract (Nano
# Transfer-Auto) and the i4 attention contract (Cosmos3.5 Ultra).
MULTIVIEW_SCHEMA_OCT7 = "vllm_omni_oct7"
MULTIVIEW_SCHEMA_I4 = "i4"

# The fixed 11-camera MADS rig order the v1 checkpoint was exported with.
COSMOS3_MADS_CAMERAS = (
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
    "camera_rear_left_70fov",
    "camera_cross_left_120fov",
    "camera_front_tele_30fov",
    "camera_front_fisheye_200fov",
    "camera_left_fisheye_200fov",
    "camera_right_fisheye_200fov",
    "camera_rear_fisheye_200fov",
)


# Stable camera attributes the per-camera caption headers are rendered from
# (the MADS rig description the separate-view-caption checkpoints trained on).
COSMOS3_MADS_CAMERA_ATTRIBUTES: dict[str, dict[str, str | int]] = {
    "camera_front_wide_120fov": {
        "camera_role": "front",
        "camera_type": "wide",
        "facing": "forward",
        "fov_degrees": 120,
    },
    "camera_cross_right_120fov": {
        "camera_role": "right_side",
        "camera_type": "wide",
        "facing": "right",
        "fov_degrees": 120,
    },
    "camera_rear_right_70fov": {
        "camera_role": "rear_right",
        "camera_type": "standard",
        "facing": "rear_right",
        "fov_degrees": 70,
    },
    "camera_rear_tele_30fov": {
        "camera_role": "rear",
        "camera_type": "telephoto",
        "facing": "backward",
        "fov_degrees": 30,
    },
    "camera_rear_left_70fov": {
        "camera_role": "rear_left",
        "camera_type": "standard",
        "facing": "rear_left",
        "fov_degrees": 70,
    },
    "camera_cross_left_120fov": {
        "camera_role": "left_side",
        "camera_type": "wide",
        "facing": "left",
        "fov_degrees": 120,
    },
    "camera_front_tele_30fov": {
        "camera_role": "front",
        "camera_type": "telephoto",
        "facing": "forward",
        "fov_degrees": 30,
    },
    "camera_front_fisheye_200fov": {
        "camera_role": "front",
        "camera_type": "fisheye",
        "facing": "forward",
        "fov_degrees": 200,
    },
    "camera_left_fisheye_200fov": {
        "camera_role": "left_side",
        "camera_type": "fisheye",
        "facing": "left",
        "fov_degrees": 200,
    },
    "camera_right_fisheye_200fov": {
        "camera_role": "right_side",
        "camera_type": "fisheye",
        "facing": "right",
        "fov_degrees": 200,
    },
    "camera_rear_fisheye_200fov": {
        "camera_role": "rear",
        "camera_type": "fisheye",
        "facing": "backward",
        "fov_degrees": 200,
    },
}

# Schema-2 exports carry these request defaults; the sampling params fall
# back to them for fields the request leaves unset.
COSMOS3_MULTIVIEW_INFERENCE_DEFAULT_KEYS = frozenset(
    {
        "resolution",
        "fps",
        "num_steps",
        "guidance",
        "shift",
        "control_guidance",
        "emphasize_control_in_prompt",
        "guidance_interval",
        "control_guidance_interval",
        "normalize_cfg",
    }
)
COSMOS3_MULTIVIEW_RESOLUTIONS = ("480", "720")
COSMOS3_MULTIVIEW_ASPECT_RATIO_LABELS = ("1,1", "4,3", "3,4", "16,9", "9,16")

# The V1.2 LiDAR tokenizer contract the joint checkpoints were exported with.
COSMOS3_LIDAR_TOKENIZER_VERSION = "1.2"
COSMOS3_LIDAR_RANGE_PROJECTION = {
    "semantic_width": 1800,
    "model_width": 1808,
    "native_height": 128,
    "model_width_transform": "circular_pad",
    "intensity_encoding": "unit",
}


class Cosmos3MultiviewDeploymentConfig(msgspec.Struct, frozen=True):
    """The validated ``multiview`` block of ``transformer/config.json``."""

    #: The exported rig. Empty under the i4 contract when the checkpoint names
    #: no rig: any unique camera keys are then accepted, as i4 inference does
    #: for checkpoints without rig view embeddings.
    cameras: tuple[str, ...]
    #: A key of another view is visible when captured at most this many seconds
    #: before the query (same instant included); 0 keeps cross-view attention to
    #: the same instant. None when the window is given as general bounds.
    cross_view_past_window_seconds: float | None
    #: Physical rig identity table: ``camera_ids`` maps every exported camera to
    #: its trained embedding row, ``lidar_id`` is the final row. None for
    #: checkpoints trained without rig view embeddings (i4 ``num_view_embeddings=0``).
    rig_view_embedding: dict[str, Any] | None
    inference_defaults: dict[str, Any]
    lidar: dict[str, Any] | None = None
    #: LiDAR token patch ``(height, width)`` in latent cells; present with ``lidar``.
    lidar_latent_patch_size_hw: tuple[int, int] | None = None
    #: i4's general cross-view window ``(start, end)`` in key-minus-query seconds.
    temporal_window_bounds: tuple[float, float] | None = None
    #: i4 ``mask.cross_view.attends_first_frame``.
    attends_first_frame: bool = False
    attention_scope: str = "decomposed"
    lidar_attends_captions: bool = True
    #: ``MULTIVIEW_SCHEMA_OCT7`` (Nano Transfer-Auto) or ``MULTIVIEW_SCHEMA_I4`` (Ultra).
    schema: str = MULTIVIEW_SCHEMA_OCT7

    @property
    def num_views(self) -> int:
        return len(self.cameras)

    @property
    def window_bounds(self) -> tuple[float, float] | None:
        """The cross-view window as ``(start, end)``, or None for the frame-index form."""
        if self.temporal_window_bounds is not None:
            return self.temporal_window_bounds
        if self.cross_view_past_window_seconds is not None:
            return (-float(self.cross_view_past_window_seconds), 0.0)
        return None

    def describe_window(self) -> str:
        bounds = self.window_bounds
        if bounds is None:
            return "same frame index"
        text = f"[{bounds[0]:+.2f}, {bounds[1]:+.2f}] s"
        return text + (" + first frame" if self.attends_first_frame else "")

    @property
    def is_i4_contract(self) -> bool:
        return self.schema == MULTIVIEW_SCHEMA_I4

    @property
    def requires_mads_cameras(self) -> bool:
        """Whether every camera must be a MADS key (Oct-7: rig header per caption)."""
        return not self.is_i4_contract

    def rig_view_ids(self, camera_keys: Sequence[str]) -> list[int] | None:
        """Trained embedding rows for the request's cameras, in request order.

        Subsets and reordered views keep each camera's physical id. None when the
        checkpoint carries no rig view embedding.
        """
        if self.rig_view_embedding is None:
            return None
        table = self.rig_view_embedding["camera_ids"]
        unknown = [key for key in camera_keys if key not in table]
        if unknown:
            raise ValueError(
                f"Cosmos3 multiview rig view embedding has no row for cameras {unknown}."
            )
        return [int(table[key]) for key in camera_keys]

    @property
    def supports_lidar(self) -> bool:
        return self.lidar is not None

    def inference_default(self, name: str, fallback: Any) -> Any:
        """A request default the export declares, else ``fallback``."""
        value = self.inference_defaults.get(name)
        return fallback if value is None else value


def _required_field(config: Mapping[str, Any], name: str) -> Any:
    if name not in config:
        raise ValueError(
            f"Cosmos3 multiview transformer config requires field {name!r}."
        )
    return config[name]


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(value)
    )


def validate_lidar_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exported V1.2 LiDAR tokenizer contract.

    ``version`` was written by the Sep/Oct-1 exporters and dropped by the Oct-7
    contract; when present it must name the V1.2 tokenizer.
    """
    required = {
        "fps",
        "latent_channels",
        "temporal_compression_factor",
        "spatial_compression",
        "network_config",
        "range_projection",
        "streaming_chunk_frames",
        "streaming_context_frames",
    }
    if missing := required - set(config):
        raise ValueError(
            f"Incomplete Cosmos3 LiDAR metadata: missing {sorted(missing)}."
        )
    if (
        "version" in config
        and str(config["version"]) != COSMOS3_LIDAR_TOKENIZER_VERSION
    ):
        raise ValueError(
            f"Only the V{COSMOS3_LIDAR_TOKENIZER_VERSION} LiDAR tokenizer is supported, "
            f"got {config['version']!r}."
        )
    projection = config["range_projection"]
    if not isinstance(projection, Mapping):
        raise TypeError("Cosmos3 LiDAR range_projection must be an object.")
    mismatched = {
        key: projection.get(key)
        for key, value in COSMOS3_LIDAR_RANGE_PROJECTION.items()
        if projection.get(key) != value
    }
    if mismatched:
        raise ValueError(
            "Cosmos3 LiDAR V1.2 requires the 128x1800 metric grid circularly padded "
            f"to 1808 with unit intensity; got {mismatched}."
        )
    minimum, maximum = projection.get("min_range_m"), projection.get("max_range_m")
    if not (_is_number(minimum) and _is_number(maximum)) or maximum <= minimum:
        raise ValueError(
            "Cosmos3 LiDAR range normalization requires finite min_range_m < max_range_m."
        )
    network = config["network_config"]
    if not isinstance(network, Mapping):
        raise TypeError("Cosmos3 LiDAR network_config must be an object.")
    spatial = list(config["spatial_compression"])
    if (
        list(network.get("resolution", ())) != [128, 1808]
        or list(network.get("patch_size", ())) != [2, 2]
        or len(network.get("depths", ())) != 4
        or spatial != [16, 16]
        or int(config["temporal_compression_factor"]) != 1
        or network.get("z_dim") != config["latent_channels"]
        or network.get("in_channels") != 3
        or any(network.get("temporal_downsample", [True]))
    ):
        raise ValueError(
            "Cosmos3 LiDAR architecture and V1.2 compression metadata disagree."
        )
    if not _is_number(config["fps"]) or config["fps"] <= 0:
        raise ValueError("Cosmos3 LiDAR fps must be finite and positive.")
    chunk, context = (
        config["streaming_chunk_frames"],
        config["streaming_context_frames"],
    )
    if (
        isinstance(chunk, bool)
        or not isinstance(chunk, int)
        or chunk < 1
        or (
            context is not None
            and (
                isinstance(context, bool)
                or not isinstance(context, int)
                or context < chunk
            )
        )
    ):
        raise ValueError(
            "Cosmos3 LiDAR streaming context must be at least the positive chunk length."
        )
    return dict(config)


def _validate_inference_defaults(
    raw: Any, *, schema: str = MULTIVIEW_SCHEMA_OCT7
) -> dict[str, Any]:
    if schema == MULTIVIEW_SCHEMA_I4:
        # Absent keys fall back to i4's own multiview defaults.
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise TypeError("Cosmos3 multiview inference_defaults must be an object.")
        if unknown := set(raw) - set(COSMOS3_MULTIVIEW_I4_INFERENCE_DEFAULTS):
            raise ValueError(
                f"Unknown Cosmos3 multiview inference_defaults {sorted(unknown)}."
            )
        raw = {**COSMOS3_MULTIVIEW_I4_INFERENCE_DEFAULTS, **raw}
        num_frames = raw["num_frames"]
        if not _positive_int(num_frames) or num_frames < 2:
            raise ValueError(
                "Cosmos3 multiview inference_defaults.num_frames must be an integer > 1, "
                f"got {num_frames!r}."
            )
        if str(raw["aspect_ratio"]) not in COSMOS3_MULTIVIEW_ASPECT_RATIO_LABELS:
            raise ValueError(
                "Cosmos3 multiview inference_defaults.aspect_ratio must be one of "
                f"{list(COSMOS3_MULTIVIEW_ASPECT_RATIO_LABELS)}, got {raw['aspect_ratio']!r}."
            )
    if not isinstance(raw, Mapping):
        raise TypeError("Cosmos3 multiview inference_defaults must be an object.")
    if missing := COSMOS3_MULTIVIEW_INFERENCE_DEFAULT_KEYS - set(raw):
        raise ValueError(
            f"Incomplete Cosmos3 multiview inference_defaults: missing {sorted(missing)}."
        )
    if str(raw["resolution"]) not in COSMOS3_MULTIVIEW_RESOLUTIONS:
        raise ValueError(
            "Cosmos3 multiview inference_defaults.resolution must be one of "
            f"{list(COSMOS3_MULTIVIEW_RESOLUTIONS)}, got {raw['resolution']!r}."
        )
    numeric = ["fps", "num_steps", "guidance", "shift", "control_guidance"]
    if "sigma_max" in raw:
        numeric.append("sigma_max")
    for name in numeric:
        if not _is_number(raw[name]) or raw[name] < 0:
            raise ValueError(
                f"Cosmos3 multiview inference_defaults.{name} must be finite and non-negative."
            )
    if raw["fps"] == 0 or raw["num_steps"] < 1 or raw["shift"] == 0:
        raise ValueError(
            "Cosmos3 multiview inference_defaults requires positive fps, num_steps, and shift."
        )
    defaults = dict(raw)
    defaults["resolution"] = str(defaults["resolution"])
    return defaults


def _positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _validated_lidar_patch(raw: Mapping[str, Any]) -> tuple[int, int] | None:
    """The LiDAR stream's ``(height, width)`` patch; required with, and only with, ``lidar``."""
    name = "lidar_latent_patch_size_hw"
    if raw.get("lidar") is None:
        if name in raw:
            raise ValueError(f"Cosmos3 multiview {name} requires a lidar block.")
        return None
    patch = _required_field(raw, name)
    if (
        not isinstance(patch, list | tuple)
        or len(patch) != 2
        or not all(_positive_int(side) for side in patch)
    ):
        raise ValueError(
            f"Cosmos3 multiview {name} must be two positive integers, got {patch!r}."
        )
    return (int(patch[0]), int(patch[1]))


def _validated_rig_view_embedding(
    raw: Mapping[str, Any], cameras: Sequence[str], *, required: bool = True
) -> dict[str, Any] | None:
    """Validate the physical rig-id table: one row per exported camera plus a final LiDAR row."""
    if not required and raw.get("rig_view_embedding") is None:
        return None
    rig = _required_field(raw, "rig_view_embedding")
    if not isinstance(rig, Mapping):
        raise TypeError("Cosmos3 multiview rig_view_embedding must be an object.")
    if unknown := set(rig) - {"num_embeddings", "camera_ids", "lidar_id"}:
        raise ValueError(
            f"Unknown Cosmos3 multiview rig_view_embedding fields: {sorted(unknown)}."
        )
    num_embeddings = rig.get("num_embeddings")
    if not _positive_int(num_embeddings) or num_embeddings < 2:
        raise ValueError(
            "Cosmos3 multiview rig_view_embedding.num_embeddings must be an integer "
            f">= 2, got {num_embeddings!r}."
        )
    camera_ids = rig.get("camera_ids")
    if not isinstance(camera_ids, Mapping):
        raise TypeError(
            "Cosmos3 multiview rig_view_embedding.camera_ids must be an object."
        )
    if set(camera_ids) != set(cameras):
        raise ValueError(
            "Cosmos3 multiview rig_view_embedding.camera_ids must name exactly the "
            f"exported cameras: expected={sorted(cameras)}, got={sorted(camera_ids)}."
        )
    for camera, row in camera_ids.items():
        # The final row is reserved for LiDAR.
        if (
            isinstance(row, bool)
            or not isinstance(row, int)
            or not 0 <= row <= num_embeddings - 2
        ):
            raise ValueError(
                f"Cosmos3 multiview rig_view_embedding.camera_ids[{camera!r}] must be an "
                f"integer in [0, {num_embeddings - 2}], got {row!r}."
            )
    if len(set(camera_ids.values())) != len(camera_ids):
        raise ValueError(
            "Cosmos3 multiview rig_view_embedding.camera_ids must assign distinct rows."
        )
    lidar_id = rig.get("lidar_id")
    if (
        isinstance(lidar_id, bool)
        or not isinstance(lidar_id, int)
        or lidar_id != num_embeddings - 1
    ):
        raise ValueError(
            "Cosmos3 multiview rig_view_embedding.lidar_id must be the final row "
            f"{num_embeddings - 1}, got {lidar_id!r}."
        )
    return {
        "num_embeddings": int(num_embeddings),
        "camera_ids": {str(camera): int(row) for camera, row in camera_ids.items()},
        "lidar_id": int(lidar_id),
    }


def parse_multiview_deployment_config(
    transformer_config: Mapping[str, Any],
) -> Cosmos3MultiviewDeploymentConfig:
    """Validate the exported multiview contract before any weights are loaded.

    The backbone is inspected first so selecting this pipeline for another
    Cosmos3 variant reports the actual mismatch. ``cosmos3_multiview`` exports
    serve the vLLM-Omni Oct-7 contract; Cosmos3.5 Ultra exports the i4 one.
    """
    backbone_type = transformer_config.get("backbone_type")
    if backbone_type not in COSMOS3_MULTIVIEW_BACKBONE_TYPES:
        raise ValueError(
            "Cosmos3MultiviewPipeline requires transformer/config.json "
            f"backbone_type in {list(COSMOS3_MULTIVIEW_BACKBONE_TYPES)}, "
            f"got {backbone_type!r}."
        )
    raw = transformer_config.get("multiview")
    if not isinstance(raw, Mapping):
        raise ValueError(
            "Cosmos3 multiview transformer config must contain a 'multiview' object."
        )
    if backbone_type == COSMOS3_5_ULTRA_BACKBONE_TYPE:
        return _parse_i4_multiview_contract(raw)
    if unknown := set(raw) - COSMOS3_MULTIVIEW_CONTRACT_FIELDS:
        raise ValueError(
            f"Unknown Cosmos3 multiview contract fields {sorted(unknown)}; this build "
            "serves the Oct-7 2026 contract only (HF da7c96b). Re-export the checkpoint."
        )

    window = _required_field(raw, "cross_view_past_window_seconds")
    if isinstance(window, bool) or not isinstance(window, int | float):
        raise TypeError(
            "Cosmos3 multiview cross_view_past_window_seconds must be a number."
        )
    if not math.isfinite(window) or window < 0:
        raise ValueError(
            "Cosmos3 multiview cross_view_past_window_seconds must be finite and "
            "non-negative."
        )

    cameras = _required_field(raw, "cameras")
    if (
        not isinstance(cameras, list)
        or not cameras
        or not all(isinstance(camera, str) and camera for camera in cameras)
    ):
        raise TypeError(
            "Cosmos3 multiview cameras must be a non-empty list of strings."
        )
    if len(cameras) != len(set(cameras)):
        raise ValueError("Cosmos3 multiview cameras must be unique.")
    unlabeled = [c for c in cameras if c not in COSMOS3_MADS_CAMERA_ATTRIBUTES]
    if unlabeled:
        raise ValueError(
            "Cosmos3 multiview per-camera captions need rig attributes for every "
            f"exported camera; none are known for {unlabeled}."
        )

    rig_view_embedding = _validated_rig_view_embedding(raw, cameras)
    inference_defaults = _validate_inference_defaults(
        _required_field(raw, "inference_defaults")
    )
    lidar = raw.get("lidar")
    if lidar is not None:
        lidar = validate_lidar_config(lidar)
    lidar_patch = _validated_lidar_patch(raw)

    return Cosmos3MultiviewDeploymentConfig(
        cameras=tuple(cameras),
        cross_view_past_window_seconds=float(window),
        rig_view_embedding=rig_view_embedding,
        inference_defaults=inference_defaults,
        lidar=lidar,
        lidar_latent_patch_size_hw=lidar_patch,
    )


def _parse_cameras(raw: Mapping[str, Any], *, required: bool) -> tuple[str, ...]:
    if not required and raw.get("cameras") is None:
        return ()
    cameras = _required_field(raw, "cameras")
    if (
        not isinstance(cameras, list)
        or not cameras
        or not all(isinstance(camera, str) and camera for camera in cameras)
    ):
        raise TypeError(
            "Cosmos3 multiview cameras must be a non-empty list of strings."
        )
    if len(cameras) != len(set(cameras)):
        raise ValueError("Cosmos3 multiview cameras must be unique.")
    return tuple(cameras)


def _parse_i4_multiview_contract(
    raw: Mapping[str, Any],
) -> Cosmos3MultiviewDeploymentConfig:
    """The i4-contract ``multiview`` block of a Cosmos3.5 Ultra export.

    Mirrors i4 ``MultiviewAttentionMaskConfig``: ``attention_scope``, a cross-view
    window given either as i4's ``(start, end)`` bounds or the past-only seconds
    of the Oct-7 contract, ``attends_first_frame`` (requires the decomposed
    scope and a window, as i4's ``FlexMetadata``), and ``lidar_attends_captions``.
    """
    if unknown := set(raw) - COSMOS3_MULTIVIEW_I4_CONTRACT_FIELDS:
        raise ValueError(
            f"Unknown Cosmos3 multiview contract fields {sorted(unknown)} for the "
            "Cosmos3.5 i4 multiview contract."
        )
    from sglang.multimodal_gen.runtime.models.dits.cosmos3_simulation.multiview_layout import (
        resolve_temporal_window_bounds,
    )

    scope = raw.get("attention_scope", "decomposed")
    if scope not in COSMOS3_MULTIVIEW_ATTENTION_SCOPES:
        raise ValueError(
            "Cosmos3 multiview attention_scope must be one of "
            f"{list(COSMOS3_MULTIVIEW_ATTENTION_SCOPES)}, got {scope!r}."
        )
    past = raw.get("cross_view_past_window_seconds")
    if past is not None and (
        isinstance(past, bool) or not isinstance(past, int | float)
    ):
        raise TypeError(
            "Cosmos3 multiview cross_view_past_window_seconds must be a number."
        )
    bounds = resolve_temporal_window_bounds(
        None if past is None else float(past),
        raw.get("cross_view_temporal_window_bounds"),
    )
    first_frame = raw.get("cross_view_attends_first_frame", False)
    if not isinstance(first_frame, bool):
        raise TypeError(
            "Cosmos3 multiview cross_view_attends_first_frame must be boolean."
        )
    if first_frame and (scope != "decomposed" or bounds is None):
        raise ValueError(
            "Cosmos3 multiview cross_view_attends_first_frame requires the decomposed "
            "scope and a cross-view temporal window."
        )
    lidar_attends_captions = raw.get("lidar_attends_captions", True)
    if not isinstance(lidar_attends_captions, bool):
        raise TypeError("Cosmos3 multiview lidar_attends_captions must be boolean.")
    cameras = _parse_cameras(raw, required=False)
    rig = raw.get("rig_view_embedding")
    rig_view_embedding = None
    if rig is not None:
        if not isinstance(rig, Mapping) or not isinstance(
            rig.get("camera_ids"), Mapping
        ):
            raise TypeError(
                "Cosmos3 multiview rig_view_embedding must be an object with camera_ids."
            )
        rig_view_embedding = _validated_rig_view_embedding(
            raw, cameras or tuple(rig["camera_ids"])
        )
    lidar = raw.get("lidar")
    if lidar is not None:
        lidar = validate_lidar_config(lidar)
    return Cosmos3MultiviewDeploymentConfig(
        cameras=cameras,
        cross_view_past_window_seconds=None if past is None else float(past),
        rig_view_embedding=rig_view_embedding,
        inference_defaults=_validate_inference_defaults(
            raw.get("inference_defaults"), schema=MULTIVIEW_SCHEMA_I4
        ),
        lidar=lidar,
        lidar_latent_patch_size_hw=_validated_lidar_patch(raw),
        temporal_window_bounds=None if past is not None else bounds,
        attends_first_frame=first_frame,
        attention_scope=scope,
        lidar_attends_captions=lidar_attends_captions,
        schema=MULTIVIEW_SCHEMA_I4,
    )


def _plain(value: Any) -> Any:
    """JSON-plain copy of a saved config node, dropping ``_target_``-style keys."""
    if isinstance(value, Mapping):
        return {
            str(k): _plain(v) for k, v in value.items() if not str(k).startswith("_")
        }
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def _saved_cross_view(mask: Mapping[str, Any]) -> tuple[Any, bool]:
    """``(temporal_window_bounds, attends_first_frame)`` of a saved i4 mask config.

    Mirrors i4 ``upgrade_saved_multiview_attention``: the current
    ``mask.cross_view`` block, else the earlier ``cross_view_temporal_window``
    (``bounds``, ``includes_first_frame``), else ``decomposed_temporal_window_seconds``
    (a scalar ``W`` meaning ``(-W, 0)``).
    """
    cross_view = mask.get("cross_view")
    if isinstance(cross_view, Mapping):
        return cross_view.get("temporal_window_bounds"), bool(
            cross_view.get("attends_first_frame", False)
        )
    window = mask.get("cross_view_temporal_window")
    if isinstance(window, Mapping):
        return window.get("bounds"), bool(window.get("includes_first_frame", False))
    seconds = mask.get("decomposed_temporal_window_seconds")
    first_frame = bool(
        mask.get("decomposed_temporal_window_includes_first_frame", False)
    )
    if _is_number(seconds):
        if seconds < 0:
            raise ValueError(
                "Saved decomposed_temporal_window_seconds must be non-negative."
            )
        return [-float(seconds), 0.0], first_frame
    return seconds, first_frame


def multiview_block_from_i4_model_config(
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    """The i4-contract ``multiview`` block an Ultra multiview checkpoint implies.

    ``model_config`` is i4's ``model.config`` as a checkpoint's root
    ``config.json`` embeds it. i4's Diffusers exporter writes the multiview block
    for the Nano Transfer-Auto contract only, so an Ultra multiview export has to
    carry this block in ``transformer/config.json`` itself. Rig rows need a
    camera table, which the model config does not name, so a recipe with
    ``num_view_embeddings`` is refused here.
    """
    if model_config.get("joint_attn_implementation") != "multiview":
        raise ValueError(
            "The i4 model config does not train multiview attention "
            f"(joint_attn_implementation={model_config.get('joint_attn_implementation')!r})."
        )
    attention = model_config.get("multiview_attention") or {}
    geometry = attention.get("geometry_position_encoding", "baseline")
    if geometry != "baseline":
        raise ValueError(
            f"Multiview geometry_position_encoding={geometry!r} (RigRoPE/PRoPE) is not served; "
            "the reference inference route refuses camera-geometry sidecars."
        )
    mask = attention.get("mask") or {}
    configured = sorted(
        name
        for name in (
            "sensor_to_sensor_window",
            "sensor_to_control_window",
            "control_to_control_window",
            "control_to_sensor_window",
            "same_view_temporal_window",
        )
        if mask.get(name) is not None
    )
    if configured:
        raise ValueError(
            f"Saved same-view temporal windows {configured} are not supported."
        )
    bounds, first_frame = _saved_cross_view(mask)
    block: dict[str, Any] = {
        # i4 MultiviewAttentionMaskConfig defaults: all views, captions on.
        "attention_scope": mask.get("attention_scope", "all_views"),
        "cross_view_attends_first_frame": first_frame,
        "lidar_attends_captions": bool(mask.get("lidar_attends_captions", True)),
    }
    if bounds is not None:
        block["cross_view_temporal_window_bounds"] = [float(b) for b in bounds]
    expert = model_config.get("diffusion_expert_config") or {}
    if expert.get("num_view_embeddings"):
        raise ValueError(
            "The recipe trains rig view embeddings; add rig_view_embedding with its "
            "camera table to the multiview block by hand."
        )
    if model_config.get("radar_state_ch") is not None:
        raise ValueError("Radar streams are not served (no radar tokenizer export).")
    tokenizer = model_config.get("lidar_tokenizer")
    if model_config.get("lidar_state_ch") is not None and tokenizer:
        patch = expert.get("lidar_patch_spatial_hw") or expert.get("patch_spatial", 2)
        if isinstance(patch, int):
            patch = [patch, patch]
        # i4 tools/export/multiview_export.build_lidar_export_config.
        block["lidar"] = {
            "dtype": "float32",
            "sample_posterior": False,
            "apply_validity_mask": bool(tokenizer.get("apply_validity_mask", True)),
            "fps": float(model_config["lidar_fps"]),
            "latent_channels": int(model_config["lidar_state_ch"]),
            "temporal_compression_factor": 1,
            "spatial_compression": [16, 16],
            "network_config": _plain(tokenizer["network_config"]),
            "range_projection": _plain(tokenizer["range_projection"]),
            "streaming_chunk_frames": tokenizer["streaming_chunk_frames"],
            "streaming_context_frames": tokenizer.get("streaming_context_frames"),
        }
        block["lidar_latent_patch_size_hw"] = [int(side) for side in patch]
    return block


def multiview_attention_heads(
    transformer_config: Mapping[str, Any],
) -> tuple[int, int] | None:
    """``(query heads, GEN key/value heads)`` the GEN attention runs with, if declared."""
    heads = transformer_config.get("num_attention_heads")
    kv_heads = transformer_config.get(
        "gen_num_key_value_heads"
    ) or transformer_config.get("num_key_value_heads")
    if not (_positive_int(heads) and _positive_int(kv_heads)):
        return None
    return int(heads), int(kv_heads)


def resolve_multiview_backend_request(explicit: str | None) -> tuple[str, str]:
    """Explicit config field, then the env override, then ``auto``; with its source.

    ``auto`` is resolved by the pipeline against the device it runs on. Shared
    by every pipeline that runs the multiview kernels (Transfer-Auto, Sim-Auto).
    """
    backend = explicit
    source = "pipeline config multiview_attention_backend"
    if backend is None:
        backend = envs.SGLANG_DIFFUSION_COSMOS3_MULTIVIEW_ATTENTION_BACKEND
        source = MULTIVIEW_BACKEND_ENV_VAR
    if not backend:
        backend = COSMOS3_MULTIVIEW_AUTO_BACKEND
        source = "device default"
    allowed = (*COSMOS3_MULTIVIEW_ATTENTION_BACKENDS, COSMOS3_MULTIVIEW_AUTO_BACKEND)
    if backend not in allowed:
        raise ValueError(
            "Cosmos3 multiview attention backend must be one of "
            f"{list(allowed)}, got {backend!r} (from {source})."
        )
    return backend, source


@dataclass
class Cosmos3MultiviewConfig(Cosmos3Config):
    """Cosmos3 Multiview-AV: 11-camera WSM-to-RGB transfer in one denoising pass."""

    # Single-task pipeline; the parent's multi-task declaration (TI2V, T2I, V2V)
    # does not apply, so requests resolve to task_type alone.
    supported_task_types: ClassVar[tuple[ModelTaskType, ...] | None] = None

    # Same weights, different GEN cross-attention: the loader instantiates the
    # multiview transformer subclass instead of the checkpoint's class name.
    transformer_class_override: str | None = "Cosmos3MultiviewTransformer"

    # The multiview tokenization stage owns prompt formatting: transfer system
    # prompt, duration/resolution sentences, and the WSM emphasis sentence.
    use_duration_template: bool = True
    use_system_prompt: bool = True

    # Kernel behind the multiview attention: "triton", "fa4", or None for the
    # env override / device default. Both kernels compute the same attention, so
    # this is safe for A/B measurement.
    multiview_attention_backend: str | None = None

    # Parsed once from transformer/config.json in update_config_from_dict.
    multiview_deployment: Cosmos3MultiviewDeploymentConfig | None = None
    # (query heads, GEN key/value heads) of the checkpoint, which bound the
    # Ulysses degree the multiview attention can shard heads over.
    multiview_attention_heads: tuple[int, int] | None = None

    def update_config_from_dict(self, args, prefix: str = "") -> None:
        super().update_config_from_dict(args, prefix)
        if self.model_path:
            transformer_config = _transformer_config(self.model_path)
            self.multiview_deployment = parse_multiview_deployment_config(
                transformer_config
            )
            self.multiview_attention_heads = multiview_attention_heads(
                transformer_config
            )
            if self.distilled_sigmas is not None:
                raise ValueError(
                    "Cosmos3 multiview expects the FlowUniPC schedule; a distilled "
                    "fixed-step scheduler is not supported."
                )
            # Fail on a bad backend name or family at launch, not on the first request.
            backend, source = self._resolve_multiview_backend()
            logger.info(
                "Cosmos3 multiview attention backend: %s (from %s)", backend, source
            )

    def _resolve_multiview_backend(self) -> tuple[str, str]:
        if self.multiview_deployment is None:
            raise ValueError(
                "Cosmos3 multiview deployment config has not been resolved; "
                "set model_path first."
            )
        return resolve_multiview_backend_request(self.multiview_attention_backend)

    def resolved_multiview_backend(self) -> str:
        """``triton``, ``fa4``, or ``auto`` (the device decides)."""
        return self._resolve_multiview_backend()[0]

    def validate_server_args(self, server_args: Any) -> None:
        super().validate_server_args(server_args)
        # Ulysses shards the GEN stream over ranks and trades sequence for heads
        # around every attention call; the mask spans the whole camera-major
        # sequence, so ring attention and tensor parallelism are not supported.
        # CFG parallel composes with it: each branch runs on its own SP group.
        for name, value in (
            ("tp_size", server_args.tp_size),
            ("ring_degree", server_args.ring_degree),
        ):
            if int(value or 1) > 1:
                raise ValueError(
                    f"Cosmos3 multiview does not support {name} > 1; use "
                    "--ulysses-degree for sequence parallelism and "
                    "--enable-cfg-parallel for the two CFG branches."
                )
        ulysses = int(server_args.ulysses_degree or 1)
        sp_degree = int(server_args.sp_degree or 1)
        if sp_degree != ulysses:
            raise ValueError(
                "Cosmos3 multiview sequence parallelism is Ulysses only: sp_degree "
                f"({sp_degree}) must equal ulysses_degree ({ulysses})."
            )
        if ulysses > 1:
            if self.multiview_attention_heads is None:
                # Nano Transfer-Auto (32 query / 8 key-value heads) when the
                # checkpoint has not been read yet.
                if ulysses not in (2, 4, 8):
                    raise ValueError(
                        "Cosmos3 multiview --ulysses-degree must divide the 8 key/value "
                        f"heads (2, 4 or 8), got {ulysses}."
                    )
            else:
                from sglang.multimodal_gen.runtime.models.dits.cosmos3_multiview import (
                    validate_multiview_ulysses_degree,
                )

                validate_multiview_ulysses_degree(
                    ulysses, *self.multiview_attention_heads
                )

    def adjust_num_frames(self, num_frames: int, *, log_adjustment: bool = True) -> int:
        """i4-contract exports round *up* to the 4k+1 grid, as i4 inference does.

        Oct-7 exports keep the Cosmos3 rounding (down).
        """
        deployment = self.multiview_deployment
        if deployment is None or not deployment.is_i4_contract or num_frames <= 1:
            return super().adjust_num_frames(num_frames, log_adjustment=log_adjustment)
        return (num_frames - 1 + 3) // 4 * 4 + 1

    def supports_action_endpoint(self) -> bool:
        return False

    def supports_dynamic_batching(self) -> bool:
        return False


def register():
    from sglang.multimodal_gen.configs.sample.cosmos3_multiview import (
        Cosmos3MultiviewSamplingParams,
    )
    from sglang.multimodal_gen.registry import register_configs

    # Cosmos3 Multiview-AV: 11-camera WSM transfer with maskless cross-camera
    # attention. The hf path is the pre-release folder name, which must sort
    # before the shorter "nvidia/Cosmos3-Nano" in the partial-path match.
    register_configs(
        sampling_param_cls=Cosmos3MultiviewSamplingParams,
        pipeline_config_cls=Cosmos3MultiviewConfig,
        hf_model_paths=["nvidia/Cosmos3-Nano-Transfer-Auto"],
        # Matches the ``Cosmos3MultiviewPipeline`` ``_class_name`` of the checkpoint.
        model_detectors=[lambda hf_id: "cosmos3multiviewpipeline" in hf_id.lower()],
    )
