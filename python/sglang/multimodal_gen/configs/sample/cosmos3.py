# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 sampling parameters.

A single ``SamplingParams`` class serves T2V, I2V, V2V, T2I, and
action-conditioned variants.  Per-request mode is dispatched in the pipeline
from ``num_frames`` (``== 1`` → T2I), ``image_path`` (set → I2V),
``video_path`` (set → V2V), and ``action_mode`` (set → action-conditioned).
For ``num_frames == 1`` the output ``data_type`` flips to ``IMAGE``
so the file extension and decode path agree.
"""

import json
import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

import msgspec

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.task_type import DataType, ModelTaskType
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

COSMOS3_DEFAULT_GUIDANCE_SCALE = 4.0
# Transfer hint keys in imaginaire4 ``TransferHintKey`` order: controls are packed,
# and named in the emphasis sentence, in this order whatever the request order.
COSMOS3_TRANSFER_HINT_KEYS = ("edge", "blur", "depth", "seg", "wsm")
# Hints imaginaire4 can compute from the input video when no control_path is given.
COSMOS3_TRANSFER_ON_THE_FLY_HINTS = frozenset({"edge", "blur"})
COSMOS3_TRANSFER_PRESET_FIELDS = {
    "edge": (
        "preset_edge_threshold",
        ("very_low", "low", "medium", "high", "very_high"),
    ),
    "blur": (
        "preset_blur_strength",
        ("none", "very_low", "low", "medium", "high", "very_high"),
    ),
}
COSMOS3_EDGE_T2I_GUIDANCE_SCALE = 7.0
COSMOS3_EDGE_T2V_GUIDANCE_SCALE = 5.0
COSMOS3_EDGE_T2V_WIDTH = 832
COSMOS3_EDGE_T2V_HEIGHT = 480
COSMOS3_EDGE_T2I_SIZE = 640
# Spatial downsampling of the Wan VAE; Cosmos3LatentPreparationStage rejects
# other sizes.
COSMOS3_SIZE_MULTIPLE = 16

# Image generation applies guidance only over a high-noise window; guiding the
# low-noise steps degrades sample quality (Kynkaeaenniemi et al. 2024). Video
# modes guide at every step.
COSMOS3_T2I_GUIDANCE_INTERVAL = (400.0, 1000.0)

# Edge is trained at 256p/480p only; larger frames push the spatial mRoPE grid
# past its trained range and shatter the output.
COSMOS3_EDGE_SUPPORTED_RESOLUTIONS = [
    (832, 480),
    (480, 832),
    (640, 480),
    (480, 640),
    (480, 480),
    (640, 640),
    (448, 256),
    (256, 448),
    (256, 256),
]


def _parse_request_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return value


def _optional_int_list(value: Any) -> list[int] | None:
    value = _parse_request_value(value)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


class Cosmos3TransferHint(msgspec.Struct, frozen=True):
    """One control input of a transfer request.

    ``key`` is None only for the untyped ``control_path`` form; ``control_path``
    is None when an edge/blur control is computed from the input video.
    """

    key: str | None
    control_path: str | None = None
    weight: float = 1.0
    preset: str = "medium"


def _transfer_weight(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got {value!r}")
    try:
        weight = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(weight) or weight < 0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return weight


def parse_transfer_hint_object(key: str, value: Any) -> Cosmos3TransferHint:
    """One per-hint request object (``edge: {...}``) as imaginaire4 accepts it.

    ``true`` / ``{}`` select the hint with defaults; a string is its
    ``control_path``; an object may carry ``control_path``, ``weight`` and the
    hint's preset (``preset_edge_threshold`` / ``preset_blur_strength``).
    """
    value = _parse_request_value(value)
    if value is True:
        value = {}
    elif isinstance(value, str):
        value = {"control_path": value}
    if not isinstance(value, dict):
        raise ValueError(
            f"Cosmos3 transfer hint {key!r} must be an object, a control path or true, "
            f"got {value!r}."
        )
    preset_field, presets = COSMOS3_TRANSFER_PRESET_FIELDS.get(key, (None, ()))
    allowed = {"control_path", "weight"} | ({preset_field} if preset_field else set())
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            f"Unsupported Cosmos3 transfer hint {key!r} fields {sorted(unknown)}; "
            f"supported: {sorted(allowed)}."
        )
    control_path = value.get("control_path")
    if control_path is not None and (
        not isinstance(control_path, str) or not control_path.strip()
    ):
        raise ValueError(f"{key}.control_path must be a non-empty string.")
    weight = value.get("weight")
    preset = value.get(preset_field) if preset_field else None
    if preset is None:
        preset = "medium"
    elif preset not in presets:
        raise ValueError(
            f"{key}.{preset_field} must be one of {list(presets)}, got {preset!r}."
        )
    return Cosmos3TransferHint(
        key=key,
        control_path=control_path,
        weight=1.0 if weight is None else _transfer_weight(weight, f"{key}.weight"),
        preset=str(preset),
    )


@dataclass
class Cosmos3SamplingParams(SamplingParams):
    """Cosmos3 sampling parameters (T2V defaults; also used for I2V / V2V / T2I).

    ``height``/``width`` default to ``None`` so the variant (Edge vs. base) can
    pick the right resolution at request time in
    :meth:`_resolve_variant_defaults`.
    """

    _explicit_fields: ClassVar[set[str] | None] = None
    # Single-view transfer request lowering (WSM length/fps defaults; otherwise the
    # output follows the control clip unless the video request sets num_frames or
    # seconds). Subclasses with their own frame contract opt out.
    _single_view_transfer: ClassVar[bool] = True

    height: int | None = None
    width: int | None = None
    num_frames: int = 81
    fps: int = 24

    guidance_scale: float = COSMOS3_DEFAULT_GUIDANCE_SCALE
    num_inference_steps: int = 35

    negative_prompt: str = ""

    use_duration_template: bool | None = None
    use_resolution_template: bool | None = None
    use_system_prompt: bool | None = None
    use_guardrails: bool | None = None
    sound_duration: float = 0.0

    # Optional CFG window — T2I requests typically pass e.g. ``(400, 1000)`` to
    # skip guidance at low noise levels. T2V / I2V / V2V leave it unset.
    guidance_interval: tuple[float, float] | None = None

    # V2V conditioning: which latent-frame indices stay locked to the input
    # video. ``None`` resolves to ``[0]`` for I2V (single frame) and ``[0, 1]``
    # for V2V. ``condition_video_keep`` controls whether the first or last
    # source frames are used when the input video is longer than needed.
    condition_frame_indexes: list[int] | None = None
    condition_video_keep: str = "first"

    # Transfer (control-video) conditioning. ``control_path`` points to one or
    # more pre-computed control videos (e.g. edge / blur / depth / seg / wsm
    # maps). When set, each control clip is VAE-encoded and packed as clean
    # vision tokens that prefix the target clip in the GEN sequence; multiple
    # paths drive multi-hint transfer (e.g. edge + depth). Control clips reuse
    # ``proj_in``, so every Cosmos3 checkpoint supports transfer.
    control_path: str | list[str] | None = None

    # Hint type(s) parallel to ``control_path`` (``edge`` / ``blur`` / ``depth``
    # / ``seg`` / ``wsm``); named in the control-emphasis sentence.
    control_hint: str | list[str] | None = None
    # Optional per-control weights parallel to ``control_path``, normalized to sum 1.
    control_weight: float | list[float] | None = None

    # The imaginaire4 per-hint request objects, an alternative to the lists
    # above: ``{"control_path": ..., "weight": ..., "preset_edge_threshold" /
    # "preset_blur_strength": ...}``, or ``{}`` / ``true``. An edge or blur hint
    # without ``control_path`` is computed from ``video_path``.
    edge: Any = None
    blur: Any = None
    depth: Any = None
    seg: Any = None
    wsm: Any = None

    # Append the control-emphasis sentence to the prompt (default on).
    emphasize_control_in_prompt: bool | None = None

    # Control-CFG scale for transfer. ``1.0`` (default) disables the extra
    # control-dropped forward; values > 1.0 amplify the control map's influence
    # by blending the with-control and without-control predictions on the
    # generated span: ``cond_nc + control_guidance * (cond_full - cond_nc)``.
    control_guidance: float = 1.0

    # Optional timestep window ``(lo, hi)`` restricting where control-CFG is
    # applied (analogous to ``guidance_interval`` for text CFG). ``None`` applies
    # it at every step.
    control_guidance_interval: tuple[float, float] | None = None

    # Long-video transfer controls. Chunks overlap by
    # ``num_conditional_frames`` pixel frames; overlap frames from the previous
    # decoded chunk are kept clean in the next chunk. The output follows the
    # control clip's length, capped by ``max_frames`` (and by ``num_frames``
    # when the request sets it).
    num_video_frames_per_chunk: int = 93
    num_conditional_frames: int = 1
    num_first_chunk_conditional_frames: int = 0
    max_frames: int = 5000
    show_control_condition: bool = False
    show_input: bool = False
    share_vision_temporal_positions: bool = True

    # Tuned per-hint defaults applied when exactly one typed control input is
    # given and the corresponding field was not set explicitly (mirrors the
    # cosmos-framework ``_TRANSFER_DEFAULTS`` table). ``shift`` maps to
    # ``flow_shift``. Multi-hint transfer keeps the request's own values.
    _TRANSFER_DEFAULTS: ClassVar[dict[str, dict[str, float | int]]] = {
        "edge": {"guidance": 3.0, "control_guidance": 1.5, "shift": 10.0},
        "blur": {"guidance": 3.0, "control_guidance": 1.5, "shift": 10.0},
        "depth": {"guidance": 3.0, "control_guidance": 1.5, "shift": 10.0},
        "seg": {"guidance": 3.0, "control_guidance": 2.0, "shift": 10.0},
        "wsm": {
            "guidance": 1.0,
            "control_guidance": 3.0,
            "shift": 10.0,
            "num_frames": 101,
            "fps": 10,
            "num_video_frames_per_chunk": 101,
        },
    }

    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (1280, 720),
            (720, 1280),
            (832, 480),
            (480, 832),
            (1024, 1024),
            (640, 640),
        ]
    )

    # Action modality (requires action_gen=True in the model checkpoint)
    # action_mode: "forward_dynamics" | "policy" | "inverse_dynamics"
    action_mode: str | None = None
    domain_id: int | None = None
    domain_name: str | None = None
    raw_action_dim: int | None = None
    action_fps: float | None = None
    # Action data for forward_dynamics: [T, D] nested list (API) or JSON string
    # (CLI via --action). Ignored by the other action modes.
    action: Any = None
    # Viewpoint phrasing for the structured action caption.
    action_view_point: str = "ego_view"
    # Optional dataset-derived action stats (JSON) for (de)normalization. When
    # set, input actions are normalized and predicted actions de-normalized
    # into physical units with ``action_normalization``.
    action_stats_path: str | None = None
    action_normalization: str = "quantile"

    def is_explicit(self, name: str) -> bool:
        return self._explicit_fields is not None and name in self._explicit_fields

    @classmethod
    def image_request_extra_fields(cls) -> frozenset[str]:
        return frozenset(
            {
                "guidance_interval",
                "use_duration_template",
                "use_guardrails",
                "use_resolution_template",
                "use_system_prompt",
            }
        )

    @classmethod
    def default_image_output_format(cls) -> str:
        return "png"

    @classmethod
    def default_image_response_format(cls) -> str:
        return "b64_json"

    @classmethod
    def video_request_extra_fields(cls) -> frozenset[str]:
        return cls.image_request_extra_fields() | frozenset(
            {
                "action",
                "action_fps",
                "action_mode",
                "action_normalization",
                "action_view_point",
                "condition_frame_indexes",
                "condition_frame_indexes_vision",
                "condition_video_keep",
                "blur",
                "control_guidance",
                "control_guidance_interval",
                "control_hint",
                "control_path",
                "control_weight",
                "depth",
                "domain_id",
                "domain_name",
                "edge",
                "emphasize_control_in_prompt",
                "generate_sound",
                "guardrails",
                "max_frames",
                "num_conditional_frames",
                "num_first_chunk_conditional_frames",
                "num_video_frames_per_chunk",
                "raw_action_dim",
                "seg",
                "share_vision_temporal_positions",
                "show_control_condition",
                "show_input",
                "sound_duration",
                "wsm",
            }
        )

    def _resolve_control_paths(self) -> list[str]:
        cp = self.control_path
        if cp is None:
            return []
        if isinstance(cp, str):
            return [cp] if cp else []
        return [p for p in cp if isinstance(p, str) and p]

    def _resolve_control_hints(self) -> list[str]:
        hint = self.control_hint
        if hint is None:
            return []
        hints = [hint] if isinstance(hint, str) else list(hint)
        hints = [h for h in hints if h]
        for h in hints:
            if h not in COSMOS3_TRANSFER_HINT_KEYS:
                raise ValueError(
                    f"Unknown control_hint {h!r}; expected one of "
                    f"{list(COSMOS3_TRANSFER_HINT_KEYS)}"
                )
        return hints

    def _resolve_control_weights(self) -> list[float]:
        weight = _parse_request_value(self.control_weight)
        if weight is None:
            return []
        weights = list(weight) if isinstance(weight, (list, tuple)) else [weight]
        return [_transfer_weight(w, "control_weight") for w in weights]

    def resolve_transfer_hints(self) -> list[Cosmos3TransferHint]:
        """The request's control inputs in ``COSMOS3_TRANSFER_HINT_KEYS`` order.

        Accepts either the per-hint objects (``edge`` ... ``wsm``) or the
        ``control_path`` / ``control_hint`` / ``control_weight`` lists. Untyped
        ``control_path`` entries keep their request order with ``key=None``.
        """
        objects = {
            key: getattr(self, key)
            for key in COSMOS3_TRANSFER_HINT_KEYS
            if getattr(self, key) is not None
        }
        paths = self._resolve_control_paths()
        hints = self._resolve_control_hints()
        weights = self._resolve_control_weights()
        if objects and (paths or hints or weights):
            raise ValueError(
                "Cosmos3 transfer takes either the per-hint objects "
                f"({', '.join(COSMOS3_TRANSFER_HINT_KEYS)}) or control_path/"
                "control_hint/control_weight, not both."
            )
        if objects:
            return [
                parse_transfer_hint_object(key, value) for key, value in objects.items()
            ]
        if hints and len(hints) != len(paths):
            raise ValueError(
                "control_hint must contain exactly one entry per control_path "
                f"(got {len(hints)} hint(s) for {len(paths)} path(s))"
            )
        if weights and len(weights) != len(paths):
            raise ValueError(
                "control_weight must contain exactly one entry per control_path "
                f"(got {len(weights)} weight(s) for {len(paths)} path(s))"
            )
        if len(set(hints)) != len(hints):
            raise ValueError(f"control_hint entries must be unique, got {hints}")
        resolved = [
            Cosmos3TransferHint(
                key=hints[i] if hints else None,
                control_path=path,
                weight=weights[i] if weights else 1.0,
            )
            for i, path in enumerate(paths)
        ]
        if hints:
            resolved.sort(key=lambda hint: COSMOS3_TRANSFER_HINT_KEYS.index(hint.key))
        return resolved

    def _apply_transfer_hint_defaults(self, hints: list[Cosmos3TransferHint]) -> bool:
        """Fill tuned per-hint defaults for a single, typed control input.

        Defaults apply only to fields the request did not pass explicitly; multi-hint
        and untyped transfer keep the request's own values. Returns whether the
        hint's default set ``num_frames``.
        """
        if len(hints) != 1 or hints[0].key is None:
            return False
        defaults = self._TRANSFER_DEFAULTS[hints[0].key]
        if not self.is_explicit("control_guidance"):
            self.control_guidance = defaults["control_guidance"]
        if not self.is_explicit("guidance_scale"):
            self.guidance_scale = defaults["guidance"]
        if not self.is_explicit("flow_shift") and self.flow_shift is None:
            self.flow_shift = defaults["shift"]
        set_num_frames = False
        for field_name in ("num_frames", "fps", "num_video_frames_per_chunk"):
            if field_name in defaults and not self.is_explicit(field_name):
                setattr(self, field_name, defaults[field_name])
                set_num_frames = set_num_frames or field_name == "num_frames"
        return set_num_frames

    def _apply_transfer_defaults(self, hints: list[Cosmos3TransferHint]) -> None:
        """Validate a transfer request and resolve its emphasis and length.

        Edge/blur controls without a file are computed from ``video_path``. The
        control-emphasis sentence names every hint, so it defaults on only when
        all controls are typed. The output follows the control clip, capped by
        ``max_frames`` and by ``num_frames`` when the request (or the hint's
        default) sets it.
        """
        for hint in hints:
            if hint.control_path is not None:
                continue
            if hint.key not in COSMOS3_TRANSFER_ON_THE_FLY_HINTS:
                raise ValueError(
                    f"Missing pre-computed control input for '{hint.key}'. "
                    "Provide a control_path for it."
                )
            if self.video_path is None:
                raise ValueError(
                    f"The '{hint.key}' control is computed from video_path; provide "
                    "video_path or a pre-computed control_path."
                )
        if sum(hint.weight for hint in hints) <= 0:
            raise ValueError(
                "Cosmos3 transfer control weights must have a positive sum."
            )
        typed = all(hint.key is not None for hint in hints)
        if self.emphasize_control_in_prompt is None:
            self.emphasize_control_in_prompt = typed
        elif self.emphasize_control_in_prompt and not typed:
            raise ValueError(
                "emphasize_control_in_prompt names the control types; set control_hint "
                f"({', '.join(COSMOS3_TRANSFER_HINT_KEYS)}) for every control_path."
            )
        cap_num_frames = self._apply_transfer_hint_defaults(hints)
        if cap_num_frames or self.is_explicit("num_frames"):
            self.max_frames = min(int(self.max_frames), int(self.num_frames))

    @classmethod
    def lower_video_request_kwargs(
        cls, request: Any, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        kwargs = super().lower_video_request_kwargs(request, dict(kwargs))
        extras = getattr(request, "model_extra", None) or {}

        if "use_guardrails" not in kwargs and extras.get("guardrails") is not None:
            kwargs["use_guardrails"] = _parse_request_value(extras["guardrails"])

        condition_indexes = kwargs.get("condition_frame_indexes")
        if condition_indexes is None:
            condition_indexes = extras.get("condition_frame_indexes_vision")
        condition_indexes = _optional_int_list(condition_indexes)
        if condition_indexes is not None:
            kwargs["condition_frame_indexes"] = condition_indexes

        if "sound_duration" in kwargs:
            kwargs["sound_duration"] = float(
                _parse_request_value(kwargs["sound_duration"])
            )
        generate_sound = _parse_request_value(extras.get("generate_sound"))
        if generate_sound is False:
            kwargs["sound_duration"] = 0.0
        elif generate_sound is True and "sound_duration" not in kwargs:
            kwargs["sound_duration"] = float(kwargs["num_frames"]) / float(
                kwargs["fps"]
            )

        for name in ("control_path", "control_hint"):
            value = _parse_request_value(kwargs.get(name))
            if isinstance(value, (list, tuple)):
                value = [str(item) for item in value if str(item).strip()]
            elif value is not None and not isinstance(value, str):
                value = str(value)
            if isinstance(value, str):
                value = value if value.strip() else None
            if value:
                kwargs[name] = value
            else:
                kwargs.pop(name, None)

        for name in (*COSMOS3_TRANSFER_HINT_KEYS, "control_weight"):
            if name in kwargs:
                value = _parse_request_value(kwargs[name])
                if value is None or (isinstance(value, str) and not value.strip()):
                    kwargs.pop(name)
                else:
                    kwargs[name] = value

        if "control_guidance" in kwargs:
            kwargs["control_guidance"] = float(
                _parse_request_value(kwargs["control_guidance"])
            )
        if "control_guidance_interval" in kwargs:
            interval = _parse_request_value(kwargs["control_guidance_interval"])
            if interval is None or (isinstance(interval, str) and not interval.strip()):
                kwargs.pop("control_guidance_interval")
            else:
                if not isinstance(interval, (list, tuple)):
                    interval = [interval]
                kwargs["control_guidance_interval"] = tuple(
                    float(item) for item in interval
                )

        for name in (
            "num_video_frames_per_chunk",
            "num_conditional_frames",
            "num_first_chunk_conditional_frames",
            "max_frames",
        ):
            value = _parse_request_value(kwargs.get(name))
            if value is not None and value != "":
                kwargs[name] = int(value)

        for name in (
            "show_control_condition",
            "show_input",
            "share_vision_temporal_positions",
            "emphasize_control_in_prompt",
        ):
            value = _parse_request_value(kwargs.get(name))
            if value is None or (isinstance(value, str) and not value.strip()):
                kwargs.pop(name, None)
            elif isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"1", "true", "yes", "on"}:
                    kwargs[name] = True
                elif normalized in {"0", "false", "no", "off"}:
                    kwargs[name] = False
                else:
                    raise ValueError(f"Invalid boolean value: {value!r}")
            else:
                kwargs[name] = bool(value)

        for name in (
            "condition_video_keep",
            "action_mode",
            "domain_id",
            "domain_name",
            "raw_action_dim",
            "action_fps",
            "action",
            "action_view_point",
            "action_normalization",
        ):
            value = _parse_request_value(kwargs.get(name))
            if isinstance(value, str) and not value.strip():
                kwargs.pop(name, None)
            elif value is not None:
                kwargs[name] = value

        if not cls._single_view_transfer:
            return kwargs
        hint = kwargs.get("control_hint")
        paths = kwargs.get("control_path")
        hints = [hint] if isinstance(hint, str) else list(hint or [])
        hints += [name for name in COSMOS3_TRANSFER_HINT_KEYS if name in kwargs]
        control_paths = [paths] if isinstance(paths, str) else list(paths or [])
        is_transfer = bool(control_paths or hints)
        if hints == ["wsm"] and len(control_paths) <= 1:
            defaults = cls._TRANSFER_DEFAULTS["wsm"]
            if request.num_frames is None:
                kwargs["num_frames"] = defaults["num_frames"]
            if request.fps is None:
                kwargs["fps"] = defaults["fps"]
        elif (
            is_transfer
            and getattr(request, "num_frames", None) is None
            and getattr(request, "seconds", None) is None
        ):
            # The video API fills num_frames from its default duration; transfer
            # instead follows the control clip unless the request sets a length.
            kwargs.pop("num_frames", None)
        return kwargs

    def _adjust(self, server_args) -> None:
        if (
            self.task_type is None
            and self.num_frames == 1
            and self.image_path is None
            and self.video_path is None
            and self.action_mode is None
        ):
            self.task_type = ModelTaskType.T2I

        # adjust distil and edge args — read from the pre-computed config fields
        # so no checkpoint download happens at request time.
        pipeline_config = server_args.pipeline_config
        if self.action_stats_path is None:
            self.action_stats_path = getattr(pipeline_config, "action_stats_path", None)
        distilled_sigmas = pipeline_config.distilled_sigmas
        if distilled_sigmas is not None:
            self.num_inference_steps = len(distilled_sigmas)
        self._resolve_variant_defaults(
            bool(pipeline_config.is_edge),
            is_distilled=distilled_sigmas is not None,
        )

        # adjust action args
        action_output = False
        if self.action_mode is not None:
            self.action_mode = str(self.action_mode).strip().lower()
            if self.action_mode not in (
                "policy",
                "forward_dynamics",
                "inverse_dynamics",
            ):
                raise ValueError(
                    f"Unsupported action_mode={self.action_mode!r}; expected "
                    "'policy', 'forward_dynamics', or 'inverse_dynamics'."
                )
            action_output = self.action_mode != "forward_dynamics"
            if action_output and self.num_outputs_per_prompt > 1:
                raise ValueError(
                    "Cosmos3 policy and inverse-dynamics requests return one "
                    "prediction per input; num_outputs_per_prompt must be 1"
                )

        transfer_hints = self.resolve_transfer_hints()
        if transfer_hints:
            if pipeline_config.distilled_sigmas is not None:
                raise ValueError(
                    "Cosmos3 distilled checkpoints do not support transfer inference"
                )
            if pipeline_config.is_edge:
                raise ValueError(
                    "Cosmos3 Edge checkpoints do not support transfer inference"
                )
            if self.num_frames == 1:
                raise ValueError(
                    "Cosmos3 transfer inference is supported only for video outputs"
                )
            if self.image_path is not None:
                raise ValueError(
                    "Cosmos3 transfer accepts control videos and an optional source "
                    "video, not an image input"
                )
            if self.action_mode is not None:
                raise ValueError(
                    "Cosmos3 transfer cannot be combined with action generation"
                )
            if float(self.sound_duration or 0.0) > 0.0:
                raise ValueError(
                    "Cosmos3 transfer cannot be combined with sound generation"
                )
            self._apply_transfer_defaults(transfer_hints)
        super()._adjust(server_args)

        # Policy and inverse dynamics produce actions. Forward dynamics consumes
        # actions to produce video and therefore remains a visual request.
        if action_output:
            self.data_type = DataType.ACTION
            self.save_output = False
            self.return_file_paths_only = False
            self.return_frames = False
            self.output_file_name = None
            self.output_compression = 0

    def _adjust_visual_fields(self, server_args, pipeline_config) -> None:
        super()._adjust_visual_fields(server_args, pipeline_config)
        self._align_resolution(pipeline_config)

    def prepare_synthetic_warmup_request_for_queue(self, req, server_args) -> None:
        # Warmup overwrites the dimensions after _adjust has run.
        self._align_resolution(server_args.pipeline_config)

    def _align_resolution(self, pipeline_config) -> None:
        from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import (
            Cosmos3Config,
        )

        # The simulation pipelines (Dreams, multiview, sim-auto) subclass
        # Cosmos3Config and validate their own canvas.
        if type(pipeline_config) is not Cosmos3Config:
            return
        if self.width is None or self.height is None:
            return
        if self.width <= 0 or self.height <= 0:
            return
        m = COSMOS3_SIZE_MULTIPLE
        requested_size = (self.width, self.height)
        self.width = max(m, self.width // m * m)
        self.height = max(m, self.height // m * m)
        if (self.width, self.height) != requested_size:
            logger.warning(
                "Cosmos3 requires dimensions divisible by %s; adjusted requested "
                "resolution from %sx%s to %sx%s",
                m,
                *requested_size,
                self.width,
                self.height,
            )

    def _validate(self) -> None:
        super()._validate()
        self.resolve_transfer_hints()
        if self.emphasize_control_in_prompt is not None and not isinstance(
            self.emphasize_control_in_prompt, bool
        ):
            raise ValueError("emphasize_control_in_prompt must be a boolean or null.")
        if self.control_guidance_interval is not None:
            if len(self.control_guidance_interval) != 2:
                raise ValueError(
                    "control_guidance_interval must contain exactly two values"
                )
            lo, hi = self.control_guidance_interval
            if float(lo) > float(hi):
                raise ValueError(
                    "control_guidance_interval must be ordered as (low, high)"
                )
        if self.num_video_frames_per_chunk <= 0:
            raise ValueError("num_video_frames_per_chunk must be positive")
        if self.num_conditional_frames < 0:
            raise ValueError("num_conditional_frames must be non-negative")
        if self.num_conditional_frames >= self.num_video_frames_per_chunk:
            raise ValueError(
                "num_conditional_frames must be smaller than num_video_frames_per_chunk"
            )
        if self.num_first_chunk_conditional_frames < 0:
            raise ValueError("num_first_chunk_conditional_frames must be non-negative")
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive")

    def _guidance_is_explicit(self) -> bool:
        explicit = getattr(self, "_explicit_fields", None)
        return explicit is not None and "guidance_scale" in explicit

    def _resolve_variant_defaults(
        self, is_edge: bool, is_distilled: bool = False
    ) -> None:
        """Fill unset resolution/guidance with the variant's defaults.

        Base resolution defaulting (``supported_resolutions[0]``) covers the
        non-Edge path; only Edge and guidance need explicit handling here.
        """
        is_t2i = self.num_frames == 1
        if is_distilled:
            # Guidance is distilled into the model; run a single forward.
            self.guidance_scale = 1.0
        elif is_edge and not self._guidance_is_explicit():
            self.guidance_scale = (
                COSMOS3_EDGE_T2I_GUIDANCE_SCALE
                if is_t2i
                else COSMOS3_EDGE_T2V_GUIDANCE_SCALE
            )
        if is_t2i and not is_distilled and self.guidance_interval is None:
            self.guidance_interval = COSMOS3_T2I_GUIDANCE_INTERVAL
        if is_edge:
            self.supported_resolutions = COSMOS3_EDGE_SUPPORTED_RESOLUTIONS
            if self.height is None and self.width is None:
                if is_t2i:
                    self.width = self.height = COSMOS3_EDGE_T2I_SIZE
                else:
                    self.width, self.height = (
                        COSMOS3_EDGE_T2V_WIDTH,
                        COSMOS3_EDGE_T2V_HEIGHT,
                    )

    def _set_output_file_name(self) -> None:
        # Action outputs never need a visual filename. This also avoids hashing
        # in-memory observation images while base visual adjustment is running.
        if self.action_mode in ("policy", "inverse_dynamics"):
            return
        # The pipeline config's ``task_type=TI2V`` drives ``data_type`` to
        # VIDEO, but a single-frame request is a T2I and must pick the IMAGE
        # extension. Flip before the base derives the file name.
        if self.num_frames == 1:
            self.data_type = DataType.IMAGE
        super()._set_output_file_name()
