# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 sampling parameters.

A single ``SamplingParams`` class serves T2V, I2V, V2V, T2I, and
action-conditioned variants.  Per-request mode is dispatched in the pipeline
from ``num_frames`` (``== 1`` → T2I), ``image_path`` (set → I2V),
``video_path`` (set → V2V), and ``action_mode`` (set → action-conditioned).
For ``num_frames == 1`` the output ``data_type`` flips to ``IMAGE``
so the file extension and decode path agree.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)


@dataclass
class Cosmos3SamplingParams(SamplingParams):
    """Cosmos3 sampling parameters (T2V defaults; also used for I2V / V2V / T2I)."""

    height: int = 720
    width: int = 1280
    num_frames: int = 81
    fps: int = 24

    guidance_scale: float = 4.0
    num_inference_steps: int = 35

    negative_prompt: str = ""

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

    # Optional hint type(s) parallel to ``control_path`` (one of
    # ``edge`` / ``blur`` / ``depth`` / ``seg`` / ``wsm``). Used only to apply
    # tuned per-hint defaults (``guidance`` / ``control_guidance`` / ``shift``)
    # when exactly one control input is given and the user left those unset.
    control_hint: str | list[str] | None = None

    # Control-CFG scale for transfer. ``1.0`` (default) disables the extra
    # control-dropped forward; values > 1.0 amplify the control map's influence
    # by blending the with-control and without-control predictions on the
    # generated span: ``cond_nc + control_guidance * (cond_full - cond_nc)``.
    control_guidance: float = 1.0

    # Optional timestep window ``(lo, hi)`` restricting where control-CFG is
    # applied (analogous to ``guidance_interval`` for text CFG). ``None`` applies
    # it at every step.
    control_guidance_interval: tuple[float, float] | None = None

    # Tuned per-hint defaults applied when exactly one control input is given
    # and the corresponding field was not set explicitly (mirrors the
    # cosmos-framework ``_TRANSFER_DEFAULTS`` table). ``shift`` maps to
    # ``flow_shift``. Multi-hint transfer keeps the request's own values.
    _TRANSFER_DEFAULTS: ClassVar[dict[str, dict[str, float]]] = {
        "edge": {"guidance": 3.0, "control_guidance": 1.5, "shift": 10.0},
        "blur": {"guidance": 3.0, "control_guidance": 1.5, "shift": 10.0},
        "depth": {"guidance": 3.0, "control_guidance": 1.5, "shift": 10.0},
        "seg": {"guidance": 3.0, "control_guidance": 2.0, "shift": 10.0},
        "wsm": {"guidance": 3.0, "control_guidance": 3.0, "shift": 10.0},
    }

    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (1280, 720),
            (720, 1280),
            (832, 480),
            (480, 832),
            (1024, 1024),
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
            if h not in self._TRANSFER_DEFAULTS:
                raise ValueError(
                    f"Unknown control_hint {h!r}; expected one of "
                    f"{sorted(self._TRANSFER_DEFAULTS)}"
                )
        return hints

    def _apply_transfer_hint_defaults(self) -> None:
        """Fill tuned per-hint defaults for a single, typed control input.

        Mirrors cosmos-framework: defaults apply only when there is exactly one
        control input with a known hint type, and only to fields the user did
        not pass explicitly (tracked via ``_explicit_fields``). Multi-hint
        transfer keeps the request's own ``guidance`` / ``control_guidance`` /
        ``flow_shift``.
        """
        if len(self._resolve_control_paths()) != 1:
            return
        hints = self._resolve_control_hints()
        if len(hints) != 1:
            return
        defaults = self._TRANSFER_DEFAULTS.get(hints[0])
        if defaults is None:
            return
        explicit = getattr(self, "_explicit_fields", None) or set()
        if "control_guidance" not in explicit:
            self.control_guidance = defaults["control_guidance"]
        if "guidance_scale" not in explicit:
            self.guidance_scale = defaults["guidance"]
        if "flow_shift" not in explicit and self.flow_shift is None:
            self.flow_shift = defaults["shift"]

    def _adjust(self, server_args):
        # Apply transfer per-hint defaults before the base resolves remaining
        # fields (e.g. flow_shift per mode), so an unset flow_shift can pick up
        # the hint's tuned shift.
        self._apply_transfer_hint_defaults()
        super()._adjust(server_args)

    def _set_output_file_name(self) -> None:
        # The pipeline config's ``task_type=TI2V`` drives ``data_type`` to
        # VIDEO, but a single-frame request is a T2I and must pick the IMAGE
        # extension. Flip before the base derives the file name.
        if self.num_frames == 1:
            self.data_type = DataType.IMAGE
        super()._set_output_file_name()
