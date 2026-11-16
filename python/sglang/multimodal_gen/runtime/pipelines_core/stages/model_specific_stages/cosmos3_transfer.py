# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 single-view transfer: control media, on-the-fly controls, prompt pieces.

Mirrors the bidirectional Transfer route of imaginaire4
(``cosmos_framework/inference/preprocessing/{control_media,control_hints,control_requests,
control_chunks}.py``): the input and control clips are decoded with PyAV, resized
with torchvision's antialiased bilinear resize on uint8 pixels and center-cropped;
edge and blur controls are computed from the resized input when a hint has no
``control_path``; every transfer caption carries the Transfer system prompt and,
by default, the control-emphasis sentence.
"""

import math
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as TF

# imaginaire4 ``data/reasoner/processors/qwen3_vl.py:_SYSTEM_PROMPT_TRANSFER``, verbatim.
COSMOS3_TRANSFER_SYSTEM_PROMPT = (
    "You are a helpful assistant that generates images or videos following the user's instructions"
    " and control signals (edge maps, blur, depth, or segmentation)."
)
# imaginaire4 ``control_requests.append_control_emphasis``, verbatim.
COSMOS3_CONTROL_EMPHASIS_TEMPLATE = (
    " Follow the {hints} control video precisely: shape, contour, silhouette,"
    " position, and motion of every visible structure must align with the {hints}"
    " signal at every frame."
)

# Canny thresholds per ``preset_edge_threshold`` (imaginaire4 ``AddControlInputEdge``).
COSMOS3_EDGE_THRESHOLDS: dict[str, tuple[int, int]] = {
    "very_low": (20, 50),
    "low": (50, 100),
    "medium": (100, 200),
    "high": (200, 300),
    "very_high": (300, 400),
}
# Per ``preset_blur_strength`` (imaginaire4 ``AddControlInputBlur``, min/max factors 4/16):
# the downscale before the bilateral filter and the final cubic down/up factor.
COSMOS3_BLUR_PRE_DOWNSCALE: dict[str, int] = {
    "none": 1,
    "very_low": 1,
    "low": 4,
    "medium": 2,
    "high": 1,
    "very_high": 4,
}
COSMOS3_BLUR_DOWNUP: dict[str, int] = {
    "none": 1,
    "very_low": 4,
    "low": 4,
    "medium": 10,
    "high": 16,
    "very_high": 16,
}
# Inference bilateral filter (imaginaire4 ``BilateralFilterConfig`` defaults), tuned
# for a 720-pixel longest side and scaled linearly with the frame size.
_BILATERAL_D = 30
_BILATERAL_SIGMA_COLOR = 150
_BILATERAL_SIGMA_SPACE = 100
_BILATERAL_REFERENCE_SIDE = 720

_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
# Frames resized together; the resize is per frame, so this only bounds the
# source-resolution frames held at once.
_RESIZE_BATCH_FRAMES = 16


def append_control_emphasis(prompt: str, hint_keys: Sequence[str]) -> str:
    """The positive caption followed by the control-emphasis sentence."""
    hints = ", ".join(hint_keys)
    return prompt.rstrip() + COSMOS3_CONTROL_EMPHASIS_TEMPLATE.format(hints=hints)


def normalize_control_weights(weights: Sequence[float]) -> list[float]:
    """Non-negative weights with a positive sum, each divided by the sum."""
    values = [float(weight) for weight in weights]
    if any(not math.isfinite(w) or w < 0 for w in values):
        raise ValueError(
            f"control weights must be finite and non-negative, got {values}"
        )
    total = sum(values)
    if total <= 0:
        raise ValueError(f"control weights must have a positive sum, got {values}")
    return [weight / total for weight in values]


def resize_center_crop_uint8(
    frames_tchw: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """Aspect-preserving resize then center crop of uint8 ``[T, 3, H, W]`` frames.

    Same calls as imaginaire4 ``images._resize_and_center_crop`` on uint8 pixels:
    ``TF.resize`` (bilinear, antialiased) to ``ceil(scale * size)`` and
    ``TF.center_crop`` (offsets rounded half to even).
    """
    orig_h, orig_w = int(frames_tchw.shape[2]), int(frames_tchw.shape[3])
    scale = max(width / orig_w, height / orig_h)
    resize_h = int(math.ceil(scale * orig_h))
    resize_w = int(math.ceil(scale * orig_w))
    resized = TF.resize(frames_tchw, [resize_h, resize_w], antialias=True)
    return TF.center_crop(resized, [height, width])


def _iter_rgb_frames(path: str, max_frames: int) -> tuple[Iterator[np.ndarray], float]:
    """RGB ``[H, W, 3]`` frames of a video in presentation order, and its frame rate.

    Decodes like ``torchvision.io.read_video``: PyAV ``to_rgb`` frames, frames
    with a negative timestamp skipped.
    """
    import av

    container = av.open(path, metadata_errors="ignore")
    if not container.streams.video:
        container.close()
        raise ValueError(f"No video stream in {path!r}")
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 0.0

    def frames() -> Iterator[np.ndarray]:
        try:
            count = 0
            for frame in container.decode(stream):
                if frame.pts is not None and frame.pts < 0:
                    continue
                yield frame.to_rgb().to_ndarray()
                count += 1
                if count >= max_frames:
                    break
        finally:
            container.close()

    return frames(), fps


def read_transfer_media_uint8(
    path: str, *, height: int, width: int, max_frames: int
) -> tuple[torch.Tensor, float]:
    """Decode an image or video and fit it to ``height x width``.

    Returns uint8 ``[3, T, height, width]`` (at most ``max_frames`` frames) and
    the clip's frame rate (1.0 for an image), as imaginaire4
    ``control_media.read_and_resize_media`` does.
    """
    if max_frames <= 0:
        raise ValueError(f"max_frames must be positive, got {max_frames}")
    if Path(path).suffix.lower() in _IMAGE_EXTENSIONS:
        with PIL.Image.open(path) as image:
            pixels = np.array(image.convert("RGB"))
        frames = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
        resized = resize_center_crop_uint8(frames, height, width)
        return resized.permute(1, 0, 2, 3).contiguous(), 1.0

    frame_iter, fps = _iter_rgb_frames(path, max_frames)
    batches: list[torch.Tensor] = []
    pending: list[np.ndarray] = []
    for array in frame_iter:
        pending.append(array)
        if len(pending) == _RESIZE_BATCH_FRAMES:
            pixels = torch.from_numpy(np.stack(pending)).permute(0, 3, 1, 2)
            batches.append(resize_center_crop_uint8(pixels, height, width))
            pending.clear()
    if pending:
        pixels = torch.from_numpy(np.stack(pending)).permute(0, 3, 1, 2)
        batches.append(resize_center_crop_uint8(pixels, height, width))
    if not batches:
        raise ValueError(f"No frames decoded from transfer media {path!r}")
    return torch.cat(batches, dim=0).permute(1, 0, 2, 3).contiguous(), fps


def edge_control_uint8(frames_cthw: torch.Tensor, preset: str) -> torch.Tensor:
    """Canny edge control of uint8 ``[3, T, H, W]`` frames, as uint8 ``[3, T, H, W]``."""
    import cv2

    if preset not in COSMOS3_EDGE_THRESHOLDS:
        raise ValueError(
            f"Unknown edge preset {preset!r}; expected one of {list(COSMOS3_EDGE_THRESHOLDS)}"
        )
    lower, upper = COSMOS3_EDGE_THRESHOLDS[preset]
    frames = frames_cthw.cpu().numpy().astype(np.uint8)
    edges = np.stack(
        [cv2.Canny(image, lower, upper) for image in frames.transpose((1, 2, 3, 0))]
    )
    return torch.from_numpy(edges)[None].expand(3, -1, -1, -1).contiguous()


def _resize_each(
    frames: np.ndarray, size: tuple[int, int], interpolation: int
) -> np.ndarray:
    """``cv2.resize`` every ``[C, T, H, W]`` frame to ``size = (width, height)``."""
    import cv2

    channels, num_frames = frames.shape[:2]
    out = np.empty((channels, num_frames, size[1], size[0]), dtype=frames.dtype)
    for t in range(num_frames):
        frame = np.ascontiguousarray(frames[:, t].transpose(1, 2, 0))
        resized = cv2.resize(frame, size, interpolation=interpolation)
        if resized.ndim == 2:
            resized = resized[..., None]
        out[:, t] = resized.transpose(2, 0, 1)
    return out


def _bilateral_each(frames: np.ndarray) -> np.ndarray:
    import cv2

    longest = int(max(frames.shape[2], frames.shape[3]))
    scale = longest / _BILATERAL_REFERENCE_SIDE if longest > 0 else 1.0
    d = max(1, int(round(_BILATERAL_D * scale)))
    d = d + 1 if d % 2 == 0 else d
    sigma_color = max(1.0, _BILATERAL_SIGMA_COLOR * scale)
    sigma_space = max(1.0, _BILATERAL_SIGMA_SPACE * scale)
    out = np.empty_like(frames)
    for t in range(frames.shape[1]):
        frame = np.ascontiguousarray(frames[:, t].transpose(1, 2, 0))
        frame = cv2.bilateralFilter(frame, d, sigma_color, sigma_space)
        if frame.ndim == 2:
            frame = frame[..., None]
        out[:, t] = frame.transpose(2, 0, 1)
    return out


def blur_control_uint8(frames_cthw: torch.Tensor, preset: str) -> torch.Tensor:
    """Bilateral-blur control of uint8 ``[3, T, H, W]`` frames, as uint8 ``[3, T, H, W]``."""
    import cv2

    if preset not in COSMOS3_BLUR_DOWNUP:
        raise ValueError(
            f"Unknown blur preset {preset!r}; expected one of {list(COSMOS3_BLUR_DOWNUP)}"
        )
    frames = frames_cthw.cpu().numpy()
    if preset == "none":
        return torch.from_numpy(frames.copy())
    height, width = frames.shape[2], frames.shape[3]
    factor = COSMOS3_BLUR_PRE_DOWNSCALE[preset]
    frames = _resize_each(frames, (width // factor, height // factor), cv2.INTER_AREA)
    frames = _bilateral_each(frames)
    frames = _resize_each(frames, (width, height), cv2.INTER_LINEAR)
    downup = COSMOS3_BLUR_DOWNUP[preset]
    small = _resize_each(
        frames, (int(width / downup), int(height / downup)), cv2.INTER_CUBIC
    )
    return torch.from_numpy(_resize_each(small, (width, height), cv2.INTER_CUBIC))


def on_the_fly_control_uint8(
    frames_cthw: torch.Tensor, *, hint: str, preset: str
) -> torch.Tensor:
    """The edge or blur control computed from the resized input clip."""
    if hint == "edge":
        return edge_control_uint8(frames_cthw, preset)
    if hint == "blur":
        return blur_control_uint8(frames_cthw, preset)
    raise ValueError(
        f"Missing pre-computed control input for '{hint}'. Provide a control_path for it."
    )
