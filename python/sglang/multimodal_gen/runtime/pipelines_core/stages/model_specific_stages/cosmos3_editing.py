# SPDX-License-Identifier: Apache-2.0
"""Cosmos3.5 instruction editing: source loading and the per-request edit plan.

An edit request packs the source as one fully clean vision item ahead of the
generated target, as cosmos-framework inference does: the source latent rides
the Cosmos3 DiT's clean ``control_latents`` prefix. Image edits place source
and target on consecutive mRoPE time steps; video edits give both items one
shared temporal grid. Sources are loaded and sized exactly as
cosmos-framework ``inference/preprocessing/editing.py`` does.
"""

import json
import math
import os
from io import BytesIO

import msgspec
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from sglang.multimodal_gen.configs.sample.cosmos3_editing import (
    COSMOS3_EDIT_DEFAULTS,
    COSMOS3_IMAGE_EDIT_SYSTEM_PROMPT,
    COSMOS3_VIDEO_EDIT_SYSTEM_PROMPT,
    EDIT_MODE_IMAGE,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.srt.multimodal.media_io import get_image_bytes

COSMOS3_EDIT_PLAN_KEY = "cosmos3_edit"
COSMOS3_EDIT_SOURCE_KEY = "cosmos3_edit_source"
COSMOS3_EDIT_FIRST_FRAME_KEY = "cosmos3_edit_first_frame"

# Bounds the float32 resize buffer of long or large video sources; resizing is
# per frame, so the chunk size does not change the result.
_RESIZE_CHUNK_FRAMES = 16


class Cosmos3EditPlan(msgspec.Struct, frozen=True):
    mode: str
    share_temporal_positions: bool
    # ``None`` falls back to the pipeline's generic system-prompt policy.
    system_prompt: str | None
    has_edited_first_frame: bool = False


def max_pixels_resized_size(
    width: int, height: int, *, max_pixels: int, padding_constant: int
) -> tuple[int, int]:
    """Aspect-preserving ``(width, height)`` capped at ``max_pixels``, rounded down to the multiple.

    Port of cosmos-framework ``image_resize.get_max_pixels_resized_size``; never upscales.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {width}x{height}")
    if padding_constant <= 0:
        raise ValueError(f"padding_constant must be positive, got {padding_constant}")
    if max_pixels < padding_constant * padding_constant:
        raise ValueError(
            f"max_pixels={max_pixels} is too small for padding_constant={padding_constant}"
        )
    scale = min(1.0, math.sqrt(max_pixels / (width * height)))
    resized_width = int(width * scale) // padding_constant * padding_constant
    resized_height = int(height * scale) // padding_constant * padding_constant
    resized_width = max(resized_width, padding_constant)
    resized_height = max(resized_height, padding_constant)
    while resized_width * resized_height > max_pixels:
        if resized_width >= resized_height and resized_width > padding_constant:
            resized_width -= padding_constant
        elif resized_height > padding_constant:
            resized_height -= padding_constant
        else:
            break
    return resized_width, resized_height


def valid_video_frame_count(available_frames: int, max_num_frames: int) -> int:
    """The largest positive ``4N+1`` frame count within both limits, else 0."""
    capped = min(available_frames, max_num_frames)
    if capped < 1:
        return 0
    return 1 + 4 * ((capped - 1) // 4)


def aligned_frame_indices(
    *, num_frames: int, average_fps: float, target_fps: float, count: int
) -> list[int]:
    """Nearest source frame of each output frame on a shared ``target_fps`` timeline.

    Python ``round`` (half to even) on ``i * average_fps / target_fps``, as cosmos-framework does.
    """
    if target_fps <= 0 or average_fps <= 0:
        raise ValueError("FPS values must be positive")
    indices = [round(index * average_fps / target_fps) for index in range(count)]
    return [min(max(index, 0), num_frames - 1) for index in indices]


def probe_frame_clock(video_path: str) -> tuple[int, float]:
    """``(num_frames, average_fps)`` of the first video stream from a packet scan.

    Same values as TorchCodec's exact-seek metadata, which cosmos-framework reads:
    ``num_frames`` counts packets, and ``average_fps`` divides it by the span from
    the first pts to the latest ``pts + duration``. A clip of 70 frames at 30 fps
    therefore reports 29.999999999999996, and that value decides rounding ties in
    :func:`aligned_frame_indices`.
    """
    import av

    with av.open(video_path) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {video_path!r}")
        stream = container.streams.video[0]
        time_base = stream.time_base
        begin = end = None
        count = 0
        for packet in container.demux(stream):
            if packet.size == 0 or packet.is_discard:
                continue
            pts = packet.pts if packet.pts is not None else packet.dts
            if pts is None:
                continue
            count += 1
            begin = pts if begin is None else min(begin, pts)
            end = (
                pts + packet.duration
                if end is None
                else max(end, pts + packet.duration)
            )
        header_fps = float(stream.average_rate) if stream.average_rate else None
    if count == 0:
        return 0, header_fps or 0.0
    # TorchCodec ptsToSeconds: double(pts) * num / den, multiplication first.
    duration = (
        float(end) * time_base.numerator / time_base.denominator
        - float(begin) * time_base.numerator / time_base.denominator
    )
    if duration == 0.0:
        return count, header_fps or 0.0
    return count, count / duration


def decode_video_frames(video_path: str, indices: list[int]) -> torch.Tensor:
    """Decode the frames at ``indices`` (presentation order) as uint8 ``[T, 3, H, W]``."""
    import av

    wanted = set(indices)
    last = max(indices)
    frames: dict[int, torch.Tensor] = {}
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_type = "AUTO"
        for index, frame in enumerate(container.decode(stream)):
            if index in wanted:
                frames[index] = torch.from_numpy(frame.to_ndarray(format="rgb24"))
            if index >= last:
                break
    missing = sorted(wanted - frames.keys())
    if missing:
        raise ValueError(
            f"Decoded fewer frames than the container reports from {video_path!r}; "
            f"missing frame indices {missing[:8]}"
        )
    return torch.stack([frames[index] for index in indices]).permute(0, 3, 1, 2)


def load_video_edit_source(
    video_path: str,
    *,
    fps: float,
    max_num_frames: int,
    max_pixels: int,
    padding_constant: int,
) -> torch.Tensor:
    """The source of a video edit as bf16 ``[3, T, H, W]`` in ``[-1, 1]``.

    cosmos-framework ``load_video_editing_source``: frames aligned to ``fps``, a
    ``4N+1`` count capped by ``max_num_frames``, a max-pixels size, a bilinear
    resize without antialiasing.
    """
    num_frames, average_fps = probe_frame_clock(video_path)
    if num_frames < 1:
        raise ValueError(f"Decoded zero frames from video editing source: {video_path}")
    if average_fps <= 0:
        raise ValueError(
            f"Video editing source FPS must be positive, got {average_fps}: {video_path}"
        )
    available = max(1, math.floor(num_frames / average_fps * fps))
    count = valid_video_frame_count(available, max_num_frames)
    if count < 1:
        raise ValueError(
            f"Video editing source has no usable frames: available={available}, "
            f"max_num_frames={max_num_frames}"
        )
    indices = aligned_frame_indices(
        num_frames=num_frames, average_fps=average_fps, target_fps=fps, count=count
    )
    frames = decode_video_frames(video_path, indices)  # [T, 3, H, W] uint8
    width, height = max_pixels_resized_size(
        int(frames.shape[3]),
        int(frames.shape[2]),
        max_pixels=max_pixels,
        padding_constant=padding_constant,
    )
    resized = torch.cat(
        [
            F.interpolate(
                chunk.float(),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            for chunk in frames.split(_RESIZE_CHUNK_FRAMES)
        ]
    )  # [T, 3, H', W'] float32
    return (resized.permute(1, 0, 2, 3).contiguous() / 127.5 - 1.0).to(torch.bfloat16)


def open_rgb_image(source: str | bytes | PIL.Image.Image) -> PIL.Image.Image:
    """Open an image the way cosmos-framework does: RGB, without applying EXIF orientation."""
    if isinstance(source, PIL.Image.Image):
        image = source
    elif isinstance(source, str) and os.path.isfile(source):
        image = PIL.Image.open(source)
    else:
        image = PIL.Image.open(BytesIO(get_image_bytes(source)))
    return image.convert("RGB")


def load_image_edit_source(
    image: PIL.Image.Image, *, max_pixels: int, padding_constant: int
) -> torch.Tensor:
    """The source of an image edit at its native size under the budget, float32 ``[3, H, W]`` in ``[-1, 1]``.

    cosmos-framework ``load_image_edit_source`` without a bucket: a direct
    LANCZOS resize to the max-pixels size, no crop.
    """
    size = max_pixels_resized_size(
        image.width,
        image.height,
        max_pixels=max_pixels,
        padding_constant=padding_constant,
    )
    if size != image.size:
        image = image.resize(size, PIL.Image.LANCZOS)
    pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
    return pixels / 127.5 - 1.0


def load_conditioning_frame(
    image: PIL.Image.Image, height: int, width: int
) -> torch.Tensor:
    """An image resized to fill ``height x width`` and center-cropped, float32 ``[3, H, W]`` in ``[-1, 1]``.

    cosmos-framework ``load_conditioning_image``: antialiased bilinear resize on
    float pixels, center crop, rounding to uint8 levels.
    """
    pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()[None]
    orig_h, orig_w = pixels.shape[2], pixels.shape[3]
    scale = max(width / orig_w, height / orig_h)
    resize_h, resize_w = math.ceil(scale * orig_h), math.ceil(scale * orig_w)
    pixels = TF.resize(pixels, [resize_h, resize_w], antialias=True)
    pixels = TF.center_crop(pixels, [height, width])
    return pixels[0].round().clamp(0, 255) / 127.5 - 1.0


def edit_caption_metadata(
    prompt: str, *, num_frames: int, fps: float, height: int, width: int
) -> str | None:
    """A JSON-object instruction with generation metadata, as cosmos-framework formats it.

    New keys follow the native insertion order (duration, fps, resolution); the
    JSON text is tokenized, so the order changes the token ids. Returns ``None``
    for prompts that are not a JSON object.
    """
    try:
        caption = json.loads(prompt)
    except (TypeError, ValueError):
        return None
    if not isinstance(caption, dict):
        return None
    if num_frames > 1:
        caption["duration"] = f"{int(num_frames / fps) if fps > 0 else 0}s"
        caption["fps"] = float(fps)
    else:
        caption.pop("duration", None)
        caption.pop("fps", None)
    caption["resolution"] = {"H": int(height), "W": int(width)}
    return json.dumps(caption)


def resolve_image_edit_system_prompt(system_prompt: str | None) -> str | None:
    """``None`` selects the image-edit prompt, ``""`` disables it, any other value is used as is."""
    if system_prompt is None:
        return COSMOS3_IMAGE_EDIT_SYSTEM_PROMPT
    return system_prompt or None


def _single_path(path: str | list[str] | None) -> str | None:
    if isinstance(path, (list, tuple)):
        return path[0] if path else None
    return path


def prepare_edit_inputs(batch: Req) -> None:
    """Load the edit source, fix the output geometry, and record the edit plan on ``batch``."""
    params = batch.sampling_params
    mode = params.edit_mode
    defaults = COSMOS3_EDIT_DEFAULTS[mode]
    max_pixels = params.max_pixels or defaults.max_pixels
    padding_constant = params.padding_constant or defaults.padding_constant
    first_frame = None
    if mode == EDIT_MODE_IMAGE:
        image = open_rgb_image(_single_path(batch.image_path))
        if batch.height is not None and batch.width is not None:
            pixels = load_conditioning_frame(image, batch.height, batch.width)
        else:
            pixels = load_image_edit_source(
                image, max_pixels=max_pixels, padding_constant=padding_constant
            )
        source = pixels[None, :, None]  # [1, 3, 1, H, W]
        plan = Cosmos3EditPlan(
            mode=mode,
            share_temporal_positions=False,
            system_prompt=resolve_image_edit_system_prompt(params.system_prompt),
        )
    else:
        video_path = _single_path(batch.video_path)
        video = load_video_edit_source(
            video_path,
            fps=float(batch.fps),
            max_num_frames=int(batch.num_frames),
            max_pixels=max_pixels,
            padding_constant=padding_constant,
        )
        if video.shape[1] < 2:
            raise ValueError(
                "Video editing needs at least two aligned source frames at the requested "
                f"fps; {video_path!r} decoded to one frame"
            )
        if params.edited_first_frame_path is not None:
            # Frame counts are 4N+1, so a source of at least two frames leaves
            # generated frames after the edited one.
            first_frame = load_conditioning_frame(
                open_rgb_image(params.edited_first_frame_path),
                int(video.shape[2]),
                int(video.shape[3]),
            )[None]  # [1, 3, H, W]
        source = video[None]  # [1, 3, T, H, W]
        plan = Cosmos3EditPlan(
            mode=mode,
            share_temporal_positions=True,
            system_prompt=COSMOS3_VIDEO_EDIT_SYSTEM_PROMPT,
            has_edited_first_frame=first_frame is not None,
        )
    batch.num_frames = int(source.shape[2])
    batch.height = int(source.shape[3])
    batch.width = int(source.shape[4])
    batch.extra[COSMOS3_EDIT_PLAN_KEY] = plan
    batch.extra[COSMOS3_EDIT_SOURCE_KEY] = source
    if first_frame is not None:
        batch.extra[COSMOS3_EDIT_FIRST_FRAME_KEY] = first_frame
