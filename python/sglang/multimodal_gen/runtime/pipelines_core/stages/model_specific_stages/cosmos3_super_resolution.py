# SPDX-License-Identifier: Apache-2.0
"""Cosmos3.5 LR-to-HR super-resolution: supplied-LR conditioning.

The LR media is fitted into its declared bucket, VAE-encoded and packed as a
clean GEN prefix ahead of the generated HR latent, with the HR item's temporal
positions and its spatial positions scaled into HR patch coordinates. The
decoded HR canvas is then cropped to the region the LR content covers.

Mirrors cosmos-framework ``inference/preprocessing/super_resolution.py`` and
``inference/postprocessing/super_resolution.py``, including the resolution
adapter; the internal-only LR degradation is not ported.
"""

import contextlib
import math
import os
import tempfile
from collections.abc import Iterator
from io import BytesIO
from urllib.parse import unquote, urlparse

import numpy as np
import PIL.Image
import requests
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from sglang.multimodal_gen.configs.sample.cosmos3_super_resolution import (
    SR_IMAGE_EXTENSIONS,
    Cosmos3SRAdapterPlan,
    Cosmos3SRPlan,
    build_resolution_adapter_plan,
    fit_super_resolution_content_size,
    lr_video_frame_indices,
    media_extension,
    parse_size,
    resolve_sr_plan,
    scale_super_resolution_content_size,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.srt.multimodal.media_io import get_image_bytes

# Request-scoped keys this stage writes into ``Req.extra``.
SR_OUTPUT_SIZE_KEY = "super_resolution_output_size"
CONTROL_LATENTS_KEY = "control_latents"
TRANSFORMER_EXTRA_KWARGS_KEY = "transformer_extra_kwargs"
_VIDEO_DOWNLOAD_TIMEOUT_S = 60
_VAE_PHASE = "super_resolution_encode"


def uint8_to_normalized_bf16(pixels: torch.Tensor) -> torch.Tensor:
    """uint8 [0, 255] -> [-1, 1] in bfloat16 arithmetic, as cosmos-framework does."""
    return pixels.to(dtype=torch.bfloat16) / 127.5 - 1.0


def adapt_super_resolution_pixels(
    pixels: torch.Tensor,  # [C,T,H_src,W_src]
    *,
    target_width: int,
    target_height: int,
) -> torch.Tensor:  # [C,T,H_target,W_target]
    """Same-aspect down-resize for the resolution adapter, clamped to the source range."""
    if pixels.ndim != 4:
        raise ValueError(
            f"SR boundary adaptation requires [C,T,H,W] pixels, got {pixels.ndim} dimensions."
        )
    source_height, source_width = int(pixels.shape[-2]), int(pixels.shape[-1])
    if target_width <= 0 or target_height <= 0:
        raise ValueError(
            "SR adapter target dimensions must be positive, got "
            f"{target_width}x{target_height}."
        )
    if target_width > source_width or target_height > source_height:
        raise ValueError(
            f"SR boundary adaptation must not upscale {source_width}x{source_height} "
            f"pixels to {target_width}x{target_height}."
        )
    if source_width * target_height != target_width * source_height:
        raise ValueError(
            "SR boundary adaptation cannot change aspect ratio from "
            f"{source_width}x{source_height} to {target_width}x{target_height}."
        )
    if (source_width, source_height) == (target_width, target_height):
        return pixels
    source_min, source_max = float(pixels.min()), float(pixels.max())
    resized = F.interpolate(
        pixels.permute(1, 0, 2, 3).to(torch.float32),
        size=(target_height, target_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(source_min, source_max)
    return resized.permute(1, 0, 2, 3).to(dtype=pixels.dtype)


def fit_super_resolution_pixels(
    pixels: torch.Tensor,  # [C,T,H_src,W_src] normalized
    *,
    expected_width: int,
    expected_height: int,
    adapter_source_size: tuple[int, int] | None = None,
) -> torch.Tensor:  # [C,T,H_expected,W_expected]
    """Fit normalized LR pixels into the LR bucket and pad bottom/right like training.

    With a resolution-adapter plan the content is only ever down-resized.
    """
    source_height, source_width = int(pixels.shape[-2]), int(pixels.shape[-1])
    content_width, content_height = fit_super_resolution_content_size(
        (source_width, source_height), (expected_width, expected_height)
    )
    if adapter_source_size is not None:
        pixels = adapt_super_resolution_pixels(
            pixels, target_width=content_width, target_height=content_height
        )
    elif (content_width, content_height) != (source_width, source_height):
        frames = F.interpolate(
            pixels.permute(1, 0, 2, 3).to(torch.float32),
            size=(content_height, content_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )  # [T,C,H_content,W_content]
        pixels = frames.clamp(-1.0, 1.0).permute(1, 0, 2, 3).to(dtype=pixels.dtype)
    pad_right = expected_width - content_width
    pad_bottom = expected_height - content_height
    if pad_right == 0 and pad_bottom == 0:
        return pixels
    # Reflection needs a pad smaller than the content; fall back to edge padding.
    mode = (
        "edge"
        if pad_right >= content_width or pad_bottom >= content_height
        else "reflect"
    )
    return TF.pad(pixels, [0, 0, pad_right, pad_bottom], padding_mode=mode).contiguous()


def canonicalize_lr_pixels(pixels: torch.Tensor) -> torch.Tensor:
    """``[1,3,T,H,W]`` CPU bfloat16 LR item, finite and within [-1, 1]."""
    if pixels.ndim != 5 or pixels.shape[0] != 1 or pixels.shape[1] != 3:
        raise ValueError(
            f"LR pixels must have shape [1,3,T,H,W], got {tuple(pixels.shape)}."
        )
    canonical = pixels.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    if not torch.isfinite(canonical.float()).all():
        raise ValueError("LR pixels must contain only finite values.")
    if float(canonical.min()) < -1.0 or float(canonical.max()) > 1.0:
        raise ValueError("LR pixels must be normalized to [-1, 1].")
    return canonical


def _open_lr_image(source: str) -> PIL.Image.Image:
    # No EXIF transpose: cosmos-framework reads the stored pixel grid.
    if os.path.isfile(source):
        with open(source, "rb") as file:
            return PIL.Image.open(BytesIO(file.read())).convert("RGB")
    return PIL.Image.open(BytesIO(get_image_bytes(source))).convert("RGB")


@contextlib.contextmanager
def _local_video(source: str) -> Iterator[str]:
    if os.path.isfile(source):
        yield source
        return
    if not source.startswith(("http://", "https://")):
        raise ValueError(f"LR video {source!r} is neither a file nor an http(s) URL.")
    suffix = os.path.splitext(unquote(urlparse(source).path))[1] or ".mp4"
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, f"lr_video{suffix}")
        with contextlib.closing(
            requests.get(source, stream=True, timeout=_VIDEO_DOWNLOAD_TIMEOUT_S)
        ) as response:
            if response.status_code != 200:
                raise ValueError(
                    f"Failed to download LR video. Status code: {response.status_code}"
                )
            with open(path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    file.write(chunk)
        yield path


def _video_dimensions(path: str) -> tuple[int, int]:
    import av

    with av.open(path) as container:
        frame = next(container.decode(container.streams.video[0]))
        return int(frame.width), int(frame.height)


def probe_super_resolution_source_size(source: str) -> tuple[int, int]:
    """(width, height) of LR media as stored (no EXIF transpose)."""
    if media_extension(source) in SR_IMAGE_EXTENSIONS:
        return _open_lr_image(source).size
    with _local_video(source) as path:
        return _video_dimensions(path)


def _check_adapter_source_size(
    source_size: tuple[int, int], adapter_source_size: tuple[int, int] | None
) -> None:
    if adapter_source_size is not None and tuple(source_size) != tuple(
        adapter_source_size
    ):
        raise ValueError(
            f"LR source is {source_size[0]}x{source_size[1]}; expected {adapter_source_size}."
        )


def probe_video(path: str) -> tuple[int, float]:
    """``(num_frames, average_fps)`` from a full packet scan.

    Matches TorchCodec's exact seek mode: frames counted from the content and
    the rate taken over ``[first pts, last pts + duration)``.
    """
    import av

    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {path!r}")
        stream = container.streams.video[0]
        time_base = stream.time_base
        num_frames = 0
        begin = end = None
        for packet in container.demux(stream):
            if packet.pts is None or packet.size == 0:
                continue
            num_frames += 1
            start = packet.pts * time_base
            stop = (packet.pts + (packet.duration or 0)) * time_base
            begin = start if begin is None else min(begin, start)
            end = stop if end is None else max(end, stop)
        header_fps = float(stream.average_rate) if stream.average_rate else 0.0
    if num_frames == 0:
        raise ValueError(f"No frames in LR video {path!r}")
    if begin is None or end is None or end <= begin:
        return num_frames, header_fps
    return num_frames, float(num_frames / (end - begin))


def _decode_video_frames(path: str, indices: list[int]) -> torch.Tensor:
    """``uint8 [T,3,H,W]`` frames at presentation-order ``indices``."""
    import av

    wanted = {index: position for position, index in enumerate(indices)}
    frames: list[np.ndarray | None] = [None] * len(indices)
    last = max(indices)
    with av.open(path) as container:
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index in wanted:
                frames[wanted[frame_index]] = frame.to_ndarray(format="rgb24")
            if frame_index >= last:
                break
    if any(frame is None for frame in frames):
        raise ValueError(f"Decoded fewer LR frames than requested from {path!r}.")
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()


def load_super_resolution_source(
    source: str,
    *,
    expected_width: int,
    expected_height: int,
    target_fps: float,
    num_frames: int,
    adapter_source_size: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """LR media fitted to its bucket: ``([C,T,H,W] bf16, source (width, height))``."""
    if num_frames == 1:
        image = _open_lr_image(source)
        _check_adapter_source_size(image.size, adapter_source_size)
        pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(1)
        source_size = image.size
    else:
        with _local_video(source) as path:
            _check_adapter_source_size(_video_dimensions(path), adapter_source_size)
            num_source_frames, average_fps = probe_video(path)
            indices = lr_video_frame_indices(
                average_fps=average_fps,
                num_source_frames=num_source_frames,
                target_fps=target_fps,
                num_frames=num_frames,
            )
            frames = _decode_video_frames(path, indices)  # [T,C,H,W]
        pixels = frames.permute(1, 0, 2, 3).contiguous()  # [C,T,H,W]
        source_size = (int(pixels.shape[-1]), int(pixels.shape[-2]))
    fitted = fit_super_resolution_pixels(
        uint8_to_normalized_bf16(pixels),
        expected_width=expected_width,
        expected_height=expected_height,
        adapter_source_size=adapter_source_size,
    )
    return fitted, source_size


def crop_scaled_condition_latent(
    lr_latent: torch.Tensor,  # [B,C,T,h,w]
    *,
    target_latent_size: tuple[int, int],  # (h, w) of the HR latent
    spatial_position_scale: tuple[float, float],
) -> torch.Tensor:
    """Keep the LR rows/columns that cover the HR latent (``crop_scaled_condition_items``)."""
    target_height, target_width = target_latent_size
    crop_height = min(
        int(lr_latent.shape[-2]), math.ceil(target_height / spatial_position_scale[0])
    )
    crop_width = min(
        int(lr_latent.shape[-1]), math.ceil(target_width / spatial_position_scale[1])
    )
    return lr_latent[..., :crop_height, :crop_width].contiguous()


def super_resolution_output_size(
    *,
    source_size: tuple[int, int],
    lr_bucket_size: tuple[int, int],
    hr_bucket_size: tuple[int, int],
) -> tuple[int, int]:
    """(width, height) of the HR content generated from a padded LR bucket."""
    lr_content = fit_super_resolution_content_size(source_size, lr_bucket_size)
    return scale_super_resolution_content_size(
        lr_content, lr_bucket_size, hr_bucket_size
    )


def adapt_super_resolution_output(
    decoded: torch.Tensor,  # [B,C,T,H_canvas,W_canvas]
    output: tuple[tuple[int, int], tuple[int, int]],
) -> torch.Tensor:  # [B,C,T,H_out,W_out]
    """Keep the generated content of the HR canvas, then apply the adapter's down-resize."""
    (content_width, content_height), (final_width, final_height) = output
    decoded = decoded[..., :content_height, :content_width]
    if (final_width, final_height) == (content_width, content_height):
        return decoded
    return torch.stack(
        [
            adapt_super_resolution_pixels(
                item, target_width=final_width, target_height=final_height
            )
            for item in decoded
        ]
    )


def encode_lr_latent(vae, pixels: torch.Tensor) -> torch.Tensor:
    """Normalized VAE latent (posterior mode) of ``[B,3,T,H,W]`` pixels."""
    latent = vae.encode(pixels).mode()
    mean = (
        torch.as_tensor(vae.config.latents_mean)
        .view(1, -1, 1, 1, 1)
        .to(latent.device, latent.dtype)
    )
    std = (
        torch.as_tensor(vae.config.latents_std)
        .view(1, -1, 1, 1, 1)
        .to(latent.device, latent.dtype)
    )
    return (latent - mean) / std


def plan_for_request(batch: Req, server_args: ServerArgs) -> Cosmos3SRPlan | None:
    """The resolved SR route of a request, or ``None`` for any other request."""
    params = batch.sampling_params
    if params.lr_vision_path is None:
        return None
    return resolve_sr_plan(
        server_args.pipeline_config.super_resolution,
        modality=params.sr_modality,
        hr_size=(int(batch.width), int(batch.height)),
        num_frames=int(batch.num_frames),
        lr_resolution=params.lr_resolution,
    )


class Cosmos3SuperResolutionStage(PipelineStage):
    """Encode the supplied LR into a clean, position-scaled GEN prefix.

    Runs after latent preparation: the HR latent is pure noise of the HR
    canvas. Writes the LR latent to ``control_latents`` with its spatial
    scale in ``transformer_extra_kwargs`` and the HR content size the decoder
    crops to. No-op for requests without ``lr_vision_path``.
    """

    parallelism_type = StageParallelismType.REPLICATED

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        plan = plan_for_request(batch, server_args)
        if plan is None:
            return batch
        if batch.latents is None:
            raise ValueError("Super-resolution needs the prepared HR latent.")
        adapter = self._adapter_plan(batch, server_args, plan)
        lr_width, lr_height = plan.lr_size
        lr_pixels, source_size = load_super_resolution_source(
            batch.sampling_params.lr_vision_path,
            expected_width=lr_width,
            expected_height=lr_height,
            target_fps=float(batch.fps or 24.0),
            num_frames=plan.num_frames,
            adapter_source_size=(
                adapter.requested_input_size if adapter is not None else None
            ),
        )
        lr_pixels = canonicalize_lr_pixels(lr_pixels.unsqueeze(0))

        device = get_local_torch_device()
        with self.use_declared_component(
            component_name="vae", module=self.vae, phase=_VAE_PHASE
        ):
            vae_dtype = next(self.vae.parameters()).dtype
            with torch.no_grad():
                lr_latent = encode_lr_latent(
                    self.vae, lr_pixels.to(device=device, dtype=vae_dtype)
                )
        scale = plan.spatial_position_scale
        lr_latent = crop_scaled_condition_latent(
            lr_latent,
            target_latent_size=tuple(batch.latents.shape[-2:]),
            spatial_position_scale=scale,
        ).to(batch.latents.dtype)
        if lr_latent.shape[0] != batch.latents.shape[0]:
            lr_latent = lr_latent.expand(batch.latents.shape[0], -1, -1, -1, -1)

        extra_kwargs = dict(batch.extra.get(TRANSFORMER_EXTRA_KWARGS_KEY) or {})
        extra_kwargs["control_spatial_position_scales"] = [scale]
        batch.extra[TRANSFORMER_EXTRA_KWARGS_KEY] = extra_kwargs
        batch.extra[CONTROL_LATENTS_KEY] = [lr_latent]
        if adapter is not None:
            output = (adapter.model_output_content_size, adapter.requested_output_size)
        else:
            content = super_resolution_output_size(
                source_size=source_size,
                lr_bucket_size=plan.lr_size,
                hr_bucket_size=plan.hr_size,
            )
            output = (content, content)
        batch.extra[SR_OUTPUT_SIZE_KEY] = output
        self.log_info(
            f"Super-resolution {plan.modality}: LR {source_size[0]}x{source_size[1]} -> "
            f"bucket {lr_width}x{lr_height} ({plan.lr_resolution}), HR canvas "
            f"{plan.hr_size[0]}x{plan.hr_size[1]} ({plan.hr_resolution}), LR latent "
            f"{tuple(lr_latent.shape)}, output {batch.extra[SR_OUTPUT_SIZE_KEY]}"
        )
        return batch

    @staticmethod
    def _adapter_plan(
        batch: Req, server_args: ServerArgs, plan: Cosmos3SRPlan
    ) -> Cosmos3SRAdapterPlan | None:
        params = batch.sampling_params
        if params.resolution_adapter_policy is None:
            return None
        adapter = build_resolution_adapter_plan(
            policy=params.resolution_adapter_policy,
            capabilities=server_args.pipeline_config.super_resolution.capabilities,
            modality=plan.modality,
            num_frames=plan.num_frames,
            aspect_ratio=plan.aspect_ratio,
            requested_input_size=probe_super_resolution_source_size(
                params.lr_vision_path
            ),
            requested_output_size=parse_size(params.resolution_adapter_output_size),
        )
        if (adapter.model_input_size, adapter.model_output_size) != (
            plan.lr_size,
            plan.hr_size,
        ):
            raise ValueError(
                "Resolution adapter plan does not match the resolved LR/HR route."
            )
        return adapter

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "vae",
                phase=_VAE_PHASE,
                allow_prefetch=False,
                start_at_stage_entry=False,
            )
        ]
