# SPDX-License-Identifier: Apache-2.0
"""Cosmos3.5 LR-to-HR super-resolution: request contract, LR conditioning, positions.

CPU only. Element-wise parity with cosmos-framework (capability resolution,
LR fit/pad, video frame selection, packed positions) is checked by the WS3
parity harness; these cases pin the derived rules the pipeline relies on.
"""

import contextlib
import os
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import Cosmos3Config
from sglang.multimodal_gen.configs.sample.cosmos3 import Cosmos3SamplingParams
from sglang.multimodal_gen.configs.sample.cosmos3_super_resolution import (
    Cosmos3SuperResolutionConfig,
    lr_video_frame_indices,
    parse_super_resolution_config,
    plan_resolution_adapter,
    resolve_sr_capability,
    resolve_sr_plan,
)
from sglang.multimodal_gen.configs.task_type import DataType
from sglang.multimodal_gen.runtime.models.dits.cosmos3video import (
    Cosmos3OmniTransformer,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
    Cosmos3DecodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_super_resolution import (
    SR_OUTPUT_SIZE_KEY,
    Cosmos3SuperResolutionStage,
    adapt_super_resolution_output,
    adapt_super_resolution_pixels,
    fit_super_resolution_pixels,
    probe_video,
    super_resolution_output_size,
    uint8_to_normalized_bf16,
)

_VIDEO_MODES = ("text2video", "image2video", "video2video")
_IMAGE_MODES = ("text2image", "image2image")
_ENCODE_CHUNK_FRAMES = {"256": 68, "480": 24, "720": 8, "544": 8, "1080": 4, "1440": 4}


def _capabilities(lr: str, hr: str, frames=(17, 21, 397), modes=None):
    caps = []
    for mode in modes or _IMAGE_MODES + _VIDEO_MODES:
        for num_frames in (1,) if mode in _IMAGE_MODES else frames:
            caps.append(
                dict(
                    modality=mode,
                    hr_resolution=hr,
                    num_frames=num_frames,
                    tokenizer_resolution=hr,
                    lr_resolution=lr,
                )
            )
    return caps


def _published(block, *, flow_shift=7.0, budget=None, encode=None):
    settings = {
        "resolution": "1080",
        "tokenizer": {"encode_chunk_frames": encode or _ENCODE_CHUNK_FRAMES},
    }
    if block is not None:
        settings["high_resolution"] = block
    if budget is not None:
        settings["max_num_tokens_after_packing"] = budget
    return parse_super_resolution_config(
        {"model": {"config": settings}}, {"flow_shift": flow_shift}
    )


def _sr_config(capabilities=None, *, inference=True, budget=1_000_000, **kwargs):
    return _published(
        dict(
            enabled=True,
            inference_capabilities_enabled=inference,
            sr_workflow="sequence_lr_conditioning",
            capabilities=capabilities
            or _capabilities("544", "1080") + _capabilities("720", "1440"),
        ),
        budget=budget,
        **kwargs,
    )


def _server_args(sr_config):
    return types.SimpleNamespace(
        model_id=None,
        model_path="/models/cosmos3-sr",
        served_model_name="cosmos3",
        backend=None,
        pipeline_class_name=None,
        output_path=None,
        comfyui_mode=False,
        num_gpus=8,
        tp_size=1,
        sp_degree=1,
        ulysses_degree=1,
        ring_degree=1,
        batching_max_size=1,
        pipeline_config=Cosmos3Config(super_resolution=sr_config),
    )


def _adjusted(sr_config, **kwargs):
    params = Cosmos3SamplingParams(prompt="a red cube", **kwargs)
    params._explicit_fields = set(kwargs)
    params._adjust(_server_args(sr_config))
    return params


class TestCosmos3SuperResolutionConfig(unittest.TestCase):
    def test_published_schema(self):
        """The published block is validated as cosmos-framework HighResolutionConfig does."""
        disabled = _published(None)
        self.assertFalse(disabled.enabled)
        self.assertEqual(disabled.capabilities, ())
        typed = _published(
            {
                "_type": "HighResolutionConfig",
                "enabled": True,
                "sr_workflow": "sequence_lr_conditioning",
                "capabilities": [{**_capabilities("544", "1080")[0], "_type": "cap"}],
            }
        )
        self.assertEqual(len(typed.capabilities), 1)
        cap = _capabilities("544", "1080")[0]
        invalid = {
            "enabled without workflow": dict(enabled=True, sr_workflow="none"),
            "workflow without capabilities": dict(
                enabled=True, sr_workflow="sequence_lr_conditioning", capabilities=[]
            ),
            "duplicate": dict(
                enabled=True,
                sr_workflow="sequence_lr_conditioning",
                capabilities=[cap, cap],
            ),
            "unknown modality": dict(
                enabled=True,
                sr_workflow="sequence_lr_conditioning",
                capabilities=[{**cap, "modality": "video_editing"}],
            ),
            "unknown key": dict(
                enabled=True,
                sr_workflow="sequence_lr_conditioning",
                capabilities=[cap],
                extra=1,
            ),
        }
        for name, block in invalid.items():
            with self.subTest(name), self.assertRaises((TypeError, ValueError)):
                _published(block)

    def test_checkpoint_metadata(self):
        """The bound shift is the scheduler's; an unpublished packing budget is i4's 13312."""
        config = _published(None, flow_shift=10.0)
        self.assertEqual(config.inference_flow_shift, 10.0)
        self.assertEqual(config.max_num_tokens_after_packing, 13312)
        self.assertEqual(config.default_resolution, "1080")
        self.assertEqual(config.tokenizer_routes, frozenset(_ENCODE_CHUNK_FRAMES))


class TestCosmos3SuperResolutionCapability(unittest.TestCase):
    @staticmethod
    def _resolve(config, **kwargs):
        args = dict(
            modality="video2video",
            resolution="1080",
            aspect_ratio="16,9",
            num_frames=17,
            lr_resolution=None,
        )
        args.update(kwargs)
        return resolve_sr_capability(config, **args)

    def test_training_only_capabilities_are_refused(self):
        with self.assertRaisesRegex(ValueError, "training and architecture validation"):
            self._resolve(_sr_config(inference=False))

    def test_lr_tier_must_be_unambiguous(self):
        config = _sr_config(
            _capabilities("544", "1080")
            + _capabilities("720", "1080", modes=("video2video",))
        )
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self._resolve(config)
        self.assertEqual(
            self._resolve(config, lr_resolution="720").lr_resolution, "720"
        )
        # Only video2video declares the 720 route.
        self.assertEqual(
            self._resolve(config, modality="text2video").lr_resolution, "544"
        )

    def test_frame_counts_are_exact(self):
        config = _sr_config()
        self.assertIsNotNone(self._resolve(config, num_frames=21))
        with self.assertRaisesRegex(ValueError, "does not declare"):
            self._resolve(config, num_frames=25)
        with self.assertRaisesRegex(ValueError, "Image output requires one frame"):
            self._resolve(config, modality="image2image")

    def test_routes_need_tokenizer_chunks(self):
        config = _sr_config(encode={"720": 8, "1080": 4})
        with self.assertRaisesRegex(ValueError, "LR tokenizer route '544'"):
            self._resolve(config)

    def test_undeclared_tier_is_not_an_sr_route(self):
        self.assertIsNone(self._resolve(_sr_config(), resolution="720"))


class TestCosmos3SuperResolutionRequest(unittest.TestCase):
    def test_video_request_binds_route(self):
        params = _adjusted(
            _sr_config(),
            lr_vision_path="/data/lr.mp4",
            width=1920,
            height=1088,
            num_frames=17,
        )
        self.assertEqual(params.sr_modality, "video2video")
        self.assertEqual(params.lr_resolution, "544")
        self.assertEqual(params.flow_shift, 7.0)
        self.assertEqual(params.num_frames, 17)
        self.assertEqual(params.data_type, DataType.VIDEO)
        # The declared count is exact; 8-GPU alignment must not move it.
        self.assertFalse(params.adjust_frames)
        # Declared canvases are on the 16-pixel latent grid, so alignment keeps them.
        self.assertEqual((params.width, params.height), (1920, 1088))

    def test_image_lr_is_a_single_frame_image_route(self):
        params = _adjusted(
            _sr_config(), lr_vision_path="/data/lr.png", width=2560, height=1440
        )
        self.assertEqual(params.sr_modality, "image2image")
        self.assertEqual(params.lr_resolution, "720")
        self.assertEqual(params.num_frames, 1)
        self.assertEqual(params.data_type, DataType.IMAGE)

    def test_image2video_source_image_is_not_conditioning(self):
        params = _adjusted(
            _sr_config(),
            lr_vision_path="/data/lr.mp4",
            image_path="/data/first_frame.png",
            width=1920,
            height=1088,
            num_frames=17,
        )
        self.assertEqual(params.sr_modality, "image2video")
        self.assertIsNone(params.image_path)
        with self.assertRaisesRegex(ValueError, "requires one source image"):
            _adjusted(
                _sr_config(),
                lr_vision_path="/data/lr.mp4",
                sr_modality="image2video",
                width=1920,
                height=1088,
                num_frames=17,
            )

    def test_shift_is_bound_to_the_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "does not match model inference shift"):
            _adjusted(
                _sr_config(),
                lr_vision_path="/data/lr.mp4",
                width=1920,
                height=1088,
                num_frames=17,
                flow_shift=10.0,
            )

    def test_declared_canvas_requires_lr(self):
        with self.assertRaisesRegex(ValueError, "requires lr_vision_path"):
            _adjusted(_sr_config(), width=1920, height=1088, num_frames=17)
        # 1920x1090 would be floored onto the 1920x1088 canvas.
        with self.assertRaisesRegex(ValueError, "requires lr_vision_path"):
            _adjusted(_sr_config(), width=1920, height=1090, num_frames=17)
        # Undeclared canvases and checkpoints without SR keep the ordinary path.
        plain = _adjusted(_sr_config(), width=1280, height=720, num_frames=17)
        self.assertIsNone(plain.lr_resolution)
        _adjusted(Cosmos3SuperResolutionConfig(), width=1920, height=1088)

    def test_rejected_requests(self):
        lr = dict(lr_vision_path="/data/lr.mp4", width=1920, height=1088, num_frames=17)
        cases = {
            "no SR checkpoint": (Cosmos3SuperResolutionConfig(), lr, "capability"),
            "transfer": (_sr_config(), dict(lr, control_path="/c.mp4"), "transfer"),
            "sound": (_sr_config(), dict(lr, sound_duration=1.0), "sound"),
            "action": (_sr_config(), dict(lr, action_mode="policy"), "action"),
            "source video": (_sr_config(), dict(lr, video_path="/v.mp4"), "video_path"),
            # An image in with one frame out is otherwise an image2image edit.
            "source image on an image route": (
                _sr_config(),
                dict(
                    lr_vision_path="/data/lr.png",
                    image_path="/data/source.png",
                    width=1920,
                    height=1088,
                ),
                "does not accept image_path",
            ),
            "image LR for video": (
                _sr_config(),
                dict(lr, lr_vision_path="/data/lr.png", sr_modality="video2video"),
                "video media",
            ),
            "undeclared canvas": (
                _sr_config(),
                dict(lr, width=1280, height=720),
                "not a declared SR canvas",
            ),
            "LR fields without LR": (
                _sr_config(),
                dict(width=1280, height=720, lr_resolution="544"),
                "apply only",
            ),
        }
        for name, (config, kwargs, message) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                _adjusted(config, **kwargs)

    def test_packed_token_budget(self):
        """QHD at 397 frames needs 100 * 45 * 80 + 2 tokens before the LR item."""
        with self.assertRaisesRegex(ValueError, "360002 packed tokens"):
            resolve_sr_plan(
                _sr_config(budget=360_001),
                modality="video2video",
                hr_size=(2560, 1440),
                num_frames=397,
                lr_resolution=None,
            )
        plan = resolve_sr_plan(
            _sr_config(budget=360_002),
            modality="video2video",
            hr_size=(2560, 1440),
            num_frames=397,
            lr_resolution=None,
        )
        self.assertEqual(
            (plan.lr_size, plan.spatial_position_scale), ((1280, 720), (2.0, 2.0))
        )


class TestCosmos3SuperResolutionAdapter(unittest.TestCase):
    """A 720p source asks for 1080p while the checkpoint declares 544->1080 and 720->1440."""

    def _plan(self, policy, **kwargs):
        args = dict(
            policy=policy,
            modality="video2video",
            hr_size=(1920, 1088),
            num_frames=17,
            requested_input_size=(1280, 720),
            requested_output_size=None,
        )
        args.update(kwargs)
        return plan_resolution_adapter(_sr_config(), **args)

    def test_cost_first_down_resizes_the_source(self):
        plan = self._plan("cost_first")
        self.assertEqual(plan.route, "cost_first")
        self.assertEqual(plan.capability_lr_resolution, "544")
        self.assertEqual(plan.model_input_content_size, (960, 540))
        self.assertEqual(plan.requested_output_size, (1920, 1080))
        self.assertFalse(plan.resizes_output)

    def test_quality_first_keeps_source_pixels(self):
        plan = self._plan("quality_first")
        self.assertEqual(plan.capability_hr_resolution, "1440")
        self.assertEqual(plan.model_input_content_size, (1280, 720))
        self.assertEqual(plan.model_output_content_size, (2560, 1440))
        self.assertTrue(plan.resizes_output)

    def test_never_upscales_outside_the_model(self):
        with self.assertRaisesRegex(
            ValueError, "No 'quality_first' resolution-adapter route"
        ):
            self._plan("quality_first", requested_output_size=(3200, 1800))
        with self.assertRaisesRegex(ValueError, "must not upscale"):
            adapt_super_resolution_pixels(
                torch.zeros(3, 1, 4, 4), target_width=8, target_height=8
            )

    def test_request_binds_the_selected_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lr.png")
            Image.fromarray(np.zeros((720, 1280, 3), dtype=np.uint8)).save(path)
            params = _adjusted(
                _sr_config(),
                lr_vision_path=path,
                width=1920,
                height=1088,
                resolution_adapter_policy="quality_first",
            )
            self.assertEqual((params.width, params.height), (2560, 1440))
            self.assertEqual(params.lr_resolution, "720")
            self.assertEqual(params.resolution_adapter_output_size, (1920, 1080))
            with self.assertRaisesRegex(ValueError, "conflicts"):
                _adjusted(
                    _sr_config(),
                    lr_vision_path=path,
                    width=1920,
                    height=1088,
                    lr_resolution="544",
                    resolution_adapter_policy="quality_first",
                )
        with self.assertRaisesRegex(ValueError, "requires an explicit"):
            _adjusted(
                _sr_config(),
                lr_vision_path="/data/lr.png",
                width=1920,
                height=1088,
                resolution_adapter_output_size="1280x720",
            )

    def test_output_is_cropped_then_down_resized(self):
        decoded = torch.rand(1, 3, 1, 1440, 2560) * 2 - 1
        decoded[..., 1400:, :] = (
            5.0  # bucket padding must not leak into the clamp range
        )
        out = adapt_super_resolution_output(decoded, ((2560, 1400), (1920, 1050)))
        self.assertEqual(tuple(out.shape), (1, 3, 1, 1050, 1920))
        self.assertLessEqual(float(out.max()), float(decoded[..., :1400, :].max()))


class TestCosmos3SuperResolutionLRConditioning(unittest.TestCase):
    def test_bucket_padding(self):
        rows = torch.arange(540).remainder(256).to(torch.uint8)
        pixels = rows.view(1, 1, 540, 1).expand(3, 1, 540, 960)
        fitted = fit_super_resolution_pixels(
            uint8_to_normalized_bf16(pixels), expected_width=960, expected_height=544
        )
        self.assertEqual(tuple(fitted.shape), (3, 1, 544, 960))
        self.assertEqual(fitted.dtype, torch.bfloat16)
        # The 4 padding rows reflect the content rows above them.
        self.assertTrue(
            torch.equal(fitted[..., 540:, :], fitted[..., 535:539, :].flip(-2))
        )
        # A pad at least as large as the content repeats the edge instead.
        wide = torch.full((3, 1, 500, 2000), 9, dtype=torch.uint8)
        wide[..., -1, :] = 200
        fitted = fit_super_resolution_pixels(
            uint8_to_normalized_bf16(wide), expected_width=960, expected_height=544
        )
        self.assertTrue(
            torch.equal(
                fitted[..., 240:, :],
                fitted[..., 239:240, :].expand_as(fitted[..., 240:, :]),
            )
        )

    def test_frame_selection_never_repeats(self):
        self.assertEqual(
            lr_video_frame_indices(
                average_fps=30, num_source_frames=50, target_fps=24, num_frames=5
            ),
            [0, 1, 2, 4, 5],
        )
        with self.assertRaisesRegex(ValueError, "without repeating"):
            lr_video_frame_indices(
                average_fps=24, num_source_frames=12, target_fps=24, num_frames=17
            )
        with self.assertRaisesRegex(ValueError, "distinct"):
            lr_video_frame_indices(
                average_fps=24, num_source_frames=60, target_fps=30, num_frames=17
            )

    def test_output_keeps_lr_content_region(self):
        self.assertEqual(
            super_resolution_output_size(
                source_size=(960, 540),
                lr_bucket_size=(960, 544),
                hr_bucket_size=(1920, 1088),
            ),
            (1920, 1080),
        )
        self.assertEqual(
            super_resolution_output_size(
                source_size=(1000, 1000),
                lr_bucket_size=(1280, 720),
                hr_bucket_size=(2560, 1440),
            ),
            (1440, 1440),
        )

    def test_probe_counts_content_frames(self):
        try:
            import av
        except ImportError:
            self.skipTest("PyAV is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clip.mp4")
            with av.open(path, "w") as container:
                stream = container.add_stream("libx264", rate=8)
                stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
                stream.options = {"bf": "2"}
                for t in range(10):
                    frame = np.full((48, 64, 3), t * 20, dtype=np.uint8)
                    for packet in stream.encode(
                        av.VideoFrame.from_ndarray(frame, format="rgb24")
                    ):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            num_frames, fps = probe_video(path)
        self.assertEqual(num_frames, 10)
        self.assertAlmostEqual(fps, 8.0, places=6)


class TestCosmos3SuperResolutionPositions(unittest.TestCase):
    @staticmethod
    def _model():
        model = Cosmos3OmniTransformer.__new__(Cosmos3OmniTransformer)
        model.temporal_margin = 15000
        model.base_fps = 24.0
        model.temporal_compression_factor = 4
        return model

    def test_lr_patches_sit_on_hr_patch_centers(self):
        model = self._model()
        text_mask = torch.ones(1, 4)
        T = 3
        _, base = model._compute_rope_position_ids(
            text_mask, T, 4, 6, fps=16.0, device=torch.device("cpu")
        )
        _, positions = model._compute_rope_position_ids(
            text_mask,
            T,
            4,
            6,
            fps=16.0,
            device=torch.device("cpu"),
            control_frames=[T],
            control_grids=[(2, 3)],
            control_spatial_position_scales=[(2.0, 2.0)],
        )
        lr = positions[:, 0, : T * 6].reshape(3, T, 2, 3)
        hr = positions[:, 0, T * 6 :]
        self.assertTrue(torch.equal(hr, base[:, 0]))
        self.assertTrue(torch.equal(lr[0, :, 0, 0], hr.reshape(3, T, 4, 6)[0, :, 0, 0]))
        self.assertEqual(lr[1, 0, :, 0].tolist(), [0.5, 2.5])
        self.assertEqual(lr[2, 0, 0, :].tolist(), [0.5, 2.5, 4.5])

    def test_non_integer_scale_on_images(self):
        _, positions = self._model()._compute_rope_position_ids(
            torch.ones(1, 4),
            1,
            3,
            3,
            fps=None,
            device=torch.device("cpu"),
            control_frames=[1],
            control_grids=[(2, 2)],
            control_spatial_position_scales=[(1.5, 1.5)],
        )
        lr = positions[:, 0, :4].reshape(3, 2, 2)
        self.assertTrue(torch.allclose(lr[1, :, 0], torch.tensor([0.25, 1.75])))
        self.assertTrue(torch.equal(lr[0], positions[0, 0, 4:].reshape(3, 3)[:2, :2]))


class _LatentVAE(torch.nn.Module):
    """Encodes to a 16x-downsampled latent and records its input."""

    def __init__(self, channels=4):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        self.config = types.SimpleNamespace(
            latents_mean=[0.5] * channels, latents_std=[2.0] * channels
        )
        self.channels = channels
        self.inputs = []

    def encode(self, pixels):
        self.inputs.append(pixels)
        b, _, t, h, w = pixels.shape
        latent = torch.full((b, self.channels, 1 + (t - 1) // 4, h // 16, w // 16), 2.5)
        return types.SimpleNamespace(mode=lambda: latent.to(pixels.dtype))

    def decode(self, z):
        return torch.zeros(z.shape[0], 3, 1, z.shape[-2] * 16, z.shape[-1] * 16)


class TestCosmos3SuperResolutionStages(unittest.TestCase):
    def _batch(self, lr_path, **params):
        sampling = Cosmos3SamplingParams(
            prompt="p", lr_vision_path=lr_path, sr_modality="image2image", **params
        )
        return types.SimpleNamespace(
            sampling_params=sampling,
            width=1920,
            height=1088,
            num_frames=1,
            fps=24,
            latents=torch.zeros(1, 4, 1, 68, 120, dtype=torch.bfloat16),
            extra={},
        )

    def test_lr_becomes_a_scaled_clean_prefix(self):
        vae = _LatentVAE()
        stage = Cosmos3SuperResolutionStage(vae)
        stage.use_declared_component = lambda **_kwargs: contextlib.nullcontext()
        server_args = types.SimpleNamespace(
            pipeline_config=types.SimpleNamespace(super_resolution=_sr_config())
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lr.png")
            Image.fromarray(np.full((540, 960, 3), 255, dtype=np.uint8)).save(path)
            with mock.patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages."
                "model_specific_stages.cosmos3_super_resolution.get_local_torch_device",
                return_value=torch.device("cpu"),
            ):
                batch = stage.forward(self._batch(path), server_args)
        (pixels,) = vae.inputs
        self.assertEqual(tuple(pixels.shape), (1, 3, 1, 544, 960))
        self.assertEqual(pixels.dtype, torch.bfloat16)
        (lr_latent,) = batch.extra["control_latents"]
        self.assertEqual(tuple(lr_latent.shape), (1, 4, 1, 34, 60))
        # Normalized with the VAE latent statistics: (2.5 - 0.5) / 2.
        self.assertTrue(torch.all(lr_latent == 1.0))
        self.assertEqual(
            batch.extra["transformer_extra_kwargs"]["control_spatial_position_scales"],
            [(2.0, 2.0)],
        )
        self.assertEqual(batch.extra[SR_OUTPUT_SIZE_KEY], ((1920, 1080), (1920, 1080)))

        plain = types.SimpleNamespace(
            sampling_params=Cosmos3SamplingParams(prompt="p"), extra={}
        )
        self.assertIs(stage.forward(plain, server_args), plain)
        self.assertEqual(plain.extra, {})

    def test_decode_crops_to_the_lr_content(self):
        stage = Cosmos3DecodingStage.__new__(Cosmos3DecodingStage)
        stage.vae = _LatentVAE()
        stage.vae.supports_decode_on_frames = True
        stage.sound_tokenizer = None
        stage._guardrails = False
        stage._latents_mean = stage._latents_std = None
        stage.log_info = stage.log_debug = lambda *_args, **_kwargs: None
        stage.use_declared_component = lambda **_kwargs: contextlib.nullcontext()
        batch = types.SimpleNamespace(
            data_type=DataType.IMAGE,
            latents=torch.zeros(1, 4, 1, 68, 120),
            extra={SR_OUTPUT_SIZE_KEY: ((1920, 1080), (1920, 1080))},
            sampling_params=Cosmos3SamplingParams(prompt="p"),
            action_latents=None,
            audio_latents=None,
            use_guardrails=None,
            metrics=None,
            rollout_trajectory_data=None,
        )
        with mock.patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages."
            "model_specific_stages.cosmos3.streamed_video_output_path",
            return_value="/tmp/out.mp4",
        ):
            self.assertIsNone(
                stage._encode_while_decoding(batch, types.SimpleNamespace())
            )
            output = stage.forward(batch, types.SimpleNamespace())
        self.assertEqual(tuple(output.output.shape), (1, 3, 1, 1080, 1920))


if __name__ == "__main__":
    unittest.main()
