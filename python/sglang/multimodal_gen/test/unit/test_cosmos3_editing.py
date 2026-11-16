# SPDX-License-Identifier: Apache-2.0
"""Cosmos3.5 instruction editing: request contract, source loading and packing.

Expected values marked "i4" were produced by cosmos-framework inference
(``ddacbb1d4a3``) on the same inputs.
"""

import contextlib
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import av
import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import Cosmos3Config
from sglang.multimodal_gen.configs.sample.cosmos3 import Cosmos3SamplingParams
from sglang.multimodal_gen.configs.sample.cosmos3_editing import (
    COSMOS3_IMAGE_EDIT_SYSTEM_PROMPT,
    COSMOS3_VIDEO_EDIT_SYSTEM_PROMPT,
)
from sglang.multimodal_gen.configs.task_type import DataType, ModelTaskType
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3video import (
    Cosmos3OmniTransformer,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    cosmos3 as stages,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    cosmos3_editing as editing,
)

_CPU = torch.device("cpu")
_STAGES_MODULE = stages.__name__


def _pattern(width: int, height: int, phase: float = 0.0) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    r = 127.5 + 127.5 * np.sin(x / 23.0 + phase)
    g = 127.5 + 127.5 * np.cos(y / 17.0 - 0.5 * phase)
    b = (x * 7 + y * 13 + int(phase * 31)) % 256
    return np.stack([r, g, b], axis=-1).round().clip(0, 255).astype(np.uint8)


def _write_video(path: Path, *, fps: int, frames: int, width: int, height: int):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for i in range(frames):
            frame = av.VideoFrame.from_ndarray(
                _pattern(width, height, 0.37 * i), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _server_args(distilled_sigmas=None):
    config = Cosmos3Config()
    config.is_edge = False
    config.is_nano = False
    config.distilled_sigmas = distilled_sigmas
    return types.SimpleNamespace(
        pipeline_config=config,
        model_path="cosmos3",
        output_path=None,
        num_gpus=1,
        comfyui_mode=False,
        enable_cfg_parallel=False,
    )


def _params(distilled_sigmas=None, **kwargs) -> Cosmos3SamplingParams:
    sp = Cosmos3SamplingParams(**kwargs)
    sp._explicit_fields = set(kwargs)
    sp._adjust(_server_args(distilled_sigmas))
    return sp


def _stage(cls, **attrs):
    stage = cls.__new__(cls)
    stage.log_info = lambda *a, **k: None
    stage.use_declared_component = lambda **_kw: contextlib.nullcontext()
    for name, value in attrs.items():
        setattr(stage, name, value)
    return stage


class _Posterior:
    def __init__(self, latent):
        self._latent = latent

    def mode(self):
        return self._latent


class _PoolVAE(torch.nn.Module):
    """Causal stand-in: latent 0 sees pixel frame 0, latent k sees frames 4k-3..4k."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.config = types.SimpleNamespace(
            latents_mean=[0.0] * 48,
            latents_std=[1.0] * 48,
            scale_factor_temporal=4,
            scale_factor_spatial=16,
        )
        self.encoded = []

    def encode(self, video):
        self.encoded.append(video.clone())
        b, _, t, h, w = video.shape
        chunks = [video[:, :, :1]] + [
            video[:, :, 1 + 4 * i : 5 + 4 * i] for i in range((t - 1) // 4)
        ]
        lat = torch.stack([c.float().mean(2) for c in chunks], 2)
        lat = torch.nn.functional.avg_pool2d(lat.flatten(1, 2), 16)
        lat = lat.view(b, 3, len(chunks), h // 16, w // 16).repeat(1, 16, 1, 1, 1)
        return _Posterior(lat)


class TestCosmos3EditSourceRules(unittest.TestCase):
    def test_max_pixels_size_matches_i4(self):
        cases = {
            # (w, h, max_pixels, padding): i4 output size
            (1920, 1080, 1024 * 1024, 32): (1344, 768),
            (640, 480, 1024 * 1024, 32): (640, 480),
            (1000, 1500, 1024 * 1024, 32): (832, 1248),
            (300, 200, 1024 * 1024, 32): (288, 192),
            (640, 360, 512 * 512, 16): (640, 352),
            (1280, 720, 512 * 512, 16): (672, 384),
            (320, 240, 512 * 512, 16): (320, 240),
            (10, 10, 1024, 32): (32, 32),
        }
        for (w, h, max_pixels, padding), expected in cases.items():
            with self.subTest(size=(w, h)):
                self.assertEqual(
                    editing.max_pixels_resized_size(
                        w, h, max_pixels=max_pixels, padding_constant=padding
                    ),
                    expected,
                )

    def test_valid_video_frame_count_is_4n_plus_1(self):
        for available, cap, expected in [
            (37, 81, 37),
            (100, 81, 81),
            (20, 81, 17),
            (4, 81, 1),
            (0, 81, 0),
        ]:
            self.assertEqual(editing.valid_video_frame_count(available, cap), expected)

    def test_aligned_frame_indices_use_python_rounding(self):
        # i4: TorchCodec reports 29.999999999999996 for 70 frames at 30 fps, so
        # output frame 4 maps to 7.4999... -> 7; an exact 30.0 ties 7.5 -> 8.
        inexact = editing.aligned_frame_indices(
            num_frames=70, average_fps=29.999999999999996, target_fps=16, count=10
        )
        self.assertEqual(inexact, [0, 2, 4, 6, 7, 9, 11, 13, 15, 17])
        exact = editing.aligned_frame_indices(
            num_frames=70, average_fps=30.0, target_fps=16, count=5
        )
        self.assertEqual(exact[4], 8)
        # Indices past the source clamp to its last frame.
        self.assertEqual(
            editing.aligned_frame_indices(
                num_frames=3, average_fps=24.0, target_fps=8, count=3
            ),
            [0, 2, 2],
        )

    def test_image_source_is_a_direct_lanczos_resize(self):
        image = Image.fromarray(_pattern(1920, 1080))
        pixels = editing.load_image_edit_source(
            image, max_pixels=1024 * 1024, padding_constant=32
        )
        expected = image.resize((1344, 768), Image.LANCZOS)
        expected = torch.from_numpy(np.array(expected)).permute(2, 0, 1).float()
        self.assertEqual(tuple(pixels.shape), (3, 768, 1344))
        self.assertEqual(pixels.dtype, torch.float32)
        self.assertTrue(torch.equal(pixels, expected / 127.5 - 1.0))

    def test_edit_caption_keys_follow_native_order(self):
        # The JSON text is tokenized, so key order changes the token ids;
        # cosmos-framework appends duration, fps, then resolution.
        caption = editing.edit_caption_metadata(
            '{"edit": "make it snow"}', num_frames=17, fps=16.0, height=240, width=320
        )
        self.assertEqual(
            caption,
            '{"edit": "make it snow", "duration": "1s", "fps": 16.0, '
            '"resolution": {"H": 240, "W": 320}}',
        )
        image = editing.edit_caption_metadata(
            '{"edit": "x", "fps": 8}', num_frames=1, fps=24.0, height=64, width=96
        )
        self.assertEqual(image, '{"edit": "x", "resolution": {"H": 64, "W": 96}}')
        self.assertIsNone(
            editing.edit_caption_metadata(
                "Make it snow.", num_frames=17, fps=16.0, height=240, width=320
            )
        )

    def test_open_rgb_image_ignores_exif_orientation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rotated.jpg"
            exif = Image.Exif()
            exif[0x0112] = 6  # rotate 90 degrees on display
            Image.fromarray(_pattern(64, 32)).save(path, exif=exif)
            self.assertEqual(editing.open_rgb_image(str(path)).size, (64, 32))

    def test_video_clock_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            _write_video(path, fps=30, frames=70, width=320, height=176)
            num_frames, average_fps = editing.probe_frame_clock(str(path))
            self.assertEqual(num_frames, 70)
            self.assertAlmostEqual(average_fps, 30.0, places=9)
            video = editing.load_video_edit_source(
                str(path),
                fps=16,
                max_num_frames=81,
                max_pixels=512 * 512,
                padding_constant=16,
            )
            # floor(70 / 30 * 16) = 37 frames, already 4N+1.
            self.assertEqual(tuple(video.shape), (3, 37, 176, 320))
            self.assertEqual(video.dtype, torch.bfloat16)
            indices = editing.aligned_frame_indices(
                num_frames=70, average_fps=average_fps, target_fps=16, count=37
            )
            frames = editing.decode_video_frames(str(path), indices)
            expected = (frames.float().permute(1, 0, 2, 3) / 127.5 - 1.0).bfloat16()
            self.assertTrue(torch.equal(video, expected))


class TestCosmos3EditRequestContract(unittest.TestCase):
    def test_image_input_with_one_frame_is_an_image_edit(self):
        sp = _params(
            prompt="Make the sky purple.", image_path="/tmp/a.png", num_frames=1
        )
        self.assertEqual(sp.edit_mode, "image")
        self.assertEqual(sp.task_type, ModelTaskType.I2I)
        self.assertEqual(sp.data_type, DataType.IMAGE)
        # i4 defaults/image2image.
        self.assertEqual(sp.num_inference_steps, 50)
        self.assertEqual(sp.guidance_scale, 4.0)
        self.assertEqual(sp.flow_shift, 3.0)
        self.assertIsNone(sp.guidance_interval)
        self.assertEqual((sp.max_pixels, sp.padding_constant), (1024 * 1024, 32))
        self.assertFalse(sp.use_duration_template)
        self.assertFalse(sp.use_resolution_template)
        # The output follows the source; the stage fills the size in.
        self.assertIsNone(sp.width)
        self.assertIsNone(sp.height)

    def test_editing_instruction_becomes_the_prompt(self):
        sp = _params(
            prompt="",
            editing_instruction="  Remove the car.  ",
            image_path="/tmp/a.png",
            num_frames=1,
        )
        self.assertEqual(sp.prompt, "Remove the car.")
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            _params(
                prompt="Something else",
                editing_instruction="Remove the car.",
                image_path="/tmp/a.png",
                num_frames=1,
            )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            _params(
                prompt="",
                editing_instruction="  ",
                image_path="/tmp/a.png",
                num_frames=1,
            )

    def test_explicit_values_override_the_recipe(self):
        sp = _params(
            prompt="edit",
            image_path="/tmp/a.png",
            num_frames=1,
            guidance_scale=2.5,
            num_inference_steps=10,
            max_pixels=960 * 960,
            padding_constant=16,
            system_prompt="",
        )
        self.assertEqual(sp.guidance_scale, 2.5)
        self.assertEqual(sp.num_inference_steps, 10)
        self.assertEqual((sp.max_pixels, sp.padding_constant), (960 * 960, 16))
        self.assertEqual(sp.system_prompt, "")

    def test_video_path_with_instruction_is_a_video_edit(self):
        sp = _params(
            prompt="",
            editing_instruction="Change the pitcher to red.",
            video_path="/tmp/clip.mp4",
        )
        self.assertEqual(sp.edit_mode, "video")
        self.assertEqual(sp.task_type, ModelTaskType.V2V)
        self.assertEqual(sp.data_type, DataType.VIDEO)
        self.assertEqual(sp.prompt, "Change the pitcher to red.")
        # i4 defaults/video_editing.
        self.assertEqual(sp.num_inference_steps, 35)
        self.assertEqual(sp.guidance_scale, 7.0)
        self.assertEqual(sp.flow_shift, 5.0)
        self.assertEqual(sp.fps, 16)
        self.assertEqual(sp.num_frames, 81)
        self.assertEqual((sp.max_pixels, sp.padding_constant), (512 * 512, 16))
        self.assertFalse(sp.adjust_frames)

    def test_generation_modes_are_unchanged(self):
        t2i = _params(prompt="a cat", num_frames=1)
        self.assertIsNone(t2i.edit_mode)
        self.assertEqual(t2i.task_type, ModelTaskType.T2I)
        self.assertEqual(t2i.guidance_interval, (400.0, 1000.0))
        i2v = _params(prompt="a cat", image_path="/tmp/a.png")
        self.assertIsNone(i2v.edit_mode)
        self.assertEqual(i2v.task_type, ModelTaskType.TI2V)
        v2v = _params(prompt="continue", video_path="/tmp/clip.mp4")
        self.assertIsNone(v2v.edit_mode)
        self.assertEqual(v2v.task_type, ModelTaskType.V2V)
        self.assertEqual(v2v.guidance_scale, 4.0)

    def test_distilled_checkpoint_keeps_single_forward(self):
        sp = _params(
            distilled_sigmas=[1.0, 0.5],
            prompt="edit",
            image_path="/tmp/a.png",
            num_frames=1,
        )
        self.assertEqual(sp.guidance_scale, 1.0)
        self.assertEqual(sp.num_inference_steps, 2)

    def test_rejects_edit_fields_outside_an_edit(self):
        for kwargs in (
            {"prompt": "a cat", "system_prompt": "x"},
            {"prompt": "a cat", "editing_instruction": "x"},
            {"prompt": "a cat", "image_path": "/tmp/a.png", "max_pixels": 1024},
            {
                "prompt": "go",
                "video_path": "/tmp/c.mp4",
                "edited_first_frame_path": "/tmp/e.png",
            },
        ):
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(ValueError, "require"),
            ):
                _params(**kwargs)

    def test_rejects_inputs_the_edit_routes_do_not_take(self):
        video = {"prompt": "", "editing_instruction": "x", "video_path": "/tmp/c.mp4"}
        for extra, message in (
            ({"control_path": "/tmp/edge.mp4"}, "control_path"),
            ({"system_prompt": "custom"}, "system_prompt"),
            ({"sound_duration": 2.0}, "sound"),
            ({"edited_first_frame_path": "/tmp/e.mp4"}, "must be an image"),
            ({"width": 640, "height": 352}, "derives the output size"),
            ({"condition_frame_indexes": [0, 1]}, "condition_frame_indexes"),
        ):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, message):
                _params(**video, **extra)
        image = {"prompt": "x", "image_path": "/tmp/a.png", "num_frames": 1}
        with self.assertRaisesRegex(ValueError, "edited_first_frame_path"):
            _params(**image, edited_first_frame_path="/tmp/e.png")
        with self.assertRaisesRegex(ValueError, "both width and height"):
            _params(**image, width=640)
        with self.assertRaisesRegex(ValueError, "multiple of 16"):
            # Source-derived sizes are multiples of padding_constant and must
            # sit on the 16-px latent grid.
            _params(**image, padding_constant=24)
        with self.assertRaisesRegex(ValueError, "exactly one source image"):
            _params(prompt="x", image_path=["/tmp/a.png", "/tmp/b.png"], num_frames=1)

    def test_http_video_edit_uses_the_editing_clock(self):
        kwargs = {
            "prompt": "",
            "video_path": "/tmp/clip.mp4",
            "fps": 24,
            "num_frames": 96,
            "editing_instruction": "Make it snow.",
        }
        request = VideoGenerationsRequest(prompt="", video_path="/tmp/clip.mp4")
        lowered = Cosmos3SamplingParams.lower_video_request_kwargs(
            request, dict(kwargs)
        )
        self.assertEqual((lowered["fps"], lowered["num_frames"]), (16, 81))
        # The length comes from the source: seconds does not set the cap.
        request = VideoGenerationsRequest(
            prompt="", video_path="/tmp/clip.mp4", seconds=2, fps=10
        )
        lowered = Cosmos3SamplingParams.lower_video_request_kwargs(
            request, dict(kwargs, fps=10, num_frames=20)
        )
        self.assertEqual((lowered["fps"], lowered["num_frames"]), (10, 81))
        # Not an edit: the generic HTTP clock is kept.
        plain = dict(kwargs, editing_instruction=None)
        lowered = Cosmos3SamplingParams.lower_video_request_kwargs(
            VideoGenerationsRequest(prompt="go", video_path="/tmp/clip.mp4"), plain
        )
        self.assertEqual((lowered["fps"], lowered["num_frames"]), (24, 96))

    def test_transport_decoded_values_are_coerced(self):
        # Multipart forms JSON-decode values: a JSON-object instruction arrives as
        # a dict and must stay the instruction text; numbers may arrive as text.
        sp = _params(
            prompt="",
            editing_instruction={"edit": "night"},
            video_path="/tmp/clip.mp4",
            max_pixels="4096",
        )
        self.assertEqual(sp.prompt, '{"edit": "night"}')
        self.assertEqual(sp.max_pixels, 4096)


class _ChatTokenizer:
    """Records the conversations; ids encode the content lengths."""

    pad_token_id = 0
    eos_token_id = 7

    def __init__(self):
        self.conversations = []

    def convert_tokens_to_ids(self, token):
        return 9

    def apply_chat_template(self, conversations, tokenize, add_generation_prompt):
        self.conversations.append(conversations)
        return [len(turn["content"]) + 10 for turn in conversations]


class TestCosmos3EditStages(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch(
            f"{_STAGES_MODULE}.get_local_torch_device", return_value=_CPU
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _run(self, sp):
        req = Req(sampling_params=sp)
        server_args = _server_args()
        _stage(stages.Cosmos3ImagePreprocessStage).forward(req, server_args)
        source = req.extra[editing.COSMOS3_EDIT_SOURCE_KEY].clone()
        tokenizer = _ChatTokenizer()
        _stage(stages.Cosmos3TokenizationStage, tokenizer=tokenizer).forward(
            req, server_args
        )
        vae = _PoolVAE()
        _stage(
            stages.Cosmos3LatentPreparationStage,
            vae=vae,
            transformer=types.SimpleNamespace(latent_channel=48),
        ).forward(req, server_args)
        return req, source, tokenizer, vae

    def test_image_edit_packs_the_source_as_a_clean_prefix(self):
        path = self.root / "src.png"
        Image.fromarray(_pattern(1920, 1080)).save(path)
        sp = _params(prompt="Make the sky purple.", image_path=str(path), num_frames=1)
        req, source, tokenizer, vae = self._run(sp)
        self.assertEqual((req.height, req.width, req.num_frames), (768, 1344, 1))
        self.assertEqual(tuple(source.shape), (1, 3, 1, 768, 1344))
        plan = req.extra[editing.COSMOS3_EDIT_PLAN_KEY]
        self.assertFalse(plan.share_temporal_positions)
        # Caption: the bare instruction with the image-edit system prompt.
        cond, uncond = tokenizer.conversations
        self.assertEqual(
            cond,
            [
                {"role": "system", "content": COSMOS3_IMAGE_EDIT_SYSTEM_PROMPT},
                {"role": "user", "content": "Make the sky purple."},
            ],
        )
        self.assertEqual(uncond[1], {"role": "user", "content": ""})
        # The source is encoded into the clean prefix; the target starts from noise.
        (control,) = req.extra["control_latents"]
        self.assertEqual(tuple(control.shape), (1, 48, 1, 48, 84))
        self.assertEqual(tuple(req.latents.shape), (1, 48, 1, 48, 84))
        self.assertTrue(torch.equal(vae.encoded[0].float(), source))
        self.assertNotIn("velocity_mask", req.extra)

    def test_image_edit_system_prompt_states(self):
        path = self.root / "src.png"
        Image.fromarray(_pattern(64, 64)).save(path)
        for system_prompt, expected in (
            (None, COSMOS3_IMAGE_EDIT_SYSTEM_PROMPT),
            ("Custom.", "Custom."),
            ("", None),
        ):
            kwargs = {"prompt": "edit", "image_path": str(path), "num_frames": 1}
            if system_prompt is not None:
                kwargs["system_prompt"] = system_prompt
            _, _, tokenizer, _ = self._run(_params(**kwargs))
            cond = tokenizer.conversations[0]
            with self.subTest(system_prompt=system_prompt):
                if expected is None:
                    # Generic policy; the pipeline default has no system prompt.
                    self.assertEqual([turn["role"] for turn in cond], ["user"])
                else:
                    self.assertEqual(cond[0], {"role": "system", "content": expected})

    def test_explicit_size_uses_the_conditioning_crop(self):
        path = self.root / "src.png"
        image = Image.fromarray(_pattern(300, 200))
        image.save(path)
        sp = _params(
            prompt="edit", image_path=str(path), num_frames=1, width=160, height=128
        )
        req, source, _, _ = self._run(sp)
        self.assertEqual((req.height, req.width), (128, 160))
        expected = editing.load_conditioning_frame(image, 128, 160)
        self.assertTrue(torch.equal(source[0, :, 0], expected))

    def test_video_edit_shares_the_temporal_grid_and_conditions_frame_zero(self):
        video_path = self.root / "clip.mp4"
        _write_video(video_path, fps=16, frames=20, width=320, height=240)
        edited_path = self.root / "edited.png"
        edited = Image.fromarray(_pattern(800, 600, 3.0))
        edited.save(edited_path)
        sp = _params(
            prompt="",
            editing_instruction="Make it snow.",
            video_path=str(video_path),
            edited_first_frame_path=str(edited_path),
        )
        req, source, tokenizer, vae = self._run(sp)
        self.assertEqual((req.num_frames, req.height, req.width), (17, 240, 320))
        self.assertTrue(
            req.extra[editing.COSMOS3_EDIT_PLAN_KEY].share_temporal_positions
        )
        self.assertEqual(
            tokenizer.conversations[0][0],
            {"role": "system", "content": COSMOS3_VIDEO_EDIT_SYSTEM_PROMPT},
        )
        self.assertEqual(tokenizer.conversations[0][1]["content"], "Make it snow.")
        (control,) = req.extra["control_latents"]
        self.assertEqual(tuple(control.shape), (1, 48, 5, 15, 20))
        # Target latent 0 is the encoded edited frame and stays clean.
        velocity = req.extra["velocity_mask"][0, 0, :, 0, 0]
        self.assertEqual(velocity.tolist(), [0.0, 1.0, 1.0, 1.0, 1.0])
        first = editing.load_conditioning_frame(edited, 240, 320)
        self.assertTrue(torch.equal(vae.encoded[1][0, :, 0].float(), first))
        self.assertTrue(
            torch.equal(req.latents[:, :, 0], req.extra["condition_latents"][:, :, 0])
        )

    def test_video_edit_json_instruction_uses_native_key_order(self):
        # Edits format JSON instructions as cosmos-framework does, not with the
        # generic Cosmos3 caption helper (which puts resolution first).
        video_path = self.root / "clip.mp4"
        _write_video(video_path, fps=16, frames=20, width=320, height=240)
        sp = _params(
            prompt="",
            editing_instruction='{"edit": "make it snow"}',
            video_path=str(video_path),
        )
        _, _, tokenizer, _ = self._run(sp)
        self.assertEqual(
            tokenizer.conversations[0][1]["content"],
            '{"edit": "make it snow", "duration": "1s", "fps": 16.0, '
            '"resolution": {"H": 240, "W": 320}}',
        )

    def test_video_edit_needs_two_aligned_frames(self):
        # Four frames at 16 fps align to a single 4N+1 frame: nothing to edit.
        video_path = self.root / "short.mp4"
        _write_video(video_path, fps=16, frames=4, width=64, height=64)
        sp = _params(prompt="", editing_instruction="x", video_path=str(video_path))
        with self.assertRaisesRegex(ValueError, "at least two aligned source frames"):
            _stage(stages.Cosmos3ImagePreprocessStage).forward(
                Req(sampling_params=sp), _server_args()
            )


class TestCosmos3EditPositions(unittest.TestCase):
    """The DiT prefix layout reproduces i4's packed mRoPE positions."""

    def _positions(self, text_len, t, hp, wp, fps, share):
        dit = types.SimpleNamespace(
            temporal_margin=15000, base_fps=24.0, temporal_compression_factor=4
        )
        text_pos, vis_pos = Cosmos3OmniTransformer._compute_rope_position_ids(
            dit,
            torch.ones(1, text_len, dtype=torch.long),
            t,
            hp,
            wp,
            fps,
            _CPU,
            control_frames=[t],
            share_vision_temporal_positions=share,
        )
        return text_pos[:, 0], vis_pos[:, 0]

    def test_image_edit_items_are_consecutive(self):
        # i4 img_landscape: 36 text tokens, a 24x42 patch grid per item.
        text_pos, vis = self._positions(36, 1, 24, 42, 24.0, share=False)
        self.assertEqual(text_pos[:, -1].tolist(), [35, 35, 35])
        n = 24 * 42
        self.assertEqual(vis[:, 0].tolist(), [15036, 0, 0])
        self.assertEqual(vis[:, n - 1].tolist(), [15036, 23, 41])
        self.assertEqual(vis[:, n].tolist(), [15037, 0, 0])
        self.assertEqual(vis[:, -1].tolist(), [15037, 23, 41])

    def test_video_edit_items_share_fps_scaled_positions(self):
        # i4 vid_30fps: 37 text tokens, 10 latent frames of 11x20 at 16 fps.
        _, vis = self._positions(37, 10, 11, 20, 16.0, share=True)
        n = 10 * 11 * 20
        self.assertTrue(torch.equal(vis[:, :n], vis[:, n:]))
        self.assertEqual(vis[:, 0].tolist(), [15037.0, 0.0, 0.0])
        self.assertEqual(vis[:, n - 1].tolist(), [15050.5, 10.0, 19.0])


class TestCosmos3EditDenoise(unittest.TestCase):
    def test_both_cfg_branches_carry_the_source_with_the_plan_layout(self):
        stage = stages.Cosmos3DenoisingStage.__new__(stages.Cosmos3DenoisingStage)
        seen = []

        def fake_run(*, latents, cache_key, control_latents, **_):
            seen.append(
                (cache_key, control_latents, stage._share_vision_temporal_positions)
            )
            return torch.zeros_like(latents)

        stage.transformer = mock.Mock()
        stage._run_transformer = fake_run
        stage.scheduler = mock.Mock(step=lambda noise_pred, t, latents, **_: (latents,))
        stage.progress_bar = lambda iterable, **_: iterable
        stage.log_info = lambda *_: None
        stage.check_cancelled = lambda _: None
        stage._logged_parallel_config = True
        stage._logged_cfg_split = True
        stage._current_batch_is_warmup = False
        text = torch.zeros(1, 4, dtype=torch.long)
        source = torch.ones(1, 48, 1, 2, 2)
        plan = editing.Cosmos3EditPlan(
            mode="image", share_temporal_positions=False, system_prompt=None
        )
        batch = types.SimpleNamespace(
            sampling_params=types.SimpleNamespace(
                guidance_interval=None,
                control_guidance=1.0,
                share_vision_temporal_positions=True,
            ),
            extra={
                editing.COSMOS3_EDIT_PLAN_KEY: plan,
                "control_latents": [source],
                "cond_text_ids": text,
                "cond_text_mask": torch.ones_like(text),
                "uncond_text_ids": text,
                "uncond_text_mask": torch.ones_like(text),
                "cond_text_seq_len": 4,
                "uncond_text_seq_len": 4,
                "video_shape": (1, 2, 2),
            },
            latents=torch.zeros(1, 48, 1, 2, 2),
            audio_latents=None,
            timesteps=torch.tensor([999.0, 500.0]),
            guidance_scale=4.0,
            generator=None,
            seed=None,
            scheduler=None,
            rollout=False,
            profile=False,
            is_warmup=False,
        )
        server_args = types.SimpleNamespace(
            enable_cfg_parallel=False, pipeline_config=None
        )
        patches = {
            "get_sp_world_size": 1,
            "rollout_step_kwargs": {},
            "prepare_rollout_if_enabled": None,
            "init_rollout_env_if_enabled": None,
            "finalize_rollout_if_enabled": None,
        }
        with contextlib.ExitStack() as stack:
            for name, value in patches.items():
                stack.enter_context(
                    mock.patch(f"{_STAGES_MODULE}.{name}", return_value=value)
                )
            stage._denoise_once(batch, server_args)
        self.assertEqual([key for key, _, _ in seen], ["cond", "uncond"] * 2)
        for _, control, share in seen:
            self.assertEqual(len(control), 1)
            self.assertIs(control[0], source)
            self.assertFalse(share)


if __name__ == "__main__":
    unittest.main()
