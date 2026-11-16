# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 joint video + action prediction (``action_mode="wam"``).

The ``*MatchesNative`` cases hold values dumped from the native implementation
(imaginaire4 ``ddacbb1d4a3``: ``inference/preprocessing/action.py::build_action_batch``,
``transforms.py::build_sequence_plan_from_mode``, ``get_3d_mrope_ids_vae_tokens``,
``ActionProcessor.postprocess_action``) for the same inputs.
"""

import contextlib
import json
import types
import unittest
from unittest import mock

import numpy as np
import torch

from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import Cosmos3Config
from sglang.multimodal_gen.configs.sample.cosmos3 import (
    COSMOS3_WAM_DEFAULTS,
    Cosmos3SamplingParams,
)
from sglang.multimodal_gen.configs.task_type import DataType
from sglang.multimodal_gen.runtime.entrypoints.action.protocol import (
    action_generation_response,
    action_metadata,
    action_raw_response,
    build_action_sampling_params,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3video import (
    compute_mrope_position_ids_action,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
    Cosmos3DecodingStage,
    Cosmos3ImagePreprocessStage,
    Cosmos3LatentPreparationStage,
    Cosmos3TokenizationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_action import (
    build_action_prompt,
    canonical_aspect_ratio,
)

_STAGES = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3"
)


def _server_args(*, is_edge=False, batching_max_size=1):
    config = Cosmos3Config()
    config.is_edge = is_edge
    return types.SimpleNamespace(
        model_id=None,
        model_path="nvidia/Cosmos3-Nano",
        served_model_name="cosmos3-production",
        backend=None,
        pipeline_class_name=None,
        output_path=None,
        comfyui_mode=False,
        num_gpus=1,
        tp_size=1,
        sp_degree=1,
        ulysses_degree=1,
        ring_degree=1,
        batching_max_size=batching_max_size,
        pipeline_config=config,
    )


def _user_params(explicit: dict, **fields) -> Cosmos3SamplingParams:
    """A request as the user-facing builders create it: ``explicit`` was set by the user."""
    params = Cosmos3SamplingParams(**explicit, **fields)
    params._explicit_fields = set(explicit)
    return params


# (prompt, view_point, num_frames, fps, canvas H, canvas W, format_prompt_as_json, native caption)
_NATIVE_CAPTIONS = [
    (
        "Pick up the banana and place it in the bowl",
        "ego_view",
        17,
        24,
        192,
        320,
        True,
        '{"cinematography": {"framing": "This video is captured from a first-person perspective looking at the scene."}, "actions": [{"time": "0:00-0:01", "description": "Pick up the banana and place it in the bowl."}], "duration": "0s", "fps": 24.0, "resolution": {"H": 192, "W": 320}, "aspect_ratio": "16,9"}',
    ),
    (
        "pick up the cup.",
        "concat_view",
        17,
        5,
        480,
        832,
        True,
        '{"cinematography": {"framing": "This video contains concatenated views from multiple camera perspectives."}, "actions": [{"time": "0:00-0:03", "description": "pick up the cup."}], "duration": "3s", "fps": 5.0, "resolution": {"H": 480, "W": 832}, "aspect_ratio": "16,9"}',
    ),
    (
        "  open the drawer  ",
        "wrist_view",
        33,
        15,
        640,
        640,
        True,
        '{"cinematography": {"framing": "This video is captured from a wrist-mounted camera."}, "actions": [{"time": "0:00-0:02", "description": "open the drawer."}], "duration": "2s", "fps": 15.0, "resolution": {"H": 640, "W": 640}, "aspect_ratio": "1,1"}',
    ),
    (
        "Fold the towel!",
        "third_person_view",
        17,
        10,
        544,
        736,
        True,
        '{"cinematography": {"framing": "This video is captured from a third-person perspective looking towards the agent from the front."}, "actions": [{"time": "0:00-0:02", "description": "Fold the towel!"}], "duration": "1s", "fps": 10.0, "resolution": {"H": 544, "W": 736}, "aspect_ratio": "4,3"}',
    ),
    (
        "stack the blocks",
        "top_down_2d_view",
        17,
        24,
        256,
        256,
        True,
        '{"cinematography": {"framing": "This video is captured from a top-down view of a flat 2D scene."}, "actions": [{"time": "0:00-0:01", "description": "stack the blocks."}], "duration": "0s", "fps": 24.0, "resolution": {"H": 256, "W": 256}, "aspect_ratio": "1,1"}',
    ),
    (
        "drive forward",
        "video_game_view",
        61,
        30,
        720,
        1280,
        True,
        '{"cinematography": {"framing": "This video is captured from an in-game camera perspective in a virtual environment."}, "actions": [{"time": "0:00-0:02", "description": "drive forward."}], "duration": "2s", "fps": 30.0, "resolution": {"H": 720, "W": 1280}, "aspect_ratio": "16,9"}',
    ),
    (
        "pour the water",
        "unknown_view",
        17,
        24,
        192,
        320,
        True,
        '{"actions": [{"time": "0:00-0:01", "description": "pour the water."}], "duration": "0s", "fps": 24.0, "resolution": {"H": 192, "W": 320}, "aspect_ratio": "16,9"}',
    ),
    ("", "ego_view", 17, 24, 192, 320, True, ""),
    (
        "Pick up the banana and place it in the bowl",
        "ego_view",
        17,
        24,
        192,
        320,
        False,
        "Pick up the banana and place it in the bowl",
    ),
    (
        "long horizon task with a duration over a minute",
        "ego_view",
        81,
        1,
        192,
        320,
        True,
        '{"cinematography": {"framing": "This video is captured from a first-person perspective looking at the scene."}, "actions": [{"time": "0:00-1:21", "description": "long horizon task with a duration over a minute."}], "duration": "81s", "fps": 1.0, "resolution": {"H": 192, "W": 320}, "aspect_ratio": "16,9"}',
    ),
]

# Native caption aspect keys of canvases whose reduced ratio differs from the key.
_NATIVE_ASPECT_LABELS = {
    (416, 304): "4,3",
    (480, 272): "16,9",
    (960, 544): "16,9",
    (544, 960): "9,16",
    (1184, 880): "4,3",
    (1360, 768): "16,9",
    (1920, 1088): "16,9",
    (1968, 1472): "4,3",
    (2272, 1280): "16,9",
    (3160, 2368): "4,3",
    (3640, 2048): "16,9",
    (7280, 4096): "16,9",
    (2208, 1664): "4,3",
    (1104, 832): "4,3",
    (512, 768): "2,3",
    (320, 192): "16,9",
}

# Native wam plan (video_length, action_length): every action row is generated,
# the first latent frame is clean, and action row k sits on video frame k + 1.
_NATIVE_WAM_PLANS = [
    (
        17,
        16,
        {
            "condition_frame_indexes_vision": [0],
            "condition_frame_indexes_action": [],
            "action_start_frame_offset": 1,
        },
    ),
    (
        33,
        32,
        {
            "condition_frame_indexes_vision": [0],
            "condition_frame_indexes_action": [],
            "action_start_frame_offset": 1,
        },
    ),
    (
        5,
        4,
        {
            "condition_frame_indexes_vision": [0],
            "condition_frame_indexes_action": [],
            "action_start_frame_offset": 1,
        },
    ),
]

# get_3d_mrope_ids_vae_tokens(grid_t, 1, 1, temporal_offset, fps=fps, base_fps=24,
# temporal_compression_factor=1, base_temporal_compression_factor=4, start_frame_offset=1)[0][0]
_NATIVE_ACTION_MROPE_T = [
    (16, 0, None, [float(i) for i in range(1, 17)]),
    (16, 0, 24.0, [0.25 * i for i in range(1, 17)]),
    (
        16,
        0,
        5.0,
        [
            1.2000000476837158,
            2.4000000953674316,
            3.6000001430511475,
            4.800000190734863,
            6.0,
            7.200000286102295,
            8.399999618530273,
            9.600000381469727,
            10.799999237060547,
            12.0,
            13.200000762939453,
            14.40000057220459,
            15.59999942779541,
            16.799999237060547,
            18.0,
            19.200000762939453,
        ],
    ),
    (32, 0, 15.0, [0.4 * i for i in range(1, 33)]),
    (16, 3, 24.0, [3.0 + 0.25 * i for i in range(1, 17)]),
]


class TestCosmos3WamMatchesNative(unittest.TestCase):
    def test_caption_matches_native_request_builder(self):
        for prompt, view, frames, fps, h, w, as_json, expected in _NATIVE_CAPTIONS:
            with self.subTest(prompt=prompt, view=view, as_json=as_json):
                caption = build_action_prompt(
                    prompt, view, frames, fps, h, w, format_prompt_as_json=as_json
                )
                self.assertEqual(caption, expected)

    def test_batched_captions_match_one_by_one(self):
        prompts = [case[0] for case in _NATIVE_CAPTIONS[:2]]
        captions = build_action_prompt(prompts, "ego_view", 17, 24, 192, 320)
        self.assertEqual(
            captions,
            [build_action_prompt(p, "ego_view", 17, 24, 192, 320) for p in prompts],
        )

    def test_aspect_labels_match_native_size_table(self):
        for (width, height), label in _NATIVE_ASPECT_LABELS.items():
            with self.subTest(size=f"{width}x{height}"):
                self.assertEqual(canonical_aspect_ratio(width, height), label)

    def test_action_mrope_matches_native(self):
        for grid_t, offset, fps, expected in _NATIVE_ACTION_MROPE_T:
            with self.subTest(grid_t=grid_t, offset=offset, fps=fps):
                ids, _ = compute_mrope_position_ids_action(
                    grid_t=grid_t,
                    temporal_offset=offset,
                    action_fps=fps,
                    device=torch.device("cpu"),
                    base_fps=24.0,
                    base_temporal_compression_factor=4,
                    start_frame_offset=1,
                )
                torch.testing.assert_close(
                    ids[0].float(), torch.tensor(expected), rtol=0, atol=1e-5
                )
                self.assertTrue(torch.all(ids[1:] == 0))


class TestCosmos3WamSequenceLayout(unittest.TestCase):
    """wam lays out the native plan and conditions exactly like policy."""

    @classmethod
    def setUpClass(cls):
        cls.stage = Cosmos3LatentPreparationStage.__new__(Cosmos3LatentPreparationStage)
        cls.stage.transformer = types.SimpleNamespace(action_dim=64)
        cls.stage.log_info = lambda *a, **k: None

    def _prepare(self, mode, num_frames, seed=0):
        sp = Cosmos3SamplingParams(
            prompt="t",
            num_frames=num_frames,
            action_mode=mode,
            domain_name="droid_lerobot",
        )
        batch = types.SimpleNamespace(
            sampling_params=sp,
            num_frames=num_frames,
            raw_latent_shape=(1, 48, 1, 1, 1),
            extra={},
        )
        gen = torch.Generator().manual_seed(seed)
        self.stage._prepare_action_latents(
            batch, gen, torch.device("cpu"), torch.float32
        )
        return batch

    def test_plan_matches_native(self):
        for video_length, action_length, plan in _NATIVE_WAM_PLANS:
            with self.subTest(video_length=video_length):
                batch = self._prepare("wam", video_length)
                self.assertEqual(batch.action_latents.shape[1], action_length)
                noisy = batch.extra["action_velocity_mask"][0, :, 0]
                clean_rows = [i for i in range(action_length) if noisy[i] == 0]
                self.assertEqual(clean_rows, plan["condition_frame_indexes_action"])
                self.assertEqual(
                    batch.extra["action_start_frame_offset"],
                    plan["action_start_frame_offset"],
                )
                # Native action conditioning for wam is the all-zero clean action.
                self.assertTrue(torch.all(batch.extra["action_condition_latents"] == 0))

    def test_wam_and_policy_draw_identical_action_noise(self):
        wam = self._prepare("wam", 17, seed=3)
        policy = self._prepare("policy", 17, seed=3)
        torch.testing.assert_close(wam.action_latents, policy.action_latents)
        torch.testing.assert_close(
            wam.extra["action_velocity_mask"], policy.extra["action_velocity_mask"]
        )
        self.assertEqual(wam.extra["raw_action_dim"], 10)

    def test_wam_shares_the_policy_observation_preprocessing(self):
        from PIL import Image

        image = Image.fromarray(
            np.random.default_rng(0).integers(0, 255, (40, 60, 3), dtype=np.uint8)
        )

        def preprocess(action_mode, data_type):
            stage = Cosmos3ImagePreprocessStage.__new__(Cosmos3ImagePreprocessStage)
            stage.log_info = lambda *_args, **_kwargs: None
            batch = types.SimpleNamespace(
                image_path=image,
                video_path=None,
                height=32,
                width=48,
                data_type=data_type,
                sampling_params=types.SimpleNamespace(action_mode=action_mode),
                preprocessed_image=None,
            )
            with mock.patch(
                f"{_STAGES}.get_local_torch_device", return_value=torch.device("cpu")
            ):
                stage.forward(batch, types.SimpleNamespace())
            return batch.preprocessed_image

        wam_video = preprocess("wam", DataType.VIDEO)
        wam_action = preprocess("wam", DataType.ACTION)
        policy = preprocess("policy", DataType.ACTION)
        torch.testing.assert_close(wam_video, policy)
        torch.testing.assert_close(wam_action, policy)


class TestCosmos3WamSamplingParams(unittest.TestCase):
    def test_video_request_stays_visual_with_native_defaults(self):
        params = _user_params(
            {"prompt": "pick", "image_path": "obs.png", "action_mode": "WAM"}
        )
        params._adjust(_server_args())

        self.assertEqual(params.action_mode, "wam")
        self.assertEqual(params.data_type, DataType.VIDEO)
        for name, value in COSMOS3_WAM_DEFAULTS.items():
            self.assertEqual(getattr(params, name), value, name)

    def test_explicit_fields_win_over_native_defaults(self):
        params = _user_params(
            {
                "prompt": "pick",
                "image_path": "obs.png",
                "action_mode": "wam",
                "guidance_scale": 3.0,
                "num_inference_steps": 4,
                "num_frames": 33,
                "fps": 15,
                "width": 832,
                "height": 480,
            }
        )
        params._adjust(_server_args())

        self.assertEqual(
            (params.guidance_scale, params.num_inference_steps, params.num_frames),
            (3.0, 4, 33),
        )
        self.assertEqual((params.fps, params.width, params.height), (15, 832, 480))

    def test_edge_keeps_native_wam_guidance(self):
        params = _user_params(
            {"prompt": "pick", "image_path": "obs.png", "action_mode": "wam"}
        )
        params._adjust(_server_args(is_edge=True))

        self.assertEqual(params.guidance_scale, 1.0)

    def test_action_payload_request_returns_actions(self):
        params = Cosmos3SamplingParams(
            prompt="pick",
            image_path="obs.png",
            action_mode="wam",
            num_frames=17,
            return_action_payload=True,
        )
        params._adjust(_server_args())

        self.assertEqual(params.data_type, DataType.ACTION)
        self.assertFalse(params.save_output)
        self.assertIsNone(params.output_file_name)

    def test_wam_requires_one_observation_image(self):
        for fields in (
            {"prompt": "pick"},
            {"prompt": "pick", "image_path": "obs.png", "video_path": "obs.mp4"},
        ):
            with self.subTest(fields=fields):
                params = _user_params({**fields, "action_mode": "wam"})
                with self.assertRaisesRegex(ValueError, "observation image"):
                    params._adjust(_server_args())

    def test_video_api_lowering_uses_native_clip_defaults(self):
        request = VideoGenerationsRequest(
            prompt="pick", input_reference="obs.png", action_mode="wam"
        )
        kwargs = Cosmos3SamplingParams.lower_video_request_kwargs(
            request, {"num_frames": 96, "fps": 24, "action_mode": "wam"}
        )
        self.assertEqual((kwargs["num_frames"], kwargs["fps"]), (17, 24))

        request = VideoGenerationsRequest(
            prompt="pick",
            input_reference="obs.png",
            action_mode="wam",
            seconds=2,
            fps=8,
        )
        kwargs = Cosmos3SamplingParams.lower_video_request_kwargs(
            request,
            {
                "num_frames": 16,
                "fps": 8,
                "action_mode": "wam",
                "format_prompt_as_json": "false",
            },
        )
        self.assertEqual((kwargs["num_frames"], kwargs["fps"]), (16, 8))
        self.assertIs(kwargs["format_prompt_as_json"], False)

    def test_video_api_accepts_wam_fields(self):
        self.assertIn(
            "format_prompt_as_json", Cosmos3SamplingParams.video_request_extra_fields()
        )
        self.assertNotIn(
            "return_action_payload", Cosmos3SamplingParams.video_request_extra_fields()
        )


class TestCosmos3WamActionEndpoint(unittest.TestCase):
    @staticmethod
    def _payload(batch_size=1, mode="wam", **parameters):
        images = np.zeros((batch_size, 8, 8, 3), dtype=np.uint8)
        return {
            "input": {"prompt": "pick up the block", "input_reference": images},
            "parameters": {
                "action_mode": mode,
                "domain_name": "droid_lerobot",
                **parameters,
            },
        }

    def test_wam_request_builds_action_payload_params(self):
        params = build_action_sampling_params(
            self._payload(format_prompt_as_json=False), _server_args()
        )

        self.assertEqual(params.action_mode, "wam")
        self.assertEqual(params.data_type, DataType.ACTION)
        self.assertTrue(params.return_action_payload)
        self.assertFalse(params.format_prompt_as_json)
        self.assertEqual((params.num_frames, params.guidance_scale), (17, 1.0))

    def test_wam_batches_like_policy(self):
        params = build_action_sampling_params(
            self._payload(batch_size=2), _server_args(batching_max_size=2)
        )

        self.assertEqual(len(params.image_path), 2)
        self.assertEqual(params.prompt, ["pick up the block"] * 2)

    def test_wam_requires_observation_image(self):
        payload = {
            "input": {"prompt": "pick", "observation": {}},
            "parameters": {"action_mode": "wam", "domain_name": "droid_lerobot"},
        }
        with self.assertRaisesRegex(
            ValueError, "wam input requires an observation image"
        ):
            build_action_sampling_params(payload, _server_args())

    def test_metadata_lists_wam(self):
        capabilities = action_metadata(_server_args())["capabilities"]

        self.assertIn("wam", capabilities["action_modes"])
        self.assertIn("wam", capabilities["batched_action_modes"])

    def _output(self, batch_size):
        shape = (batch_size, 4, 3) if batch_size > 1 else (4, 3)
        video_shape = (batch_size, 5, 2, 2, 3) if batch_size > 1 else (5, 2, 2, 3)
        return {
            "request_id": "wam-1",
            "actions": np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
            "video": np.full(video_shape, 7, dtype=np.uint8),
            "fps": 5,
            "action_mode": "wam",
            "domain_id": 8,
            "raw_action_dim": 3,
        }

    def test_response_carries_video_beside_actions(self):
        response = action_generation_response(self._output(1), _server_args())
        item = response["data"][0]

        self.assertEqual(item["action"]["shape"], [4, 3])
        self.assertEqual(item["action"]["action_mode"], "wam")
        self.assertEqual(item["video"]["shape"], [5, 2, 2, 3])
        self.assertEqual(item["video"]["dtype"], "uint8")
        self.assertEqual(item["video"]["fps"], 5)
        self.assertEqual(item["video"]["values"][0][0][0], [7, 7, 7])

    def test_msgpack_response_keeps_numpy_frames(self):
        response = action_generation_response(
            self._output(2), _server_args(), preserve_numpy=True
        )

        self.assertEqual(len(response["data"]), 2)
        for item in response["data"]:
            self.assertIsInstance(item["video"]["values"], np.ndarray)
            self.assertEqual(item["video"]["values"].shape, (5, 2, 2, 3))

    def test_response_rejects_video_batch_mismatch(self):
        output = self._output(2)
        output["video"] = output["video"][:1]
        with self.assertRaisesRegex(ValueError, "wam video output"):
            action_generation_response(output, _server_args())

    def test_policy_response_has_no_video(self):
        output = self._output(1)
        del output["video"]
        response = action_generation_response(output, _server_args())
        self.assertNotIn("video", response["data"][0])

    def test_raw_response_lists_video(self):
        response = action_raw_response(self._output(1))
        self.assertIsInstance(response["video"], list)
        self.assertIsInstance(response["actions"], list)


class TestCosmos3WamDecode(unittest.TestCase):
    @staticmethod
    def _stage(decoded):
        stage = Cosmos3DecodingStage.__new__(Cosmos3DecodingStage)
        stage.vae = mock.Mock()
        stage.sound_tokenizer = None
        stage._guardrails = False
        stage.log_info = lambda *_args, **_kwargs: None
        stage.log_debug = lambda *_args, **_kwargs: None
        stage.use_declared_component = lambda **_kwargs: contextlib.nullcontext()
        stage._decode_latents = mock.Mock(return_value=decoded)
        return stage

    @staticmethod
    def _batch(data_type, batch_size=1):
        return types.SimpleNamespace(
            data_type=data_type,
            latents=torch.zeros(batch_size, 4, 2, 2, 2),
            action_latents=torch.arange(batch_size * 24, dtype=torch.float32).reshape(
                batch_size, 4, 6
            ),
            audio_latents=None,
            extra={
                "raw_action_dim": 3,
                "action_domain_ids": torch.tensor([8] * batch_size),
            },
            sampling_params=Cosmos3SamplingParams(
                prompt="t", action_mode="wam", domain_name="droid_lerobot"
            ),
            request_id="wam-decode",
            num_inference_steps=30,
            num_frames=5,
            fps=5,
            metrics=None,
            use_guardrails=None,
            rollout_trajectory_data=None,
        )

    def test_action_payload_carries_decoded_frames(self):
        decoded = torch.ones(1, 3, 5, 2, 2)  # [-1, 1] decoder range: white frames
        stage = self._stage(decoded)

        output = stage.forward(self._batch(DataType.ACTION), types.SimpleNamespace())
        payload = output.output[0]

        self.assertEqual(payload["actions"].shape, (4, 3))
        self.assertEqual(payload["video"].shape, (5, 2, 2, 3))
        self.assertEqual(payload["video"].dtype, np.uint8)
        self.assertTrue(np.all(payload["video"] == 255))
        self.assertEqual(payload["fps"], 5)
        self.assertEqual(payload["action_mode"], "wam")
        stage._decode_latents.assert_called_once()

    def test_batched_action_payload_keeps_batch_dimension(self):
        stage = self._stage(torch.zeros(2, 3, 5, 2, 2))

        output = stage.forward(
            self._batch(DataType.ACTION, batch_size=2), types.SimpleNamespace()
        )

        self.assertEqual(output.output[0]["video"].shape, (2, 5, 2, 2, 3))
        self.assertEqual(output.output[0]["actions"].shape, (2, 4, 3))

    def test_video_request_returns_video_and_actions(self):
        stage = self._stage(torch.zeros(1, 3, 5, 2, 2))
        server_args = types.SimpleNamespace(output_path=None)

        output = stage.forward(self._batch(DataType.VIDEO), server_args)

        self.assertEqual(tuple(output.output.shape), (1, 3, 5, 2, 2))
        torch.testing.assert_close(
            output.action_pred,
            torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)[:, :, :3],
        )
        self.assertEqual(output.action_mode, "wam")
        self.assertEqual(output.action_domain_id, 8)


class TestCosmos3WamTokenization(unittest.TestCase):
    def _captions(self, *, negative_prompt, format_prompt_as_json=True):
        seen = []

        def fake_tokenize(text, *args, **kwargs):
            seen.append(text)
            return (
                torch.zeros(1, 4, dtype=torch.long),
                torch.ones(1, 4, dtype=torch.long),
                4,
            )

        stage = Cosmos3TokenizationStage.__new__(Cosmos3TokenizationStage)
        batch = types.SimpleNamespace(
            prompt="pick up the cup",
            negative_prompt=negative_prompt,
            max_sequence_length=512,
            use_duration_template=None,
            use_resolution_template=None,
            negative_metadata_mode=None,
            use_system_prompt=None,
            fps=24,
            num_frames=17,
            height=192,
            width=320,
            data_type=DataType.VIDEO,
            sampling_params=Cosmos3SamplingParams(
                action_mode="wam", format_prompt_as_json=format_prompt_as_json
            ),
            extra={},
        )
        server_args = types.SimpleNamespace(pipeline_config=Cosmos3Config())
        with (
            mock.patch.object(stage, "_tokenize_prompt", fake_tokenize),
            mock.patch.object(stage, "log_info"),
        ):
            stage.forward(batch, server_args)
        return seen

    def test_unconditional_branch_is_the_empty_caption(self):
        cond, uncond = self._captions(negative_prompt="")
        self.assertEqual(
            json.loads(cond)["actions"][0]["description"], "pick up the cup."
        )
        self.assertEqual(uncond, "")

    def test_explicit_negative_prompt_is_kept(self):
        _, uncond = self._captions(negative_prompt="blurry")
        self.assertEqual(uncond, "blurry")

    def test_plain_caption_is_sent_as_given(self):
        cond, _ = self._captions(negative_prompt="", format_prompt_as_json=False)
        self.assertEqual(cond, "pick up the cup")


if __name__ == "__main__":
    unittest.main()
