# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Cosmos3 inverse dynamics: human-motion domains, the
Image2Action (frame-0 pose) layout, action normalizers, plain action captions,
bucket input fitting and the unified_v1 action schema."""

import json
import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import Cosmos3Config
from sglang.multimodal_gen.configs.sample.cosmos3 import Cosmos3SamplingParams
from sglang.multimodal_gen.configs.task_type import DataType
from sglang.multimodal_gen.runtime.entrypoints.action.protocol import (
    build_action_sampling_params,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
    Cosmos3DecodingStage,
    Cosmos3ImagePreprocessStage,
    Cosmos3LatentPreparationStage,
    Cosmos3TokenizationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_action import (
    ASPECT_RATIO_SIZE_INFO,
    EMBODIMENT_TO_DOMAIN_ID,
    EMBODIMENT_TO_RAW_ACTION_DIM,
    VIDEO_RES_SIZE_INFO,
    action_bucket_size,
    action_layout,
    add_action_mode_metadata,
    build_plain_action_caption,
    default_action_stats_key,
    denormalize_action_rows,
    fit_action_video,
    load_action_normalizer,
    load_action_stats,
    normalize_action,
    render_action_prompt,
    resolve_action_normalizer,
    resolve_raw_action_dim,
    select_action_stats_path,
    unified_action_valid_mask,
    unified_to_native_action,
)

HAND_CAPTION_POSTFIX = (
    "The video is captured from a real-world environment. "
    "The video shows a human actor. "
    "This video is captured from a static perspective looking towards the actor. "
    "The action performed by the human actor is defined as the wrist and fingertip "
    "motion of the actor's hands, mapped to the right and left wrist-pose and "
    "fingertip components."
)


def _server_args(config=None):
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
        batching_max_size=1,
        pipeline_config=config or Cosmos3Config(),
    )


def _write_stats(directory: str, name: str, blocks: dict) -> str:
    path = Path(directory) / name
    path.write_text(
        json.dumps(
            {
                key: {k: np.asarray(v, dtype=np.float32).tolist() for k, v in b.items()}
                for key, b in blocks.items()
            }
        )
    )
    return str(path)


class TestHumanDomainTable(unittest.TestCase):
    def test_human_domain_ids_and_widths(self):
        self.assertEqual(EMBODIMENT_TO_DOMAIN_ID["webhumanaction_hand"], 31)
        self.assertEqual(EMBODIMENT_TO_DOMAIN_ID["webhumanaction_body"], 24)
        self.assertEqual(EMBODIMENT_TO_DOMAIN_ID["hand_pose"], 3)
        self.assertEqual(EMBODIMENT_TO_RAW_ACTION_DIM["webhumanaction_hand"], 48)
        self.assertEqual(EMBODIMENT_TO_RAW_ACTION_DIM["webhumanaction_body"], 57)
        # hand_pose (Mecka / VITRA) has a per-dataset width.
        self.assertNotIn("hand_pose", EMBODIMENT_TO_RAW_ACTION_DIM)

    def test_raw_action_dim_resolution(self):
        self.assertEqual(resolve_raw_action_dim("webhumanaction_hand", None), 48)
        self.assertEqual(resolve_raw_action_dim(" WebHumanAction_Body ", 57), 57)
        with self.assertRaisesRegex(ValueError, "does not match"):
            resolve_raw_action_dim("webhumanaction_hand", 57)
        self.assertIsNone(resolve_raw_action_dim("hand_pose", None))
        self.assertEqual(resolve_raw_action_dim("hand_pose", 57), 57)
        self.assertEqual(resolve_raw_action_dim(None, 12), 12)

    def test_every_domain_with_a_width_has_an_id(self):
        self.assertTrue(
            set(EMBODIMENT_TO_RAW_ACTION_DIM) <= set(EMBODIMENT_TO_DOMAIN_ID)
        )


class TestActionLayout(unittest.TestCase):
    def test_inverse_dynamics_default_is_one_row_per_transition(self):
        layout = action_layout("inverse_dynamics", 73)
        self.assertEqual(layout.num_rows, 72)
        self.assertEqual(layout.condition_rows, ())
        self.assertEqual(layout.start_frame_offset, 1)

    def test_image2action_adds_frame0_row_on_frame0(self):
        layout = action_layout("inverse_dynamics", 73, predict_initial_state=True)
        self.assertEqual(layout.num_rows, 73)
        self.assertEqual(layout.condition_rows, ())
        self.assertEqual(layout.start_frame_offset, 0)

    def test_forward_dynamics_conditions_every_row(self):
        layout = action_layout("forward_dynamics", 17)
        self.assertEqual(layout.condition_rows, tuple(range(16)))

    def test_initial_state_rejected_for_policy_and_forward_dynamics(self):
        for mode in ("policy", "forward_dynamics"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                action_layout(mode, 17, predict_initial_state=True)

    def test_single_frame_keeps_one_row_on_frame0(self):
        self.assertEqual(action_layout("policy", 1), (1, (), 0))


class TestActionNormalizers(unittest.TestCase):
    STATS = {
        "q01": torch.tensor([-1.0, 0.0, 2.0]),
        "q99": torch.tensor([1.0, 4.0, 2.0]),
        "min": torch.tensor([-5.0, -1.0, 1.0]),
        "max": torch.tensor([5.0, 9.0, 3.0]),
        "mean": torch.tensor([0.0, 2.0, 2.0]),
        "std": torch.tensor([1.0, 0.5, 0.0]),
    }

    def test_quantile_matches_range_formula_and_clamps(self):
        x = torch.tensor([[0.5, 3.0, 2.0], [3.0, -2.0, 2.0]])
        out = normalize_action(x, "quantile", self.STATS)
        q01, q99 = self.STATS["q01"], self.STATS["q99"]
        legacy = (2.0 * (x - q01) / (q99 - q01).clamp(min=1e-8) - 1.0).clamp(-1, 1)
        torch.testing.assert_close(out[:, :2], legacy[:, :2])
        self.assertTrue(bool((out.abs() <= 1.0).all()))

    def test_piecewise_asinh_is_linear_inside_and_compresses_tails(self):
        normalizer = resolve_action_normalizer("piecewise_asinh_rot", self.STATS)
        x = torch.tensor([[0.5, 2.0, 2.0], [7.0, -6.0, 2.0]])
        z = (x - normalizer.offset) / normalizer.scale
        y = normalizer.normalize(x)
        torch.testing.assert_close(y[0, :2], z[0, :2])
        expected_tail = torch.sign(z[1, :2]) * (1.0 + torch.asinh(z[1, :2].abs() - 1))
        torch.testing.assert_close(y[1, :2], expected_tail)
        torch.testing.assert_close(normalizer.denormalize(y), x)

    def test_unclamped_methods_round_trip(self):
        x = torch.tensor([[0.3, 7.5], [-4.0, -2.0]], dtype=torch.float64)
        for method in ("asinh_rot", "piecewise_asinh_rot", "meanstd"):
            with self.subTest(method=method):
                normalizer = resolve_action_normalizer(method, self.STATS)
                normalizer = normalizer._replace(
                    offset=normalizer.offset[:2], scale=normalizer.scale[:2]
                )
                torch.testing.assert_close(
                    normalizer.denormalize(normalizer.normalize(x)), x
                )

    def test_asinh_unit_keeps_quantile_bounds_at_one(self):
        stats = {"q01": torch.tensor([-1.0, 0.0]), "q99": torch.tensor([1.0, 4.0])}
        torch.testing.assert_close(
            normalize_action(stats["q99"], "asinh_rot", stats), torch.ones(2)
        )
        y = normalize_action(torch.tensor([3.0, 2.0]), "asinh_rot", stats)
        self.assertAlmostEqual(float(y[0]), math.asinh(3.0) / math.asinh(1.0), places=5)

    def test_scale_floor_widens_heavy_tailed_channels(self):
        normalizer = resolve_action_normalizer("quantile_rot_scale_floor", self.STATS)
        # max(|min - offset|, |max - offset|) / 5 is 1.0 and 1.4, below the
        # quantile half-ranges 1.0 and 2.0, so the scale is unchanged.
        torch.testing.assert_close(normalizer.scale[:2], torch.tensor([1.0, 2.0]))
        stats = dict(self.STATS, max=torch.tensor([50.0, 9.0, 3.0]))
        normalizer = resolve_action_normalizer("quantile_rot_scale_floor", stats)
        self.assertAlmostEqual(float(normalizer.scale[0]), 10.0, places=5)

    def test_global_asinh_requires_unified_width(self):
        with self.assertRaisesRegex(ValueError, "59"):
            resolve_action_normalizer("global_asinh", self.STATS)
        stats = {"q01": -torch.ones(59), "q99": torch.ones(59)}
        y = normalize_action(torch.full((2, 59), 3.0), "global_asinh", stats)
        torch.testing.assert_close(
            y, torch.full((2, 59), math.asinh(3.0) / math.asinh(1.0))
        )

    def test_unknown_method_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown action normalization"):
            resolve_action_normalizer("global_quantile", self.STATS)

    def test_stats_block_follows_method(self):
        self.assertEqual(default_action_stats_key("piecewise_asinh_rot"), "global_raw")
        self.assertEqual(default_action_stats_key("quantile_rot"), "global_raw")
        self.assertEqual(default_action_stats_key("quantile"), "global")
        self.assertEqual(default_action_stats_key("global_asinh"), "global")
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_stats(
                tmp,
                "stats.json",
                {
                    "global": {"q01": [-1.0], "q99": [1.0]},
                    "global_raw": {"q01": [-2.0], "q99": [2.0]},
                },
            )
            self.assertEqual(
                float(load_action_normalizer("quantile", path).scale[0]), 1.0
            )
            self.assertEqual(
                float(load_action_normalizer("piecewise_asinh_rot", path).scale[0]),
                2.0,
            )
            flat = Path(tmp) / "flat.json"
            flat.write_text(json.dumps({"q01": [-3.0], "q99": [3.0]}))
            self.assertEqual(
                float(load_action_normalizer("quantile_rot", str(flat)).scale[0]), 3.0
            )
            empty = Path(tmp) / "empty.json"
            empty.write_text(json.dumps({"metadata": {}}))
            with self.assertRaises(KeyError):
                load_action_stats(str(empty), "global_raw")

    def test_initial_state_row_uses_its_own_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = _write_stats(
                tmp,
                "base.json",
                {"global_raw": {"q01": [-1.0, -1.0], "q99": [1.0, 1.0]}},
            )
            a0 = _write_stats(
                tmp, "a0.json", {"global_raw": {"q01": [9.0, 9.0], "q99": [11.0, 11.0]}}
            )
            pred = torch.zeros(1, 4, 2)
            out = denormalize_action_rows(
                pred,
                method="piecewise_asinh_rot",
                stats_path=base,
                initial_state_stats_path=a0,
                has_initial_state=True,
            )
            torch.testing.assert_close(out[0, 0], torch.tensor([10.0, 10.0]))
            torch.testing.assert_close(out[0, 1:], torch.zeros(3, 2))
            plain = denormalize_action_rows(
                pred,
                method="piecewise_asinh_rot",
                stats_path=base,
                initial_state_stats_path=a0,
                has_initial_state=False,
            )
            torch.testing.assert_close(plain, torch.zeros(1, 4, 2))
            with self.assertRaisesRegex(ValueError, "initial_state"):
                denormalize_action_rows(
                    pred,
                    method="piecewise_asinh_rot",
                    stats_path=base,
                    initial_state_stats_path=None,
                    has_initial_state=True,
                )
        # Without stats the actions stay in model space.
        self.assertIs(
            denormalize_action_rows(
                pred,
                method="quantile",
                stats_path=None,
                initial_state_stats_path=None,
                has_initial_state=True,
            ),
            pred,
        )

    def test_per_domain_stats_selection(self):
        self.assertEqual(select_action_stats_path("a.json", "hand_pose"), "a.json")
        self.assertIsNone(select_action_stats_path(None, "hand_pose"))
        mapping = {"WebHumanAction_Hand": "hand.json", "hand_pose": "mecka.json"}
        self.assertEqual(
            select_action_stats_path(mapping, "webhumanaction_hand"), "hand.json"
        )
        with self.assertRaisesRegex(ValueError, "No action stats"):
            select_action_stats_path(mapping, "webhumanaction_body")


class TestActionCaptions(unittest.TestCase):
    def test_semantic_caption_matches_lance_training_caption(self):
        caption = build_plain_action_caption(
            "The person folds a towel.",
            domain_name="webhumanaction_hand",
            height=480,
            width=832,
            append_semantics=True,
        )
        self.assertEqual(
            caption,
            "The person folds a towel. "
            + HAND_CAPTION_POSTFIX
            + " This video is of 480x832 resolution.",
        )

    def test_caption_without_terminal_punctuation_gets_a_period(self):
        caption = build_plain_action_caption(
            "fold the towel  ",
            domain_name="hand_pose",
            height=480,
            width=832,
            append_semantics=True,
        )
        self.assertTrue(caption.startswith("fold the towel. The video is captured"))
        self.assertIn("head-mounted camera", caption)

    def test_plain_caption_is_verbatim_without_semantics(self):
        self.assertEqual(
            build_plain_action_caption(
                "fold", domain_name=None, height=1, width=1, append_semantics=False
            ),
            "fold",
        )
        self.assertEqual(
            build_plain_action_caption(
                "", domain_name="hand_pose", height=1, width=1, append_semantics=True
            ),
            "",
        )
        with self.assertRaisesRegex(ValueError, "semantics"):
            build_plain_action_caption(
                "x",
                domain_name="droid_lerobot",
                height=1,
                width=1,
                append_semantics=True,
            )

    def test_mode_specific_prompt(self):
        self.assertEqual(
            add_action_mode_metadata("fold the towel", "inverse_dynamics"),
            "Predict the underlying action given the video.",
        )
        self.assertEqual(
            add_action_mode_metadata("fold the towel", "policy"),
            "fold the towel. Predict future action given the image.",
        )
        self.assertEqual(
            add_action_mode_metadata("", "policy"),
            "Predict future action given the image.",
        )
        caption = render_action_prompt(
            "fold the towel",
            format_prompt_as_json=False,
            append_semantics=True,
            mode_prompt="inverse_dynamics",
            domain_name="webhumanaction_hand",
            view_point="ego_view",
            num_frames=17,
            fps=15,
            height=480,
            width=832,
        )
        self.assertEqual(
            caption,
            "Predict the underlying action given the video. "
            + HAND_CAPTION_POSTFIX
            + " This video is of 480x832 resolution.",
        )

    def test_json_prompt_unchanged_and_rejects_mode_prompt(self):
        prompt = render_action_prompt(
            "fold",
            format_prompt_as_json=True,
            append_semantics=False,
            mode_prompt=None,
            domain_name="hand_pose",
            view_point="ego_view",
            num_frames=17,
            fps=15,
            height=480,
            width=832,
        )
        self.assertEqual(json.loads(prompt)["resolution"], {"H": 480, "W": 832})
        with self.assertRaises(ValueError):
            render_action_prompt(
                "fold",
                format_prompt_as_json=True,
                append_semantics=False,
                mode_prompt="inverse_dynamics",
                domain_name=None,
                view_point="ego_view",
                num_frames=17,
                fps=15,
                height=480,
                width=832,
            )


class TestActionInputFitting(unittest.TestCase):
    def test_bucket_tables(self):
        self.assertEqual(list(ASPECT_RATIO_SIZE_INFO)[:4], ["256", "272", "544", "480"])
        self.assertEqual(ASPECT_RATIO_SIZE_INFO["480"]["2,3"], (512, 768))
        # The shared canvas table other Cosmos3 paths use stays unchanged.
        self.assertNotIn("2,3", VIDEO_RES_SIZE_INFO["480"])
        self.assertEqual(action_bucket_size(720, 1280, "480"), (832, 480))
        self.assertEqual(action_bucket_size(1080, 720, 480), (512, 768))
        with self.assertRaises(ValueError):
            action_bucket_size(720, 1280, "999")

    def test_downscale_pads_bottom_and_crops_latent(self):
        frames = torch.randint(0, 256, (3, 3, 720, 1280), dtype=torch.uint8)
        pixels, fit = fit_action_video(frames, resolution="480", num_frames=5)
        self.assertEqual(tuple(pixels.shape), (3, 5, 480, 832))
        self.assertEqual(fit.bucket_hw, (480, 832))
        self.assertEqual(fit.content_hw, (468, 832))
        self.assertEqual(fit.latent_hw, (29, 52))
        self.assertEqual(fit.generation_hw, (464, 832))
        self.assertGreaterEqual(float(pixels.min()), -1.0)
        self.assertLessEqual(float(pixels.max()), 1.0)
        # Short clips repeat the last frame; padding mirrors the content rows.
        torch.testing.assert_close(pixels[:, 3], pixels[:, 2])
        torch.testing.assert_close(pixels[:, :, 468], pixels[:, :, 466])

    def test_small_input_is_not_upscaled_and_edge_padded(self):
        frames = torch.zeros(3, 2, 240, 320, dtype=torch.uint8)
        frames[:, :, :, -1] = 255
        pixels, fit = fit_action_video(frames, resolution="480", num_frames=2)
        self.assertEqual(fit.bucket_hw, (544, 736))
        self.assertEqual(fit.content_hw, (240, 320))
        self.assertEqual(fit.latent_hw, (15, 20))
        # Padding wider than the content falls back to edge replication.
        self.assertTrue(bool((pixels[:, :, :240, 320:] == 1.0).all()))

    def test_long_clip_is_truncated(self):
        frames = torch.zeros(3, 9, 64, 64, dtype=torch.uint8)
        pixels, _ = fit_action_video(frames, resolution="256", num_frames=5)
        self.assertEqual(pixels.shape[1], 5)

    def test_rejects_float_frames(self):
        with self.assertRaises(ValueError):
            fit_action_video(torch.zeros(3, 1, 8, 8), resolution="256", num_frames=1)


class TestUnifiedV1(unittest.TestCase):
    def test_human_slot_masks(self):
        hand = unified_action_valid_mask("webhumanaction_hand")
        self.assertEqual(int(hand.sum()), 48)
        self.assertFalse(bool(hand[:9].any()))
        self.assertFalse(bool(hand[33]) or bool(hand[58]))
        body = unified_action_valid_mask("webhumanaction_body")
        self.assertEqual(int(body.sum()), 57)
        torch.testing.assert_close(body, unified_action_valid_mask("hand_pose"))
        with self.assertRaises(ValueError):
            unified_action_valid_mask("pusht")

    def test_robot_slot_masks(self):
        single = unified_action_valid_mask("droid_lerobot")
        self.assertEqual(torch.nonzero(single).flatten().tolist(), [*range(9, 18), 33])
        dual = unified_action_valid_mask("xdof_yam")
        self.assertEqual(
            torch.nonzero(dual).flatten().tolist(),
            [*range(9, 18), 33, *range(34, 43), 58],
        )
        humanoid = unified_action_valid_mask("agibotworld")
        torch.testing.assert_close(humanoid[9:], dual[9:])
        self.assertTrue(bool(humanoid[:9].all()))
        self.assertEqual(int(unified_action_valid_mask("av").sum()), 9)

    def test_native_round_trip(self):
        for domain, width in (("webhumanaction_hand", 48), ("hand_pose", 57)):
            with self.subTest(domain=domain):
                mask = unified_action_valid_mask(domain)
                unified = torch.zeros(2, 59)
                unified[:, mask] = torch.arange(width, dtype=torch.float32)
                native = unified_to_native_action(unified, domain)
                torch.testing.assert_close(
                    native, torch.arange(width, dtype=torch.float32).expand(2, -1)
                )


class TestInverseDynamicsSamplingParams(unittest.TestCase):
    def _adjust(self, config=None, **kwargs):
        sp = Cosmos3SamplingParams(prompt="t", num_frames=17, **kwargs)
        sp._adjust(_server_args(config))
        return sp

    def test_initial_state_only_for_inverse_dynamics(self):
        sp = self._adjust(
            action_mode="inverse_dynamics",
            domain_name="webhumanaction_hand",
            predict_initial_state=True,
        )
        self.assertTrue(sp.predict_initial_state)
        with self.assertRaisesRegex(ValueError, "predict_initial_state"):
            self._adjust(
                action_mode="policy",
                domain_name="droid_lerobot",
                predict_initial_state=True,
            )
        with self.assertRaisesRegex(ValueError, "require an action_mode"):
            self._adjust(predict_initial_state=True)
        config = Cosmos3Config()
        config.action_stats_path = "/stats/delta.json"
        with self.assertRaisesRegex(ValueError, "action_initial_state_stats_path"):
            self._adjust(
                config,
                action_mode="inverse_dynamics",
                domain_name="webhumanaction_hand",
                predict_initial_state=True,
            )

    def test_action_resolution_validation(self):
        sp = self._adjust(
            action_mode="inverse_dynamics",
            domain_name="hand_pose",
            raw_action_dim=57,
            action_resolution=480,
        )
        self.assertEqual(sp.action_resolution, "480")
        with self.assertRaisesRegex(ValueError, "Unknown action_resolution"):
            self._adjust(
                action_mode="inverse_dynamics",
                domain_name="av",
                action_resolution="999",
            )
        sp = self._adjust(
            action_mode="policy", domain_name="droid_lerobot", action_resolution="256"
        )
        self.assertEqual(sp.action_resolution, "256")

    def test_plain_caption_flags_need_plain_captions(self):
        with self.assertRaisesRegex(ValueError, "plain captions"):
            self._adjust(
                action_mode="inverse_dynamics",
                domain_name="hand_pose",
                action_caption_semantics=True,
            )

    def test_unified_schema_and_per_domain_stats_come_from_server(self):
        config = Cosmos3Config()
        config.action_schema = "unified_v1"
        config.action_stats_path = {"webhumanaction_hand": "/stats/global.json"}
        config.action_initial_state_stats_path = {
            "webhumanaction_hand": "/stats/a0.json"
        }
        sp = self._adjust(
            config, action_mode="inverse_dynamics", domain_name="webhumanaction_hand"
        )
        self.assertEqual(sp.action_schema, "unified_v1")
        self.assertEqual(sp.action_stats_path, "/stats/global.json")
        self.assertEqual(sp.action_initial_state_stats_path, "/stats/a0.json")
        with self.assertRaisesRegex(ValueError, "known slots"):
            self._adjust(config, action_mode="inverse_dynamics", domain_name="pusht")
        with self.assertRaisesRegex(ValueError, "forward dynamics"):
            self._adjust(
                config,
                action_mode="forward_dynamics",
                domain_name="webhumanaction_hand",
                action=[[0.0] * 59] * 16,
            )
        with self.assertRaisesRegex(ValueError, "unified_v1 width"):
            self._adjust(
                config,
                action_mode="inverse_dynamics",
                domain_name="webhumanaction_hand",
                raw_action_dim=48,
            )

    def test_action_endpoint_passes_inverse_dynamics_fields(self):
        payload = {
            "input": {
                "task": "fold the towel",
                "observation": {"video": "hands.mp4"},
            },
            "parameters": {
                "action_mode": "inverse_dynamics",
                "num_frames": 73,
                "fps": 15,
                "domain_name": "webhumanaction_hand",
                "predict_initial_state": True,
                "format_prompt_as_json": False,
                "action_caption_semantics": True,
                "enable_mode_specific_prompt": True,
                "action_resolution": "480",
            },
        }
        params = build_action_sampling_params(payload, _server_args())
        self.assertEqual(params.data_type, DataType.ACTION)
        self.assertTrue(params.predict_initial_state)
        self.assertFalse(params.format_prompt_as_json)
        self.assertTrue(params.action_caption_semantics)
        self.assertTrue(params.enable_mode_specific_prompt)
        self.assertEqual(params.action_resolution, "480")

    def test_http_lowering_parses_new_fields(self):
        request = types.SimpleNamespace(model_extra={}, num_frames=None, fps=None)
        kwargs = Cosmos3SamplingParams.lower_video_request_kwargs(
            request,
            {
                "num_frames": 17,
                "fps": 15,
                "predict_initial_state": "true",
                "format_prompt_as_json": "false",
                "action_resolution": "480",
            },
        )
        self.assertIs(kwargs["predict_initial_state"], True)
        self.assertIs(kwargs["format_prompt_as_json"], False)
        self.assertEqual(kwargs["action_resolution"], 480)
        for field in (
            "predict_initial_state",
            "format_prompt_as_json",
            "action_caption_semantics",
            "enable_mode_specific_prompt",
            "action_resolution",
        ):
            self.assertIn(field, Cosmos3SamplingParams.video_request_extra_fields())
        self.assertNotIn(
            "action_initial_state_stats_path",
            Cosmos3SamplingParams.video_request_extra_fields(),
        )
        self.assertNotIn(
            "action_schema", Cosmos3SamplingParams.video_request_extra_fields()
        )


class TestInverseDynamicsStages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage = Cosmos3LatentPreparationStage.__new__(Cosmos3LatentPreparationStage)
        cls.stage.transformer = types.SimpleNamespace(action_dim=64)
        cls.stage.log_info = lambda *a, **k: None

    def _prepare(self, num_frames=17, **sp_kwargs):
        sp = Cosmos3SamplingParams(prompt="t", num_frames=num_frames, **sp_kwargs)
        batch = types.SimpleNamespace(
            sampling_params=sp,
            num_frames=num_frames,
            raw_latent_shape=(1, 48, 1, 1, 1),
            extra={},
        )
        gen = torch.Generator().manual_seed(0)
        self.stage._prepare_action_latents(
            batch, gen, torch.device("cpu"), torch.float32
        )
        return batch

    def test_image2action_latents(self):
        batch = self._prepare(
            num_frames=73,
            action_mode="inverse_dynamics",
            domain_name="webhumanaction_hand",
            predict_initial_state=True,
        )
        self.assertEqual(tuple(batch.action_latents.shape), (1, 73, 64))
        self.assertEqual(batch.extra["action_start_frame_offset"], 0)
        self.assertTrue(batch.extra["action_has_initial_state"])
        self.assertEqual(batch.extra["raw_action_dim"], 48)
        self.assertEqual(int(batch.extra["action_domain_ids"][0]), 31)
        self.assertTrue(torch.all(batch.extra["action_velocity_mask"] == 1))
        self.assertTrue(torch.all(batch.action_latents[:, :, 48:] == 0))
        self.assertIsNone(batch.extra["action_valid_mask"])

    def test_default_inverse_dynamics_keeps_offset_one(self):
        batch = self._prepare(
            action_mode="inverse_dynamics", domain_name="webhumanaction_body"
        )
        self.assertEqual(tuple(batch.action_latents.shape), (1, 16, 64))
        self.assertEqual(batch.extra["action_start_frame_offset"], 1)
        self.assertFalse(batch.extra["action_has_initial_state"])
        self.assertEqual(int(batch.extra["action_domain_ids"][0]), 24)

    def test_hand_pose_needs_explicit_width(self):
        with self.assertRaisesRegex(ValueError, "raw-action-dim"):
            self._prepare(action_mode="inverse_dynamics", domain_name="hand_pose")
        batch = self._prepare(
            action_mode="inverse_dynamics", domain_name="hand_pose", raw_action_dim=57
        )
        self.assertEqual(batch.extra["raw_action_dim"], 57)

    def test_unified_v1_masks_noise_per_slot(self):
        batch = self._prepare(
            action_mode="inverse_dynamics",
            domain_name="webhumanaction_hand",
            action_schema="unified_v1",
        )
        mask = batch.extra["action_valid_mask"]
        self.assertEqual(batch.extra["raw_action_dim"], 59)
        self.assertEqual(int(mask.sum()), 48)
        self.assertTrue(torch.all(batch.action_latents[..., ~mask] == 0))
        self.assertTrue(bool((batch.action_latents[..., mask] != 0).any()))

    def _decode(self, action_latents, extra, **sp_kwargs):
        stage = Cosmos3DecodingStage.__new__(Cosmos3DecodingStage)
        stage.log_info = lambda *a, **k: None
        batch = types.SimpleNamespace(
            data_type=DataType.ACTION,
            action_latents=action_latents,
            extra={"action_domain_ids": torch.tensor([31]), **extra},
            sampling_params=Cosmos3SamplingParams(prompt="t", **sp_kwargs),
            request_id="id",
            num_inference_steps=1,
            num_frames=action_latents.shape[1],
            metrics=None,
        )
        return stage.forward(batch, types.SimpleNamespace(vae_cpu_offload=False))

    def test_decode_denormalizes_initial_state_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = _write_stats(
                tmp, "base.json", {"global_raw": {"q01": [-1.0] * 2, "q99": [1.0] * 2}}
            )
            a0 = _write_stats(
                tmp, "a0.json", {"global_raw": {"q01": [4.0] * 2, "q99": [6.0] * 2}}
            )
            output = self._decode(
                torch.zeros(1, 3, 4),
                {"raw_action_dim": 2, "action_has_initial_state": True},
                action_mode="inverse_dynamics",
                action_normalization="piecewise_asinh_rot",
                action_stats_path=base,
                action_initial_state_stats_path=a0,
            )
        actions = output.output[0]["actions"]
        self.assertEqual(actions.shape, (3, 2))
        np.testing.assert_allclose(actions[0], [5.0, 5.0])
        np.testing.assert_allclose(actions[1:], np.zeros((2, 2)))

    def test_decode_zeroes_unfilled_unified_slots(self):
        mask = torch.zeros(64, dtype=torch.bool)
        mask[:59] = unified_action_valid_mask("webhumanaction_hand")
        with tempfile.TemporaryDirectory() as tmp:
            stats = _write_stats(
                tmp, "global.json", {"global": {"q01": [-1.0] * 59, "q99": [1.0] * 59}}
            )
            output = self._decode(
                torch.full((1, 2, 64), 0.5),
                {"raw_action_dim": 59, "action_valid_mask": mask},
                action_mode="inverse_dynamics",
                action_normalization="global_asinh",
                action_stats_path=stats,
            )
        actions = torch.as_tensor(output.output[0]["actions"])
        self.assertEqual(tuple(actions.shape), (2, 59))
        self.assertTrue(torch.all(actions[:, ~mask[:59]] == 0))
        torch.testing.assert_close(
            actions[:, mask[:59]],
            torch.full((2, 48), math.sinh(0.5 * math.asinh(1.0))),
        )

    def test_preprocess_bucket_fits_inverse_dynamics_video(self):
        stage = Cosmos3ImagePreprocessStage.__new__(Cosmos3ImagePreprocessStage)
        stage.log_info = lambda *a, **k: None
        frames = [np.full((720, 1280, 3), 100, dtype=np.uint8) for _ in range(3)]
        batch = types.SimpleNamespace(
            image_path=None,
            video_path="hands.mp4",
            data_type=DataType.ACTION,
            height=480,
            width=832,
            num_frames=17,
            extra={},
            sampling_params=Cosmos3SamplingParams(
                prompt="t",
                num_frames=17,
                action_mode="inverse_dynamics",
                action_resolution="480",
            ),
            preprocessed_video=None,
        )
        with mock.patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages."
            "model_specific_stages.cosmos3._iter_video_frames_pyav",
            side_effect=lambda *_: (f for f in frames),
        ):
            stage.forward(batch, types.SimpleNamespace())
        self.assertEqual(tuple(batch.preprocessed_video.shape), (1, 3, 17, 480, 832))
        self.assertEqual((batch.height, batch.width), (464, 832))
        self.assertEqual(batch.extra["action_caption_hw"], (480, 832))
        self.assertEqual(batch.extra["action_latent_crop_hw"], (29, 52))

    def test_preprocess_bucket_fits_policy_images(self):
        stage = Cosmos3ImagePreprocessStage.__new__(Cosmos3ImagePreprocessStage)
        stage.log_info = lambda *a, **k: None

        def run(sizes):
            batch = types.SimpleNamespace(
                image_path=["a.png", "b.png"],
                video_path=None,
                data_type=DataType.ACTION,
                height=480,
                width=832,
                extra={},
                sampling_params=Cosmos3SamplingParams(
                    prompt="t", action_mode="policy", action_resolution="480"
                ),
                preprocessed_image=None,
            )
            images = iter(Image.new("RGB", size, (200, 10, 10)) for size in sizes)
            with mock.patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages."
                "model_specific_stages.cosmos3.load_image",
                side_effect=lambda _p: next(images),
            ):
                stage.forward(batch, types.SimpleNamespace())
            return batch

        batch = run([(1280, 720), (1280, 720)])
        self.assertEqual(tuple(batch.preprocessed_image.shape), (2, 3, 480, 832))
        self.assertEqual((batch.height, batch.width), (464, 832))
        self.assertEqual(batch.extra["action_latent_crop_hw"], (29, 52))
        with self.assertRaisesRegex(ValueError, "same resolution bucket"):
            run([(1280, 720), (640, 480)])

    def test_tokenization_uses_bucket_size_in_plain_caption(self):
        captured = []

        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 1

            @staticmethod
            def convert_tokens_to_ids(_token):
                return 2

            @staticmethod
            def apply_chat_template(conversations, **_kwargs):
                captured.append(conversations[-1]["content"])
                return [3, 4, 5]

        stage = Cosmos3TokenizationStage.__new__(Cosmos3TokenizationStage)
        stage.tokenizer = Tokenizer()
        stage.log_info = lambda *a, **k: None
        batch = types.SimpleNamespace(
            prompt="fold the towel",
            negative_prompt="",
            height=464,
            width=832,
            num_frames=17,
            fps=15,
            data_type=DataType.ACTION,
            extra={"action_caption_hw": (480, 832)},
            sampling_params=Cosmos3SamplingParams(
                prompt="fold the towel",
                action_mode="inverse_dynamics",
                domain_name="webhumanaction_hand",
                format_prompt_as_json=False,
                action_caption_semantics=True,
                enable_mode_specific_prompt=True,
            ),
        )
        stage.forward(
            batch, types.SimpleNamespace(pipeline_config=types.SimpleNamespace())
        )
        self.assertEqual(
            captured[0],
            "Predict the underlying action given the video. "
            + HAND_CAPTION_POSTFIX
            + " This video is of 480x832 resolution.",
        )


if __name__ == "__main__":
    unittest.main()
