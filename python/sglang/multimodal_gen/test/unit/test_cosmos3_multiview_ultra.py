# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Cosmos3 multiview on the Cosmos3.5 Ultra (Qwen3-VL-MoE) backbone.

Covers i4's attention contract (general cross-view window, first-frame
visibility), the i4-contract deployment block, hint-less (uncontrolled)
multiview requests, per-view caption formatting, layout-ordered stream packing,
and the Ulysses split of the Ultra's four key/value heads. Parity against the
imaginaire4 reference lives in the WS4 parity suite, which imports both trees.
"""

import copy
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.multimodal_gen.configs.pipeline_configs.cosmos3_multiview import (
    COSMOS3_MULTIVIEW_I4_INFERENCE_DEFAULTS,
    MULTIVIEW_SCHEMA_I4,
    Cosmos3MultiviewConfig,
    multiview_attention_heads,
    parse_multiview_deployment_config,
)
from sglang.multimodal_gen.configs.sample.cosmos3_multiview import (
    Cosmos3MultiviewSamplingParams,
    validate_multiview_request,
)
from sglang.multimodal_gen.runtime.models.dits import cosmos3_multiview as mv_dit
from sglang.multimodal_gen.runtime.models.dits.cosmos3_multiview import (
    Cosmos3MultiviewCrossAttention,
    Cosmos3MultiviewTransformer,
    multiview_kv_head_repeats,
    validate_multiview_ulysses_degree,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_simulation.multiview_attention import (
    _make_pair_allowed,
    build_multiview_flex_metadata,
    padded_multiview_flex_attention,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_simulation.multiview_layout import (
    MaskItem,
    MultiviewAttentionContext,
    MultiviewLayout,
    resolve_temporal_window_bounds,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_simulation.lidar_outputs import (
    lidar_range_preview_video,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_simulation.multiview import (
    format_per_view_prompts,
    format_view_prompt,
    multiview_condition_indexes,
    multiview_layout_for,
    multiview_mask_items,
    uses_mads_rig_header,
)
from sglang.multimodal_gen.test.unit.test_cosmos3_multiview import LIDAR_BLOCK

ULTRA_BACKBONE = "cosmos3_5_qwen_3_vl_moe"
# The cross-view contract of the Ultra v1p2..v1p6 midtrain recipes
# (cosmos3p5_ga_470bm235b_v1_midtrain.py): decomposed scope, (-0.4, 0) window,
# first frame visible.
ULTRA_BLOCK = {
    "cross_view_temporal_window_bounds": [-0.4, 0.0],
    "cross_view_attends_first_frame": True,
}


def _ultra_config(block=None):
    return {
        "backbone_type": ULTRA_BACKBONE,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "multiview": copy.deepcopy(ULTRA_BLOCK if block is None else block),
    }


def _metadata(items, *, num_und=2, **kwargs):
    offsets = [num_und]
    for item in items:
        offsets.append(offsets[-1] + item.num_tokens)
    return build_multiview_flex_metadata(
        seq_len=offsets[-1],
        full_q_offsets=tuple(offsets),
        items_per_sample=items,
        device="cpu",
        num_und=num_und,
        **kwargs,
    )


def multiview_pair_predicate(metadata, q_index, kv_index):
    """The exact token visibility predicate the block masks are built from."""
    pair_allowed = _make_pair_allowed(
        metadata.query_vectors(),
        metadata.key_vectors(),
        metadata.attention_scope,
        metadata.temporal_window_bounds,
        metadata.control_attends_sensor,
        metadata.lidar_attends_captions,
        metadata.includes_first_frame,
    )
    return pair_allowed(q_index, kv_index)


def _dense(metadata):
    return multiview_pair_predicate(
        metadata,
        torch.arange(metadata.q_len)[:, None],
        torch.arange(metadata.kv_len)[None, :],
    )


class TestCrossViewWindow(unittest.TestCase):
    """i4 ``_multiview_pair_predicate``: window bounds and the first-frame anchor."""

    CAMERA = MaskItem((6, 1, 1), 2, seconds_per_frame=0.4)  # 2 views x 3 frames
    CONTROL = MaskItem((6, 1, 1), 2, is_control=True, seconds_per_frame=0.4)

    def test_past_only_seconds_equal_bounds(self):
        for items in ((self.CAMERA,), (self.CONTROL, self.CAMERA)):
            for window in (0.0, 0.4, 0.8):
                with self.subTest(items=len(items), window=window):
                    past = _dense(
                        _metadata(
                            items,
                            decomposed_temporal_window_seconds=window,
                            control_attends_sensor=True,
                        )
                    )
                    bounds = _dense(
                        _metadata(
                            items,
                            temporal_window_bounds=(-window, 0.0),
                            control_attends_sensor=True,
                        )
                    )
                    self.assertTrue(torch.equal(past, bounds))

    def test_symmetric_window_admits_future_keys(self):
        # Query: view 0 frame 1 (t=0.4); keys of view 1 at t=0.0, 0.4, 0.8.
        query, keys = 1, [2 + 3, 2 + 4, 2 + 5]
        past = _metadata((self.CAMERA,), temporal_window_bounds=(-0.4, 0.0))
        symmetric = _metadata((self.CAMERA,), temporal_window_bounds=(-0.4, 0.4))
        narrow = _metadata((self.CAMERA,), temporal_window_bounds=(-0.39, 0.39))
        for metadata, expected in (
            (past, [True, True, False]),
            (symmetric, [True, True, True]),
            (narrow, [False, True, False]),
        ):
            self.assertEqual(
                multiview_pair_predicate(
                    metadata, torch.tensor(query), torch.tensor(keys)
                ).tolist(),
                expected,
            )

    def test_first_frame_reaches_every_sensor_query(self):
        """attends_first_frame: frame 0 of the other views' sensor streams, whatever
        the window, never their control and never padding."""
        items = (self.CONTROL, self.CAMERA)
        plain = _dense(_metadata(items, temporal_window_bounds=(-0.4, 0.0)))
        anchored = _dense(
            _metadata(
                items, temporal_window_bounds=(-0.4, 0.0), includes_first_frame=True
            )
        )
        # Query: target view 0 frame 2 (t=0.8); view 1 frame 0 (t=0.0) is outside
        # the 0.4 s window, inside the anchor.
        target = 6
        query = target + 2
        first_other = 2 + target + 3
        self.assertFalse(bool(plain[query, first_other]))
        self.assertTrue(bool(anchored[query, first_other]))
        # Control keys of another view stay out; control queries are unaffected.
        self.assertFalse(bool(anchored[query, 2 + 3]))
        self.assertTrue(torch.equal(plain[:target], anchored[:target]))
        # The anchor only adds pairs.
        self.assertTrue(bool((anchored | plain).eq(anchored).all()))
        added = anchored & ~plain
        added_keys = added.nonzero()[:, 1] - 2
        self.assertTrue(bool((added_keys >= target).all()))
        self.assertEqual(sorted({int(k) % 3 for k in added_keys - target}), [0])

    def test_first_frame_needs_decomposed_window(self):
        with self.assertRaisesRegex(ValueError, "includes_first_frame"):
            _metadata((self.CAMERA,), includes_first_frame=True)
        with self.assertRaisesRegex(ValueError, "includes_first_frame"):
            _metadata(
                (self.CAMERA,),
                attention_scope="all_views",
                temporal_window_bounds=(-0.4, 0.0),
                includes_first_frame=True,
            )
        with self.assertRaisesRegex(ValueError, "includes_first_frame"):
            MultiviewLayout(
                num_views=2,
                latent_frames=6,
                patch_height=1,
                patch_width=1,
                includes_first_frame=True,
            )

    def test_window_spellings_are_exclusive_and_validated(self):
        self.assertEqual(resolve_temporal_window_bounds(0.4, None), (-0.4, 0.0))
        self.assertEqual(
            resolve_temporal_window_bounds(None, [-0.67, 0.67]), (-0.67, 0.67)
        )
        self.assertIsNone(resolve_temporal_window_bounds(None, None))
        with self.assertRaisesRegex(ValueError, "not both"):
            resolve_temporal_window_bounds(0.4, (-0.4, 0.0))
        with self.assertRaisesRegex(ValueError, "start <= end"):
            resolve_temporal_window_bounds(None, (0.4, -0.4))
        with self.assertRaisesRegex(ValueError, "two finite"):
            resolve_temporal_window_bounds(None, (0.0,))

    def test_layout_cache_key_tracks_the_contract(self):
        base = dict(num_views=2, latent_frames=6, patch_height=1, patch_width=1)
        keys = {
            MultiviewLayout(**base).cache_key(),
            MultiviewLayout(temporal_window_bounds=(-0.4, 0.0), **base).cache_key(),
            MultiviewLayout(
                temporal_window_bounds=(-0.4, 0.0), includes_first_frame=True, **base
            ).cache_key(),
            MultiviewLayout(decomposed_temporal_window_seconds=0.4, **base).cache_key(),
        }
        self.assertEqual(len(keys), 4)
        layout = MultiviewLayout(temporal_window_bounds=[-0.4, 0], **base)
        self.assertEqual(layout.temporal_window_bounds, (-0.4, 0.0))
        self.assertEqual(layout.window_bounds, (-0.4, 0.0))
        self.assertEqual(
            MultiviewLayout(
                decomposed_temporal_window_seconds=0.4, **base
            ).window_bounds,
            (-0.4, 0.0),
        )

    def test_hintless_layout_matches_dense_oracle(self):
        """Target cameras only, per-view captions, (-0.4, 0) window, first frame."""
        torch.manual_seed(0)
        camera = (6, 1, 2)
        layout = MultiviewLayout(
            num_views=2,
            latent_frames=6,
            patch_height=1,
            patch_width=2,
            temporal_window_bounds=(-0.4, 0.0),
            includes_first_frame=True,
            control_attends_sensor=True,
            seconds_per_frame=0.4,
            backend="triton",
            max_und_tokens=70,
            items=(MaskItem(camera, 2, seconds_per_frame=0.4),),
            caption_lengths=(3, 2),
        )
        heads, kv_heads, dim = 4, 1, 16
        gen, und = layout.gen_tokens, 5
        q = torch.randn(1, gen, heads, dim)
        k, v = torch.randn(2, 1, gen, kv_heads, dim)
        k_und, v_und = torch.randn(2, 1, und, kv_heads, dim)
        out = padded_multiview_flex_attention(
            q, k, v, k_und, v_und, MultiviewAttentionContext(layout, {}, {})
        )
        padded_und = 128
        offsets = (padded_und, padded_und + gen)
        metadata = build_multiview_flex_metadata(
            seq_len=padded_und + 64,
            full_q_offsets=offsets,
            items_per_sample=layout.items,
            device="cpu",
            num_und=und,
            temporal_window_bounds=(-0.4, 0.0),
            includes_first_frame=True,
            control_attends_sensor=True,
            caption_lengths=(3, 2),
        )
        keys = torch.cat([torch.arange(und), padded_und + torch.arange(gen)])
        mask = multiview_pair_predicate(
            metadata, torch.arange(gen)[:, None], keys[None]
        )
        scores = torch.einsum(
            "bqhd,bkhd->bhqk",
            q,
            torch.cat([k_und, k], dim=1).repeat_interleave(heads, dim=2),
        ) / (dim**0.5)
        probs = scores.masked_fill(~mask, float("-inf")).softmax(-1)
        expected = torch.einsum(
            "bhqk,bkhd->bqhd",
            probs,
            torch.cat([v_und, v], dim=1).repeat_interleave(heads, dim=2),
        )
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


class TestUlyssesKvHeads(unittest.TestCase):
    def test_repeat_counts_follow_i4(self):
        self.assertEqual(multiview_kv_head_repeats(4, 4), 1)
        self.assertEqual(multiview_kv_head_repeats(4, 2), 1)
        self.assertEqual(multiview_kv_head_repeats(4, 8), 2)
        self.assertEqual(multiview_kv_head_repeats(4, 16), 4)
        self.assertEqual(multiview_kv_head_repeats(8, 8), 1)
        with self.assertRaisesRegex(ValueError, "multiple"):
            multiview_kv_head_repeats(4, 6)
        validate_multiview_ulysses_degree(8, 64, 4)
        validate_multiview_ulysses_degree(4, 32, 4)
        with self.assertRaisesRegex(ValueError, "query heads"):
            validate_multiview_ulysses_degree(3, 64, 4)

    def _emulate(self, world_size, heads=8, kv_heads=2):
        """Run ``_forward_multiview`` once per simulated Ulysses rank."""
        torch.manual_seed(0)
        layout = MultiviewLayout(
            num_views=2,
            latent_frames=4,
            patch_height=1,
            patch_width=2,
            temporal_window_bounds=(-0.4, 0.0),
            includes_first_frame=True,
            control_attends_sensor=True,
            seconds_per_frame=0.4,
            max_und_tokens=40,
            items=(
                MaskItem((4, 1, 2), 2, is_control=True, seconds_per_frame=0.4),
                MaskItem((4, 1, 2), 2, seconds_per_frame=0.4),
            ),
        )
        gen, dim = layout.gen_tokens, 8
        q = torch.randn(1, gen, heads, dim)
        k, v = torch.randn(2, 1, gen, kv_heads, dim)
        k_und, v_und = torch.randn(2, 1, 6, kv_heads, dim)
        full = padded_multiview_flex_attention(
            q, k, v, k_und, v_und, MultiviewAttentionContext(layout, {}, {})
        )
        outputs = []
        for rank in range(world_size):

            def to_heads(x, head_dim=2, rank=rank):
                local = x.shape[head_dim] // world_size
                return x.narrow(head_dim, rank * local, local)

            with mock.patch.multiple(
                mv_dit,
                get_ulysses_parallel_world_size=lambda: world_size,
                get_ring_parallel_world_size=lambda: 1,
                get_ulysses_parallel_rank=lambda rank=rank: rank,
                _usp_input_all_to_all=to_heads,
                _usp_output_all_to_all=lambda x, head_dim=2: x,
            ):
                outputs.append(
                    Cosmos3MultiviewCrossAttention._forward_multiview(
                        None,
                        q,
                        k,
                        v,
                        k_und,
                        v_und,
                        MultiviewAttentionContext(layout, {}, {}),
                    )
                )
        torch.testing.assert_close(
            torch.cat(outputs, dim=2), full, atol=1e-6, rtol=1e-6
        )

    def test_heads_shard_without_repetition(self):
        self._emulate(2)

    def test_fewer_kv_heads_than_ranks_repeat_each_head(self):
        # Ultra's 4 KV heads under --ulysses-degree 8, scaled down: 2 KV heads
        # over 4 ranks, each rank owning one copy that serves its query heads.
        self._emulate(4)
        self._emulate(8)


class TestI4Deployment(unittest.TestCase):
    def test_ultra_block_defaults_follow_i4(self):
        deployment = parse_multiview_deployment_config(_ultra_config())
        self.assertEqual(deployment.schema, MULTIVIEW_SCHEMA_I4)
        self.assertTrue(deployment.is_i4_contract)
        self.assertFalse(deployment.requires_mads_cameras)
        self.assertEqual(deployment.window_bounds, (-0.4, 0.0))
        self.assertTrue(deployment.attends_first_frame)
        self.assertEqual(deployment.attention_scope, "decomposed")
        self.assertIsNone(deployment.rig_view_embedding)
        self.assertIsNone(deployment.rig_view_ids(["anything"]))
        self.assertEqual(deployment.cameras, ())
        self.assertIsNone(deployment.lidar)
        self.assertEqual(
            deployment.inference_defaults, COSMOS3_MULTIVIEW_I4_INFERENCE_DEFAULTS
        )
        self.assertEqual(deployment.inference_default("fps", 30), 10.0)
        self.assertEqual(deployment.inference_default("resolution", "480"), "720")
        self.assertIn("+ first frame", deployment.describe_window())

    def test_ultra_block_variants(self):
        symmetric = parse_multiview_deployment_config(
            _ultra_config(
                {
                    "cross_view_temporal_window_bounds": [-0.67, 0.67],
                    "cross_view_attends_first_frame": True,
                    "cameras": ["view_00", "view_01"],
                    "inference_defaults": {"resolution": "480", "num_frames": 61},
                }
            )
        )
        self.assertEqual(symmetric.window_bounds, (-0.67, 0.67))
        self.assertEqual(symmetric.cameras, ("view_00", "view_01"))
        self.assertEqual(symmetric.inference_default("resolution", None), "480")
        self.assertEqual(symmetric.inference_default("num_frames", None), 61)
        past = parse_multiview_deployment_config(
            _ultra_config({"cross_view_past_window_seconds": 0.4})
        )
        self.assertEqual(past.window_bounds, (-0.4, 0.0))
        self.assertIsNone(past.temporal_window_bounds)
        frame_index = parse_multiview_deployment_config(_ultra_config({}))
        self.assertIsNone(frame_index.window_bounds)
        self.assertEqual(frame_index.describe_window(), "same frame index")

    def test_ultra_block_validation(self):
        for block, message in (
            ({"cross_view_attends_first_frame": True}, "first_frame"),
            (
                {
                    "cross_view_past_window_seconds": 0.4,
                    "cross_view_temporal_window_bounds": [-0.4, 0],
                },
                "not both",
            ),
            ({"attention_scope": "diagonal"}, "attention_scope"),
            ({"rig_view_embeddings": {}}, "Unknown"),
            ({"inference_defaults": {"sampler": "euler"}}, "Unknown"),
            ({"inference_defaults": {"num_frames": 1}}, "num_frames"),
            ({"inference_defaults": {"aspect_ratio": "2,1"}}, "aspect_ratio"),
            ({"cameras": ["a", "a"]}, "unique"),
        ):
            with self.subTest(block=block):
                with self.assertRaisesRegex((ValueError, TypeError), message):
                    parse_multiview_deployment_config(_ultra_config(block))

    def test_rig_embedding_is_optional_and_validated(self):
        rig = {
            "num_embeddings": 3,
            "camera_ids": {"view_00": 0, "view_01": 1},
            "lidar_id": 2,
        }
        deployment = parse_multiview_deployment_config(
            _ultra_config({**ULTRA_BLOCK, "rig_view_embedding": rig})
        )
        self.assertEqual(deployment.rig_view_ids(["view_01", "view_00"]), [1, 0])
        with self.assertRaisesRegex(ValueError, "no row"):
            deployment.rig_view_ids(["view_02"])

    def test_backbone_gate_and_heads(self):
        with self.assertRaisesRegex(ValueError, "backbone_type"):
            parse_multiview_deployment_config(
                {**_ultra_config(), "backbone_type": "cosmos3_nano"}
            )
        self.assertEqual(multiview_attention_heads(_ultra_config()), (64, 4))
        self.assertEqual(
            multiview_attention_heads(
                {**_ultra_config(), "gen_num_key_value_heads": 64}
            ),
            (64, 64),
        )
        self.assertIsNone(multiview_attention_heads({}))

    def test_server_args_ulysses_follow_checkpoint_heads(self):
        config = Cosmos3MultiviewConfig()
        config.multiview_attention_heads = (64, 4)
        args = SimpleNamespace(tp_size=1, ring_degree=1, ulysses_degree=8, sp_degree=8)
        with mock.patch.object(
            Cosmos3MultiviewConfig.__mro__[1], "validate_server_args"
        ):
            config.validate_server_args(args)
            args.ulysses_degree = args.sp_degree = 3
            with self.assertRaisesRegex(ValueError, "query heads"):
                config.validate_server_args(args)

    def test_i4_contract_rounds_frames_up(self):
        config = Cosmos3MultiviewConfig()
        self.assertEqual(config.adjust_num_frames(94), 93)  # unresolved: Cosmos3 rule
        config.multiview_deployment = parse_multiview_deployment_config(_ultra_config())
        self.assertEqual(config.adjust_num_frames(94), 97)
        self.assertEqual(config.adjust_num_frames(93), 93)
        self.assertEqual(config.adjust_num_frames(1), 1)

    def test_layout_follows_the_contract(self):
        deployment = parse_multiview_deployment_config(_ultra_config())
        items = [MaskItem((6, 1, 1), 2, seconds_per_frame=0.4)]
        layout = multiview_layout_for(
            deployment,
            items=items,
            num_views=2,
            latent_frames=6,
            patch_hw=(1, 1),
            seconds_per_frame=0.4,
            backend="triton",
        )
        self.assertEqual(layout.window_bounds, (-0.4, 0.0))
        self.assertIsNone(layout.decomposed_temporal_window_seconds)
        self.assertTrue(layout.includes_first_frame)
        self.assertTrue(layout.control_attends_sensor)
        self.assertEqual(layout.items, tuple(items))
        past = parse_multiview_deployment_config(
            _ultra_config({"cross_view_past_window_seconds": 0.4})
        )
        layout = multiview_layout_for(
            past,
            items=items,
            num_views=2,
            latent_frames=6,
            patch_hw=(1, 1),
            seconds_per_frame=0.4,
            backend="triton",
        )
        self.assertEqual(layout.decomposed_temporal_window_seconds, 0.4)
        self.assertIsNone(layout.temporal_window_bounds)


class TestHintlessRequests(unittest.TestCase):
    def _params(self, **kwargs):
        params = Cosmos3MultiviewSamplingParams(**kwargs)
        deployment = parse_multiview_deployment_config(_ultra_config())
        params._apply_deployment_defaults(deployment)
        params._resolve_canvas(deployment)
        params._validate_request_mode()
        params._apply_guidance_policy(deployment)
        return params

    def test_i4_defaults(self):
        views = [{"camera_key": "view_00", "prompt": "A road."}]
        params = self._params(multiview={"views": views})
        self.assertEqual(params.num_frames, 93)
        self.assertEqual(params.fps, 10)
        self.assertEqual((params.width, params.height), (1280, 720))
        self.assertEqual(params.aspect_ratio, "16,9")
        self.assertEqual(params.num_inference_steps, 35)
        self.assertEqual(params.guidance_scale, 6.0)
        self.assertEqual(params.flow_shift, 10.0)
        explicit = self._params(
            multiview={"views": views, "aspect_ratio": "4:3", "resolution": "480"},
            num_frames=61,
            fps=24,
        )
        self.assertEqual((explicit.width, explicit.height), (736, 544))
        self.assertEqual((explicit.num_frames, explicit.fps), (61, 24))

    def test_mode_rules(self):
        hintless = [
            {"camera_key": "view_00", "prompt": "a", "vision_path": "a.mp4"},
            {"camera_key": "view_01", "prompt": "b", "vision_path": "b.mp4"},
        ]
        self._params(multiview={"views": hintless})
        with self.assertRaisesRegex(ValueError, "wsm requires a control_path"):
            self._params(multiview={"views": hintless}, wsm={})
        completion = copy.deepcopy(hintless)
        del completion[1]["vision_path"]
        with self.assertRaisesRegex(ValueError, "view completion requires"):
            self._params(multiview={"views": completion})
        controlled = copy.deepcopy(completion)
        for index, view in enumerate(controlled):
            view["control_path"] = f"wsm_{index}.mp4"
        self._params(multiview={"views": controlled}, wsm={})
        self.assertEqual(
            [
                view.control
                for view in Cosmos3MultiviewSamplingParams(
                    multiview={"views": hintless}
                ).resolve_views()
            ],
            [None, None],
        )
        self.assertEqual(len(validate_multiview_request({"views": hintless})), 2)


class TestPerViewCaptionFormatting(unittest.TestCase):
    META = dict(num_frames=93, fps=10.0, height=720, width=1280)

    def test_i4_template_edge_cases(self):
        # i4 _format_prompt_with_template: strip, then rstrip(".") + ". " per sentence.
        self.assertEqual(
            format_view_prompt("A road.", **self.META),
            "A road. The video is 9.0 seconds long and is of 10 FPS. "
            "This video is of 720x1280 resolution.",
        )
        self.assertEqual(
            format_view_prompt("A road ...  ", **self.META),
            "A road . The video is 9.0 seconds long and is of 10 FPS. "
            "This video is of 720x1280 resolution.",
        )
        self.assertEqual(
            format_view_prompt("", **self.META),
            ". The video is 9.0 seconds long and is of 10 FPS. "
            "This video is of 720x1280 resolution.",
        )

    def test_json_caption_gets_structured_metadata(self):
        out = format_view_prompt(
            '{"subjects": "car"}', aspect_ratio="16,9", **self.META
        )
        self.assertEqual(
            out,
            '{"subjects": "car", "duration": "9s", "fps": 10.0, '
            '"resolution": {"H": 720, "W": 1280}, "aspect_ratio": "16,9"}',
        )

    def test_rig_header_only_for_mads_keys(self):
        self.assertTrue(
            uses_mads_rig_header(["camera_front_wide_120fov", "camera_rear_tele_30fov"])
        )
        self.assertFalse(uses_mads_rig_header(["camera_front_wide_120fov", "view_01"]))
        generic = format_per_view_prompts(
            ["A road.", "Trees."], ["view_00", "view_01"], emphasis=None, **self.META
        )
        self.assertEqual(
            generic[1],
            "Trees. The video is 9.0 seconds long and is of 10 FPS. "
            "This video is of 720x1280 resolution.",
        )
        mads = format_per_view_prompts(
            ["A road."], ["camera_front_wide_120fov"], emphasis="Follow.", **self.META
        )
        self.assertTrue(mads[0].startswith("This multiview driving sequence"))
        self.assertTrue(mads[0].endswith("resolution. Follow."))


class TestLidarOnlyRequests(unittest.TestCase):
    """LiDAR transfer (HD-map control) and generation on the LiDAR clock."""

    def _deployment(self):
        return parse_multiview_deployment_config(
            _ultra_config(
                {
                    **ULTRA_BLOCK,
                    "lidar": {
                        **LIDAR_BLOCK,
                        "streaming_chunk_frames": 9,
                        "streaming_context_frames": 9,
                    },
                    "lidar_latent_patch_size_hw": [1, 1],
                }
            )
        )

    def _adjust(self, params):
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(multiview_deployment=self._deployment())
        )
        with mock.patch.object(
            Cosmos3MultiviewSamplingParams.__mro__[2], "_adjust"
        ) as base_adjust:
            params._adjust(server_args)
        base_adjust.assert_called_once()
        return params

    def test_defaults_follow_the_legacy_lidar_recipe(self):
        params = self._adjust(
            Cosmos3MultiviewSamplingParams(
                prompt="A street.", lidar={"control_path": "hdmap.safetensors"}
            )
        )
        self.assertTrue(params.is_lidar_only())
        self.assertEqual((params.num_frames, params.fps), (96, 10))
        self.assertEqual(params.guidance_scale, 3.0)
        self.assertEqual(params.flow_shift, 5.0)
        self.assertEqual(params.num_inference_steps, 35)
        self.assertEqual(params.negative_metadata_mode, "none")
        self.assertFalse(params.emphasize_control_in_prompt)
        # Sweep counts are not rounded to the camera VAE's 4k+1 grid.
        params = self._adjust(Cosmos3MultiviewSamplingParams(lidar={}, num_frames=134))
        self.assertEqual(params.num_frames, 134)
        self.assertEqual(params.resolved_lidar()["control_path"], None)

    def test_mode_rules(self):
        self.assertFalse(
            Cosmos3MultiviewSamplingParams(
                multiview={"views": [{"camera_key": "view_00"}]},
                lidar={"control_path": "hdmap.safetensors"},
            ).is_lidar_only()
        )
        with self.assertRaisesRegex(ValueError, "control_path"):
            # Joint requests still need the HD-map control.
            Cosmos3MultiviewSamplingParams(
                multiview={"views": [{"camera_key": "view_00"}]}, lidar={}
            )
        with self.assertRaisesRegex(ValueError, "10 Hz"):
            params = Cosmos3MultiviewSamplingParams(lidar={}, fps=30)
            params._explicit_fields = {"fps"}
            self._adjust(params)
        params = Cosmos3MultiviewSamplingParams(lidar={})
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(
                multiview_deployment=parse_multiview_deployment_config(_ultra_config())
            )
        )
        with self.assertRaisesRegex(ValueError, "LiDAR encoder"):
            params._adjust(server_args)

    def test_mask_items_place_lidar_past_the_cameras(self):
        camera = (6, 1, 1)
        lidar = (5, 1, 1)
        joint = multiview_mask_items(
            num_views=2,
            camera_token_shape=camera,
            camera_seconds_per_frame=0.4,
            has_camera_control=True,
            lidar_token_shape=lidar,
            lidar_seconds_per_frame=0.1,
            has_lidar_control=True,
        )
        self.assertEqual(
            [(i.is_control, i.is_lidar, i.view_offset) for i in joint],
            [(True, False, 0), (False, False, 0), (True, True, 2), (False, True, 2)],
        )
        transfer = multiview_mask_items(
            num_views=0,
            camera_token_shape=camera,
            camera_seconds_per_frame=0.4,
            has_camera_control=False,
            lidar_token_shape=lidar,
            lidar_seconds_per_frame=0.1,
            has_lidar_control=True,
        )
        self.assertEqual(
            [(i.is_control, i.view_offset, i.num_views) for i in transfer],
            [(True, 0, 1), (False, 0, 1)],
        )
        generation = multiview_mask_items(
            num_views=0,
            camera_token_shape=camera,
            camera_seconds_per_frame=0.4,
            has_camera_control=False,
            lidar_token_shape=lidar,
            lidar_seconds_per_frame=0.1,
        )
        self.assertEqual([i.is_control for i in generation], [False])
        with self.assertRaisesRegex(ValueError, "camera or LiDAR"):
            multiview_mask_items(
                num_views=0,
                camera_token_shape=camera,
                camera_seconds_per_frame=0.4,
                has_camera_control=False,
            )

    def test_condition_indexes(self):
        self.assertEqual(
            multiview_condition_indexes(
                known_views=[],
                local_indexes=[0, 1],
                num_views=3,
                latent_frames_per_view=4,
            ),
            [0, 1, 4, 5, 8, 9],
        )
        self.assertEqual(
            multiview_condition_indexes(
                known_views=[1], local_indexes=[], num_views=3, latent_frames_per_view=2
            ),
            [2, 3],
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            multiview_condition_indexes(
                known_views=[], local_indexes=[4], num_views=1, latent_frames_per_view=4
            )

    def test_range_preview_video(self):
        clip = torch.zeros(3, 2, 4, 6)
        clip[0, :, :, :3] = 50.0
        clip[2, :, :, :3] = 1.0
        video = lidar_range_preview_video(clip, min_range_m=5.0, max_range_m=100.0)
        self.assertEqual(tuple(video.shape), (1, 3, 2, 8, 3))
        self.assertGreaterEqual(float(video.min()), -1.0)
        self.assertLessEqual(float(video.max()), 1.0)
        # Dropped rays render black.
        self.assertTrue(bool((video[0, :, :, :, -1] == -1.0).all()))


class TestLayoutStreams(unittest.TestCase):
    """``Cosmos3MultiviewTransformer._layout_streams`` pairs items with latents."""

    def _streams(self, items, targets, controls=None, lidar_control=None, lidar=True):
        layout = MultiviewLayout(
            num_views=2,
            latent_frames=4,
            patch_height=1,
            patch_width=1,
            decomposed_temporal_window_seconds=0.4,
            items=tuple(items),
        )
        owner = SimpleNamespace(lidar_config={} if lidar else None)
        return Cosmos3MultiviewTransformer._layout_streams(
            owner, layout, targets, controls, lidar_control
        )

    def test_pairs_items_in_layout_order(self):
        camera = MaskItem((4, 1, 1), 2, seconds_per_frame=0.4)
        control = MaskItem((4, 1, 1), 2, is_control=True, seconds_per_frame=0.4)
        lidar = MaskItem(
            (3, 1, 1), 1, view_offset=2, seconds_per_frame=0.1, is_lidar=True
        )
        lidar_control = MaskItem(
            (3, 1, 1),
            1,
            view_offset=2,
            is_control=True,
            seconds_per_frame=0.1,
            is_lidar=True,
        )
        cam_t, cam_c, lid_t, lid_c = (torch.full((1,), float(i)) for i in range(4))
        # Hint-less: the camera target alone.
        streams, items = self._streams([camera], [cam_t])
        self.assertEqual([s.item() for s in streams], [0.0])
        # Transfer, and its dropped-control CFG branch.
        streams, items = self._streams([control, camera], [cam_t], [cam_c])
        self.assertEqual([s.item() for s in streams], [1.0, 0.0])
        streams, items = self._streams([control, camera], [cam_t], None)
        self.assertEqual((len(streams), items), (1, (camera,)))
        # Joint camera/LiDAR transfer: controls in layout order.
        streams, _ = self._streams(
            [control, camera, lidar_control, lidar], [cam_t, lid_t], [cam_c], lid_c
        )
        self.assertEqual([s.item() for s in streams], [1.0, 0.0, 3.0, 2.0])
        # LiDAR-only transfer: the LiDAR control rides in control_latents.
        streams, _ = self._streams([lidar_control, lidar], [lid_t], [lid_c])
        self.assertEqual([s.item() for s in streams], [3.0, 2.0])
        with self.assertRaisesRegex(ValueError, "do not match"):
            self._streams([control, camera], [cam_t, lid_t], [cam_c])
        with self.assertRaisesRegex(ValueError, "LiDAR projections"):
            self._streams([lidar], [lid_t], lidar=False)


if __name__ == "__main__":
    unittest.main()
