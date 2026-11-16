# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Cosmos3 config, weight mapping, and sampling params."""

import contextlib
import dataclasses
import importlib.util
import json
import types
import unittest
from unittest import mock

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.configs.models.dits.cosmos3video import (
    _build_cosmos3_param_names_mapping,
)
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import (
    Cosmos3Config,
    is_nano_checkpoint,
)
from sglang.multimodal_gen.configs.pipeline_configs.cosmos_dreams import (
    CosmosDreamsConfig,
)
from sglang.multimodal_gen.configs.sample.cosmos3 import (
    COSMOS3_EDGE_SUPPORTED_RESOLUTIONS,
    Cosmos3SamplingParams,
    Cosmos3TransferHint,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.task_type import DataType
from sglang.multimodal_gen.registry import (
    _PIPELINE_REGISTRY,
    _discover_and_register_pipelines,
    _get_config_info,
    get_non_diffusers_pipeline_name,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.entrypoints.action.protocol import (
    action_generation_response,
    action_metadata,
    build_action_sampling_params,
)
from sglang.multimodal_gen.runtime.entrypoints.openai import utils as openai_utils
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
    VideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import request_model_kwargs
from sglang.multimodal_gen.runtime.entrypoints.openai.video_api import (
    _multipart_video_extras,
    _resolve_video_path,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    Qwen3VLTextRotaryEmbedding,
)
from sglang.multimodal_gen.runtime.loader.component_loaders import scheduler_loader
from sglang.multimodal_gen.runtime.loader.component_loaders.scheduler_loader import (
    SchedulerLoader,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.models.dits import cosmos3video as cosmos3_dit
from sglang.multimodal_gen.runtime.models.dits.cosmos3video import (
    Cosmos3CrossAttention,
    Cosmos3OmniTransformer,
    DomainAwareLinear,
    _can_enable_t1_fused_qk_norm_rope,
    build_multi_control_layout,
    compute_mrope_position_ids_action,
    compute_mrope_position_ids_sound,
    compute_mrope_position_ids_vision,
    sequence_shard_padding,
    shard_sequence,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
    Cosmos3DecodingStage,
    Cosmos3DenoisingStage,
    Cosmos3ImagePreprocessStage,
    Cosmos3LatentPreparationStage,
    Cosmos3TimestepPreparationStage,
    Cosmos3TokenizationStage,
    _inject_caption_metadata,
    _pad_transfer_frames,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_action import (
    EMBODIMENT_TO_DOMAIN_ID,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_guardrails import (
    is_cosmos_guardrail_available,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_transfer import (
    COSMOS3_TRANSFER_SYSTEM_PROMPT,
    blur_control_uint8,
    edge_control_uint8,
    resize_center_crop_uint8,
)
from sglang.multimodal_gen.runtime.warmup_request_builder import build_warmup_reqs
from sglang.srt.environ import envs
from sglang.test.test_utils import CustomTestCase


def _apply(mapping_fn, key):
    """Return (target_key, merge_index, total_splits) for a diffusers weight key."""
    return mapping_fn(key)


def _cosmos3_server_args(config=None, batching_max_size=1):
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
        pipeline_config=config or Cosmos3Config(),
    )


class _RopeOnlyTransformer(torch.nn.Module):
    _maybe_install_step_mixed_precision = (
        Cosmos3OmniTransformer._maybe_install_step_mixed_precision
    )

    def __init__(self, device, meta_rope, rope_theta, head_dim):
        super().__init__()
        self.rope_theta = rope_theta
        self.hf_config = {}
        self.modelopt_fp8_checkpoint = False
        self.language_model = torch.nn.Module()
        with torch.device("meta" if meta_rope else "cpu"):
            self.language_model.rotary_emb = Qwen3VLTextRotaryEmbedding(
                head_dim=head_dim, rope_theta=rope_theta
            )
        self.language_model.embed_tokens = torch.nn.Embedding(
            num_embeddings=2, embedding_dim=2, device=device
        )
        self.time_embedder = torch.nn.Linear(
            in_features=2, out_features=2, device=device
        )
        self.proj_in = torch.nn.Linear(in_features=2, out_features=2, device=device)
        self.proj_out = torch.nn.Linear(in_features=2, out_features=2, device=device)


@unittest.skipUnless(
    torch.cuda.is_available(), "requires CUDA/ROCm for CPU/GPU RoPE parity"
)
class TestCosmos3RopePlacement(CustomTestCase):
    def test_rope_table_does_not_depend_on_dit_placement(self):
        """A CPU-offloaded DiT must build the RoPE table a resident one builds."""
        rope_theta, head_dim = 100_000_000, 128
        exponents = torch.arange(
            start=0,
            end=head_dim,
            step=2,
            dtype=torch.float32,
            device=get_local_torch_device(),
        )
        reference = (1.0 / (rope_theta ** (exponents / head_dim))).cpu()
        for device in ("cpu", "cuda"):
            for meta_rope in (True, False):
                with self.subTest(device=device, meta_rope=meta_rope):
                    model = _RopeOnlyTransformer(
                        device=device,
                        meta_rope=meta_rope,
                        rope_theta=rope_theta,
                        head_dim=head_dim,
                    )
                    Cosmos3OmniTransformer.post_load_weights(model)
                    rotary_emb = model.language_model.rotary_emb
                    self.assertEqual(rotary_emb.inv_freq.device.type, device)
                    self.assertEqual(rotary_emb.inv_freq.dtype, torch.float32)
                    self.assertTrue(torch.equal(rotary_emb.inv_freq.cpu(), reference))
                    self.assertNotIn("inv_freq", rotary_emb.state_dict())


class TestCosmos3T1FusedQKNormRoPE(CustomTestCase):
    def _can_enable(
        self,
        *,
        is_blackwell=False,
        is_hopper=False,
        is_sm121=False,
        hidden_act="silu",
        tp_size=1,
        sp_size=1,
        is_compiled=False,
        hidden_size=0,
    ):
        return _can_enable_t1_fused_qk_norm_rope(
            is_blackwell=is_blackwell,
            is_hopper=is_hopper,
            is_sm121=is_sm121,
            hidden_act=hidden_act,
            tp_size=tp_size,
            sp_size=sp_size,
            is_compiled=is_compiled,
            hidden_size=hidden_size,
        )

    def test_blackwell_remains_enabled(self):
        self.assertTrue(
            self._can_enable(
                is_blackwell=True,
                hidden_act="relu2",
                tp_size=2,
                sp_size=2,
            )
        )

    def test_hopper_swiglu_single_gpu_enabled(self):
        self.assertTrue(self._can_enable(is_hopper=True))

    def test_hopper_dense_mlp_disabled(self):
        self.assertFalse(self._can_enable(is_hopper=True, hidden_act="relu2"))

    def test_hopper_super_t2i_tp2_enabled(self):
        self.assertTrue(self._can_enable(is_hopper=True, hidden_size=5120, tp_size=2))

    def test_hopper_super_t2i_unsupported_topologies_disabled(self):
        for overrides in (
            {"hidden_act": "relu2"},
            {"hidden_act": "gelu"},
            {"hidden_size": 4096},
            {"tp_size": 4},
            {"sp_size": 2},
            {"is_compiled": True},
            {"is_hopper": False},
        ):
            with self.subTest(overrides=overrides):
                settings = dict(is_hopper=True, hidden_size=5120, tp_size=2)
                settings.update(overrides)
                self.assertFalse(self._can_enable(**settings))

    def test_hopper_edge_single_gpu_enabled(self):
        self.assertTrue(
            self._can_enable(is_hopper=True, hidden_act="relu2", hidden_size=2048)
        )

    def test_hopper_edge_parallel_and_compiled_disabled(self):
        for override in ({"tp_size": 2}, {"sp_size": 2}, {"is_compiled": True}):
            with self.subTest(override=override):
                self.assertFalse(
                    self._can_enable(
                        is_hopper=True,
                        hidden_act="relu2",
                        hidden_size=2048,
                        **override,
                    )
                )

    def test_hopper_larger_dense_mlp_disabled(self):
        self.assertFalse(
            self._can_enable(is_hopper=True, hidden_act="relu2", hidden_size=4096)
        )

    def test_hopper_other_tensor_parallel_shapes_disabled(self):
        self.assertFalse(self._can_enable(is_hopper=True, tp_size=2))

    def test_hopper_sequence_parallel_disabled(self):
        self.assertFalse(self._can_enable(is_hopper=True, sp_size=2))

    def test_compiled_path_disabled(self):
        self.assertFalse(self._can_enable(is_hopper=True, is_compiled=True))

    def test_sm121_single_gpu_nano_and_edge_enabled(self):
        self.assertTrue(self._can_enable(is_sm121=True, hidden_size=4096))
        self.assertTrue(
            self._can_enable(is_sm121=True, hidden_act="relu2", hidden_size=2048)
        )

    def test_sm121_unmeasured_shapes_and_topologies_disabled(self):
        for overrides in (
            {"hidden_size": 5120},
            {"hidden_size": 5120, "tp_size": 2},
            {"hidden_act": "relu2", "hidden_size": 4096},
            {"hidden_size": 4096, "tp_size": 2},
            {"hidden_size": 4096, "sp_size": 2},
            {"hidden_size": 4096, "is_compiled": True},
        ):
            with self.subTest(overrides=overrides):
                self.assertFalse(self._can_enable(is_sm121=True, **overrides))

    def test_other_architecture_disabled(self):
        self.assertFalse(self._can_enable())


class TestCosmos3ParamNamesMapping(unittest.TestCase):
    """Verify diffusers → sglang weight key translations."""

    @classmethod
    def setUpClass(cls):
        cls.fn = staticmethod(
            get_param_names_mapping(_build_cosmos3_param_names_mapping())
        )

    # --- skipped / dropped weights ---

    def test_lm_head_dropped(self):
        key, idx, total = _apply(self.fn, "lm_head.weight")
        self.assertEqual(key, "")

    def test_norm_dropped(self):
        key, idx, total = _apply(self.fn, "norm.weight")
        self.assertEqual(key, "")

    # --- top-level pass-through ---

    def test_audio_proj_in_passthrough(self):
        key, *_ = _apply(self.fn, "audio_proj_in.weight")
        self.assertEqual(key, "audio_proj_in.weight")

    def test_action_proj_in_passthrough(self):
        key, *_ = _apply(self.fn, "action_proj_in.fc.weight")
        self.assertEqual(key, "action_proj_in.fc.weight")

    def test_embed_tokens(self):
        key, *_ = _apply(self.fn, "embed_tokens.weight")
        self.assertEqual(key, "language_model.embed_tokens.weight")

    def test_norm_moe_gen(self):
        key, *_ = _apply(self.fn, "norm_moe_gen.weight")
        self.assertEqual(key, "norm_moe_gen.weight")

    # --- time embedder (pass-through: checkpoint already uses linear_1/2) ---

    def test_time_embedder_linear_1_passthrough(self):
        key, *_ = _apply(self.fn, "time_embedder.linear_1.weight")
        self.assertEqual(key, "time_embedder.linear_1.weight")

    def test_time_embedder_linear_2_passthrough(self):
        key, *_ = _apply(self.fn, "time_embedder.linear_2.bias")
        self.assertEqual(key, "time_embedder.linear_2.bias")

    # --- GEN pathway: Q/K/V merge (must not be claimed by UND catch-all) ---

    def test_gen_q_proj_key_and_merge_index(self):
        key, idx, total = _apply(self.fn, "layers.3.self_attn.add_q_proj.weight")
        self.assertEqual(key, "gen_layers.3.cross_attention.to_qkv.weight")
        self.assertEqual(idx, 0)
        self.assertEqual(total, 3)

    def test_gen_k_proj_merge_index(self):
        _, idx, total = _apply(self.fn, "layers.0.self_attn.add_k_proj.weight")
        self.assertEqual(idx, 1)
        self.assertEqual(total, 3)

    def test_gen_v_proj_merge_index(self):
        _, idx, total = _apply(self.fn, "layers.0.self_attn.add_v_proj.weight")
        self.assertEqual(idx, 2)
        self.assertEqual(total, 3)

    def test_gen_o_proj(self):
        key, idx, total = _apply(self.fn, "layers.5.self_attn.to_add_out.weight")
        self.assertEqual(key, "gen_layers.5.cross_attention.to_out.weight")
        self.assertIsNone(idx)

    def test_gen_norm_added_q(self):
        key, idx, _ = _apply(self.fn, "layers.2.self_attn.norm_added_q.weight")
        self.assertEqual(key, "gen_layers.2.cross_attention.norm_q.weight")
        self.assertIsNone(idx)

    def test_gen_norm_added_k(self):
        key, idx, _ = _apply(self.fn, "layers.2.self_attn.norm_added_k.weight")
        self.assertEqual(key, "gen_layers.2.cross_attention.norm_k.weight")
        self.assertIsNone(idx)

    def test_gen_mlp_gate_proj(self):
        key, idx, total = _apply(self.fn, "layers.2.mlp_moe_gen.gate_proj.weight")
        self.assertEqual(key, "gen_layers.2.mlp.gate_up_proj.weight")
        self.assertEqual(idx, 0)
        self.assertEqual(total, 2)

    def test_gen_mlp_up_proj(self):
        key, idx, total = _apply(self.fn, "layers.2.mlp_moe_gen.up_proj.weight")
        self.assertEqual(key, "gen_layers.2.mlp.gate_up_proj.weight")
        self.assertEqual(idx, 1)
        self.assertEqual(total, 2)

    def test_gen_mlp_down_proj_passthrough(self):
        key, idx, _ = _apply(self.fn, "layers.2.mlp_moe_gen.down_proj.weight")
        self.assertEqual(key, "gen_layers.2.mlp.down_proj.weight")
        self.assertIsNone(idx)

    # --- UND pathway: Q/K/V merge ---

    def test_und_q_proj_key_and_merge_index(self):
        key, idx, total = _apply(self.fn, "layers.7.self_attn.to_q.weight")
        self.assertEqual(key, "language_model.layers.7.self_attn.to_qkv.weight")
        self.assertEqual(idx, 0)
        self.assertEqual(total, 3)

    def test_und_k_proj_merge_index(self):
        _, idx, total = _apply(self.fn, "layers.0.self_attn.to_k.weight")
        self.assertEqual(idx, 1)
        self.assertEqual(total, 3)

    def test_und_v_proj_merge_index(self):
        _, idx, total = _apply(self.fn, "layers.0.self_attn.to_v.weight")
        self.assertEqual(idx, 2)
        self.assertEqual(total, 3)

    def test_und_mlp_gate_proj(self):
        key, idx, total = _apply(self.fn, "layers.1.mlp.gate_proj.weight")
        self.assertEqual(key, "language_model.layers.1.mlp.gate_up_proj.weight")
        self.assertEqual(idx, 0)
        self.assertEqual(total, 2)

    def test_und_mlp_up_proj(self):
        _, idx, total = _apply(self.fn, "layers.1.mlp.up_proj.weight")
        self.assertEqual(idx, 1)
        self.assertEqual(total, 2)

    def test_und_layernorm_catch_all(self):
        key, idx, _ = _apply(self.fn, "layers.0.input_layernorm.weight")
        self.assertEqual(key, "language_model.layers.0.input_layernorm.weight")
        self.assertIsNone(idx)

    # --- ordering: GEN patterns must not be swallowed by UND catch-all ---

    def test_gen_layernorm_not_mapped_to_und(self):
        key, *_ = _apply(self.fn, "layers.0.input_layernorm_moe_gen.weight")
        self.assertIn("gen_layers", key)
        self.assertNotIn("language_model", key)

    def test_gen_post_attention_layernorm_not_mapped_to_und(self):
        key, *_ = _apply(self.fn, "layers.4.post_attention_layernorm_moe_gen.weight")
        self.assertIn("gen_layers", key)
        self.assertNotIn("language_model", key)


class TestCosmos3DenseParamNamesMapping(unittest.TestCase):
    """Dense (squared-ReLU) checkpoints ship no gate_proj; up/down_proj must
    pass through unmerged."""

    @classmethod
    def setUpClass(cls):
        cls.fn = staticmethod(
            get_param_names_mapping(_build_cosmos3_param_names_mapping(gated_mlp=False))
        )

    def test_und_mlp_up_proj_unmerged(self):
        key, idx, _ = _apply(self.fn, "layers.1.mlp.up_proj.weight")
        self.assertEqual(key, "language_model.layers.1.mlp.up_proj.weight")
        self.assertIsNone(idx)

    def test_und_mlp_down_proj_unmerged(self):
        key, idx, _ = _apply(self.fn, "layers.1.mlp.down_proj.weight")
        self.assertEqual(key, "language_model.layers.1.mlp.down_proj.weight")
        self.assertIsNone(idx)

    def test_gen_mlp_up_proj_unmerged(self):
        key, idx, _ = _apply(self.fn, "layers.2.mlp_moe_gen.up_proj.weight")
        self.assertEqual(key, "gen_layers.2.mlp.up_proj.weight")
        self.assertIsNone(idx)

    def test_gen_mlp_down_proj_unmerged(self):
        key, idx, _ = _apply(self.fn, "layers.2.mlp_moe_gen.down_proj.weight")
        self.assertEqual(key, "gen_layers.2.mlp.down_proj.weight")
        self.assertIsNone(idx)

    def test_qkv_merge_still_applies(self):
        key, idx, total = _apply(self.fn, "layers.0.self_attn.to_q.weight")
        self.assertEqual(key, "language_model.layers.0.self_attn.to_qkv.weight")
        self.assertEqual((idx, total), (0, 3))


class TestCosmos3AdjustNumFrames(unittest.TestCase):
    """Verify VAE-aligned frame rounding in Cosmos3Config."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = Cosmos3Config()

    def test_single_frame_t2i_bypass(self):
        self.assertEqual(self.cfg.adjust_num_frames(1), 1)

    def test_already_aligned(self):
        # (81 - 1) = 80, 80 % 4 == 0
        self.assertEqual(self.cfg.adjust_num_frames(81), 81)

    def test_rounds_down_to_nearest_aligned(self):
        # (83 - 1) = 82 → floor(82/4)*4 + 1 = 81
        self.assertEqual(self.cfg.adjust_num_frames(83), 81)
        # (6 - 1) = 5 → floor(5/4)*4 + 1 = 5
        self.assertEqual(self.cfg.adjust_num_frames(6), 5)

    def test_minimum_video_frame_count(self):
        # 2 frames: (2-1)=1, 1//4=0, 0*4+1=1 → rounds to 1, but 1 is T2I — still valid
        self.assertEqual(self.cfg.adjust_num_frames(2), 1)


class TestCosmos3SchedulerConfig(unittest.TestCase):
    """Verify Cosmos3 scheduler class and flow-shift defaults."""

    def test_config_overrides_checkpoint_scheduler_class(self):
        cfg = Cosmos3Config()
        self.assertEqual(cfg.scheduler_class_override, "FlowUniPCMultistepScheduler")
        self.assertIsNone(cfg.flow_shift)

    def test_scheduler_loader_uses_configured_class_override(self):
        class FakeScheduler:
            def __init__(self, **config):
                self.config = config

        server_args = types.SimpleNamespace(
            pipeline_config=types.SimpleNamespace(
                scheduler_class_override="FlowUniPCMultistepScheduler",
                flow_shift=None,
            )
        )
        with (
            mock.patch.object(
                scheduler_loader,
                "get_diffusers_component_config",
                return_value={"_class_name": "CheckpointScheduler", "foo": "bar"},
            ),
            mock.patch.object(
                scheduler_loader.ModelRegistry,
                "resolve_model_cls",
                return_value=(FakeScheduler, None),
            ) as resolve,
        ):
            scheduler = SchedulerLoader().load_customized("unused", server_args)

        resolve.assert_called_once_with("FlowUniPCMultistepScheduler")
        self.assertEqual(scheduler.config["foo"], "bar")

    @staticmethod
    def _stage():
        stage = Cosmos3TimestepPreparationStage.__new__(Cosmos3TimestepPreparationStage)
        stage.scheduler = types.SimpleNamespace(
            config=types.SimpleNamespace(flow_shift=1.0)
        )
        return stage

    @staticmethod
    def _batch(**kwargs):
        sp_kwargs = kwargs.pop("sp_kwargs", {})
        return types.SimpleNamespace(
            sampling_params=Cosmos3SamplingParams(prompt="t", **sp_kwargs),
            data_type=kwargs.pop("data_type", DataType.VIDEO),
            preprocessed_image=kwargs.pop("preprocessed_image", None),
            preprocessed_video=kwargs.pop("preprocessed_video", None),
        )

    def test_per_mode_flow_shift_defaults(self):
        stage = self._stage()
        self.assertEqual(
            stage._default_flow_shift_for_mode(
                self._batch(data_type=DataType.IMAGE), is_edge=False
            ),
            3.0,
        )
        self.assertEqual(
            stage._default_flow_shift_for_mode(
                self._batch(preprocessed_image=torch.empty(1)), is_edge=False
            ),
            10.0,
        )
        self.assertEqual(
            stage._default_flow_shift_for_mode(
                self._batch(preprocessed_video=torch.empty(1)), is_edge=False
            ),
            10.0,
        )
        self.assertEqual(
            stage._default_flow_shift_for_mode(self._batch(), is_edge=False), 10.0
        )
        self.assertEqual(
            stage._default_flow_shift_for_mode(
                self._batch(sp_kwargs={"action_mode": "policy"}), is_edge=False
            ),
            10.0,
        )

    def test_edge_flow_shift_default(self):
        stage = self._stage()
        # Edge uses 3.0 for T2I and every video mode (T2V/I2V/V2V); action stays high.
        self.assertEqual(
            stage._default_flow_shift_for_mode(self._batch(), is_edge=True), 3.0
        )
        self.assertEqual(
            stage._default_flow_shift_for_mode(
                self._batch(data_type=DataType.IMAGE), is_edge=True
            ),
            3.0,
        )
        self.assertEqual(
            stage._default_flow_shift_for_mode(
                self._batch(preprocessed_image=torch.empty(1)), is_edge=True
            ),
            3.0,
        )
        self.assertEqual(
            stage._default_flow_shift_for_mode(
                self._batch(preprocessed_video=torch.empty(1)), is_edge=True
            ),
            3.0,
        )
        self.assertEqual(
            stage._default_flow_shift_for_mode(
                self._batch(sp_kwargs={"action_mode": "policy"}), is_edge=True
            ),
            10.0,
        )


class TestCosmos3EdgeSamplingDefaults(unittest.TestCase):
    """Edge variant fills its own resolution/guidance defaults; base is untouched."""

    def test_edge_t2v_defaults(self):
        sp = Cosmos3SamplingParams(prompt="t", num_frames=81)
        sp._resolve_variant_defaults(is_edge=True)
        self.assertEqual((sp.width, sp.height), (832, 480))
        self.assertEqual(sp.guidance_scale, 5.0)

    def test_edge_t2i_defaults(self):
        sp = Cosmos3SamplingParams(prompt="t", num_frames=1)
        sp._resolve_variant_defaults(is_edge=True)
        self.assertEqual((sp.width, sp.height), (640, 640))
        self.assertEqual(sp.guidance_scale, 7.0)

    def test_edge_restricts_supported_resolutions(self):
        sp = Cosmos3SamplingParams(prompt="t", num_frames=1)
        sp._resolve_variant_defaults(is_edge=True)
        self.assertEqual(sp.supported_resolutions, COSMOS3_EDGE_SUPPORTED_RESOLUTIONS)
        # Base high-res sizes are excluded so they trip the "unsupported" warning.
        self.assertNotIn((1280, 720), sp.supported_resolutions)
        self.assertNotIn((1024, 1024), sp.supported_resolutions)

    def test_non_edge_defers_resolution_to_base(self):
        sp = Cosmos3SamplingParams(prompt="t", num_frames=81)
        sp._resolve_variant_defaults(is_edge=False)
        self.assertIsNone(sp.width)
        self.assertIsNone(sp.height)
        self.assertEqual(sp.guidance_scale, 4.0)

    def test_explicit_resolution_preserved_for_edge(self):
        sp = Cosmos3SamplingParams(prompt="t", num_frames=81, width=1024, height=576)
        sp._resolve_variant_defaults(is_edge=True)
        self.assertEqual((sp.width, sp.height), (1024, 576))


def _cosmos3_size_server_args(config=None):
    server_args = _cosmos3_server_args(config)
    server_args.backend = "sglang"
    server_args.performance_mode = "manual"
    server_args.enable_cfg_parallel = False
    server_args.enable_breakable_cuda_graph = False
    server_args.enable_torch_compile = False
    server_args.enable_trace = False
    server_args.attention_backend_config = {}
    server_args.warmup_steps = 1
    server_args.warmup_num_frames = None
    return server_args


def _build_cosmos3_sampling(server_args, data_type, fields):
    with (
        mock.patch.object(
            openai_utils, "get_global_server_args", return_value=server_args
        ),
        mock.patch.object(
            SamplingParams, "from_pretrained", return_value=Cosmos3SamplingParams()
        ),
    ):
        return openai_utils.build_sampling_params(
            "size", prompt="a cat", request_data_type=data_type, **fields
        )


class TestCosmos3RequestSizeAlignment(unittest.TestCase):
    """Requested sizes reach the latent stage as multiples of 16."""

    def _assert_latent_stage_accepts(self, sampling, server_args):
        stage = Cosmos3LatentPreparationStage(vae=None, transformer=None)
        result = stage.verify_input(Req(sampling_params=sampling), server_args)
        self.assertTrue(result.is_valid(), result.get_failure_summary())

    def test_http_size_aligns_to_the_latent_grid(self):
        server_args = _cosmos3_size_server_args()
        for data_type, fields, expected in (
            (DataType.IMAGE, {"size": "1920x1080", "num_frames": 1}, (1920, 1072)),
            (DataType.IMAGE, {"size": "854x480", "num_frames": 1}, (848, 480)),
            (DataType.VIDEO, {"size": "1920x1080", "num_frames": 17}, (1920, 1072)),
            (DataType.VIDEO, {"width": 854, "height": 480}, (848, 480)),
            (DataType.IMAGE, {"size": "1280x720", "num_frames": 1}, (1280, 720)),
            (DataType.VIDEO, {"size": "832x480", "num_frames": 17}, (832, 480)),
            (DataType.VIDEO, {}, (1280, 720)),
        ):
            with self.subTest(data_type=data_type, fields=fields):
                sampling = _build_cosmos3_sampling(server_args, data_type, fields)
                self.assertEqual((sampling.width, sampling.height), expected)
                self._assert_latent_stage_accepts(sampling, server_args)

    def test_simulation_configs_keep_requested_size(self):
        server_args = _cosmos3_size_server_args(CosmosDreamsConfig())
        sampling = _build_cosmos3_sampling(
            server_args, DataType.VIDEO, {"size": "854x480", "num_frames": 17}
        )
        self.assertEqual((sampling.width, sampling.height), (854, 480))

    def test_warmup_resolution_aligns_to_the_latent_grid(self):
        server_args = _cosmos3_size_server_args()
        with mock.patch(
            "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
            return_value=Cosmos3SamplingParams(),
        ):
            reqs = build_warmup_reqs(
                server_args,
                warmup_resolutions=["1920x1080"],
                warmup_input_path="warmup.png",
                server_based_warmup=True,
            )
        self.assertEqual([(r.width, r.height) for r in reqs], [(1920, 1072)])


class TestCosmos3SamplingParamsDataType(unittest.TestCase):
    """Verify num_frames==1 flips data_type to IMAGE before file name derivation."""

    def test_single_frame_sets_image_data_type(self):
        params = Cosmos3SamplingParams(prompt="test", num_frames=1)
        params._set_output_file_name()
        self.assertEqual(params.data_type, DataType.IMAGE)
        self.assertTrue(
            params.output_file_name.endswith((".png", ".jpg", ".jpeg", ".webp")),
            f"Expected image extension, got: {params.output_file_name}",
        )

    def test_multi_frame_keeps_video_data_type(self):
        params = Cosmos3SamplingParams(prompt="test", num_frames=81)
        params._set_output_file_name()
        self.assertEqual(params.data_type, DataType.VIDEO)

    def test_default_num_frames_is_video(self):
        params = Cosmos3SamplingParams(prompt="test")
        params._set_output_file_name()
        self.assertEqual(params.data_type, DataType.VIDEO)

    def test_policy_adjusts_to_action_output(self):
        params = Cosmos3SamplingParams(
            prompt="test",
            action_mode="policy",
            num_frames=17,
            image_path="observation.png",
        )

        params._adjust(_cosmos3_server_args())

        self.assertEqual(params.data_type, DataType.ACTION)
        self.assertFalse(params.save_output)
        self.assertFalse(params.return_file_paths_only)
        self.assertIsNone(params.output_file_name)
        self.assertEqual(params.num_frames, 17)

    def test_forward_dynamics_remains_video_output(self):
        params = Cosmos3SamplingParams(
            prompt="test",
            action_mode="forward_dynamics",
            num_frames=17,
            image_path="observation.png",
        )

        params._adjust(_cosmos3_server_args())

        self.assertEqual(params.data_type, DataType.VIDEO)


class TestCosmos3ActionEndpoint(unittest.TestCase):
    @staticmethod
    def _policy_payload(
        batch_size=2,
        prompt="pick up the block",
        *,
        tensor_payload=False,
    ):
        images = torch.zeros(batch_size, 8, 8, 3, dtype=torch.uint8).numpy()
        input_reference = (
            {
                "dtype": "uint8",
                "shape": list(images.shape),
                "values": images.tolist(),
            }
            if tensor_payload
            else images
        )
        return {
            "input": {"prompt": prompt, "input_reference": input_reference},
            "parameters": {
                "action_mode": "policy",
                "domain_name": "droid_lerobot",
            },
        }

    def test_policy_request_builds_action_sampling_params(self):
        image = torch.zeros(8, 8, 3, dtype=torch.uint8).numpy()
        payload = {
            "request_id": "cosmos-action-1",
            "input": {
                "task": "pick up the block",
                "observation": {
                    "image": {
                        "dtype": "uint8",
                        "shape": [8, 8, 3],
                        "values": image.tolist(),
                    }
                },
            },
            "parameters": {
                "action_mode": "policy",
                "action_horizon": 16,
                "domain_name": "droid_lerobot",
                "num_inference_steps": 30,
                "height": 480,
                "width": 832,
                "fps": 5,
                "seed": 7,
            },
        }

        params = build_action_sampling_params(payload, _cosmos3_server_args())

        self.assertIsInstance(params, Cosmos3SamplingParams)
        self.assertEqual(params.data_type, DataType.ACTION)
        self.assertEqual(params.prompt, "pick up the block")
        self.assertEqual(params.action_mode, "policy")
        self.assertEqual(params.domain_name, "droid_lerobot")
        self.assertEqual(params.num_frames, 17)
        self.assertEqual(params.num_inference_steps, 30)
        self.assertEqual(params.seed, 7)
        self.assertEqual(params.image_path.size, (8, 8))

    def test_batched_policy_request_preserves_input_pairing(self):
        payload = self._policy_payload(
            prompt=["pick up the block", "close the drawer"],
            tensor_payload=True,
        )

        params = build_action_sampling_params(
            payload, _cosmos3_server_args(batching_max_size=2)
        )

        self.assertEqual(params.prompt, ["pick up the block", "close the drawer"])
        self.assertEqual(len(params.image_path), 2)
        self.assertTrue(
            all(isinstance(image, Image.Image) for image in params.image_path)
        )

    def test_batched_policy_request_broadcasts_scalar_prompt(self):
        params = build_action_sampling_params(
            self._policy_payload(), _cosmos3_server_args(batching_max_size=2)
        )

        self.assertEqual(params.prompt, ["pick up the block"] * 2)

    def test_batched_policy_request_rejects_cardinality_mismatch(self):
        with self.assertRaisesRegex(ValueError, "one prompt per image"):
            build_action_sampling_params(
                self._policy_payload(prompt=["one", "two", "three"]),
                _cosmos3_server_args(batching_max_size=3),
            )

    def test_batched_policy_request_honors_server_batch_limit(self):
        with self.assertRaisesRegex(ValueError, "--batching-max-size=1"):
            build_action_sampling_params(self._policy_payload(), _cosmos3_server_args())

    def test_single_item_image_batch_keeps_single_input_contract(self):
        params = build_action_sampling_params(
            self._policy_payload(batch_size=1, prompt="pick"),
            _cosmos3_server_args(),
        )

        self.assertEqual(params.prompt, "pick")
        self.assertIsInstance(params.image_path, Image.Image)

    def test_inverse_dynamics_maps_video_input(self):
        payload = {
            "input": {
                "task": "infer the robot motion",
                "observation": {"video": "observation.mp4"},
            },
            "parameters": {
                "action_mode": "inverse_dynamics",
                "num_frames": 61,
                "domain_name": "av",
            },
        }

        params = build_action_sampling_params(payload, _cosmos3_server_args())

        self.assertEqual(params.data_type, DataType.ACTION)
        self.assertEqual(params.video_path, "observation.mp4")
        self.assertEqual(params.num_frames, 61)

    def test_inverse_dynamics_rejects_prompt_batch(self):
        payload = {
            "input": {
                "prompt": ["first", "second"],
                "video": "observation.mp4",
            },
            "parameters": {
                "action_mode": "inverse_dynamics",
                "domain_name": "droid_lerobot",
            },
        }

        with self.assertRaisesRegex(ValueError, "prompt must be a string"):
            build_action_sampling_params(payload, _cosmos3_server_args())

    def test_forward_dynamics_is_rejected_by_action_endpoint(self):
        payload = {
            "input": {
                "task": "predict the next frames",
                "observation": {"image": "observation.png"},
            },
            "parameters": {"action_mode": "forward_dynamics"},
        }

        with self.assertRaisesRegex(ValueError, "/v1/videos"):
            build_action_sampling_params(payload, _cosmos3_server_args())

    def test_metadata_describes_cosmos_action_contract(self):
        metadata = action_metadata(_cosmos3_server_args(batching_max_size=4))

        self.assertEqual(metadata["model"], "cosmos3-production")
        self.assertEqual(metadata["policy_family"], "cosmos3")
        self.assertEqual(metadata["input"]["modalities"], ["image", "video"])
        self.assertEqual(metadata["output"]["action_horizon"], 16)
        self.assertEqual(metadata["output"]["padded_action_dim"], 64)
        self.assertFalse(metadata["capabilities"]["openpi_websocket"])
        self.assertTrue(metadata["capabilities"]["batch_inputs"])
        self.assertEqual(metadata["capabilities"]["max_batch_size"], 4)
        self.assertEqual(metadata["capabilities"]["batched_action_modes"], ["policy"])

    def test_metadata_keeps_batching_opt_in(self):
        metadata = action_metadata(_cosmos3_server_args())

        self.assertFalse(metadata["capabilities"]["batch_inputs"])
        self.assertEqual(metadata["capabilities"]["max_batch_size"], 1)

    def test_action_response_includes_cosmos_metadata(self):
        output = {
            "request_id": "cosmos-action-2",
            "actions": torch.zeros(16, 10).numpy(),
            "action_mode": "policy",
            "domain_id": 8,
            "raw_action_dim": 10,
            "parameters": {"num_inference_steps": 30},
        }

        response = action_generation_response(output, _cosmos3_server_args())
        action = response["data"][0]["action"]

        self.assertEqual(action["shape"], [16, 10])
        self.assertEqual(action["action_mode"], "policy")
        self.assertEqual(action["domain_id"], 8)
        self.assertEqual(action["raw_action_dim"], 10)
        self.assertEqual(response["usage"]["denoise_steps"], 30)

    def test_batched_action_response_emits_one_data_item_per_input(self):
        output = {
            "request_id": "cosmos-action-batch",
            "actions": torch.arange(24, dtype=torch.float32).reshape(2, 4, 3).numpy(),
            "action_mode": "policy",
            "domain_id": 8,
            "raw_action_dim": 3,
        }

        response = action_generation_response(output, _cosmos3_server_args())

        self.assertEqual(len(response["data"]), 2)
        self.assertEqual(response["data"][0]["action"]["shape"], [4, 3])
        self.assertEqual(response["data"][1]["input_index"], 1)
        self.assertEqual(response["usage"]["batch_size"], 2)
        self.assertEqual(response["usage"]["action_horizon"], 4)
        self.assertEqual(response["usage"]["action_dim"], 3)

    def test_action_response_rejects_empty_batch(self):
        output = {
            "request_id": "cosmos-action-empty",
            "actions": torch.empty(0, 4, 3).numpy(),
        }

        with self.assertRaisesRegex(ValueError, "dimensions must be non-zero"):
            action_generation_response(output, _cosmos3_server_args())

    def test_action_decode_skips_vae(self):
        class FailIfDecoded:
            def decode(self, _latents):
                raise AssertionError("VAE decode must not run for action output")

        stage = Cosmos3DecodingStage.__new__(Cosmos3DecodingStage)
        stage.vae = FailIfDecoded()
        stage.sound_tokenizer = None
        stage._guardrails = False
        stage.log_info = lambda *_args, **_kwargs: None
        batch = types.SimpleNamespace(
            data_type=DataType.ACTION,
            action_latents=torch.arange(24, dtype=torch.float32).reshape(1, 4, 6),
            extra={
                "raw_action_dim": 3,
                "action_domain_ids": torch.tensor([8]),
            },
            sampling_params=Cosmos3SamplingParams(
                prompt="test",
                action_mode="policy",
                domain_name="droid_lerobot",
            ),
            request_id="cosmos-action-3",
            num_inference_steps=30,
            num_frames=5,
            metrics=None,
        )

        output = stage.forward(batch, types.SimpleNamespace(vae_cpu_offload=False))

        self.assertEqual(output.output[0]["actions"].shape, (4, 3))
        self.assertEqual(output.output[0]["domain_id"], 8)
        self.assertEqual(output.action_pred.shape, (1, 4, 3))

    def test_batched_action_decode_keeps_batch_dimension(self):
        stage = Cosmos3DecodingStage.__new__(Cosmos3DecodingStage)
        stage.vae = mock.Mock()
        stage.sound_tokenizer = None
        stage._guardrails = False
        stage.log_info = lambda *_args, **_kwargs: None
        batch = types.SimpleNamespace(
            data_type=DataType.ACTION,
            action_latents=torch.arange(48, dtype=torch.float32).reshape(2, 4, 6),
            extra={
                "raw_action_dim": 3,
                "action_domain_ids": torch.tensor([8, 8]),
            },
            sampling_params=Cosmos3SamplingParams(
                prompt=["one", "two"],
                action_mode="policy",
                domain_name="droid_lerobot",
            ),
            request_id="cosmos-action-batch",
            num_inference_steps=30,
            num_frames=5,
            metrics=None,
        )

        output = stage.forward(batch, types.SimpleNamespace(vae_cpu_offload=False))

        self.assertEqual(output.output[0]["actions"].shape, (2, 4, 3))
        self.assertEqual(output.action_pred.shape, (2, 4, 3))


class TestCosmos3ModelResolution(unittest.TestCase):
    """Verify Cosmos3 checkpoints resolve to the native SGLang pipeline."""

    def test_nano_architecture_detection_does_not_depend_on_model_path(self):
        cases = (
            (
                {
                    "hidden_size": 4096,
                    "num_hidden_layers": 36,
                    "num_attention_heads": 32,
                },
                True,
            ),
            (
                {
                    "hidden_size": 5120,
                    "num_hidden_layers": 64,
                    "num_attention_heads": 64,
                },
                False,
            ),
            (
                {
                    "hidden_size": 2048,
                    "num_hidden_layers": 28,
                    "num_attention_heads": 16,
                },
                False,
            ),
        )
        for index, (transformer_config, expected) in enumerate(cases):
            with self.subTest(transformer_config=transformer_config):
                is_nano_checkpoint.cache_clear()
                with mock.patch(
                    "sglang.multimodal_gen.configs.pipeline_configs.cosmos3._transformer_config",
                    return_value=transformer_config,
                ):
                    self.assertEqual(
                        is_nano_checkpoint(f"/models/checkpoint-{index}"), expected
                    )

    def test_hf_checkpoint_uses_registered_native_pipeline_config(self):
        for model_path in (
            "nvidia/Cosmos3-Nano",
            "nvidia/Cosmos3-Nano-Policy-DROID",
            "nvidia/Cosmos3-Super",
            "nvidia/Cosmos3-Super-Text2Image",
            "nvidia/Cosmos3-Super-Image2Video",
            "nvidia/Cosmos3-Edge",
        ):
            with self.subTest(model_path=model_path):
                self.assertIsNone(get_non_diffusers_pipeline_name(model_path))
                config_info = _get_config_info(model_path)
                self.assertIsNotNone(config_info)
                self.assertIs(config_info.sampling_param_cls, Cosmos3SamplingParams)
                self.assertIs(config_info.pipeline_config_cls, Cosmos3Config)

    def test_class_name_detection_matches_legacy_and_new(self):
        """Unregistered checkpoints resolve via ``_class_name``: both the legacy
        ``Cosmos3OmniDiffusersPipeline`` and the current ``Cosmos3OmniPipeline``
        map to the native Cosmos3 config."""
        for idx, class_name in enumerate(
            ("Cosmos3OmniDiffusersPipeline", "Cosmos3OmniPipeline")
        ):
            model_path = f"acme/mystery-ckpt-{idx}"
            with self.subTest(class_name=class_name):
                with mock.patch(
                    "sglang.multimodal_gen.registry.maybe_download_model_index",
                    return_value={"_class_name": class_name},
                ):
                    config_info = _get_config_info(model_path)
                self.assertIsNotNone(config_info)
                self.assertIs(config_info.pipeline_config_cls, Cosmos3Config)

    def test_legacy_and_new_pipeline_names_both_registered(self):
        """Both ``_class_name`` spellings resolve to a native pipeline class so
        old (Nano/Super) and new (Edge) checkpoints load."""
        _discover_and_register_pipelines()
        for pipeline_name in ("Cosmos3OmniPipeline", "Cosmos3OmniDiffusersPipeline"):
            with self.subTest(pipeline_name=pipeline_name):
                self.assertIn(pipeline_name, _PIPELINE_REGISTRY)


class TestCosmos3OpenAIProtocol(unittest.TestCase):
    """Verify Cosmos3 modality knobs stay model-specific video extras."""

    def test_cosmos3_template_fields_remain_extra_fields(self):
        base_fields = {field.name for field in dataclasses.fields(SamplingParams)}
        cosmos_fields = {
            field.name for field in dataclasses.fields(Cosmos3SamplingParams)
        }
        for request_cls in (ImageGenerationsRequest, VideoGenerationsRequest):
            with self.subTest(request_cls=request_cls.__name__):
                self.assertIn("max_sequence_length", request_cls.model_fields)
                self.assertIn("flow_shift", request_cls.model_fields)
                self.assertNotIn("use_duration_template", request_cls.model_fields)
                self.assertNotIn("use_resolution_template", request_cls.model_fields)
                self.assertNotIn("use_system_prompt", request_cls.model_fields)
                self.assertNotIn("use_guardrails", request_cls.model_fields)
        for field_name in (
            "sound_duration",
            "use_duration_template",
            "use_resolution_template",
            "use_system_prompt",
            "use_guardrails",
        ):
            self.assertNotIn(field_name, base_fields)
            self.assertIn(field_name, cosmos_fields)

    def test_cosmos3_modal_fields_are_model_specific_video_extras(self):
        for field_name in (
            "generate_sound",
            "sound_duration",
            "condition_frame_indexes",
            "condition_frame_indexes_vision",
            "condition_video_keep",
            "control_path",
            "control_hint",
            "control_guidance",
            "control_guidance_interval",
            "num_video_frames_per_chunk",
            "num_conditional_frames",
            "num_first_chunk_conditional_frames",
            "max_frames",
            "show_control_condition",
            "show_input",
            "share_vision_temporal_positions",
            "action_mode",
            "domain_id",
            "domain_name",
            "raw_action_dim",
            "action_fps",
            "action",
            "action_view_point",
            "action_normalization",
        ):
            with self.subTest(field_name=field_name):
                self.assertNotIn(field_name, VideoGenerationsRequest.model_fields)
                self.assertIn(
                    field_name, Cosmos3SamplingParams.video_request_extra_fields()
                )

        self.assertIn("video_path", VideoGenerationsRequest.model_fields)
        self.assertIn("video_url", VideoGenerationsRequest.model_fields)
        self.assertNotIn("action_stats_path", VideoGenerationsRequest.model_fields)
        self.assertNotIn(
            "action_stats_path", Cosmos3SamplingParams.video_request_extra_fields()
        )

    def test_cosmos3_http_aliases_map_to_sampling_params(self):
        req = VideoGenerationsRequest(
            prompt="test",
            video_url="https://example.com/input.mp4",
            generate_sound=True,
            condition_frame_indexes_vision=[0, 2],
            condition_video_keep="last",
            action_mode="policy",
            domain_name="umi",
            raw_action_dim=9,
            action_fps=30.0,
            action_view_point="ego_view",
            control_path=["edge.mp4", "depth.mp4"],
            control_hint=["edge", "depth"],
            control_guidance=1.5,
            control_guidance_interval=[0.0, 500.0],
            num_video_frames_per_chunk=97,
            num_conditional_frames=5,
            num_first_chunk_conditional_frames=2,
            max_frames=1200,
            show_control_condition="true",
            show_input="false",
            share_vision_temporal_positions="false",
        )

        self.assertEqual(_resolve_video_path(req), "https://example.com/input.mp4")

        kwargs = request_model_kwargs(req, Cosmos3SamplingParams, "video")
        kwargs.update(num_frames=48, fps=24)
        kwargs = Cosmos3SamplingParams.lower_video_request_kwargs(req, kwargs)
        self.assertEqual(kwargs["sound_duration"], 2.0)
        self.assertEqual(kwargs["condition_frame_indexes"], [0, 2])
        self.assertEqual(kwargs["condition_video_keep"], "last")
        self.assertEqual(kwargs["action_mode"], "policy")
        self.assertEqual(kwargs["domain_name"], "umi")
        self.assertEqual(kwargs["raw_action_dim"], 9)
        self.assertEqual(kwargs["action_fps"], 30.0)
        self.assertEqual(kwargs["action_view_point"], "ego_view")
        self.assertEqual(kwargs["control_path"], ["edge.mp4", "depth.mp4"])
        self.assertEqual(kwargs["control_hint"], ["edge", "depth"])
        self.assertEqual(kwargs["control_guidance"], 1.5)
        self.assertEqual(kwargs["control_guidance_interval"], (0.0, 500.0))
        self.assertEqual(kwargs["num_video_frames_per_chunk"], 97)
        self.assertEqual(kwargs["num_conditional_frames"], 5)
        self.assertEqual(kwargs["num_first_chunk_conditional_frames"], 2)
        self.assertEqual(kwargs["max_frames"], 1200)
        self.assertTrue(kwargs["show_control_condition"])
        self.assertFalse(kwargs["show_input"])
        self.assertFalse(kwargs["share_vision_temporal_positions"])

    def test_cosmos3_multipart_extras_are_model_specific(self):
        raw_form = {
            "generate_sound": "true",
            "control_path": '["edge.mp4", "depth.mp4"]',
            "control_hint": '["edge", "depth"]',
            "action_mode": "policy",
            "action_stats_path": "/tmp/action_stats.json",
        }

        generic = _multipart_video_extras(
            raw_form,
            extra_body=None,
            extra_params=None,
            sampling_params_cls=SamplingParams,
        )
        self.assertNotIn("generate_sound", generic)
        self.assertNotIn("control_path", generic)
        self.assertNotIn("action_mode", generic)

        cosmos = _multipart_video_extras(
            raw_form,
            extra_body=None,
            extra_params=None,
            sampling_params_cls=Cosmos3SamplingParams,
        )
        self.assertIs(cosmos["generate_sound"], True)
        self.assertEqual(cosmos["control_path"], ["edge.mp4", "depth.mp4"])
        self.assertEqual(cosmos["control_hint"], ["edge", "depth"])
        self.assertEqual(cosmos["action_mode"], "policy")
        self.assertNotIn("action_stats_path", cosmos)

    def test_generate_sound_false_disables_sound_duration(self):
        req = VideoGenerationsRequest(
            prompt="test", generate_sound=False, sound_duration=3.0
        )
        kwargs = request_model_kwargs(req, Cosmos3SamplingParams, "video")
        kwargs.update(num_frames=48, fps=24)
        kwargs = Cosmos3SamplingParams.lower_video_request_kwargs(req, kwargs)
        self.assertEqual(kwargs["sound_duration"], 0.0)


class TestCosmos3Guardrails(unittest.TestCase):
    """Verify optional guardrail dependency handling."""

    def setUp(self):
        is_cosmos_guardrail_available.cache_clear()

    def tearDown(self):
        is_cosmos_guardrail_available.cache_clear()

    def test_guardrail_availability_matches_package_spec(self):
        self.assertEqual(
            is_cosmos_guardrail_available(),
            importlib.util.find_spec("cosmos_guardrail") is not None,
        )

    @mock.patch("importlib.util.find_spec", return_value=None)
    def test_missing_guardrail_package_reports_unavailable(self, _):
        self.assertFalse(is_cosmos_guardrail_available())


_COSMOS3_STAGES = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3"
)


class _FrameVAE(torch.nn.Module):
    """Decodes one latent frame at a time and hands each to ``on_frames``."""

    supports_decode_on_frames = True

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = types.SimpleNamespace(scaling_factor=1.0)

    def decode(self, z, on_frames=None):
        out = torch.tanh(3 * z[:, :3])
        if on_frames is not None:
            for t in range(out.shape[2]):
                on_frames(out[:, :, t : t + 1].clone())
        return out.clone()


class _RecordingStream:
    def __init__(self, save_file_path, batch, finish_frames=None):
        self.save_file_path = save_file_path
        self.finish_frames = finish_frames
        self.frames = []

    def __call__(self, frames):
        self.frames.append(self.finish_frames(frames))

    def finish(self):
        return [self.save_file_path]

    def abort(self):
        raise AssertionError("the decode must not fail")


class TestCosmos3EncodeWhileDecoding(unittest.TestCase):
    """The plain video decode feeds the MP4 encoder the frames it would save."""

    @staticmethod
    def _stage():
        stage = Cosmos3DecodingStage.__new__(Cosmos3DecodingStage)
        stage.vae = _FrameVAE()
        stage.sound_tokenizer = None
        stage._guardrails = False
        stage._latents_mean = stage._latents_std = None
        stage.log_info = stage.log_debug = lambda *_args, **_kwargs: None
        stage.use_declared_component = lambda **_kwargs: contextlib.nullcontext()
        return stage

    @staticmethod
    def _batch(**overrides):
        fields = dict(
            data_type=DataType.VIDEO,
            latents=torch.randn(
                1, 4, 5, 6, 8, generator=torch.Generator().manual_seed(0)
            ),
            extra={},
            sampling_params=Cosmos3SamplingParams(prompt="a robot"),
            action_latents=None,
            audio_latents=None,
            use_guardrails=None,
            metrics=None,
            rollout_trajectory_data=None,
        )
        fields.update(overrides)
        return types.SimpleNamespace(**fields)

    def test_streamed_frames_equal_the_saved_output(self):
        stage = self._stage()
        with mock.patch(
            f"{_COSMOS3_STAGES}.streamed_video_output_path", return_value=None
        ):
            reference = stage.forward(self._batch(), types.SimpleNamespace())
        streams = []

        def open_stream(*args, **kwargs):
            streams.append(_RecordingStream(*args, **kwargs))
            return streams[-1]

        with (
            mock.patch(
                f"{_COSMOS3_STAGES}.streamed_video_output_path",
                return_value="/tmp/cosmos3.mp4",
            ),
            mock.patch(
                f"{_COSMOS3_STAGES}.EncodeWhileDecoding", side_effect=open_stream
            ),
        ):
            streamed = stage.forward(self._batch(), types.SimpleNamespace())

        self.assertIsNone(streamed.output)
        self.assertEqual(streamed.output_file_paths, ["/tmp/cosmos3.mp4"])
        self.assertIsNone(reference.output_file_paths)
        self.assertTrue(
            torch.equal(torch.cat(streams[0].frames, dim=2), reference.output)
        )

    def test_outputs_that_need_the_frames_are_not_streamed(self):
        stage = self._stage()
        cases = {
            "guardrails": (dict(_guardrails=True), {}),
            "audio": (
                dict(sound_tokenizer=object()),
                dict(audio_latents=torch.zeros(1)),
            ),
            "action": ({}, dict(action_latents=torch.zeros(1, 2, 3))),
            "transfer": ({}, dict(extra={"transfer_plan": {"total_frames": 5}})),
        }
        with mock.patch(
            f"{_COSMOS3_STAGES}.streamed_video_output_path",
            return_value="/tmp/cosmos3.mp4",
        ):
            self.assertIsNotNone(
                stage._encode_while_decoding(self._batch(), types.SimpleNamespace())
            )
            for name, (stage_fields, batch_fields) in cases.items():
                with self.subTest(name):
                    case_stage = self._stage()
                    vars(case_stage).update(stage_fields)
                    self.assertIsNone(
                        case_stage._encode_while_decoding(
                            self._batch(**batch_fields), types.SimpleNamespace()
                        )
                    )


class TestCosmos3DecodeFastPath(unittest.TestCase):
    """The VAE decode runs on the fast path exactly when the quality allows it."""

    def test_fast_path_follows_request_quality(self):
        class _VAE(torch.nn.Module):
            supports_decode_on_frames = False

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.config = types.SimpleNamespace(scaling_factor=1.0)

            def decode(self, z):
                return z[:, :3].clamp(-1, 1)

        stage = Cosmos3DecodingStage.__new__(Cosmos3DecodingStage)
        stage.vae = _VAE()
        stage.sound_tokenizer = None
        stage._guardrails = False
        stage._latents_mean = stage._latents_std = None
        stage.log_info = stage.log_debug = lambda *_args, **_kwargs: None
        stage.use_declared_component = lambda **_kwargs: contextlib.nullcontext()
        gates = []

        @contextlib.contextmanager
        def record_gate(vae, enabled):
            gates.append((vae, enabled))
            yield

        for quality in ("lossless", "high", "exact"):
            batch = types.SimpleNamespace(
                data_type=DataType.VIDEO,
                latents=torch.zeros(1, 4, 2, 3, 3),
                extra={},
                sampling_params=Cosmos3SamplingParams(prompt="a", quality=quality),
                action_latents=None,
                audio_latents=None,
                metrics=None,
                rollout_trajectory_data=None,
            )
            with mock.patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages."
                "model_specific_stages.cosmos3.use_vae_fast_path",
                record_gate,
            ):
                stage.forward(batch, types.SimpleNamespace())
        self.assertEqual(
            gates, [(stage.vae, True), (stage.vae, True), (stage.vae, False)]
        )


class TestCosmos3MRoPE(unittest.TestCase):
    """mRoPE position-ID computation for vision / sound / action token grids."""

    DEVICE = torch.device("cpu")

    def test_vision_default_args_unchanged(self):
        # With base_temporal_compression_factor=None and start_frame_offset=0
        # the token and base rates cancel, so t-index == frame index.
        pos, _ = compute_mrope_position_ids_vision(
            grid_t=21,
            grid_h=1,
            grid_w=1,
            temporal_offset=0,
            device=self.DEVICE,
            fps=24.0,
            base_fps=24.0,
            temporal_compression_factor=4,
        )
        self.assertEqual(tuple(pos.shape), (3, 21))
        self.assertAlmostEqual(float(pos[0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(pos[0, 20]), 20.0, places=5)

    def test_sound_grid_shape_and_scaling(self):
        pos, _ = compute_mrope_position_ids_sound(
            grid_t=10,
            temporal_offset=0,
            sound_latent_fps=25.0,
            device=self.DEVICE,
            base_fps=24.0,
            temporal_compression_factor_sound=1,
        )
        # (T, 1, 1) grid -> spatial axes are all zero.
        self.assertEqual(tuple(pos.shape), (3, 10))
        self.assertTrue(torch.all(pos[1] == 0))
        self.assertTrue(torch.all(pos[2] == 0))
        # t-index = i / sound_fps * base_fps = i / 25 * 24.
        self.assertAlmostEqual(float(pos[0, 5]), 5 / 25 * 24, places=4)

    def test_action_uses_video_base_compression_and_offset(self):
        # Action runs at frame rate (tcf=1) but is scaled by the video's
        # base_temporal_compression_factor=4, and shifted by start_frame_offset.
        pos, _ = compute_mrope_position_ids_action(
            grid_t=16,
            temporal_offset=0,
            action_fps=10.0,
            device=self.DEVICE,
            base_fps=24.0,
            base_temporal_compression_factor=4,
            start_frame_offset=1,
        )
        self.assertEqual(tuple(pos.shape), (3, 16))
        self.assertTrue(torch.all(pos[1] == 0))
        self.assertTrue(torch.all(pos[2] == 0))
        # t-index[i] = (i + start_frame_offset) / action_fps * (base_fps / base_tcf)
        #            = (i + 1) / 10 * (24 / 4) = (i + 1) * 0.6
        self.assertAlmostEqual(float(pos[0, 0]), 0.6, places=4)
        self.assertAlmostEqual(float(pos[0, 15]), 16 * 0.6, places=4)

    def test_action_offset_zero(self):
        pos, _ = compute_mrope_position_ids_action(
            grid_t=8,
            temporal_offset=0,
            action_fps=10.0,
            device=self.DEVICE,
            base_fps=24.0,
            base_temporal_compression_factor=4,
            start_frame_offset=0,
        )
        self.assertAlmostEqual(float(pos[0, 0]), 0.0, places=5)

    def test_action_aligns_with_video_positions(self):
        # Action frames at frame rate should share the video's temporal frame:
        # every 4th action token lands on the next video latent-frame position.
        media_offset = 100
        vid, _ = compute_mrope_position_ids_vision(
            grid_t=5,
            grid_h=1,
            grid_w=1,
            temporal_offset=media_offset,
            device=self.DEVICE,
            fps=24.0,
            base_fps=24.0,
            temporal_compression_factor=4,
        )
        act, _ = compute_mrope_position_ids_action(
            grid_t=16,
            temporal_offset=media_offset,
            action_fps=24.0,
            device=self.DEVICE,
            base_fps=24.0,
            base_temporal_compression_factor=4,
            start_frame_offset=0,
        )
        # video latent frame 1 sits at media_offset+1; action frame 4 (4 frames
        # per latent at tcf=4) lands at the same temporal position.
        self.assertAlmostEqual(float(vid[0, 1]), float(act[0, 4]), places=4)


class TestCosmos3Transfer(unittest.TestCase):
    """Transfer (control-video) conditioning: rope packing, control-CFG, defaults."""

    DEVICE = torch.device("cpu")

    @staticmethod
    def _transformer_self() -> Cosmos3OmniTransformer:
        model = Cosmos3OmniTransformer.__new__(Cosmos3OmniTransformer)
        model.temporal_margin = 15000
        model.base_fps = 24.0
        model.temporal_compression_factor = 4
        model.sound_latent_fps = 25.0
        model.temporal_compression_factor_sound = 1
        return model

    def test_single_control_shares_video_positions(self):
        model = self._transformer_self()
        text_mask = torch.ones(1, 4)
        T, Hp, Wp = 3, 2, 2
        tpc = T * Hp * Wp

        _, base = model._compute_rope_position_ids(
            text_mask, T, Hp, Wp, fps=None, device=self.DEVICE, control_frames=0
        )
        _, with_ctrl = model._compute_rope_position_ids(
            text_mask, T, Hp, Wp, fps=None, device=self.DEVICE, control_frames=T
        )
        self.assertEqual(tuple(with_ctrl.shape), (3, 1, 2 * tpc))
        # control prefix == video block, and video block is unchanged.
        self.assertTrue(torch.equal(with_ctrl[:, :, :tpc], with_ctrl[:, :, tpc:]))
        self.assertTrue(torch.equal(with_ctrl[:, :, tpc:], base))

    def test_multi_control_prepended_in_order(self):
        model = self._transformer_self()
        text_mask = torch.ones(1, 4)
        T, Hp, Wp = 3, 2, 2
        tpc = T * Hp * Wp

        _, base = model._compute_rope_position_ids(
            text_mask, T, Hp, Wp, fps=None, device=self.DEVICE, control_frames=0
        )
        _, multi = model._compute_rope_position_ids(
            text_mask, T, Hp, Wp, fps=None, device=self.DEVICE, control_frames=[T, T]
        )
        self.assertEqual(tuple(multi.shape), (3, 1, 3 * tpc))
        c0 = multi[:, :, :tpc]
        c1 = multi[:, :, tpc : 2 * tpc]
        vid = multi[:, :, 2 * tpc :]
        self.assertTrue(torch.equal(c0, vid))
        self.assertTrue(torch.equal(c1, vid))
        self.assertTrue(torch.equal(vid, base))

    def test_control_positions_can_be_sequential(self):
        model = self._transformer_self()
        text_mask = torch.ones(1, 4)
        T, Hp, Wp = 3, 1, 1

        _, positions = model._compute_rope_position_ids(
            text_mask,
            T,
            Hp,
            Wp,
            fps=None,
            device=self.DEVICE,
            control_frames=[T, T],
            share_vision_temporal_positions=False,
        )
        c0 = positions[0, 0, :T]
        c1 = positions[0, 0, T : 2 * T]
        video = positions[0, 0, 2 * T :]
        self.assertLess(c0.max().item(), c1.min().item())
        self.assertLess(c1.max().item(), video.min().item())

    def test_transfer_chunk_count_and_reflection_padding(self):
        self.assertEqual(
            Cosmos3ImagePreprocessStage._get_transfer_num_chunks(93, 93, 1),
            (1, 93),
        )
        self.assertEqual(
            Cosmos3ImagePreprocessStage._get_transfer_num_chunks(186, 93, 1),
            (3, 92),
        )
        frames = torch.tensor([0, 1, 2], dtype=torch.uint8).view(1, 1, 3, 1, 1)
        frames = frames.expand(1, 3, 3, 1, 1)
        padded = _pad_transfer_frames(frames, 5)
        self.assertEqual(padded[0, 0, :, 0, 0].tolist(), [0, 1, 2, 2, 1])

    def test_transfer_resize_antialiases_and_rounds_crop_like_imaginaire4(self):
        """Controls are fit like imaginaire4's ``_resize_and_center_crop``.

        Plain bilinear samples single columns of a 3x downscale (returning
        one-pixel stripes as 0/255), and a floored crop offset shifts odd
        margins by a row; both changed the control tokens of 1080p WSM clips.
        """
        stripes = torch.zeros(2, 3, 6, 18, dtype=torch.uint8)
        stripes[..., 1::2] = 255
        resized = resize_center_crop_uint8(stripes, 2, 6)
        self.assertEqual(tuple(resized.shape), (2, 3, 2, 6))
        self.assertEqual(resized.dtype, torch.uint8)
        self.assertTrue(100 <= resized.min() and resized.max() <= 155)

        # 7 rows cropped to 4: torchvision rounds the 1.5-row offset to 2.
        rows = torch.arange(0, 70, 10, dtype=torch.uint8).reshape(1, 1, 7, 1)
        cropped = resize_center_crop_uint8(rows.expand(1, 3, 7, 4).contiguous(), 4, 4)
        self.assertEqual(cropped[0, 0, :, 0].tolist(), [20, 30, 40, 50])

    def test_transfer_chunks_stitch_without_duplicate_overlap(self):
        class Scheduler:
            def __init__(self):
                self.timesteps = torch.tensor([])
                self.calls = 0

            def set_timesteps(self, steps, device):
                self.calls += 1
                self.timesteps = torch.arange(steps, device=device)

        stage = Cosmos3DenoisingStage.__new__(Cosmos3DenoisingStage)
        stage.vae = torch.nn.Linear(1, 1, bias=False)
        stage.transformer = torch.nn.Linear(1, 1, bias=False)
        stage.scheduler = Scheduler()
        stage._prepare_transfer_chunk = mock.Mock(side_effect=[0, 1])
        stage._denoise_once = mock.Mock(side_effect=lambda batch, *_a, **_k: batch)
        stage._decode_transfer_latents = mock.Mock(
            side_effect=[
                torch.arange(5).view(1, 1, 5, 1, 1).float() / 10,
                torch.arange(5, 10).view(1, 1, 5, 1, 1).float() / 10,
            ]
        )
        generator = torch.Generator(device="cpu").manual_seed(7)
        batch = types.SimpleNamespace(
            latents=torch.zeros(1),
            generator=generator,
            seed=7,
            num_inference_steps=2,
            is_warmup=False,
            extra={
                "transfer_plan": {
                    "num_chunks": 2,
                    "total_frames": 9,
                }
            },
        )
        server_args = types.SimpleNamespace(vae_cpu_offload=False)
        stage.server_args = server_args

        stage._forward_transfer(batch, server_args)

        stitched = batch.extra["transfer_decoded_output"].flatten()
        self.assertTrue(
            torch.allclose(
                stitched,
                torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]),
            )
        )
        self.assertEqual(stage.scheduler.calls, 2)
        # imaginaire4 seeds chunk k with seed + k: chunk 0 keeps the request
        # generator, chunk 1 draws from a fresh seed-8 generator.
        prepare_gens = [c.args[3] for c in stage._prepare_transfer_chunk.call_args_list]
        denoise_gens = [
            c.kwargs["generator"] for c in stage._denoise_once.call_args_list
        ]
        self.assertIs(prepare_gens[0], generator)
        self.assertEqual(prepare_gens, denoise_gens)
        self.assertIsNot(prepare_gens[1], generator)
        expected = torch.Generator(device="cpu").manual_seed(8)
        self.assertTrue(
            torch.equal(
                torch.randn(4, generator=prepare_gens[1]),
                torch.randn(4, generator=expected),
            )
        )

    def test_transfer_display_composes_input_controls_and_output(self):
        output = torch.full((1, 3, 2, 1, 2), 0.5)
        control = torch.full((1, 3, 2, 1, 2), 255, dtype=torch.uint8)
        source = torch.zeros((1, 3, 2, 1, 2), dtype=torch.uint8)
        batch = types.SimpleNamespace(
            sampling_params=types.SimpleNamespace(
                show_control_condition=True,
                show_input=True,
            ),
            extra={
                "transfer_plan": {"total_frames": 2},
                "preprocessed_control": [control],
                "preprocessed_transfer_video": source,
            },
        )

        composed = Cosmos3DecodingStage._compose_transfer_display(output, batch)

        self.assertEqual(tuple(composed.shape), (1, 3, 2, 1, 6))
        self.assertTrue(torch.equal(composed[..., :2], torch.zeros_like(output)))
        self.assertTrue(torch.equal(composed[..., 2:4], torch.ones_like(output)))
        self.assertTrue(torch.equal(composed[..., 4:], output))

    def test_control_cfg_blend_math(self):
        stage = Cosmos3DenoisingStage.__new__(Cosmos3DenoisingStage)

        # Per-branch forward values (each run un-batched, bs=1): cond_full
        # (control in, cond text) -> 20; cond_nc (control dropped) -> 10; uncond
        # (control in, uncond text) -> 2. The unified executor reduces the
        # coefficient-weighted branch sum built by _control_cfg_branches.
        def fake_run(**kw):
            bs = kw["latents"].shape[0]
            if kw["control_latents"] is None:
                return torch.full((bs,), 10.0)  # cond_nc, control dropped
            if kw["cache_key"] == "uncond":
                return torch.full((bs,), 2.0)  # uncond, control in
            return torch.full((bs,), 20.0)  # cond_full, control in

        stage._run_transformer = fake_run
        cond_text_ids = torch.zeros(1)
        cond_text_mask = torch.ones(1)
        uncond_text_ids = torch.zeros(1)
        uncond_text_mask = torch.ones(1)
        control_latents = [torch.zeros(1)]

        def _run(text_g, control_g):
            branches = stage._control_cfg_branches(
                cond_text_ids,
                cond_text_mask,
                uncond_text_ids,
                uncond_text_mask,
                cond_text_seq_len=None,
                uncond_text_seq_len=None,
                control_latents=control_latents,
                text_guidance_scale=text_g,
                control_guidance_scale=control_g,
            )
            return stage._predict_noise_cfg(
                branches,
                latents=torch.zeros(1),
                timestep=torch.zeros(1),
                video_shape=(1, 1, 1),
                fps=24.0,
                cfg_rank=0,
                cfg_world_size=1,
            )

        # control-CFG only (g=1): cg*cond_full + (1-cg)*cond_nc
        #   = 2*20 + (-1)*10 = 30
        out = _run(text_g=1.0, control_g=2.0)
        self.assertTrue(torch.allclose(out, torch.full((1,), 30.0)))

        # control-CFG + text CFG (g=3), imaginaire4's executor: the no-control mix
        # uses eff = 1 + (cg-1)/g before the text blend, so the control increment
        # is (cg-1)*(full-nc), not g*(cg-1)*(full-nc):
        #   (g+cg-1)*cond_full + (1-cg)*cond_nc + (1-g)*uncond
        #   = 4*20 + (-1)*10 + (-2)*2 = 66
        out2 = _run(text_g=3.0, control_g=2.0)
        self.assertTrue(torch.allclose(out2, torch.full((1,), 66.0)))

        with self.assertRaisesRegex(ValueError, "text guidance scale 0.0"):
            _run(text_g=0.0, control_g=2.0)

    def test_text_cfg_batched_single_gpu(self):
        """Single-GPU text CFG batches both branches into one bs=2 forward."""
        stage = Cosmos3DenoisingStage.__new__(Cosmos3DenoisingStage)

        def fake_run(**kw):
            self.assertIsNone(kw["control_latents"])
            self.assertEqual(kw["latents"].shape[0], 2)  # batched [uncond, cond]
            return torch.tensor([2.0, 20.0])

        stage._run_transformer = fake_run
        out = stage._predict_noise_cfg_batched(
            latents=torch.zeros(1),
            timestep=torch.zeros(1),
            cond_text_ids=torch.zeros(1),
            cond_text_mask=torch.ones(1),
            uncond_text_ids=torch.zeros(1),
            uncond_text_mask=torch.ones(1),
            video_shape=(1, 1, 1),
            fps=24.0,
            guidance_scale=3.0,
        )
        # uncond + g*(cond - uncond) = 2 + 3*(20-2) = 56
        self.assertTrue(torch.allclose(out, torch.full((1,), 56.0)))

    def test_control_cfg_parallel_distribution(self):
        """Round-robin branches are all-gathered and combined in serial order."""
        stage = Cosmos3DenoisingStage.__new__(Cosmos3DenoisingStage)
        seen = {}

        def make_run(rank):
            seen[rank] = []

            def fake_run(**kw):
                seen[rank].append(kw["cache_key"])
                bs = kw["latents"].shape[0]
                if kw["control_latents"] is None:
                    return torch.full((bs,), 10.0)  # cond_nc
                if kw["cache_key"] == "uncond":
                    return torch.full((bs,), 2.0)  # uncond
                return torch.full((bs,), 20.0)  # cond_full

            return fake_run

        def run(rank, world):
            stage._run_transformer = make_run(rank)
            branches = stage._control_cfg_branches(
                torch.zeros(1),
                torch.ones(1),
                torch.zeros(1),
                torch.ones(1),
                cond_text_seq_len=None,
                uncond_text_seq_len=None,
                control_latents=[torch.zeros(1)],
                text_guidance_scale=3.0,
                control_guidance_scale=2.0,
            )
            return stage._predict_noise_cfg(
                branches,
                latents=torch.zeros(1),
                timestep=torch.zeros(1),
                video_shape=(1, 1, 1),
                fps=24.0,
                cfg_rank=rank,
                cfg_world_size=world,
            )

        target = (
            "sglang.multimodal_gen.runtime.pipelines_core.stages."
            "model_specific_stages.cosmos3.cfg_model_parallel_all_gather"
        )
        values = {"cond": 20.0, "cond_nc": 10.0, "uncond": 2.0}
        branch_order = ["cond", "cond_nc", "uncond"]
        sent = []

        def fake_gather(world):
            # Return what every rank would send for the same slot, so each
            # rank combines all branches as a real all-gather would let it.
            def gather(pred, dim, separate_tensors):
                sent.append(pred)
                slot = len(sent) - 1
                bids = [slot * world + r for r in range(world)]
                return [
                    (
                        torch.full_like(pred, values[branch_order[bid]])
                        if bid < len(branch_order)
                        else torch.zeros_like(pred)
                    )
                    for bid in bids
                ]

            return gather

        # 2 ranks: rank 0 runs the two control-in forwards (cond_full +
        # uncond), rank 1 runs only the control-dropped cond_nc.
        for rank in range(2):
            sent.clear()
            with mock.patch(target, side_effect=fake_gather(2)):
                out = run(rank, 2)
            self.assertTrue(torch.allclose(out, torch.full((1,), 66.0)))
        self.assertEqual(seen[0], ["cond", "uncond"])
        self.assertEqual(seen[1], ["cond_nc"])
        self.assertTrue(torch.equal(sent[1], torch.zeros(1)))

        seen.clear()
        # 4 ranks for 3 branches: rank 3 is idle and sends zeros.
        for rank in range(4):
            sent.clear()
            with mock.patch(target, side_effect=fake_gather(4)):
                out = run(rank, 4)
            self.assertTrue(torch.allclose(out, torch.full((1,), 66.0)))
        self.assertEqual(seen[0], ["cond"])
        self.assertEqual(seen[1], ["cond_nc"])
        self.assertEqual(seen[2], ["uncond"])
        self.assertEqual(seen[3], [])
        self.assertTrue(torch.equal(sent[0], torch.zeros(1)))

    @staticmethod
    def _transfer_params(explicit=(), **kwargs):
        sp = Cosmos3SamplingParams(prompt="t", **kwargs)
        sp._explicit_fields = {"prompt", *kwargs, *explicit}
        return sp

    def test_single_hint_defaults_applied(self):
        sp = self._transfer_params(control_path="edge.mp4", control_hint="edge")
        sp._apply_transfer_hint_defaults(sp.resolve_transfer_hints())
        self.assertEqual(sp.control_guidance, 1.5)
        self.assertEqual(sp.guidance_scale, 3.0)
        self.assertEqual(sp.flow_shift, 10.0)

    def test_wsm_uses_transfer_spec_defaults(self):
        for kwargs in (
            dict(control_path="wsm.mp4", control_hint="wsm"),
            dict(wsm={"control_path": "wsm.mp4"}),
        ):
            sp = self._transfer_params(**kwargs)
            sp._apply_transfer_hint_defaults(sp.resolve_transfer_hints())
            self.assertEqual(sp.guidance_scale, 1.0)
            self.assertEqual(sp.control_guidance, 3.0)
            self.assertEqual(sp.flow_shift, 10.0)
            self.assertEqual(sp.num_frames, 101)
            self.assertEqual(sp.fps, 10)
            self.assertEqual(sp.num_video_frames_per_chunk, 101)

        for extra in (
            {"control_path": "wsm.mp4", "control_hint": "wsm"},
            {"wsm": '{"control_path": "wsm.mp4"}'},
        ):
            lowered = Cosmos3SamplingParams.lower_video_request_kwargs(
                types.SimpleNamespace(num_frames=None, fps=None),
                {"num_frames": 121, "fps": 24, **extra},
            )
            self.assertEqual(lowered["num_frames"], 101)
            self.assertEqual(lowered["fps"], 10)

    def test_explicit_value_overrides_hint_default(self):
        sp = self._transfer_params(
            control_path="seg.mp4", control_hint="seg", control_guidance=4.2
        )
        sp._apply_transfer_hint_defaults(sp.resolve_transfer_hints())
        self.assertEqual(sp.control_guidance, 4.2)

    def test_multi_hint_defaults_not_applied(self):
        sp = self._transfer_params(
            control_path=["edge.mp4", "depth.mp4"], control_hint=["edge", "depth"]
        )
        base_guidance = sp.guidance_scale
        sp._apply_transfer_hint_defaults(sp.resolve_transfer_hints())
        self.assertEqual(sp.control_guidance, 1.0)
        self.assertEqual(sp.guidance_scale, base_guidance)

    def test_transfer_emphasis_and_length_defaults(self):
        """The emphasis names every hint, so it defaults on only for typed controls;
        num_frames caps the control-length output only when set by the request or
        by the WSM hint default."""
        sp = self._transfer_params(control_path="e.mp4", control_hint="edge")
        sp._apply_transfer_defaults(sp.resolve_transfer_hints())
        self.assertIs(sp.emphasize_control_in_prompt, True)
        self.assertEqual(sp.max_frames, 5000)

        sp = self._transfer_params(control_path="w.mp4", control_hint="wsm")
        sp._apply_transfer_defaults(sp.resolve_transfer_hints())
        self.assertEqual(sp.max_frames, 101)

        sp = self._transfer_params(
            control_path="e.mp4",
            control_hint="edge",
            emphasize_control_in_prompt=False,
            num_frames=121,
        )
        sp._apply_transfer_defaults(sp.resolve_transfer_hints())
        self.assertIs(sp.emphasize_control_in_prompt, False)
        self.assertEqual(sp.max_frames, 121)

        sp = self._transfer_params(control_path="a.mp4")
        sp._apply_transfer_defaults(sp.resolve_transfer_hints())
        self.assertIs(sp.emphasize_control_in_prompt, False)
        sp = self._transfer_params(
            control_path="a.mp4", emphasize_control_in_prompt=True
        )
        with self.assertRaisesRegex(ValueError, "names the control types"):
            sp._apply_transfer_defaults(sp.resolve_transfer_hints())

    def test_transfer_hints_resolve_in_imaginaire4_order(self):
        """Controls pack (and the emphasis names them) in TransferHintKey order."""
        sp = self._transfer_params(
            control_path=["d.mp4", "e.mp4"],
            control_hint=["depth", "edge"],
            control_weight=[3.0, 1.0],
        )
        self.assertEqual(
            sp.resolve_transfer_hints(),
            [
                Cosmos3TransferHint(key="edge", control_path="e.mp4", weight=1.0),
                Cosmos3TransferHint(key="depth", control_path="d.mp4", weight=3.0),
            ],
        )
        sp = self._transfer_params(
            wsm={"control_path": "w.mp4", "weight": 2},
            edge={"preset_edge_threshold": "high"},
        )
        self.assertEqual(
            sp.resolve_transfer_hints(),
            [
                Cosmos3TransferHint(key="edge", preset="high"),
                Cosmos3TransferHint(key="wsm", control_path="w.mp4", weight=2.0),
            ],
        )

    def test_transfer_request_contract_refusals(self):
        bad = [
            dict(control_path="x.mp4", control_hint="bogus"),
            dict(control_path=["a.mp4", "b.mp4"], control_hint=["edge", "edge"]),
            dict(control_path="a.mp4", control_hint="edge", control_weight=[1, 2]),
            dict(control_path="a.mp4", control_hint="edge", wsm={}),
            dict(edge={"preset_edge_threshold": "extreme"}),
            dict(blur={"preset_edge_threshold": "low"}),
            dict(depth={"weight": -1}),
        ]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Cosmos3SamplingParams(prompt="t", **kwargs)

        # Only edge/blur are computed on the fly, and only from video_path.
        for kwargs, message in (
            (dict(depth={}), "Missing pre-computed control input for 'depth'"),
            (dict(edge={}), "computed from video_path"),
        ):
            sp = self._transfer_params(**kwargs)
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(ValueError, message),
            ):
                sp._apply_transfer_defaults(sp.resolve_transfer_hints())
        sp = self._transfer_params(edge={}, video_path="in.mp4")
        sp._apply_transfer_defaults(sp.resolve_transfer_hints())

    def test_video_request_length_follows_control_unless_set(self):
        kwargs = {"num_frames": 96, "fps": 24, "depth": '{"control_path": "d.mp4"}'}
        request = types.SimpleNamespace(num_frames=None, seconds=None, model_extra={})
        lowered = Cosmos3SamplingParams.lower_video_request_kwargs(
            request, dict(kwargs)
        )
        self.assertNotIn("num_frames", lowered)
        self.assertEqual(lowered["depth"], {"control_path": "d.mp4"})
        request = types.SimpleNamespace(num_frames=96, seconds=None, model_extra={})
        lowered = Cosmos3SamplingParams.lower_video_request_kwargs(
            request, dict(kwargs)
        )
        self.assertEqual(lowered["num_frames"], 96)

    def test_prepare_transfer_plan_follows_control_length_and_source_fps(self):
        """Output length is the first control's (num_frames does not cap it), the
        source video sets the output fps, and an edge hint without a file is
        computed from the resized source."""
        source = torch.randint(0, 256, (3, 9, 16, 32), dtype=torch.uint8)
        depth = torch.randint(0, 256, (3, 7, 16, 32), dtype=torch.uint8)
        media = {"in.mp4": (source, 29.97), "depth.mp4": (depth, 30.0)}
        batch = types.SimpleNamespace(
            height=16,
            width=32,
            fps=24,
            num_frames=81,
            extra={},
            sampling_params=types.SimpleNamespace(
                max_frames=5000,
                num_video_frames_per_chunk=5,
                num_conditional_frames=1,
                num_first_chunk_conditional_frames=0,
            ),
        )
        hints = [
            Cosmos3TransferHint(key="edge", weight=1.0),
            Cosmos3TransferHint(key="depth", control_path="depth.mp4", weight=3.0),
        ]
        with mock.patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages."
            "model_specific_stages.cosmos3.read_transfer_media_uint8",
            side_effect=lambda path, **_kw: media[path],
        ):
            Cosmos3ImagePreprocessStage()._prepare_transfer_plan(batch, hints, "in.mp4")
        plan = batch.extra["transfer_plan"]
        self.assertEqual(plan["total_frames"], 9)
        self.assertEqual(batch.num_frames, 9)
        self.assertEqual((batch.fps, plan["output_fps"]), (30, 30))
        self.assertEqual(plan["hints"], ["edge", "depth"])
        self.assertEqual(plan["control_weights"], [0.25, 0.75])
        controls = batch.extra["preprocessed_control"]
        self.assertTrue(
            torch.equal(controls[0][0, :, :9], edge_control_uint8(source, "medium"))
        )
        # The shorter depth clip is reflect-padded to the output length.
        self.assertTrue(torch.equal(controls[1][0, :, :7], depth))
        self.assertTrue(torch.equal(controls[1][0, :, 7:9], depth[:, [6, 5]]))

    def test_transfer_prompt_has_system_prompt_and_emphasis(self):
        """Both captions use the Transfer system prompt; only the positive one ends
        with the emphasis sentence naming the hints in order."""
        conversations = []

        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 1

            def convert_tokens_to_ids(self, token):
                return 2

            def apply_chat_template(self, conversation, **kwargs):
                conversations.append(conversation)
                return [5, 6, 7]

        stage = Cosmos3TokenizationStage(Tokenizer())
        batch = types.SimpleNamespace(
            prompt="A road.",
            negative_prompt="",
            max_sequence_length=512,
            use_duration_template=None,
            use_system_prompt=None,
            fps=30,
            num_frames=41,
            height=480,
            width=832,
            data_type=DataType.VIDEO,
            sampling_params=types.SimpleNamespace(
                action_mode=None, emphasize_control_in_prompt=True
            ),
            extra={"transfer_plan": {"chunk_frames": 93, "hints": ["edge", "wsm"]}},
        )
        server_args = types.SimpleNamespace(
            pipeline_config=types.SimpleNamespace(
                use_duration_template=True,
                use_system_prompt=False,
            )
        )
        with mock.patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages."
            "model_specific_stages.cosmos3.get_local_torch_device",
            return_value=torch.device("cpu"),
        ):
            stage.forward(batch, server_args)
        positive, negative = conversations
        self.assertEqual(
            positive[0], {"role": "system", "content": COSMOS3_TRANSFER_SYSTEM_PROMPT}
        )
        self.assertEqual(negative[0], positive[0])
        # imaginaire4 control_requests.append_control_emphasis, verbatim.
        self.assertEqual(
            positive[1]["content"],
            "A road. The video is 3.1 seconds long and is of 30 FPS. Follow the edge, "
            "wsm control video precisely: shape, contour, silhouette, position, and "
            "motion of every visible structure must align with the edge, wsm signal at "
            "every frame.",
        )
        self.assertEqual(negative[1]["content"], "")

    def test_multi_control_attention_weights_independent_control_passes(self):
        """imaginaire4 multi_control_two_way_attention: control i attends to
        [text | ctrl_i | target] only, and the target's output is the weighted
        sum of those per-control passes (not one joint softmax over all controls)."""
        self.assertIsNone(build_multi_control_layout([4], 6, None))
        layout = build_multi_control_layout([3, 2], 4, [0.25, 0.75])
        self.assertEqual(layout.control_ranges, ((0, 3), (3, 5)))
        self.assertEqual(layout.target_range, (5, 9))
        with self.assertRaisesRegex(ValueError, "control_weights"):
            build_multi_control_layout([3, 2], 4, [1.0])

        def sdpa(q, k, v, _metadata):
            k = k.repeat_interleave(q.shape[2] // k.shape[2], dim=2)
            v = v.repeat_interleave(q.shape[2] // v.shape[2], dim=2)
            out = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            )
            return out.transpose(1, 2)

        attention = types.SimpleNamespace(
            attn=types.SimpleNamespace(attn_impl=types.SimpleNamespace(forward=sdpa))
        )
        gen = torch.Generator().manual_seed(0)
        q, k, v = (
            torch.randn(1, 9, 4, 8, generator=gen, dtype=torch.float64)
            for _ in range(3)
        )
        k_und, v_und = (
            torch.randn(1, 3, 2, 8, generator=gen, dtype=torch.float64)
            for _ in range(2)
        )
        k, v = k[:, :, :2], v[:, :, :2]

        def plain(rows, keys):
            return sdpa(
                q[:, rows],
                torch.cat([k_und, k[:, keys]], 1),
                torch.cat([v_und, v[:, keys]], 1),
                None,
            )

        target = list(range(5, 9))
        pass1 = plain(target, [0, 1, 2, *target])
        pass2 = plain(target, [3, 4, *target])
        with (
            mock.patch.object(
                cosmos3_dit,
                "get_forward_context",
                return_value=types.SimpleNamespace(attn_metadata=None),
            ),
            mock.patch.object(
                cosmos3_dit, "get_ulysses_parallel_world_size", return_value=1
            ),
        ):
            out = Cosmos3CrossAttention._forward_multi_control(
                attention, q, k, v, k_und, v_und, layout
            )
        self.assertTrue(
            torch.allclose(out[:, 0:3], plain([0, 1, 2], [0, 1, 2, *target]))
        )
        self.assertTrue(torch.allclose(out[:, 3:5], plain([3, 4], [3, 4, *target])))
        self.assertTrue(torch.allclose(out[:, 5:9], 0.25 * pass1 + 0.75 * pass2))

    def test_on_the_fly_controls_follow_imaginaire4_presets(self):
        """Edge/blur presets are imaginaire4's AddControlInputEdge/Blur inference values."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
            cosmos3_transfer,
        )

        self.assertEqual(
            cosmos3_transfer.COSMOS3_EDGE_THRESHOLDS,
            {
                "very_low": (20, 50),
                "low": (50, 100),
                "medium": (100, 200),
                "high": (200, 300),
                "very_high": (300, 400),
            },
        )
        self.assertEqual(
            (
                cosmos3_transfer.COSMOS3_BLUR_PRE_DOWNSCALE,
                cosmos3_transfer.COSMOS3_BLUR_DOWNUP,
            ),
            (
                {
                    "none": 1,
                    "very_low": 1,
                    "low": 4,
                    "medium": 2,
                    "high": 1,
                    "very_high": 4,
                },
                {
                    "none": 1,
                    "very_low": 4,
                    "low": 4,
                    "medium": 10,
                    "high": 16,
                    "very_high": 16,
                },
            ),
        )
        frames = torch.randint(0, 256, (3, 2, 40, 64), dtype=torch.uint8)
        edge = edge_control_uint8(frames, "medium")
        self.assertEqual((edge.shape, edge.dtype), (frames.shape, torch.uint8))
        self.assertTrue(torch.equal(edge[0], edge[2]))
        self.assertTrue(set(edge.unique().tolist()) <= {0, 255})
        self.assertTrue(torch.equal(blur_control_uint8(frames, "none"), frames))
        blurred = blur_control_uint8(frames, "medium")
        self.assertEqual((blurred.shape, blurred.dtype), (frames.shape, torch.uint8))


class TestCosmos3DomainAwareLinear(unittest.TestCase):
    """Per-domain action projection."""

    def test_rank3_and_rank2_shapes(self):
        layer = DomainAwareLinear(input_size=7, output_size=64, num_domains=32)
        x3 = torch.randn(2, 16, 7)
        out3 = layer(x3, torch.tensor([1, 5]))
        self.assertEqual(tuple(out3.shape), (2, 16, 64))
        x2 = torch.randn(3, 7)
        out2 = layer(x2, torch.tensor([0, 1, 2]))
        self.assertEqual(tuple(out2.shape), (3, 64))

    def test_distinct_domains_give_distinct_outputs(self):
        torch.manual_seed(0)
        layer = DomainAwareLinear(input_size=4, output_size=8, num_domains=4)
        x = torch.randn(1, 3, 4)
        out_a = layer(x, torch.tensor([0]))
        out_b = layer(x, torch.tensor([2]))
        self.assertFalse(torch.allclose(out_a, out_b))

    def test_scalar_domain_id_promoted(self):
        layer = DomainAwareLinear(input_size=4, output_size=8, num_domains=4)
        out = layer(torch.randn(1, 2, 4), torch.tensor(3))
        self.assertEqual(tuple(out.shape), (1, 2, 8))


class TestCosmos3ConditionIndexes(unittest.TestCase):
    """Vision condition-frame resolution across V2V and action modes."""

    @staticmethod
    def _batch(num_frames=61, **sp_kwargs):
        sp = Cosmos3SamplingParams(prompt="t", num_frames=num_frames, **sp_kwargs)
        return types.SimpleNamespace(sampling_params=sp, num_frames=num_frames)

    def test_v2v_default(self):
        idx = Cosmos3ImagePreprocessStage._resolve_condition_indexes(self._batch())
        self.assertEqual(idx, [0, 1])

    def test_v2v_explicit_sorted_unique(self):
        idx = Cosmos3ImagePreprocessStage._resolve_condition_indexes(
            self._batch(condition_frame_indexes=[2, 0, 2])
        )
        self.assertEqual(idx, [0, 2])

    def test_inverse_dynamics_conditions_all_latent_frames(self):
        # 61 frames -> (61-1)//4 + 1 = 16 latent frames, all locked.
        idx = Cosmos3ImagePreprocessStage._resolve_condition_indexes(
            self._batch(num_frames=61, action_mode="inverse_dynamics")
        )
        self.assertEqual(idx, list(range(16)))


class TestCosmos3DomainResolution(unittest.TestCase):
    """Embodiment domain-id resolution for action generation."""

    @staticmethod
    def _batch(**sp_kwargs):
        sp = Cosmos3SamplingParams(prompt="t", **sp_kwargs)
        return types.SimpleNamespace(sampling_params=sp)

    def test_explicit_domain_id(self):
        self.assertEqual(
            Cosmos3LatentPreparationStage._resolve_domain_id(self._batch(domain_id=7)),
            7,
        )

    def test_domain_name_lookup(self):
        self.assertEqual(
            Cosmos3LatentPreparationStage._resolve_domain_id(
                self._batch(domain_name="av")
            ),
            EMBODIMENT_TO_DOMAIN_ID["av"],
        )
        self.assertEqual(
            Cosmos3LatentPreparationStage._resolve_domain_id(
                self._batch(domain_name="umi")
            ),
            EMBODIMENT_TO_DOMAIN_ID["umi"],
        )

    def test_missing_domain_raises(self):
        with self.assertRaises(ValueError):
            Cosmos3LatentPreparationStage._resolve_domain_id(self._batch())

    def test_unknown_domain_name_raises(self):
        with self.assertRaises(ValueError):
            Cosmos3LatentPreparationStage._resolve_domain_id(
                self._batch(domain_name="not_a_robot")
            )


class TestCosmos3ActionLatentPrep(unittest.TestCase):
    """Action latent / mask preparation per action mode."""

    @classmethod
    def setUpClass(cls):
        # Bypass PipelineStage.__init__ (needs global server args); only the
        # transformer's action_dim and log_info are used by _prepare_action_latents.
        cls.stage = Cosmos3LatentPreparationStage.__new__(Cosmos3LatentPreparationStage)
        cls.stage.transformer = types.SimpleNamespace(action_dim=64)
        cls.stage.log_info = lambda *a, **k: None
        cls.device = torch.device("cpu")
        cls.dtype = torch.float32

    def _run(self, num_frames=17, batch_size=1, **sp_kwargs):
        sp = Cosmos3SamplingParams(prompt="t", num_frames=num_frames, **sp_kwargs)
        batch = types.SimpleNamespace(
            sampling_params=sp,
            num_frames=num_frames,
            raw_latent_shape=(batch_size, 48, 1, 1, 1),
            extra={},
        )
        gen = torch.Generator(device=self.device).manual_seed(0)
        self.stage._prepare_action_latents(batch, gen, self.device, self.dtype)
        return batch

    def test_forward_dynamics_clean_conditioning(self):
        batch = self._run(
            action_mode="forward_dynamics",
            domain_name="agibotworld",
            action=[[0.1] * 29 for _ in range(16)],
        )
        # action_chunk_size = num_frames - 1 = 16; padded to action_dim 64.
        self.assertEqual(tuple(batch.action_latents.shape), (1, 16, 64))
        # raw_action_dim is derived from the embodiment (agibotworld -> 29).
        self.assertEqual(batch.extra["raw_action_dim"], 29)
        self.assertEqual(batch.extra["action_start_frame_offset"], 1)
        self.assertEqual(
            int(batch.extra["action_domain_ids"][0]),
            EMBODIMENT_TO_DOMAIN_ID["agibotworld"],
        )
        # forward_dynamics: action is clean conditioning -> velocity mask all zero.
        self.assertTrue(torch.all(batch.extra["action_velocity_mask"] == 0))

    def test_policy_denoises_from_noise(self):
        batch = self._run(
            action_mode="policy", domain_name="droid_lerobot", raw_action_dim=10
        )
        self.assertEqual(tuple(batch.action_latents.shape), (1, 16, 64))
        self.assertEqual(batch.extra["raw_action_dim"], 10)
        # policy: action fully denoised -> velocity mask all one.
        self.assertTrue(torch.all(batch.extra["action_velocity_mask"] == 1))
        # padding dims beyond raw_action_dim start at zero.
        self.assertTrue(torch.all(batch.action_latents[:, :, 10:] == 0))

    def test_batched_policy_prepares_one_action_stream_per_observation(self):
        batch = self._run(
            batch_size=3,
            action_mode="policy",
            domain_name="droid_lerobot",
        )

        self.assertEqual(tuple(batch.action_latents.shape), (3, 16, 64))
        self.assertEqual(tuple(batch.extra["action_domain_ids"].shape), (3,))
        self.assertEqual(tuple(batch.extra["action_velocity_mask"].shape), (3, 16, 1))

    def test_inverse_dynamics_denoises_from_noise(self):
        batch = self._run(
            num_frames=61,
            action_mode="inverse_dynamics",
            domain_name="av",
            raw_action_dim=9,
        )
        self.assertEqual(tuple(batch.action_latents.shape), (1, 60, 64))
        self.assertTrue(torch.all(batch.extra["action_velocity_mask"] == 1))

    def test_forward_dynamics_requires_action(self):
        with self.assertRaises(ValueError):
            self._run(action_mode="forward_dynamics", domain_name="agibotworld")

    def test_policy_requires_raw_action_dim(self):
        # policy has no input action to infer from, so it needs raw_action_dim
        # from either the embodiment or an explicit value. With only a numeric
        # domain_id (no embodiment name) and no raw_action_dim, it must raise.
        with self.assertRaises(ValueError):
            self._run(action_mode="policy", domain_id=0)

    def test_policy_raw_action_dim_from_embodiment(self):
        # droid_lerobot -> 10, so policy no longer needs an explicit raw dim.
        batch = self._run(action_mode="policy", domain_name="droid_lerobot")
        self.assertEqual(batch.extra["raw_action_dim"], 10)

    def test_unknown_action_mode_raises(self):
        with self.assertRaises(ValueError):
            self._run(action_mode="teleport", domain_id=0)


class TestCosmos3BatchedActionStages(unittest.TestCase):
    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

        @staticmethod
        def convert_tokens_to_ids(_token):
            return 2

        @staticmethod
        def apply_chat_template(conversations, **_kwargs):
            text = conversations[-1]["content"]
            return list(range(3, 3 + len(text.split())))

    @staticmethod
    def _preprocess_images(data_type, action_mode=None):
        stage = Cosmos3ImagePreprocessStage.__new__(Cosmos3ImagePreprocessStage)
        stage.log_info = lambda *_args, **_kwargs: None
        seen = []
        batch = types.SimpleNamespace(
            image_path=["first.png", "second.png"],
            video_path=None,
            height=16,
            width=16,
            data_type=data_type,
            sampling_params=types.SimpleNamespace(action_mode=action_mode),
            preprocessed_image=None,
        )

        def fake_load_image(path):
            seen.append(path)
            return Image.new("RGB", (16, 16))

        with mock.patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages."
            "model_specific_stages.cosmos3.load_image",
            side_effect=fake_load_image,
        ):
            stage.forward(batch, types.SimpleNamespace())
        return seen, batch.preprocessed_image

    def test_tokenization_rejects_unequal_prompt_lengths(self):
        stage = Cosmos3TokenizationStage.__new__(Cosmos3TokenizationStage)
        stage.tokenizer = self._Tokenizer()

        with self.assertRaisesRegex(ValueError, "same length"):
            stage._tokenize_prompt(
                ["pick block", "close the top drawer"],
                max_sequence_length=16,
                device=torch.device("cpu"),
            )

    def test_tokenization_preserves_equal_length_prompt_batch(self):
        stage = Cosmos3TokenizationStage.__new__(Cosmos3TokenizationStage)
        stage.tokenizer = self._Tokenizer()

        input_ids, attention_mask, seq_len = stage._tokenize_prompt(
            ["pick block", "push cube"],
            max_sequence_length=16,
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(input_ids.shape), (2, 16))
        self.assertEqual(tuple(attention_mask.shape), (2, 16))
        self.assertEqual(seq_len, 4)

    def test_cfg_duplicates_batched_timesteps(self):
        stage = Cosmos3DenoisingStage.__new__(Cosmos3DenoisingStage)
        captured = {}

        def fake_run_transformer(**kwargs):
            captured.update(kwargs)
            return kwargs["latents"]

        stage._run_transformer = fake_run_transformer
        latents = torch.zeros(3, 1, 1, 1, 1)
        text_ids = torch.ones(3, 4, dtype=torch.long)
        text_mask = torch.ones_like(text_ids)

        output = stage._predict_noise_cfg_batched(
            latents=latents,
            timestep=torch.ones(3),
            cond_text_ids=text_ids,
            cond_text_mask=text_mask,
            uncond_text_ids=text_ids,
            uncond_text_mask=text_mask,
            video_shape=(1, 1, 1),
            fps=20.0,
            guidance_scale=2.0,
        )

        self.assertEqual(tuple(captured["timestep"].shape), (6,))
        self.assertEqual(tuple(output.shape), tuple(latents.shape))

    def test_visual_i2v_keeps_single_conditioning_image(self):
        seen, image = self._preprocess_images(DataType.VIDEO)

        self.assertEqual(seen, ["first.png"])
        self.assertEqual(tuple(image.shape), (1, 3, 16, 16))

    def test_action_preprocess_stacks_all_images(self):
        seen, image = self._preprocess_images(DataType.ACTION, "policy")

        self.assertEqual(seen, ["first.png", "second.png"])
        self.assertEqual(tuple(image.shape), (2, 3, 16, 16))


class TestCosmos3CaptionMetadata(unittest.TestCase):
    """Structured captions get generation metadata; prose prompts opt out."""

    def test_video_caption_gets_resolution_duration_and_fps(self):
        prompt = json.dumps({"temporal_caption": "a cone melts"})
        caption = json.loads(
            _inject_caption_metadata(
                prompt, num_frames=189, fps=24.0, height=480, width=832
            )
        )
        self.assertEqual(caption["resolution"], {"H": 480, "W": 832})
        # Whole seconds, matching the caption format the model was trained on.
        self.assertEqual(caption["duration"], "7s")
        self.assertEqual(caption["fps"], 24.0)
        self.assertEqual(caption["temporal_caption"], "a cone melts")

    def test_image_caption_drops_temporal_fields(self):
        prompt = json.dumps({"subjects": ["hands"], "duration": "5s", "fps": 24.0})
        caption = json.loads(
            _inject_caption_metadata(
                prompt, num_frames=1, fps=24.0, height=768, width=768
            )
        )
        self.assertEqual(caption["resolution"], {"H": 768, "W": 768})
        self.assertNotIn("duration", caption)
        self.assertNotIn("fps", caption)
        self.assertEqual(caption["subjects"], ["hands"])

    def test_request_resolution_overrides_caption(self):
        prompt = json.dumps({"resolution": {"H": 111, "W": 222}})
        caption = json.loads(
            _inject_caption_metadata(
                prompt, num_frames=1, fps=24.0, height=640, width=640
            )
        )
        self.assertEqual(caption["resolution"], {"H": 640, "W": 640})

    def test_unrelated_caption_fields_are_preserved(self):
        prompt = json.dumps({"aspect_ratio": "1,1", "subjects": ["a vase"]})
        caption = json.loads(
            _inject_caption_metadata(
                prompt, num_frames=1, fps=24.0, height=640, width=640
            )
        )
        self.assertEqual(caption["aspect_ratio"], "1,1")
        self.assertEqual(caption["subjects"], ["a vase"])

    def test_zero_fps_does_not_divide_by_zero(self):
        prompt = json.dumps({"temporal_caption": "a cone melts"})
        caption = json.loads(
            _inject_caption_metadata(
                prompt, num_frames=81, fps=0.0, height=480, width=832
            )
        )
        self.assertEqual(caption["duration"], "0s")

    def test_non_structured_prompts_are_declined(self):
        for prompt in (
            "A curious raccoon in a field of sunflowers.",
            json.dumps(["not", "an", "object"]),
            json.dumps("a bare string"),
            "",
        ):
            with self.subTest(prompt=prompt):
                self.assertIsNone(
                    _inject_caption_metadata(
                        prompt, num_frames=81, fps=24.0, height=480, width=832
                    )
                )


class TestCosmos3DurationTemplateSuppression(unittest.TestCase):
    """A structured caption must not also get the prose duration suffix."""

    def _run_prompt_stage(self, prompt: str, use_duration_template: bool) -> str:
        """Drive Cosmos3TokenizationStage.forward far enough to see the prompt."""
        seen = {}

        def fake_tokenize(prompt_text, *args, **kwargs):
            seen.setdefault("prompt", prompt_text)
            return (
                torch.zeros(1, 4, dtype=torch.long),
                torch.ones(1, 4, dtype=torch.long),
                4,
            )

        stage = Cosmos3TokenizationStage.__new__(Cosmos3TokenizationStage)
        batch = types.SimpleNamespace(
            prompt=prompt,
            negative_prompt="bad",
            max_sequence_length=512,
            use_duration_template=use_duration_template,
            use_system_prompt=False,
            fps=24.0,
            num_frames=189,
            height=480,
            width=832,
            data_type=DataType.VIDEO,
            sampling_params=types.SimpleNamespace(action_mode=None),
            extra={},
        )
        server_args = types.SimpleNamespace(
            pipeline_config=types.SimpleNamespace(
                use_duration_template=True, use_system_prompt=False
            )
        )
        with (
            mock.patch.object(stage, "_tokenize_prompt", fake_tokenize),
            mock.patch.object(stage, "log_info"),
        ):
            stage.forward(batch, server_args)
        return seen["prompt"]

    def test_structured_caption_stays_valid_json(self):
        prompt = json.dumps({"temporal_caption": "a cone melts"})
        final = self._run_prompt_stage(prompt, use_duration_template=True)
        caption = json.loads(final)  # would raise if the suffix were appended
        self.assertEqual(caption["duration"], "7s")
        self.assertNotIn("seconds long", final)

    def test_prose_prompt_still_gets_the_duration_suffix(self):
        final = self._run_prompt_stage("A curious raccoon.", use_duration_template=True)
        self.assertIn("seconds long", final)


class TestCosmos3SequenceSharding(unittest.TestCase):
    def test_sequence_sharding_helpers(self):
        """Every GEN modality (video, control, action, sound) rides one pad-and-
        shard rule under Ulysses: tokens pad with zeros, RoPE positions pad by
        repeating the last entry, each rank gets one contiguous chunk, and
        degree 1 returns the input itself."""
        self.assertEqual(sequence_shard_padding(10, 1), 0)
        self.assertEqual(sequence_shard_padding(10, 4), 2)
        self.assertEqual(sequence_shard_padding(12, 4), 0)
        tokens = torch.arange(10.0).view(1, 10, 1)
        self.assertEqual(
            shard_sequence(tokens, 4, 0, dim=1, pad_last=False).flatten().tolist(),
            [0.0, 1.0, 2.0],
        )
        self.assertEqual(
            shard_sequence(tokens, 4, 3, dim=1, pad_last=False).flatten().tolist(),
            [9.0, 0.0, 0.0],
        )
        positions = torch.arange(10).view(1, 1, 10).expand(3, 1, 10)
        last = shard_sequence(positions, 4, 3, dim=2, pad_last=True)
        self.assertEqual(last.shape, (3, 1, 3))
        self.assertEqual(last[0, 0].tolist(), [9, 9, 9])
        self.assertIs(shard_sequence(tokens, 1, 0, dim=1, pad_last=False), tokens)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None, "requires CUDA"
)
class TestCosmos3VideoPreprocess(CustomTestCase):
    def test_gpu_preprocess_switch(self):
        """CPU padding and deferred GPU padding preserve the same source pixels."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
            cosmos3,
        )

        stage = Cosmos3ImagePreprocessStage.__new__(Cosmos3ImagePreprocessStage)
        stage.log_info = lambda *_args, **_kwargs: None
        frames = np.random.default_rng(816).integers(
            0, 256, (5, 32, 48, 3), dtype=np.uint8
        )
        outputs = []
        for force_cpu in (True, False):
            with (
                self.subTest(force_cpu=force_cpu),
                envs.SGLANG_FORCE_CPU_IMAGE_PREPROCESSING.override(force_cpu),
                mock.patch.object(
                    cosmos3,
                    "_iter_video_frames_pyav",
                    side_effect=lambda *_: (frame for frame in frames),
                ),
            ):
                batch = types.SimpleNamespace(
                    image_path=None,
                    video_path="input.mp4",
                    data_type=DataType.VIDEO,
                    height=24,
                    width=40,
                    num_frames=45,
                    sampling_params=types.SimpleNamespace(),
                )
                stage.forward(batch, types.SimpleNamespace())
                actual = batch.preprocessed_video
                self.assertEqual(actual.device.type, "cpu" if force_cpu else "cuda")
                self.assertEqual(actual.shape, (1, 3, 45 if force_cpu else 5, 24, 40))
                outputs.append(actual[:, :, :5].cpu())
        self.assertTrue(torch.equal(outputs[0], outputs[1]))

    def test_vae_cache_key(self):
        """Video content and VAE settings distinguish Cosmos3 cache entries."""
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

        from sglang.multimodal_gen.runtime.cache import conditioning

        class VAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = types.SimpleNamespace(
                    latents_mean=[0, 0, 0], latents_std=[1, 1, 1]
                )
                self.calls = 0
                self.use_tiling = False

            @conditioning.cached_vae_encode
            def encode(self, pixels):
                self.calls += 1
                mean = pixels * (2 if self.use_tiling else 1)
                return DiagonalGaussianDistribution(
                    torch.cat([mean, torch.zeros_like(mean)], dim=1)
                )

        stage = Cosmos3LatentPreparationStage.__new__(Cosmos3LatentPreparationStage)
        stage.vae = VAE().eval()
        cache = conditioning.ConditioningCache(1024 * 1024)
        video = torch.zeros(1, 3, 5, 16, 16, device="cuda")
        video[:, :, 0] = 1
        with torch.no_grad(), cache.scope():
            self.assertTrue(torch.equal(stage._vae_encode(video), video))
            hit = stage._vae_encode(video.clone())
            self.assertTrue(torch.equal(hit, video))
            self.assertEqual(stage.vae.calls, 1)
            changed = video.clone()
            changed[:, :, -1] = 0.5
            self.assertTrue(torch.equal(stage._vae_encode(changed), changed))
            self.assertEqual(stage.vae.calls, 2)
            stage.vae.use_tiling = True
            self.assertTrue(torch.equal(stage._vae_encode(video), video * 2))
            self.assertEqual(stage.vae.calls, 3)


if __name__ == "__main__":
    unittest.main()
