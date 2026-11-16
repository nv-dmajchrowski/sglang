"""Unit tests for the audio-capable Cosmos3.5 Reasoner (CosmosQwen3VLMoeAudioForConditionalGeneration).

The audio branch of the export is a Qwen3-Omni audio tower read before its
``proj1``/``proj2`` projection, followed by a ``LayerNorm -> Linear -> GELU ->
Linear`` projector. These tests check the ``cosmos_audio`` config contract, the
checkpoint key mapping, and that the SGLang tower matches the Hugging Face
tower on CPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.layers.dp_attention import initialize_dp_attention_flags
from sglang.srt.layers.rotary_embedding.mrope_rope_index import (
    _get_feat_extract_output_lengths,
)
from sglang.srt.managers.req_types import MultimodalDataItem
from sglang.srt.models.cosmos3_audio import (
    CosmosAudioSpec,
    CosmosQwen3VLMoeAudioForConditionalGeneration,
    CosmosSoundUnderstandingModel,
)
from sglang.srt.multimodal.modality import Modality
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

# A small Qwen3-Omni tower with the released chunking (n_window 50, n_window_infer 800).
ENCODER = dict(
    num_mel_bins=128,
    encoder_layers=2,
    encoder_attention_heads=2,
    encoder_ffn_dim=32,
    d_model=16,
    dropout=0.0,
    attention_dropout=0.0,
    activation_function="gelu",
    activation_dropout=0.0,
    scale_embedding=False,
    initializer_range=0.02,
    max_source_positions=1500,
    n_window=50,
    output_dim=24,
    n_window_infer=800,
    conv_chunksize=500,
    downsample_hidden_size=8,
)
PROJECTION_HIDDEN, TEXT_HIDDEN = 12, 20


def exported_cosmos_audio(**audio_overrides):
    """``config.cosmos_audio`` as written by i4 ``audio_export.prepare_audio_export``."""
    encoder_config = {
        **ENCODER,
        # Generic PretrainedConfig keys that ``to_dict()`` serializes as well.
        "model_type": "qwen3_omni_moe_audio_encoder",
        "return_dict": True,
        "id2label": {"0": "LABEL_0", "1": "LABEL_1"},
        "transformers_version": "4.57.1",
    }
    audio = {
        "model_type": "qwen3_omni_thinking_audio",
        "encoder_config": encoder_config,
        "projection_hidden_size": PROJECTION_HIDDEN,
        "out_hidden_size": TEXT_HIDDEN,
        "encoder_output_mode": "pre_projection",
    }
    audio.update(audio_overrides)
    return {
        "config": audio,
        "token_ids": [151669, 151670, 151671],
        "sampling_rate": 16000,
        "layout": "separate_no_timestamps",
    }


class TestCosmosAudioSpec(CustomTestCase):
    def test_parses_exported_config(self):
        spec = CosmosAudioSpec.from_hf_config(
            SimpleNamespace(cosmos_audio=exported_cosmos_audio())
        )
        self.assertEqual(spec.encoder_config, ENCODER)
        self.assertEqual(
            (spec.audio_start_token_id, spec.audio_token_id, spec.audio_end_token_id),
            (151669, 151670, 151671),
        )
        self.assertEqual(spec.encoder_output_size, ENCODER["d_model"])
        self.assertEqual(spec.projection_hidden_size, PROJECTION_HIDDEN)
        self.assertEqual(spec.out_hidden_size, TEXT_HIDDEN)
        native = CosmosAudioSpec.from_hf_config(
            SimpleNamespace(
                cosmos_audio=exported_cosmos_audio(encoder_output_mode="native")
            )
        )
        self.assertEqual(native.encoder_output_size, ENCODER["output_dim"])

    def test_rejects_unsupported_exports(self):
        with self.assertRaisesRegex(ValueError, "cosmos_audio"):
            CosmosAudioSpec.from_hf_config(SimpleNamespace())
        broken = exported_cosmos_audio()
        broken["token_ids"] = [1, 2]
        with self.assertRaisesRegex(ValueError, "token_ids"):
            CosmosAudioSpec.from_hf_config(SimpleNamespace(cosmos_audio=broken))
        broken = exported_cosmos_audio()
        broken["sampling_rate"] = 44100
        with self.assertRaisesRegex(ValueError, "sampling_rate"):
            CosmosAudioSpec.from_hf_config(SimpleNamespace(cosmos_audio=broken))
        broken = exported_cosmos_audio()
        broken["layout"] = "interleaved_timestamps"
        with self.assertRaises(NotImplementedError):
            CosmosAudioSpec.from_hf_config(SimpleNamespace(cosmos_audio=broken))
        with self.assertRaisesRegex(ValueError, "encoder_output_mode"):
            CosmosAudioSpec.from_hf_config(
                SimpleNamespace(
                    cosmos_audio=exported_cosmos_audio(encoder_output_mode="other")
                )
            )

    def test_token_lengths_follow_the_reference_formula(self):
        # i4 get_qwen3_omni_audio_output_lengths: 13 tokens per 100 mel frames.
        def reference(length):
            leave = length % 100
            feat = (leave - 1) // 2 + 1
            return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (length // 100) * 13

        lengths = torch.arange(1, 30001)
        expected = torch.tensor([reference(int(n)) for n in lengths])
        self.assertTrue(
            torch.equal(_get_feat_extract_output_lengths(lengths), expected)
        )
        self.assertEqual(
            int(_get_feat_extract_output_lengths(torch.tensor(30000))), 3900
        )


class _HFReferenceAudioModel(nn.Module):
    """The i4 Qwen3OmniAudioModel in pre-projection mode, on the transformers tower."""

    def __init__(self):
        super().__init__()
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoderConfig,
        )
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoder,
        )

        config = Qwen3OmniMoeAudioEncoderConfig(**ENCODER)
        config._attn_implementation = "eager"
        self.encoder = nn.Module()
        self.encoder.encoder = Qwen3OmniMoeAudioEncoder(config)
        self.projector = nn.Module()
        self.projector.norm = nn.LayerNorm(ENCODER["d_model"], eps=1e-6)
        self.projector.linear_fc1 = nn.Linear(ENCODER["d_model"], PROJECTION_HIDDEN)
        self.projector.linear_fc2 = nn.Linear(PROJECTION_HIDDEN, TEXT_HIDDEN)

    def forward(self, features, lengths):
        tower = self.encoder.encoder
        packed = torch.cat([f[:n].T for f, n in zip(features, lengths)], dim=1)
        captured = []
        hook = tower.proj1.register_forward_pre_hook(
            lambda _m, inputs: captured.append(inputs[0])
        )
        try:
            tower(input_features=packed, feature_lens=torch.tensor(lengths))
        finally:
            hook.remove()
        hidden = self.projector.norm(captured[0])
        hidden = self.projector.linear_fc2(
            nn.functional.gelu(self.projector.linear_fc1(hidden))
        )
        return hidden


def _randomize(module, seed):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, param in module.named_parameters():
            param.copy_(torch.randn(param.shape, generator=generator) * 0.1)


class TestCosmosAudioTower(CustomTestCase):
    def setUp(self):
        try:
            import transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe  # noqa: F401
        except ImportError:
            self.skipTest("transformers has no Qwen3-Omni")
        reset_context()
        self.addCleanup(reset_context)
        server_args = ServerArgs(
            model_path="dummy", device="cpu", tp_size=1, mm_attention_backend="sdpa"
        )
        publish(server_args, role="test")
        initialize_dp_attention_flags(server_args)
        torch.manual_seed(0)
        self.spec = CosmosAudioSpec.from_hf_config(
            SimpleNamespace(cosmos_audio=exported_cosmos_audio())
        )
        self.model = object.__new__(CosmosQwen3VLMoeAudioForConditionalGeneration)
        nn.Module.__init__(self.model)
        self.model.sound_und_model = CosmosSoundUnderstandingModel(self.spec)
        self.reference = _HFReferenceAudioModel()
        _randomize(self.reference, seed=1)
        self.checkpoint = list(self.reference.state_dict().items())

    def _items(self, features, lengths, token_lengths=None):
        token_lengths = (
            token_lengths
            or _get_feat_extract_output_lengths(torch.tensor(lengths)).tolist()
        )
        return [
            MultimodalDataItem(
                modality=Modality.AUDIO,
                feature=feature[:length],
                model_specific_data={
                    "audio_feature_lengths": length,
                    "audio_token_lengths": tokens,
                },
            )
            for feature, length, tokens in zip(features, lengths, token_lengths)
        ]

    def test_matches_hf_tower_and_projector(self):
        self.model._load_audio_weights(self.checkpoint)
        # One clip spans several 100-frame chunks and two 104-token attention windows;
        # 100 frames fills exactly one chunk (empty tail chunk).
        lengths = [1603, 237, 100, 33]
        features = [torch.randn(n, 128) for n in lengths]
        with torch.no_grad():
            actual = self.model.get_audio_feature(self._items(features, lengths))
            expected = self.reference(features, lengths)
        self.assertEqual(
            actual.shape,
            (
                int(_get_feat_extract_output_lengths(torch.tensor(lengths)).sum()),
                TEXT_HIDDEN,
            ),
        )
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_checkpoint_must_match_exactly(self):
        with self.assertRaisesRegex(ValueError, "Unexpected audio weight"):
            self.model._load_audio_weights(
                self.checkpoint + [("projector.extra.weight", torch.zeros(1))]
            )
        with self.assertRaisesRegex(ValueError, "Missing 1 audio weight"):
            self.model._load_audio_weights(
                [kv for kv in self.checkpoint if kv[0] != "projector.norm.bias"]
            )

    def test_rejects_token_count_mismatch(self):
        self.model._load_audio_weights(self.checkpoint)
        features = [torch.randn(237, 128)]
        with self.assertRaisesRegex(ValueError, "audio_token_lengths"):
            self.model.get_audio_feature(self._items(features, [237], [30]))


if __name__ == "__main__":
    unittest.main()
