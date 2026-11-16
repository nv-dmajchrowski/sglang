"""Unit tests for the Cosmos3.5 Reasoner audio processor.

The contract comes from the i4 audio export (``audio_export.extract_audio_features``
and the HF/vLLM processors): a request's clips share one Whisper log-mel batch,
each ``<audio_pad>`` expands to the clip's encoder length, audio spans keep text
M-RoPE positions, and video audio becomes a separate span after its video.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import asyncio
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import GenerateReqInput  # noqa: E402
from sglang.srt.managers.multimodal_processor import (  # noqa: E402
    get_mm_processor_cls,
    import_processors,
)
from sglang.srt.multimodal.processors.cosmos3_audio import (  # noqa: E402
    CosmosQwen3VLMoeAudioProcessor,
    build_cosmos_audio_frontend,
    expand_audio_placeholders,
    extract_cosmos_audio_features,
    interleave_video_audio,
)
from sglang.srt.runtime_context import publish, reset_context  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402

ARCH = "CosmosQwen3VLMoeAudioForConditionalGeneration"
VOCAB = [
    "<unk>",
    "<|vision_start|>",
    "<|image_pad|>",
    "<|vision_end|>",
    "hello",
    "<|video_pad|>",
    "<pad>",
    "<audio_start>",
    "<audio_pad>",
    "<audio_end>",
]
AUDIO_START, AUDIO_PAD, AUDIO_END = 7, 8, 9


class TestCosmosAudioFrontend(CustomTestCase):
    def setUp(self):
        self.frontend = build_cosmos_audio_frontend()

    def test_lengths_and_token_counts(self):
        # 1 s = 100 frames = 13 tokens; 2.37 s = 237 frames = 31 tokens; 0.33 s = 5 tokens.
        rng = np.random.default_rng(0)
        clips = [
            rng.standard_normal(n).astype(np.float32) * 0.1
            for n in (16000, 37920, 5280)
        ]
        features, lengths, tokens = extract_cosmos_audio_features(self.frontend, clips)
        self.assertEqual(lengths.tolist(), [100, 237, 33])
        self.assertEqual(tokens.tolist(), [13, 31, 5])
        self.assertEqual(
            [tuple(f.shape) for f in features], [(100, 128), (237, 128), (33, 128)]
        )
        self.assertTrue(all(f.dtype == torch.float32 for f in features))

    def test_clip_length_depends_on_the_request_batch(self):
        # The Whisper mask is the sample mask strided by 160 and trimmed once when the
        # padded length is not a multiple of 160, so a 5281-sample clip has 33 frames
        # alone but 34 when padded to a 16000-sample clip. Serving must therefore
        # extract a request's clips together, like the reference HF/vLLM processors.
        clip = np.full(5281, 0.1, dtype=np.float32)
        _, alone, _ = extract_cosmos_audio_features(self.frontend, [clip])
        _, batched, _ = extract_cosmos_audio_features(
            self.frontend, [clip, np.full(16000, 0.1, dtype=np.float32)]
        )
        self.assertEqual(alone.tolist(), [33])
        self.assertEqual(batched.tolist(), [34, 100])

    def test_rejects_non_mono_and_empty_clips(self):
        with self.assertRaisesRegex(ValueError, "mono"):
            extract_cosmos_audio_features(
                self.frontend, [np.zeros((100, 2), np.float32)]
            )
        with self.assertRaisesRegex(ValueError, "empty"):
            extract_cosmos_audio_features(self.frontend, [np.zeros(0, np.float32)])


class TestCosmosAudioPrompt(CustomTestCase):
    def test_expand_audio_placeholders(self):
        text = "a<s><p><e>b<s><p><e>"
        self.assertEqual(
            expand_audio_placeholders(text, "<p>", [2, 3]),
            "a<s><p><p><e>b<s><p><p><p><e>",
        )
        with self.assertRaisesRegex(ValueError, "2 <p> placeholder"):
            expand_audio_placeholders(text, "<p>", [2])

    def test_video_audio_spans_match_the_reference_plugin(self):
        # The cases of i4 audio_vllm/model_test.py, written with token strings:
        # V=<|video_pad|>, E=<|vision_end|>, I=<|image_pad|>, A=<s><p><e>.
        span = "<s><p><e>"

        def run(text, explicit, video_audios):
            return interleave_video_audio(
                text,
                video_token="V",
                vision_end_token="E",
                audio_token="<p>",
                audio_span=span,
                explicit_audios=explicit,
                video_audios=video_audios,
            )

        self.assertEqual(run("SVE.", [], ["v0"]), ("SVE" + span + ".", ["v0"]))
        self.assertEqual(run("SIESVE.", [], ["v0"]), ("SIESVE" + span + ".", ["v0"]))
        self.assertEqual(
            run("SVE" + span + "SVE.", ["x0"], ["v0", "v1"]),
            ("SVE" + span + span + "SVE" + span + ".", ["v0", "x0", "v1"]),
        )
        # A video without an audio track adds nothing.
        self.assertEqual(
            run("SVE" + span + ".", ["x0"], [None]), ("SVE" + span + ".", ["x0"])
        )
        with self.assertRaisesRegex(ValueError, "video"):
            run("SVE.", [], ["v0", "v1"])


class TestCosmosAudioProcessor(CustomTestCase):
    def setUp(self):
        from tokenizers import Tokenizer, decoders
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from transformers import (
            PreTrainedTokenizerFast,
            Qwen2VLImageProcessor,
            Qwen3VLMoeConfig,
            Qwen3VLProcessor,
            Qwen3VLVideoProcessor,
        )

        backend = Tokenizer(
            WordLevel({t: i for i, t in enumerate(VOCAB)}, unk_token="<unk>")
        )
        backend.pre_tokenizer, backend.decoder = WhitespaceSplit(), decoders.Fuse()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="<unk>",
            pad_token="<pad>",
            additional_special_tokens=VOCAB[1:4] + [VOCAB[5]] + VOCAB[7:],
        )
        hf_processor = Qwen3VLProcessor(
            image_processor=Qwen2VLImageProcessor(),
            video_processor=Qwen3VLVideoProcessor(),
            tokenizer=tokenizer,
        )
        config = Qwen3VLMoeConfig(
            vision_start_token_id=1,
            image_token_id=2,
            vision_end_token_id=3,
            video_token_id=5,
            architectures=[ARCH],
        )
        config.cosmos_audio = {
            "config": {
                "encoder_config": {"d_model": 1280, "output_dim": 2048},
                "projection_hidden_size": 8,
                "out_hidden_size": 8,
                "encoder_output_mode": "pre_projection",
            },
            "token_ids": [AUDIO_START, AUDIO_PAD, AUDIO_END],
            "sampling_rate": 16000,
            "layout": "separate_no_timestamps",
        }
        reset_context()
        self.addCleanup(reset_context)
        server_args = ServerArgs(
            model_path="dummy",
            model_impl="sglang",
            mm_feature_transport="cpu",
            mm_process_config={},
            allowed_media_domains=[],
            disable_fast_image_processor=True,
        )
        publish(server_args, role="tokenizer")
        import_processors("sglang.srt.multimodal.processors")
        self.processor_cls = get_mm_processor_cls(config)
        self.processor = CosmosQwen3VLMoeAudioProcessor(
            config, server_args, hf_processor, None, skip_mm_pool=True
        )
        self.tmp = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(0)
        self.paths = []
        for index, samples in enumerate((37920, 5281)):
            path = self.tmp / f"clip{index}.wav"
            sf.write(
                path, (rng.standard_normal(samples) * 0.1).astype(np.float32), 16000
            )
            self.paths.append(str(path))

    def test_architecture_resolves_this_processor(self):
        self.assertIs(self.processor_cls, CosmosQwen3VLMoeAudioProcessor)

    def test_audio_items_offsets_and_positions(self):
        span = "<audio_start><audio_pad><audio_end>"
        prompt = f"hello {span} hello {span} hello"
        output = asyncio.run(
            self.processor.process_mm_data_async(
                image_data=[],
                input_text=prompt,
                request_obj=GenerateReqInput(text=prompt, audio_data=self.paths),
            )
        )
        # 37920 samples -> 237 frames -> 31 tokens; 5281 samples batched -> 34 frames -> 5 tokens.
        expected_ids = (
            [4, AUDIO_START]
            + [AUDIO_PAD] * 31
            + [AUDIO_END, 4, AUDIO_START]
            + [AUDIO_PAD] * 5
            + [AUDIO_END, 4]
        )
        self.assertEqual(output.input_ids, expected_ids)
        self.assertEqual(
            [item.offsets for item in output.mm_items], [[(2, 32)], [(36, 40)]]
        )
        self.assertEqual(
            [
                (item.audio_feature_lengths, item.audio_token_lengths)
                for item in output.mm_items
            ],
            [(237, 31), (34, 5)],
        )
        self.assertEqual(
            [tuple(item.feature.shape) for item in output.mm_items],
            [(237, 128), (34, 128)],
        )
        # Audio spans keep ordinary text positions on all three M-RoPE axes.
        positions = torch.arange(len(expected_ids)).expand(3, -1)
        self.assertTrue(torch.equal(output.mrope_positions.long(), positions))
        self.assertEqual(int(output.mrope_position_delta.reshape(-1)[0]), 0)

    def test_placeholder_count_must_match_the_audio(self):
        prompt = "hello <audio_start><audio_pad><audio_end> hello"
        with self.assertRaisesRegex(ValueError, "1 <audio_pad> placeholder"):
            asyncio.run(
                self.processor.process_mm_data_async(
                    image_data=[],
                    input_text=prompt,
                    request_obj=GenerateReqInput(text=prompt, audio_data=self.paths),
                )
            )


if __name__ == "__main__":
    unittest.main()
