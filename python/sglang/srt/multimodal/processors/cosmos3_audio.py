"""Multimodal processor for the audio-capable Cosmos3.5 Reasoner.

Images and videos go through the stock Qwen3-VL path. Audio follows the
reference export (``audio_export.extract_audio_features`` and the HF/vLLM
processors): every clip of a request is turned into Qwen3-Omni log-mels in one
padded batch, and each ``<audio_pad>`` placeholder is expanded to the clip's
encoder output length. With ``use_audio_in_video``, the audio track of every
video becomes a separate ``<audio_start><audio_pad><audio_end>`` span placed
right after that video, as the reference vLLM plugin does.
"""

import asyncio
import contextvars
import logging
import re
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import WhisperFeatureExtractor

from sglang.srt.layers.rotary_embedding.mrope_rope_index import (
    _get_feat_extract_output_lengths,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.req_types import MultimodalDataItem
from sglang.srt.models.cosmos3_audio import (
    COSMOS_AUDIO_SAMPLING_RATE,
    CosmosAudioSpec,
    CosmosQwen3VLMoeAudioForConditionalGeneration,
)
from sglang.srt.multimodal.audio_from_video import extract_audio_from_video_bytes
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
from sglang.srt.utils.video_decoder import VideoDecoderWrapper

logger = logging.getLogger(__name__)

# The training frontend keeps whole clips up to its 300 s window (3900 tokens).
COSMOS_AUDIO_MAX_SECONDS = 300

# Per-request flag, set around the uncached path so `load_mm_data` can read it.
_USE_AUDIO_IN_VIDEO: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "cosmos3_audio_use_audio_in_video", default=False
)


def build_cosmos_audio_frontend() -> WhisperFeatureExtractor:
    """The Qwen3-Omni log-mel frontend with the parameters the Reasoner was trained with."""
    return WhisperFeatureExtractor(
        feature_size=128,
        sampling_rate=COSMOS_AUDIO_SAMPLING_RATE,
        hop_length=160,
        chunk_length=COSMOS_AUDIO_MAX_SECONDS,
        n_fft=400,
        padding_value=0.0,
        dither=0.0,
        return_attention_mask=True,
        padding_side="right",
    )


def extract_cosmos_audio_features(
    frontend: WhisperFeatureExtractor, audios: List[np.ndarray]
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Return per-clip log-mels ``[T_i, 128]`` (float32), mel lengths and audio token counts.

    The clips are extracted as one right-padded batch: the valid length and the
    last frames of a clip depend on the batch padding, so a request's clips must
    not be extracted one by one.
    """
    waveforms = [_as_mono_waveform(audio, index) for index, audio in enumerate(audios)]
    extracted = frontend(
        waveforms,
        sampling_rate=COSMOS_AUDIO_SAMPLING_RATE,
        return_tensors="pt",
        return_attention_mask=True,
        padding=True,
        truncation=False,
    )
    features = torch.as_tensor(extracted["input_features"], dtype=torch.float32)
    feature_lengths = torch.as_tensor(extracted["attention_mask"]).sum(
        dim=1, dtype=torch.long
    )
    token_lengths = _get_feat_extract_output_lengths(feature_lengths)
    # [A, 128, T] -> A x [T_i, 128]
    clips = [
        features[index, :, :length].T.contiguous()
        for index, length in enumerate(feature_lengths.tolist())
    ]
    return clips, feature_lengths, token_lengths


def _as_mono_waveform(audio: np.ndarray, index: int) -> np.ndarray:
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError(
            f"Audio clip {index} must be a mono waveform, got shape {waveform.shape}"
        )
    if waveform.size == 0:
        raise ValueError(f"Audio clip {index} is empty")
    if not np.isfinite(waveform).all():
        raise ValueError(f"Audio clip {index} contains NaN or infinite samples")
    if waveform.size > COSMOS_AUDIO_MAX_SECONDS * COSMOS_AUDIO_SAMPLING_RATE:
        logger.warning(
            "Audio clip %d lasts %.1f s, beyond the %d s training window.",
            index,
            waveform.size / COSMOS_AUDIO_SAMPLING_RATE,
            COSMOS_AUDIO_MAX_SECONDS,
        )
    return waveform


def expand_audio_placeholders(
    text: str, audio_token: str, token_lengths: List[int]
) -> str:
    """Repeat the i-th ``audio_token`` ``token_lengths[i]`` times."""
    pieces = text.split(audio_token)
    if len(pieces) - 1 != len(token_lengths):
        raise ValueError(
            f"The prompt has {len(pieces) - 1} {audio_token} placeholder(s) "
            f"but {len(token_lengths)} audio clip(s) were provided."
        )
    return pieces[0] + "".join(
        audio_token * count + piece for count, piece in zip(token_lengths, pieces[1:])
    )


def interleave_video_audio(
    text: str,
    *,
    video_token: str,
    vision_end_token: str,
    audio_token: str,
    audio_span: str,
    explicit_audios: list,
    video_audios: list,
) -> Tuple[str, list]:
    """Insert ``audio_span`` after each video whose ``video_audios`` entry is not None.

    Returns the new prompt and all audio clips in prompt order. An inserted span
    goes right after the video's ``vision_end_token`` (before any explicit audio
    that follows the video), matching the reference vLLM plugin.
    """
    pattern = re.compile(
        "({video}(?:{vision_end})?|{audio})".format(
            video=re.escape(video_token),
            vision_end=re.escape(vision_end_token),
            audio=re.escape(audio_token),
        )
    )
    parts = pattern.split(text)
    markers = parts[1::2]
    num_audio_markers = sum(part == audio_token for part in markers)
    num_video_markers = len(markers) - num_audio_markers
    if (num_audio_markers, num_video_markers) != (
        len(explicit_audios),
        len(video_audios),
    ):
        raise ValueError(
            f"The prompt has {num_audio_markers} audio and "
            f"{num_video_markers} video placeholder(s) but "
            f"{len(explicit_audios)} audio clip(s) and {len(video_audios)} video(s) "
            "were provided."
        )
    explicit = iter(explicit_audios)
    videos = iter(video_audios)
    pieces, audios = [], []
    for index, part in enumerate(parts):
        pieces.append(part)
        if index % 2 == 0:
            continue
        if part == audio_token:
            audios.append(next(explicit))
            continue
        video_audio = next(videos)
        if video_audio is not None:
            pieces.append(audio_span)
            audios.append(video_audio)
    return "".join(pieces), audios


class CosmosQwen3VLMoeAudioProcessor(QwenVLImageProcessor):
    models = [CosmosQwen3VLMoeAudioForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        spec = CosmosAudioSpec.from_hf_config(hf_config)
        self.audio_start_token, self.audio_token, self.audio_end_token = (
            self._tokenizer.convert_ids_to_tokens(
                [
                    spec.audio_start_token_id,
                    spec.audio_token_id,
                    spec.audio_end_token_id,
                ]
            )
        )
        self.audio_span = (
            self.audio_start_token + self.audio_token + self.audio_end_token
        )
        self.audio_start_token_id = spec.audio_start_token_id
        self.audio_token_id = spec.audio_token_id
        self.vision_end_token = self._tokenizer.convert_ids_to_tokens(
            [self.vision_end_token_id]
        )[0]
        qwen_tokens = self.mm_tokens
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=qwen_tokens.image_token,
            image_token_id=qwen_tokens.image_token_id,
            image_token_regex=qwen_tokens.image_token_regex,
            video_token_id=qwen_tokens.video_token_id,
            audio_token_id=spec.audio_token_id,
        ).build(_processor)
        self.audio_frontend = build_cosmos_audio_frontend()

    async def _process_mm_data_uncached(
        self, image_data, input_text, request_obj, *args, **kwargs
    ):
        use_audio_in_video = (
            isinstance(request_obj, GenerateReqInput) and request_obj.use_audio_in_video
        )
        token = _USE_AUDIO_IN_VIDEO.set(use_audio_in_video)
        try:
            return await super()._process_mm_data_uncached(
                image_data, input_text, request_obj, *args, **kwargs
            )
        finally:
            _USE_AUDIO_IN_VIDEO.reset(token)

    async def load_mm_data(
        self, prompt, multimodal_tokens, audio_data=None, **kwargs
    ) -> BaseMultiModalProcessorOutput:
        # The generic loader ignores clips beyond the placeholders; the reference
        # processor rejects that, and so do we.
        num_placeholders = (
            prompt.count(self.audio_token_id)
            if isinstance(prompt, list)
            else prompt.count(self.audio_token)
        )
        if num_placeholders != len(audio_data or []):
            raise ValueError(
                f"The prompt has {num_placeholders} {self.audio_token} placeholder(s) "
                f"but {len(audio_data or [])} audio clip(s) were provided."
            )
        base_output = await super().load_mm_data(
            prompt=prompt,
            multimodal_tokens=multimodal_tokens,
            audio_data=audio_data,
            **kwargs,
        )
        if _USE_AUDIO_IN_VIDEO.get() and base_output.videos:
            await self._attach_video_audio(base_output)
        return base_output

    async def _attach_video_audio(self, base_output: BaseMultiModalProcessorOutput):
        loop = asyncio.get_running_loop()
        video_audios: List[Optional[np.ndarray]] = []
        for index, video in enumerate(base_output.videos):
            source = (
                video.source_bytes if isinstance(video, VideoDecoderWrapper) else None
            )
            audio = None
            if source is not None:
                audio = await loop.run_in_executor(
                    self.io_executor,
                    extract_audio_from_video_bytes,
                    source,
                    COSMOS_AUDIO_SAMPLING_RATE,
                )
            if audio is None:
                logger.warning("Video %d has no decodable audio track; skipped.", index)
            video_audios.append(audio)
        base_output.input_text, base_output.audios = interleave_video_audio(
            base_output.input_text,
            video_token=self.mm_tokens.video_token,
            vision_end_token=self.vision_end_token,
            audio_token=self.audio_token,
            audio_span=self.audio_span,
            explicit_audios=list(base_output.audios),
            video_audios=video_audios,
        )

    def _process_and_collect_mm_items(
        self, input_text: str, images=None, audios=None, videos=None, **kwargs
    ) -> Tuple[List[MultimodalDataItem], torch.Tensor, dict]:
        if not audios:
            return super()._process_and_collect_mm_items(
                input_text, images=images, videos=videos, **kwargs
            )
        clips, feature_lengths, token_lengths = extract_cosmos_audio_features(
            self.audio_frontend, audios
        )
        input_text = expand_audio_placeholders(
            input_text, self.audio_token, token_lengths.tolist()
        )
        mm_items, input_ids, ret = super()._process_and_collect_mm_items(
            input_text, images=images, videos=videos, **kwargs
        )
        spans = self.get_mm_items_offset(input_ids, self.audio_token_id)
        span_lengths = [end - start + 1 for start, end in spans]
        if span_lengths != token_lengths.tolist():
            raise ValueError(
                f"Audio placeholder spans {span_lengths} do not match the audio "
                f"token counts {token_lengths.tolist()}."
            )
        for clip, feature_length, token_length, span in zip(
            clips, feature_lengths, token_lengths, spans
        ):
            mm_items.append(
                MultimodalDataItem(
                    modality=Modality.AUDIO,
                    feature=clip,
                    offsets=[span],
                    model_specific_data={
                        "audio_feature_lengths": int(feature_length),
                        "audio_token_lengths": int(token_length),
                    },
                )
            )
        return mm_items, input_ids, ret
