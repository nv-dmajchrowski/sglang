# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inference-only Cosmos3.5 Reasoner with audio understanding.

The audio-capable Cosmos3.5 Reasoner export
(``CosmosQwen3VLMoeAudioForConditionalGeneration``) is a stock Hugging Face
Qwen3-VL-MoE checkpoint plus a ``sound_und_model`` branch:

* ``sound_und_model.encoder.encoder``: the Qwen3-Omni (Thinking) audio tower,
  read *before* its native ``proj1``/``proj2`` output projection
  (``encoder_output_mode="pre_projection"``);
* ``sound_und_model.projector``: ``LayerNorm -> Linear -> GELU -> Linear`` into
  the LLM hidden size.

``config.json`` keeps ``model_type: qwen3_vl_moe`` and serializes the audio
branch under ``cosmos_audio`` (audio config, the ``<audio_start>``,
``<audio_pad>``, ``<audio_end>`` token ids, 16 kHz, separate audio spans).
Projected audio rows replace the ``<audio_pad>`` embeddings. Audio spans take
ordinary text M-RoPE positions and contribute nothing to deepstack.
"""

import logging
from typing import Any, Iterable, List, Optional, Tuple

import msgspec
import torch
import torch.nn as nn

from sglang.srt.configs.qwen3_omni import Qwen3OmniMoeAudioEncoderConfig
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding.mrope_rope_index import (
    _get_feat_extract_output_lengths,
)
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.managers.req_types import MultimodalDataItem
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeAudioEncoder
from sglang.srt.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration

logger = logging.getLogger(__name__)

COSMOS_AUDIO_SAMPLING_RATE = 16000
COSMOS_AUDIO_LAYOUT = "separate_no_timestamps"

# The Qwen3-Omni audio-tower fields consumed by the SGLang encoder.
_ENCODER_CONFIG_FIELDS = (
    "num_mel_bins",
    "encoder_layers",
    "encoder_attention_heads",
    "encoder_ffn_dim",
    "d_model",
    "dropout",
    "attention_dropout",
    "activation_function",
    "activation_dropout",
    "scale_embedding",
    "initializer_range",
    "max_source_positions",
    "n_window",
    "output_dim",
    "n_window_infer",
    "conv_chunksize",
    "downsample_hidden_size",
)


class CosmosAudioSpec(msgspec.Struct, frozen=True):
    """The ``cosmos_audio`` branch of an audio-capable Cosmos3.5 Reasoner config."""

    encoder_config: dict
    encoder_output_mode: str
    projection_hidden_size: int
    out_hidden_size: int
    audio_start_token_id: int
    audio_token_id: int
    audio_end_token_id: int

    @property
    def encoder_output_size(self) -> int:
        if self.encoder_output_mode == "pre_projection":
            return int(self.encoder_config["d_model"])
        return int(self.encoder_config["output_dim"])

    @classmethod
    def from_hf_config(cls, hf_config: Any) -> "CosmosAudioSpec":
        cosmos_audio = getattr(hf_config, "cosmos_audio", None)
        if not isinstance(cosmos_audio, dict):
            raise ValueError(
                "CosmosQwen3VLMoeAudioForConditionalGeneration requires the "
                "exported `cosmos_audio` config branch."
            )
        token_ids = [int(token_id) for token_id in cosmos_audio["token_ids"]]
        if len(token_ids) != 3 or len(set(token_ids)) != 3:
            raise ValueError(
                "cosmos_audio.token_ids must hold the distinct <audio_start>, "
                f"<audio_pad> and <audio_end> ids, got {token_ids}."
            )
        sampling_rate = int(
            cosmos_audio.get("sampling_rate", COSMOS_AUDIO_SAMPLING_RATE)
        )
        if sampling_rate != COSMOS_AUDIO_SAMPLING_RATE:
            raise ValueError(
                f"cosmos_audio.sampling_rate must be {COSMOS_AUDIO_SAMPLING_RATE}, "
                f"got {sampling_rate}."
            )
        layout = cosmos_audio.get("layout", COSMOS_AUDIO_LAYOUT)
        if layout != COSMOS_AUDIO_LAYOUT:
            raise NotImplementedError(
                f"cosmos_audio.layout {layout!r} is not supported; only "
                f"{COSMOS_AUDIO_LAYOUT!r} (separate audio spans) is."
            )
        audio_config = cosmos_audio["config"]
        encoder_config = audio_config["encoder_config"]
        encoder_config = {
            key: encoder_config[key]
            for key in _ENCODER_CONFIG_FIELDS
            if key in encoder_config
        }
        encoder_output_mode = audio_config.get("encoder_output_mode", "native")
        if encoder_output_mode not in ("native", "pre_projection"):
            raise ValueError(
                f"Unsupported cosmos_audio encoder_output_mode {encoder_output_mode!r}."
            )
        return cls(
            encoder_config=encoder_config,
            encoder_output_mode=encoder_output_mode,
            projection_hidden_size=int(audio_config["projection_hidden_size"]),
            out_hidden_size=int(audio_config["out_hidden_size"]),
            audio_start_token_id=token_ids[0],
            audio_token_id=token_ids[1],
            audio_end_token_id=token_ids[2],
        )


class CosmosAudioProjector(nn.Module):
    """Project audio-tower features into the LLM hidden space (Qwen-VL merger MLP shape)."""

    def __init__(
        self,
        input_hidden_size: int,
        projection_hidden_size: int,
        out_hidden_size: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_hidden_size, eps=1e-6)
        self.linear_fc1 = ReplicatedLinear(
            input_hidden_size,
            projection_hidden_size,
            bias=True,
            prefix=add_prefix("linear_fc1", prefix),
        )
        self.act_fn = nn.GELU()
        self.linear_fc2 = ReplicatedLinear(
            projection_hidden_size,
            out_hidden_size,
            bias=True,
            prefix=add_prefix("linear_fc2", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states, _ = self.linear_fc1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states, _ = self.linear_fc2(hidden_states)
        return hidden_states


class CosmosAudioEncoder(nn.Module):
    """Holds the Qwen3-Omni tower under ``encoder`` to keep the checkpoint key layout."""

    def __init__(self, spec: CosmosAudioSpec) -> None:
        super().__init__()
        # Replicated on every rank: the ~0.6B tower is small next to the LLM,
        # and its 20 heads do not divide every TP size.
        self.encoder = Qwen3OmniMoeAudioEncoder(
            Qwen3OmniMoeAudioEncoderConfig(**spec.encoder_config),
            use_data_parallel=True,
            output_pre_projection=spec.encoder_output_mode == "pre_projection",
        )


class CosmosSoundUnderstandingModel(nn.Module):
    def __init__(self, spec: CosmosAudioSpec, prefix: str = "") -> None:
        super().__init__()
        self.encoder = CosmosAudioEncoder(spec)
        self.projector = CosmosAudioProjector(
            spec.encoder_output_size,
            spec.projection_hidden_size,
            spec.out_hidden_size,
            prefix=add_prefix("projector", prefix),
        )

    def forward(
        self, features: List[torch.Tensor], feature_lengths: List[int]
    ) -> torch.Tensor:
        """Encode per-clip log-mels ``[T_i, 128]`` into ``[sum N_i, H]`` audio embeddings."""
        tower = self.encoder.encoder
        weight = tower.conv2d1.weight
        packed = torch.cat(
            [feature[:length] for feature, length in zip(features, feature_lengths)]
        )
        packed = packed.to(device=weight.device, dtype=weight.dtype).T.contiguous()
        lengths = torch.tensor(feature_lengths, dtype=torch.long, device=weight.device)
        hidden_states = tower(packed, feature_lens=lengths).last_hidden_state
        # VisionAttention keeps a leading batch dim of 1: [1, N, D] -> [N, D].
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        hidden_states = hidden_states.to(self.projector.linear_fc1.weight.dtype)
        return self.projector(hidden_states)


class CosmosQwen3VLMoeAudioForConditionalGeneration(Qwen3VLMoeForConditionalGeneration):
    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        self.audio_spec = CosmosAudioSpec.from_hf_config(config)
        if self.audio_spec.out_hidden_size != self.config.hidden_size:
            raise ValueError(
                "cosmos_audio.config.out_hidden_size "
                f"({self.audio_spec.out_hidden_size}) must equal the text hidden "
                f"size ({self.config.hidden_size})."
            )
        if self.language_model_only:
            self.sound_und_model = None
        else:
            self.sound_und_model = CosmosSoundUnderstandingModel(
                self.audio_spec, prefix=add_prefix("sound_und_model", prefix)
            )

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        if self.sound_und_model is None:
            raise RuntimeError(
                "Checkpoint was loaded without its audio tower "
                "(language_model_only=True); audio inputs are not supported."
            )
        features, feature_lengths, token_lengths = [], [], []
        for item in items:
            feature = item.feature
            features.append(feature.reshape(-1, feature.shape[-1]))
            feature_lengths.append(int(item.audio_feature_lengths))
            token_lengths.append(int(item.audio_token_lengths))
        expected = _get_feat_extract_output_lengths(
            torch.tensor(feature_lengths, dtype=torch.long)
        ).tolist()
        if expected != token_lengths:
            raise ValueError(
                "audio_token_lengths do not match the Qwen3-Omni encoder output "
                f"lengths: processor={token_lengths}, encoder={expected}"
            )
        return self.sound_und_model(features, feature_lengths)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        audio_weights: List[Tuple[str, torch.Tensor]] = []

        def backbone_weights():
            for name, loaded_weight in weights:
                if name.startswith("sound_und_model."):
                    audio_weights.append(
                        (name[len("sound_und_model.") :], loaded_weight)
                    )
                else:
                    yield name, loaded_weight

        super().load_weights(backbone_weights())
        if self.sound_und_model is not None:
            self._load_audio_weights(audio_weights)

    def _load_audio_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        # The audio branch must load exactly, as in the reference serving plugin.
        stacked_params_mapping = [
            (".qkv_proj.", ".q_proj.", "q"),
            (".qkv_proj.", ".k_proj.", "k"),
            (".qkv_proj.", ".v_proj.", "v"),
        ]
        params_dict = dict(self.sound_und_model.named_parameters())
        loaded = set()
        for name, loaded_weight in weights:
            name = name.replace(".self_attn.out_proj.", ".self_attn.proj.")
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    raise ValueError(f"Unexpected audio weight sound_und_model.{name}")
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name not in params_dict:
                    raise ValueError(f"Unexpected audio weight sound_und_model.{name}")
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded.add(name)
        missing = sorted(set(params_dict) - loaded)
        if missing:
            raise ValueError(
                f"Missing {len(missing)} audio weight(s), e.g. sound_und_model.{missing[0]}"
            )


EntryClass = CosmosQwen3VLMoeAudioForConditionalGeneration
