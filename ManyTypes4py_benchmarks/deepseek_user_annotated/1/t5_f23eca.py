"""
An implementation of [T5](https://api.semanticscholar.org/CorpusID:204838007), adapted from [HuggingFace]
(https://github.com/huggingface/transformers/blob/4c32f9f26e6a84f0d9843fec8757e6ce640bb44e/src/transformers/models/t5/modeling_t5.py).
"""  # noqa: E401

import logging
from typing import Optional, Tuple, List, Union, Dict, TYPE_CHECKING, NamedTuple, Callable, Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from allennlp.common import FromParams, Params, Lazy, Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.modules.transformer.transformer_module import TransformerModule
from allennlp.modules.transformer.attention_module import (
    T5Attention,
    AttentionOutput,
)
from allennlp.modules.transformer.util import (
    get_extended_attention_mask,
    FloatT,
    IntT,
    BoolT,
)
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn.parallel import DdpAccelerator
from allennlp.nn.checkpoint import CheckpointWrapper

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)


class T5LayerNorm(TransformerModule, FromParams):
    """T5-style layer norm does not have bias and does not subtract the mean."""

    def __init__(self, hidden_size: int = 512, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> FloatT:
        # layer norm should always be calculated in float32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into float16 if necessary
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states.to(torch.float16)
        return self.weight * hidden_states


class T5FeedForwardProjection(TransformerModule, Registrable):
    def forward(self, hidden_states: torch.Tensor) -> FloatT:
        raise NotImplementedError


@T5FeedForwardProjection.register("relu")
class T5DenseReluDense(TransformerModule, FromParams):
    def __init__(self, hidden_size: int = 512, ff_size: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.wi = nn.Linear(hidden_size, ff_size, bias=False)
        self.wi.weight.data.normal_(mean=0.0, std=hidden_size**-0.5)
        self.wo = nn.Linear(ff_size, hidden_size, bias=False)
        self.wo.weight.data.normal_(mean=0.0, std=ff_size**-0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> FloatT:
        hidden_states = self.wi(hidden_states)
        hidden_states = F.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


@T5FeedForwardProjection.register("gated-gelu")
class T5DenseGatedGeluDense(TransformerModule, FromParams):
    def __init__(self, hidden_size: int = 512, ff_size: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.wi_0 = nn.Linear(hidden_size, ff_size, bias=False)
        self.wi_0.weight.data.normal_(mean=0.0, std=hidden_size**-0.5)
        self.wi_1 = nn.Linear(hidden_size, ff_size, bias=False)
        self.wi_1.weight.data.normal_(mean=0.0, std=hidden_size**-0.5)
        self.wo = nn.Linear(ff_size, hidden_size, bias=False)
        self.wo.weight.data.normal_(mean=0.0, std=ff_size**-0.5)
        self.dropout = nn.Dropout(dropout)
        from allennlp.nn import Activation

        self.gelu_act = Activation.by_name("gelu_new")()

    def forward(self, hidden_states: torch.Tensor) -> FloatT:
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(TransformerModule, FromParams):
    _pretrained_mapping = {"DenseReluDense": "ff_proj"}

    def __init__(
        self,
        ff_proj: Optional[T5FeedForwardProjection] = None,
        layer_norm: Optional[T5LayerNorm] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ff_proj = ff_proj or T5DenseReluDense()
        self.layer_norm = layer_norm or T5LayerNorm()
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> FloatT:
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.ff_proj(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5LayerSelfAttentionOutput(NamedTuple):
    hidden_states: FloatT
    attn_key_value_state: Optional[Tuple[FloatT, FloatT]]
    attn_position_bias: FloatT
    attn_weights: Optional[FloatT] = None


class T5LayerSelfAttention(TransformerModule, FromParams):
    _pretrained_mapping = {"SelfAttention": "self_attention"}

    def __init__(
        self,
        self_attention: Optional[T5Attention] = None,
        layer_norm: Optional[T5LayerNorm] = None,
        dropout: float = 0.1,
        has_relative_attention_bias: bool = False,
    ) -> None:
        super().__init__()
        self.self_attention = self_attention or T5Attention(
            has_relative_attention_bias=has_relative_attention_bias
        )
        self.layer_norm = layer_norm or T5LayerNorm(hidden_size=self.self_attention.hidden_size)
        self.dropout = nn.Dropout(dropout)

    @property
    def hidden_size(self) -> int:
        return self.self_attention.hidden_size

    def forward(
        self,
        hidden_states: FloatT,
        attention_mask: Optional[torch.BoolTensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.BoolTensor] = None,
        past_key_value: Optional[Tuple[FloatT]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> T5LayerSelfAttentionOutput:

        normed_hidden_states = self.layer_norm(hidden_states)

        attention_output: AttentionOutput = self.self_attention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        hidden_states = hidden_states + self.dropout(attention_output.hidden_states)

        return T5LayerSelfAttentionOutput(
            hidden_states,
            attention_output.key_value_state,
            attention_output.position_bias,
            attention_output.attention_probs,
        )


class T5LayerCrossAttentionOutput(NamedTuple):
    hidden_states: FloatT
    attn_key_value_state: Optional[Tuple[FloatT, FloatT]]
    attn_position_bias: FloatT
    attn_weights: Optional[FloatT] = None


class T5LayerCrossAttention(TransformerModule, FromParams):
    _pretrained_mapping = {"EncDecAttention": "enc_dec_attention"}

    def __init__(
        self,
        enc_dec_attention: Optional[T5Attention] = None,
        layer_norm: Optional[T5LayerNorm] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.enc_dec_attention = enc_dec_attention or T5Attention(
            is_decoder=True,
            has_relative_attention_bias=False,
            is_cross_attention=True,
        )
        self.layer_norm = layer_norm or T5LayerNorm(hidden_size=self.enc_dec_attention.hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: FloatT,
        key_value_states: Optional[FloatT],
        attention_mask: Optional[torch.BoolTensor] = None,
        position_bias: Optional[FloatT] = None,
        layer_head_mask: Optional[torch.BoolTensor] = None,
        past_key_value: Optional[Tuple[Tuple[FloatT]]] = None,
        use_cache: bool = False,
        query_length: Optional[int] = None,
        output_attentions: bool = False,
    ) -> T5LayerCrossAttentionOutput:
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output: AttentionOutput = self.enc_dec_attention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output.hidden_states)

        return T5LayerCrossAttentionOutput(
            layer_output,
            attention_output.key_value_state,
            attention_output.position_bias,
            attention_output.attention_probs,
        )


KeyValueStates = Union[
    Tuple[FloatT, FloatT],  # without cross attention
    Tuple[FloatT, FloatT, FloatT, FloatT],  # with cross attention
]


class T5BlockOutput(NamedTuple):
    hidden_states: FloatT
    present_key_value_states: Optional[KeyValueStates]
    self_attn_weights: Optional[FloatT]
    self_attn_position_bias: Optional[FloatT]
    cross_attn_weights: Optional[FloatT] = None
    cross_attn_position_bias: Optional[FloatT] = None


class T5Block(TransformerModule, FromParams):
    def __init__(
        self,
        attention: Optional[T5LayerSelfAttention] = None,
        cross_attention: Optional[T5LayerCrossAttention] = None,
        ff: Optional[T5LayerFF] = None,
    ) -> None:
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(attention or T5LayerSelfAttention())
        if cross_attention is None:
            self.is_decoder = False
        else:
            self.layer.append(cross_attention)
            self.is_decoder = True
        self.layer.append(ff or T5LayerFF())

    @property
    def hidden_size(self) -> int:
        return self.layer[0].hidden_size

    def forward(
        self,
        hidden_states: FloatT,
        attention_mask: Optional[torch.BoolTensor] = None,
        position_bias: Optional[FloatT] = None,
        encoder_hidden_states: Optional[FloatT] = None,
        encoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_decoder_position_bias: Optional[FloatT] = None,
        layer_head_mask: Optional[torch.BoolTensor] = None,
        encoder_layer_head_mask: Optional[torch.BoolTensor] = None,
        past_key_value: Optional[KeyValueStates] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> T5BlockOutput:
        if past_key_value is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            error_message = f"There should be {expected_num_past_key_values} past states. "
            error_message += "2 (past / key) for self attention. "
            if expected_num_past_key_values == 4:
                error_message += "2 (past / key) for cross attention. "
            error_message += f"Got {len(past_key_value)} past key / value states"
            assert len(past_key_value) == expected_num_past_key_values, error_message

        self_attention_outputs: T5LayerSelfAttentionOutput = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=None if past_key_value is None else past_key_value[:2],
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = self_attention_outputs.hidden_states
        present_key_value_state: Optional[
            Tuple[FloatT, FloatT]
        ] = self_attention_outputs.attn_key_value_state

        # clamp inf values to enable fp16 training
        if torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs: T5LayerCrossAttentionOutput = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=encoder_layer_head_mask,
                past_key_value=None if past_key_value is None else past_key_value[2:],
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs.hidden_states
            if torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if (
                present_key_value_state is not None
                and cross_attention_outputs.attn_key_value_state is not None
            ):
                present_key_value_state: KeyValueStates = (  # type: ignore[no-redef]
                    present_key_value_state + cross_attention_outputs.attn_key_value_state
                )

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)
        if torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        output = T5BlockOutput(
            hidden_states,
            present_key_value_state,
            self_attention_outputs.attn_weights,
            self_attention_outputs.attn_position_bias,
            cross_attn_weights=(
                None if not do_cross_attention else cross_attention_outputs.attn_weights
            ),
            cross_attn_position_bias=(
                None if not do_cross_attention else cross_attention_outputs.attn_position_bias
            ),
        )
        return output


class T5StackOutput(NamedTuple):
    last_hidden_state: FloatT
    past_key_values: Optional[List[KeyValueStates]] = None
    all_hidden_states: Optional[List[FloatT]] = None
    attentions: Optional[List[FloatT]] = None
    cross_attentions: Optional[List[FloatT]] = None


class T5Stack(TransformerModule, FromParams):
    _pretrained_mapping = {"embed_tokens": "token_embeddings", "block": "blocks"}

    def __init__(
        self,
        token_embeddings: nn.Embedding,
        blocks: List[T5Block],
        final_layer_norm: Optional[T5LayerNorm] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.is_decoder = blocks[0].is_decoder
        if not all(b.is_decoder == self.is_decoder for b in blocks):
            raise ConfigurationError("Found mismatched blocks in stack.")
        self.blocks = nn.ModuleList(blocks)
        self.token_embeddings = token_embeddings
        self.final_layer_norm = final_layer_norm or T5LayerNorm(hidden_size=self.hidden_size)
        self.dropout = nn.Dropout(dropout)

    @property
    def num_blocks(self) -> int:
        return len