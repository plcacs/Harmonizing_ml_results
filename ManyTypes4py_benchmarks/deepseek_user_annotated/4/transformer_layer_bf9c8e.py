from typing import Union, Optional, TYPE_CHECKING, Tuple, Dict, Any
from dataclasses import dataclass

import torch
from torch import nn

from allennlp.common import FromParams
from allennlp.modules.transformer.transformer_module import TransformerModule
from allennlp.modules.transformer.activation_layer import ActivationLayer
from allennlp.modules.transformer.attention_module import SelfAttention, AttentionOutput
from allennlp.modules.transformer.output_layer import OutputLayer
from allennlp.modules.transformer.util import FloatT

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig


class AttentionLayer(TransformerModule, FromParams):
    _pretrained_relevant_module: str = "encoder.layer.0.attention"
    _pretrained_mapping: Dict[str, str] = {"layer": "layers"}

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        is_cross_attention: bool = False,
        is_decoder: bool = False,
    ) -> None:
        super().__init__()
        self.self: SelfAttention = SelfAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout,
            is_cross_attention=is_cross_attention,
            is_decoder=is_decoder,
        )
        self.output: OutputLayer = OutputLayer(hidden_size, hidden_size, hidden_dropout)

    def forward(
        self,
        input_tensor: torch.Tensor,
        attention_mask: torch.BoolTensor,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.BoolTensor] = None,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        if encoder_hidden_states is not None:
            attention_mask = encoder_attention_mask

        self_output: AttentionOutput = self.self(
            input_tensor,
            source_states=encoder_hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )

        attention_output: torch.Tensor = self.output(self_output.hidden_states, input_tensor)
        outputs: AttentionOutput = AttentionOutput(
            attention_output,
            self_output.key_value_state,
            self_output.position_bias,
            self_output.attention_probs,
        )
        return outputs

    @classmethod
    def _from_config(cls, config: "PretrainedConfig", **kwargs: Any) -> "AttentionLayer":
        final_kwargs: Dict[str, Any] = {}

        final_kwargs["hidden_size"] = config.hidden_size
        final_kwargs["num_attention_heads"] = config.num_attention_heads
        final_kwargs["attention_dropout"] = config.attention_probs_dropout_prob
        final_kwargs["hidden_dropout"] = config.hidden_dropout_prob

        final_kwargs.update(**kwargs)
        return cls(**final_kwargs)


@dataclass
class TransformerLayerOutput:
    hidden_states: FloatT
    self_attention_probs: Optional[FloatT] = None
    cross_attention_probs: Optional[FloatT] = None


class TransformerLayer(TransformerModule, FromParams):
    _pretrained_relevant_module: str = "encoder.layer.0"
    _pretrained_mapping: Dict[str, str] = {
        "layer": "layers",
        "intermediate_act_fn": "act_fn",
        "crossattention": "cross_attention",
    }

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        activation: Union[str, torch.nn.Module] = "relu",
        add_cross_attention: bool = False,
    ) -> None:
        super().__init__()

        self.attention: AttentionLayer = AttentionLayer(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
        )

        if add_cross_attention:
            self.cross_attention: AttentionLayer = AttentionLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                attention_dropout=attention_dropout,
                hidden_dropout=hidden_dropout,
                is_cross_attention=True,
                is_decoder=True,
            )

        self.intermediate: ActivationLayer = ActivationLayer(
            hidden_size=hidden_size, intermediate_size=intermediate_size, activation=activation
        )
        self.output: OutputLayer = OutputLayer(
            input_size=intermediate_size, hidden_size=hidden_size, dropout=hidden_dropout
        )

    def get_output_dim(self) -> int:
        return self.output.get_output_dim()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> TransformerLayerOutput:
        attention_outputs: AttentionOutput = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output: torch.Tensor = attention_outputs.hidden_states
        self_attention_probs: Optional[torch.Tensor] = attention_outputs.attention_probs
        cross_attention_probs: Optional[torch.Tensor] = None

        if encoder_hidden_states is not None:
            assert hasattr(
                self, "cross_attention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated "
            "with cross-attention layers by setting `config.add_cross_attention=True`"

            cross_attention_outputs: AttentionOutput = self.cross_attention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs.hidden_states
            cross_attention_probs = cross_attention_outputs.attention_probs

        intermediate_output: torch.Tensor = self.intermediate(attention_output)
        layer_output: torch.Tensor = self.output(intermediate_output, attention_output)

        outputs: TransformerLayerOutput = TransformerLayerOutput(
            layer_output, self_attention_probs, cross_attention_probs
        )
        return outputs

    @classmethod
    def _from_config(cls, config: "PretrainedConfig", **kwargs: Any) -> "TransformerLayer":
        final_kwargs: Dict[str, Any] = {}
        final_kwargs["hidden_size"] = config.hidden_size
        final_kwargs["num_attention_heads"] = config.num_attention_heads
        final_kwargs["attention_dropout"] = config.attention_probs_dropout_prob
        final_kwargs["hidden_dropout"] = config.hidden_dropout_prob
        final_kwargs["intermediate_size"] = config.intermediate_size
        final_kwargs["activation"] = config.hidden_act
        final_kwargs["add_cross_attention"] = config.add_cross_attention
        final_kwargs.update(**kwargs)
        return cls(**final_kwargs)
