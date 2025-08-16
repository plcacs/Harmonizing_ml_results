from typing import Union, Optional, TYPE_CHECKING
from dataclasses import dataclass
import torch
from allennlp.common import FromParams
from allennlp.modules.transformer.transformer_module import TransformerModule
from allennlp.modules.transformer.activation_layer import ActivationLayer
from allennlp.modules.transformer.attention_module import SelfAttention, AttentionOutput
from allennlp.modules.transformer.output_layer import OutputLayer
from allennlp.modules.transformer.util import FloatT
if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig

class AttentionLayer(TransformerModule, FromParams):
    hidden_size: int
    num_attention_heads: int
    attention_dropout: float
    hidden_dropout: float

    def __init__(self, hidden_size: int, num_attention_heads: int, attention_dropout: float = 0.0, hidden_dropout: float = 0.0, is_cross_attention: bool = False, is_decoder: bool = False) -> None:

    def forward(self, input_tensor: torch.Tensor, attention_mask: Optional[torch.BoolTensor], head_mask: Optional[torch.BoolTensor] = None, encoder_hidden_states: Optional[torch.Tensor] = None, encoder_attention_mask: Optional[torch.Tensor] = None, output_attentions: bool = False) -> AttentionOutput:

    @classmethod
    def _from_config(cls, config: 'PretrainedConfig', **kwargs) -> 'AttentionLayer':

@dataclass
class TransformerLayerOutput:
    self_attention_probs: Optional[torch.Tensor] = None
    cross_attention_probs: Optional[torch.Tensor] = None

class TransformerLayer(TransformerModule, FromParams):
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    attention_dropout: float
    hidden_dropout: float
    activation: Union[str, torch.nn.Module]
    add_cross_attention: bool

    def __init__(self, hidden_size: int, intermediate_size: int, num_attention_heads: int, attention_dropout: float = 0.0, hidden_dropout: float = 0.0, activation: Union[str, torch.nn.Module] = 'relu', add_cross_attention: bool = False) -> None:

    def get_output_dim(self) -> int:

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.BoolTensor], head_mask: Optional[torch.BoolTensor] = None, encoder_hidden_states: Optional[torch.Tensor] = None, encoder_attention_mask: Optional[torch.Tensor] = None, output_attentions: bool = False) -> TransformerLayerOutput:

    @classmethod
    def _from_config(cls, config: 'PretrainedConfig', **kwargs) -> 'TransformerLayer':
