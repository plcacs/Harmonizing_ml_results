import math
from typing import Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass
import torch
from allennlp.common import FromParams
from allennlp.modules.transformer.util import FloatT, IntT, BoolT

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig

@dataclass
class AttentionOutput:
    key_value_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = ...
    position_bias: Optional[torch.Tensor] = ...
    attention_probs: Optional[torch.Tensor] = ...

class AttentionModule(TransformerModule, FromParams):
    def __init__(self, hidden_size: int = 512, attention_head_size: int = 64, num_attention_heads: int = 8, scoring_func: str = 'scaled_dot_product', output_linear: bool = False, dropout: float = 0.0, bias: bool = True, normalize_weights: bool = False, is_decoder: bool = False, is_cross_attention: bool = False, relative_attention_num_buckets: Optional[int] = None) -> None: ...

    def forward(self, query_states: torch.Tensor, past_key_states: Optional[torch.Tensor] = None, past_value_states: Optional[torch.Tensor] = None, attention_mask: Optional[BoolT] = None, source_states: Optional[torch.Tensor] = None, source_attention_mask: Optional[BoolT] = None, head_mask: Optional[BoolT] = None, position_bias: Optional[FloatT] = None, output_attentions: bool = False, use_cache: bool = False, query_length: Optional[int] = None) -> AttentionOutput: ...

class T5Attention(AttentionModule):
    def __init__(self, is_decoder: bool, hidden_size: int, key_value_proj_dim: int, num_heads: int, has_relative_attention_bias: bool, relative_attention_num_buckets: int = 32, dropout: float = 0.1, normalize: bool = True, is_cross_attention: bool = False) -> None: ...

    def forward(self, hidden_states: torch.Tensor, mask: Optional[BoolT] = None, key_value_states: Optional[torch.Tensor] = None, position_bias: Optional[FloatT] = None, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, layer_head_mask: Optional[BoolT] = None, query_length: Optional[int] = None, use_cache: bool = False, output_attentions: bool = False) -> AttentionOutput: ...

class SelfAttention(AttentionModule):
    def __init__(self, hidden_size: int, num_attention_heads: int, dropout: float = 0.0, scoring_func: str = 'scaled_dot_product', output_linear: bool = False, is_decoder: bool = False, is_cross_attention: bool = False) -> None: ...

    @classmethod
    def _from_config(cls, config, **kwargs) -> 'SelfAttention': ...