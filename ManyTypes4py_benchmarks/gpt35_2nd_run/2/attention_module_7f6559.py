import math
from typing import Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from allennlp.common import FromParams
from allennlp.common.checks import ConfigurationError
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp.modules.transformer.transformer_module import TransformerModule
from allennlp.modules.transformer.util import apply_mask

@dataclass
class AttentionOutput:
    key_value_state: Optional[torch.Tensor] = None
    position_bias: Optional[torch.Tensor] = None
    attention_probs: Optional[torch.Tensor] = None

class AttentionModule(TransformerModule, FromParams):
    hidden_size: int = 512
    attention_head_size: int = 64
    num_attention_heads: int = 8
    scoring_func: str = 'scaled_dot_product'
    output_linear: bool = False
    dropout: float = 0.0
    bias: bool = True
    normalize_weights: bool = False
    is_decoder: bool = False
    is_cross_attention: bool = False
    relative_attention_num_buckets: Optional[int] = None

    def __init__(self, hidden_size=512, attention_head_size=64, num_attention_heads=8, scoring_func='scaled_dot_product', output_linear=False, dropout=0.0, bias=True, normalize_weights=False, is_decoder=False, is_cross_attention=False, relative_attention_num_buckets=None):
        ...

    def _normalize(self):
        ...

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def _query_layer(self, query_states: torch.Tensor) -> torch.Tensor:
        ...

    def _project(self, hidden_states: torch.Tensor, layer: torch.nn.Linear, source_states: Optional[torch.Tensor] = None, past_key_or_value: Optional[torch.Tensor] = None) -> torch.Tensor:
        ...

    def _position_bias(self, position_bias: Optional[torch.Tensor], seq_lengths: Tuple[int, int, int], past_key_states: Optional[torch.Tensor], attention_scores: torch.Tensor) -> torch.Tensor:
        ...

    def _get_attention_probs(self, query_layer: torch.Tensor, key_layer: torch.Tensor, attention_mask: Optional[torch.BoolTensor], head_mask: Optional[torch.BoolTensor], seq_lengths: Tuple[int, int, int], position_bias: Optional[torch.Tensor] = None, past_key_states: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    def _output_layer(self, attention_probs: torch.Tensor, value_layer: torch.Tensor) -> torch.Tensor:
        ...

    def _get_lengths(self, query_states: torch.Tensor, past_key_states: Optional[torch.Tensor] = None, source_states: Optional[torch.Tensor] = None, query_length: Optional[int] = None) -> Tuple[int, int, int]:
        ...

    def forward(self, query_states: torch.Tensor, past_key_states: Optional[torch.Tensor] = None, past_value_states: Optional[torch.Tensor] = None, attention_mask: Optional[torch.BoolTensor] = None, source_states: Optional[torch.Tensor] = None, source_attention_mask: Optional[torch.BoolTensor] = None, head_mask: Optional[torch.BoolTensor] = None, position_bias: Optional[torch.Tensor] = None, output_attentions: bool = False, use_cache: bool = False, query_length: Optional[int] = None) -> AttentionOutput:
        ...

    @staticmethod
    def _relative_position_bucket(relative_position: torch.Tensor, bidirectional: bool = True, num_buckets: int = 32, max_distance: int = 128) -> torch.Tensor:
        ...

    def compute_bias(self, query_length: int, key_length: int) -> torch.Tensor:
        ...

class T5Attention(AttentionModule):
    _pretrained_relevant_module = ['encoder.block.0.layer.0.SelfAttention']
    _pretrained_mapping = {'q': 'query', 'k': 'key', 'v': 'value', 'o': 'output'}

    def __init__(self, is_decoder=False, hidden_size=512, key_value_proj_dim=64, num_heads=8, has_relative_attention_bias=False, relative_attention_num_buckets=32, dropout=0.1, normalize=True, is_cross_attention=False):
        ...

    def forward(self, hidden_states: torch.Tensor, mask: Optional[torch.BoolTensor] = None, key_value_states: Optional[torch.Tensor] = None, position_bias: Optional[torch.Tensor] = None, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, layer_head_mask: Optional[torch.BoolTensor] = None, query_length: Optional[int] = None, use_cache: bool = False, output_attentions: bool = False) -> AttentionOutput:
        ...

    @classmethod
    def _from_config(cls, config, **kwargs):
        ...

class SelfAttention(AttentionModule):
    _pretrained_relevant_module = ['encoder.layers.0.attention.self', 'encoder.layers.0.attention']
    _pretrained_mapping = {'layer': 'layers', 'q_lin': 'query', 'k_lin': 'key', 'v_lin': 'value', 'out_lin': 'output', 'transformer': 'encoder'}

    def __init__(self, hidden_size: int, num_attention_heads: int, dropout: float = 0.0, scoring_func: str = 'scaled_dot_product', output_linear: bool = False, is_decoder: bool = False, is_cross_attention: bool = False):
        ...

    @classmethod
    def _from_config(cls, config, **kwargs):
        ...
