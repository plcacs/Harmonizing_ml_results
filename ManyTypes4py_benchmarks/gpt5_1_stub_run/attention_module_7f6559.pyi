from typing import Any, Optional, Tuple, ClassVar, Dict, List
import torch
from allennlp.common import FromParams
from allennlp.modules.transformer.transformer_module import TransformerModule

class AttentionOutput:
    key_value_state: Any
    position_bias: Any
    attention_probs: Any
    ...

class AttentionModule(TransformerModule, FromParams):
    def __init__(
        self,
        hidden_size: int = ...,
        attention_head_size: int = ...,
        num_attention_heads: int = ...,
        scoring_func: str = ...,
        output_linear: bool = ...,
        dropout: float = ...,
        bias: bool = ...,
        normalize_weights: bool = ...,
        is_decoder: bool = ...,
        is_cross_attention: bool = ...,
        relative_attention_num_buckets: Optional[int] = ...,
    ) -> None: ...
    def _normalize(self) -> None: ...
    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor: ...
    def _query_layer(self, query_states: torch.Tensor) -> torch.Tensor: ...
    def _project(
        self,
        hidden_states: torch.Tensor,
        layer: Any,
        source_states: Optional[torch.Tensor] = ...,
        past_key_or_value: Optional[torch.Tensor] = ...,
    ) -> torch.Tensor: ...
    def _position_bias(
        self,
        position_bias: Optional[torch.Tensor],
        seq_lengths: Tuple[int, int, int],
        past_key_states: Optional[torch.Tensor],
        attention_scores: torch.Tensor,
    ) -> torch.Tensor: ...
    def _get_attention_probs(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        head_mask: Optional[torch.Tensor],
        seq_lengths: Tuple[int, int, int],
        position_bias: Optional[torch.Tensor] = ...,
        past_key_states: Optional[torch.Tensor] = ...,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...
    def _output_layer(self, attention_probs: torch.Tensor, value_layer: torch.Tensor) -> torch.Tensor: ...
    def _get_lengths(
        self,
        query_states: torch.Tensor,
        past_key_states: Optional[torch.Tensor] = ...,
        source_states: Optional[torch.Tensor] = ...,
        query_length: Optional[int] = ...,
    ) -> Tuple[int, int, int]: ...
    def forward(
        self,
        query_states: torch.Tensor,
        past_key_states: Optional[torch.Tensor] = ...,
        past_value_states: Optional[torch.Tensor] = ...,
        attention_mask: Optional[torch.Tensor] = ...,
        source_states: Optional[torch.Tensor] = ...,
        source_attention_mask: Optional[torch.Tensor] = ...,
        head_mask: Optional[torch.Tensor] = ...,
        position_bias: Optional[torch.Tensor] = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
        query_length: Optional[int] = ...,
    ) -> AttentionOutput: ...
    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        bidirectional: bool = ...,
        num_buckets: int = ...,
        max_distance: int = ...,
    ) -> torch.Tensor: ...
    def compute_bias(self, query_length: int, key_length: int) -> torch.Tensor: ...

class T5Attention(AttentionModule):
    _pretrained_relevant_module: ClassVar[List[str]]
    _pretrained_mapping: ClassVar[Dict[str, str]]
    def __init__(
        self,
        is_decoder: bool = ...,
        hidden_size: int = ...,
        key_value_proj_dim: int = ...,
        num_heads: int = ...,
        has_relative_attention_bias: bool = ...,
        relative_attention_num_buckets: int = ...,
        dropout: float = ...,
        normalize: bool = ...,
        is_cross_attention: bool = ...,
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = ...,
        key_value_states: Optional[torch.Tensor] = ...,
        position_bias: Optional[torch.Tensor] = ...,
        past_key_value: Any = ...,
        layer_head_mask: Optional[torch.Tensor] = ...,
        query_length: Optional[int] = ...,
        use_cache: bool = ...,
        output_attentions: bool = ...,
    ) -> AttentionOutput: ...
    @classmethod
    def _from_config(cls, config: Any, **kwargs: Any) -> "T5Attention": ...

class SelfAttention(AttentionModule):
    _pretrained_relevant_module: ClassVar[List[str]]
    _pretrained_mapping: ClassVar[Dict[str, str]]
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = ...,
        scoring_func: str = ...,
        output_linear: bool = ...,
        is_decoder: bool = ...,
        is_cross_attention: bool = ...,
    ) -> None: ...
    @classmethod
    def _from_config(cls, config: Any, **kwargs: Any) -> "SelfAttention": ...