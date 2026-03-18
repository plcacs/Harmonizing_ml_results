from typing import Any, Optional, Tuple
from allennlp.common import FromParams
from allennlp.modules.transformer.transformer_module import TransformerModule

class AttentionOutput:
    context_layer: Any
    key_value_state: Any
    position_bias: Any
    attention_probs: Any
    def __init__(self, context_layer: Any, key_value_state: Any = ..., position_bias: Any = ..., attention_probs: Any = ...) -> None: ...

class AttentionModule(TransformerModule, FromParams):
    hidden_size: int
    num_attention_heads: int
    attention_head_size: int
    all_head_size: int
    scoring_func: str
    relative_attention_num_buckets: Optional[int]
    dropout: float
    is_decoder: bool
    is_cross_attention: bool
    def __init__(
        self,
        hidden_size: int = 512,
        attention_head_size: int = 64,
        num_attention_heads: int = 8,
        scoring_func: str = 'scaled_dot_product',
        output_linear: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
        normalize_weights: bool = False,
        is_decoder: bool = False,
        is_cross_attention: bool = False,
        relative_attention_num_buckets: Optional[int] = None
    ) -> None: ...
    def _normalize(self) -> None: ...
    def _transpose_for_scores(self, x: Any) -> Any: ...
    def _query_layer(self, query_states: Any) -> Any: ...
    def _project(
        self,
        hidden_states: Any,
        layer: Any,
        source_states: Optional[Any] = ...,
        past_key_or_value: Optional[Any] = ...
    ) -> Any: ...
    def _position_bias(
        self,
        position_bias: Any,
        seq_lengths: Tuple[int, int, int],
        past_key_states: Optional[Any],
        attention_scores: Any
    ) -> Any: ...
    def _get_attention_probs(
        self,
        query_layer: Any,
        key_layer: Any,
        attention_mask: Optional[Any],
        head_mask: Optional[Any],
        seq_lengths: Tuple[int, int, int],
        position_bias: Optional[Any] = ...,
        past_key_states: Optional[Any] = ...,
        **kwargs: Any
    ) -> Tuple[Any, Any]: ...
    def _output_layer(self, attention_probs: Any, value_layer: Any) -> Any: ...
    def _get_lengths(
        self,
        query_states: Any,
        past_key_states: Optional[Any] = ...,
        source_states: Optional[Any] = ...,
        query_length: Optional[int] = ...
    ) -> Tuple[int, int, int]: ...
    def forward(
        self,
        query_states: Any,
        past_key_states: Optional[Any] = ...,
        past_value_states: Optional[Any] = ...,
        attention_mask: Optional[Any] = ...,
        source_states: Optional[Any] = ...,
        source_attention_mask: Optional[Any] = ...,
        head_mask: Optional[Any] = ...,
        position_bias: Optional[Any] = ...,
        output_attentions: bool = False,
        use_cache: bool = False,
        query_length: Optional[int] = ...
    ) -> AttentionOutput: ...
    @staticmethod
    def _relative_position_bucket(
        relative_position: Any,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128
    ) -> Any: ...
    def compute_bias(self, query_length: int, key_length: int) -> Any: ...

class T5Attention(AttentionModule):
    _pretrained_relevant_module: Any
    _pretrained_mapping: Any
    def __init__(
        self,
        is_decoder: bool = False,
        hidden_size: int = 512,
        key_value_proj_dim: int = 64,
        num_heads: int = 8,
        has_relative_attention_bias: bool = False,
        relative_attention_num_buckets: int = 32,
        dropout: float = 0.1,
        normalize: bool = True,
        is_cross_attention: bool = False
    ) -> None: ...
    def forward(
        self,
        hidden_states: Any,
        mask: Optional[Any] = ...,
        key_value_states: Optional[Any] = ...,
        position_bias: Optional[Any] = ...,
        past_key_value: Optional[Any] = ...,
        layer_head_mask: Optional[Any] = ...,
        query_length: Optional[int] = ...,
        use_cache: bool = False,
        output_attentions: bool = False
    ) -> AttentionOutput: ...
    @classmethod
    def _from_config(cls, config: Any, **kwargs: Any) -> "T5Attention": ...

class SelfAttention(AttentionModule):
    _pretrained_relevant_module: Any
    _pretrained_mapping: Any
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = 0.0,
        scoring_func: str = 'scaled_dot_product',
        output_linear: bool = False,
        is_decoder: bool = False,
        is_cross_attention: bool = False
    ) -> None: ...
    @classmethod
    def _from_config(cls, config: Any, **kwargs: Any) -> "SelfAttention": ...