from typing import Optional, Tuple, List, Union, Dict, Callable

class T5LayerNorm(TransformerModule, FromParams):
    def __init__(self, hidden_size: int = 512, eps: float = 1e-06):
        ...

class T5FeedForwardProjection(TransformerModule, Registrable):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...

@T5FeedForwardProjection.register('relu')
class T5DenseReluDense(TransformerModule, FromParams):
    def __init__(self, hidden_size: int = 512, ff_size: int = 2048, dropout: float = 0.1):
        ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...

@T5FeedForwardProjection.register('gated-gelu')
class T5DenseGatedGeluDense(TransformerModule, FromParams):
    def __init__(self, hidden_size: int = 512, ff_size: int = 2048, dropout: float = 0.1):
        ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...

class T5LayerFF(TransformerModule, FromParams):
    def __init__(self, ff_proj: Optional[T5FeedForwardProjection] = None, layer_norm: Optional[T5LayerNorm] = None, dropout: float = 0.1):
        ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...

class T5LayerSelfAttentionOutput(NamedTuple):
    attn_weights: Optional[torch.Tensor] = None

class T5LayerSelfAttention(TransformerModule, FromParams):
    def __init__(self, self_attention: Optional[T5Attention] = None, layer_norm: Optional[T5LayerNorm] = None, dropout: float = 0.1, has_relative_attention_bias: bool = False):
        ...
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_bias: Optional[torch.Tensor] = None, layer_head_mask: Optional[torch.Tensor] = None, past_key_value: Optional[KeyValueStates] = None, use_cache: bool = False, output_attentions: bool = False) -> T5LayerSelfAttentionOutput:
        ...

class T5LayerCrossAttentionOutput(NamedTuple):
    attn_weights: Optional[torch.Tensor] = None

class T5LayerCrossAttention(TransformerModule, FromParams):
    def __init__(self, enc_dec_attention: Optional[T5Attention] = None, layer_norm: Optional[T5LayerNorm] = None, dropout: float = 0.1):
        ...
    def forward(self, hidden_states: torch.Tensor, key_value_states: FloatT, attention_mask: Optional[torch.Tensor] = None, position_bias: Optional[torch.Tensor] = None, layer_head_mask: Optional[torch.Tensor] = None, past_key_value: Optional[KeyValueStates] = None, use_cache: bool = False, query_length: Optional[int] = None, output_attentions: bool = False) -> T5LayerCrossAttentionOutput:
        ...

KeyValueStates = Union[Tuple[FloatT, FloatT], Tuple[FloatT, FloatT, FloatT, FloatT]]

class T5BlockOutput(NamedTuple):
    cross_attn_weights: Optional[torch.Tensor] = None
    cross_attn_position_bias: Optional[torch.Tensor] = None

class T5Block(TransformerModule, FromParams):
    def __init__(self, attention: Optional[T5LayerSelfAttention] = None, cross_attention: Optional[T5LayerCrossAttention] = None, ff: Optional[T5LayerFF] = None):
        ...
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_bias: Optional[torch.Tensor] = None, encoder_hidden_states: Optional[torch.Tensor] = None, encoder_attention_mask: Optional[torch.Tensor] = None, encoder_decoder_position_bias: Optional[torch.Tensor] = None, layer_head_mask: Optional[torch.Tensor] = None, encoder_layer_head_mask: Optional[torch.Tensor] = None, past_key_value: Optional[KeyValueStates] = None, use_cache: bool = False, output_attentions: bool = False) -> T5BlockOutput:
        ...

class T5StackOutput(NamedTuple):
    past_key_values: Optional[List[KeyValueStates]] = None
    all_hidden_states: Optional[List[torch.Tensor]] = None
    attentions: Optional[List[torch.Tensor]] = None
    cross_attentions: Optional[List[torch.Tensor]] = None

class T5Stack(TransformerModule, FromParams):
    def __init__(self, token_embeddings: nn.Embedding, blocks: List[T5Block], final_layer_norm: Optional[T5LayerNorm] = None, dropout: float = 0.1):
        ...
    def forward(self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, encoder_hidden_states: Optional[torch.Tensor] = None, encoder_attention_mask: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None, head_mask: Optional[torch.Tensor] = None, encoder_head_mask: Optional[torch.Tensor] = None, past_key_values: Optional[List[KeyValueStates]] = None, use_cache: bool = False, output_attentions: bool = False, output_all_hidden_states: bool = False) -> T5StackOutput:
        ...

class T5EncoderStack(T5Stack, FromParams):
    def __init__(self, token_embeddings: nn.Embedding, blocks: List[T5Block], final_layer_norm: Optional[T5LayerNorm] = None, dropout: float = 0.1):
        ...

    @classmethod
    def basic_encoder(cls, token_embeddings: nn.Embedding, num_blocks: int = 6, block_self_attention: Lazy[T5Attention], final_layer_norm: Optional[T5LayerNorm] = None, block_ff: Lazy[T5LayerFF], dropout: float = 0.1, ddp_accelerator: Optional[DdpAccelerator] = None, checkpoint_wrapper: Optional[CheckpointWrapper] = None):
        ...

class T5DecoderStack(T5Stack, FromParams):
    def __init__(self, token_embeddings: nn.Embedding, blocks: List[T5Block], final_layer_norm: Optional[T5LayerNorm] = None, dropout: float = 0.1):
        ...

    @classmethod
    def basic_decoder(cls, token_embeddings: nn.Embedding, num_blocks: int = 6, block_self_attention: Lazy[T5Attention], block_cross_attention: Lazy[T5Attention], final_layer_norm: Optional[T5LayerNorm] = None, block_ff: Lazy[T5LayerFF], dropout: float = 0.1, ddp_accelerator: Optional[DdpAccelerator] = None, checkpoint_wrapper: Optional[CheckpointWrapper] = None):
        ...

class T5Output(NamedTuple):
    encoder_all_hidden_states: Optional[torch.Tensor] = None
    decoder_last_hidden_state: Optional[torch.Tensor] = None
    decoder_all_hidden_states: Optional[torch.Tensor] = None
    encoder_attentions: Optional[torch.Tensor] = None
    decoder_attentions: Optional[torch.Tensor] = None
    cross_attentions: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    predictions: Optional[torch.Tensor] = None
    predicted_log_probs: Optional[torch.Tensor] = None

class T5(TransformerModule, Registrable):
    def __init__(self, token_embeddings: Optional[nn.Embedding] = None, encoder: Lazy[T5EncoderStack], decoder: Lazy[T5DecoderStack], decoder_start_token_id: int = 0, pad_token_id: int = 0, eos_token_id: int = 1, vocab_size: int = 32128, model_dim: int = 512, output_attentions: bool = False, output_all_hidden_states: bool = False, beam_search: Lazy[BeamSearch], ddp_accelerator: Optional[DdpAccelerator] = None, checkpoint_wrapper: Optional[CheckpointWrapper] = None, tie_word_embeddings: bool = True):
        ...

    def resize_token_embeddings(self, new_size: int, init_fn: Callable = torch.nn.init.normal_):
        ...

    def _post_load_state_dict(self, missing_keys: List[str], unexpected_keys: List[str]) -> Tuple[List[str], List[str]]:
        ...

    @classmethod
    def _from_config(cls, config, **kwargs):
        ...

    def _shift_right(self, input_ids: torch.Tensor, start_value: int) -> torch.Tensor:
        ...

    def _get_lm_logits(self, decoder_last_hidden_state: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, input_ids: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor], labels: Optional[torch.Tensor], decoder_attention_mask: Optional[torch.Tensor]) -> T5Output:
        ...

    def take_search_step(self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor], step: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ...

    @staticmethod
    def _decoder_cache_to_dict(decoder_cache: List[KeyValueStates]) -> Dict[str, torch.Tensor]:
        ...

    def _dict_to_decoder_cache(self, cache_dict: Dict[str, torch.Tensor]) -> List[KeyValueStates]:
        ...
