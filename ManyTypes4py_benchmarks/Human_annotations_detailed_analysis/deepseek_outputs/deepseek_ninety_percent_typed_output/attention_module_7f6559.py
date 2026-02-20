import math
from typing import Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from allennlp.common import FromParams
from allennlp.common.checks import ConfigurationError
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp.modules.transformer.transformer_module import TransformerModule
from allennlp.modules.transformer.util import apply_mask, FloatT, IntT, BoolT

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig


@dataclass
class AttentionOutput:
    """
    Encapsulates the outputs of the `Attention` module.
    """

    hidden_states: FloatT
    key_value_state: Optional[Tuple[FloatT, FloatT]] = None
    position_bias: Optional[FloatT] = None
    attention_probs: Optional[FloatT] = None


class AttentionModule(TransformerModule, FromParams):
    """
    This module computes self-attention (or cross-attention), similar to the architecture in BERT.
    Details in the paper:
    [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Devlin et al, 2019]
    (https://api.semanticscholar.org/CorpusID:52967399)

    Additionally, it has the following functionality:

    * the attention scoring function can be specified.
    * it can be used in encoders as well as decoders.
    * `position_bias` can be used, which makes it suitable for
    [T5-style attention](https://api.semanticscholar.org/CorpusID:204838007) as well.

    # Parameters

    hidden_size: `int` (default = `512`)
        The size of the expected input tensor.
    attention_head_size: `int` (default = `64`)
        The size of a single attention head.
    num_attention_heads: `int` (default = `8`)
        The number of attention heads.
    scoring_func: `str` (default = `scaled_dot_product`)
        The name of the attention-calculating function to be used.
        Eg. `additive`, `linear`, etc. For a complete list, please check
        :mod:`allennlp.modules.matrix_attention.matrix_attention`.
    output_linear: `bool` (default = `False`)
        Whether to add an additional output linear layer at the end.
    dropout: `float` (default = `0.0`)
        The dropout probability.
    bias: `bool` (default = `True`)
        Whether to include bias weights in query, key, value (and output) linear layers.
    normalize_weights: `bool` (default = `False`)
        Whether to normalize the initial weights.
    is_decoder: `bool` (default = `False`)
        Whether this module is being used in a decoder stack or not.
    is_cross_attention: `bool` (default = `False`)
        Whether this module is being used for cross-attention in a decoder stack or not.
        If `is_cross_attention` is `True`, then `is_decoder` must also be `True`.
    relative_attention_num_buckets: `int`,  optional (default = `None`)
        The number of buckets to use in relative attention; if `None`, relative attention
        will not be applied.
    """

    def __init__(
        self,
        hidden_size: int = 512,
        attention_head_size: int = 64,
        num_attention_heads: int = 8,
        scoring_func: str = "scaled_dot_product",
        output_linear: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
        normalize_weights: bool = False,
        is_decoder: bool = False,
        is_cross_attention: bool = False,
        relative_attention_num_buckets: Optional[int] = None,
    ):

        super().__init__()

        if hidden_size % num_attention_heads != 0:
            raise ConfigurationError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )

        if is_cross_attention and not is_decoder:
            raise ConfigurationError(
                "The attention layer can be a cross-attention layer only "
                "if it is within a decoder."
            )

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = torch.nn.Linear(hidden_size, self.all_head_size, bias=bias)
        self.key = torch.nn.Linear(hidden_size, self.all_head_size, bias=bias)
        self.value = torch.nn.Linear(hidden_size, self.all_head_size, bias=bias)

        # out linear layer for distilbert, T5 etc.
        if output_linear:
            self.output = torch.nn.Linear(self.all_head_size, hidden_size, bias=bias)

        self.scoring_func = scoring_func
        self.attn = MatrixAttention.by_name(self.scoring_func)()

        self.relative_attention_num_buckets = relative_attention_num_buckets

        if self.relative_attention_num_buckets is not None:
            self.relative_attention_bias = torch.nn.Embedding(
                self.relative_attention_num_buckets, self.num_attention_heads
            )

        self.dropout = dropout

        self.is_decoder = is_decoder
        self.is_cross_attention = is_cross_attention

        if normalize_weights:
            self._normalize()

    def _normalize(self) -> None:
        self.query.weight.data.normal_(
            mean=0.0, std=(self.hidden_size * self.attention_head_size) ** -0.5
        )
        self.key.weight.data.normal_(mean=0.0, std=self.hidden_size**-0.5)
        self.value.weight.data.normal_(mean=0.0, std=self.hidden_size**-0.5)

        if hasattr(self, "output"):
            self.output.weight.data.normal_(
                mean=0.0, std=(self.num_attention_heads * self.attention_head_size) ** -0.5
            )

        if hasattr(self, "relative_attention_bias"):
            self.relative_attention_bias.weight.data.normal_(mean=0.0, std=self.hidden_size**-0.5)

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def _query_layer(self, query_states: torch.Tensor) -> torch.Tensor:
        mixed_query_layer = self.query(query_states)
        query_layer = self._transpose_for_scores(mixed_query_layer)
        return query_layer

    def _project(
        self,
        hidden_states: torch.Tensor,
        layer: torch.nn.Linear,
        source_states: Optional[torch.Tensor] = None,
        past_key_or_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # TODO: clarify logic in terms of is_decoder and is_cross_attention
        # to make it more readable.
        if source_states is None:
            # self-attn
            # (batch_size, num_heads, seq_length, dim_per_head)
            hidden_states = self._transpose_for_scores(layer(hidden_states))
        elif past_key_or_value is None:
            # cross-attn
            # (batch_size, num_heads, seq_length, dim_per_head)
            hidden_states = self._transpose_for_scores(layer(source_states))

        if past_key_or_value is not None:
            if source_states is None:
                # self-attn
                # (batch_size, num_heads, key_length, dim_per_head)
                # if len(past_key_or_value.shape) == 3:
                #     past_key_or_value = self._transpose_for_scores(past_key_or_value)
                hidden_states = torch.cat([past_key_or_value, hidden_states], dim=2)
            else:
                # cross-attn
                hidden_states = past_key_or_value
        return hidden_states

    def _position_bias(
        self,
        position_bias: Optional[torch.Tensor],
        seq_lengths: Tuple[int, int, int],
        past_key_states: Optional[torch.Tensor],
        attention_scores: torch.Tensor,
    ) -> torch.Tensor:
        seq_length, real_seq_length, key_length = seq_lengths

        if position_bias is None:
            if self.relative_attention_num_buckets is not None:
                position_bias = self.compute_bias(real_seq_length, key_length)
            else:
                position_bias = torch.zeros(
                    (1, self.num_attention_heads, real_seq_length, key_length),
                    device=attention_scores.device,
                    dtype=attention_scores.dtype,
                )

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_states is not None:
                position_bias = position_bias[:, :, -seq_length:, :]
        return position_bias

    def _get_attention_probs(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        attention_mask: Optional[torch.BoolTensor],
        head_mask: Optional[torch.Tensor],
        seq_lengths: Tuple[int, int, int],
        position_bias: Optional[torch.Tensor] = None,
        past_key_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_scores = self.attn(query_layer, key_layer)

        position_bias = self._position_bias(
            position_bias, seq_lengths, past_key_states, attention_scores
        )

        if attention_mask is not None:
            # Shape: (batch_size, num_heads, seq_length, key_length)
            position_bias = apply_mask(position_bias, attention_mask)
        attention_scores += position_bias

        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = F.dropout(attention_probs, p=self.dropout, training=self.training)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        return attention_probs, position_bias

    def _output_layer(self, attention_probs: torch.Tensor, value_layer: torch.Tensor) -> torch.Tensor:
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if hasattr(self, "output"):
            context_layer = self.output(context_layer)

        return context_layer

    def _get_lengths(
        self,
        query_states: torch.Tensor,
        past_key_states: Optional[torch.Tensor] = None,
        source_states: Optional[torch.Tensor] = None,
        query_length: Optional[int] = None,
    ) -> Tuple[int, int, int]:

        seq_length = query_states.shape[1]
        effective_seq_len = seq_length

        if past_key_states is not None:
            # TODO: query_length from up the stack: move logic here.
            # TODO: clarify the logic here in terms of encoder/decoder case.
            effective_seq_len += past_key_states.shape[2] if query_length is None else query_length

        key_length = effective_seq_len if source_states is None else source_states.shape[1]

        return (seq_length, effective_seq_len, key_length)

    def forward(
        self,
        query_states: torch.Tensor,
        past_key_states: Optional[torch.Tensor] = None,
        past_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        source_states: Optional[torch.Tensor] = None,
        source_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        query_length: Optional[int] = None,
    ) -> AttentionOutput:
        """
        # Parameters

        query_states : `torch.Tensor`
            Shape `batch_size x seq_len x hidden_dim`
        past_key_states : `torch.Tensor`, optional
            Shape `batch_size x seq_len x hidden_dim`
            These are the key_states from the previous step of the decoder.
        past_value_states : `torch.Tensor`, optional
            Shape `batch_size x seq_len x hidden_dim`
            These are the value_states from the previous step of the decoder.
        attention_mask : `torch.BoolTensor`, optional
            Shape `batch_size x seq_len`
        source_states : `torch.Tensor`, optional
            Shape `batch_size x source_seq_len x hidden_dim`
            This is from the final state of attention over the source (encoder);
            it is passed when this module is being used for cross-attention.
        source_attention_mask : `torch.BoolTensor`, optional
            Shape `batch_size x source_seq_len`
        head_mask : `torch.BoolTensor`, optional
        position_bias : `torch.Tensor`, optional
        output_attentions : `bool`
            Whether to also return the attention probabilities, default = `False`

        !!! Note
            `source_states` needs to be passed in case of cross-attention.

        """
        query_layer = self._query_layer(query_states)

        key_layer = self._project(
            query_states,
            self.key,
            source_states,
            past_key_states,
        )

        value_layer = self._project(
            query_states,
            self.value,
            source_states,
            past_value_states,
        )

        if self.is_cross_attention:
            attention_mask = source_attention_mask

        seq_lengths = self._get_lengths(query_states, past_key_states, source_states, query_length)

        attention_probs, position_bias = self._get_attention_probs(
            query_layer,
            key_layer,
            attention_mask,
            head_mask,
            seq_lengths,
            position_bias,
            past_key_states,
        )

        context_layer = self._output_layer(attention_probs, value_layer)

        present_key_value_state = (
            (key_layer, value_layer) if (self.is_decoder and use_cache) else None
        )

        if not output_attentions:
            attention_probs = None

        outputs = AttentionOutput(
            context_layer, present_key_value_state, position_bias, attention_probs
        )

        return outputs

    @staticmethod
    def _relative_position_bucket(
        relative_position: IntT,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> IntT:
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the
        attended-to position. If bidirectional=False, then positive relative positions are invalid. We use smaller
        buckets for small absolute relative_position and larger buckets for larger absolute relative_positions. All
        relative positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the
        same bucket. This should allow for more graceful generalization to longer sequences than the model has been
        trained on.

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range
            [0, num_buckets)
        """
        relative_buckets = relative_position.new_zeros(relative_position.shape)
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int) -> FloatT:
        """Compute binned relative position bias"""
        context_position = torch.arange(query_length, dtype=torch.l