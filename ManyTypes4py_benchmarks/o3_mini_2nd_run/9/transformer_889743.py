from dataclasses import dataclass
from typing import Optional, Tuple, List, Any, Callable
import torch as pt
import sockeye.layers
from sockeye import constants as C
from . import config

@dataclass
class TransformerConfig(config.Config):
    decoder_type: str = C.TRANSFORMER_TYPE
    block_prepended_cross_attention: bool = False
    use_lhuc: bool = False
    depth_key_value: int = 0
    use_glu: bool = False

class TransformerEncoderBlock(pt.nn.Module):
    """
    A transformer encoder block consists self-attention and a feed-forward layer with pre/post process blocks
    in between.
    """

    def __init__(self, config: TransformerConfig, inference_only: bool = False, dtype: Optional[Any] = None, clamp_to_dtype: bool = False) -> None:
        super().__init__()
        self.pre_self_attention = TransformerProcessBlock(
            sequence=config.preprocess_sequence,
            dropout=config.dropout_prepost,
            num_hidden=config.model_size,
            dtype=dtype,
            clamp_to_dtype=clamp_to_dtype,
        )
        self.self_attention = sockeye.layers.MultiHeadSelfAttention(
            depth_att=config.model_size,
            heads=config.attention_heads,
            depth_out=config.model_size,
            dropout=config.dropout_attention,
            dtype=dtype,
            clamp_to_dtype=clamp_to_dtype,
        )
        self.post_self_attention = TransformerProcessBlock(
            sequence=config.postprocess_sequence,
            dropout=config.dropout_prepost,
            num_hidden=config.model_size,
            dtype=dtype,
            clamp_to_dtype=clamp_to_dtype,
        )
        self.pre_ff = TransformerProcessBlock(
            sequence=config.preprocess_sequence,
            dropout=config.dropout_prepost,
            num_hidden=config.model_size,
            dtype=dtype,
            clamp_to_dtype=clamp_to_dtype,
        )
        self.ff = TransformerFeedForward(
            num_hidden=config.feed_forward_num_hidden,
            num_model=config.model_size,
            act_type=config.act_type,
            dropout=config.dropout_act,
            use_glu=config.use_glu,
            inference_only=inference_only,
            dtype=dtype,
            clamp_to_dtype=clamp_to_dtype,
        )
        self.post_ff = TransformerProcessBlock(
            sequence=config.postprocess_sequence,
            dropout=config.dropout_prepost,
            num_hidden=config.model_size,
            dtype=dtype,
            clamp_to_dtype=clamp_to_dtype,
        )
        self.lhuc: Optional[pt.nn.Module] = None
        if config.use_lhuc:
            self.lhuc = sockeye.layers.LHUC(config.model_size, dtype=dtype)

    def forward(self, data: pt.Tensor, att_mask: Optional[pt.Tensor] = None) -> pt.Tensor:
        """
        :param data: Input tensor of shape (length, batch_size, hidden)
        :param att_mask: Optional data length mask of shape (batch_size * self.heads, 1, length)
                         to mask self-attention scores. True for padding positions.
        """
        data_self_att, _ = self.self_attention(
            inputs=self.pre_self_attention(data),
            previous_states=None,
            mask=att_mask,
            bias=None,
        )
        data = self.post_self_attention(data_self_att, data)
        data_ff = self.ff(self.pre_ff(data))
        data = self.post_ff(data_ff, data)
        if self.lhuc is not None:
            data = self.lhuc(data)
        return data

class TransformerDecoderBlock(pt.nn.Module):
    """
    A transformer decoder block consists of an autoregressive attention block, encoder attention,
    and a feed-forward layer with pre/post process blocks in between.
    """

    def __init__(self, config: TransformerConfig, inference_only: bool, dtype: Optional[Any] = None, clamp_to_dtype: bool = False) -> None:
        super().__init__()
        self.decoder_type: str = config.decoder_type
        self.inference_only: bool = inference_only
        self.autoregr_layer: pt.nn.Module
        if self.decoder_type == C.TRANSFORMER_TYPE:
            self.autoregr_layer = sockeye.layers.MultiHeadSelfAttention(
                depth_att=config.model_size,
                heads=config.attention_heads,
                depth_out=config.model_size,
                dropout=config.dropout_attention,
                dtype=dtype,
                clamp_to_dtype=clamp_to_dtype,
            )
        elif self.decoder_type == C.SSRU_TRANSFORMER:
            self.autoregr_layer = sockeye.layers.SSRU(
                model_size=config.model_size,
                inference_only=inference_only,
                dtype=dtype,
                clamp_to_dtype=clamp_to_dtype,
            )
        else:
            raise ValueError('Invalid decoder type.')
        self.pre_autoregr_layer = TransformerProcessBlock(
            sequence=config.preprocess_sequence,
            dropout=config.dropout_prepost,
            num_hidden=config.model_size,
            dtype=dtype,
            clamp_to_dtype=clamp_to_dtype,
        )
        self.post_autoregr_layer = TransformerProcessBlock(
            sequence=config.postprocess_sequence,
            dropout=config.dropout_prepost,
            num_hidden=config.model_size,
            dtype=dtype,
            clamp_to_dtype=clamp_to_dtype,
        )
        self.pre_enc_attention = TransformerProcessBlock(
            sequence=config.preprocess_sequence,
            dropout=config.dropout_prepost,
            num_hidden=config.model_size,
            dtype=dtype,
            clamp_to_dtype=clamp_to_dtype,
        )
        self.enc_attention = sockeye.layers.MultiHeadAttention(
            depth_att=config.model_size,
            heads=config.attention_heads,
            depth_out=config.model_size,
            dropout=config.dropout_attention,
            depth_key_value=config.depth_key_value,
            dtype=dtype,
            clamp_to_dtype=clamp_to_dtype,
        )
        self.post_enc_attention = TransformerProcessBlock(
            sequence=config.postprocess_sequence,
            dropout=config.dropout_prepost,
            num_hidden=config.model_size,
            dtype=dtype,
            clamp_to_dtype=clamp_to_dtype,
        )
        self.pre_ff = TransformerProcessBlock(
            sequence=config.preprocess_sequence,
            dropout=config.dropout_prepost,
            num_hidden=config.model_size,
            dtype=dtype,
            clamp_to_dtype=clamp_to_dtype,
        )
        self.ff = TransformerFeedForward(
            num_hidden=config.feed_forward_num_hidden,
            num_model=config.model_size,
            act_type=config.act_type,
            dropout=config.dropout_act,
            use_glu=config.use_glu,
            inference_only=inference_only,
            dtype=dtype,
            clamp_to_dtype=clamp_to_dtype,
        )
        self.post_ff = TransformerProcessBlock(
            sequence=config.postprocess_sequence,
            dropout=config.dropout_prepost,
            num_hidden=config.model_size,
            dtype=dtype,
            clamp_to_dtype=clamp_to_dtype,
        )
        self.lhuc: Optional[pt.nn.Module] = None
        if config.use_lhuc:
            self.lhuc = sockeye.layers.LHUC(config.model_size, dtype=dtype)

    def set_inference_only(self, inference_only: bool) -> None:
        """
        Set inference_only.
        """
        self.inference_only = inference_only
        if self.decoder_type == C.SSRU_TRANSFORMER:
            # type: ignore[attr-defined]
            self.autoregr_layer.set_inference_only(inference_only)

    @property
    def num_state_tensors(self) -> int:
        """ Number of state tensors returned by the layer """
        # type: ignore[attr-defined]
        return self.autoregr_layer.num_state_tensors

    @property
    def needs_mask(self) -> bool:
        """ Whether the block makes use of a mask tensor or not """
        # type: ignore[attr-defined]
        return self.autoregr_layer.needs_mask

    def get_states_shape(self, batch_size: int) -> Tuple[int, ...]:
        """
        :param batch_size: current batch size
        :return: dimensions of an output state (assuming all of them have the same shape)
        """
        # type: ignore[attr-defined]
        return self.autoregr_layer.get_state_shape(batch_size)

    def forward(
        self,
        target: pt.Tensor,
        target_mask: pt.Tensor,
        source: pt.Tensor,
        source_mask: pt.Tensor,
        autoregr_states: Any,
        enc_att_kv: Optional[pt.Tensor] = None,
    ) -> Tuple[pt.Tensor, List[Any]]:
        target_autoregr, *new_autoregr_states = self.autoregr_layer(
            inputs=self.pre_autoregr_layer(target),
            previous_states=autoregr_states,
            mask=target_mask,
        )
        target = self.post_autoregr_layer(target_autoregr, target)
        target_enc_att = self.enc_attention(
            queries=self.pre_enc_attention(target),
            key_values=source,
            mask=source_mask,
            projected_memory_kv=enc_att_kv,
        )
        target = self.post_enc_attention(target_enc_att, target)
        target_ff = self.ff(self.pre_ff(target))
        target = self.post_ff(target_ff, target)
        if self.lhuc:
            target = self.lhuc(target)
        return (target, new_autoregr_states)

class TransformerProcessBlock(pt.nn.Module):
    """
    Block to perform pre/post processing on layer inputs.
    The processing steps are determined by the sequence argument, which can contain one of the three operations:
    n: layer normalization
    r: residual connection
    d: dropout
    """

    def __init__(
        self,
        sequence: str,
        dropout: float,
        num_hidden: int = 0,
        dtype: Optional[Any] = None,
        clamp_to_dtype: bool = False,
    ) -> None:
        super().__init__()
        self.sequence: str = sequence
        self.clamp_to_dtype: bool = clamp_to_dtype
        self.layer_norm: Optional[pt.nn.LayerNorm] = None
        if 'n' in sequence:
            self.layer_norm = pt.nn.LayerNorm(num_hidden, eps=1e-06, dtype=dtype)
        self.dropout: float = dropout
        self.drop: pt.nn.Dropout = pt.nn.Dropout(p=dropout)

    def forward(self, data: pt.Tensor, prev: Optional[pt.Tensor] = None) -> pt.Tensor:
        """
        Apply processing sequence to data with optional previous input.

        :param data: Input data. Shape: (batch, length, num_hidden).
        :param prev: Previous data. Shape: (batch, length, num_hidden).
        :return: Processed data. Shape: (batch, length, num_hidden).
        """
        if not self.sequence:
            return data
        if prev is None:
            assert 'r' not in self.sequence, 'Residual connection not allowed if no previous value given.'
        for step in self.sequence:
            if step == 'r':
                data = data + prev  # type: ignore
            elif step == 'n':
                data = self.layer_norm(data)  # type: ignore
            elif step == 'd':
                data = self.drop(data)
            else:
                raise ValueError('Unknown step in sequence: %s' % step)
        if self.clamp_to_dtype:
            data = sockeye.layers.clamp_to_dtype_min_max(data)
        return data

class TransformerFeedForward(pt.nn.Module):

    def __init__(
        self,
        num_hidden: int,
        num_model: int,
        act_type: str,
        dropout: float,
        use_glu: bool = False,
        inference_only: bool = False,
        dtype: Optional[Any] = None,
        clamp_to_dtype: bool = False,
    ) -> None:
        super().__init__()
        self.use_glu: bool = use_glu
        self.clamp_to_dtype: bool = clamp_to_dtype
        self.ff1: pt.nn.Linear = pt.nn.Linear(in_features=num_model, out_features=num_hidden, dtype=dtype)
        self.act: Callable[[pt.Tensor], pt.Tensor] = sockeye.layers.get_activation(act_type)
        if self.use_glu:
            self.linear: pt.nn.Linear = pt.nn.Linear(in_features=num_model, out_features=num_hidden, dtype=dtype)  # type: ignore
        self.drop: pt.nn.Dropout = pt.nn.Dropout(p=dropout)
        self.ff2: pt.nn.Linear = pt.nn.Linear(in_features=num_hidden, out_features=num_model, dtype=dtype)

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        h: pt.Tensor = self.ff1(x)
        h = self.act(h)
        if self.use_glu:
            h = h * self.linear(x)  # type: ignore
        h = self.drop(h)
        y: pt.Tensor = self.ff2(h)
        if self.clamp_to_dtype:
            y = sockeye.layers.clamp_to_dtype_min_max(y)
        return y

class AutoRegressiveMask(pt.nn.Module):

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        """ Input tensor with length on dimension 1 """
        mask: pt.Tensor = pt.full((x.shape[1], x.shape[1]), fill_value=1, device=x.device, dtype=pt.bool)
        mask = pt.triu(mask, diagonal=1)
        return mask.detach()