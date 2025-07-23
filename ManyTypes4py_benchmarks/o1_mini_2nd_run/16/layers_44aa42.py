import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import torch as pt
import torch.nn.functional as F
from sockeye import constants as C, utils
from . import config

logger = logging.getLogger(__name__)

def get_activation(act_type: str) -> pt.nn.Module:
    if act_type == C.SWISH1:
        return pt.nn.SiLU()
    if act_type == C.GELU:
        return pt.nn.GELU()
    return pt.nn.ReLU()

class LHUC(pt.nn.Module):
    """
    Learning Hidden Unit Contribution

    David Vilar. "Learning Hidden Unit Contribution for Adapting Neural
    Machine Translation Models" NAACL 2018

    :param num_hidden: Number of hidden units of the layer to be modified.
    """

    def __init__(self, num_hidden: int, dtype: Optional[torch.dtype] = None) -> None:
        super().__init__()
        self.weight: pt.nn.Parameter = pt.nn.Parameter(pt.empty(num_hidden, dtype=dtype))

    def forward(self, data: pt.Tensor) -> pt.Tensor:
        weight = 2 * pt.sigmoid(self.weight)
        return weight * data

class OutputLayer(pt.nn.Module):
    """
    Final output layer of seq2seq models. Supports vocabulary selection that caches reduced weight/bias
    across multiple invocations if selected vocabulary ids do not change.

    :param hidden_size: Input hidden size.
    :param vocab_size: Target vocabulary size.
    :param weight: Optional shared weight Parameter.
    :param dtype: Torch data type for parameters.
    """

    def __init__(
        self, 
        hidden_size: int, 
        vocab_size: int, 
        weight: Optional[pt.Tensor] = None, 
        dtype: Optional[torch.dtype] = None
    ) -> None:
        super().__init__()
        self.vocab_size: int = vocab_size
        self.in_features: int = hidden_size
        self.out_features: int = vocab_size
        if weight is None:
            self.weight: pt.nn.Parameter = pt.nn.Parameter(pt.empty(vocab_size, hidden_size, dtype=dtype))
        else:
            self.weight = weight
        self.bias: pt.nn.Parameter = pt.nn.Parameter(pt.empty(vocab_size, dtype=dtype))
        self.previous_slice_ids: pt.Tensor = pt.empty(0)
        self.reduced_weight: pt.Tensor = pt.empty(0)
        self.reduced_bias: pt.Tensor = pt.empty(0)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None} dtype={self.weight.dtype}'

    def _is_new_slice(self, x: pt.Tensor) -> bool:
        if x.size() != self.previous_slice_ids.size() or pt.any(x != self.previous_slice_ids):
            return True
        return False

    def _take_slice(self, vocab_slice_ids: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        weight = self.weight[vocab_slice_ids]
        bias = self.bias[vocab_slice_ids]
        return (weight, bias)

    def forward(self, data: pt.Tensor, vocab_slice_ids: Optional[pt.Tensor] = None) -> pt.Tensor:
        if vocab_slice_ids is not None:
            if self._is_new_slice(vocab_slice_ids):
                self.previous_slice_ids = vocab_slice_ids
                weight, bias = self.reduced_weight, self.reduced_bias = self._take_slice(vocab_slice_ids)
            else:
                weight, bias = (self.reduced_weight, self.reduced_bias)
        else:
            weight, bias = (self.weight, self.bias)
        return F.linear(data, weight, bias)

class KNN(pt.nn.Module):
    """
    An alternative output layer that can produce a output distribution over the vocabulary
    by using the decoder hidden state to query into an index.
    For more details, see: https://arxiv.org/abs/2010.00710.

    :param keys_index: faiss index used for k-NN query.
    :param vals: a list of word indexes that maps key ids to their corresponding vocabulary ids.
    :param vocab_size: the size of the output vocabulary.
    :param k: number of candidates to be retrieved by k-nearest neighbors query.
    :param temperature: temperature that controls the smoothness of the output distribution.
    :param state_store: an optional state store object that is used to compute the exact distance
                        between the query and the index.
    """

    def __init__(
        self, 
        keys_index: any, 
        vals: np.ndarray, 
        vocab_size: int, 
        k: int = 64, 
        temperature: float = 10.0, 
        state_store: Optional[np.ndarray] = None
    ) -> None:
        super().__init__()
        self.keys_index = keys_index
        self.vals: np.ndarray = vals
        self.vocab_size: int = vocab_size
        self.k: int = k
        self.temperature: float = temperature
        self.state_store = state_store

    def forward(self, data: pt.Tensor) -> pt.Tensor:
        distances: np.ndarray
        indices: np.ndarray
        distances, indices = self.keys_index.search(data.cpu().numpy().astype(np.float32), self.k)
        y: np.ndarray = self.vals[(indices + 1) % len(self.vals)]
        y[y == C.BOS_ID] = C.EOS_ID
        if self.state_store is not None:
            raw_keys: pt.Tensor = pt.from_numpy(self.state_store[indices]).to(device=data.device)
            distances = pt.norm(data.unsqueeze(1) - raw_keys, p=2, dim=-1)
        else:
            distances = np.sqrt(distances)
            distances = pt.from_numpy(distances).to(device=data.device)
        y_tensor: pt.Tensor = pt.from_numpy(y).to(device=data.device).long()
        probs = pt.exp(-distances / self.temperature)
        full_probs = pt.zeros((data.shape[0], self.vocab_size), device=data.device)
        full_probs.scatter_add_(src=probs, index=y_tensor.squeeze(2), dim=-1)
        z = pt.sum(full_probs, dim=-1).unsqueeze(-1)
        z[z < C.KNN_EPSILON] = C.KNN_EPSILON
        full_probs.div_(z)
        return full_probs

@dataclass
class LengthRatioConfig(config.Config):
    pass

class LengthRatio(pt.nn.Module):
    """
    Defines the length-ratio prediction layer of Sockeye.

    :param hidden_size: Encoder hidden size.
    :param num_layers: Number of layers.
    :param dtype: Torch data type for parameters.
    """

    def __init__(self, hidden_size: int, num_layers: int, dtype: Optional[torch.dtype] = None) -> None:
        utils.check_condition(num_layers >= 1, "LengthRatio's num_layers has to be >=1.")
        super().__init__()
        self.num_layers: int = num_layers
        self.hidden_size: int = hidden_size
        modules: List[pt.nn.Module] = []
        for _ in range(num_layers - 1):
            modules.append(pt.nn.Linear(in_features=hidden_size, out_features=hidden_size, dtype=dtype))
            modules.append(pt.nn.Tanh())
        modules.append(pt.nn.Linear(in_features=hidden_size, out_features=1, dtype=dtype))
        modules.append(pt.nn.Softplus())
        self.layers: pt.nn.Sequential = pt.nn.Sequential(*modules)

    def forward(
        self, 
        source_encoded: pt.Tensor, 
        source_encoded_length: pt.Tensor
    ) -> pt.Tensor:
        """
        Transformation to the length ratio. Returns a vector.

        :param source_encoded: Encoder representation for n elements. Shape: (n, source_encoded_length, hidden_size).
        :param source_encoded_length: A vector of encoded sequence lengths. Shape: (n,).
        :return: Predictions of the ratio length(hypothesis)/length(reference). Shape(n, 1).
        """
        mask: pt.Tensor = pt.arange(source_encoded.size()[1], device=source_encoded_length.device)[None, :, None] >= source_encoded_length[:, None, None]
        source_masked: pt.Tensor = source_encoded.masked_fill(mask, 0.0)
        data: pt.Tensor = source_masked.sum(dim=1, keepdim=False) / source_encoded_length.unsqueeze(1)
        data = self.layers(data).squeeze(1)
        return data

@pt.jit.script
def interleaved_matmul_encdec_qk(q: pt.Tensor, kv: pt.Tensor, heads: int) -> pt.Tensor:
    """
    Simple port of npx.interleaved_matmul_encdec_qk with PyTorch.

    :param q: (qlen, batch, hidden)
    :param kv: (kvlen, batch, hidden * 2) -- interleaved
    :param heads: number of attention heads
    :return: (batch * heads, qlen, klen)
    """
    qlen, batch, hidden = q.size()
    head_dim = hidden // heads
    q = q.contiguous().view(qlen, batch * heads, head_dim).transpose(0, 1)
    q = q * head_dim ** (-0.5)
    tmp = kv.reshape(-1, batch, heads, 2, head_dim)
    k = tmp[:, :, :, 0, :]
    k = k.permute(1, 2, 3, 0)
    k = k.reshape(batch * heads, head_dim, -1)
    return pt.bmm(q, k)

@pt.jit.script
def interleaved_matmul_encdec_valatt(kv: pt.Tensor, att: pt.Tensor, heads: int) -> pt.Tensor:
    """
    Simple port of npx.interleaved_matmul_encdec_valatt with PyTorch.
    There is probably something to be gained by using views more
    efficiently but this is placeholder code anyway.

    :param kv: (kvlen, batch, hidden * 2)
    :param att: (batch * heads, qlen, kvlen)
    :param heads: number of attention heads
    :return: (qlen, batch, hidden)
    """
    kvlen, batch, hidden2 = kv.size()
    hidden = hidden2 // 2
    head_dim = hidden // heads
    tmp = kv.reshape(kvlen, batch, heads, 2, -1)
    v = tmp[:, :, :, 1, :]
    v = v.permute(1, 2, 0, 3)
    v = v.reshape(-1, kvlen, head_dim)
    output = pt.bmm(att, v)
    output = output.transpose(0, 1).contiguous().view(-1, batch, hidden)
    return output

class DotAttentionCell(pt.nn.Module):

    def __init__(self, dropout: float = 0.0, heads: int = 1) -> None:
        super().__init__()
        self.dropout: pt.nn.Dropout = pt.nn.Dropout(p=dropout)
        self.heads: int = heads

    def forward(
        self, 
        queries: pt.Tensor, 
        key_values: pt.Tensor, 
        mask: Optional[pt.Tensor] = None
    ) -> pt.Tensor:
        """
        :param queries: Query tensor of shape (query_length, batch_size, hidden)
        :param key_values: Interleaved Key & value tensor of shape (key/value_length, batch_size, hidden * 2)
        :param mask: Optional boolean tensor for attention masking of shape (batch * heads, <qlen>, <kvlen>).
                     If this is cross-attention, <qlen> dimension can be 1 for broadcasting,
                     i.e. (batch * heads, 1, kvlen). For self-attention on the decoder side an autoregressive mask
                     should be provided of shape (1, len, len) or (len, len).
                     Value of this mask is True for positions that should be masked out (padding positions),
                     False for valid positions.
        """
        logits: pt.Tensor = interleaved_matmul_encdec_qk(queries, key_values, heads=self.heads)
        if mask is not None:
            logits = logits.masked_fill(mask, -C.LARGE_VALUES[logits.dtype]
                                        )
        probs: pt.Tensor = F.softmax(logits, dim=-1)
        if self.dropout is not None:
            probs = self.dropout(probs)
        return interleaved_matmul_encdec_valatt(key_values, probs, heads=self.heads)

def prepare_source_length_mask(
    lengths: pt.Tensor, 
    heads: int, 
    max_length: int, 
    expand: bool = True, 
    mask_prepended_tokens: bool = False
) -> pt.Tensor:
    """
    Prepare source length masks where positions of invalid tokens are marked as True.

    :param lengths: Total source length and prepended source length. Shape: (batch_size, 2)
    :param heads: Number of attention heads.
    :param max_length: Maximum sequence length.
    :param expand: Expand to the heads.
    :param mask_prepended_tokens: Mask prepended tokens.
    :return: Source length mask.
    """
    mask: pt.Tensor = ~(pt.arange(max_length, device=lengths.device).unsqueeze(0) < lengths[:, :1])
    if mask_prepended_tokens:
        prepended_token_mask: pt.Tensor = pt.arange(max_length, device=lengths.device).unsqueeze(0) < lengths[:, 1:2]
        mask = mask | prepended_token_mask
    if expand:
        mask = mask.unsqueeze(1).expand(-1, heads, -1).reshape((-1, max_length)).unsqueeze(1)
    return mask

class MultiHeadAttentionBase(pt.nn.Module):
    """
    Base class for Multi-head attention.

    :param depth_att: Attention depth / number of hidden units.
    :param heads: Number of attention heads.
    :param depth_out: Output depth / number of output units.
    :param dropout: Dropout probability on attention scores.
    :param dtype: Torch data type for parameters.
    :param clamp_to_dtype: Avoid -inf/inf by clamping outputs to min/max finite
                           values for their dtype.
    """

    def __init__(
        self, 
        depth_att: int = 512, 
        heads: int = 8, 
        depth_out: int = 512, 
        dropout: float = 0.0, 
        dtype: Optional[torch.dtype] = None, 
        clamp_to_dtype: bool = False
    ) -> None:
        super().__init__()
        utils.check_condition(depth_att % heads == 0, 'Number of heads (%d) must divide attention depth (%d)' % (heads, depth_att))
        self.depth: int = depth_att
        self.heads: int = heads
        self.depth_out: int = depth_out
        self.depth_per_head: int = self.depth // self.heads
        self.clamp_to_dtype: bool = clamp_to_dtype
        self.dot_att: DotAttentionCell = DotAttentionCell(dropout=dropout, heads=heads)
        self.ff_out: pt.nn.Linear = pt.nn.Linear(in_features=depth_att, out_features=depth_out, bias=False, dtype=dtype)

    def _attend(
        self, 
        queries: pt.Tensor, 
        key_values: pt.Tensor, 
        mask: Optional[pt.Tensor] = None
    ) -> pt.Tensor:
        """
        Returns context vectors of multi-head dot attention.

        :param queries: Query tensor. Shape: (queries_length, batch_size, depth).
        :param key_values: Keys/Values. Shape: (keys_values_length, batch_size, depth * 2).
        :param mask: Optional boolean attention mask. See DotAttentionCell for shape requirements.
        :return: Context vectors. Shape: (batch_size, query_max_length, output_depth).
        """
        contexts: pt.Tensor = self.dot_att(queries=queries, key_values=key_values, mask=mask)
        contexts = self.ff_out(contexts)
        if self.clamp_to_dtype:
            contexts = clamp_to_dtype_min_max(contexts)
        return contexts

class AutoregressiveLayer(pt.nn.Module):

    @property
    @abstractmethod
    def num_state_tensors(self) -> int:
        """ Number of state tensors returned by the layer """
        raise NotImplementedError

    @property
    @abstractmethod
    def needs_mask(self) -> bool:
        """ Whether the layer makes use of a mask tensor or not """
        raise NotImplementedError

    @abstractmethod
    def get_state_shape(self, batch_size: int) -> Tuple[int, int, int]:
        """
        :param batch_size: current batch size
        :return: dimensions of each output state (assuming all of them have the same shape)
        """
        raise NotImplementedError

    @abstractmethod
    def set_inference_only(self, inference_only: bool) -> None:
        """
        Set inference_only.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, 
        inputs: pt.Tensor, 
        previous_states: Union[List[pt.Tensor], Tuple[pt.Tensor, ...]], 
        *args: Any
    ) -> Tuple[pt.Tensor, Union[List[pt.Tensor], Tuple[pt.Tensor, ...]]]:
        """
        :param inputs: layer input
        :param previous_states: Previous states array or list of arrays
        :param args: layer-specific arguments and/or arguments to be ignored
        :return: layer output and new states
        """
        raise NotImplementedError

class MultiHeadSelfAttention(MultiHeadAttentionBase, AutoregressiveLayer):
    """
    Multi-head self-attention. Independent linear projections of inputs serve as
    queries, keys, and values for the attention.

    :param depth_att: Attention depth / number of hidden units.
    :param heads: Number of attention heads.
    :param depth_out: Output depth / number of output units.
    :param dropout: Dropout probability on attention scores.
    :param dtype: Torch data type for parameters.
    :param clamp_to_dtype: Avoid -inf/inf by clamping outputs to min/max finite
                           values for their dtype.
    """

    def __init__(
        self, 
        depth_att: int = 512, 
        heads: int = 8, 
        depth_out: int = 512, 
        dropout: float = 0.0, 
        dtype: Optional[torch.dtype] = None, 
        clamp_to_dtype: bool = False
    ) -> None:
        super().__init__(depth_att, heads, depth_out, dropout, dtype, clamp_to_dtype)
        self.depth_att: int = depth_att
        self.ff_in: pt.nn.Linear = pt.nn.Linear(in_features=depth_att, out_features=depth_att * 3, bias=False, dtype=dtype)
        self._drop_p: float = dropout
        self.kv_interleaved: bool = False

    def set_inference_only(self, inference_only: bool) -> None:
        """
        Set inference_only. Not needed for MultiHeadSelfAttention.
        """
        raise NotImplementedError

    def separate_kv(self) -> None:
        """ write kv input projection parameters in non-interleaved format (compatible with F.multi_head_attention) """
        assert self.kv_interleaved
        with pt.no_grad():
            kv: pt.Tensor = self.ff_in.weight.data[self.depth:, :]
            k, v = kv.view(self.heads, 2 * self.depth_per_head, self.depth).split(self.depth_per_head, dim=1)
            k = k.reshape(self.depth, self.depth)
            v = v.reshape(self.depth, self.depth)
        self.ff_in.weight.data[self.depth:, :] = pt.cat((k, v), dim=0)
        self.kv_interleaved = False

    def interleave_kv(self) -> None:
        """ write kv input projection parameters in interleaved format (compatible with interleaved matmul) """
        assert not self.kv_interleaved
        with pt.no_grad():
            _, k, v = self.ff_in.weight.data.split(self.depth, dim=0)
            k = k.reshape(self.heads, -1, self.depth)
            v = v.reshape(self.heads, -1, self.depth)
        self.ff_in.weight.data[self.depth:, :] = pt.cat((k, v), dim=1).reshape(self.depth * 2, self.depth)
        self.kv_interleaved = True

    def train(self, mode: bool = True) -> pt.nn.Module:
        """
        Overrides super().train() to ensure key-value parameters are stored in non-interleaved format during training
        and interleaved format during inference (mod.eval()).
        """
        if mode and self.kv_interleaved:
            self.separate_kv()
        elif not mode and (not self.kv_interleaved):
            self.interleave_kv()
        return super().train(mode)

    @property
    def num_state_tensors(self) -> int:
        """ Number of state tensors returned by the layer """
        return 1

    @property
    def needs_mask(self) -> bool:
        """ Whether the layer makes use of a mask tensor or not """
        return True

    def get_state_shape(self, batch_size: int) -> Tuple[int, int, int]:
        """
        :param batch_size: current batch size
        :return: dimensions of each output state (assuming all of them have the same shape)
        """
        return (0, batch_size, self.depth_out * 2)

    def forward(
        self, 
        inputs: pt.Tensor, 
        previous_states: Optional[Union[List[pt.Tensor], Tuple[pt.Tensor, ...]]] = None, 
        mask: Optional[pt.Tensor] = None, 
        **args: Any
    ) -> Tuple[pt.Tensor, Union[List[pt.Tensor], Tuple[pt.Tensor, ...]]]:
        """
        Computes multi-head attention on a set of inputs, serving as queries, keys, and values.
        If sequence lengths are provided, they will be used to mask the attention scores.
        A bias mask may also be used to mask the attention scores.
        May also use a cache of previously computed inputs.
        Returns a tensor of shape (max_length, batch, output_depth).

        :param inputs: Input Data. Shape: (length, batch, input_depth).
        :param previous_states: Optional list with two tensors - previous input's keys and values.
                                Shape: 2 * (batch, max_length+1, depth_att).
        :param mask: Optional attention mask. See DotAttentionCell for shape information.
        :return: tensor of shape (max_length, batch, output_depth).
        """
        if self.training:
            assert not self.kv_interleaved
            contexts, _ = F.multi_head_attention_forward(
                query=inputs, 
                key=inputs, 
                value=inputs, 
                embed_dim_to_check=self.depth, 
                num_heads=self.heads, 
                in_proj_weight=self.ff_in.weight, 
                in_proj_bias=None, 
                bias_k=None, 
                bias_v=None, 
                add_zero_attn=False, 
                dropout_p=self._drop_p, 
                out_proj_weight=self.ff_out.weight, 
                out_proj_bias=self.ff_out.bias, 
                training=self.training, 
                key_padding_mask=None, 
                need_weights=False, 
                attn_mask=mask, 
                use_separate_proj_weight=False, 
                q_proj_weight=None, 
                k_proj_weight=None, 
                v_proj_weight=None
            )
            return (contexts, contexts)
        else:
            proj: pt.Tensor = self.ff_in(inputs)
            queries, states = proj.split((self.depth_att, 2 * self.depth_att), dim=2)
            if previous_states is not None:
                states = pt.cat(previous_states, states, dim=0)
            return (self._attend(queries=queries, key_values=states, mask=mask), states)

class MultiHeadAttention(MultiHeadAttentionBase):
    """
    Multi-head attention layer for queries independent from keys/values.

    :param depth_att: Attention depth / number of hidden units.
    :param heads: Number of attention heads.
    :param depth_out: Output depth / number of output units.
    :param depth_key_value: Dimension of input key and value vectors.
    :param dropout: Dropout probability on attention scores.
    :param dtype: Torch data type for parameters.
    :param clamp_to_dtype: Avoid -inf/inf by clamping outputs to min/max finite
                           values for their dtype.
    """

    def __init__(
        self, 
        depth_att: int = 512, 
        heads: int = 8, 
        depth_out: int = 512, 
        dropout: float = 0.0, 
        depth_key_value: int = 512, 
        dtype: Optional[torch.dtype] = None, 
        clamp_to_dtype: bool = False
    ) -> None:
        super().__init__(depth_att, heads, depth_out, dropout, dtype, clamp_to_dtype)
        self.ff_q: pt.nn.Linear = pt.nn.Linear(in_features=depth_out, out_features=depth_att, bias=False, dtype=dtype)
        self.ff_kv: pt.nn.Linear = pt.nn.Linear(in_features=depth_key_value, out_features=depth_att * 2, bias=False, dtype=dtype)
        self._drop_p: float = dropout
        self._depth_key_value: int = depth_key_value
        self.kv_interleaved: bool = False

    def separate_kv(self) -> None:
        """Writes kv input projection parameters in non-interleaved format (compatible with F.multi_head_attention). """
        assert self.kv_interleaved
        with pt.no_grad():
            k, v = self.ff_kv.weight.data.view(self.heads, 2 * self.depth_per_head, self._depth_key_value).split(self.depth_per_head, dim=1)
            k = k.reshape(self.depth, self._depth_key_value)
            v = v.reshape(self.depth, self._depth_key_value)
        self.ff_kv.weight.data[:] = pt.cat((k, v), dim=0)
        self.kv_interleaved = False

    def interleave_kv(self) -> None:
        """Writes kv input projection parameters in interleaved format (compatible with interleaved matmul). """
        assert not self.kv_interleaved
        with pt.no_grad():
            k, v = self.ff_kv.weight.data.split(self.depth, dim=0)
            k = k.reshape(self.heads, -1, self.depth)
            v = v.reshape(self.heads, -1, self.depth)
        self.ff_kv.weight.data[:] = pt.cat((k, v), dim=1).reshape(self.depth * 2, self._depth_key_value)
        self.kv_interleaved = True

    def train(self, mode: bool = True) -> pt.nn.Module:
        """
        Overrides super().train() to ensure key-value parameters are stored in non-interleaved format during training
        and interleaved format during inference (mod.eval()).
        """
        if mode and self.kv_interleaved:
            self.separate_kv()
        elif not mode and (not self.kv_interleaved):
            self.interleave_kv()
        return super().train(mode)

    def forward(
        self, 
        queries: pt.Tensor, 
        key_values: pt.Tensor, 
        mask: Optional[pt.Tensor] = None, 
        projected_memory_kv: Optional[pt.Tensor] = None
    ) -> pt.Tensor:
        """
        Computes multi-head attention for queries given a memory tensor.
        If sequence lengths are provided, they will be used to mask the attention scores.
        A bias mask may also be used to mask the attention scores.
        Returns an tensor of shape (max_length, batch, output_depth).

        :param queries: Query tensor. Shape: (queries_length, batch, input_depth).
        :param key_values: Memory data to attend to. Shape: (key_values_length, batch, input_depth).
        :param mask: Optional attention mask. See DotAttentionCell for shape information.
        :param projected_memory_kv: Optional previously projected memory keys and values.
        :return: tensor of shape (query_seq_len, batch, output_depth).
        """
        if self.training:
            assert not self.kv_interleaved
            assert projected_memory_kv is None, 'caching not supported in training'
            contexts, _ = F.multi_head_attention_forward(
                query=queries, 
                key=key_values, 
                value=key_values, 
                embed_dim_to_check=self.depth, 
                num_heads=self.heads, 
                in_proj_weight=None, 
                in_proj_bias=None, 
                bias_k=None, 
                bias_v=None, 
                add_zero_attn=False, 
                dropout_p=self._drop_p, 
                out_proj_weight=self.ff_out.weight, 
                out_proj_bias=self.ff_out.bias, 
                training=self.training, 
                key_padding_mask=None, 
                need_weights=False, 
                attn_mask=mask, 
                use_separate_proj_weight=True, 
                q_proj_weight=self.ff_q.weight, 
                k_proj_weight=self.ff_kv.weight[:self.depth, :], 
                v_proj_weight=self.ff_kv.weight[self.depth:, :]
            )
            return contexts
        else:
            queries_proj: pt.Tensor = self.ff_q(queries)
            key_values_proj: pt.Tensor = projected_memory_kv if projected_memory_kv is not None else self.ff_kv(key_values)
            return self._attend(queries=queries_proj, key_values=key_values_proj, mask=mask)

def interleave_kv(module: pt.nn.Module) -> None:
    """ Writes kv input projection parameters in interleaved format (compatible with interleaved matmul). """
    if isinstance(module, MultiHeadAttention) or isinstance(module, MultiHeadSelfAttention):
        if not module.kv_interleaved:
            module.interleave_kv()

def separate_kv(module: pt.nn.Module) -> None:
    """ Writes kv input projection parameters in non-interleaved format (compatible with F.multi_head_attention). """
    if isinstance(module, MultiHeadAttention) or isinstance(module, MultiHeadSelfAttention):
        if module.kv_interleaved:
            module.separate_kv()

@pt.jit.script
def get_positional_embeddings(length: int, depth: int) -> pt.Tensor:
    utils.check_condition(depth % 2 == 0, 'Positional embeddings require an even embedding size it is however %d.' % depth)
    channels: pt.Tensor = pt.arange(depth // 2).unsqueeze(0)
    positions: pt.Tensor = pt.arange(0, length).unsqueeze(1)
    scaled_positions: pt.Tensor = positions / pt.pow(10000, 2 * channels / depth)
    sin: pt.Tensor = pt.sin(scaled_positions)
    cos: pt.Tensor = pt.cos(scaled_positions)
    encodings: pt.Tensor = pt.hstack([sin, cos])
    return encodings

class PositionalEmbeddings(pt.nn.Module):
    """
    Takes an encoded sequence and adds sinusoidal or learned positional embeddings as in Vaswani et al, 2017 to it.

    :param weight_type: type of embeddings, fixed or learned.
    :param num_embed: Embedding size.
    :param max_seq_len: Maximum sequence length.
    :param scale_up_input: If True, scales input data up by num_embed ** 0.5.
    :param scale_down_positions: If True, scales positional embeddings down by num_embed ** -0.5.
    :param dtype: Torch data type for parameters.
    """

    def __init__(
        self, 
        weight_type: str, 
        num_embed: int, 
        max_seq_len: int, 
        scale_up_input: bool, 
        scale_down_positions: bool, 
        dtype: Optional[torch.dtype] = None
    ) -> None:
        utils.check_condition(num_embed % 2 == 0, 'Positional embeddings require an even embedding size it is however %d.' % num_embed)
        super().__init__()
        self.weight_type: str = weight_type
        self.num_embed: int = num_embed
        self.max_seq_len: int = max_seq_len
        self.scale_up_input: bool = scale_up_input
        self.scale_down_positions: bool = scale_down_positions
        if self.weight_type == C.FIXED_POSITIONAL_EMBEDDING:
            weight: pt.Tensor = get_positional_embeddings(length=self.max_seq_len, depth=self.num_embed)
            if self.scale_down_positions:
                weight = weight * (self.num_embed ** (-0.5))
            if dtype is not None:
                weight = weight.to(dtype)
            self.weight: pt.nn.Parameter = pt.nn.Parameter(weight, requires_grad=False)
        elif self.weight_type == C.LEARNED_POSITIONAL_EMBEDDING:
            self.weight: pt.nn.Parameter = pt.nn.Parameter(pt.empty(self.max_seq_len, self.num_embed, dtype=dtype))
        else:
            raise ValueError(f"weight_type '{self.weight_type}' is not supported!")

    def forward(self, data: pt.Tensor, steps: Optional[pt.Tensor] = None) -> pt.Tensor:
        """
        Applies positional embeddings to input data.

        :param data: Input data. Shape: (batch, length or 1, num_embed)
        :param steps: Optional steps input. If given, shape is (batch_size or 1, seq_len,)
        :return: Data with positional embeddings added
        """
        if steps is None:
            pos_embedding: pt.Tensor = self.weight.unsqueeze(0)[:, :data.size(1)]
        else:
            steps = pt.clip(steps, max=self.max_seq_len - 1)
            pos_embedding = F.embedding(steps, self.weight)
        if self.weight_type == C.FIXED_POSITIONAL_EMBEDDING:
            pos_embedding = pos_embedding.detach()
        if self.scale_up_input:
            data = data * (self.num_embed ** 0.5)
        return data + pos_embedding

class SSRU(AutoregressiveLayer):
    """
    Simpler Simple Recurrent Unit

    Kim et al, "From Research to Production and Back: Ludicrously Fast Neural Machine Translation" WNGT 2019

    Variant of an LSTM cell aimed at reducing computational dependency across time steps.
    Formally described as:

    (1) f[t] = sigmoid(W1[t] * x[t] + b[t])
    (2) c[t] = f[t] . c[t-1] + (1 - f[t]) . W2[t] * x[t]
    (3) h = ReLU(c[t])

    where:
        . represents elementwise multiplication;
        x[t] is the input at time step t;
        f[t] is the output of the forget gate at time step t;
        c[t] is the cell state at time step t;
        h is the output of the unit.

    :param model_size: number of hidden units
    :param inference_only: flag used to indicate execution at inference time.
    :param dtype: Torch data type for parameters.
    :param clamp_to_dtype: Avoid -inf/inf by clamping outputs to min/max finite
                           values for their dtype.
    """

    def __init__(
        self, 
        model_size: int, 
        inference_only: bool, 
        dtype: Optional[torch.dtype] = None, 
        clamp_to_dtype: bool = False
    ) -> None:
        super().__init__()
        self.model_size: int = model_size
        self.clamp_to_dtype: bool = clamp_to_dtype
        self.set_inference_only(inference_only)
        self.forget_gate: pt.nn.Linear = pt.nn.Linear(in_features=model_size, out_features=model_size, bias=True, dtype=dtype)
        self.forget_gate_act: pt.nn.Sigmoid = pt.nn.Sigmoid()
        self.linear: pt.nn.Linear = pt.nn.Linear(in_features=model_size, out_features=model_size, bias=False, dtype=dtype)
        self.relu: pt.nn.ReLU = pt.nn.ReLU(inplace=False)

    def set_inference_only(self, inference_only: bool) -> None:
        """
        Set inference_only.
        """
        self.inference_only: bool = inference_only
        if inference_only:
            self.cell_state_transform = self._inference_cell_state_transform
        else:
            self.cell_state_transform = self._training_cell_state_transform

    @property
    def num_state_tensors(self) -> int:
        """ Number of state tensors returned by the layer """
        return 1

    @property
    def needs_mask(self) -> bool:
        """ Whether the layer makes use of a mask tensor or not """
        return False

    def get_state_shape(self, batch_size: int) -> Tuple[int, int, int]:
        """
        :param batch_size: current batch size
        :return: dimensions of each output state (assuming all of them have the same shape)
        """
        return (1, batch_size, self.model_size)

    @staticmethod
    @pt.jit.script_if_tracing
    def _training_cell_state_transform(
        previous_cell_state: pt.Tensor, 
        weighted_inputs: pt.Tensor, 
        forget_rates: pt.Tensor
    ) -> Tuple[pt.Tensor, pt.Tensor]:
        """Update SSRU cell at training time"""
        steps: int = weighted_inputs.size(0)
        cell_state: pt.Tensor = previous_cell_state.squeeze(0)
        states: List[pt.Tensor] = []
        for t in range(steps):
            cell_state = forget_rates[t, :, :] * cell_state + weighted_inputs[t, :, :]
            states.append(cell_state)
        states_tensor: pt.Tensor = pt.stack(states, dim=0)
        return (states_tensor, cell_state.unsqueeze(0))

    @staticmethod
    def _inference_cell_state_transform(
        previous_cell_state: pt.Tensor, 
        weighted_inputs: pt.Tensor, 
        forget_rates: pt.Tensor
    ) -> Tuple[pt.Tensor, pt.Tensor]:
        """Update SSRU cell at inference time"""
        new_step_state: pt.Tensor = forget_rates * previous_cell_state + weighted_inputs
        return (new_step_state, new_step_state)

    def forward(
        self, 
        inputs: pt.Tensor, 
        previous_states: Union[List[pt.Tensor], Tuple[pt.Tensor, ...]], 
        **args: Any
    ) -> Tuple[pt.Tensor, pt.Tensor]:
        """
        :param inputs: input data. Shape: (max_length, batch, input_depth).
        :param previous_states: previous cell states. Shape: (max_length, batch, input_depth)
        :return: cell output and new cell states.  Both with shape (max_length, batch, input_depth).
        """
        forget_rates: pt.Tensor = self.forget_gate_act(self.forget_gate(inputs))
        weighted_inputs: pt.Tensor = (1 - forget_rates) * self.linear(inputs)
        cell_state, last_step_state = self.cell_state_transform(previous_states, weighted_inputs, forget_rates)
        cell_state = self.relu(cell_state)
        if self.clamp_to_dtype:
            cell_state = clamp_to_dtype_min_max(cell_state)
        return (cell_state, last_step_state)

def clamp_to_dtype_min_max(data: pt.Tensor) -> pt.Tensor:
    """
    Clamp a tensor's values to the min and max for its dtype. This effectively
    pushes overflowed (infinite) values back into the finite range.

    See: https://discuss.huggingface.co/t/t5-fp16-issue-is-fixed/3139
    """
    return pt.clamp(data, min=pt.finfo(data.dtype).min, max=pt.finfo(data.dtype).max)
