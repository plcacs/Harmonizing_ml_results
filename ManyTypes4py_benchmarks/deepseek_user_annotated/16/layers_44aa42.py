# Copyright 2017--2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, Dict, Union, cast

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

    def __init__(self, num_hidden: int, dtype: Optional[pt.dtype] = None) -> None:
        super().__init__()
        self.weight = pt.nn.Parameter(pt.empty(num_hidden, dtype=dtype))

    def forward(self, data: pt.Tensor) -> pt.Tensor:
        # We use a sigmoid with amplitude 2 for weighting the hidden units. The
        # activation is dampened when the value of the sigmoid is close to 0, and
        # strengthened when it's close to 2 (see also original paper)
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

    def __init__(self,
                 hidden_size: int,
                 vocab_size: int,
                 weight: Optional[pt.nn.Parameter] = None,
                 dtype: Optional[pt.dtype] = None) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.in_features = hidden_size
        self.out_features = vocab_size

        if weight is None:
            self.weight = pt.nn.Parameter(pt.empty(vocab_size, hidden_size, dtype=dtype))
        else:
            self.weight = weight
        self.bias = pt.nn.Parameter(pt.empty(vocab_size, dtype=dtype))

        self.previous_slice_ids: pt.Tensor = pt.empty(0)
        self.reduced_weight: pt.Tensor = pt.empty(0)
        self.reduced_bias: pt.Tensor = pt.empty(0)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={} dtype={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.weight.dtype)

    def _is_new_slice(self, x: pt.Tensor) -> bool:
        if x.size() != self.previous_slice_ids.size() or pt.any(x != self.previous_slice_ids):
            return True
        return False

    def _take_slice(self, vocab_slice_ids: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        weight = self.weight[vocab_slice_ids]  # Shape: (len(vocab_slice_ids), hidden)
        bias = self.bias[vocab_slice_ids]
        return weight, bias

    def forward(self, data: pt.Tensor, vocab_slice_ids: Optional[pt.Tensor] = None) -> pt.Tensor:
        if vocab_slice_ids is not None:
            # Imperative, reduced matrix multiplication for vocabulary selection.
            # vocab_slice_ids is constant across decoder step calls, so we cache the result of _take_slice
            # across decoder steps. If a new vocab_slice_ids tensor is observed, we re-run _take_slice.
            # This significantly reduces latency for CPU decoding.
            if self._is_new_slice(vocab_slice_ids):
                self.previous_slice_ids = vocab_slice_ids
                weight, bias = self.reduced_weight, self.reduced_bias = self._take_slice(vocab_slice_ids)
            else:
                weight, bias = self.reduced_weight, self.reduced_bias
        else:
            weight, bias = self.weight, self.bias

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

    def __init__(self,
                 keys_index: Any,  # faiss.Index type
                 vals: np.memmap,
                 vocab_size: int,
                 k: int = 64,
                 temperature: int = 10,
                 state_store: Optional[np.memmap] = None) -> None:
        super().__init__()
        self.keys_index = keys_index
        self.vals = vals
        self.vocab_size = vocab_size
        self.k = k
        self.temperature = temperature
        self.state_store = state_store

    def forward(self, data: pt.Tensor) -> pt.Tensor:
        # faiss only supports float32
        distances, indices = self.keys_index.search(data.cpu().numpy().astype(np.float32), self.k)
        # Map indices to tokens
        y = self.vals[(indices + 1) % len(self.vals)]
        # no EOS is inserted in generated data store, so we need to use the BOS of the next sentence as EOS
        y[y == C.BOS_ID] = C.EOS_ID

        # use exact distance when state_store is available
        if self.state_store is not None:
            raw_keys = pt.from_numpy(self.state_store[indices]).to(device=data.device)  # (data.shape[0], k, dim)
            distances = pt.norm(data.unsqueeze(1) - raw_keys, p=2, dim=-1)  # data lacks k axis, so need to create one
        else:
            distances = np.sqrt(distances)  # unlike pytorch, faiss doesn't do sqrt for us
            distances = pt.from_numpy(distances).to(device=data.device)

        # pytorch expects long for indexes
        y = pt.from_numpy(y).to(device=data.device).long()

        probs = pt.exp(-distances / self.temperature)
        full_probs = pt.zeros((data.shape[0], self.vocab_size), device=data.device)
        full_probs.scatter_add_(src=probs, index=y.squeeze(2), dim=-1)
        z = pt.sum(full_probs, dim=-1).unsqueeze(-1)
        z[z < C.KNN_EPSILON] = C.KNN_EPSILON  # avoid div by 0 (which may happen when distances of all items are large)
        full_probs.div_(z)
        return full_probs


@dataclass
class LengthRatioConfig(config.Config):
    num_layers: int  # Number of layers
    weight: float  # Weight of this loss


class LengthRatio(pt.nn.Module):
    """
    Defines the length-ratio prediction layer of Sockeye.

    :param hidden_size: Encoder hidden size.
    :param num_layers: Number of layers.
    :param dtype: Torch data type for parameters.
    """

    def __init__(self,
                 hidden_size: int,
                 num_layers: int,
                 dtype: Optional[pt.dtype] = None) -> None:
        utils.check_condition(num_layers >= 1, "LengthRatio's num_layers has to be >=1.")
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        modules: List[pt.nn.Module] = []
        for _ in range(num_layers - 1):
            modules.append(pt.nn.Linear(in_features=hidden_size, out_features=hidden_size, dtype=dtype))
            modules.append(pt.nn.Tanh())
        modules.append(pt.nn.Linear(in_features=hidden_size, out_features=1, dtype=dtype))
        modules.append(pt.nn.Softplus())  # SoftReLU activation to ensure positiveness of the predicted length ratio
        self.layers = pt.nn.Sequential(*modules)

    def forward(self, source_encoded: pt.Tensor, source_encoded_length: pt.Tensor) -> pt.Tensor:
        """
        Transformation to the length ratio. Returns a vector.

        :param source_encoded: Encoder representation for n elements. Shape: (n, source_encoded_length, hidden_size).
        :param source_encoded_length: A vector of encoded sequence lengths. Shape: (n,).
        :return: Predictions of the ratio length(hypothesis)/length(reference). Shape(n, 1).
        """
        # True when outside length. Shape: (n, source_encoded_length, 1)
        mask = pt.arange(source_encoded.size()[1], device=source_encoded_length.device)[None, :, None] >= source_encoded_length[:, None, None]
        source_masked = source_encoded.masked_fill(mask, 0.)

        # data: (n, hidden_size)
        data = source_masked.sum(dim=1, keepdim=False) / source_encoded_length.unsqueeze(1)
        data = self.layers(data).squeeze(1)  # (n, 1)
        return data


@pt.jit.script
def interleaved_matmul_encdec_qk(q: pt.Tensor,
                                 kv: pt.Tensor,
                                 heads: int) -> pt.Tensor:
    """
    Simple port of npx.interleaved_matmul_encdec_qk with PyTorch.

    :param q: (qlen, batch, hidden)
    :param kv: (kvlen, batch, hidden * 2) -- interleaved
    :param heads: number of attention heads
    :return: (batch * heads, qlen, klen)
    """
    qlen, batch, hidden = q.size()
    head_dim = hidden // heads

    # batch * heads, qlen, head_dim)
    q = q.contiguous().view(qlen, batch * heads, head_dim).transpose(0, 1)
    q = q * head_dim ** -0.5

    tmp = kv.reshape(-1, batch, heads, 2, head_dim)
    k = tmp[:, :, :, 0, :]  # pick keys
    k = k.permute(1, 2, 3, 0)  # (batch, heads, head_dim, kvlen)
    k = k.reshape(batch * heads, head_dim, -1)  # (batch * heads, head_dim, kvlen)

    return pt.bmm(q, k)  # (batch * heads, qlen, klen)


@pt.jit.script
def interleaved_matmul_encdec_valatt(kv: pt.Tensor,
                                     att: pt.Tensor,
                                     heads: int) -> pt.Tensor:
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
    v = tmp[:, :, :, 1, :]  # pick values
    v = v.permute(1, 2, 0, 3)  # bsz, heads, kvlen, head_dim
    v = v.reshape(-1, kvlen, head_dim)  # bsz * heads, kvlen, head_dim

    output = pt.bmm(att, v)  # bsz * heads, qlen, head_dim
    output = output.transpose(0, 1).contiguous().view(-1, batch, hidden)
    return output


class DotAttentionCell(pt.nn.Module):

    def __init__(self, dropout: float = 0.0, heads: int = 1) -> None:
        super().__init__()
        self.dropout = pt.nn.Dropout(p=dropout)
        self.heads = heads

    def forward(self,
                queries: pt.Tensor,
                key_values: pt.Tensor,
                mask: Optional[pt.Tensor] = None) -> pt.Tensor:
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
        # (batch * heads, qlen, klen)
        logits = interleaved_matmul_encdec_qk(queries, key_values, heads=self.heads)

        if mask is not None:
            logits = logits.masked_fill(mask, -C.LARGE_VALUES[logits.dtype])

        probs = F.softmax(logits, dim=-1)

        probs = self.dropout(probs) if self.dropout is not None else probs

        # key_values: (lk, n, dv * 2)
        # probs: (n*h, lq, lk)
        # result: (n, lq, dv)
        return interleaved_matmul_encdec_valatt(key_values, probs, heads=self.heads)


def prepare_source_length_mask(lengths: pt.Tensor, heads: int, max_length: int, expand: bool = True,
                               mask_prepended_tokens: bool = False) -> pt.Tensor:
    """
    Prepare source length masks where positions of invalid tokens are marked as True.

    :param lengths: Total source length and prepended source length. Shape: (batch_size, 2)
    :param heads: Number of attention heads.
    :param max_length: Maximum sequence length.
    :param expand: Expand to the heads.
    :param mask_prepended_tokens: Mask prepended tokens.
    :return: Source length mask.
    """
    # (batch_size, max_len)
    mask = ~(pt.arange(max_length, device=lengths.device).unsqueeze(0) < lengths[:, :1])
    if mask_prepended_tokens:
        prepended_token_mask = pt.arange(max_length, device=lengths.device).unsqueeze(0) < lengths[:, 1:2]
        mask |= prepended_token_mask
    if expand:
        # (batch_size * heads, 1, max_len)
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
    def __init__(self,
                 depth_att: int = 512,
                 heads: int = 8,
                 depth_out: int = 512,
                 dropout: float = 0.0,
                 dtype: Optional[pt.dtype] = None,
                 clamp_to_dtype: bool = False) -> None:
        super().__init__()
        utils.check_condition(depth_att % heads == 0,
                              "Number of heads (%d) must divide attention depth (%d)" % (heads, depth_att))
        self.depth = depth_att
        self.heads = heads
        self.depth_out = depth_out
        self.depth_per_head = self.depth // self.heads
        self.clamp_to_dtype = clamp_to_dtype

        self.dot_att = DotAttentionCell(dropout=dropout, heads=heads)
        self.ff_out = pt.nn.Linear(in_features=depth_att, out_features=depth_out,