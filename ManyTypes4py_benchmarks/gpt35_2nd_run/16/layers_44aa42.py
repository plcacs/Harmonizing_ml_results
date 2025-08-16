import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import torch as pt
import torch.nn.functional as F
from sockeye import constants as C
from . import config
logger: logging.Logger = logging.getLogger(__name__)

def get_activation(act_type: str) -> pt.nn.Module:
    if act_type == C.SWISH1:
        return pt.nn.SiLU()
    if act_type == C.GELU:
        return pt.nn.GELU()
    return pt.nn.ReLU()

class LHUC(pt.nn.Module):
    def __init__(self, num_hidden: int, dtype: Optional[pt.dtype] = None):
        super().__init__()
        self.weight = pt.nn.Parameter(pt.empty(num_hidden, dtype=dtype))

    def forward(self, data: pt.Tensor) -> pt.Tensor:
        weight = 2 * pt.sigmoid(self.weight)
        return weight * data

class OutputLayer(pt.nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, weight: Optional[pt.nn.Parameter] = None, dtype: Optional[pt.dtype] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.in_features = hidden_size
        self.out_features = vocab_size
        if weight is None:
            self.weight = pt.nn.Parameter(pt.empty(vocab_size, hidden_size, dtype=dtype))
        else:
            self.weight = weight
        self.bias = pt.nn.Parameter(pt.empty(vocab_size, dtype=dtype))
        self.previous_slice_ids = pt.empty(0)
        self.reduced_weight = pt.empty(0)
        self.reduced_bias = pt.empty(0)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={} dtype={}'.format(self.in_features, self.out_features, self.bias is not None, self.weight.dtype)

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
    def __init__(self, keys_index: Any, vals: List[int], vocab_size: int, k: int = 64, temperature: int = 10, state_store: Optional[Any] = None):
        super().__init__()
        self.keys_index = keys_index
        self.vals = vals
        self.vocab_size = vocab_size
        self.k = k
        self.temperature = temperature
        self.state_store = state_store

    def forward(self, data: pt.Tensor) -> pt.Tensor:
        distances, indices = self.keys_index.search(data.cpu().numpy().astype(np.float32), self.k)
        y = self.vals[(indices + 1) % len(self.vals)]
        y[y == C.BOS_ID] = C.EOS_ID
        if self.state_store is not None:
            raw_keys = pt.from_numpy(self.state_store[indices]).to(device=data.device)
            distances = pt.norm(data.unsqueeze(1) - raw_keys, p=2, dim=-1)
        else:
            distances = np.sqrt(distances)
            distances = pt.from_numpy(distances).to(device=data.device)
        y = pt.from_numpy(y).to(device=data.device).long()
        probs = pt.exp(-distances / self.temperature)
        full_probs = pt.zeros((data.shape[0], self.vocab_size), device=data.device)
        full_probs.scatter_add_(src=probs, index=y.squeeze(2), dim=-1)
        z = pt.sum(full_probs, dim=-1).unsqueeze(-1)
        z[z < C.KNN_EPSILON] = C.KNN_EPSILON
        full_probs.div_(z)
        return full_probs

@dataclass
class LengthRatioConfig(config.Config):
    pass

class LengthRatio(pt.nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dtype: Optional[pt.dtype] = None):
        utils.check_condition(num_layers >= 1, "LengthRatio's num_layers has to be >=1.")
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        modules = []
        for _ in range(num_layers - 1):
            modules.append(pt.nn.Linear(in_features=hidden_size, out_features=hidden_size, dtype=dtype))
            modules.append(pt.nn.Tanh())
        modules.append(pt.nn.Linear(in_features=hidden_size, out_features=1, dtype=dtype))
        modules.append(pt.nn.Softplus())
        self.layers = pt.nn.Sequential(*modules)

    def forward(self, source_encoded: pt.Tensor, source_encoded_length: pt.Tensor) -> pt.Tensor:
        mask = pt.arange(source_encoded.size()[1], device=source_encoded_length.device)[None, :, None] >= source_encoded_length[:, None, None]
        source_masked = source_encoded.masked_fill(mask, 0.0)
        data = source_masked.sum(dim=1, keepdim=False) / source_encoded_length.unsqueeze(1)
        data = self.layers(data).squeeze(1)
        return data

@pt.jit.script
def interleaved_matmul_encdec_qk(q: pt.Tensor, kv: pt.Tensor, heads: int) -> pt.Tensor:
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
    def __init__(self, dropout: float = 0.0, heads: int = 1):
        super().__init__()
        self.dropout = pt.nn.Dropout(p=dropout)
        self.heads = heads

    def forward(self, queries: pt.Tensor, key_values: pt.Tensor, mask: Optional[pt.Tensor] = None) -> pt.Tensor:
        logits = interleaved_matmul_encdec_qk(queries, key_values, heads=self.heads)
        if mask is not None:
            logits = logits.masked_fill(mask, -C.LARGE_VALUES[logits.dtype])
        probs = F.softmax(logits, dim=-1)
        probs = self.dropout(probs) if self.dropout is not None else probs
        return interleaved_matmul_encdec_valatt(key_values, probs, heads=self.heads)

def prepare_source_length_mask(lengths: pt.Tensor, heads: int, max_length: int, expand: bool = True, mask_prepended_tokens: bool = False) -> pt.Tensor:
    mask = ~(pt.arange(max_length, device=lengths.device).unsqueeze(0) < lengths[:, :1])
    if mask_prepended_tokens:
        prepended_token_mask = pt.arange(max_length, device=lengths.device).unsqueeze(0) < lengths[:, 1:2]
        mask |= prepended_token_mask
    if expand:
        mask = mask.unsqueeze(1).expand(-1, heads, -1).reshape((-1, max_length)).unsqueeze(1)
    return mask

class MultiHeadAttentionBase(pt.nn.Module):
    def __init__(self, depth_att: int = 512, heads: int = 8, depth_out: int = 512, dropout: float = 0.0, dtype: Optional[pt.dtype] = None, clamp_to_dtype: bool = False):
        super().__init__()
        utils.check_condition(depth_att % heads == 0, 'Number of heads (%d) must divide attention depth (%d)' % (heads, depth_att))
        self.depth = depth_att
        self.heads = heads
        self.depth_out = depth_out
        self.depth_per_head = self.depth // self.heads
        self.clamp_to_dtype = clamp_to_dtype
        self.dot_att = DotAttentionCell(dropout=dropout, heads=heads)
        self.ff_out = pt.nn.Linear(in_features=depth_att, out_features=depth_out, bias=False, dtype=dtype)

    def _attend(self, queries: pt.Tensor, key_values: pt.Tensor, mask: Optional[pt.Tensor] = None) -> pt.Tensor:
        contexts = self.dot_att(queries=queries, key_values=key_values, mask=mask)
        contexts = self.ff_out(contexts)
        if self.clamp_to_dtype:
            contexts = clamp_to_dtype_min_max(contexts)
        return contexts

class AutoregressiveLayer(pt.nn.Module):
    @property
    @abstractmethod
    def num_state_tensors(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def needs_mask(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_state_shape(self, batch_size: int) -> Tuple[int, int, int]:
        raise NotImplementedError

    @abstractmethod
    def set_inference_only(self, inference_only: bool) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs: pt.Tensor, previous_states, *args) -> Tuple[pt.Tensor, Any]:
        raise NotImplementedError

class MultiHeadSelfAttention(MultiHeadAttentionBase, AutoregressiveLayer):
    def __init__(self, depth_att: int = 512, heads: int = 8, depth_out: int = 512, dropout: float = 0.0, dtype: Optional[pt.dtype] = None, clamp_to_dtype: bool = False):
        super().__init__(depth_att, heads, depth_out, dropout, dtype, clamp_to_dtype)
        self.depth_att = depth_att
        self.ff_in = pt.nn.Linear(in_features=depth_att, out_features=depth_att * 3, bias=False, dtype=dtype)
        self._drop_p = dropout
        self.kv_interleaved = False

    def set_inference_only(self, inference_only: bool) -> None:
        raise NotImplementedError

    def separate_kv(self) -> None:
        assert self.kv_interleaved
        with pt.no_grad():
            kv = self.ff_in.weight.data[self.depth:, :]
            k, v = kv.view(self.heads, 2 * self.depth_per_head, self.depth).split(self.depth_per_head, dim=1)
            k = k.reshape(self.depth, self.depth)
            v = v.reshape(self.depth, self.depth)
        self.ff_in.weight.data[self.depth:, :] = pt.cat((k, v), dim=0)
        self.kv_interleaved = False

    def interleave_kv(self) -> None:
        assert not self.kv_interleaved
        with pt.no_grad():
            _, k, v = self.ff_in.weight.data.split(self.depth, dim=0)
            k = k.reshape(self.heads, -1, self.depth)
            v = v.reshape(self.heads, -1, self.depth)
        self.ff_in.weight.data[self.depth:, :] = pt.cat((k, v), dim=1).reshape(self.depth * 2, self.depth)
        self.kv_interleaved = True

    def train(self, mode: bool = True) -> None:
        if mode and self.kv_interleaved:
            self.separate_kv()
        elif not mode and (not self.kv_interleaved):
            self.interleave_kv()
        return super().train(mode)

    @property
    def num_state_tensors(self) -> int:
        return 1

    @property
    def needs_mask(self) -> bool:
        return True

    def get_state_shape(self, batch_size: int) -> Tuple[int, int, int]:
        return (0, batch_size, self.depth_out * 2)

    def forward(self, inputs: pt.Tensor, previous_states: Optional[pt.Tensor] = None, mask: Optional[pt.Tensor] = None, **args) -> pt.Tensor:
        if self.training:
            assert not self.kv_interleaved
            contexts, _ = F.multi_head_attention_forward(query=inputs, key=inputs, value=inputs, embed_dim_to_check=self.depth, num_heads=self.heads, in_proj_weight=self.ff_in.weight, in_proj_bias=None, bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=self._drop_p, out_proj_weight=self.ff_out.weight, out_proj_bias=self.ff_out.bias, training=self.training, key_padding_mask=None, need_weights=False, attn_mask=mask, use_separate_proj_weight=False, q_proj_weight=None, k_proj_weight=None, v_proj_weight=None)
            return contexts
        else:
            proj = self.ff_in(inputs)
            queries, states = proj.split((self.depth_att, 2 * self.depth_att), dim=2)
            if previous_states is not None:
                states = pt.cat((previous_states, states), dim=0)
            return (self._attend(queries=queries, key_values=states, mask=mask), states)

class MultiHeadAttention(MultiHeadAttentionBase):
    def __init__(self, depth_att: int = 512, heads: int = 8, depth_out: int = 512, dropout: float = 0.0, depth_key_value: int = 512, dtype: Optional[pt.dtype] = None, clamp_to_dtype: bool = False):
        super().__init__(depth_att, heads, depth_out, dropout, dtype, clamp_to_dtype)
        self.ff_q = pt.nn.Linear(in_features=depth_out, out_features=depth_att, bias=False, dtype=dtype)
        self.ff_kv = pt.nn.Linear(in_features=depth_key_value, out_features=depth_att * 2, bias=False, dtype=dtype)
        self._drop_p = dropout
        self._depth_key_value = depth_key_value
        self.kv_interleaved = False

    def separate_kv(self) -> None:
        if not self.kv_interleaved:
            module.interleave_kv()

    def interleave_kv(self) -> None:
        if self.kv_interleaved:
            module.separate_kv()

    def train(self, mode: bool = True) -> None:
        if mode and self.kv_interleaved:
            self.separate_kv()
        elif not mode and (not self.kv_interleaved):
            self.interleave_kv()
        return super().train(mode)

    def forward(self, queries: pt.Tensor, key_values: pt.Tensor, mask: Optional[pt.Tensor] = None, projected_memory_kv: Optional[pt.Tensor] = None) -> pt.Tensor:
        if self.training:
            assert not self.kv_interleaved
            assert projected_memory_kv is None, 'caching not supported in training'
            contexts, _ = F.multi_head_attention_forward(query=queries, key=key_values, value=key_values, embed_dim_to_check=self.depth, num_heads=self.heads, in_proj_weight=None, in_proj_bias=None, bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=self._drop_p, out_proj_weight=self.ff_out.weight, out_proj_bias=self.ff_out.bias, training=self.training, key_padding_mask=None, need_weights=False, attn_mask=mask, use_separate_proj_weight=True, q_proj_weight=self.ff_q.weight, k_proj_weight=self.ff_kv.weight[:self.depth, :], v_proj_weight=self.ff_kv.weight[self.depth:, :])
            return contexts
        else:
            queries = self.ff_q(queries)
            key_values = projected_memory_kv if projected_memory_kv is not None else self.ff_kv(key_values)
            return self._attend(queries=queries, key_values=key_values, mask=mask)

def interleave_kv(module: pt.nn.Module) -> None:
    if isinstance(module, MultiHeadAttention) or isinstance(module, MultiHeadSelfAttention):
        if not module.kv_interleaved:
            module.interleave_kv()

def separate_kv(module: pt.nn.Module) -> None:
    if isinstance(module, MultiHeadAttention) or isinstance(module, MultiHeadSelfAttention):
        if module.kv_interleaved:
            module.separate_kv()

@pt.jit.script
def get_positional_embeddings(length: int, depth: int) -> pt.Tensor:
    utils.check_condition(depth % 2 == 0, 'Positional embeddings require an even embedding size it is however %d.' % depth)
    channels = pt.arange(depth // 2).unsqueeze(0)
    positions = pt.arange(0, length).unsqueeze(1)
    scaled_positions = positions / pt.pow(10000, 2 * channels / depth)
    sin = pt.sin(scaled_positions)
    cos = pt.cos(scaled_positions)
    encodings = pt.hstack([sin, cos])
    return encodings

class PositionalEmbeddings(pt.nn.Module):
    def __init__(self, weight_type: str, num_embed: int, max_seq_len: int, scale_up_input: bool, scale_down_positions: bool, dtype: Optional[pt.dtype] = None):
        utils.check_condition(num_embed % 2 == 0, 'Positional embeddings require an even embedding size it is however %d.' % num_embed)
        super().__init__()
        self.weight_type = weight_type
        self.num_embed = num_embed
        self.max_seq_len = max_seq_len
        self.scale_up_input = scale_up_input
        self.scale_down_positions = scale_down_positions
        if self.weight_type == C.FIXED_POSITIONAL_EMBEDDING:
            weight = get_positional_embeddings(length=self.max_seq_len, depth=self.num_embed)
            if self.scale_down_positions:
                weight *= self.num_embed ** (-