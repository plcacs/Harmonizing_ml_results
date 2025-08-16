from dataclasses import dataclass
from typing import Optional, Tuple
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
    def __init__(self, config: TransformerConfig, inference_only: bool = False, dtype: Optional[pt.dtype] = None, clamp_to_dtype: bool = False):
        ...

    def forward(self, data: pt.Tensor, att_mask: Optional[pt.Tensor] = None) -> pt.Tensor:
        ...

class TransformerDecoderBlock(pt.nn.Module):
    def __init__(self, config: TransformerConfig, inference_only: bool, dtype: Optional[pt.dtype] = None, clamp_to_dtype: bool = False):
        ...

    def set_inference_only(self, inference_only: bool):
        ...

    @property
    def num_state_tensors(self) -> int:
        ...

    @property
    def needs_mask(self) -> bool:
        ...

    def get_states_shape(self, batch_size: int) -> Tuple[int, ...]:
        ...

    def forward(self, target: pt.Tensor, target_mask: pt.Tensor, source: pt.Tensor, source_mask: pt.Tensor, autoregr_states: Tuple[pt.Tensor, ...], enc_att_kv: Optional[pt.Tensor] = None) -> Tuple[pt.Tensor, Tuple[pt.Tensor, ...]]:
        ...

class TransformerProcessBlock(pt.nn.Module):
    def __init__(self, sequence: str, dropout: float, num_hidden: int = 0, dtype: Optional[pt.dtype] = None, clamp_to_dtype: bool = False):
        ...

    def forward(self, data: pt.Tensor, prev: Optional[pt.Tensor] = None) -> pt.Tensor:
        ...

class TransformerFeedForward(pt.nn.Module):
    def __init__(self, num_hidden: int, num_model: int, act_type: str, dropout: float, use_glu: bool = False, inference_only: bool = False, dtype: Optional[pt.dtype] = None, clamp_to_dtype: bool = False):
        ...

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        ...

class AutoRegressiveMask(pt.nn.Module):
    def forward(self, x: pt.Tensor) -> pt.Tensor:
        ...
