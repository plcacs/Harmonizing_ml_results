"""
Stub file for 'augmented_lstm_c8e957' module
"""

from typing import Optional, Tuple, List
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

class AugmentedLSTMCell:
    input_linearity: torch.nn.Linear
    state_linearity: torch.nn.Linear
    
    def __init__(self, embed_dim: int, lstm_dim: int, use_highway: bool = True, use_bias: bool = True) -> None:
        ...
    
    def reset_parameters(self) -> None:
        ...
    
    def forward(self, x: Tensor, states: Tuple[Tensor, Tensor], variational_dropout_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        ...

class AugmentedLstm:
    cell: AugmentedLSTMCell
    
    def __init__(self, input_size: int, hidden_size: int, go_forward: bool = True, recurrent_dropout_probability: float = 0.0, use_highway: bool = True, use_input_projection_bias: bool = True) -> None:
        ...
    
    def forward(self, inputs: PackedSequence, states: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[PackedSequence, Tuple[Tensor, Tensor]]:
        ...

class BiAugmentedLstm:
    forward_layers: torch.nn.ModuleList
    backward_layers: Optional[torch.nn.ModuleList]
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True, recurrent_dropout_probability: float = 0.0, bidirectional: bool = False, padding_value: float = 0.0, use_highway: bool = True) -> None:
        ...
    
    def forward(self, inputs: PackedSequence, states: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[PackedSequence, Tuple[Tensor, Tensor]]:
        ...
    
    def _forward_bidirectional(self, inputs: PackedSequence, states: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[PackedSequence, Tuple[Tensor, Tensor]]:
        ...
    
    def _forward_unidirectional(self, inputs: PackedSequence, states: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[PackedSequence, Tuple[Tensor, Tensor]]:
        ...