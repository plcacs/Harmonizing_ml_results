from typing import Optional, Tuple
import torch
from torch.nn.utils.rnn import PackedSequence

class AugmentedLSTMCell(torch.nn.Module):
    embed_dim: int
    lstm_dim: int
    use_highway: bool
    use_bias: bool
    input_linearity: torch.nn.Module
    state_linearity: torch.nn.Module
    def __init__(self, embed_dim: int, lstm_dim: int, use_highway: bool = ..., use_bias: bool = ...) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward(self, x: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] = ..., variational_dropout_mask: Optional[torch.Tensor] = ...) -> Tuple[torch.Tensor, torch.Tensor]: ...

class AugmentedLstm(torch.nn.Module):
    embed_dim: int
    lstm_dim: int
    go_forward: bool
    use_highway: bool
    recurrent_dropout_probability: float
    cell: AugmentedLSTMCell
    def __init__(self, input_size: int, hidden_size: int, go_forward: bool = ..., recurrent_dropout_probability: float = ..., use_highway: bool = ..., use_input_projection_bias: bool = ...) -> None: ...
    def forward(self, inputs: PackedSequence, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = ...) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]: ...

class BiAugmentedLstm(torch.nn.Module):
    input_size: int
    padding_value: float
    hidden_size: int
    num_layers: int
    bidirectional: bool
    recurrent_dropout_probability: float
    use_highway: bool
    use_bias: bool
    representation_dim: int
    forward_layers: torch.nn.ModuleList
    backward_layers: torch.nn.ModuleList
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True, recurrent_dropout_probability: float = 0.0, bidirectional: bool = False, padding_value: float = 0.0, use_highway: bool = True) -> None: ...
    def forward(self, inputs: PackedSequence, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = ...) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]: ...
    def _forward_bidirectional(self, inputs: PackedSequence, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = ...) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]: ...
    def _forward_unidirectional(self, inputs: PackedSequence, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = ...) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]: ...