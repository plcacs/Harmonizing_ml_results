```python
from typing import Optional, Tuple, List
import torch
from torch.nn.utils.rnn import PackedSequence

class AugmentedLSTMCell(torch.nn.Module):
    embed_dim: int
    lstm_dim: int
    use_highway: bool
    use_bias: bool
    input_linearity: torch.nn.Linear
    state_linearity: torch.nn.Linear
    _highway_inp_proj_start: int
    _highway_inp_proj_end: int
    
    def __init__(
        self,
        embed_dim: int,
        lstm_dim: int,
        use_highway: bool = True,
        use_bias: bool = True
    ) -> None: ...
    
    def reset_parameters(self) -> None: ...
    
    def forward(
        self,
        x: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor],
        variational_dropout_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

class AugmentedLstm(torch.nn.Module):
    embed_dim: int
    lstm_dim: int
    go_forward: bool
    use_highway: bool
    recurrent_dropout_probability: float
    cell: AugmentedLSTMCell
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        go_forward: bool = True,
        recurrent_dropout_probability: float = 0.0,
        use_highway: bool = True,
        use_input_projection_bias: bool = True
    ) -> None: ...
    
    def forward(
        self,
        inputs: PackedSequence,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]: ...

class BiAugmentedLstm(torch.nn.Module):
    input_size: int
    padding_value: float
    hidden_size: int
    num_layers: int
    bidirectional: bool
    recurrent_dropout_probability: float
    use_highway: bool
    use_bias: bool
    forward_layers: torch.nn.ModuleList
    backward_layers: torch.nn.ModuleList
    representation_dim: int
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        recurrent_dropout_probability: float = 0.0,
        bidirectional: bool = False,
        padding_value: float = 0.0,
        use_highway: bool = True
    ) -> None: ...
    
    def forward(
        self,
        inputs: PackedSequence,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]: ...
    
    def _forward_bidirectional(
        self,
        inputs: PackedSequence,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]: ...
    
    def _forward_unidirectional(
        self,
        inputs: PackedSequence,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]: ...
```