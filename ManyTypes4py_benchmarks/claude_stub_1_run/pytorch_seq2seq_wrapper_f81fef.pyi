```pyi
from typing import Any, Optional, Tuple, Union, List
import torch
from torch.nn.utils.rnn import PackedSequence
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

class PytorchSeq2SeqWrapper(Seq2SeqEncoder):
    _module: Any
    _is_bidirectional: bool
    _num_directions: int
    
    def __init__(self, module: Any, stateful: bool = ...) -> None: ...
    def get_input_dim(self) -> int: ...
    def get_output_dim(self) -> int: ...
    def is_bidirectional(self) -> bool: ...
    def forward(
        self,
        inputs: Union[torch.Tensor, PackedSequence],
        mask: Optional[torch.Tensor],
        hidden_state: Optional[torch.Tensor] = ...
    ) -> torch.Tensor: ...

class GruSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = ...,
        bias: bool = ...,
        dropout: float = ...,
        bidirectional: bool = ...,
        stateful: bool = ...
    ) -> None: ...

class LstmSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = ...,
        bias: bool = ...,
        dropout: float = ...,
        bidirectional: bool = ...,
        stateful: bool = ...
    ) -> None: ...

class RnnSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = ...,
        nonlinearity: str = ...,
        bias: bool = ...,
        dropout: float = ...,
        bidirectional: bool = ...,
        stateful: bool = ...
    ) -> None: ...

class AugmentedLstmSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        go_forward: bool = ...,
        recurrent_dropout_probability: float = ...,
        use_highway: bool = ...,
        use_input_projection_bias: bool = ...,
        stateful: bool = ...
    ) -> None: ...

class StackedAlternatingLstmSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        recurrent_dropout_probability: float = ...,
        use_highway: bool = ...,
        use_input_projection_bias: bool = ...,
        stateful: bool = ...
    ) -> None: ...

class StackedBidirectionalLstmSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        recurrent_dropout_probability: float = ...,
        layer_dropout_probability: float = ...,
        use_highway: bool = ...,
        stateful: bool = ...
    ) -> None: ...
```