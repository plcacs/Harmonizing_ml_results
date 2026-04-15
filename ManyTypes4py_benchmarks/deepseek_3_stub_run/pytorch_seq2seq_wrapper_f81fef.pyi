import torch
from typing import Optional, Tuple, Union, List
from torch.nn.utils.rnn import PackedSequence
from allennlp.common.checks import ConfigurationError
from allennlp.modules.augmented_lstm import AugmentedLstm
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.stacked_alternating_lstm import StackedAlternatingLstm
from allennlp.modules.stacked_bidirectional_lstm import StackedBidirectionalLstm

class PytorchSeq2SeqWrapper(Seq2SeqEncoder):
    _module: torch.nn.Module
    _is_bidirectional: bool
    _num_directions: int
    
    def __init__(self, module: torch.nn.Module, stateful: bool = False) -> None: ...
    
    def get_input_dim(self) -> int: ...
    
    def get_output_dim(self) -> int: ...
    
    def is_bidirectional(self) -> bool: ...
    
    def forward(
        self, 
        inputs: Union[PackedSequence, torch.Tensor], 
        mask: Optional[torch.Tensor], 
        hidden_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor: ...

@Seq2SeqEncoder.register('gru')
class GruSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int = 1, 
        bias: bool = True, 
        dropout: float = 0.0, 
        bidirectional: bool = False, 
        stateful: bool = False
    ) -> None: ...

@Seq2SeqEncoder.register('lstm')
class LstmSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int = 1, 
        bias: bool = True, 
        dropout: float = 0.0, 
        bidirectional: bool = False, 
        stateful: bool = False
    ) -> None: ...

@Seq2SeqEncoder.register('rnn')
class RnnSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int = 1, 
        nonlinearity: str = 'tanh', 
        bias: bool = True, 
        dropout: float = 0.0, 
        bidirectional: bool = False, 
        stateful: bool = False
    ) -> None: ...

@Seq2SeqEncoder.register('augmented_lstm')
class AugmentedLstmSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        go_forward: bool = True, 
        recurrent_dropout_probability: float = 0.0, 
        use_highway: bool = True, 
        use_input_projection_bias: bool = True, 
        stateful: bool = False
    ) -> None: ...

@Seq2SeqEncoder.register('alternating_lstm')
class StackedAlternatingLstmSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int, 
        recurrent_dropout_probability: float = 0.0, 
        use_highway: bool = True, 
        use_input_projection_bias: bool = True, 
        stateful: bool = False
    ) -> None: ...

@Seq2SeqEncoder.register('stacked_bidirectional_lstm')
class StackedBidirectionalLstmSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int, 
        recurrent_dropout_probability: float = 0.0, 
        layer_dropout_probability: float = 0.0, 
        use_highway: bool = True, 
        stateful: bool = False
    ) -> None: ...