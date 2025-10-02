import torch
from allennlp.common.checks import ConfigurationError
from allennlp.modules.augmented_lstm import AugmentedLstm
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.stacked_alternating_lstm import StackedAlternatingLstm
from allennlp.modules.stacked_bidirectional_lstm import StackedBidirectionalLstm
from torch.nn.modules.rnn import RNNBase
from torch.nn.modules.rnn import LSTM
from torch.nn.modules.rnn import GRU
from torch.nn.modules.rnn import RNN

class PytorchSeq2VecWrapper(Seq2VecEncoder):
    def __init__(self, module: RNNBase):
        super().__init__(stateful=False)
        self._module = module

    def get_input_dim(self) -> int:
        return self._module.input_size

    def get_output_dim(self) -> int:
        is_bidirectional = getattr(self._module, 'bidirectional', False)
        return self._module.hidden_size * (2 if is_bidirectional else 1)

    def forward(self, inputs, mask, hidden_state=None):
        if mask is None:
            return self._module(inputs, hidden_state)[0][:, -1, :]
        batch_size = mask.size(0)
        _, state, restoration_indices = self.sort_and_run_forward(self._module, inputs, mask, hidden_state)
        if isinstance(state, tuple):
            state = state[0]
        num_layers_times_directions, num_valid, encoding_dim = state.size()
        if num_valid < batch_size:
            zeros = state.new_zeros(num_layers_times_directions, batch_size - num_valid, encoding_dim)
            state = torch.cat([state, zeros], 1)
        unsorted_state = state.transpose(0, 1).index_select(0, restoration_indices)
        last_state_index = 2 if getattr(self._module, 'bidirectional', False) else 1
        last_layer_state = unsorted_state[:, -last_state_index:, :]
        return last_layer_state.contiguous().view([-1, self.get_output_dim()])

@Seq2VecEncoder.register('gru')
class GruSeq2VecEncoder(PytorchSeq2VecWrapper):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True, dropout: float = 0.0, bidirectional: bool = False):
        module = GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        super().__init__(module=module)

@Seq2VecEncoder.register('lstm')
class LstmSeq2VecEncoder(PytorchSeq2VecWrapper):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True, dropout: float = 0.0, bidirectional: bool = False):
        module = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        super().__init__(module=module)

@Seq2VecEncoder.register('rnn')
class RnnSeq2VecEncoder(PytorchSeq2VecWrapper):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, nonlinearity: str = 'tanh', bias: bool = True, dropout: float = 0.0, bidirectional: bool = False):
        module = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, nonlinearity=nonlinearity, bias=bias, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        super().__init__(module=module)

@Seq2VecEncoder.register('augmented_lstm')
class AugmentedLstmSeq2VecEncoder(PytorchSeq2VecWrapper):
    def __init__(self, input_size: int, hidden_size: int, go_forward: bool = True, recurrent_dropout_probability: float = 0.0, use_highway: bool = True, use_input_projection_bias: bool = True):
        module = AugmentedLstm(input_size=input_size, hidden_size=hidden_size, go_forward=go_forward, recurrent_dropout_probability=recurrent_dropout_probability, use_highway=use_highway, use_input_projection_bias=use_input_projection_bias)
        super().__init__(module=module)

@Seq2VecEncoder.register('alternating_lstm')
class StackedAlternatingLstmSeq2VecEncoder(PytorchSeq2VecWrapper):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, recurrent_dropout_probability: float = 0.0, use_highway: bool = True, use_input_projection_bias: bool = True):
        module = StackedAlternatingLstm(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, recurrent_dropout_probability=recurrent_dropout_probability, use_highway=use_highway, use_input_projection_bias=use_input_projection_bias)
        super().__init__(module=module)

@Seq2VecEncoder.register('stacked_bidirectional_lstm')
class StackedBidirectionalLstmSeq2VecEncoder(PytorchSeq2VecWrapper):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, recurrent_dropout_probability: float = 0.0, layer_dropout_probability: float = 0.0, use_highway: bool = True):
        module = StackedBidirectionalLstm(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, recurrent_dropout_probability=recurrent_dropout_probability, layer_dropout_probability=layer_dropout_probability, use_highway=use_highway)
        super().__init__(module=module)
