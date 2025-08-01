import torch
from typing import Any, Optional, Tuple
from allennlp.common.checks import ConfigurationError
from allennlp.modules.augmented_lstm import AugmentedLstm
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.stacked_alternating_lstm import StackedAlternatingLstm
from allennlp.modules.stacked_bidirectional_lstm import StackedBidirectionalLstm

class PytorchSeq2VecWrapper(Seq2VecEncoder):
    """
    Pytorch's RNNs have two outputs: the final hidden state for every time step,
    and the hidden state at the last time step for every layer.
    We just want the final hidden state of the last time step.
    This wrapper pulls out that output, and adds a `get_output_dim` method, which is useful if you
    want to, e.g., define a linear + softmax layer on top of this to get some distribution over a
    set of labels.  The linear layer needs to know its input dimension before it is called, and you
    can get that from `get_output_dim`.

    Also, there are lots of ways you could imagine going from an RNN hidden state at every
    timestep to a single vector - you could take the last vector at all layers in the stack, do
    some kind of pooling, take the last vector of the top layer in a stack, or many other  options.
    We just take the final hidden state vector, or in the case of a bidirectional RNN cell, we
    concatenate the forward and backward final states together. TODO(mattg): allow for other ways
    of wrapping RNNs.

    In order to be wrapped with this wrapper, a class must have the following members:

        - `self.input_size: int`
        - `self.hidden_size: int`
        - `def forward(inputs: PackedSequence, hidden_state: torch.tensor) ->
          Tuple[PackedSequence, torch.Tensor]`.
        - `self.bidirectional: bool` (optional)

    This is what pytorch's RNN's look like - just make sure your class looks like those, and it
    should work.

    Note that we *require* you to pass a binary `mask` of shape
    (batch_size, sequence_length) when you call this module, to avoid subtle
    bugs around masking. If you already have a `PackedSequence` you can pass
    `None` as the second parameter.
    """
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__(stateful=False)
        self._module = module
        try:
            if not self._module.batch_first:
                raise ConfigurationError('Our encoder semantics assumes batch is always first!')
        except AttributeError:
            pass

    def get_input_dim(self) -> int:
        return self._module.input_size

    def get_output_dim(self) -> int:
        try:
            is_bidirectional: bool = self._module.bidirectional
        except AttributeError:
            is_bidirectional = False
        return self._module.hidden_size * (2 if is_bidirectional else 1)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor], hidden_state: Optional[Any] = None) -> torch.Tensor:
        if mask is None:
            return self._module(inputs, hidden_state)[0][:, -1, :]
        batch_size: int = mask.size(0)
        # Assuming sort_and_run_forward returns a tuple of (output, state, restoration_indices)
        _, state, restoration_indices = self.sort_and_run_forward(self._module, inputs, mask, hidden_state)
        if isinstance(state, tuple):
            state = state[0]
        num_layers_times_directions, num_valid, encoding_dim = state.size()
        if num_valid < batch_size:
            zeros = state.new_zeros(num_layers_times_directions, batch_size - num_valid, encoding_dim)
            state = torch.cat([state, zeros], 1)
        unsorted_state = state.transpose(0, 1).index_select(0, restoration_indices)
        try:
            last_state_index: int = 2 if self._module.bidirectional else 1
        except AttributeError:
            last_state_index = 1
        last_layer_state = unsorted_state[:, -last_state_index:, :]
        return last_layer_state.contiguous().view(-1, self.get_output_dim())

@Seq2VecEncoder.register('gru')
class GruSeq2VecEncoder(PytorchSeq2VecWrapper):
    """
    Registered as a `Seq2VecEncoder` with name "gru".
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True,
                 dropout: float = 0.0, bidirectional: bool = False) -> None:
        module: torch.nn.GRU = torch.nn.GRU(input_size=input_size,
                                             hidden_size=hidden_size,
                                             num_layers=num_layers,
                                             bias=bias,
                                             batch_first=True,
                                             dropout=dropout,
                                             bidirectional=bidirectional)
        super().__init__(module=module)

@Seq2VecEncoder.register('lstm')
class LstmSeq2VecEncoder(PytorchSeq2VecWrapper):
    """
    Registered as a `Seq2VecEncoder` with name "lstm".
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True,
                 dropout: float = 0.0, bidirectional: bool = False) -> None:
        module: torch.nn.LSTM = torch.nn.LSTM(input_size=input_size,
                                               hidden_size=hidden_size,
                                               num_layers=num_layers,
                                               bias=bias,
                                               batch_first=True,
                                               dropout=dropout,
                                               bidirectional=bidirectional)
        super().__init__(module=module)

@Seq2VecEncoder.register('rnn')
class RnnSeq2VecEncoder(PytorchSeq2VecWrapper):
    """
    Registered as a `Seq2VecEncoder` with name "rnn".
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, nonlinearity: str = 'tanh',
                 bias: bool = True, dropout: float = 0.0, bidirectional: bool = False) -> None:
        module: torch.nn.RNN = torch.nn.RNN(input_size=input_size,
                                             hidden_size=hidden_size,
                                             num_layers=num_layers,
                                             nonlinearity=nonlinearity,
                                             bias=bias,
                                             batch_first=True,
                                             dropout=dropout,
                                             bidirectional=bidirectional)
        super().__init__(module=module)

@Seq2VecEncoder.register('augmented_lstm')
class AugmentedLstmSeq2VecEncoder(PytorchSeq2VecWrapper):
    """
    Registered as a `Seq2VecEncoder` with name "augmented_lstm".
    """
    def __init__(self, input_size: int, hidden_size: int, go_forward: bool = True,
                 recurrent_dropout_probability: float = 0.0, use_highway: bool = True,
                 use_input_projection_bias: bool = True) -> None:
        module: AugmentedLstm = AugmentedLstm(input_size=input_size,
                                               hidden_size=hidden_size,
                                               go_forward=go_forward,
                                               recurrent_dropout_probability=recurrent_dropout_probability,
                                               use_highway=use_highway,
                                               use_input_projection_bias=use_input_projection_bias)
        super().__init__(module=module)

@Seq2VecEncoder.register('alternating_lstm')
class StackedAlternatingLstmSeq2VecEncoder(PytorchSeq2VecWrapper):
    """
    Registered as a `Seq2VecEncoder` with name "alternating_lstm".
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 recurrent_dropout_probability: float = 0.0, use_highway: bool = True,
                 use_input_projection_bias: bool = True) -> None:
        module: StackedAlternatingLstm = StackedAlternatingLstm(input_size=input_size,
                                                                hidden_size=hidden_size,
                                                                num_layers=num_layers,
                                                                recurrent_dropout_probability=recurrent_dropout_probability,
                                                                use_highway=use_highway,
                                                                use_input_projection_bias=use_input_projection_bias)
        super().__init__(module=module)

@Seq2VecEncoder.register('stacked_bidirectional_lstm')
class StackedBidirectionalLstmSeq2VecEncoder(PytorchSeq2VecWrapper):
    """
    Registered as a `Seq2VecEncoder` with name "stacked_bidirectional_lstm".
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 recurrent_dropout_probability: float = 0.0, layer_dropout_probability: float = 0.0,
                 use_highway: bool = True) -> None:
        module: StackedBidirectionalLstm = StackedBidirectionalLstm(input_size=input_size,
                                                                    hidden_size=hidden_size,
                                                                    num_layers=num_layers,
                                                                    recurrent_dropout_probability=recurrent_dropout_probability,
                                                                    layer_dropout_probability=layer_dropout_probability,
                                                                    use_highway=use_highway)
        super().__init__(module=module)