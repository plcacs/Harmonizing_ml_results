import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from typing import Optional, Tuple, Union, List
from allennlp.common.checks import ConfigurationError
from allennlp.modules.augmented_lstm import AugmentedLstm
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.stacked_alternating_lstm import StackedAlternatingLstm
from allennlp.modules.stacked_bidirectional_lstm import StackedBidirectionalLstm

class PytorchSeq2SeqWrapper(Seq2SeqEncoder):
    """
    Pytorch's RNNs have two outputs: the hidden state for every time step, and the hidden state at
    the last time step for every layer.  We just want the first one as a single output.  This
    wrapper pulls out that output, and adds a `get_output_dim` method, which is useful if you
    want to, e.g., define a linear + softmax layer on top of this to get some distribution over a
    set of labels.  The linear layer needs to know its input dimension before it is called, and you
    can get that from `get_output_dim`.

    In order to be wrapped with this wrapper, a class must have the following members:

        - `self.input_size: int`
        - `self.hidden_size: int`
        - `def forward(inputs: PackedSequence, hidden_state: torch.Tensor) ->
          Tuple[PackedSequence, torch.Tensor]`.
        - `self.bidirectional: bool` (optional)

    This is what pytorch's RNN's look like - just make sure your class looks like those, and it
    should work.

    Note that we *require* you to pass a binary mask of shape (batch_size, sequence_length)
    when you call this module, to avoid subtle bugs around masking.  If you already have a
    `PackedSequence` you can pass `None` as the second parameter.

    We support stateful RNNs where the final state from each batch is used as the initial
    state for the subsequent batch by passing `stateful=True` to the constructor.
    """
    def __init__(self, module: Module, stateful: bool = False) -> None:
        super().__init__(stateful)
        self._module: Module = module
        try:
            if not self._module.batch_first:
                raise ConfigurationError('Our encoder semantics assumes batch is always first!')
        except AttributeError:
            pass
        try:
            self._is_bidirectional: bool = self._module.bidirectional  # type: ignore
        except AttributeError:
            self._is_bidirectional = False
        if self._is_bidirectional:
            self._num_directions: int = 2
        else:
            self._num_directions = 1

    def get_input_dim(self) -> int:
        return self._module.input_size  # type: ignore

    def get_output_dim(self) -> int:
        return self._module.hidden_size * self._num_directions  # type: ignore

    def is_bidirectional(self) -> bool:
        return self._is_bidirectional

    def forward(
        self, 
        inputs: Union[Tensor, PackedSequence], 
        mask: Optional[Tensor], 
        hidden_state: Optional[Tensor] = None
    ) -> Tensor:
        if self.stateful and mask is None:
            raise ValueError('Always pass a mask with stateful RNNs.')
        if self.stateful and hidden_state is not None:
            raise ValueError('Stateful RNNs provide their own initial hidden_state.')
        if mask is None:
            output, _ = self._module(inputs, hidden_state)  # type: ignore
            return output
        batch_size, total_sequence_length = mask.size()
        # The following function is assumed to sort and run the forward pass.
        # It returns a tuple of (PackedSequence, final_states, restoration_indices).
        packed_sequence_output, final_states, restoration_indices = self.sort_and_run_forward(
            self._module, inputs, mask, hidden_state
        )  # type: Tuple[PackedSequence, Union[Tensor, List[Tensor]], Tensor]
        unpacked_sequence_tensor, _ = pad_packed_sequence(packed_sequence_output, batch_first=True)
        num_valid: int = unpacked_sequence_tensor.size(0)
        if not isinstance(final_states, (list, tuple)) and self.stateful:
            final_states = [final_states]
        if num_valid < batch_size:
            _, length, output_dim = unpacked_sequence_tensor.size()
            zeros: Tensor = unpacked_sequence_tensor.new_zeros(batch_size - num_valid, length, output_dim)
            unpacked_sequence_tensor = torch.cat([unpacked_sequence_tensor, zeros], 0)
            if self.stateful:
                new_states: List[Tensor] = []
                for state in final_states:  # type: Tensor
                    num_layers, _, state_dim = state.size()
                    zeros_state: Tensor = state.new_zeros(num_layers, batch_size - num_valid, state_dim)
                    new_states.append(torch.cat([state, zeros_state], 1))
                final_states = new_states
        sequence_length_difference: int = total_sequence_length - unpacked_sequence_tensor.size(1)
        if sequence_length_difference > 0:
            zeros_seq: Tensor = unpacked_sequence_tensor.new_zeros(
                batch_size, sequence_length_difference, unpacked_sequence_tensor.size(-1)
            )
            unpacked_sequence_tensor = torch.cat([unpacked_sequence_tensor, zeros_seq], 1)
        if self.stateful:
            self._update_states(final_states, restoration_indices)
        return unpacked_sequence_tensor.index_select(0, restoration_indices)

@Seq2SeqEncoder.register("gru")
class GruSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    """
    Registered as a `Seq2SeqEncoder` with name "gru".
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        stateful: bool = False
    ) -> None:
        module = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        super().__init__(module=module, stateful=stateful)

@Seq2SeqEncoder.register("lstm")
class LstmSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    """
    Registered as a `Seq2SeqEncoder` with name "lstm".
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        stateful: bool = False
    ) -> None:
        module = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        super().__init__(module=module, stateful=stateful)

@Seq2SeqEncoder.register("rnn")
class RnnSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    """
    Registered as a `Seq2SeqEncoder` with name "rnn".
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        stateful: bool = False
    ) -> None:
        module = torch.nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        super().__init__(module=module, stateful=stateful)

@Seq2SeqEncoder.register("augmented_lstm")
class AugmentedLstmSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    """
    Registered as a `Seq2SeqEncoder` with name "augmented_lstm".
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        go_forward: bool = True,
        recurrent_dropout_probability: float = 0.0,
        use_highway: bool = True,
        use_input_projection_bias: bool = True,
        stateful: bool = False
    ) -> None:
        module = AugmentedLstm(
            input_size=input_size,
            hidden_size=hidden_size,
            go_forward=go_forward,
            recurrent_dropout_probability=recurrent_dropout_probability,
            use_highway=use_highway,
            use_input_projection_bias=use_input_projection_bias
        )
        super().__init__(module=module, stateful=stateful)

@Seq2SeqEncoder.register("alternating_lstm")
class StackedAlternatingLstmSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    """
    Registered as a `Seq2SeqEncoder` with name "alternating_lstm".
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        recurrent_dropout_probability: float = 0.0,
        use_highway: bool = True,
        use_input_projection_bias: bool = True,
        stateful: bool = False
    ) -> None:
        module = StackedAlternatingLstm(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            recurrent_dropout_probability=recurrent_dropout_probability,
            use_highway=use_highway,
            use_input_projection_bias=use_input_projection_bias
        )
        super().__init__(module=module, stateful=stateful)

@Seq2SeqEncoder.register("stacked_bidirectional_lstm")
class StackedBidirectionalLstmSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    """
    Registered as a `Seq2SeqEncoder` with name "stacked_bidirectional_lstm".
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        recurrent_dropout_probability: float = 0.0,
        layer_dropout_probability: float = 0.0,
        use_highway: bool = True,
        stateful: bool = False
    ) -> None:
        module = StackedBidirectionalLstm(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            recurrent_dropout_probability=recurrent_dropout_probability,
            layer_dropout_probability=layer_dropout_probability,
            use_highway=use_highway
        )
        super().__init__(module=module, stateful=stateful)