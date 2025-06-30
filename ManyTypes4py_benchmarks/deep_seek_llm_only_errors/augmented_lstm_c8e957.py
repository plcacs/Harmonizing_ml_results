from typing import Optional, Tuple, List, Union
import torch
from allennlp.common.checks import ConfigurationError
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from allennlp.nn.initializers import block_orthogonal
from allennlp.nn.util import get_dropout_mask

class AugmentedLSTMCell(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        lstm_dim: int,
        use_highway: bool = True,
        use_bias: bool = True
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.lstm_dim = lstm_dim
        self.use_highway = use_highway
        self.use_bias = use_bias
        if use_highway:
            self._highway_inp_proj_start = 5 * self.lstm_dim
            self._highway_inp_proj_end = 6 * self.lstm_dim
            self.input_linearity = torch.nn.Linear(self.embed_dim, self._highway_inp_proj_end, bias=self.use_bias)
            self.state_linearity = torch.nn.Linear(self.lstm_dim, self._highway_inp_proj_start, bias=True)
        else:
            self.input_linearity = torch.nn.Linear(self.embed_dim, 4 * self.lstm_dim, bias=self.use_bias)
            self.state_linearity = torch.nn.Linear(self.lstm_dim, 4 * self.lstm_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        block_orthogonal(self.input_linearity.weight.data, [self.lstm_dim, self.embed_dim])
        block_orthogonal(self.state_linearity.weight.data, [self.lstm_dim, self.lstm_dim])
        self.state_linearity.bias.data.fill_(0.0)
        self.state_linearity.bias.data[self.lstm_dim:2 * self.lstm_dim].fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor],
        variational_dropout_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_state, memory_state = states
        if variational_dropout_mask is not None and self.training:
            hidden_state = hidden_state * variational_dropout_mask
        projected_input = self.input_linearity(x)
        projected_state = self.state_linearity(hidden_state)
        input_gate = forget_gate = memory_init = output_gate = highway_gate = None
        if self.use_highway:
            fused_op = projected_input[:, :5 * self.lstm_dim] + projected_state
            fused_chunked = torch.chunk(fused_op, 5, 1)
            input_gate, forget_gate, memory_init, output_gate, highway_gate = fused_chunked
            highway_gate = torch.sigmoid(highway_gate)
        else:
            fused_op = projected_input + projected_state
            input_gate, forget_gate, memory_init, output_gate = torch.chunk(fused_op, 4, 1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        memory_init = torch.tanh(memory_init)
        output_gate = torch.sigmoid(output_gate)
        memory = input_gate * memory_init + forget_gate * memory_state
        timestep_output = output_gate * torch.tanh(memory)
        if self.use_highway:
            highway_input_projection = projected_input[:, self._highway_inp_proj_start:self._highway_inp_proj_end]
            timestep_output = highway_gate * timestep_output + (1 - highway_gate) * highway_input_projection
        return (timestep_output, memory)

class AugmentedLstm(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        go_forward: bool = True,
        recurrent_dropout_probability: float = 0.0,
        use_highway: bool = True,
        use_input_projection_bias: bool = True
    ) -> None:
        super().__init__()
        self.embed_dim = input_size
        self.lstm_dim = hidden_size
        self.go_forward = go_forward
        self.use_highway = use_highway
        self.recurrent_dropout_probability = recurrent_dropout_probability
        self.cell = AugmentedLSTMCell(self.embed_dim, self.lstm_dim, self.use_highway, use_input_projection_bias)

    def forward(
        self,
        inputs: PackedSequence,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]:
        if not isinstance(inputs, PackedSequence):
            raise ConfigurationError('inputs must be PackedSequence but got %s' % type(inputs))
        sequence_tensor, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
        batch_size = sequence_tensor.size()[0]
        total_timesteps = sequence_tensor.size()[1]
        output_accumulator = sequence_tensor.new_zeros(batch_size, total_timesteps, self.lstm_dim)
        if states is None:
            full_batch_previous_memory = sequence_tensor.new_zeros(batch_size, self.lstm_dim)
            full_batch_previous_state = sequence_tensor.data.new_zeros(batch_size, self.lstm_dim)
        else:
            full_batch_previous_state = states[0].squeeze(0)
            full_batch_previous_memory = states[1].squeeze(0)
        current_length_index = batch_size - 1 if self.go_forward else 0
        if self.recurrent_dropout_probability > 0.0:
            dropout_mask = get_dropout_mask(self.recurrent_dropout_probability, full_batch_previous_memory)
        else:
            dropout_mask = None
        for timestep in range(total_timesteps):
            index = timestep if self.go_forward else total_timesteps - timestep - 1
            if self.go_forward:
                while batch_lengths[current_length_index] <= index:
                    current_length_index -= 1
            else:
                while current_length_index < len(batch_lengths) - 1 and batch_lengths[current_length_index + 1] > index:
                    current_length_index += 1
            previous_memory = full_batch_previous_memory[0:current_length_index + 1].clone()
            previous_state = full_batch_previous_state[0:current_length_index + 1].clone()
            timestep_input = sequence_tensor[0:current_length_index + 1, index]
            timestep_output, memory = self.cell(timestep_input, (previous_state, previous_memory), dropout_mask[0:current_length_index + 1] if dropout_mask is not None else None)
            full_batch_previous_memory = full_batch_previous_memory.data.clone()
            full_batch_previous_state = full_batch_previous_state.data.clone()
            full_batch_previous_memory[0:current_length_index + 1] = memory
            full_batch_previous_state[0:current_length_index + 1] = timestep_output
            output_accumulator[0:current_length_index + 1, index, :] = timestep_output
        output_accumulator = pack_padded_sequence(output_accumulator, batch_lengths, batch_first=True)
        final_state = (full_batch_previous_state.unsqueeze(0), full_batch_previous_memory.unsqueeze(0))
        return (output_accumulator, final_state)

class BiAugmentedLstm(torch.nn.Module):
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
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.padding_value = padding_value
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.recurrent_dropout_probability = recurrent_dropout_probability
        self.use_highway = use_highway
        self.use_bias = bias
        num_directions = int(self.bidirectional) + 1
        self.forward_layers = torch.nn.ModuleList()
        if self.bidirectional:
            self.backward_layers = torch.nn.ModuleList()
        lstm_embed_dim = self.input_size
        for _ in range(self.num_layers):
            self.forward_layers.append(AugmentedLstm(lstm_embed_dim, self.hidden_size, go_forward=True, recurrent_dropout_probability=self.recurrent_dropout_probability, use_highway=self.use_highway, use_input_projection_bias=self.use_bias))
            if self.bidirectional:
                self.backward_layers.append(AugmentedLstm(lstm_embed_dim, self.hidden_size, go_forward=False, recurrent_dropout_probability=self.recurrent_dropout_probability, use_highway=self.use_highway, use_input_projection_bias=self.use_bias))
            lstm_embed_dim = self.hidden_size * num_directions
        self.representation_dim = lstm_embed_dim

    def forward(
        self,
        inputs: PackedSequence,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]:
        if not isinstance(inputs, PackedSequence):
            raise ConfigurationError('inputs must be PackedSequence but got %s' % type(inputs))
        if self.bidirectional:
            return self._forward_bidirectional(inputs, states)
        return self._forward_unidirectional(inputs, states)

    def _forward_bidirectional(
        self,
        inputs: PackedSequence,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]:
        output_sequence = inputs
        final_h: List[torch.Tensor] = []
        final_c: List[torch.Tensor] = []
        if not states:
            hidden_states: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * self.num_layers
        elif states[0].size()[0] != self.num_layers:
            raise RuntimeError('Initial states were passed to forward() but the number of initial states does not match the number of layers.')
        else:
            hidden_states = list(zip(states[0].chunk(self.num_layers, 0), states[1].chunk(self.num_layers, 0)))
        for i, state in enumerate(hidden_states):
            if state:
                forward_state = state[0].chunk(2, -1)
                backward_state = state[1].chunk(2, -1)
            else:
                forward_state = backward_state = None
            forward_layer = self.forward_layers[i]
            backward_layer = self.backward_layers[i]
            forward_output, final_forward_state = forward_layer(output_sequence, forward_state)
            backward_output, final_backward_state = backward_layer(output_sequence, backward_state)
            forward_output, lengths = pad_packed_sequence(forward_output, batch_first=True)
            backward_output, _ = pad_packed_sequence(backward_output, batch_first=True)
            output_sequence = torch.cat([forward_output, backward_output], -1)
            output_sequence = pack_padded_sequence(output_sequence, lengths, batch_first=True)
            final_h.extend([final_forward_state[0], final_backward_state[0]])
            final_c.extend([final_forward_state[1], final_backward_state[1]])
        final_h_tensor = torch.cat(final_h, dim=0)
        final_c_tensor = torch.cat(final_c, dim=0)
        final_state_tuple = (final_h_tensor, final_c_tensor)
        output_sequence, batch_lengths = pad_packed_sequence(output_sequence, padding_value=self.padding_value, batch_first=True)
        output_sequence = pack_padded_sequence(output_sequence, batch_lengths, batch_first=True)
        return (output_sequence, final_state_tuple)

    def _forward_unidirectional(
        self,
        inputs: PackedSequence,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]:
        output_sequence = inputs
        final_h: List[torch.Tensor] = []
        final_c: List[torch.Tensor] = []
        if not states:
            hidden_states: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * self.num_layers
        elif states[0].size()[0] != self.num_layers:
            raise RuntimeError('Initial states were passed to forward() but the number of initial states does not match the number of layers.')
        else:
            hidden_states = list(zip(states[0].chunk(self.num_layers, 0), states[1].chunk(self.num_layers, 0)))
        for i, state in enumerate(hidden_states):
            forward_layer = self.forward_layers[i]
            forward_output, final_forward_state = forward_layer(output_sequence, state)
            output_sequence = forward_output
            final_h.append(final_forward_state[0])
            final_c.append(final_forward_state[1])
        final_h_tensor = torch.cat(final_h, dim=0)
        final_c_tensor = torch.cat(final_c, dim=0)
        final_state_tuple = (final_h_tensor, final_c_tensor)
        output_sequence, batch_lengths = pad_packed_sequence(output_sequence, padding_value=self.padding_value, batch_first=True)
        output_sequence = pack_padded_sequence(output_sequence, batch_lengths, batch_first=True)
        return (output_sequence, final_state_tuple)
