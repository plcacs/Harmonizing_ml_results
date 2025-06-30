from typing import Tuple, Union, Optional, Callable, Any, List, cast
import torch
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, sort_batch_by_length
RnnState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
RnnStateStorage = Tuple[torch.Tensor, ...]
RnnInputs = Tuple[PackedSequence, Optional[RnnState]]
RnnOutputs = Tuple[Union[PackedSequence, torch.Tensor], RnnState]

class _EncoderBase(torch.nn.Module):
    def __init__(self, stateful: bool = False) -> None:
        super().__init__()
        self.stateful: bool = stateful
        self._states: Optional[RnnStateStorage] = None

    def sort_and_run_forward(
        self,
        module: Callable[..., RnnOutputs],
        inputs: torch.Tensor,
        mask: torch.BoolTensor,
        hidden_state: Optional[RnnState] = None
    ) -> Tuple[Union[torch.Tensor, PackedSequence], Optional[RnnState], torch.LongTensor]:
        batch_size: int = mask.size(0)
        num_valid: int = torch.sum(mask[:, 0]).int().item()
        sequence_lengths: torch.Tensor = get_lengths_from_binary_sequence_mask(mask)
        sorted_inputs: torch.Tensor
        sorted_sequence_lengths: torch.Tensor
        restoration_indices: torch.LongTensor
        sorting_indices: torch.LongTensor
        sorted_inputs, sorted_sequence_lengths, restoration_indices, sorting_indices = sort_batch_by_length(inputs, sequence_lengths)
        packed_sequence_input: PackedSequence = pack_padded_sequence(
            sorted_inputs[:num_valid, :, :],
            sorted_sequence_lengths[:num_valid].data.tolist(),
            batch_first=True
        )
        initial_states: Optional[RnnState]
        if not self.stateful:
            if hidden_state is None:
                initial_states = hidden_state
            elif isinstance(hidden_state, tuple):
                initial_states = [state.index_select(1, sorting_indices)[:, :num_valid, :].contiguous() for state in hidden_state]
                initial_states = cast(RnnState, tuple(initial_states))
            else:
                initial_states = hidden_state.index_select(1, sorting_indices)[:, :num_valid, :].contiguous()
        else:
            initial_states = self._get_initial_states(batch_size, num_valid, sorting_indices)
        module_output: Union[torch.Tensor, PackedSequence]
        final_states: RnnState
        module_output, final_states = module(packed_sequence_input, initial_states)
        return (module_output, final_states, restoration_indices)

    def _get_initial_states(
        self,
        batch_size: int,
        num_valid: int,
        sorting_indices: torch.LongTensor
    ) -> Optional[RnnState]:
        if self._states is None:
            return None
        if batch_size > self._states[0].size(1):
            num_states_to_concat: int = batch_size - self._states[0].size(1)
            resized_states: List[torch.Tensor] = []
            for state in self._states:
                zeros: torch.Tensor = state.new_zeros(state.size(0), num_states_to_concat, state.size(2))
                resized_states.append(torch.cat([state, zeros], 1))
            self._states = tuple(resized_states)
            correctly_shaped_states: RnnStateStorage = self._states
        elif batch_size < self._states[0].size(1):
            correctly_shaped_states = tuple((state[:, :batch_size, :] for state in self._states))
        else:
            correctly_shaped_states = self._states
        if len(self._states) == 1:
            correctly_shaped_state: torch.Tensor = correctly_shaped_states[0]
            sorted_state: torch.Tensor = correctly_shaped_state.index_select(1, sorting_indices)
            return sorted_state[:, :num_valid, :].contiguous()
        else:
            sorted_states: List[torch.Tensor] = [state.index_select(1, sorting_indices) for state in correctly_shaped_states]
            return tuple((state[:, :num_valid, :].contiguous() for state in sorted_states))

    def _update_states(
        self,
        final_states: RnnStateStorage,
        restoration_indices: torch.LongTensor
    ) -> None:
        new_unsorted_states: List[torch.Tensor] = [state.index_select(1, restoration_indices) for state in final_states]
        if self._states is None:
            self._states = tuple((state.data for state in new_unsorted_states))
        else:
            current_state_batch_size: int = self._states[0].size(1)
            new_state_batch_size: int = final_states[0].size(1)
            used_new_rows_mask: List[torch.Tensor] = [
                (state[0, :, :].sum(-1) != 0.0).float().view(1, new_state_batch_size, 1)
                for state in new_unsorted_states
            ]
            new_states: List[torch.Tensor] = []
            if current_state_batch_size > new_state_batch_size:
                for old_state, new_state, used_mask in zip(self._states, new_unsorted_states, used_new_rows_mask):
                    masked_old_state: torch.Tensor = old_state[:, :new_state_batch_size, :] * (1 - used_mask)
                    old_state[:, :new_state_batch_size, :] = new_state + masked_old_state
                    new_states.append(old_state.detach())
            else:
                new_states = []
                for old_state, new_state, used_mask in zip(self._states, new_unsorted_states, used_new_rows_mask):
                    masked_old_state = old_state * (1 - used_mask)
                    new_state += masked_old_state
                    new_states.append(new_state.detach())
            self._states = tuple(new_states)

    def reset_states(self, mask: Optional[torch.BoolTensor] = None) -> None:
        if mask is None:
            self._states = None
        else:
            mask_batch_size: int = mask.size(0)
            mask = mask.view(1, mask_batch_size, 1)
            new_states: List[torch.Tensor] = []
            assert self._states is not None
            for old_state in self._states:
                old_state_batch_size: int = old_state.size(1)
                if old_state_batch_size != mask_batch_size:
                    raise ValueError(f'Trying to reset states using mask with incorrect batch size. Expected batch size: {old_state_batch_size}. Provided batch size: {mask_batch_size}.')
                new_state: torch.Tensor = ~mask * old_state
                new_states.append(new_state.detach())
            self._states = tuple(new_states)
