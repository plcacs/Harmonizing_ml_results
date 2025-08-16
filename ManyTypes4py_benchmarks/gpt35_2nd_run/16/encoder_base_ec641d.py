from typing import Tuple, Union, Optional, Callable, Any
import torch
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, sort_batch_by_length

RnnState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
RnnStateStorage = Tuple[torch.Tensor, ...]

class _EncoderBase(torch.nn.Module):
    def __init__(self, stateful: bool = False) -> None:
        super().__init__()
        self.stateful: bool = stateful
        self._states: Optional[RnnStateStorage] = None

    def sort_and_run_forward(self, module: Callable[[PackedSequence, Optional[RnnState]], Tuple[Union[PackedSequence, torch.Tensor], RnnState]], 
                             inputs: torch.Tensor, mask: torch.BoolTensor, hidden_state: Optional[RnnState] = None) -> Tuple[Union[torch.Tensor, PackedSequence], Optional[RnnState], torch.LongTensor]:
        ...

    def _get_initial_states(self, batch_size: int, num_valid: int, sorting_indices: torch.LongTensor) -> Optional[RnnStateStorage]:
        ...

    def _update_states(self, final_states: RnnStateStorage, restoration_indices: torch.LongTensor) -> None:
        ...

    def reset_states(self, mask: Optional[torch.BoolTensor] = None) -> None:
        ...
