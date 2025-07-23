"""
Assorted utilities for working with neural networks in AllenNLP.
"""
import copy
from collections import defaultdict, OrderedDict
from itertools import chain
import json
import logging
from os import PathLike
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union, NamedTuple, cast, overload
import math
import numpy
import torch
import torch.distributed as dist
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import int_to_device, is_distributed, is_global_primary

logger = logging.getLogger(__name__)
T = TypeVar('T')
V = TypeVar('V', int, float, torch.Tensor)
StateDictType = Union[Dict[str, torch.Tensor], 'OrderedDict[str, torch.Tensor]']

def move_to_device(obj: Any, device: Union[int, torch.device]) -> Any:
    """
    Given a structure (possibly) containing Tensors,
    move all the Tensors to the specified device (or do nothing, if they are already on
    the target device).
    """
    device = int_to_device(device)
    if isinstance(obj, torch.Tensor):
        return obj if obj.device == device else obj.to(device=device)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = move_to_device(value, device)
        return obj
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            obj[i] = move_to_device(item, device)
        return obj
    elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
        return obj.__class__(*(move_to_device(item, device) for item in obj))
    elif isinstance(obj, tuple):
        return tuple((move_to_device(item, device) for item in obj))
    else:
        return obj

def clamp_tensor(tensor: torch.Tensor, minimum: float, maximum: float) -> torch.Tensor:
    """
    Supports sparse and dense tensors.
    Returns a tensor with values clamped between the provided minimum and maximum,
    without modifying the original tensor.
    """
    if tensor.is_sparse:
        coalesced_tensor = tensor.coalesce()
        coalesced_tensor._values().clamp_(minimum, maximum)
        return coalesced_tensor
    else:
        return tensor.clamp(minimum, maximum)

def batch_tensor_dicts(tensor_dicts: List[Dict[str, torch.Tensor]], remove_trailing_dimension: bool = False) -> Dict[str, torch.Tensor]:
    """
    Takes a list of tensor dictionaries, where each dictionary is assumed to have matching keys,
    and returns a single dictionary with all tensors with the same key batched together.
    """
    key_to_tensors = defaultdict(list)
    for tensor_dict in tensor_dicts:
        for key, tensor in tensor_dict.items():
            key_to_tensors[key].append(tensor)
    batched_tensors = {}
    for key, tensor_list in key_to_tensors.items():
        batched_tensor = torch.stack(tensor_list)
        if remove_trailing_dimension and all((tensor.size(-1) == 1 for tensor in tensor_list)):
            batched_tensor = batched_tensor.squeeze(-1)
        batched_tensors[key] = batched_tensor
    return batched_tensors

def get_lengths_from_binary_sequence_mask(mask: torch.BoolTensor) -> torch.LongTensor:
    """
    Compute sequence lengths for each batch element in a tensor using a
    binary mask.
    """
    return mask.sum(-1)

def get_mask_from_sequence_lengths(sequence_lengths: torch.LongTensor, max_length: int) -> torch.BoolTensor:
    """
    Given a variable of shape `(batch_size,)` that represents the sequence lengths of each batch
    element, this function returns a `(batch_size, max_length)` mask variable.
    """
    ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
    range_tensor = ones.cumsum(dim=1)
    return sequence_lengths.unsqueeze(1) >= range_tensor

def sort_batch_by_length(tensor: torch.FloatTensor, sequence_lengths: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """
    Sort a batch first tensor by some specified lengths.
    """
    if not isinstance(tensor, torch.Tensor) or not isinstance(sequence_lengths, torch.Tensor):
        raise ConfigurationError('Both the tensor and sequence lengths must be torch.Tensors.')
    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)
    index_range = torch.arange(0, len(sequence_lengths), device=sequence_lengths.device)
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return (sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index)

def get_final_encoder_states(encoder_outputs: torch.Tensor, mask: torch.BoolTensor, bidirectional: bool = False) -> torch.Tensor:
    """
    Given the output from a `Seq2SeqEncoder`, with shape `(batch_size, sequence_length,
    encoding_dim)`, this method returns the final hidden state for each element of the batch,
    giving a tensor of shape `(batch_size, encoding_dim)`.
    """
    last_word_indices = mask.sum(1) - 1
    batch_size, _, encoder_output_dim = encoder_outputs.size()
    expanded_indices = last_word_indices.view(-1, 1, 1).expand(batch_size, 1, encoder_output_dim)
    final_encoder_output = encoder_outputs.gather(1, expanded_indices)
    final_encoder_output = final_encoder_output.squeeze(1)
    if bidirectional:
        final_forward_output = final_encoder_output[:, :encoder_output_dim // 2]
        final_backward_output = encoder_outputs[:, 0, encoder_output_dim // 2:]
        final_encoder_output = torch.cat([final_forward_output, final_backward_output], dim=-1)
    return final_encoder_output

def get_dropout_mask(dropout_probability: float, tensor_for_masking: torch.Tensor) -> torch.FloatTensor:
    """
    Computes and returns an element-wise dropout mask for a given tensor.
    """
    binary_mask = (torch.rand(tensor_for_masking.size()) > dropout_probability).to(tensor_for_masking.device)
    dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
    return dropout_mask

def masked_softmax(vector: torch.Tensor, mask: Optional[torch.BoolTensor], dim: int = -1, memory_efficient: bool = False) -> torch.Tensor:
    """
    `torch.nn.functional.softmax(vector)` does not work if some elements of `vector` should be
    masked. This performs a softmax on just the non-masked portions of `vector`.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + tiny_value_of_dtype(result.dtype))
        else:
            masked_vector = vector.masked_fill(~mask, min_value_of_dtype(vector.dtype))
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result

def masked_log_softmax(vector: torch.Tensor, mask: Optional[torch.BoolTensor], dim: int = -1) -> torch.Tensor:
    """
    `torch.nn.functional.log_softmax(vector)` does not work if some elements of `vector` should be
    masked. This performs a log_softmax on just the non-masked portions of `vector`.
    """
    if mask is not None:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        vector = vector + (mask + tiny_value_of_dtype(vector.dtype)).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)

def masked_max(vector: torch.Tensor, mask: torch.BoolTensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """
    To calculate max along certain dimensions on masked values
    """
    replaced_vector = vector.masked_fill(~mask, min_value_of_dtype(vector.dtype))
    max_value, _ = replaced_vector.max(dim=dim, keepdim=keepdim)
    return max_value

def masked_mean(vector: torch.Tensor, mask: torch.BoolTensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """
    To calculate mean along certain dimensions on masked values
    """
    replaced_vector = vector.masked_fill(~mask, 0.0)
    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    value_count = torch.sum(mask, dim=dim, keepdim=keepdim)
    return value_sum / value_count.float().clamp(min=tiny_value_of_dtype(torch.float))

def masked_flip(padded_sequence: torch.Tensor, sequence_lengths: torch.Tensor) -> torch.Tensor:
    """
    Flips a padded tensor along the time dimension without affecting masked entries.
    """
    assert padded_sequence.size(0) == len(sequence_lengths), f'sequence_lengths length ${len(sequence_lengths)} does not match batch size ${padded_sequence.size(0)}'
    num_timesteps = padded_sequence.size(1)
    flipped_padded_sequence = torch.flip(padded_sequence, [1])
    sequences = [flipped_padded_sequence[i, num_timesteps - length:] for i, length in enumerate(sequence_lengths)]
    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)

def viterbi_decode(tag_sequence: torch.Tensor, transition_matrix: torch.Tensor, tag_observations: Optional[List[int]] = None, allowed_start_transitions: Optional[torch.Tensor] = None, allowed_end_transitions: Optional[torch.Tensor] = None, top_k: Optional[int] = None) -> Union[Tuple[List[int], torch.Tensor], Tuple[List[List[int]], torch.Tensor]]:
    """
    Perform Viterbi decoding in log space over a sequence given a transition matrix
    specifying pairwise (transition) potentials between tags.
    """
    if top_k is None:
        top_k = 1
        flatten_output = True
    elif top_k >= 1:
        flatten_output = False
    else:
        raise ValueError(f'top_k must be either None or an integer >=1. Instead received {top_k}')
    sequence_length, num_tags = list(tag_sequence.size())
    has_start_end_restrictions = allowed_end_transitions is not None or allowed_start_transitions is not None
    if has_start_end_restrictions:
        if allowed_end_transitions is None:
            allowed_end_transitions = torch.zeros(num_tags)
        if allowed_start_transitions is None:
            allowed_start_transitions = torch.zeros(num_tags)
        num_tags = num_tags + 2
        new_transition_matrix = torch.zeros(num_tags, num_tags)
        new_transition_matrix[:-2, :-2] = transition_matrix
        allowed_start_transitions = torch.cat([allowed_start_transitions, torch.tensor([-math.inf, -math.inf])])
        allowed_end_transitions = torch.cat([allowed_end_transitions, torch.tensor([-math.inf, -math.inf])])
        new_transition_matrix[-2, :] = allowed_start_transitions
        new_transition_matrix[-1, :] = -math.inf
        new_transition_matrix[:, -1] = allowed_end_transitions
        new_transition_matrix[:, -2] = -math.inf
        transition_matrix = new_transition_matrix
    if tag_observations:
        if len(tag_observations) != sequence_length:
            raise ConfigurationError('Observations were provided, but they were not the same length as the sequence. Found sequence of length: {} and evidence: {}'.format(sequence_length, tag_observations))
    else:
        tag_observations = [-1 for _ in range(sequence_length)]
    if has_start_end_restrictions:
        tag_observations = [num_tags - 2] + tag_observations + [num_tags - 1]
        zero_sentinel = torch.zeros(1, num_tags)
        extra_tags_sentinel = torch.ones(sequence_length, 2) * -math.inf
        tag_sequence = torch.cat([tag_sequence, extra_tags_sentinel], -1)
        tag_sequence = torch.cat([zero_sentinel, tag_sequence, zero_sentinel], 0)
        sequence_length = tag_sequence.size(0)
    path_scores = []
    path_indices = []
    if tag_observations[0] != -1:
        one_hot = torch.zeros(num_tags)
        one_hot[tag_observations[0]] = 100000.0
        path_scores.append(one_hot.unsqueeze(0))
    else:
        path_scores.append(tag_sequence[0, :].unsqueeze(0))
    for timestep in range(1, sequence_length):
        summed_potentials = path_scores[timestep - 1].unsqueeze(2) + transition_matrix
        summed_potentials = summed_potentials.view(-1, num_tags)
        max_k = min(summed_potentials.size()[0], top_k)
        scores, paths = torch.topk(summed_potentials, k=max_k, dim=0)
        observation = tag_observations[timestep]
        if tag_observations[timestep - 1] != -1 and observation != -1:
            if transition_matrix[tag_observations[timestep - 1], observation] < -10000:
                logger.warning('The pairwise potential between tags you have passed as observations is extremely unlikely. Double check your evidence or transition potentials!')
        if observation != -1:
            one_hot = torch.zeros(num_tags)
            one_hot[observation] = 100000.0
            path_scores.append(one_hot.unsqueeze(0))
        else:
            path_scores.append(tag_sequence[timestep, :] + scores)
        path_indices.append(paths.squeeze())
    path_scores_v = path_scores[-1].view(-1)
    max_k = min(path_scores_v.size()[0], top_k)
    viterbi_scores, best_paths = torch.topk(path_scores_v, k=max_k, dim=0)
    viterbi_paths = []
    for i in range(max_k):
        viterbi_path = [best_paths[i]]
        for backward_timestep in reversed(path_indices):
            viterbi_path.append(int(backward_timestep.view(-1)[viterbi_path[-1]]))
        viterbi_path.reverse()
        if has_start_end_restrictions:
            viterbi_path = viterbi_path[1:-1]
        viterbi_path = [j % num_tags for j in viterbi_path]
        viterbi_paths.append(viterbi_path)
    if flatten_output:
        return (viterbi_paths[0], viterbi_scores[0])
    return (viterbi_paths, viterbi_scores)

def get_text_field_mask(text_field_tensors: Dict[str, Dict[str, torch.Tensor]], num_wrapping_dims: int = 0, padding_id: int = 0) -> torch.BoolTensor:
    """
    Takes the dictionary of tensors produced by a `TextField` and returns a mask
    with 0 where the tokens are padding, and 1 otherwise.
    """
    masks = []
    for indexer_name, indexer_tensors in text_field_tensors.items():
        if 'mask' in indexer_tensors:
            masks.append(indexer_tensors['mask'].bool())
    if len(masks) == 1:
        return masks[0]
    elif len(masks) > 1:
        raise ValueError('found two mask outputs; not sure which to use!')
    tensor_dims = [(tensor.dim(), tensor) for indexer_output in text_field_tensors.values() for tensor in indexer_output.values()]
    tensor_dims.sort(key=lambda x: x[0])
    smallest_dim = tensor_dims[0][0] - num_wrapping_dims
    if smallest_dim == 2:
        token_tensor = tensor_dims[0][1]
        return token_tensor != padding_id
    elif smallest_dim == 3:
        character_tensor = tensor_dims[0][1]
        return (character_tensor != padding_id).any(dim=-1)
    else:
        raise ValueError('Expected a tensor with dimension 2 or 3, found {}'.format(smallest_dim))

def get_token_ids_from_text_field_tensors(text_field_tensors: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
    """
    Our `TextFieldTensors` are complex output structures, because they try to handle a lot of
    potential variation. Sometimes, you just want to grab the token ids from this data structure.
    """
    for indexer_name, indexer_tensors in text_field_tensors.items():
        for argument_name, tensor in indexer_tensors.items():
            if argument_name in ['tokens', 'token_ids', 'input_ids']:
                return tensor
    raise NotImplementedError('Our heuristic for guessing the right token ids failed. Please open an issue on github with more detail on how you got this error, so we can implement more robust logic in this method.')

def weighted_sum(matrix: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    """
    Takes a matrix of vectors and a set of weights over the rows in the matrix (which we call an
    "attention" vector), and returns a weighted sum of the rows in the matrix.
    """
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.dim() ==