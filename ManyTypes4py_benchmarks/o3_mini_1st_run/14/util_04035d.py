#!/usr/bin/env python3
"""
Assorted utilities for working with neural networks in AllenNLP.
"""
import copy
import json
import logging
import math
import re
from collections import defaultdict, OrderedDict
from itertools import chain
from os import PathLike
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, TypeVar, Union

import numpy
import torch
import torch.distributed as dist

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import int_to_device, is_distributed, is_global_primary

logger = logging.getLogger(__name__)
T = TypeVar("T")
_V = TypeVar("_V", int, float, torch.Tensor)
StateDictType = Union[Dict[str, torch.Tensor], "OrderedDict[str, torch.Tensor]"]


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
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
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


def batch_tensor_dicts(
    tensor_dicts: List[Dict[str, torch.Tensor]], remove_trailing_dimension: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Takes a list of tensor dictionaries, where each dictionary is assumed to have matching keys,
    and returns a single dictionary with all tensors with the same key batched together.
    """
    key_to_tensors: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for tensor_dict in tensor_dicts:
        for key, tensor in tensor_dict.items():
            key_to_tensors[key].append(tensor)
    batched_tensors: Dict[str, torch.Tensor] = {}
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


def get_mask_from_sequence_lengths(sequence_lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    """
    Given a variable of shape `(batch_size,)` that represents the sequence lengths of each batch
    element, this function returns a `(batch_size, max_length)` mask variable.
    """
    ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
    range_tensor = ones.cumsum(dim=1)
    return sequence_lengths.unsqueeze(1) >= range_tensor


def sort_batch_by_length(
    tensor: torch.Tensor, sequence_lengths: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sort a batch first tensor by some specified lengths.
    """
    if not isinstance(tensor, torch.Tensor) or not isinstance(sequence_lengths, torch.Tensor):
        raise ConfigurationError("Both the tensor and sequence lengths must be torch.Tensors.")
    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)
    index_range = torch.arange(0, len(sequence_lengths), device=sequence_lengths.device)
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return (sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index)


def get_final_encoder_states(
    encoder_outputs: torch.Tensor, mask: torch.Tensor, bidirectional: bool = False
) -> torch.Tensor:
    """
    Given the output from a `Seq2SeqEncoder`, returns the final hidden state for each element of the batch.
    """
    last_word_indices = mask.sum(1) - 1
    batch_size, _, encoder_output_dim = encoder_outputs.size()
    expanded_indices = last_word_indices.view(-1, 1, 1).expand(batch_size, 1, encoder_output_dim)
    final_encoder_output = encoder_outputs.gather(1, expanded_indices)
    final_encoder_output = final_encoder_output.squeeze(1)
    if bidirectional:
        final_forward_output = final_encoder_output[:, : encoder_output_dim // 2]
        final_backward_output = encoder_outputs[:, 0, encoder_output_dim // 2 :]
        final_encoder_output = torch.cat([final_forward_output, final_backward_output], dim=-1)
    return final_encoder_output


def get_dropout_mask(dropout_probability: float, tensor_for_masking: torch.Tensor) -> torch.FloatTensor:
    """
    Computes and returns an element-wise dropout mask for a given tensor.
    """
    binary_mask = (torch.rand(tensor_for_masking.size()) > dropout_probability).to(tensor_for_masking.device)
    dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
    return dropout_mask


def masked_softmax(
    vector: torch.Tensor, mask: Optional[torch.Tensor], dim: int = -1, memory_efficient: bool = False
) -> torch.Tensor:
    """
    Performs a softmax on just the non-masked portions of `vector`.
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


def masked_log_softmax(vector: torch.Tensor, mask: Optional[torch.Tensor], dim: int = -1) -> torch.Tensor:
    """
    Performs a log_softmax on just the non-masked portions of `vector`.
    """
    if mask is not None:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        vector = vector + (mask + tiny_value_of_dtype(vector.dtype)).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


def masked_max(vector: torch.Tensor, mask: torch.BoolTensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """
    Calculate max along certain dimensions on masked values.
    """
    replaced_vector = vector.masked_fill(~mask, min_value_of_dtype(vector.dtype))
    max_value, _ = replaced_vector.max(dim=dim, keepdim=keepdim)
    return max_value


def masked_mean(vector: torch.Tensor, mask: torch.BoolTensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """
    Calculate mean along certain dimensions on masked values.
    """
    replaced_vector = vector.masked_fill(~mask, 0.0)
    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    value_count = torch.sum(mask, dim=dim, keepdim=keepdim)
    return value_sum / value_count.float().clamp(min=tiny_value_of_dtype(torch.float))


def masked_flip(padded_sequence: torch.Tensor, sequence_lengths: torch.Tensor) -> torch.Tensor:
    """
    Flips a padded tensor along the time dimension without affecting masked entries.
    """
    assert padded_sequence.size(0) == len(sequence_lengths), (
        f"sequence_lengths length {len(sequence_lengths)} does not match batch size {padded_sequence.size(0)}"
    )
    num_timesteps = padded_sequence.size(1)
    flipped_padded_sequence = torch.flip(padded_sequence, [1])
    sequences = [flipped_padded_sequence[i, num_timesteps - length :] for i, length in enumerate(sequence_lengths)]
    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)


def viterbi_decode(
    tag_sequence: torch.Tensor,
    transition_matrix: torch.Tensor,
    tag_observations: Optional[List[int]] = None,
    allowed_start_transitions: Optional[torch.Tensor] = None,
    allowed_end_transitions: Optional[torch.Tensor] = None,
    top_k: Optional[int] = None,
) -> Union[Tuple[List[int], torch.Tensor], Tuple[List[List[int]], torch.Tensor]]:
    """
    Perform Viterbi decoding in log space over a sequence.
    """
    if top_k is None:
        top_k = 1
        flatten_output = True
    elif top_k >= 1:
        flatten_output = False
    else:
        raise ValueError(f"top_k must be either None or an integer >=1. Instead received {top_k}")
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
            raise ConfigurationError(
                "Observations were provided, but they were not the same length as the sequence. "
                "Found sequence of length: {} and evidence: {}".format(sequence_length, tag_observations)
            )
    else:
        tag_observations = [-1 for _ in range(sequence_length)]
    if has_start_end_restrictions:
        tag_observations = [num_tags - 2] + tag_observations + [num_tags - 1]
        zero_sentinel = torch.zeros(1, num_tags)
        extra_tags_sentinel = torch.ones(sequence_length, 2) * -math.inf
        tag_sequence = torch.cat([tag_sequence, extra_tags_sentinel], -1)
        tag_sequence = torch.cat([zero_sentinel, tag_sequence, zero_sentinel], 0)
        sequence_length = tag_sequence.size(0)
    path_scores: List[torch.Tensor] = []
    path_indices: List[torch.Tensor] = []
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
                logger.warning(
                    "The pairwise potential between tags you have passed as observations is extremely unlikely. "
                    "Double check your evidence or transition potentials!"
                )
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
    viterbi_paths: List[List[int]] = []
    for i in range(max_k):
        viterbi_path = [best_paths[i].item()]
        for backward_timestep in reversed(path_indices):
            viterbi_path.append(int(backward_timestep.view(-1)[viterbi_path[-1]]))
        viterbi_path.reverse()
        if has_start_end_restrictions:
            viterbi_path = viterbi_path[1:-1]
        viterbi_path = [j % num_tags for j in viterbi_path]
        viterbi_paths.append(viterbi_path)
    if flatten_output:
        return (viterbi_paths[0], viterbi_scores)
    return (viterbi_paths, viterbi_scores)


def get_text_field_mask(
    text_field_tensors: Dict[str, Dict[str, torch.Tensor]], num_wrapping_dims: int = 0, padding_id: int = 0
) -> torch.BoolTensor:
    """
    Takes the dictionary of tensors produced by a `TextField` and returns a mask
    with 0 where the tokens are padding, and 1 otherwise.
    """
    masks: List[torch.BoolTensor] = []
    for indexer_name, indexer_tensors in text_field_tensors.items():
        if "mask" in indexer_tensors:
            masks.append(indexer_tensors["mask"].bool())
    if len(masks) == 1:
        return masks[0]
    elif len(masks) > 1:
        raise ValueError("found two mask outputs; not sure which to use!")
    tensor_dims: List[Tuple[int, torch.Tensor]] = [
        (tensor.dim(), tensor)
        for indexer_output in text_field_tensors.values()
        for tensor in indexer_output.values()
    ]
    tensor_dims.sort(key=lambda x: x[0])
    smallest_dim = tensor_dims[0][0] - num_wrapping_dims
    if smallest_dim == 2:
        token_tensor = tensor_dims[0][1]
        return token_tensor != padding_id
    elif smallest_dim == 3:
        character_tensor = tensor_dims[0][1]
        return (character_tensor != padding_id).any(dim=-1)
    else:
        raise ValueError("Expected a tensor with dimension 2 or 3, found {}".format(smallest_dim))


def get_token_ids_from_text_field_tensors(text_field_tensors: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
    """
    Retrieves token ids from a TextFieldTensors data structure.
    """
    for indexer_name, indexer_tensors in text_field_tensors.items():
        for argument_name, tensor in indexer_tensors.items():
            if argument_name in ["tokens", "token_ids", "input_ids"]:
                return tensor
    raise NotImplementedError(
        "Our heuristic for guessing the right token ids failed. Please open an issue on github with more detail on how you got this error, so we can implement more robust logic in this method."
    )


def weighted_sum(matrix: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    """
    Takes a matrix of vectors and a set of weights over the rows in the matrix,
    and returns a weighted sum of the rows in the matrix.
    """
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix)
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)


def sequence_cross_entropy_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: Union[torch.Tensor, torch.BoolTensor],
    average: Optional[str] = "batch",
    label_smoothing: Optional[float] = None,
    gamma: Optional[float] = None,
    alpha: Optional[Union[float, List[float]]] = None,
) -> torch.Tensor:
    """
    Computes the cross entropy loss of a sequence, weighted with respect to
    some user provided weights.
    """
    if average not in {None, "token", "batch"}:
        raise ValueError(f"Got average f{average}, expected one of None, 'token', or 'batch'")
    weights = weights.to(logits.dtype)
    non_batch_dims = tuple(range(1, len(weights.shape)))
    weights_batch_sum = weights.sum(dim=non_batch_dims)
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    targets_flat = targets.view(-1, 1).long()
    if gamma:
        probs_flat = log_probs_flat.exp()
        probs_flat = torch.gather(probs_flat, dim=1, index=targets_flat)
        focal_factor = (1.0 - probs_flat) ** gamma
        focal_factor = focal_factor.view(*targets.size())
        weights = weights * focal_factor
    if alpha is not None:
        if isinstance(alpha, (float, int)):
            alpha_factor = torch.tensor([1.0 - float(alpha), float(alpha)], dtype=weights.dtype, device=weights.device)
        elif isinstance(alpha, (list, numpy.ndarray, torch.Tensor)):
            alpha_factor = torch.tensor(alpha, dtype=weights.dtype, device=weights.device)
            if not alpha_factor.size():
                alpha_factor = alpha_factor.view(1)
                alpha_factor = torch.cat([1 - alpha_factor, alpha_factor])
        else:
            raise TypeError("alpha must be float, list of float, or torch.FloatTensor, {} provided.".format(type(alpha)))
        alpha_factor = torch.gather(alpha_factor, dim=0, index=targets_flat.view(-1)).view(*targets.size())
        weights = weights * alpha_factor
    if label_smoothing is not None and label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / num_classes
        smoothed_targets = torch.full_like(log_probs_flat, smoothing_value).scatter_(
            -1, targets_flat, 1.0 - label_smoothing + smoothing_value
        )
        negative_log_likelihood_flat = -log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        negative_log_likelihood_flat = -torch.gather(log_probs_flat, dim=1, index=targets_flat)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    negative_log_likelihood = negative_log_likelihood * weights
    if average == "batch":
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (
            weights_batch_sum + tiny_value_of_dtype(negative_log_likelihood.dtype)
        )
        num_non_empty_sequences = (weights_batch_sum > 0).sum() + tiny_value_of_dtype(negative_log_likelihood.dtype)
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return negative_log_likelihood.sum() / (weights_batch_sum.sum() + tiny_value_of_dtype(negative_log_likelihood.dtype))
    else:
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (
            weights_batch_sum + tiny_value_of_dtype(negative_log_likelihood.dtype)
        )
        return per_batch_loss


def replace_masked_values(tensor: torch.Tensor, mask: torch.Tensor, replace_with: Union[int, float]) -> torch.Tensor:
    """
    Replaces all masked values in `tensor` with `replace_with`.
    """
    if tensor.dim() != mask.dim():
        raise ConfigurationError("tensor.dim() (%d) != mask.dim() (%d)" % (tensor.dim(), mask.dim()))
    return tensor.masked_fill(~mask, replace_with)


def tensors_equal(tensor1: Any, tensor2: Any, tolerance: float = 1e-12) -> bool:
    """
    A check for tensor equality (by value), supporting nested containers.
    """
    if isinstance(tensor1, (list, tuple)):
        if not isinstance(tensor2, (list, tuple)) or len(tensor1) != len(tensor2):
            return False
        return all((tensors_equal(t1, t2, tolerance) for t1, t2 in zip(tensor1, tensor2)))
    elif isinstance(tensor1, dict):
        if not isinstance(tensor2, dict):
            return False
        if tensor1.keys() != tensor2.keys():
            return False
        return all((tensors_equal(tensor1[key], tensor2[key], tolerance) for key in tensor1))
    elif isinstance(tensor1, torch.Tensor):
        if not isinstance(tensor2, torch.Tensor):
            return False
        if tensor1.size() != tensor2.size():
            return False
        if tensor1.dtype == torch.bool or tensor2.dtype == torch.bool:
            return (tensor1 == tensor2).all()
        return ((tensor1 - tensor2).abs().float() < tolerance).all()
    else:
        try:
            return tensor1 == tensor2
        except RuntimeError:
            print(type(tensor1), type(tensor2))
            raise


def device_mapping(cuda_device: int) -> Callable[[torch.Storage, Any], torch.Storage]:
    """
    Returns a map_location function for torch.load to load models onto a specific device.
    """
    def inner_device_mapping(storage: torch.Storage, location: Any) -> torch.Storage:
        if cuda_device >= 0:
            return storage.cuda(cuda_device)
        else:
            return storage

    return inner_device_mapping


def read_state_dict(
    path: Union[PathLike, str],
    strip_prefix: Optional[str] = None,
    ignore: Optional[List[str]] = None,
    strict: bool = True,
    cuda_device: int = -1,
) -> StateDictType:
    """
    Read a PyTorch model state dictionary from a checkpoint at the given `path`.
    """
    state = torch.load(path, map_location=device_mapping(cuda_device))
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    if ignore is not None and (not isinstance(ignore, list)):
        raise ValueError("'ignore' parameter should be a list")
    strip_prefix_used: Optional[bool] = None
    ignore_used: Optional[List[bool]] = None
    if strict and strip_prefix is not None:
        strip_prefix_used = False
    if strict and ignore:
        ignore_used = [False] * len(ignore)
    for key in state.keys():
        ignore_key = False
        if ignore:
            for i, pattern in enumerate(ignore):
                if re.match(pattern, key):
                    if ignore_used is not None:
                        ignore_used[i] = True
                    logger.warning("ignoring %s from state dict", key)
                    ignore_key = True
                    break
        if ignore_key:
            continue
        new_key = key
        if strip_prefix and key.startswith(strip_prefix):
            strip_prefix_used = True
            new_key = key[len(strip_prefix) :]
            if not new_key:
                raise ValueError("'strip_prefix' resulted in an empty string for a key")
        out[new_key] = state[key]
    if strip_prefix_used is False:
        raise ValueError(f"'strip_prefix' of '{strip_prefix}' was never used")
    if ignore is not None and ignore_used is not None:
        for pattern, used in zip(ignore, ignore_used):
            if not used:
                raise ValueError(f"'ignore' pattern '{pattern}' didn't have any matches")
    return out


def combine_tensors(combination: str, tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Combines a list of tensors using element-wise operations and concatenation.
    """
    if len(tensors) > 9:
        raise ConfigurationError("Double-digit tensor lists not currently supported")
    combination = combination.replace("x", "1").replace("y", "2")
    to_concatenate = [_get_combination(piece, tensors) for piece in combination.split(",")]
    return torch.cat(to_concatenate, dim=-1)


def _rindex(sequence: Sequence[T], obj: T) -> int:
    """
    Return zero-based index in the sequence of the last item whose value is equal to obj.
    """
    for i in range(len(sequence) - 1, -1, -1):
        if sequence[i] == obj:
            return i
    raise ValueError(f"Unable to find {obj} in sequence {sequence}.")


def _get_combination(combination: str, tensors: List[torch.Tensor]) -> torch.Tensor:
    if combination.isdigit():
        index = int(combination) - 1
        return tensors[index]
    else:
        if len(combination) != 3:
            raise ConfigurationError("Invalid combination: " + combination)
        first_tensor = _get_combination(combination[0], tensors)
        second_tensor = _get_combination(combination[2], tensors)
        operation = combination[1]
        if operation == "*":
            return first_tensor * second_tensor
        elif operation == "/":
            return first_tensor / second_tensor
        elif operation == "+":
            return first_tensor + second_tensor
        elif operation == "-":
            return first_tensor - second_tensor
        else:
            raise ConfigurationError("Invalid operation: " + operation)


def combine_tensors_and_multiply(
    combination: str, tensors: List[torch.Tensor], weights: torch.nn.Parameter
) -> torch.Tensor:
    """
    Combines tensors and performs weighted multiplication.
    """
    if len(tensors) > 9:
        raise ConfigurationError("Double-digit tensor lists not currently supported")
    combination = combination.replace("x", "1").replace("y", "2")
    pieces = combination.split(",")
    tensor_dims = [tensor.size(-1) for tensor in tensors]
    combination_dims = [_get_combination_dim(piece, tensor_dims) for piece in pieces]
    dims_so_far = 0
    to_sum: List[torch.Tensor] = []
    for piece, combination_dim in zip(pieces, combination_dims):
        weight = weights[dims_so_far : dims_so_far + combination_dim]
        dims_so_far += combination_dim
        to_sum.append(_get_combination_and_multiply(piece, tensors, weight))
    result = to_sum[0]
    for result_piece in to_sum[1:]:
        result = result + result_piece
    return result


def _get_combination_and_multiply(
    combination: str, tensors: List[torch.Tensor], weight: torch.Tensor
) -> torch.Tensor:
    if combination.isdigit():
        index = int(combination) - 1
        return torch.matmul(tensors[index], weight)
    else:
        if len(combination) != 3:
            raise ConfigurationError("Invalid combination: " + combination)
        first_tensor = _get_combination(combination[0], tensors)
        second_tensor = _get_combination(combination[2], tensors)
        operation = combination[1]
        if operation == "*":
            if first_tensor.dim() > 4 or second_tensor.dim() > 4:
                raise ValueError("Tensors with dim > 4 not currently supported")
            desired_dim = max(first_tensor.dim(), second_tensor.dim()) - 1
            if first_tensor.dim() == 4:
                expanded_dim = _rindex(first_tensor.size(), 1)
                first_tensor = first_tensor.squeeze(expanded_dim)
            if second_tensor.dim() == 4:
                expanded_dim = _rindex(second_tensor.size(), 1)
                second_tensor = second_tensor.squeeze(expanded_dim)
            intermediate = first_tensor * weight
            result = torch.matmul(intermediate, second_tensor.transpose(-1, -2))
            if result.dim() == desired_dim + 1:
                result = result.squeeze(-1)
            return result
        elif operation == "/":
            if first_tensor.dim() > 4 or second_tensor.dim() > 4:
                raise ValueError("Tensors with dim > 4 not currently supported")
            desired_dim = max(first_tensor.dim(), second_tensor.dim()) - 1
            if first_tensor.dim() == 4:
                expanded_dim = _rindex(first_tensor.size(), 1)
                first_tensor = first_tensor.squeeze(expanded_dim)
            if second_tensor.dim() == 4:
                expanded_dim = _rindex(second_tensor.size(), 1)
                second_tensor = second_tensor.squeeze(expanded_dim)
            intermediate = first_tensor * weight
            result = torch.matmul(intermediate, second_tensor.pow(-1).transpose(-1, -2))
            if result.dim() == desired_dim + 1:
                result = result.squeeze(-1)
            return result
        elif operation == "+":
            return torch.matmul(first_tensor, weight) + torch.matmul(second_tensor, weight)
        elif operation == "-":
            return torch.matmul(first_tensor, weight) - torch.matmul(second_tensor, weight)
        else:
            raise ConfigurationError("Invalid operation: " + operation)


def get_combined_dim(combination: str, tensor_dims: List[int]) -> int:
    """
    Compute the resultant dimension when combining tensors.
    """
    if len(tensor_dims) > 9:
        raise ConfigurationError("Double-digit tensor lists not currently supported")
    combination = combination.replace("x", "1").replace("y", "2")
    return sum((_get_combination_dim(piece, tensor_dims) for piece in combination.split(",")))


def _get_combination_dim(combination: str, tensor_dims: List[int]) -> int:
    if combination.isdigit():
        index = int(combination) - 1
        return tensor_dims[index]
    else:
        if len(combination) != 3:
            raise ConfigurationError("Invalid combination: " + combination)
        first_tensor_dim = _get_combination_dim(combination[0], tensor_dims)
        second_tensor_dim = _get_combination_dim(combination[2], tensor_dims)
        operation = combination[1]
        if first_tensor_dim != second_tensor_dim:
            raise ConfigurationError('Tensor dims must match for operation "{}"'.format(operation))
        return first_tensor_dim


def logsumexp(tensor: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """
    A numerically stable computation of logsumexp.
    """
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + stable_vec.exp().sum(dim, keepdim=keepdim).log()


def get_device_of(tensor: torch.Tensor) -> int:
    """
    Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()


def flatten_and_batch_shift_indices(indices: torch.LongTensor, sequence_length: int) -> torch.LongTensor:
    """
    Returns a flattened vector that correctly indexes into the flattened target tensor.
    """
    if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
        raise ConfigurationError(f"All elements in indices should be in range (0, {sequence_length - 1})")
    offsets = get_range_vector(indices.size(0), get_device_of(indices)) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)
    offset_indices = indices + offsets
    offset_indices = offset_indices.view(-1)
    return offset_indices


def batched_index_select(
    target: torch.Tensor, indices: torch.LongTensor, flattened_indices: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Returns selected values in the target with respect to the provided indices.
    """
    if flattened_indices is None:
        flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))
    flattened_target = target.view(-1, target.size(-1))
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]
    selected_targets = flattened_selected.view(*selected_shape)
    return selected_targets


def masked_index_fill(target: torch.Tensor, indices: torch.LongTensor, mask: torch.Tensor, fill_value: int = 1) -> torch.Tensor:
    """
    Fills in the tensor at specified indices with fill_value based on mask.
    """
    mask = mask.bool()
    prev_shape = target.size()
    flattened_indices = flatten_and_batch_shift_indices(indices * mask, target.size(1))
    mask = mask.view(-1)
    flattened_target = target.view(-1, 1)
    unmasked_indices = flattened_indices[mask].unsqueeze(-1)
    flattened_target = flattened_target.scatter(0, unmasked_indices, fill_value)
    filled_target = flattened_target.reshape(prev_shape)
    return filled_target


def masked_index_replace(
    target: torch.Tensor, indices: torch.LongTensor, mask: torch.Tensor, replace: torch.Tensor
) -> torch.Tensor:
    """
    Replaces values in target at specified indices with values from replace based on mask.
    """
    target = target.clone()
    mask = mask.bool()
    prev_shape = target.size()
    flattened_indices = flatten_and_batch_shift_indices(indices * mask, target.size(1))
    flattened_target = target.view(-1, target.size(-1))
    mask = mask.view(-1)
    flattened_target[flattened_indices[mask]] = replace.view(-1, replace.size(-1))[mask]
    replaced_target = flattened_target.reshape(prev_shape)
    return replaced_target


def batched_span_select(
    target: torch.Tensor, spans: torch.LongTensor
) -> Tuple[torch.Tensor, torch.BoolTensor]:
    """
    Returns segmented spans in the target with respect to the provided span indices.
    """
    span_starts, span_ends = spans.split(1, dim=-1)
    span_widths = span_ends - span_starts
    max_batch_span_width = span_widths.max().item() + 1
    max_span_range_indices = get_range_vector(max_batch_span_width, get_device_of(target)).view(1, 1, -1)
    span_mask = max_span_range_indices <= span_widths
    raw_span_indices = span_starts + max_span_range_indices
    span_mask = span_mask & (raw_span_indices < target.size(1)) & (0 <= raw_span_indices)
    span_indices = raw_span_indices * span_mask
    span_embeddings = batched_index_select(target, span_indices)
    return (span_embeddings, span_mask)


def flattened_index_select(target: torch.Tensor, indices: torch.LongTensor) -> torch.Tensor:
    """
    Selects subsets of the target based on indices.
    """
    if indices.dim() != 2:
        raise ConfigurationError(
            "Indices passed to flattened_index_select had shape {} but only 2 dimensional inputs are supported.".format(
                indices.size()
            )
        )
    flattened_selected = target.index_select(1, indices.view(-1))
    selected = flattened_selected.view(target.size(0), indices.size(0), indices.size(1), -1)
    return selected


def get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0.
    """
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)


def bucket_values(distances: torch.Tensor, num_identity_buckets: int = 4, num_total_buckets: int = 10) -> torch.Tensor:
    """
    Buckets values into semi-logscale buckets.
    """
    logspace_index = (distances.float().log() / math.log(2)).floor().long() + (num_identity_buckets - 1)
    use_identity_mask = (distances <= num_identity_buckets).long()
    use_buckets_mask = 1 + -1 * use_identity_mask
    combined_index = use_identity_mask * distances + use_buckets_mask * logspace_index
    return combined_index.clamp(0, num_total_buckets - 1)


def add_sentence_boundary_token_ids(
    tensor: torch.Tensor, mask: torch.BoolTensor, sentence_begin_token: Any, sentence_end_token: Any
) -> Tuple[torch.Tensor, torch.BoolTensor]:
    """
    Add begin/end of sentence tokens to the batch of sentences.
    """
    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
    tensor_shape = list(tensor.data.shape)
    new_shape = list(tensor_shape)
    new_shape[1] = tensor_shape[1] + 2
    tensor_with_boundary_tokens = tensor.new_zeros(*new_shape, device=tensor.device)
    if len(tensor_shape) == 2:
        tensor_with_boundary_tokens[:, 1:-1] = tensor
        tensor_with_boundary_tokens[:, 0] = sentence_begin_token
        for i, j in enumerate(sequence_lengths):
            tensor_with_boundary_tokens[i, j + 1] = sentence_end_token
        new_mask = tensor_with_boundary_tokens != 0
    elif len(tensor_shape) == 3:
        tensor_with_boundary_tokens[:, 1:-1, :] = tensor
        sentence_begin_token = sentence_begin_token.detach().to(tensor.device)
        sentence_end_token = sentence_end_token.detach().to(tensor.device)
        for i, j in enumerate(sequence_lengths):
            tensor_with_boundary_tokens[i, 0, :] = sentence_begin_token
            tensor_with_boundary_tokens[i, j + 1, :] = sentence_end_token
        new_mask = (tensor_with_boundary_tokens > 0).sum(dim=-1) > 0
    else:
        raise ValueError("add_sentence_boundary_token_ids only accepts 2D and 3D input")
    return (tensor_with_boundary_tokens, new_mask)


def remove_sentence_boundaries(
    tensor: torch.Tensor, mask: torch.BoolTensor
) -> Tuple[torch.Tensor, torch.BoolTensor]:
    """
    Remove begin/end of sentence embeddings from the batch of sentences.
    """
    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
    tensor_shape = list(tensor.data.shape)
    new_shape = list(tensor_shape)
    new_shape[1] = tensor_shape[1] - 2
    tensor_without_boundary_tokens = tensor.new_zeros(*new_shape)
    new_mask = tensor.new_zeros((new_shape[0], new_shape[1]), dtype=torch.bool)
    for i, j in enumerate(sequence_lengths):
        if j > 2:
            tensor_without_boundary_tokens[i, : j - 2, :] = tensor[i, 1 : j - 1, :]
            new_mask[i, : j - 2] = True
    return (tensor_without_boundary_tokens, new_mask)


def add_positional_features(
    tensor: torch.Tensor, min_timescale: float = 1.0, max_timescale: float = 10000.0
) -> torch.Tensor:
    """
    Implements the frequency-based positional encoding.
    """
    _, timesteps, hidden_dim = tensor.size()
    timestep_range = get_range_vector(timesteps, get_device_of(tensor)).data.float()
    num_timescales = hidden_dim // 2
    timescale_range = get_range_vector(num_timescales, get_device_of(tensor)).data.float()
    log_timescale_increments = math.log(float(max_timescale) / float(min_timescale)) / float(num_timescales - 1)
    inverse_timescales = min_timescale * torch.exp(timescale_range * -log_timescale_increments)
    scaled_time = timestep_range.unsqueeze(1) * inverse_timescales.unsqueeze(0)
    sinusoids = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 1)
    if hidden_dim % 2 != 0:
        sinusoids = torch.cat([sinusoids, sinusoids.new_zeros(timesteps, 1)], 1)
    return tensor + sinusoids.unsqueeze(0)


def clone(module: torch.nn.Module, num_copies: int) -> torch.nn.ModuleList:
    """Produce N identical layers."""
    return torch.nn.ModuleList((copy.deepcopy(module) for _ in range(num_copies)))


def combine_initial_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor of shape (d1, ..., dn, sequence_length),
    Return a view that's (d1 * ... * dn, sequence_length).
    """
    if tensor.dim() <= 2:
        return tensor
    else:
        return tensor.view(-1, tensor.size(-1))


def uncombine_initial_dims(tensor: torch.Tensor, original_size: Sequence[int]) -> torch.Tensor:
    """
    Given a tensor of embeddings with shape (d1 * ... * dn, sequence_length, embedding_dim)
    and the original shape (d1, ..., dn, sequence_length), return the reshaped tensor.
    """
    if len(original_size) <= 2:
        return tensor
    else:
        view_args = list(original_size) + [tensor.size(-1)]
        return tensor.view(*view_args)


def inspect_parameters(module: torch.nn.Module, quiet: bool = False) -> Dict[str, Any]:
    """
    Inspects the model/module parameters and their tunability.
    """
    results: Dict[str, Any] = {}
    for name, param in sorted(module.named_parameters()):
        keys = name.split(".")
        write_to: Dict[str, Any] = results
        for key in keys[:-1]:
            if key not in write_to:
                write_to[key] = {}
            write_to = write_to[key]
        write_to[keys[-1]] = "tunable" if param.requires_grad else "frozen"
    if not quiet:
        print(json.dumps(results, indent=4))
    return results


def find_text_field_embedder(model: torch.nn.Module) -> torch.nn.Module:
    """
    Takes a `Model` and returns the `TextFieldEmbedder`.
    """
    from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder

    for module in model.modules():
        if isinstance(module, TextFieldEmbedder):
            return module
    raise ValueError("Couldn't find TextFieldEmbedder!")


def find_embedding_layer(model: torch.nn.Module) -> torch.nn.Module:
    """
    Makes a best guess about which module is the embedding layer.
    """
    from transformers.models.gpt2.modeling_gpt2 import GPT2Model
    from transformers.models.bert.modeling_bert import BertEmbeddings
    from transformers.models.albert.modeling_albert import AlbertEmbeddings
    from transformers.models.roberta.modeling_roberta import RobertaEmbeddings
    from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
    from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
    from allennlp.modules.token_embedders.embedding import Embedding

    for module in model.modules():
        if isinstance(module, BertEmbeddings):
            return module.word_embeddings
        if isinstance(module, RobertaEmbeddings):
            return module.word_embeddings
        if isinstance(module, AlbertEmbeddings):
            return module.word_embeddings
        if isinstance(module, GPT2Model):
            return module.wte
    for module in model.modules():
        if isinstance(module, TextFieldEmbedder):
            if isinstance(module, BasicTextFieldEmbedder):
                if len(module._token_embedders) == 1:
                    embedder = list(module._token_embedders.values())[0]
                    if isinstance(embedder, Embedding):
                        if embedder._projection is None:
                            return embedder
            return module
    raise RuntimeError("No embedding module found!")


def get_token_offsets_from_text_field_inputs(text_field_inputs: List[Any]) -> Optional[Any]:
    """
    Given a list of inputs to a TextFieldEmbedder, returns token offsets if available.
    """
    for input_index, text_field_input in enumerate(text_field_inputs):
        if not isinstance(text_field_input, dict):
            continue
        for input_value in text_field_input.values():
            if not isinstance(input_value, dict):
                continue
            for embedder_arg_name, embedder_arg_value in input_value.items():
                if embedder_arg_name == "offsets":
                    return embedder_arg_value
    return None


def extend_layer(layer: torch.nn.Module, new_dim: int) -> None:
    """
    Extends a layer (Linear or Bilinear) to have new_dim outputs.
    """
    valid_layers = [torch.nn.Linear, torch.nn.Bilinear]
    if not any([isinstance(layer, i) for i in valid_layers]):
        raise ConfigurationError("Inappropriate layer type")
    extend_dim = new_dim - layer.out_features
    if not extend_dim:
        return
    if isinstance(layer, torch.nn.Linear):
        new_weight = torch.FloatTensor(extend_dim, layer.in_features)
    elif isinstance(layer, torch.nn.Bilinear):
        new_weight = torch.FloatTensor(extend_dim, layer.in1_features, layer.in2_features)
    new_bias = torch.FloatTensor(extend_dim)
    torch.nn.init.xavier_uniform_(new_weight)
    torch.nn.init.zeros_(new_bias)
    device = layer.weight.device
    layer.weight = torch.nn.Parameter(torch.cat([layer.weight.data, new_weight.to(device)], dim=0), requires_grad=layer.weight.requires_grad)
    layer.bias = torch.nn.Parameter(torch.cat([layer.bias.data, new_bias.to(device)], dim=0), requires_grad=layer.bias.requires_grad)
    layer.out_features = new_dim


def masked_topk(
    input_: torch.Tensor, mask: torch.BoolTensor, k: Union[int, torch.Tensor], dim: int = -1
) -> Tuple[torch.Tensor, torch.BoolTensor, torch.Tensor]:
    """
    Extracts the top-k items along a certain dimension given a mask.
    """
    if input_.size() != mask.size():
        raise ValueError("`input_` and `mask` must have the same shape.")
    if not -input_.dim() <= dim < input_.dim():
        raise ValueError("`dim` must be in `[-input_.dim(), input_.dim())`")
    dim = (dim + input_.dim()) % input_.dim()
    max_k = k if isinstance(k, int) else k.max()
    permutation = list(range(input_.dim()))
    permutation.pop(dim)
    permutation += [dim]
    reverse_permutation = list(range(input_.dim() - 1))
    reverse_permutation.insert(dim, -1)
    other_dims_size = list(input_.size())
    other_dims_size.pop(dim)
    permuted_size = other_dims_size + [max_k]
    if isinstance(k, int):
        k = k * torch.ones(*other_dims_size, dtype=torch.long, device=mask.device)
    elif list(k.size()) != other_dims_size:
        raise ValueError("`k` must have the same shape as `input_` with dimension `dim` removed.")
    num_items = input_.size(dim)
    input_ = input_.permute(*permutation).reshape(-1, num_items)
    mask = mask.permute(*permutation).reshape(-1, num_items)
    k = k.reshape(-1)
    input_ = replace_masked_values(input_, mask, min_value_of_dtype(input_.dtype))
    _, top_indices = input_.topk(max_k, 1)
    top_indices_mask = get_mask_from_sequence_lengths(k, max_k).bool()
    fill_value, _ = top_indices.max(dim=1, keepdim=True)
    top_indices = torch.where(top_indices_mask, top_indices, fill_value)
    top_indices, _ = top_indices.sort(1)
    sequence_mask = mask.gather(1, top_indices)
    top_mask = top_indices_mask & sequence_mask
    top_input = input_.gather(1, top_indices)
    return (
        top_input.reshape(*permuted_size).permute(*reverse_permutation),
        top_mask.reshape(*permuted_size).permute(*reverse_permutation),
        top_indices.reshape(*permuted_size).permute(*reverse_permutation),
    )


def info_value_of_dtype(dtype: torch.dtype) -> Union[torch.finfo, torch.iinfo]:
    """
    Returns the finfo or iinfo object of a given PyTorch data type.
    """
    if dtype == torch.bool:
        raise TypeError("Does not support torch.bool")
    elif torch.tensor(0, dtype=dtype).is_floating_point():
        return torch.finfo(dtype)
    else:
        return torch.iinfo(dtype)


def min_value_of_dtype(dtype: torch.dtype) -> Union[float, int]:
    """
    Returns the minimum value of a given PyTorch data type.
    """
    return info_value_of_dtype(dtype).min


def max_value_of_dtype(dtype: torch.dtype) -> Union[float, int]:
    """
    Returns the maximum value of a given PyTorch data type.
    """
    return info_value_of_dtype(dtype).max


def tiny_value_of_dtype(dtype: torch.dtype) -> float:
    """
    Returns a moderately tiny value for a given PyTorch floating point data type.
    """
    if not torch.tensor(0, dtype=dtype).is_floating_point():
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 0.0001
    else:
        raise TypeError("Does not support dtype " + str(dtype))


def distributed_device() -> torch.device:
    """
    Get the correct torch.device for distributed communication.
    """
    if not is_distributed():
        raise RuntimeError("'distributed_device()' can only be called within a distributed process group")
    return int_to_device(-1 if dist.get_backend() != "nccl" else torch.cuda.current_device())


def dist_reduce(value: _V, reduce_op: Any) -> _V:
    """
    Reduces the given value across all distributed worker nodes.
    """
    if not is_distributed():
        return value
    device = distributed_device()
    if isinstance(value, torch.Tensor):
        value_tensor = value.clone().to(device)
    else:
        value_tensor = torch.tensor(value, device=device)
    dist.all_reduce(value_tensor, op=reduce_op)
    if isinstance(value, torch.Tensor):
        return value_tensor
    return value_tensor.item()  # type: ignore


def dist_reduce_sum(value: _V) -> _V:
    """
    Sums the given value across distributed worker nodes.
    """
    if not is_distributed():
        return value
    return dist_reduce(value, dist.ReduceOp.SUM)


def _collect_state_dict(
    module: torch.nn.Module, state_dict: Optional[StateDictType], recurse: bool = True, prefix: str = ""
) -> Tuple[StateDictType, List[str], List[str]]:
    """
    Collect a module's state dict across distributed processes.
    """
    dist_device = distributed_device()
    state_dict_device = int_to_device(-1)
    missing_keys: List[str] = []
    unexpected_keys: List[str] = []
    if recurse:
        current_state_dict = module.state_dict()
    else:
        current_state_dict = OrderedDict(
            chain(((n, p.data) for n, p in module.named_parameters(recurse=False)), module.named_buffers(recurse=False))
        )
    keys = list(current_state_dict.keys())
    if is_global_primary():
        assert state_dict is not None
        for key in state_dict:
            if key not in keys:
                unexpected_keys.append(key)
    for key in keys:
        tensor = current_state_dict[key]
        if is_global_primary():
            assert state_dict is not None
            if key in state_dict:
                tensor = state_dict[key]
            else:
                missing_keys.append(key)
        logger.debug("Broadcasting distributed parameter '%s'", prefix + key)
        tensor = tensor.to(dist_device).contiguous()
        dist.broadcast(tensor, 0)
        current_state_dict[key] = tensor.to(state_dict_device)
    return (current_state_dict, missing_keys, unexpected_keys)


class _IncompatibleKeys(NamedTuple):
    missing_keys: List[str]
    unexpected_keys: List[str]

    def __repr__(self) -> str:
        if not self.missing_keys and (not self.unexpected_keys):
            return "<All keys matched successfully>"
        return f"(missing_keys = {self.missing_keys}, unexpected_keys = {self.unexpected_keys})"


def _check_incompatible_keys(
    module: torch.nn.Module, missing_keys: List[str], unexpected_keys: List[str], strict: bool
) -> None:
    error_msgs = []
    if missing_keys:
        error_msgs.append("Missing key(s) in state_dict: {}".format(", ".join((f'"{k}"' for k in missing_keys))))
    if unexpected_keys:
        error_msgs.append("Unexpected key(s) in state_dict: {}".format(", ".join((f'"{k}"' for k in unexpected_keys))))
    if error_msgs and strict:
        raise RuntimeError("Error(s) in loading state_dict for {}:\n\t{}".format(module.__class__.__name__, "\n\t".join(error_msgs)))


def load_state_dict_distributed(
    module: torch.nn.Module, state_dict: Optional[StateDictType], strict: bool = True, prefix: str = ""
) -> _IncompatibleKeys:
    """
    Load a state_dict to the module within a distributed process.
    """
    if not is_distributed():
        missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=strict)
        return _IncompatibleKeys(missing_keys, unexpected_keys)  # type: ignore
    if is_global_primary():
        assert state_dict is not None
    else:
        assert state_dict is None
    missing_keys: List[str] = []
    unexpected_keys: List[str] = []
    submodules: Dict[str, torch.nn.Module] = dict(module.named_children())

    def update_key_list(original: List[str], updates: List[str]) -> None:
        for key in updates:
            if key not in original:
                original.append(key)

    from allennlp.nn.parallel.sharded_module_mixin import ShardedModuleMixin

    if isinstance(module, ShardedModuleMixin) or not submodules:
        collected_state_dict, _missing_keys, _unexpected_keys = _collect_state_dict(module, state_dict, prefix=prefix)
        assert collected_state_dict is not None
        update_key_list(missing_keys, _missing_keys)
        update_key_list(unexpected_keys, _unexpected_keys)
        _missing_keys2, _unexpected_keys2 = module.load_state_dict(collected_state_dict, strict=False)  # type: ignore
        update_key_list(missing_keys, _missing_keys2)
        update_key_list(unexpected_keys, _unexpected_keys2)
    else:
        direct_member_state_dict, _missing_keys, _unexpected_keys = _collect_state_dict(module, state_dict, recurse=False, prefix=prefix)
        _unexpected_keys = [k for k in _unexpected_keys if "." not in k or k.split(".")[0] not in submodules.keys()]
        update_key_list(missing_keys, _missing_keys)
        update_key_list(unexpected_keys, _unexpected_keys)
        _missing_keys2, _unexpected_keys2 = module.load_state_dict(direct_member_state_dict, strict=False)  # type: ignore
        update_key_list(missing_keys, _missing_keys2)
        update_key_list(unexpected_keys, _unexpected_keys2)
        for name, submodule in submodules.items():
            missing_keys = [k for k in missing_keys if not k.startswith(name + ".")]
            submodule_state_dict: Optional[StateDictType] = None
            if is_global_primary():
                assert state_dict is not None
                submodule_state_dict = {key.replace(name + ".", "", 1): value for key, value in state_dict.items() if key.startswith(name + ".")}
            _missing_keys3, _unexpected_keys3 = load_state_dict_distributed(submodule, submodule_state_dict, strict=False, prefix=prefix + name + ".")
            update_key_list(missing_keys, [f"{name}.{key}" for key in _missing_keys3])
            update_key_list(unexpected_keys, [f"{name}.{key}" for key in _unexpected_keys3])
    _check_incompatible_keys(module, missing_keys, unexpected_keys, strict)
    return _IncompatibleKeys(missing_keys, unexpected_keys)