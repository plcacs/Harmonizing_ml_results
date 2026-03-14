from inspect import signature
from typing import Any, List, Callable, Tuple, Dict, cast, TypeVar, Optional, Union
import copy
import warnings
import torch
from allennlp.common import Lazy, Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.nn.util import min_value_of_dtype

StateType = Dict[str, torch.Tensor]
StepFunctionTypeWithTimestep = Callable[[torch.Tensor, StateType, int], Tuple[torch.Tensor, StateType]]
StepFunctionTypeNoTimestep = Callable[[torch.Tensor, StateType], Tuple[torch.Tensor, StateType]]
StepFunctionType = TypeVar('StepFunctionType', StepFunctionTypeWithTimestep, StepFunctionTypeNoTimestep)
"""
The type of step function that can be passed to [`BeamSearch.search`](#search).

This can either be [`StepFunctionTypeWithTimestep`](#stepfunctiontypewithtimestep)
or [`StepFunctionTypeNoTimestep`](#stepfunctiontypenotimestep).
"""
ConstraintStateType = List[List[Dict[str, Any]]]

class Sampler(Registrable):
    """
    An abstract class that can be used to sample candidates (either nodes or beams)
    within `BeamSearch`.

    A `Sampler` just has three methods, `init_state()`, `sample_nodes()` and `sample_beams()`.

    `init_state()` takes three arguments:

    - a tensor of starting log probs with shape `(batch_size,, num_classes)`,
    - the batch size, an int,
    - and the number of classes, also an int.

    It returns a state dictionary with any state tensors needed for subsequent
    calls to `sample_nodes()` and `sample_beams()`.

    By default this method just returns an empty dictionary.

    Both `sample_nodes()` and `sample_beams()` should take three arguments:

    - tensor of normalized log probabilities with shape `(batch_size, num_examples)`,
    - an integer representing the number of samples to take for each example in the batch,
    - and a state dictionary which could contain any tensors needed for the `Sampler` to keep
      track of state.

    For `sample_nodes()`, `num_examples = num_classes`, but for `sample_beams`,
    `num_examples = beam_size * per_node_beam_size`.

    The return value should be a tuple containing:

    - a tensor of log probabilities of the sampled examples with shape `(batch_size, num_samples)`,
    - a tensor of indices of the sampled examples with shape `(batch_size, num_samples)`,
    - and the updated state dictionary.

    A default implementation of `sample_beams` is provided, which just deterministically
    picks the `k` examples with highest log probability.
    """
    default_implementation = 'deterministic'

    def init_state(self, start_class_log_probabilities: torch.Tensor, batch_size: int, num_classes: int) -> Dict[str, torch.Tensor]:
        return {}

    def sample_nodes(self, log_probs: torch.Tensor, per_node_beam_size: int, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError

    def sample_beams(self, log_probs: torch.Tensor, beam_size: int, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        selected_log_probs, selected_indices = torch.topk(log_probs, beam_size, dim=-1)
        return (selected_log_probs, selected_indices, {})

@Sampler.register('deterministic')
class DeterministicSampler(Sampler):
    """
    A `Sampler` that just deterministically returns the `k` nodes or beams with highest
    log probability.
    """

    def sample_nodes(self, log_probs: torch.Tensor, per_node_beam_size: int, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        selected_log_probs, selected_indices = torch.topk(log_probs, per_node_beam_size, dim=-1)
        return (selected_log_probs, selected_indices, {})

@Sampler.register('multinomial')
class MultinomialSampler(Sampler):
    """
    A `Sampler` which samples nodes from the given multinomial distribution. Beams are sampled
    in the default, non-deterministic way.

    # Parameters

    temperature : `float`, optional (default = `1.0`)
        A `temperature` below 1.0 produces a sharper probability distribution and a `temperature` above 1.0
        produces a flatter probability distribution.
    with_replacement : `bool`, optional (default = `False`)
        Whether to sample with replacement.
    """

    def __init__(self, temperature: float = 1.0, with_replacement: bool = False) -> None:
        self.temperature = temperature
        self.with_replacement = with_replacement

    def sample_nodes(self, log_probs: torch.Tensor, per_node_beam_size: int, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if self.temperature != 1.0:
            _probabilities = torch.nn.functional.softmax(log_probs / self.temperature, dim=-1)
        else:
            _probabilities = log_probs.exp()
        selected_indices = torch.multinomial(_probabilities, per_node_beam_size, replacement=self.with_replacement)
        return (torch.gather(log_probs, 1, selected_indices), selected_indices, state)

@Sampler.register('top-k')
class TopKSampler(Sampler):
    """
    A `Sampler` which redistributes the probability mass function for nodes among the
    top `k` choices, then samples from that subset after re-normalizing the probabilities.

    Beams are sampled in the default, deterministic way.

    # Parameters

    k : `int`, optional (default = `1`)
        The number of top choices to be selected from.
    temperature : `float`, optional (default = `1.0`)
        A `temperature` below 1.0 produces a sharper probability distribution and a `temperature`
        above 1.0 produces a flatter probability distribution.
    with_replacement: `bool`, optional, (default = `False`)
        If set to `True`, samples will be selected with replacement from the top k choices.
    """

    def __init__(self, k: int = 1, temperature: float = 1.0, with_replacement: bool = False) -> None:
        self.k = k
        self.temperature = temperature or 1.0
        self.with_replacement = with_replacement

    def sample_nodes(self, log_probs: torch.Tensor, per_node_beam_size: int, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if not per_node_beam_size <= self.k <= log_probs.size()[1]:
            raise ValueError('k must be a postive integer no less than per_node_beam_size and no greater than vocabulary size')
        top_k_log_probs, top_k_indices = log_probs.topk(self.k, dim=-1)
        if self.temperature != 1.0:
            top_k_log_probs = top_k_log_probs / self.temperature
        normalized_top_k_probs = torch.nn.functional.softmax(top_k_log_probs, dim=-1)
        sampled_indices = torch.multinomial(normalized_top_k_probs, per_node_beam_size, replacement=self.with_replacement)
        indices = top_k_indices.gather(-1, sampled_indices)
        return (log_probs.gather(1, indices), indices, state)

@Sampler.register('top-p')
class TopPSampler(Sampler):
    """
    A `Sampler` which redistributes the probability mass function for nodes among
    the top choices with a cumulative probability of at least `p`, then samples from that subset
    after re-normalizing the probabilities.

    Beams are sampled in the default, deterministic way.

    # Parameters

    p : `float`, optional (default = `0.9`)
        The cumulative probability cutoff threshold. A higher value of `p` will result in more possible
        examples to sample from. If `with_replacement` is `False` and the number of possible samples is
        insufficient to sample without replacement from when calling `sample_nodes`, then the top
        `per_node_beam_size` examples will be chosen.
    temperature : `float`, optional (default = `1.0`)
        A `temperature` below 1.0 produces a sharper probability distribution and a `temperature`
        above 1.0 produces a flatter probability distribution.
    with_replacement : `bool`, optional, (default = `False`)
        If set to `True`, samples will be selected with replacement from the top choices.
    """

    def __init__(self, p: float = 0.9, temperature: float = 1.0, with_replacement: bool = False) -> None:
        if p < 0.0 or p > 1.0:
            raise ValueError('p must be a positive float no greater than 1.0')
        self.p = p
        self.temperature = temperature or 1.0
        self.with_replacement = with_replacement

    def sample_nodes(self, log_probs: torch.Tensor, per_node_beam_size: int, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if not per_node_beam_size <= log_probs.size()[1]:
            raise ValueError('per_node_beam_size cannot be greater than vocabulary size')
        if self.temperature != 1.0:
            _log_probs = torch.nn.functional.log_softmax(log_probs / self.temperature, dim=-1)
        else:
            _log_probs = log_probs
        log_probs_descending, sorting_indices = torch.sort(_log_probs, descending=True)
        probabilities_descending = log_probs_descending.exp()
        probabilities_summed = torch.cumsum(probabilities_descending, dim=-1)
        exclusion_mask = probabilities_summed >= self.p
        exclusion_mask[..., 1:] = exclusion_mask[..., :-1].clone()
        exclusion_mask[..., 0] = False
        if not self.with_replacement:
            exclusion_mask[..., :per_node_beam_size] = False
        log_probs_descending[exclusion_mask] = min_value_of_dtype(log_probs.dtype)
        filtered_probabilities = torch.nn.functional.softmax(log_probs_descending, dim=-1)
        sampled_indices = torch.multinomial(filtered_probabilities, per_node_beam_size, replacement=self.with_replacement)
        selected_indices = sorting_indices.gather(-1, sampled_indices)
        return (torch.gather(log_probs, 1, selected_indices), selected_indices, state)

@Sampler.register('gumbel')
class GumbelSampler(Sampler):
    """
    A `Sampler` which uses the Gumbel-Top-K trick to sample without replacement. See
    [*Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for Sampling
    Sequences Without Replacement*, W Kool, H Van Hoof and M Welling, 2010]
    (https://api.semanticscholar.org/CorpusID:76662039).

    # Parameters

    temperature : `float`, optional (default = `1.0`)
        A `temperature` below 1.0 produces a sharper probability distribution and a `temperature`
        above 1.0 produces a flatter probability distribution.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def init_state(self, start_class_log_probabilities: torch.Tensor, batch_size: int, num_classes: int) -> Dict[str, torch.Tensor]:
        zeros = start_class_log_probabilities.new_zeros((batch_size, num_classes))
        G_phi_S = self.gumbel_with_max(start_class_log_probabilities, zeros)
        return {'G_phi_S': G_phi_S}

    def sample_nodes(self, log_probs: torch.Tensor, per_node_beam_size: int, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if self.temperature != 1.0:
            _log_probs = torch.nn.functional.log_softmax(log_probs / self.temperature, dim=-1)
        else:
            _log_probs = log_probs
        phi_S = state['phi_S']
        phi_S = phi_S.unsqueeze(-1).expand_as(_log_probs)
        phi_S_new = phi_S + _log_probs
        G_phi_S = state['G_phi_S'].unsqueeze(-1)
        G_phi_S_new = self.gumbel_with_max(phi_S_new, G_phi_S)
        top_G_phi_S_new, top_indices = torch.topk(G_phi_S_new, per_node_beam_size, dim=-1)
        top_log_probs = log_probs.gather(1, top_indices)
        return (top_log_probs, top_indices, {'G_phi_S': top_G_phi_S_new})

    def sample_beams(self, log_probs: torch.Tensor, beam_size: int, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns the beams with the highest perturbed log probabilities.
        """
        batch_size = log_probs.size()[0]
        G_phi_S = state['G_phi_S']
        G_phi_S = G_phi_S.reshape_as(log_probs)
        G_phi_S_new, selected_indices = torch.topk(G_phi_S, beam_size, dim=-1)
        selected_log_probs = log_probs.gather(1, selected_indices)
        selected_log_probs, sort_indices = selected_log_probs.sort(dim=-1, descending=True)
        selected_indices = selected_indices.gather(1, sort_indices)
        G_phi_S_new = G_phi_S_new.gather(1, sort_indices)
        G_phi_S_new = G_phi_S_new.reshape(batch_size * beam_size)
        phi_S = selected_log_probs.reshape(batch_size * beam_size)
        return (selected_log_probs, selected_indices, {'G_phi_S': G_phi_S_new, 'phi_S': phi_S})

    def gumbel(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Sample `Gumbel(phi)`.

        `phi` should have shape `(batch_size, num_classes)`.
        """
        return -torch.log(-torch.log(torch.rand_like(phi))) + phi

    def gumbel_with_max(self, phi: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Sample `Gumbel(phi)` conditioned on the maximum value being equal to `T`.

        `phi` should have shape `(batch_size, num_classes)` and `T` should have
        shape `(batch_size, 1)`.
        """
        G_phi = self.gumbel(phi)
        Z, _ = G_phi.max(dim=-1)
        v = T - G_phi + torch.log1p(-torch.exp(G_phi - Z.unsqueeze(-1)))
        return T - torch.nn.functional.relu(v) - torch.log1p(torch.exp(-v.abs()))

class FinalSequenceScorer(Registrable):
    """
    An abstract class that can be used to score the final generated sequences found
    by beam search. Given the predicted sequences and the corresponding log probabilities of
    those sequences, the class calculates and returns the final score of the sequences.

    The default implementation scores the sequences using the sum of the log probabilities of
    the sequence, which is passed as input.
    """
    default_implementation = 'sequence-log-prob'

    def score(self, predictions: torch.Tensor, log_probabilities: torch.Tensor, end_index: int) -> torch.Tensor:
        """
        Score the final predictions found by beam search.

        # Parameters

        predictions : `torch.Tensor`
            A tensor containing the initial predictions with shape `(batch_size, beam_size, max_steps)`.

        log_probabilities : `torch.Tensor`
            A tensor containing the log probabilities of the sequence, defined as the sum
            of the log probabilities per token, with shape `(batch_size, beam_size)`.

        end_index : `int`
            The index of the end symbol.

        # Returns

        `torch.Tensor`
            A tensor of the final sequence scores of shape `(batch_size, beam_size)`.
        """
        raise NotImplementedError

@FinalSequenceScorer.register('sequence-log-prob')
class SequenceLogProbabilityScorer(FinalSequenceScorer):
    """
    A `FinalSequenceScorer` which scores the sequences by the sum of the log probabilities
    across the sequence's tokens.
    """

    def score(self, predictions: torch.Tensor, log_probabilities: torch.Tensor, end_index: int) -> torch.Tensor:
        return log_probabilities

@FinalSequenceScorer.register('length-normalized-sequence-log-prob')
class LengthNormalizedSequenceLogProbabilityScorer(FinalSequenceScorer):
    """
    A `FinalSequenceScorer` which scores the sequences by the average log probability of the
    tokens in the sequence. It optionally includes a length penalty which promotes
    or demotes sequences based on their lengths. The final score for a sequence will
    be `(sequence_log_probability) / (sequence_length ** length_penalty)`. The sequence length
    here includes the end token.

    # Parameters

    length_penalty : `float`, optional (default = `1.0`)
        The length penalty to use. A value of 1.0 means no length penalty is used.
        A value > 1.0 favors longer sequences, and < 1.0 favors shorter sequences.
    """

    def __init__(self, length_penalty: float = 1.0) -> None:
        super().__init__()
        self.length_penalty = length_penalty

    def score(self, predictions: torch.Tensor, log