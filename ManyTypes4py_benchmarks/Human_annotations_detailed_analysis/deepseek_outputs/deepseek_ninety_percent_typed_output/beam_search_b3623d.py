from inspect import signature
from typing import Any, List, Callable, Tuple, Dict, cast, TypeVar, Optional
import copy
import warnings


import torch

from allennlp.common import Lazy, Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.nn.util import min_value_of_dtype


StateType = Dict[str, torch.Tensor]
StepFunctionTypeWithTimestep = Callable[
    [torch.Tensor, StateType, int], Tuple[torch.Tensor, StateType]
]
StepFunctionTypeNoTimestep = Callable[[torch.Tensor, StateType], Tuple[torch.Tensor, StateType]]

StepFunctionType = TypeVar(
    "StepFunctionType", StepFunctionTypeWithTimestep, StepFunctionTypeNoTimestep
)
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

    default_implementation = "deterministic"

    def init_state(
        self, start_class_log_probabilities: torch.Tensor, batch_size: int, num_classes: int
    ) -> StateType:
        return {}

    def sample_nodes(
        self, log_probs: torch.Tensor, per_node_beam_size: int, state: StateType
    ) -> Tuple[torch.Tensor, torch.Tensor, StateType]:
        raise NotImplementedError

    def sample_beams(
        self, log_probs: torch.Tensor, beam_size: int, state: StateType
    ) -> Tuple[torch.Tensor, torch.Tensor, StateType]:
        selected_log_probs, selected_indices = torch.topk(log_probs, beam_size, dim=-1)
        return selected_log_probs, selected_indices, {}


@Sampler.register("deterministic")
class DeterministicSampler(Sampler):
    """
    A `Sampler` that just deterministically returns the `k` nodes or beams with highest
    log probability.
    """

    def sample_nodes(
        self, log_probs: torch.Tensor, per_node_beam_size: int, state: StateType
    ) -> Tuple[torch.Tensor, torch.Tensor, StateType]:
        selected_log_probs, selected_indices = torch.topk(log_probs, per_node_beam_size, dim=-1)
        return selected_log_probs, selected_indices, {}


@Sampler.register("multinomial")
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

    def __init__(
        self,
        temperature: float = 1.0,
        with_replacement: bool = False,
    ) -> None:
        self.temperature = temperature
        self.with_replacement = with_replacement

    def sample_nodes(
        self, log_probs: torch.Tensor, per_node_beam_size: int, state: StateType
    ) -> Tuple[torch.Tensor, torch.Tensor, StateType]:
        if self.temperature != 1.0:
            _probabilities = torch.nn.functional.softmax(log_probs / self.temperature, dim=-1)
        else:
            _probabilities = log_probs.exp()

        selected_indices = torch.multinomial(
            _probabilities, per_node_beam_size, replacement=self.with_replacement
        )

        return torch.gather(log_probs, 1, selected_indices), selected_indices, state


@Sampler.register("top-k")
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

    def __init__(
        self,
        k: int = 1,
        temperature: float = 1.0,
        with_replacement: bool = False,
    ) -> None:
        self.k = k
        self.temperature = temperature or 1.0
        self.with_replacement = with_replacement

    def sample_nodes(
        self, log_probs: torch.Tensor, per_node_beam_size: int, state: StateType
    ) -> Tuple[torch.Tensor, torch.Tensor, StateType]:
        if not per_node_beam_size <= self.k <= log_probs.size()[1]:
            raise ValueError(
                "k must be a postive integer no less than per_node_beam_size and no greater than vocabulary size"
            )

        # shape (both): (batch_size, k)
        top_k_log_probs, top_k_indices = log_probs.topk(self.k, dim=-1)

        # Apply temperature if necessary.
        # shape: (batch_size, k)
        if self.temperature != 1.0:
            top_k_log_probs = top_k_log_probs / self.temperature

        # Re-normalize the subset.
        # shape: (batch_size, k)
        normalized_top_k_probs = torch.nn.functional.softmax(top_k_log_probs, dim=-1)

        # Sample from the re-normalized subset.
        # NOTE: These indices are not indices into `log_probs`, they are indices into `top_k_log_probs`.
        # shape: (batch_size, per_node_beam_size)
        sampled_indices = torch.multinomial(
            normalized_top_k_probs, per_node_beam_size, replacement=self.with_replacement
        )

        # Convert `sampled_indices` back to indices in the original `log_probs` tensor.
        # shape: (batch_size, per_node_beam_size)
        indices = top_k_indices.gather(-1, sampled_indices)

        return log_probs.gather(1, indices), indices, state


@Sampler.register("top-p")
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

    def __init__(
        self,
        p: float = 0.9,
        temperature: float = 1.0,
        with_replacement: bool = False,
    ) -> None:
        if p < 0.0 or p > 1.0:
            raise ValueError("p must be a positive float no greater than 1.0")
        self.p = p
        self.temperature = temperature or 1.0
        self.with_replacement = with_replacement

    def sample_nodes(
        self, log_probs: torch.Tensor, per_node_beam_size: int, state: StateType
    ) -> Tuple[torch.Tensor, torch.Tensor, StateType]:
        if not per_node_beam_size <= log_probs.size()[1]:
            raise ValueError("per_node_beam_size cannot be greater than vocabulary size")

        # First apply temperature coefficient:
        if self.temperature != 1.0:
            _log_probs = torch.nn.functional.log_softmax(log_probs / self.temperature, dim=-1)
        else:
            _log_probs = log_probs

        # Sort the probabilities in descending order to then find cumulative sum
        log_probs_descending, sorting_indices = torch.sort(_log_probs, descending=True)

        # shape: (batch_size, num_classes)
        probabilities_descending = log_probs_descending.exp()
        probabilities_summed = torch.cumsum(probabilities_descending, dim=-1)

        # Create a mask for filtering out probabilities that don't make the top `p`.
        # shape: (batch_size, num_classes)
        exclusion_mask = probabilities_summed >= self.p

        # We want to include the first index where probabilities_summed >= p, so we shift over one.
        exclusion_mask[..., 1:] = exclusion_mask[..., :-1].clone()
        exclusion_mask[..., 0] = False

        # Make sure there's at least `per_node_beam_size` options to be selected.
        if not self.with_replacement:
            exclusion_mask[..., :per_node_beam_size] = False

        log_probs_descending[exclusion_mask] = min_value_of_dtype(log_probs.dtype)

        # Now re-normalized the included log probs.
        # shape: (batch_size, num_classes)
        filtered_probabilities = torch.nn.functional.softmax(log_probs_descending, dim=-1)

        # Sample from the re-normalized subset.
        # NOTE: These indices are not indices into `log_probs`, they are indices into `log_probs_descending`.
        # shape: (batch_size, per_node_beam_size)
        sampled_indices = torch.multinomial(
            filtered_probabilities, per_node_beam_size, replacement=self.with_replacement
        )

        # Convert `sampled_indices` back to indices in the original `log_probs` tensor.
        # shape: (batch_size, per_node_beam_size)
        selected_indices = sorting_indices.gather(-1, sampled_indices)

        # Return (selected log probabilities, selected classes)
        # shape: (len(log_probs),1) , (len(log_probs), 1)
        return torch.gather(log_probs, 1, selected_indices), selected_indices, state


@Sampler.register("gumbel")
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

    def init_state(
        self, start_class_log_probabilities: torch.Tensor, batch_size: int, num_classes: int
    ) -> StateType:
        # shape: (batch_size, num_classes)
        zeros = start_class_log_probabilities.new_zeros((batch_size, num_classes))

        # shape: (batch_size, num_classes)
        G_phi_S = self.gumbel_with_max(start_class_log_probabilities, zeros)

        return {"G_phi_S": G_phi_S}

    def sample_nodes(
        self,
        log_probs: torch.Tensor,
        per_node_beam_size: int,
        state: StateType,
    ) -> Tuple[torch.Tensor, torch.Tensor, StateType]:
        # First apply temperature coefficient:
        # shape: (batch_size * beam_size, num_classes)
        if self.temperature != 1.0:
            _log_probs = torch.nn.functional.log_softmax(log_probs / self.temperature, dim=-1)
        else:
            _log_probs = log_probs

        # shape: (group_size,)
        phi_S = state["phi_S"]

        # shape: (group_size, num_classes)
        phi_S = phi_S.unsqueeze(-1).expand_as(_log_probs)

        # shape: (group_size, num_classes)
        phi_S_new = phi_S + _log_probs

        # shape: (group_size, 1)
        G_phi_S = state["G_phi_S"].unsqueeze(-1)

        # shape: (group_size, num_classes)
        G_phi_S_new = self.gumbel_with_max(phi_S_new, G_phi_S)

        # Replace NaNs with very negative number.
        # shape: (group_size, num_classes)
        #  G_phi_S_new[G_phi_S_new.isnan()] = min_value_of_dtype(G_phi_S_new.dtype)

        # shape (both): (group_size, per_node_beam_size)
        top_G_phi_S_new, top_indices = torch.topk(G_phi_S_new, per_node_beam_size, dim=-1)

        # shape: (group_size, per_node_beam_size)
        top_log_probs = log_probs.gather(1, top_indices)

        return top_log_probs, top_indices, {"G_phi_S": top_G_phi_S_new}

    def sample_beams(
        self,
        log_probs: torch.Tensor,
        beam_size: int,
        state: StateType,
    ) -> Tuple[torch.Tensor, torch.Tensor, StateType]:
        """
        Returns the beams with the highest perturbed log probabilities.
        """
        # shape (log_probs): (batch_size, beam_size * per_node_beam_size)

        batch_size = log_probs.size()[0]

        # shape: (batch_size * beam_size, per_node_beam_size)
        G_phi_S = state["G_phi_S"]

        # shape: (batch_size, beam_size * per_node_beam_size)
        G_phi_S = G_phi_S.reshape_as(log_probs)

        # shape (both): (batch_size, beam_size)
        G_phi_S_new, selected_indices = torch.topk(G_phi_S, beam_size, dim=-1)

        # shape: (batch_size, beam_size)
        selected_log_probs = log_probs.gather(1, selected_indices)

        # Now sort the selected beams by their true log prob.
        # shape (all): (batch_size, beam_size)
        selected_log_probs, sort_indices = selected_log_probs.sort(dim=-1, descending=True)
        selected_indices = selected_indices.gather(1, sort_indices)
        G_phi_S_new = G_phi_S_new.gather(1, sort_indices)

        # shape: (batch_size * beam_size,)
        G_phi_S_new = G_phi_S_new.reshape(batch_size * beam_size)

        # shape: (batch_size * beam_size,)
        phi_S = selected_log_probs.reshape(batch_size * beam_size)

        return selected_log_probs, selected_indices, {"G_phi_S": G_phi_S_new, "phi_S": phi_S}

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
