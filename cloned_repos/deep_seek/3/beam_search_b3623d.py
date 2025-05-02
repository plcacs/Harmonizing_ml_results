from inspect import signature
from typing import Any, List, Callable, Tuple, Dict, cast, TypeVar, Optional, Union
import copy
import warnings
import torch
from allennlp.common import Lazy, Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.nn.util import min_value_of_dtype
from torch import Tensor

StateType = Dict[str, torch.Tensor]
StepFunctionTypeWithTimestep = Callable[[torch.Tensor, StateType, int], Tuple[torch.Tensor, StateType]]
StepFunctionTypeNoTimestep = Callable[[torch.Tensor, StateType], Tuple[torch.Tensor, StateType]]
StepFunctionType = TypeVar('StepFunctionType', StepFunctionTypeWithTimestep, StepFunctionTypeNoTimestep)
ConstraintStateType = List[List[Dict[str, Any]]]

class Sampler(Registrable):
    default_implementation = 'deterministic'

    def init_state(self, start_class_log_probabilities: Tensor, batch_size: int, num_classes: int) -> Dict[str, Any]:
        return {}

    def sample_nodes(self, log_probs: Tensor, per_node_beam_size: int, state: Dict[str, Any]) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        raise NotImplementedError

    def sample_beams(self, log_probs: Tensor, beam_size: int, state: Dict[str, Any]) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        selected_log_probs, selected_indices = torch.topk(log_probs, beam_size, dim=-1)
        return (selected_log_probs, selected_indices, {})

@Sampler.register('deterministic')
class DeterministicSampler(Sampler):
    def sample_nodes(self, log_probs: Tensor, per_node_beam_size: int, state: Dict[str, Any]) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        selected_log_probs, selected_indices = torch.topk(log_probs, per_node_beam_size, dim=-1)
        return (selected_log_probs, selected_indices, {})

@Sampler.register('multinomial')
class MultinomialSampler(Sampler):
    def __init__(self, temperature: float = 1.0, with_replacement: bool = False) -> None:
        self.temperature = temperature
        self.with_replacement = with_replacement

    def sample_nodes(self, log_probs: Tensor, per_node_beam_size: int, state: Dict[str, Any]) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        if self.temperature != 1.0:
            _probabilities = torch.nn.functional.softmax(log_probs / self.temperature, dim=-1)
        else:
            _probabilities = log_probs.exp()
        selected_indices = torch.multinomial(_probabilities, per_node_beam_size, replacement=self.with_replacement)
        return (torch.gather(log_probs, 1, selected_indices), selected_indices, state)

@Sampler.register('top-k')
class TopKSampler(Sampler):
    def __init__(self, k: int = 1, temperature: float = 1.0, with_replacement: bool = False) -> None:
        self.k = k
        self.temperature = temperature or 1.0
        self.with_replacement = with_replacement

    def sample_nodes(self, log_probs: Tensor, per_node_beam_size: int, state: Dict[str, Any]) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
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
    def __init__(self, p: float = 0.9, temperature: float = 1.0, with_replacement: bool = False) -> None:
        if p < 0.0 or p > 1.0:
            raise ValueError('p must be a positive float no greater than 1.0')
        self.p = p
        self.temperature = temperature or 1.0
        self.with_replacement = with_replacement

    def sample_nodes(self, log_probs: Tensor, per_node_beam_size: int, state: Dict[str, Any]) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
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
    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def init_state(self, start_class_log_probabilities: Tensor, batch_size: int, num_classes: int) -> Dict[str, Any]:
        zeros = start_class_log_probabilities.new_zeros((batch_size, num_classes))
        G_phi_S = self.gumbel_with_max(start_class_log_probabilities, zeros)
        return {'G_phi_S': G_phi_S}

    def sample_nodes(self, log_probs: Tensor, per_node_beam_size: int, state: Dict[str, Any]) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
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

    def sample_beams(self, log_probs: Tensor, beam_size: int, state: Dict[str, Any]) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
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

    def gumbel(self, phi: Tensor) -> Tensor:
        return -torch.log(-torch.log(torch.rand_like(phi))) + phi

    def gumbel_with_max(self, phi: Tensor, T: Tensor) -> Tensor:
        G_phi = self.gumbel(phi)
        Z, _ = G_phi.max(dim=-1)
        v = T - G_phi + torch.log1p(-torch.exp(G_phi - Z.unsqueeze(-1)))
        return T - torch.nn.functional.relu(v) - torch.log1p(torch.exp(-v.abs()))

class FinalSequenceScorer(Registrable):
    default_implementation = 'sequence-log-prob'

    def score(self, predictions: Tensor, log_probabilities: Tensor, end_index: int) -> Tensor:
        raise NotImplementedError

@FinalSequenceScorer.register('sequence-log-prob')
class SequenceLogProbabilityScorer(FinalSequenceScorer):
    def score(self, predictions: Tensor, log_probabilities: Tensor, end_index: int) -> Tensor:
        return log_probabilities

@FinalSequenceScorer.register('length-normalized-sequence-log-prob')
class LengthNormalizedSequenceLogProbabilityScorer(FinalSequenceScorer):
    def __init__(self, length_penalty: float = 1.0) -> None:
        super().__init__()
        self.length_penalty = length_penalty

    def score(self, predictions: Tensor, log_probabilities: Tensor, end_index: int) -> Tensor:
        lengths = (predictions != end_index).long().sum(dim=2)
        is_end_token = predictions[:, :, -1] == end_index
        lengths += is_end_token.long()
        average_log_probs = log_probabilities / lengths ** self.length_penalty
        return average_log_probs

class Constraint(Registrable):
    def __init__(self, vocab: Optional[Vocabulary] = None) -> None:
        self.vocab = vocab

    def init_state(self, batch_size: int) -> ConstraintStateType:
        raise NotImplementedError

    def apply(self, state: ConstraintStateType, class_log_probabilities: Tensor) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def _copy_state(state: ConstraintStateType, batch_size: int, beam_size: int, last_backpointer: Optional[Tensor] = None) -> ConstraintStateType:
        new_state = []
        for i in range(batch_size):
            batch_state = []
            for j in range(beam_size):
                if last_backpointer is None:
                    backpointer = 0
                else:
                    backpointer = last_backpointer[i, j].item()
                batch_state.append(copy.deepcopy(state[i][backpointer]))
            new_state.append(batch_state)
        return new_state

    def update_state(self, state: ConstraintStateType, last_prediction: Tensor, last_backpointer: Optional[Tensor] = None) -> ConstraintStateType:
        batch_size, beam_size = last_prediction.size()
        new_state = self._copy_state(state, batch_size, beam_size, last_backpointer)
        return self._update_state(new_state, last_prediction)

    def _update_state(self, state: ConstraintStateType, last_prediction: Tensor) -> ConstraintStateType:
        raise NotImplementedError

@Constraint.register('repeated-ngram-blocking')
class RepeatedNGramBlockingConstraint(Constraint):
    def __init__(self, ngram_size: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.ngram_size = ngram_size

    def init_state(self, batch_size: int) -> ConstraintStateType:
        return [[{'seen_ngrams': {}, 'current_prefix': []}] for _ in range(batch_size)]

    def apply(self, state: ConstraintStateType, class_log_probabilities: Tensor) -> Tensor:
        for i, batch in enumerate(state):
            for j, beam in enumerate(batch):
                current_prefix = tuple(beam['current_prefix'])
                seen_ngrams = beam['seen_ngrams']
                try:
                    disallowed_indices = seen_ngrams[current_prefix]
                    class_log_probabilities[i, j, disallowed_indices] = min_value_of_dtype(class_log_probabilities.dtype)
                except KeyError:
                    pass
        return class_log_probabilities

    def _update_state(self, state: ConstraintStateType, last_prediction: Tensor) -> ConstraintStateType:
        for i, batch in enumerate(state):
            for j, beam in enumerate(batch):
                prediction = last_prediction[i, j].item()
                prefix = beam['current_prefix']
                seen_ngrams = beam['seen_ngrams']
                if len(prefix) == self.ngram_size - 1:
                    if tuple(prefix) not in seen_ngrams:
                        seen_ngrams[tuple(prefix)] = []
                    seen_ngrams[tuple(prefix)].append(prediction)
                prefix.append(prediction)
                if len(prefix) == self.ngram_size:
                    prefix.pop(0)
        return state

class BeamSearch(Registrable):
    default_implementation = 'beam_search'

    def __init__(
        self,
        end_index: int,
        max_steps: int = 50,
        beam_size: int = 10,
        per_node_beam_size: Optional[int] = None,
        sampler: Optional[Sampler] = None,
        min_steps: Optional[int] = None,
        final_sequence_scorer: Optional[FinalSequenceScorer] = None,
        constraints: Optional[List[Lazy[Constraint]]] = None,
        vocab: Optional[Vocabulary] = None
    ) -> None:
        if not max_steps > 0:
            raise ValueError('max_steps must be positive')
        if not beam_size > 0:
            raise ValueError('beam_size must be positive')
        if per_node_beam_size is not None and (not per_node_beam_size > 0):
            raise ValueError('per_node_beam_size must be positive')
        if min_steps is not None:
            if not min_steps >= 0:
                raise ValueError('min_steps must be non-negative')
            if not min_steps <= max_steps:
                raise ValueError('min_steps must be less than or equal to max_steps')
        self._end_index = end_index
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size or beam_size
        self.sampler = sampler or DeterministicSampler()
        self.min_steps = min_steps or 0
        self.final_sequence_scorer = final_sequence_scorer or SequenceLogProbabilityScorer()
        self.constraints = [constraint.construct(vocab=vocab) for constraint in constraints or []]

    @staticmethod
    def _reconstruct_sequences(predictions: List[Tensor], backpointers: List[Tensor]) -> List[Tensor]:
        reconstructed_predictions = [predictions[-1].unsqueeze(2)]
        if not backpointers:
            return reconstructed_predictions
        cur_backpointers = backpointers[-1]
        for timestep in range(len(predictions) - 2, 0, -1):
            cur_preds = predictions[timestep].gather(1, cur_backpointers).unsqueeze(2)
            reconstructed_predictions.append(cur_preds)
            cur_backpointers = backpointers[timestep - 1].gather(1, cur_backpointers)
        final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)
        reconstructed_predictions.append(final_preds)
        return reconstructed_predictions

    @torch.no_grad()
    def search(
        self,
        start_predictions: Tensor,
        start_state: StateType,
        step: Union[StepFunctionTypeWithTimestep, StepFunctionTypeNoTimestep]
    ) -> Tuple[Tensor, Tensor]:
        step_signature = signature(step)
        if len(step_signature.parameters) < 3:
            old_step = cast(StepFunctionTypeNoTimestep, step)

            def new_step(last_predictions: Tensor, state: StateType, time_step: int) -> Tuple[Tensor, StateType]:
                return old_step(last_predictions, state)
            return self._search(start_predictions, start_state, new_step)
        else:
            return self._search(start_predictions, start_state, cast(StepFunctionTypeWithTimestep, step))

    def _search(
        self,
        start_predictions: Tensor,
        start_state: StateType,
        step: StepFunctionTypeWithTimestep
    ) -> Tuple[Tensor, Tensor]:
        batch_size = start_predictions.size()[0]
        predictions: List[Tensor] = []
        backpointers: List[Tensor] = []
        constraint_states = [constraint.init_state(batch_size) for constraint in self.constraints]
        start_class_log_probabilities, state = step(start_predictions, start_state, 0)
        num_classes = start_class_log_probabilities.size()[1]
        if self.per_node_beam_size > num_classes:
