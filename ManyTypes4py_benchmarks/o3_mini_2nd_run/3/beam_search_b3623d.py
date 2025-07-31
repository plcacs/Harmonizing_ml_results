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
StepFunctionTypeWithTimestep = Callable[[torch.Tensor, StateType, int], Tuple[torch.Tensor, StateType]]
StepFunctionTypeNoTimestep = Callable[[torch.Tensor, StateType], Tuple[torch.Tensor, StateType]]
StepFunctionType = TypeVar('StepFunctionType', StepFunctionTypeWithTimestep, StepFunctionTypeNoTimestep)
ConstraintStateType = List[List[Dict[str, Any]]]


class Sampler(Registrable):
    """
    An abstract class that can be used to sample candidates (either nodes or beams)
    within `BeamSearch`.
    """
    default_implementation = 'deterministic'

    def init_state(self, start_class_log_probabilities: torch.Tensor, batch_size: int, num_classes: int) -> StateType:
        return {}

    def sample_nodes(self, log_probs: torch.Tensor, per_node_beam_size: int, state: StateType) -> Tuple[torch.Tensor, torch.Tensor, StateType]:
        raise NotImplementedError

    def sample_beams(self, log_probs: torch.Tensor, beam_size: int, state: StateType) -> Tuple[torch.Tensor, torch.Tensor, StateType]:
        selected_log_probs, selected_indices = torch.topk(log_probs, beam_size, dim=-1)
        return (selected_log_probs, selected_indices, {})


@Sampler.register('deterministic')
class DeterministicSampler(Sampler):
    """
    A `Sampler` that just deterministically returns the `k` nodes or beams with highest
    log probability.
    """
    def sample_nodes(self, log_probs: torch.Tensor, per_node_beam_size: int, state: StateType) -> Tuple[torch.Tensor, torch.Tensor, StateType]:
        selected_log_probs, selected_indices = torch.topk(log_probs, per_node_beam_size, dim=-1)
        return (selected_log_probs, selected_indices, {})


@Sampler.register('multinomial')
class MultinomialSampler(Sampler):
    """
    A `Sampler` which samples nodes from the given multinomial distribution. Beams are sampled
    in the default, non-deterministic way.
    """
    def __init__(self, temperature: float = 1.0, with_replacement: bool = False) -> None:
        self.temperature = temperature
        self.with_replacement = with_replacement

    def sample_nodes(self, log_probs: torch.Tensor, per_node_beam_size: int, state: StateType) -> Tuple[torch.Tensor, torch.Tensor, StateType]:
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
    """
    def __init__(self, k: int = 1, temperature: float = 1.0, with_replacement: bool = False) -> None:
        self.k = k
        self.temperature = temperature or 1.0
        self.with_replacement = with_replacement

    def sample_nodes(self, log_probs: torch.Tensor, per_node_beam_size: int, state: StateType) -> Tuple[torch.Tensor, torch.Tensor, StateType]:
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
    """
    def __init__(self, p: float = 0.9, temperature: float = 1.0, with_replacement: bool = False) -> None:
        if p < 0.0 or p > 1.0:
            raise ValueError('p must be a positive float no greater than 1.0')
        self.p = p
        self.temperature = temperature or 1.0
        self.with_replacement = with_replacement

    def sample_nodes(self, log_probs: torch.Tensor, per_node_beam_size: int, state: StateType) -> Tuple[torch.Tensor, torch.Tensor, StateType]:
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
    A `Sampler` which uses the Gumbel-Top-K trick to sample without replacement.
    """
    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def init_state(self, start_class_log_probabilities: torch.Tensor, batch_size: int, num_classes: int) -> StateType:
        zeros = start_class_log_probabilities.new_zeros((batch_size, num_classes))
        G_phi_S = self.gumbel_with_max(start_class_log_probabilities, zeros)
        return {'G_phi_S': G_phi_S}

    def sample_nodes(self, log_probs: torch.Tensor, per_node_beam_size: int, state: StateType) -> Tuple[torch.Tensor, torch.Tensor, StateType]:
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

    def sample_beams(self, log_probs: torch.Tensor, beam_size: int, state: StateType) -> Tuple[torch.Tensor, torch.Tensor, StateType]:
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
        """
        return -torch.log(-torch.log(torch.rand_like(phi))) + phi

    def gumbel_with_max(self, phi: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Sample `Gumbel(phi)` conditioned on the maximum value being equal to `T`.
        """
        G_phi = self.gumbel(phi)
        Z, _ = G_phi.max(dim=-1)
        v = T - G_phi + torch.log1p(-torch.exp(G_phi - Z.unsqueeze(-1)))
        return T - torch.nn.functional.relu(v) - torch.log1p(torch.exp(-v.abs()))


class FinalSequenceScorer(Registrable):
    """
    An abstract class that can be used to score the final generated sequences found
    by beam search.
    """
    default_implementation = 'sequence-log-prob'

    def score(self, predictions: torch.Tensor, log_probabilities: torch.Tensor, end_index: int) -> torch.Tensor:
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
    tokens in the sequence.
    """
    def __init__(self, length_penalty: float = 1.0) -> None:
        super().__init__()
        self.length_penalty = length_penalty

    def score(self, predictions: torch.Tensor, log_probabilities: torch.Tensor, end_index: int) -> torch.Tensor:
        lengths = (predictions != end_index).long().sum(dim=2)
        is_end_token = predictions[:, :, -1] == end_index
        lengths += is_end_token.long()
        average_log_probs = log_probabilities / lengths ** self.length_penalty
        return average_log_probs


class Constraint(Registrable):
    """
    An abstract class that can be used to enforce constraints on the output predictions
    by manipulating the class log probabilities during beam search.
    """
    def __init__(self, vocab: Optional[Vocabulary] = None) -> None:
        self.vocab = vocab

    def init_state(self, batch_size: int) -> ConstraintStateType:
        raise NotImplementedError

    def apply(self, state: ConstraintStateType, class_log_probabilities: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _copy_state(state: ConstraintStateType, batch_size: int, beam_size: int, last_backpointer: Optional[torch.Tensor] = None) -> ConstraintStateType:
        new_state: ConstraintStateType = []
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

    def update_state(self, state: ConstraintStateType, last_prediction: torch.Tensor, last_backpointer: Optional[torch.Tensor] = None) -> ConstraintStateType:
        batch_size, beam_size = last_prediction.size()
        new_state = self._copy_state(state, batch_size, beam_size, last_backpointer)
        return self._update_state(new_state, last_prediction)

    def _update_state(self, state: ConstraintStateType, last_prediction: torch.Tensor) -> ConstraintStateType:
        raise NotImplementedError


@Constraint.register('repeated-ngram-blocking')
class RepeatedNGramBlockingConstraint(Constraint):
    def __init__(self, ngram_size: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.ngram_size = ngram_size

    def init_state(self, batch_size: int) -> ConstraintStateType:
        return [[{'seen_ngrams': {}, 'current_prefix': []}] for _ in range(batch_size)]

    def apply(self, state: ConstraintStateType, class_log_probabilities: torch.Tensor) -> torch.Tensor:
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

    def _update_state(self, state: ConstraintStateType, last_prediction: torch.Tensor) -> ConstraintStateType:
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
    """
    Implements the beam search algorithm for decoding the most likely sequences.
    """
    default_implementation = 'beam_search'

    def __init__(self,
                 end_index: int,
                 max_steps: int = 50,
                 beam_size: int = 10,
                 per_node_beam_size: Optional[int] = None,
                 sampler: Optional[Sampler] = None,
                 min_steps: Optional[int] = None,
                 final_sequence_scorer: Optional[FinalSequenceScorer] = None,
                 constraints: Optional[List[Lazy[Constraint]]] = None,
                 vocab: Optional[Vocabulary] = None) -> None:
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
    def _reconstruct_sequences(predictions: List[torch.Tensor], backpointers: List[torch.Tensor]) -> List[torch.Tensor]:
        reconstructed_predictions: List[torch.Tensor] = [predictions[-1].unsqueeze(2)]
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
    def search(self, start_predictions: torch.Tensor, start_state: StateType, step: Callable[..., Tuple[torch.Tensor, StateType]]) -> Tuple[torch.Tensor, torch.Tensor]:
        step_signature = signature(step)
        if len(step_signature.parameters) < 3:
            old_step = cast(StepFunctionTypeNoTimestep, step)

            def new_step(last_predictions: torch.Tensor, state: StateType, time_step: int) -> Tuple[torch.Tensor, StateType]:
                return old_step(last_predictions, state)
            return self._search(start_predictions, start_state, new_step)
        else:
            return self._search(start_predictions, start_state, cast(StepFunctionTypeWithTimestep, step))

    def _search(self, start_predictions: torch.Tensor, start_state: StateType, step: StepFunctionTypeWithTimestep) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = start_predictions.size()[0]
        predictions: List[torch.Tensor] = []
        backpointers: List[torch.Tensor] = []
        constraint_states: List[ConstraintStateType] = [constraint.init_state(batch_size) for constraint in self.constraints]
        start_class_log_probabilities, state = step(start_predictions, start_state, 0)
        num_classes = start_class_log_probabilities.size()[1]
        if self.per_node_beam_size > num_classes:
            raise ConfigurationError(f'Target vocab size ({num_classes:d}) too small relative to per_node_beam_size ({self.per_node_beam_size:d}).\nPlease decrease beam_size or per_node_beam_size.')
        sampler_state = self.sampler.init_state(start_class_log_probabilities, batch_size, num_classes)
        if self.constraints:
            expanded_start_class_log_probabilities = start_class_log_probabilities.unsqueeze(1)
            for constraint, constraint_state in zip(self.constraints, constraint_states):
                expanded_start_class_log_probabilities = constraint.apply(constraint_state, expanded_start_class_log_probabilities)
            start_class_log_probabilities = expanded_start_class_log_probabilities.squeeze(1)
        if self.min_steps >= 1:
            start_class_log_probabilities[:, self._end_index] = min_value_of_dtype(start_class_log_probabilities.dtype)
        start_top_log_probabilities, start_predicted_classes, sampler_state = self.sampler.sample_beams(start_class_log_probabilities, self.beam_size, sampler_state)
        if self.beam_size == 1 and (start_predicted_classes == self._end_index).all():
            warnings.warn('Empty sequences predicted. You may want to increase the beam size or ensure your step function is working properly.', RuntimeWarning)
            return (start_predicted_classes.unsqueeze(-1), start_top_log_probabilities)
        last_log_probabilities = start_top_log_probabilities
        predictions.append(start_predicted_classes)
        log_probs_after_end = start_class_log_probabilities.new_full((batch_size * self.beam_size, num_classes), min_value_of_dtype(start_class_log_probabilities.dtype))
        log_probs_after_end[:, self._end_index] = 0.0
        self._update_initial_state(state, batch_size)
        for i, constraint in enumerate(self.constraints):
            constraint_states[i] = constraint.update_state(constraint_states[i], start_predicted_classes)
        for timestep in range(self.max_steps - 1):
            last_predictions = predictions[-1].reshape(batch_size * self.beam_size)
            if (last_predictions == self._end_index).all():
                break
            class_log_probabilities, state = step(last_predictions, state, timestep + 1)
            if self.constraints:
                reshaped_class_log_probabilities = class_log_probabilities.view(batch_size, self.beam_size, -1)
                for constraint, constraint_state in zip(self.constraints, constraint_states):
                    reshaped_class_log_probabilities = constraint.apply(constraint_state, reshaped_class_log_probabilities)
                class_log_probabilities = reshaped_class_log_probabilities.view(batch_size * self.beam_size, -1)
            if timestep + 2 <= self.min_steps:
                class_log_probabilities[:, self._end_index] = min_value_of_dtype(class_log_probabilities.dtype)
            last_predictions_expanded = last_predictions.unsqueeze(-1).expand(batch_size * self.beam_size, num_classes)
            cleaned_log_probabilities = torch.where(last_predictions_expanded == self._end_index, log_probs_after_end, class_log_probabilities)
            top_log_probabilities, predicted_classes, sampler_state = self.sampler.sample_nodes(cleaned_log_probabilities, self.per_node_beam_size, sampler_state)
            expanded_last_log_probabilities = last_log_probabilities.unsqueeze(2).expand(batch_size, self.beam_size, self.per_node_beam_size).reshape(batch_size * self.beam_size, self.per_node_beam_size)
            summed_top_log_probabilities = top_log_probabilities + expanded_last_log_probabilities
            reshaped_summed = summed_top_log_probabilities.reshape(batch_size, self.beam_size * self.per_node_beam_size)
            reshaped_predicted_classes = predicted_classes.reshape(batch_size, self.beam_size * self.per_node_beam_size)
            restricted_beam_log_probs, restricted_beam_indices, sampler_state = self.sampler.sample_beams(reshaped_summed, self.beam_size, sampler_state)
            restricted_predicted_classes = reshaped_predicted_classes.gather(1, restricted_beam_indices)
            predictions.append(restricted_predicted_classes)
            last_log_probabilities = restricted_beam_log_probs
            backpointer = torch.divide(restricted_beam_indices, self.per_node_beam_size, rounding_mode='trunc')
            backpointers.append(backpointer)
            self._update_state(state, backpointer)
            for i, constraint in enumerate(self.constraints):
                constraint_states[i] = constraint.update_state(constraint_states[i], restricted_predicted_classes, last_backpointer=backpointer)
        if not self.constraints and (not torch.isfinite(last_log_probabilities).all() or (last_log_probabilities == min_value_of_dtype(last_log_probabilities.dtype)).any()):
            warnings.warn("Negligible log probabilities encountered ('-inf' or equivalent). Some final sequences may not make sense. This can happen when the beam size is larger than the number of valid (non-zero probability) transitions that the step function produces.", RuntimeWarning)
        reconstructed_predictions = self._reconstruct_sequences(predictions, backpointers)
        all_predictions = torch.cat(list(reversed(reconstructed_predictions)), 2)
        final_scores = self.final_sequence_scorer.score(all_predictions, last_log_probabilities, self._end_index)
        sorted_final_scores, sorted_indices = torch.sort(final_scores, dim=1, descending=True)
        sorted_all_predictions = torch.gather(all_predictions, 1, sorted_indices.unsqueeze(-1).expand_as(all_predictions))
        return (sorted_all_predictions, sorted_final_scores)

    @staticmethod
    def _is_multilayer_rnn_decoder(key: str, state_tensor: torch.Tensor) -> bool:
        return state_tensor.dim() == 3 and key in {'decoder_hidden', 'decoder_context'}

    def _update_initial_state(self, state: StateType, batch_size: int) -> None:
        for key, state_tensor in state.items():
            if state_tensor is None:
                continue
            multilayer_rnn_decoder = self._is_multilayer_rnn_decoder(key, state_tensor)
            if multilayer_rnn_decoder:
                num_layers, _, *last_dims = state_tensor.size()
                state[key] = state_tensor.unsqueeze(2).expand(num_layers, batch_size, self.beam_size, *last_dims).reshape(num_layers, batch_size * self.beam_size, *last_dims)
            else:
                _, *last_dims = state_tensor.size()
                state[key] = state_tensor.unsqueeze(1).expand(batch_size, self.beam_size, *last_dims).reshape(batch_size * self.beam_size, *last_dims)

    def _update_state(self, state: StateType, backpointer: torch.Tensor) -> None:
        batch_size = backpointer.size()[0]
        for key, state_tensor in state.items():
            if state_tensor is None:
                continue
            multilayer_rnn_decoder = self._is_multilayer_rnn_decoder(key, state_tensor)
            if multilayer_rnn_decoder:
                num_layers, _, *last_dims = state_tensor.size()
                expanded_backpointer = backpointer.view(batch_size, self.beam_size, *[1] * len(last_dims)).expand(batch_size, self.beam_size, *last_dims)
                expanded_backpointer = expanded_backpointer.unsqueeze(0).repeat(num_layers, 1, 1, 1)
                state[key] = state_tensor.reshape(num_layers, batch_size, self.beam_size, *last_dims).gather(2, expanded_backpointer).reshape(num_layers, batch_size * self.beam_size, *last_dims)
            else:
                _, *last_dims = state_tensor.size()
                expanded_backpointer = backpointer.view(batch_size, self.beam_size, *[1] * len(last_dims)).expand(batch_size, self.beam_size, *last_dims)
                state[key] = state_tensor.reshape(batch_size, self.beam_size, *last_dims).gather(1, expanded_backpointer).reshape(batch_size * self.beam_size, *last_dims)


BeamSearch.register('beam_search')(BeamSearch)