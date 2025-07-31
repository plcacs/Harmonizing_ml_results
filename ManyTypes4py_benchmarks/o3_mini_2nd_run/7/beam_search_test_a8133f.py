#!/usr/bin/env python
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import pytest
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.nn.beam_search import (MultinomialSampler, BeamSearch, TopKSampler, TopPSampler, 
                                      GumbelSampler, LengthNormalizedSequenceLogProbabilityScorer, 
                                      RepeatedNGramBlockingConstraint, StepFunctionTypeWithTimestep, 
                                      StepFunctionTypeNoTimestep)
from allennlp.common.params import Params
from allennlp.nn.util import min_value_of_dtype

transition_probabilities: torch.Tensor = torch.tensor([[0.0, 0.4, 0.3, 0.2, 0.1, 0.0],
                                                       [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                                       [0.2, 0.1, 0.2, 0.2, 0.2, 0.1]])
short_sequence_transition_probabilities: torch.Tensor = torch.tensor([[0.0, 0.1, 0.0, 0.0, 0.0, 0.9],
                                                                       [0.0, 0.0, 0.1, 0.0, 0.0, 0.9],
                                                                       [0.0, 0.0, 0.0, 0.1, 0.0, 0.9],
                                                                       [0.0, 0.0, 0.0, 0.0, 0.1, 0.9],
                                                                       [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                                                       [0.2, 0.1, 0.2, 0.2, 0.2, 0.1]])
repeated_ngram_transition_probabilities_0: torch.Tensor = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                                                        [0.0, 0.0, 0.4, 0.6, 0.0, 1e-09],
                                                                        [0.0, 0.0, 0.0, 1.0, 0.0, 1e-09],
                                                                        [0.0, 1.0, 0.0, 0.0, 0.0, 1e-09],
                                                                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
repeated_ngram_transition_probabilities_1: torch.Tensor = torch.tensor([[0.0, 0.4, 0.3, 0.2, 0.1, 0.0],
                                                                        [0.0, 0.4, 0.3, 0.2, 0.1, 0.1],
                                                                        [0.0, 0.0, 0.4, 0.3, 0.2, 0.1],
                                                                        [0.0, 0.0, 0.3, 0.4, 0.2, 0.1],
                                                                        [0.0, 0.0, 0.2, 0.3, 0.4, 0.1],
                                                                        [0.2, 0.1, 0.2, 0.2, 0.2, 0.1]])
log_probabilities: torch.Tensor = torch.log(torch.tensor([[0.1, 0.3, 0.3, 0.3, 0.0, 0.0],
                                                          [0.0, 0.0, 0.4, 0.3, 0.2, 0.1]]))


def get_step_function(transition_matrix: torch.Tensor, with_timestep: bool = False
                      ) -> Union[StepFunctionTypeNoTimestep, StepFunctionTypeWithTimestep]:
    def _step_function(last_predictions: torch.Tensor, state: Any) -> Tuple[torch.Tensor, Any]:
        log_probs_list: List[torch.Tensor] = []
        for last_token in last_predictions:
            log_probs = torch.log(transition_matrix[last_token.item()])
            log_probs_list.append(log_probs)
        return torch.stack(log_probs_list), state

    if not with_timestep:
        return _step_function

    def _step_function_with_timestep(last_predictions: torch.Tensor, state: Any, timestep: int
                                    ) -> Tuple[torch.Tensor, Any]:
        return _step_function(last_predictions, state)
    return _step_function_with_timestep


take_step_no_timestep: StepFunctionTypeNoTimestep = get_step_function(transition_probabilities)
take_step_with_timestep: StepFunctionTypeWithTimestep = get_step_function(transition_probabilities, with_timestep=True)
take_short_sequence_step: StepFunctionTypeNoTimestep = get_step_function(short_sequence_transition_probabilities)


class BeamSearchTest(AllenNlpTestCase):
    def setup_method(self, method: Any = None) -> None:
        super().setup_method(method)
        self.end_index: int = transition_probabilities.size()[0] - 1
        self.beam_search: BeamSearch = BeamSearch(self.end_index, max_steps=10, beam_size=3)
        self.expected_top_k: np.ndarray = np.array([[1, 2, 3, 4, 5],
                                                     [2, 3, 4, 5, 5],
                                                     [3, 4, 5, 5, 5]])
        self.expected_log_probs: np.ndarray = np.log(np.array([0.4, 0.3, 0.2]))

    def _check_results(self,
                       batch_size: int = 5,
                       expected_top_k: Optional[np.ndarray] = None,
                       expected_log_probs: Optional[np.ndarray] = None,
                       beam_search: Optional[BeamSearch] = None,
                       state: Optional[Any] = None,
                       take_step: Union[StepFunctionTypeWithTimestep, StepFunctionTypeNoTimestep] = take_step_with_timestep
                       ) -> None:
        expected_top_k = expected_top_k if expected_top_k is not None else self.expected_top_k
        expected_log_probs = expected_log_probs if expected_log_probs is not None else self.expected_log_probs
        state = state or {}
        beam_search = beam_search or self.beam_search
        beam_size: int = beam_search.beam_size
        initial_predictions: torch.Tensor = torch.tensor([0] * batch_size)
        top_k, log_probs = beam_search.search(initial_predictions, state, take_step)
        assert list(top_k.size())[:-1] == [batch_size, beam_size]
        np.testing.assert_array_equal(top_k[0].numpy(), expected_top_k)
        assert list(log_probs.size()) == [batch_size, beam_size]
        np.testing.assert_allclose(log_probs[0].numpy(), expected_log_probs, rtol=1e-06)

    @pytest.mark.parametrize('step_function', [take_step_with_timestep, take_step_no_timestep])
    def test_search(self, step_function: Union[StepFunctionTypeWithTimestep, StepFunctionTypeNoTimestep]) -> None:
        self._check_results(take_step=step_function)

    def test_finished_state(self) -> None:
        state: Dict[str, torch.Tensor] = {}
        state['foo'] = torch.tensor([[1, 0, 1],
                                     [2, 0, 1],
                                     [0, 0, 1],
                                     [1, 1, 1],
                                     [0, 0, 0]])
        expected_finished_state: Dict[str, np.ndarray] = {}
        expected_finished_state['foo'] = np.array([[1, 0, 1],
                                                   [1, 0, 1],
                                                   [1, 0, 1],
                                                   [2, 0, 1],
                                                   [2, 0, 1],
                                                   [2, 0, 1],
                                                   [0, 0, 1],
                                                   [0, 0, 1],
                                                   [0, 0, 1],
                                                   [1, 1, 1],
                                                   [1, 1, 1],
                                                   [1, 1, 1],
                                                   [0, 0, 0],
                                                   [0, 0, 0],
                                                   [0, 0, 0]])
        self._check_results(state=state)
        for key, array in expected_finished_state.items():
            np.testing.assert_allclose(state[key].numpy(), array)

    def test_diff_shape_state(self) -> None:
        state: Dict[str, torch.Tensor] = {}
        state['decoder_hidden'] = torch.tensor([[1, 0, 1],
                                                  [2, 0, 1],
                                                  [0, 0, 1],
                                                  [1, 1, 1],
                                                  [0, 0, 0]])
        state['decoder_hidden'] = state['decoder_hidden'].unsqueeze(0).repeat(2, 1, 1)
        seq: List[List[List[int]]] = [[[1, 0, 1], [1, 0, 1], [1, 0, 1],
                                       [2, 0, 1], [2, 0, 1], [2, 0, 1],
                                       [0, 0, 1], [0, 0, 1], [0, 0, 1],
                                       [1, 1, 1], [1, 1, 1], [1, 1, 1],
                                       [0, 0, 0], [0, 0, 0], [0, 0, 0]]] * 2
        expected_finished_state: Dict[str, np.ndarray] = {}
        expected_finished_state['decoder_hidden'] = np.array(seq)
        self._check_results(state=state)
        for key, array in expected_finished_state.items():
            np.testing.assert_allclose(state[key].numpy(), array)

    def test_batch_size_of_one(self) -> None:
        self._check_results(batch_size=1)

    def test_greedy_search(self) -> None:
        beam_search: BeamSearch = BeamSearch(self.end_index, beam_size=1)
        expected_top_k: np.ndarray = np.array([[1, 2, 3, 4, 5]])
        expected_log_probs: np.ndarray = np.log(np.array([0.4]))
        self._check_results(expected_top_k=expected_top_k, expected_log_probs=expected_log_probs, beam_search=beam_search)

    def test_single_step(self) -> None:
        self.beam_search.max_steps = 1
        expected_top_k: np.ndarray = np.array([[1], [2], [3]])
        expected_log_probs: np.ndarray = np.log(np.array([0.4, 0.3, 0.2]))
        self._check_results(expected_top_k=expected_top_k, expected_log_probs=expected_log_probs)

    def test_early_stopping(self) -> None:
        beam_search: BeamSearch = BeamSearch(self.end_index, beam_size=3, max_steps=3)
        expected_top_k: np.ndarray = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        expected_log_probs: np.ndarray = np.log(np.array([0.4, 0.3, 0.2]))
        self._check_results(expected_top_k=expected_top_k, expected_log_probs=expected_log_probs, beam_search=beam_search)

    def test_take_short_sequence_step(self) -> None:
        self.beam_search.beam_size = 5
        expected_top_k: np.ndarray = np.array([[5, 5, 5, 5, 5],
                                               [1, 5, 5, 5, 5],
                                               [1, 2, 5, 5, 5],
                                               [1, 2, 3, 5, 5],
                                               [1, 2, 3, 4, 5]])
        expected_log_probs: np.ndarray = np.log(np.array([0.9, 0.09, 0.009, 0.0009, 0.0001]))
        self._check_results(expected_top_k=expected_top_k, expected_log_probs=expected_log_probs, take_step=take_short_sequence_step)

    def test_min_steps(self) -> None:
        self.beam_search.beam_size = 1
        self.beam_search.min_steps = 0
        expected_top_k: np.ndarray = np.array([[5]])
        expected_log_probs: np.ndarray = np.log(np.array([0.9]))
        with pytest.warns(RuntimeWarning, match='Empty sequences predicted'):
            self._check_results(expected_top_k=expected_top_k, expected_log_probs=expected_log_probs, take_step=take_short_sequence_step)
        self.beam_search.min_steps = 1
        expected_top_k = np.array([[1, 5]])
        expected_log_probs = np.log(np.array([0.09]))
        self._check_results(expected_top_k=expected_top_k, expected_log_probs=expected_log_probs, take_step=take_short_sequence_step)
        self.beam_search.min_steps = 2
        expected_top_k = np.array([[1, 2, 5]])
        expected_log_probs = np.log(np.array([0.009]))
        self._check_results(expected_top_k=expected_top_k, expected_log_probs=expected_log_probs, take_step=take_short_sequence_step)
        self.beam_search.beam_size = 3
        self.beam_search.min_steps = 2
        expected_top_k = np.array([[1, 2, 5, 5, 5],
                                   [1, 2, 3, 5, 5],
                                   [1, 2, 3, 4, 5]])
        expected_log_probs = np.log(np.array([0.009, 0.0009, 0.0001]))
        self._check_results(expected_top_k=expected_top_k, expected_log_probs=expected_log_probs, take_step=take_short_sequence_step)

    def test_different_per_node_beam_size(self) -> None:
        beam_search: BeamSearch = BeamSearch(self.end_index, beam_size=3, per_node_beam_size=1)
        self._check_results(beam_search=beam_search)
        beam_search = BeamSearch(self.end_index, beam_size=3, per_node_beam_size=2)
        self._check_results(beam_search=beam_search)

    def test_catch_bad_config(self) -> None:
        beam_search: BeamSearch = BeamSearch(self.end_index, beam_size=20)
        with pytest.raises(ConfigurationError):
            self._check_results(beam_search=beam_search)

    def test_warn_for_bad_log_probs(self) -> None:
        initial_predictions: torch.Tensor = torch.LongTensor([self.end_index - 1, self.end_index - 1])
        with pytest.warns(RuntimeWarning, match='Negligible log probabilities'):
            self.beam_search.search(initial_predictions, {}, take_step_no_timestep)

    def test_empty_sequences(self) -> None:
        initial_predictions: torch.Tensor = torch.LongTensor([self.end_index - 1, self.end_index - 1])
        beam_search: BeamSearch = BeamSearch(self.end_index, beam_size=1)
        with pytest.warns(RuntimeWarning, match='Empty sequences predicted'):
            predictions, log_probs = beam_search.search(initial_predictions, {}, take_step_with_timestep)
        assert list(predictions.size()) == [2, 1, 1]
        assert list(log_probs.size()) == [2, 1]
        assert (predictions == self.end_index).all()
        assert (log_probs == 0).all()

    def test_default_from_params_params(self) -> None:
        beam_search: BeamSearch = BeamSearch.from_params(Params({'beam_size': 2, 'end_index': 7}))
        assert beam_search.beam_size == 2
        assert beam_search._end_index == 7

    def test_top_p_search(self) -> None:
        initial_predictions: torch.Tensor = torch.tensor([0] * 5)
        beam_size: int = 3
        take_step: Union[StepFunctionTypeWithTimestep, StepFunctionTypeNoTimestep] = take_step_with_timestep
        p_sampler: TopPSampler = TopPSampler(p=0.8)
        top_p, log_probs = BeamSearch(self.end_index, beam_size=beam_size, max_steps=10, sampler=p_sampler).search(initial_predictions, {}, take_step)
        batch_size: int = 5
        assert list(top_p.size())[:-1] == [batch_size, beam_size]
        assert ((0 <= top_p) & (top_p <= 5)).all()
        assert list(log_probs.size()) == [batch_size, beam_size]

    @pytest.mark.parametrize('p_val', [-1.0, 1.2, 1.1, float('inf')])
    def test_p_val(self, p_val: float) -> None:
        with pytest.raises(ValueError):
            initial_predictions: torch.Tensor = torch.tensor([0] * 5)
            take_step: Union[StepFunctionTypeWithTimestep, StepFunctionTypeNoTimestep] = take_step_with_timestep
            beam_size: int = 3
            p_sampler: TopPSampler = TopPSampler(p=p_val, with_replacement=True)
            _ , _ = BeamSearch(self.end_index, beam_size=beam_size, max_steps=10, sampler=p_sampler).search(initial_predictions, {}, take_step)

    def test_top_k_search(self) -> None:
        initial_predictions: torch.Tensor = torch.tensor([0] * 5)
        beam_size: int = 3
        take_step: Union[StepFunctionTypeWithTimestep, StepFunctionTypeNoTimestep] = take_step_with_timestep
        k_sampler: TopKSampler = TopKSampler(k=5, with_replacement=True)
        top_k, log_probs = BeamSearch(self.end_index, beam_size=beam_size, max_steps=10, sampler=k_sampler).search(initial_predictions, {}, take_step)
        batch_size: int = 5
        assert list(top_k.size())[:-1] == [batch_size, beam_size]
        assert ((0 <= top_k) & (top_k <= 5)).all()
        assert list(log_probs.size()) == [batch_size, beam_size]

    @pytest.mark.parametrize('k_val', [-1, 0])
    def test_k_val(self, k_val: int) -> None:
        with pytest.raises(ValueError):
            initial_predictions: torch.Tensor = torch.tensor([0] * 5)
            take_step: Union[StepFunctionTypeWithTimestep, StepFunctionTypeNoTimestep] = take_step_with_timestep
            beam_size: int = 3
            k_sampler: TopKSampler = TopKSampler(k=k_val, with_replacement=True)
            _ , _ = BeamSearch(self.end_index, beam_size=beam_size, max_steps=10, sampler=k_sampler).search(initial_predictions, {}, take_step)

    def test_stochastic_beam_search(self) -> None:
        initial_predictions: torch.Tensor = torch.tensor([0] * 5)
        batch_size: int = 5
        beam_size: int = 3
        take_step: Union[StepFunctionTypeWithTimestep, StepFunctionTypeNoTimestep] = take_step_with_timestep
        gumbel_sampler: GumbelSampler = GumbelSampler()
        top_k, log_probs = BeamSearch(self.end_index, beam_size=beam_size, max_steps=10, sampler=gumbel_sampler).search(initial_predictions, {}, take_step)
        assert list(top_k.size())[:-1] == [batch_size, beam_size]
        assert ((0 <= top_k) & (top_k <= 5)).all()
        assert list(log_probs.size()) == [batch_size, beam_size]
        for batch in top_k:
            for beam in batch:
                reached_end: bool = False
                for token in beam:
                    if token == self.end_index:
                        reached_end = True
                    if reached_end:
                        assert token == self.end_index

    def test_params_sampling(self) -> None:
        beam_search: BeamSearch = BeamSearch.from_params(Params({'sampler': {'type': 'top-k', 'k': 4}, 'beam_size': 2, 'end_index': 7}))
        assert beam_search.beam_size == 2
        assert beam_search._end_index == 7
        assert beam_search.sampler is not None

    def test_params_p_sampling(self) -> None:
        beam_search: BeamSearch = BeamSearch.from_params(Params({'sampler': {'type': 'top-p', 'p': 0.8}, 'beam_size': 2, 'end_index': 7}))
        assert beam_search.beam_size == 2
        assert beam_search._end_index == 7
        assert beam_search.sampler is not None

    def test_multinomial_sampler(self) -> None:
        sampler: MultinomialSampler = MultinomialSampler(temperature=0.9)
        probabilities, classes, state = sampler.sample_nodes(log_probabilities, 3, {'foo': 'bar'})
        assert probabilities.size() == classes.size()
        assert classes.size() == (2, 3)
        assert all([x < 4 for x in classes[0]])
        assert all([x > 1 for x in classes[1]])

    def test_top_k_sampler(self) -> None:
        sampler: TopKSampler = TopKSampler(k=3, temperature=0.9)
        probabilities, classes, state = sampler.sample_nodes(log_probabilities, 3, {'foo': 'bar'})
        assert probabilities.size() == classes.size()
        assert classes.size() == (2, 3)
        assert all([0 < x < 4 for x in classes[0]])
        assert all([1 < x < 5 for x in classes[1]])

    def test_top_p_sampler(self) -> None:
        sampler: TopPSampler = TopPSampler(p=0.8, temperature=0.9)
        probabilities, classes, state = sampler.sample_nodes(log_probabilities, 3, {'foo': 'bar'})
        assert probabilities.size() == classes.size()
        assert classes.size() == (2, 3)
        assert all([0 < x < 4 for x in classes[0]])
        assert all([1 < x < 5 for x in classes[1]])
        sampler = TopPSampler(p=0.7, temperature=1.0)
        probabilities, classes, state = sampler.sample_nodes(log_probabilities, 2, {'foo': 'bar'})
        assert all([x in (1, 2, 3) for x in classes[0]])
        assert all([x in (2, 3) for x in classes[1]])

    def test_gumbel_sampler(self) -> None:
        sampler: GumbelSampler = GumbelSampler()
        num_classes: int = len(log_probabilities[0])
        sampler_state: Any = sampler.init_state(log_probabilities, batch_size=2, num_classes=num_classes)
        log_probs, indices, state = sampler.sample_beams(log_probabilities, 3, sampler_state)
        assert log_probs.size() == indices.size()
        assert indices.size() == (2, 3)
        _, sorted_indices = log_probs.sort(dim=-1, descending=True)
        assert (sorted_indices == torch.arange(3).unsqueeze(0)).all()
        assert all([0 <= x < 4 for x in indices[0]])
        assert all([1 < x <= 5 for x in indices[1]])

    def test_length_normalized_sequence_log_prob_scorer(self) -> None:
        self.beam_search.final_sequence_scorer = LengthNormalizedSequenceLogProbabilityScorer()
        expected_log_probs: np.ndarray = np.log(np.array([0.4, 0.3, 0.2]))
        length_normalization: np.ndarray = np.array([5, 4, 3])
        expected_scores: np.ndarray = expected_log_probs / length_normalization
        self._check_results(expected_log_probs=expected_scores)
        length_penalty: float = 2.0
        self.beam_search.final_sequence_scorer = LengthNormalizedSequenceLogProbabilityScorer(length_penalty=length_penalty)
        expected_log_probs = np.log(np.array([0.4, 0.3, 0.2]))
        length_normalization = np.array([5 ** length_penalty, 4 ** length_penalty, 3 ** length_penalty])
        expected_scores = expected_log_probs / length_normalization
        self._check_results(expected_log_probs=expected_scores)
        length_penalty = -2.0
        self.beam_search.final_sequence_scorer = LengthNormalizedSequenceLogProbabilityScorer(length_penalty=length_penalty)
        expected_top_k: np.ndarray = np.array([[3, 4, 5, 5, 5],
                                               [2, 3, 4, 5, 5],
                                               [1, 2, 3, 4, 5]])
        expected_log_probs = np.log(np.array([0.2, 0.3, 0.4]))
        length_normalization = np.array([3 ** length_penalty, 4 ** length_penalty, 5 ** length_penalty])
        expected_scores = expected_log_probs / length_normalization
        self._check_results(expected_top_k=expected_top_k, expected_log_probs=expected_scores)
        length_penalty = 2.0
        self.beam_search.max_steps = 4
        self.beam_search.final_sequence_scorer = LengthNormalizedSequenceLogProbabilityScorer(length_penalty=length_penalty)
        expected_top_k = np.array([[1, 2, 3, 4],
                                   [2, 3, 4, 5],
                                   [3, 4, 5, 5]])
        expected_log_probs = np.log(np.array([0.4, 0.3, 0.2]))
        length_normalization = np.array([4 ** length_penalty, 4 ** length_penalty, 3 ** length_penalty])
        expected_scores = expected_log_probs / length_normalization
        self._check_results(expected_top_k=expected_top_k, expected_log_probs=expected_scores)

    def test_repeated_ngram_blocking_constraint_init_state(self) -> None:
        ngram_size: int = 3
        batch_size: int = 2
        constraint: RepeatedNGramBlockingConstraint = RepeatedNGramBlockingConstraint(ngram_size)
        state: List[Any] = constraint.init_state(batch_size)
        assert len(state) == batch_size
        for beam_states in state:
            assert len(beam_states) == 1
            beam_state: Dict[str, Any] = beam_states[0]
            assert len(beam_state.keys()) == 2
            assert len(beam_state['current_prefix']) == 0
            assert len(beam_state['seen_ngrams']) == 0

    def test_repeated_ngram_blocking_constraint_apply(self) -> None:
        ngram_size: int = 3
        batch_size: int = 2
        beam_size: int = 2
        num_classes: int = 10
        constraint: RepeatedNGramBlockingConstraint = RepeatedNGramBlockingConstraint(ngram_size)
        state: List[List[Dict[str, Any]]] = [
            [{'current_prefix': [0, 1], 'seen_ngrams': {}},
             {'current_prefix': [2, 3], 'seen_ngrams': {(2, 3): [4]}}],
            [{'current_prefix': [4, 5], 'seen_ngrams': {(8, 9): []}},
             {'current_prefix': [6, 7], 'seen_ngrams': {(6, 7): [0, 1, 2]}}]
        ]
        log_probabilities: torch.Tensor = torch.rand(batch_size, beam_size, num_classes)
        constraint.apply(state, log_probabilities)
        disallowed_locations: List[List[int]] = torch.nonzero(log_probabilities == min_value_of_dtype(log_probabilities.dtype)).tolist()
        assert len(disallowed_locations) == 4
        assert [0, 1, 4] in disallowed_locations
        assert [1, 1, 0] in disallowed_locations
        assert [1, 1, 1] in disallowed_locations
        assert [1, 1, 2] in disallowed_locations

    def test_repeated_ngram_blocking_constraint_update_state(self) -> None:
        ngram_size: int = 3
        constraint: RepeatedNGramBlockingConstraint = RepeatedNGramBlockingConstraint(ngram_size)
        state: List[List[Dict[str, Any]]] = [
            [{'current_prefix': [0, 1], 'seen_ngrams': {}},
             {'current_prefix': [2, 3], 'seen_ngrams': {(2, 3): [4]}}],
            [{'current_prefix': [4, 5], 'seen_ngrams': {(8, 9): []}},
             {'current_prefix': [6, 7], 'seen_ngrams': {(6, 7): [0, 1, 2]}}]
        ]
        predictions: torch.Tensor = torch.LongTensor([[5, 6], [0, 3]])
        backpointers: torch.Tensor = torch.LongTensor([[1, 1], [0, 1]])
        expected_state: List[List[Dict[str, Any]]] = [
            [{'current_prefix': [3, 5], 'seen_ngrams': {(2, 3): [4, 5]}},
             {'current_prefix': [3, 6], 'seen_ngrams': {(2, 3): [4, 6]}}],
            [{'current_prefix': [5, 0], 'seen_ngrams': {(8, 9): [], (4, 5): [0]}},
             {'current_prefix': [7, 3], 'seen_ngrams': {(6, 7): [0, 1, 2, 3]}}]
        ]
        updated_state: List[List[Dict[str, Any]]] = constraint.update_state(state, predictions, backpointers)
        assert updated_state == expected_state

    def test_take_repeated_ngram_step(self) -> None:
        step_function: StepFunctionTypeNoTimestep = get_step_function(repeated_ngram_transition_probabilities_0)
        self.beam_search.beam_size = 2
        self.beam_search.max_steps = 5
        expected_top_k: np.ndarray = np.array([[1, 3, 1, 3, 1], [1, 2, 3, 1, 3]])
        expected_log_probs: np.ndarray = np.log(np.array([0.36, 0.24]))
        self._check_results(expected_top_k=expected_top_k, expected_log_probs=expected_log_probs, take_step=step_function)

    def test_repeated_ngram_blocking_end_to_end_unigrams(self) -> None:
        step_function: StepFunctionTypeNoTimestep = get_step_function(repeated_ngram_transition_probabilities_0)
        self.beam_search.beam_size = 2
        self.beam_search.max_steps = 3
        self.beam_search.constraints = [RepeatedNGramBlockingConstraint(ngram_size=1)]
        expected_top_k: np.ndarray = np.array([[1, 2, 3], [1, 3, 5]])
        expected_log_probs: np.ndarray = np.log(np.array([0.4, 0.6 * 1e-09]))
        self._check_results(expected_top_k=expected_top_k, expected_log_probs=expected_log_probs, take_step=step_function)
        step_function = get_step_function(repeated_ngram_transition_probabilities_1)
        self.beam_search.max_steps = 5
        expected_top_k = np.array([[1, 2, 3, 4, 5], [1, 2, 4, 3, 5]])
        expected_log_probs = np.log(np.array([0.4 * 0.3 * 0.3 * 0.2 * 0.1, 0.4 * 0.3 * 0.2 * 0.3 * 0.1]))
        self._check_results(expected_top_k=expected_top_k, expected_log_probs=expected_log_probs, take_step=step_function)

    def test_repeated_ngram_blocking_end_to_end_bigrams(self) -> None:
        step_function: StepFunctionTypeNoTimestep = get_step_function(repeated_ngram_transition_probabilities_0)
        self.beam_search.beam_size = 2
        self.beam_search.max_steps = 4
        self.beam_search.constraints = [RepeatedNGramBlockingConstraint(ngram_size=2)]
        expected_top_k: np.ndarray = np.array([[1, 2, 3, 1], [1, 3, 1, 2]])
        expected_log_probs: np.ndarray = np.log(np.array([0.4, 0.24]))
        self._check_results(expected_top_k=expected_top_k, expected_log_probs=expected_log_probs, take_step=step_function)

    def test_repeated_ngram_blocking_end_to_end_trigrams(self) -> None:
        step_function: StepFunctionTypeNoTimestep = get_step_function(repeated_ngram_transition_probabilities_0)
        self.beam_search.beam_size = 2
        self.beam_search.max_steps = 5
        self.beam_search.constraints = [RepeatedNGramBlockingConstraint(ngram_size=3)]
        expected_top_k: np.ndarray = np.array([[1, 2, 3, 1, 3], [1, 2, 3, 1, 2]])
        expected_log_probs: np.ndarray = np.log(np.array([0.24, 0.16]))
        self._check_results(expected_top_k=expected_top_k, expected_log_probs=expected_log_probs, take_step=step_function)

    def test_repeated_ngram_blocking_end_indices(self) -> None:
        step_function: StepFunctionTypeNoTimestep = get_step_function(repeated_ngram_transition_probabilities_0)
        self.beam_search.beam_size = 2
        self.beam_search.constraints = [RepeatedNGramBlockingConstraint(ngram_size=1)]
        expected_top_k: np.ndarray = np.array([[1, 3, 5, 5], [1, 2, 3, 5]])
        expected_log_probs: np.ndarray = np.log(np.array([0.6 * 1e-09, 0.4 * 1e-09]))
        self._check_results(expected_top_k=expected_top_k, expected_log_probs=expected_log_probs, take_step=step_function)
