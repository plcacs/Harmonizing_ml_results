from typing import Dict, Tuple, Union, List, Optional, Any, Set
import numpy as np
import pytest
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.nn.beam_search import MultinomialSampler, BeamSearch, TopKSampler, TopPSampler, GumbelSampler, LengthNormalizedSequenceLogProbabilityScorer, RepeatedNGramBlockingConstraint, StepFunctionTypeWithTimestep, StepFunctionTypeNoTimestep
from allennlp.common.params import Params
from allennlp.nn.util import min_value_of_dtype
from torch import Tensor

transition_probabilities: Tensor = torch.tensor([[0.0, 0.4, 0.3, 0.2, 0.1, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.2, 0.1, 0.2, 0.2, 0.2, 0.1]])
short_sequence_transition_probabilities: Tensor = torch.tensor([[0.0, 0.1, 0.0, 0.0, 0.0, 0.9], [0.0, 0.0, 0.1, 0.0, 0.0, 0.9], [0.0, 0.0, 0.0, 0.1, 0.0, 0.9], [0.0, 0.0, 0.0, 0.0, 0.1, 0.9], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.2, 0.1, 0.2, 0.2, 0.2, 0.1]])
repeated_ngram_transition_probabilities_0: Tensor = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.4, 0.6, 0.0, 1e-09], [0.0, 0.0, 0.0, 1.0, 0.0, 1e-09], [0.0, 1.0, 0.0, 0.0, 0.0, 1e-09], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
repeated_ngram_transition_probabilities_1: Tensor = torch.tensor([[0.0, 0.4, 0.3, 0.2, 0.1, 0.0], [0.0, 0.4, 0.3, 0.2, 0.1, 0.1], [0.0, 0.0, 0.4, 0.3, 0.2, 0.1], [0.0, 0.0, 0.3, 0.4, 0.2, 0.1], [0.0, 0.0, 0.2, 0.3, 0.4, 0.1], [0.2, 0.1, 0.2, 0.2, 0.2, 0.1]])
log_probabilities: Tensor = torch.log(torch.tensor([[0.1, 0.3, 0.3, 0.3, 0.0, 0.0], [0.0, 0.0, 0.4, 0.3, 0.2, 0.1]]))

def get_step_function(transition_matrix: Tensor, with_timestep: bool = False) -> Union[StepFunctionTypeWithTimestep, StepFunctionTypeNoTimestep]:
    def _step_function(last_predictions: Tensor, state: Dict[str, Any]) -> Tuple[Tensor, Dict[str, Any]]:
        log_probs_list: List[Tensor] = []
        for last_token in last_predictions:
            log_probs: Tensor = torch.log(transition_matrix[last_token.item()])
            log_probs_list.append(log_probs)
        return (torch.stack(log_probs_list), state)
    
    if not with_timestep:
        return _step_function

    def _step_function_with_timestep(last_predictions: Tensor, state: Dict[str, Any], timestep: int) -> Tuple[Tensor, Dict[str, Any]]:
        return _step_function(last_predictions, state)
    return _step_function_with_timestep

take_step_no_timestep: StepFunctionTypeNoTimestep = get_step_function(transition_probabilities)
take_step_with_timestep: StepFunctionTypeWithTimestep = get_step_function(transition_probabilities, with_timestep=True)
take_short_sequence_step: StepFunctionTypeNoTimestep = get_step_function(short_sequence_transition_probabilities)

class BeamSearchTest(AllenNlpTestCase):
    def setup_method(self) -> None:
        super().setup_method()
        self.end_index: int = transition_probabilities.size()[0] - 1
        self.beam_search: BeamSearch = BeamSearch(self.end_index, max_steps=10, beam_size=3)
        self.expected_top_k: np.ndarray = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 5], [3, 4, 5, 5, 5]])
        self.expected_log_probs: np.ndarray = np.log(np.array([0.4, 0.3, 0.2]))

    def _check_results(
        self,
        batch_size: int = 5,
        expected_top_k: Optional[np.ndarray] = None,
        expected_log_probs: Optional[np.ndarray] = None,
        beam_search: Optional[BeamSearch] = None,
        state: Optional[Dict[str, Any]] = None,
        take_step: Union[StepFunctionTypeWithTimestep, StepFunctionTypeNoTimestep] = take_step_with_timestep
    ) -> None:
        expected_top_k = expected_top_k if expected_top_k is not None else self.expected_top_k
        expected_log_probs = expected_log_probs if expected_log_probs is not None else self.expected_log_probs
        state = state or {}
        beam_search = beam_search or self.beam_search
        beam_size: int = beam_search.beam_size
        initial_predictions: Tensor = torch.tensor([0] * batch_size)
        top_k: Tensor
        log_probs: Tensor
        top_k, log_probs = beam_search.search(initial_predictions, state, take_step)
        assert list(top_k.size())[:-1] == [batch_size, beam_size]
        np.testing.assert_array_equal(top_k[0].numpy(), expected_top_k)
        assert list(log_probs.size()) == [batch_size, beam_size]
        np.testing.assert_allclose(log_probs[0].numpy(), expected_log_probs, rtol=1e-06)

    @pytest.mark.parametrize('step_function', [take_step_with_timestep, take_step_no_timestep])
    def test_search(self, step_function: Union[StepFunctionTypeWithTimestep, StepFunctionTypeNoTimestep]) -> None:
        self._check_results(take_step=step_function)

    def test_finished_state(self) -> None:
        state: Dict[str, Any] = {}
        state['foo'] = torch.tensor([[1, 0, 1], [2, 0, 1], [0, 0, 1], [1, 1, 1], [0, 0, 0]])
        expected_finished_state: Dict[str, np.ndarray] = {}
        expected_finished_state['foo'] = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1], [2, 0, 1], [2, 0, 1], [2, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self._check_results(state=state)
        for key, array in expected_finished_state.items():
            np.testing.assert_allclose(state[key].numpy(), array)

    def test_diff_shape_state(self) -> None:
        state: Dict[str, Any] = {}
        state['decoder_hidden'] = torch.tensor([[1, 0, 1], [2, 0, 1], [0, 0, 1], [1, 1, 1], [0, 0, 0]])
        state['decoder_hidden'] = state['decoder_hidden'].unsqueeze(0).repeat(2, 1, 1)
        seq: List[List[int]] = [[1, 0, 1], [1, 0, 1], [1, 0, 1], [2, 0, 1], [2, 0, 1], [2, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        seq = [seq] * 2
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
        expected_top_k: np.ndarray = np.array([[5, 5, 5, 5, 5], [1, 5, 5, 5, 5], [1, 2, 5, 5, 5], [1, 2, 3, 5, 5], [1, 2, 3, 4, 5]])
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
        expected_top_k = np.array([[1, 2, 5, 5, 5], [1, 2, 3, 5, 5], [1, 2, 3, 4, 5]])
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
        initial_predictions: Tensor = torch.LongTensor([self.end_index - 1, self.end_index - 1])
        with pytest.warns(RuntimeWarning, match='Negligible log probabilities'):
            self.beam_search.search(initial_predictions, {}, take_step_no_timestep)

    def test_empty_sequences(self) -> None:
        initial_predictions: Tensor = torch.LongTensor([self.end_index - 1, self.end_index - 1])
        beam_search: BeamSearch = BeamSearch(self.end_index, beam_size=1)
        with pytest.warns(RuntimeWarning, match='Empty sequences predicted'):
            predictions: Tensor
            log_probs: Tensor
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
        initial_predictions: Tensor = torch.tensor([0] * 5)
        beam_size: int = 3
        take_step: StepFunctionTypeWithTimestep = take_step_with_timestep
        p_sampler: TopPSampler = TopPSampler(p=0.8)
        top_p: Tensor
        log_probs: Tensor
        top_p, log_probs = BeamSearch(self.end_index, beam_size=beam_size, max_steps=10, sampler=p_sampler).search(initial_predictions, {}, take_step)
        beam_size = beam_size or 1
        batch_size: int = 5
        assert list(top_p.size())[:-1] == [batch_size, beam_size]
        assert ((0 <= top_p) & (top_p <= 5)).all()
        assert list(log_probs.size()) == [batch_size, beam_size]

    @pytest.mark.parametrize('p_val', [-1.0, 1.2, 1.1, float('inf')])
    def test_p_val(self, p_val: float) -> None