from typing import Dict, List, Tuple, Union, Optional, Callable, Any
import numpy as np
import torch
import pytest
from allennlp.common.testing import AllenNlpTestCase
from allennlp.nn.beam_search import (
    StepFunctionTypeWithTimestep,
    StepFunctionTypeNoTimestep,
    MultinomialSampler,
    TopKSampler,
    TopPSampler,
    GumbelSampler,
    LengthNormalizedSequenceLogProbabilityScorer,
    RepeatedNGramBlockingConstraint,
)
from allennlp.common.params import Params

transition_probabilities: torch.Tensor = ...
short_sequence_transition_probabilities: torch.Tensor = ...
repeated_ngram_transition_probabilities_0: torch.Tensor = ...
repeated_ngram_transition_probabilities_1: torch.Tensor = ...
log_probabilities: torch.Tensor = ...

def get_step_function(
    transition_matrix: torch.Tensor, with_timestep: bool = False
) -> Union[StepFunctionTypeNoTimestep, StepFunctionTypeWithTimestep]:
    ...

take_step_no_timestep: StepFunctionTypeNoTimestep = ...
take_step_with_timestep: StepFunctionTypeWithTimestep = ...
take_short_sequence_step: StepFunctionTypeWithTimestep = ...

class BeamSearchTest(AllenNlpTestCase):
    def setup_method(self) -> None:
        ...

    def _check_results(
        self,
        batch_size: int = 5,
        expected_top_k: np.ndarray = ...,
        expected_log_probs: np.ndarray = ...,
        beam_search: Optional[Any] = ...,
        state: Optional[Dict[str, torch.Tensor]] = ...,
        take_step: Union[StepFunctionTypeNoTimestep, StepFunctionTypeWithTimestep] = ...,
    ) -> None:
        ...

    @pytest.mark.parametrize('step_function', [take_step_with_timestep, take_step_no_timestep])
    def test_search(self, step_function: Callable[..., Union[StepFunctionTypeNoTimestep, StepFunctionTypeWithTimestep]]) -> None:
        ...

    def test_finished_state(self) -> None:
        ...

    def test_diff_shape_state(self) -> None:
        ...

    def test_batch_size_of_one(self) -> None:
        ...

    def test_greedy_search(self) -> None:
        ...

    def test_single_step(self) -> None:
        ...

    def test_early_stopping(self) -> None:
        ...

    def test_take_short_sequence_step(self) -> None:
        ...

    def test_min_steps(self) -> None:
        ...

    def test_different_per_node_beam_size(self) -> None:
        ...

    def test_catch_bad_config(self) -> None:
        ...

    def test_warn_for_bad_log_probs(self) -> None:
        ...

    def test_empty_sequences(self) -> None:
        ...

    def test_default_from_params_params(self) -> None:
        ...

    def test_top_p_search(self) -> None:
        ...

    @pytest.mark.parametrize('p_val', [-1.0, 1.2, 1.1, float('inf')])
    def test_p_val(self, p_val: float) -> None:
        ...

    def test_top_k_search(self) -> None:
        ...

    @pytest.mark.parametrize('k_val', [-1, 0])
    def test_k_val(self, k_val: int) -> None:
        ...

    def test_stochastic_beam_search(self) -> None:
        ...

    def test_params_sampling(self) -> None:
        ...

    def test_params_p_sampling(self) -> None:
        ...

    def test_multinomial_sampler(self) -> None:
        ...

    def test_top_k_sampler(self) -> None:
        ...

    def test_top_p_sampler(self) -> None:
        ...

    def test_gumbel_sampler(self) -> None:
        ...

    def test_length_normalized_sequence_log_prob_scorer(self) -> None:
        ...

    def test_repeated_ngram_blocking_constraint_init_state(self) -> None:
        ...

    def test_repeated_ngram_blocking_constraint_apply(self) -> None:
        ...

    def test_repeated_ngram_blocking_constraint_update_state(self) -> None:
        ...

    def test_take_repeated_ngram_step(self) -> None:
        ...

    def test_repeated_ngram_blocking_end_to_end_unigrams(self) -> None:
        ...

    def test_repeated_ngram_blocking_end_to_end_bigrams(self) -> None:
        ...

    def test_repeated_ngram_blocking_end_to_end_trigrams(self) -> None:
        ...

    def test_repeated_ngram_blocking_end_indices(self) -> None:
        ...