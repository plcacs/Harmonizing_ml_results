from typing import Dict, Tuple, Union
import numpy as np
import pytest
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.nn.beam_search import MultinomialSampler, BeamSearch, TopKSampler, TopPSampler, GumbelSampler, LengthNormalizedSequenceLogProbabilityScorer, RepeatedNGramBlockingConstraint, StepFunctionTypeWithTimestep, StepFunctionTypeNoTimestep
from allennlp.common.params import Params
from allennlp.nn.util import min_value_of_dtype

transition_probabilities: torch.Tensor = torch.tensor([[0.0, 0.4, 0.3, 0.2, 0.1, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.2, 0.1, 0.2, 0.2, 0.2, 0.1]])
short_sequence_transition_probabilities: torch.Tensor = torch.tensor([[0.0, 0.1, 0.0, 0.0, 0.0, 0.9], [0.0, 0.0, 0.1, 0.0, 0.0, 0.9], [0.0, 0.0, 0.0, 0.1, 0.0, 0.9], [0.0, 0.0, 0.0, 0.0, 0.1, 0.9], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.2, 0.1, 0.2, 0.2, 0.2, 0.1]])
repeated_ngram_transition_probabilities_0: torch.Tensor = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.4, 0.6, 0.0, 1e-09], [0.0, 0.0, 0.0, 1.0, 0.0, 1e-09], [0.0, 1.0, 0.0, 0.0, 0.0, 1e-09], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
repeated_ngram_transition_probabilities_1: torch.Tensor = torch.tensor([[0.0, 0.4, 0.3, 0.2, 0.1, 0.0], [0.0, 0.4, 0.3, 0.2, 0.1, 0.1], [0.0, 0.0, 0.4, 0.3, 0.2, 0.1], [0.0, 0.0, 0.3, 0.4, 0.2, 0.1], [0.0, 0.0, 0.2, 0.3, 0.4, 0.1], [0.2, 0.1, 0.2, 0.2, 0.2, 0.1]])
log_probabilities: torch.Tensor = torch.log(torch.tensor([[0.1, 0.3, 0.3, 0.3, 0.0, 0.0], [0.0, 0.0, 0.4, 0.3, 0.2, 0.1]]))

def get_step_function(transition_matrix: torch.Tensor, with_timestep: bool = False) -> Union[StepFunctionTypeNoTimestep, StepFunctionTypeWithTimestep]:

    def _step_function(last_predictions: torch.Tensor, state: Dict) -> Tuple[torch.Tensor, Dict]:
        log_probs_list = []
        for last_token in last_predictions:
            log_probs = torch.log(transition_matrix[last_token.item()])
            log_probs_list.append(log_probs)
        return (torch.stack(log_probs_list), state)
    if not with_timestep:
        return _step_function

    def _step_function_with_timestep(last_predictions: torch.Tensor, state: Dict, timestep: int) -> Tuple[torch.Tensor, Dict]:
        return _step_function(last_predictions, state)
    return _step_function_with_timestep

take_step_no_timestep: StepFunctionTypeNoTimestep = get_step_function(transition_probabilities)
take_step_with_timestep: StepFunctionTypeWithTimestep = get_step_function(transition_probabilities, with_timestep=True)
take_short_sequence_step: StepFunctionTypeNoTimestep = get_step_function(short_sequence_transition_probabilities)
