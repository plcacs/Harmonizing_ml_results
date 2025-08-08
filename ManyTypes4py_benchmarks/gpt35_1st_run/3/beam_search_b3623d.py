from typing import Any, List, Callable, Tuple, Dict, TypeVar, Optional
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
    default_implementation: str

    def init_state(self, start_class_log_probabilities: torch.Tensor, batch_size: int, num_classes: int) -> StateType:
        ...

    def sample_nodes(self, log_probs: torch.Tensor, per_node_beam_size: int, state: StateType) -> Tuple[torch.Tensor, torch.Tensor, StateType]:
        ...

    def sample_beams(self, log_probs: torch.Tensor, beam_size: int, state: StateType) -> Tuple[torch.Tensor, torch.Tensor, StateType]:
        ...

@Sampler.register('deterministic')
class DeterministicSampler(Sampler):
    ...

@Sampler.register('multinomial')
class MultinomialSampler(Sampler):
    ...

@Sampler.register('top-k')
class TopKSampler(Sampler):
    ...

@Sampler.register('top-p')
class TopPSampler(Sampler):
    ...

@Sampler.register('gumbel')
class GumbelSampler(Sampler):
    ...

class FinalSequenceScorer(Registrable):
    default_implementation: str

    def score(self, predictions: torch.Tensor, log_probabilities: torch.Tensor, end_index: int) -> torch.Tensor:
        ...

@FinalSequenceScorer.register('sequence-log-prob')
class SequenceLogProbabilityScorer(FinalSequenceScorer):
    ...

@FinalSequenceScorer.register('length-normalized-sequence-log-prob')
class LengthNormalizedSequenceLogProbabilityScorer(FinalSequenceScorer):
    ...

class Constraint(Registrable):
    def init_state(self, batch_size: int) -> ConstraintStateType:
        ...

    def apply(self, state: ConstraintStateType, class_log_probabilities: torch.Tensor) -> torch.Tensor:
        ...

    def update_state(self, state: ConstraintStateType, last_prediction: torch.Tensor, last_backpointer: Optional[torch.Tensor]) -> ConstraintStateType:
        ...

@Constraint.register('repeated-ngram-blocking')
class RepeatedNGramBlockingConstraint(Constraint):
    ...

class BeamSearch(Registrable):
    def search(self, start_predictions: torch.Tensor, start_state: StateType, step: StepFunctionType) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    def _search(self, start_predictions: torch.Tensor, start_state: StateType, step: StepFunctionType) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    def _reconstruct_sequences(self, predictions: List[torch.Tensor], backpointers: List[torch.Tensor]) -> List[torch.Tensor]:
        ...

    def _is_multilayer_rnn_decoder(self, key: str, state_tensor: torch.Tensor) -> bool:
        ...

    def _update_initial_state(self, state: StateType, batch_size: int) -> None:
        ...

    def _update_state(self, state: StateType, backpointer: torch.Tensor) -> None:
        ...

BeamSearch.register('beam_search')(BeamSearch)
