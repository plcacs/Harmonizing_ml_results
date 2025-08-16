from typing import Any, List, Callable, Tuple, Dict
import torch

StateType = Dict[str, torch.Tensor]
StepFunctionTypeWithTimestep = Callable[[torch.Tensor, StateType, int], Tuple[torch.Tensor, StateType]]
StepFunctionTypeNoTimestep = Callable[[torch.Tensor, StateType], Tuple[torch.Tensor, StateType]]
StepFunctionType = Callable[[torch.Tensor, StateType, int], Tuple[torch.Tensor, StateType]]
ConstraintStateType = List[List[Dict[str, Any]]]

class Sampler(Registrable):
    def init_state(self, start_class_log_probabilities: torch.Tensor, batch_size: int, num_classes: int) -> StateType:
        ...

    def sample_nodes(self, log_probs: torch.Tensor, per_node_beam_size: int, state: StateType) -> Tuple[torch.Tensor, torch.Tensor, StateType]:
        ...

    def sample_beams(self, log_probs: torch.Tensor, beam_size: int, state: StateType) -> Tuple[torch.Tensor, torch.Tensor, StateType]:
        ...

@Sampler.register('deterministic')
class DeterministicSampler(Sampler):
    def sample_nodes(self, log_probs: torch.Tensor, per_node_beam_size: int, state: StateType) -> Tuple[torch.Tensor, torch.Tensor, StateType]:
        ...

@Sampler.register('multinomial')
class MultinomialSampler(Sampler):
    def __init__(self, temperature: float = 1.0, with_replacement: bool = False):
        ...

    def sample_nodes(self, log_probs: torch.Tensor, per_node_beam_size: int, state: StateType) -> Tuple[torch.Tensor, torch.Tensor, StateType]:
        ...

@Sampler.register('top-k')
class TopKSampler(Sampler):
    def __init__(self, k: int = 1, temperature: float = 1.0, with_replacement: bool = False):
        ...

    def sample_nodes(self, log_probs: torch.Tensor, per_node_beam_size: int, state: StateType) -> Tuple[torch.Tensor, torch.Tensor, StateType]:
        ...

@Sampler.register('top-p')
class TopPSampler(Sampler):
    def __init__(self, p: float = 0.9, temperature: float = 1.0, with_replacement: bool = False):
        ...

    def sample_nodes(self, log_probs: torch.Tensor, per_node_beam_size: int, state: StateType) -> Tuple[torch.Tensor, torch.Tensor, StateType]:
        ...

@Sampler.register('gumbel')
class GumbelSampler(Sampler):
    def __init__(self, temperature: float = 1.0):
        ...

    def init_state(self, start_class_log_probabilities: torch.Tensor, batch_size: int, num_classes: int) -> StateType:
        ...

    def sample_nodes(self, log_probs: torch.Tensor, per_node_beam_size: int, state: StateType) -> Tuple[torch.Tensor, torch.Tensor, StateType]:
        ...

    def sample_beams(self, log_probs: torch.Tensor, beam_size: int, state: StateType) -> Tuple[torch.Tensor, torch.Tensor, StateType]:
        ...

class FinalSequenceScorer(Registrable):
    def score(self, predictions: torch.Tensor, log_probabilities: torch.Tensor, end_index: int) -> torch.Tensor:
        ...

@FinalSequenceScorer.register('sequence-log-prob')
class SequenceLogProbabilityScorer(FinalSequenceScorer):
    def score(self, predictions: torch.Tensor, log_probabilities: torch.Tensor, end_index: int) -> torch.Tensor:
        ...

@FinalSequenceScorer.register('length-normalized-sequence-log-prob')
class LengthNormalizedSequenceLogProbabilityScorer(FinalSequenceScorer):
    def __init__(self, length_penalty: float = 1.0):
        ...

    def score(self, predictions: torch.Tensor, log_probabilities: torch.Tensor, end_index: int) -> torch.Tensor:
        ...

class Constraint(Registrable):
    def init_state(self, batch_size: int) -> ConstraintStateType:
        ...

    def apply(self, state: ConstraintStateType, class_log_probabilities: torch.Tensor) -> torch.Tensor:
        ...

    def update_state(self, state: ConstraintStateType, last_prediction: torch.Tensor) -> ConstraintStateType:
        ...

@Constraint.register('repeated-ngram-blocking')
class RepeatedNGramBlockingConstraint(Constraint):
    def __init__(self, ngram_size: int, **kwargs):
        ...

    def init_state(self, batch_size: int) -> ConstraintStateType:
        ...

    def apply(self, state: ConstraintStateType, class_log_probabilities: torch.Tensor) -> torch.Tensor:
        ...

    def update_state(self, state: ConstraintStateType, last_prediction: torch.Tensor) -> ConstraintStateType:
        ...

class BeamSearch(Registrable):
    def __init__(self, end_index: int, max_steps: int = 50, beam_size: int = 10, per_node_beam_size: int = None, sampler: Sampler = None, min_steps: int = None, final_sequence_scorer: FinalSequenceScorer = None, constraints: List[Constraint] = None, vocab: Vocabulary = None):
        ...

    def search(self, start_predictions: torch.Tensor, start_state: StateType, step: StepFunctionType) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    def _search(self, start_predictions: torch.Tensor, start_state: StateType, step: StepFunctionType) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    def _reconstruct_sequences(self, predictions: List[torch.Tensor], backpointers: List[torch.Tensor]) -> List[torch.Tensor]:
        ...

    def _update_initial_state(self, state: StateType, batch_size: int):
        ...

    def _update_state(self, state: StateType, backpointer: torch.Tensor):
        ...
