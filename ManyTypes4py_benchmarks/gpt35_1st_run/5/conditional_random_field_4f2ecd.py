from typing import List, Tuple, Dict
import torch
from allennlp.common.checks import ConfigurationError
import allennlp.nn.util as util
VITERBI_DECODING = Tuple[List[int], float]

def allowed_transitions(constraint_type: str, labels: Dict[int, str]) -> List[Tuple[int, int]]:
    ...

def is_transition_allowed(constraint_type: str, from_tag: str, from_entity: str, to_tag: str, to_entity: str) -> bool:
    ...

class ConditionalRandomField(torch.nn.Module):
    def __init__(self, num_tags: int, constraints: List[Tuple[int, int]] = None, include_start_end_transitions: bool = True):
        ...

    def reset_parameters(self) -> None:
        ...

    def _input_likelihood(self, logits: torch.Tensor, transitions: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        ...

    def _joint_likelihood(self, logits: torch.Tensor, transitions: torch.Tensor, tags: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        ...

    def forward(self, inputs: torch.Tensor, tags: torch.Tensor, mask: torch.BoolTensor = None) -> torch.Tensor:
        ...

    def viterbi_tags(self, logits: torch.Tensor, mask: torch.BoolTensor = None, top_k: int = None) -> List[List[VITERBI_DECODING]]:
        ...
