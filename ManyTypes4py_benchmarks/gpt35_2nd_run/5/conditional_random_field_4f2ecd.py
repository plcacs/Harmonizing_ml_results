from typing import List, Tuple, Dict, Union
import torch
from allennlp.common.checks import ConfigurationError
import allennlp.nn.util as util
VITERBI_DECODING: Tuple[List[int], float]

def allowed_transitions(constraint_type: str, labels: Dict[int, str]) -> List[Tuple[int, int]]:
    num_labels: int = len(labels)
    start_tag: int = num_labels
    end_tag: int = num_labels + 1
    labels_with_boundaries: List[Tuple[int, str]] = list(labels.items()) + [(start_tag, 'START'), (end_tag, 'END')]
    allowed: List[Tuple[int, int]] = []
    for from_label_index, from_label in labels_with_boundaries:
        if from_label in ('START', 'END'):
            from_tag: str = from_label
            from_entity: str = ''
        else:
            from_tag: str = from_label[0]
            from_entity: str = from_label[1:]
        for to_label_index, to_label in labels_with_boundaries:
            if to_label in ('START', 'END'):
                to_tag: str = to_label
                to_entity: str = ''
            else:
                to_tag: str = to_label[0]
                to_entity: str = to_label[1:]
            if is_transition_allowed(constraint_type, from_tag, from_entity, to_tag, to_entity):
                allowed.append((from_label_index, to_label_index))
    return allowed

def is_transition_allowed(constraint_type: str, from_tag: str, from_entity: str, to_tag: str, to_entity: str) -> bool:
    if to_tag == 'START' or from_tag == 'END':
        return False
    if constraint_type == 'BIOUL':
        if from_tag == 'START':
            return to_tag in ('O', 'B', 'U')
        if to_tag == 'END':
            return from_tag in ('O', 'L', 'U')
        return any([from_tag in ('O', 'L', 'U') and to_tag in ('O', 'B', 'U'), from_tag in ('B', 'I') and to_tag in ('I', 'L') and (from_entity == to_entity)])
    elif constraint_type == 'BIO':
        if from_tag == 'START':
            return to_tag in ('O', 'B')
        if to_tag == 'END':
            return from_tag in ('O', 'B', 'I')
        return any([to_tag in ('O', 'B'), to_tag == 'I' and from_tag in ('B', 'I') and (from_entity == to_entity)])
    elif constraint_type == 'IOB1':
        if from_tag == 'START':
            return to_tag in ('O', 'I')
        if to_tag == 'END':
            return from_tag in ('O', 'B', 'I')
        return any([to_tag in ('O', 'I'), to_tag == 'B' and from_tag in ('B', 'I') and (from_entity == to_entity)])
    elif constraint_type == 'BMES':
        if from_tag == 'START':
            return to_tag in ('B', 'S')
        if to_tag == 'END':
            return from_tag in ('E', 'S')
        return any([to_tag in ('B', 'S') and from_tag in ('E', 'S'), to_tag == 'M' and from_tag in ('B', 'M') and (from_entity == to_entity), to_tag == 'E' and from_tag in ('B', 'M') and (from_entity == to_entity)])
    else:
        raise ConfigurationError(f'Unknown constraint type: {constraint_type}')

class ConditionalRandomField(torch.nn.Module):
    def __init__(self, num_tags: int, constraints: List[Tuple[int, int]] = None, include_start_end_transitions: bool = True):
        self.num_tags: int = num_tags
        self.transitions: torch.nn.Parameter = torch.nn.Parameter(torch.empty(num_tags, num_tags))
        if constraints is None:
            constraint_mask: torch.Tensor = torch.full((num_tags + 2, num_tags + 2), 1.0)
        else:
            constraint_mask: torch.Tensor = torch.full((num_tags + 2, num_tags + 2), 0.0)
            for i, j in constraints:
                constraint_mask[i, j] = 1.0
        self._constraint_mask: torch.nn.Parameter = torch.nn.Parameter(constraint_mask, requires_grad=False)
        self.include_start_end_transitions: bool = include_start_end_transitions
        if include_start_end_transitions:
            self.start_transitions: torch.nn.Parameter = torch.nn.Parameter(torch.Tensor(num_tags))
            self.end_transitions: torch.nn.Parameter = torch.nn.Parameter(torch.Tensor(num_tags))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.transitions)
        if self.include_start_end_transitions:
            torch.nn.init.normal_(self.start_transitions)
            torch.nn.init.normal_(self.end_transitions)

    def _input_likelihood(self, logits: torch.Tensor, transitions: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        ...

    def _joint_likelihood(self, logits: torch.Tensor, transitions: torch.Tensor, tags: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        ...

    def forward(self, inputs: torch.Tensor, tags: torch.Tensor, mask: torch.BoolTensor = None) -> torch.Tensor:
        ...

    def viterbi_tags(self, logits: torch.Tensor, mask: torch.BoolTensor = None, top_k: int = None) -> Union[List[List[Tuple[List[int], float]]], List[Tuple[List[int], float]]:
        ...
