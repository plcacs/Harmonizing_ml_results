from typing import List, Tuple, Dict, Union, Optional
import torch
from allennlp.common.checks import ConfigurationError
import allennlp.nn.util as util

VITERBI_DECODING = Tuple[List[int], float]

def allowed_transitions(constraint_type: str, labels: Dict[int, str]) -> List[Tuple[int, int]]:
    """
    Given labels and a constraint type, returns the allowed transitions. It will
    additionally include transitions for the start and end states, which are used
    by the conditional random field.
    """
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
            from_tag = from_label[0]
            from_entity = from_label[1:]
        for to_label_index, to_label in labels_with_boundaries:
            if to_label in ('START', 'END'):
                to_tag: str = to_label
                to_entity: str = ''
            else:
                to_tag = to_label[0]
                to_entity = to_label[1:]
            if is_transition_allowed(constraint_type, from_tag, from_entity, to_tag, to_entity):
                allowed.append((from_label_index, to_label_index))
    return allowed

def is_transition_allowed(constraint_type: str, from_tag: str, from_entity: str, to_tag: str, to_entity: str) -> bool:
    """
    Given a constraint type and strings `from_tag` and `to_tag` that
    represent the origin and destination of the transition, return whether
    the transition is allowed under the given constraint type.
    """
    if to_tag == 'START' or from_tag == 'END':
        return False
    if constraint_type == 'BIOUL':
        if from_tag == 'START':
            return to_tag in ('O', 'B', 'U')
        if to_tag == 'END':
            return from_tag in ('O', 'L', 'U')
        return any([from_tag in ('O', 'L', 'U') and to_tag in ('O', 'B', 'U'),
                    from_tag in ('B', 'I') and to_tag in ('I', 'L') and (from_entity == to_entity)])
    elif constraint_type == 'BIO':
        if from_tag == 'START':
            return to_tag in ('O', 'B')
        if to_tag == 'END':
            return from_tag in ('O', 'B', 'I')
        return any([to_tag in ('O', 'B'),
                    to_tag == 'I' and from_tag in ('B', 'I') and (from_entity == to_entity)])
    elif constraint_type == 'IOB1':
        if from_tag == 'START':
            return to_tag in ('O', 'I')
        if to_tag == 'END':
            return from_tag in ('O', 'B', 'I')
        return any([to_tag in ('O', 'I'),
                    to_tag == 'B' and from_tag in ('B', 'I') and (from_entity == to_entity)])
    elif constraint_type == 'BMES':
        if from_tag == 'START':
            return to_tag in ('B', 'S')
        if to_tag == 'END':
            return from_tag in ('E', 'S')
        return any([to_tag in ('B', 'S') and from_tag in ('E', 'S'),
                    to_tag == 'M' and from_tag in ('B', 'M') and (from_entity == to_entity),
                    to_tag == 'E' and from_tag in ('B', 'M') and (from_entity == to_entity)])
    else:
        raise ConfigurationError(f'Unknown constraint type: {constraint_type}')

class ConditionalRandomField(torch.nn.Module):
    """
    This module uses the "forward-backward" algorithm to compute
    the log-likelihood of its inputs assuming a conditional random field model.
    """
    def __init__(self, num_tags: int, constraints: Optional[List[Tuple[int, int]]] = None, include_start_end_transitions: bool = True) -> None:
        super().__init__()
        self.num_tags: int = num_tags
        self.transitions: torch.nn.Parameter = torch.nn.Parameter(torch.empty(num_tags, num_tags))
        if constraints is None:
            constraint_mask: torch.Tensor = torch.full((num_tags + 2, num_tags + 2), 1.0)
        else:
            constraint_mask = torch.full((num_tags + 2, num_tags + 2), 0.0)
            for i, j in constraints:
                constraint_mask[i, j] = 1.0
        self._constraint_mask: torch.nn.Parameter = torch.nn.Parameter(constraint_mask, requires_grad=False)
        self.include_start_end_transitions: bool = include_start_end_transitions
        if include_start_end_transitions:
            self.start_transitions: torch.nn.Parameter = torch.nn.Parameter(torch.Tensor(num_tags))
            self.end_transitions: torch.nn.Parameter = torch.nn.Parameter(torch.Tensor(num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_normal_(self.transitions)
        if self.include_start_end_transitions:
            torch.nn.init.normal_(self.start_transitions)
            torch.nn.init.normal_(self.end_transitions)

    def _input_likelihood(self, logits: torch.Tensor, transitions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the (batch_size,) denominator term Z(x), per example, for the log-likelihood.
        """
        batch_size, sequence_length, num_tags = logits.size()
        mask = mask.transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()
        if self.include_start_end_transitions:
            alpha: torch.Tensor = self.start_transitions.view(1, num_tags) + logits[0]
        else:
            alpha = logits[0]
        for i in range(1, sequence_length):
            emit_scores: torch.Tensor = logits[i].view(batch_size, 1, num_tags)
            transition_scores: torch.Tensor = transitions.view(1, num_tags, num_tags)
            broadcast_alpha: torch.Tensor = alpha.view(batch_size, num_tags, 1)
            inner: torch.Tensor = broadcast_alpha + emit_scores + transition_scores
            alpha = util.logsumexp(inner, 1) * mask[i].view(batch_size, 1) + alpha * (~mask[i]).view(batch_size, 1)
        if self.include_start_end_transitions:
            stops: torch.Tensor = alpha + self.end_transitions.view(1, num_tags)
        else:
            stops = alpha
        return util.logsumexp(stops)

    def _joint_likelihood(self, logits: torch.Tensor, transitions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the numerator term for the log-likelihood, which is just score(inputs, tags).
        """
        batch_size, sequence_length, _ = logits.data.shape
        logits = logits.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()
        if self.include_start_end_transitions:
            score: torch.Tensor = self.start_transitions.index_select(0, tags[0])
        else:
            score = 0.0
        for i in range(sequence_length - 1):
            current_tag: torch.Tensor = tags[i]
            next_tag: torch.Tensor = tags[i + 1]
            transition_score: torch.Tensor = transitions[current_tag.view(-1), next_tag.view(-1)]
            emit_score: torch.Tensor = logits[i].gather(1, current_tag.view(batch_size, 1)).squeeze(1)
            score = score + transition_score * mask[i + 1] + emit_score * mask[i]
        last_tag_index: torch.Tensor = mask.sum(0).long() - 1
        last_tags: torch.Tensor = tags.gather(0, last_tag_index.view(1, batch_size)).squeeze(0)
        if self.include_start_end_transitions:
            last_transition_score: torch.Tensor = self.end_transitions.index_select(0, last_tags)
        else:
            last_transition_score = 0.0
        last_inputs: torch.Tensor = logits[-1]
        last_input_score: torch.Tensor = last_inputs.gather(1, last_tags.view(-1, 1)).squeeze(1)
        score = score + last_transition_score + last_input_score * mask[-1]
        return score

    def forward(self, inputs: torch.Tensor, tags: torch.Tensor, mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        """
        Computes the log likelihood for the given batch of input sequences (x, y).
        """
        if mask is None:
            mask = torch.ones(*tags.size(), dtype=torch.bool, device=inputs.device)
        else:
            mask = mask.to(torch.bool)
        log_denominator: torch.Tensor = self._input_likelihood(inputs, self.transitions, mask)
        log_numerator: torch.Tensor = self._joint_likelihood(inputs, self.transitions, tags, mask)
        return torch.sum(log_numerator - log_denominator)

    def viterbi_tags(
        self,
        logits: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        top_k: Optional[int] = None
    ) -> Union[List[Tuple[List[int], float]], List[List[Tuple[List[int], float]]]]:
        """
        Uses viterbi algorithm to find most likely tags for the given inputs.
        If constraints are applied, disallows all other transitions.

        Returns a list of results, of the same size as the batch (one result per batch member).
        Each result is a List of length top_k, containing the top K viterbi decodings.
        Each decoding is a tuple (tag_sequence, viterbi_score).

        For backwards compatibility, if top_k is None, then instead returns a flat list of
        tag sequences (the top tag sequence for each batch item).
        """
        if mask is None:
            mask = torch.ones(*logits.shape[:2], dtype=torch.bool, device=logits.device)
        if top_k is None:
            top_k = 1
            flatten_output: bool = True
        else:
            flatten_output = False
        _, max_seq_length, num_tags = logits.size()
        logits = logits.data
        mask = mask.data
        start_tag: int = num_tags
        end_tag: int = num_tags + 1
        transitions: torch.Tensor = torch.full((num_tags + 2, num_tags + 2), -10000.0, device=logits.device)
        constrained_transitions: torch.Tensor = self.transitions * self._constraint_mask[:num_tags, :num_tags] + -10000.0 * (1 - self._constraint_mask[:num_tags, :num_tags])
        transitions[:num_tags, :num_tags] = constrained_transitions.data
        if self.include_start_end_transitions:
            transitions[start_tag, :num_tags] = self.start_transitions.detach() * self._constraint_mask[start_tag, :num_tags].data \
                + -10000.0 * (1 - self._constraint_mask[start_tag, :num_tags].detach())
            transitions[:num_tags, end_tag] = self.end_transitions.detach() * self._constraint_mask[:num_tags, end_tag].data \
                + -10000.0 * (1 - self._constraint_mask[:num_tags, end_tag].detach())
        else:
            transitions[start_tag, :num_tags] = -10000.0 * (1 - self._constraint_mask[start_tag, :num_tags].detach())
            transitions[:num_tags, end_tag] = -10000.0 * (1 - self._constraint_mask[:num_tags, end_tag].detach())
        best_paths: List = []
        tag_sequence: torch.Tensor = torch.empty(max_seq_length + 2, num_tags + 2, device=logits.device)
        for prediction, prediction_mask in zip(logits, mask):
            mask_indices: torch.Tensor = prediction_mask.nonzero(as_tuple=False).squeeze()
            masked_prediction: torch.Tensor = torch.index_select(prediction, 0, mask_indices)
            sequence_length: int = masked_prediction.shape[0]
            tag_sequence.fill_(-10000.0)
            tag_sequence[0, start_tag] = 0.0
            tag_sequence[1:sequence_length + 1, :num_tags] = masked_prediction
            tag_sequence[sequence_length + 1, end_tag] = 0.0
            viterbi_paths, viterbi_scores = util.viterbi_decode(
                tag_sequence=tag_sequence[:sequence_length + 2],
                transition_matrix=transitions,
                top_k=top_k
            )
            top_k_paths: List[Tuple[List[int], float]] = []
            for viterbi_path, viterbi_score in zip(viterbi_paths, viterbi_scores):
                viterbi_path = viterbi_path[1:-1]
                top_k_paths.append((viterbi_path, viterbi_score.item()))
            best_paths.append(top_k_paths)
        if flatten_output:
            return [top_k_paths[0] for top_k_paths in best_paths]
        return best_paths