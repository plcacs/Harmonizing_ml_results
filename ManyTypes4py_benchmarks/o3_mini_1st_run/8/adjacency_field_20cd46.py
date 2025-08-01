from typing import Dict, List, Set, Tuple, Optional, Any
import logging
import textwrap
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.data.fields.field import Field
from allennlp.data.fields.sequence_field import SequenceField
from allennlp.data.vocabulary import Vocabulary

logger = logging.getLogger(__name__)

class AdjacencyField(Field[torch.Tensor]):
    """
    A `AdjacencyField` defines directed adjacency relations between elements
    in a :class:`~allennlp.data.fields.sequence_field.SequenceField`.
    Because it's a labeling of some other field, we take that field as input here
    and use it to determine our padding and other things.

    This field will get converted into an array of shape (sequence_field_length, sequence_field_length),
    where the (i, j)th array element is either a binary flag indicating there is an edge from i to j,
    or an integer label k, indicating there is a label from i to j of type k.

    # Parameters

    indices : `List[Tuple[int, int]]`
    sequence_field : `SequenceField`
        A field containing the sequence that this `AdjacencyField` is labeling.  Most often,
        this is a `TextField`, for tagging edge relations between tokens in a sentence.
    labels : `List[str]`, optional, (default = `None`)
        Optional labels for the edges of the adjacency matrix.
    label_namespace : `str`, optional (default=`'labels'`)
        The namespace to use for converting tag strings into integers.  We convert tag strings to
        integers for you, and this parameter tells the `Vocabulary` object which mapping from
        strings to integers to use (so that "O" as a tag doesn't get the same id as "O" as a word).
    padding_value : `int`, optional (default = `-1`)
        The value to use as padding.
    """
    __slots__ = ['indices', 'labels', 'sequence_field', '_label_namespace', '_padding_value', '_indexed_labels']
    _already_warned_namespaces: Set[str] = set()

    def __init__(
        self,
        indices: List[Tuple[int, int]],
        sequence_field: SequenceField,
        labels: Optional[List[str]] = None,
        label_namespace: str = 'labels',
        padding_value: int = -1
    ) -> None:
        self.indices: List[Tuple[int, int]] = indices
        self.labels: Optional[List[str]] = labels
        self.sequence_field: SequenceField = sequence_field
        self._label_namespace: str = label_namespace
        self._padding_value: int = padding_value
        self._indexed_labels: Optional[List[int]] = None
        self._maybe_warn_for_namespace(label_namespace)
        field_length: int = sequence_field.sequence_length()
        if len(set(indices)) != len(indices):
            raise ConfigurationError(f'Indices must be unique, but found {indices}')
        if not all((0 <= index[1] < field_length and 0 <= index[0] < field_length for index in indices)):
            raise ConfigurationError(f'Label indices and sequence length are incompatible: {indices} and {field_length}')
        if labels is not None and len(indices) != len(labels):
            raise ConfigurationError(f'Labelled indices were passed, but their lengths do not match:  {labels}, {indices}')

    def _maybe_warn_for_namespace(self, label_namespace: str) -> None:
        if not (self._label_namespace.endswith('labels') or self._label_namespace.endswith('tags')):
            if label_namespace not in self._already_warned_namespaces:
                logger.warning(
                    "Your label namespace was '%s'. We recommend you use a namespace ending with 'labels' or 'tags', "
                    "so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for "
                    "`non_padded_namespaces` parameter in Vocabulary.",
                    self._label_namespace
                )
                self._already_warned_namespaces.add(label_namespace)

    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]) -> None:
        if self._indexed_labels is None and self.labels is not None:
            for label in self.labels:
                counter[self._label_namespace][label] += 1

    def index(self, vocab: Vocabulary) -> None:
        if self.labels is not None:
            self._indexed_labels = [vocab.get_token_index(label, self._label_namespace) for label in self.labels]

    def get_padding_lengths(self) -> Dict[str, int]:
        return {'num_tokens': self.sequence_field.sequence_length()}

    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        desired_num_tokens: int = padding_lengths['num_tokens']
        tensor: torch.Tensor = torch.ones(desired_num_tokens, desired_num_tokens) * self._padding_value
        labels: List[int] = self._indexed_labels if self._indexed_labels is not None else [1 for _ in range(len(self.indices))]
        for index, label in zip(self.indices, labels):
            tensor[index] = label
        return tensor

    def empty_field(self) -> 'AdjacencyField':
        empty_list: List[Tuple[int, int]] = []
        adjacency_field: AdjacencyField = AdjacencyField(
            empty_list,
            self.sequence_field.empty_field(),
            padding_value=self._padding_value
        )
        return adjacency_field

    def __str__(self) -> str:
        length: int = self.sequence_field.sequence_length()
        formatted_labels: str = ''.join(('\t\t' + labels + '\n' for labels in textwrap.wrap(repr(self.labels), 100)))
        formatted_indices: str = ''.join(('\t\t' + index + '\n' for index in textwrap.wrap(repr(self.indices), 100)))
        return (f"AdjacencyField of length {length}\n\t\twith indices:\n {formatted_indices}"
                f"\n\t\tand labels:\n {formatted_labels} \n\t\tin namespace: '{self._label_namespace}'.")

    def __len__(self) -> int:
        return len(self.sequence_field)

    def human_readable_repr(self) -> Dict[str, Any]:
        ret: Dict[str, Any] = {'indices': self.indices}
        if self.labels is not None:
            ret['labels'] = self.labels
        return ret