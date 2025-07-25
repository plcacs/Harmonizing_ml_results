from typing import Dict, Union, Sequence, Set, Optional, cast
import logging
import torch
from allennlp.data.fields.field import Field
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)

class MultiLabelField(Field[torch.Tensor]):
    __slots__ = ['labels', '_label_namespace', '_label_ids', '_num_labels']
    _already_warned_namespaces: Set[str] = set()

    def __init__(self, 
                 labels: Sequence[Union[str, int]], 
                 label_namespace: str = 'labels', 
                 skip_indexing: bool = False, 
                 num_labels: Optional[int] = None) -> None:
        self.labels: Sequence[Union[str, int]] = labels
        self._label_namespace: str = label_namespace
        self._label_ids: Optional[Sequence[int]] = None
        self._maybe_warn_for_namespace(label_namespace)
        self._num_labels: Optional[int] = num_labels
        if skip_indexing and self.labels:
            if not all((isinstance(label, int) for label in labels)):
                raise ConfigurationError('In order to skip indexing, your labels must be integers. Found labels = {}'.format(labels))
            if not num_labels:
                raise ConfigurationError("In order to skip indexing, num_labels can't be None.")
            if not all((cast(int, label) < num_labels for label in labels)):
                raise ConfigurationError('All labels should be < num_labels. Found num_labels = {} and labels = {} '.format(num_labels, labels))
            self._label_ids = labels
        elif not all((isinstance(label, str) for label in labels)):
            raise ConfigurationError('MultiLabelFields expects string labels if skip_indexing=False. Found labels: {}'.format(labels))

    def _maybe_warn_for_namespace(self, label_namespace: str) -> None:
        if not (label_namespace.endswith('labels') or label_namespace.endswith('tags')):
            if label_namespace not in self._already_warned_namespaces:
                logger.warning("Your label namespace was '%s'. We recommend you use a namespace ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` parameter in Vocabulary.", self._label_namespace)
                self._already_warned_namespaces.add(label_namespace)

    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]) -> None:
        if self._label_ids is None:
            for label in self.labels:
                counter[self._label_namespace][label] += 1

    def index(self, vocab: Vocabulary) -> None:
        if self._label_ids is None:
            self._label_ids = [vocab.get_token_index(label, self._label_namespace) for label in self.labels]
        if not self._num_labels:
            self._num_labels = vocab.get_vocab_size(self._label_namespace)

    def get_padding_lengths(self) -> Dict[str, int]:
        return {}

    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        tensor = torch.zeros(self._num_labels, dtype=torch.long)
        if self._label_ids:
            tensor.scatter_(0, torch.LongTensor(self._label_ids), 1)
        return tensor

    def empty_field(self) -> 'MultiLabelField':
        return MultiLabelField([], self._label_namespace, skip_indexing=True, num_labels=self._num_labels)

    def __str__(self) -> str:
        return f"MultiLabelField with labels: {self.labels} in namespace: '{self._label_namespace}'.'"

    def __len__(self) -> int:
        return 1

    def human_readable_repr(self) -> Sequence[Union[str, int]]:
        return self.labels
