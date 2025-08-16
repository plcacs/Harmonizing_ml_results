from typing import Dict, List, Set, Tuple, Optional
import logging
import textwrap
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.data.fields.field import Field
from allennlp.data.fields.sequence_field import SequenceField
from allennlp.data.vocabulary import Vocabulary
logger: logging.Logger = logging.getLogger(__name__)

class AdjacencyField(Field[torch.Tensor]):
    __slots__: List[str] = ['indices', 'labels', 'sequence_field', '_label_namespace', '_padding_value', '_indexed_labels']
    _already_warned_namespaces: Set[str] = set()

    def __init__(self, indices: List[Tuple[int, int]], sequence_field: SequenceField, labels: Optional[List[str] = None, label_namespace: str = 'labels', padding_value: int = -1) -> None:
    
    def _maybe_warn_for_namespace(self, label_namespace: str) -> None:
    
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]) -> None:
    
    def index(self, vocab: Vocabulary) -> None:
    
    def get_padding_lengths(self) -> Dict[str, int]:
    
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
    
    def empty_field(self) -> 'AdjacencyField':
    
    def __str__(self) -> str:
    
    def __len__(self) -> int:
    
    def human_readable_repr(self) -> Dict[str, List[Tuple[int, int]]]:
