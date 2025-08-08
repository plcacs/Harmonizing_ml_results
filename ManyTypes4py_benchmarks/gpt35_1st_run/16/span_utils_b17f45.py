from typing import Callable, List, Set, Tuple, TypeVar, Optional
import warnings
from allennlp.common.checks import ConfigurationError
from allennlp.data.tokenizers import Token

TypedSpan = Tuple[int, Tuple[int, int]]
TypedStringSpan = Tuple[str, Tuple[int, int]]

class InvalidTagSequence(Exception):

    def __init__(self, tag_sequence: Optional[List[str]] = None) -> None:
        super().__init__()
        self.tag_sequence = tag_sequence

    def __str__(self) -> str:
        return ' '.join(self.tag_sequence)

T = TypeVar('T', str, Token)

def enumerate_spans(sentence: List[T], offset: int = 0, max_span_width: Optional[int] = None, min_span_width: int = 1, filter_function: Optional[Callable[[List[T]], bool]] = None) -> List[TypedSpan]:
    ...

def bio_tags_to_spans(tag_sequence: List[str], classes_to_ignore: Optional[List[str]] = None) -> List[TypedStringSpan]:
    ...

def iob1_tags_to_spans(tag_sequence: List[str], classes_to_ignore: Optional[List[str]] = None) -> List[TypedStringSpan]:
    ...

def _iob1_start_of_chunk(prev_bio_tag: str, prev_conll_tag: str, curr_bio_tag: str, curr_conll_tag: str) -> bool:
    ...

def bioul_tags_to_spans(tag_sequence: List[str], classes_to_ignore: Optional[List[str]] = None) -> List[TypedStringSpan]:
    ...

def iob1_to_bioul(tag_sequence: List[str]) -> List[str]:
    ...

def to_bioul(tag_sequence: List[str], encoding: str = 'IOB1') -> List[str]:
    ...

def bmes_tags_to_spans(tag_sequence: List[str], classes_to_ignore: Optional[List[str]] = None) -> List[TypedStringSpan]:
    ...
