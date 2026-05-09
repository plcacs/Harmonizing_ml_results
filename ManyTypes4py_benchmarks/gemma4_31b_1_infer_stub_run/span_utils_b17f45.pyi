from typing import Callable, List, Set, Tuple, TypeVar, Optional, Union, Sequence
from allennlp.data.tokenizers import Token

TypedSpan = Tuple[int, Tuple[int, int]]
TypedStringSpan = Tuple[str, Tuple[int, int]]

T = TypeVar('T', str, Token)

class InvalidTagSequence(Exception):
    def __init__(self, tag_sequence: Optional[List[str]] = None) -> None: ...
    def __str__(self) -> str: ...

def enumerate_spans(
    sentence: Sequence[T],
    offset: int = 0,
    max_span_width: Optional[int] = None,
    min_span_width: int = 1,
    filter_function: Optional[Callable[[Sequence[T]], bool]] = None
) -> List[Tuple[int, int]]: ...

def bio_tags_to_spans(
    tag_sequence: List[str],
    classes_to_ignore: Optional[List[str]] = None
) -> List[TypedStringSpan]: ...

def iob1_tags_to_spans(
    tag_sequence: List[str],
    classes_to_ignore: Optional[List[str]] = None
) -> List[TypedStringSpan]: ...

def _iob1_start_of_chunk(
    prev_bio_tag: Optional[str],
    prev_conll_tag: Optional[str],
    curr_bio_tag: str,
    curr_conll_tag: str
) -> bool: ...

def bioul_tags_to_spans(
    tag_sequence: List[str],
    classes_to_ignore: Optional[List[str]] = None
) -> List[TypedStringSpan]: ...

def iob1_to_bioul(tag_sequence: List[str]) -> List[str]: ...

def to_bioul(tag_sequence: List[str], encoding: str = 'IOB1') -> List[str]: ...

def bmes_tags_to_spans(
    tag_sequence: List[str],
    classes_to_ignore: Optional[List[str]] = None
) -> List[TypedStringSpan]: ...