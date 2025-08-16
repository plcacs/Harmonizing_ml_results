from typing import Optional

def _check_is_single_character(c: str) -> str:
    ...

class OneCharStringStrategy(SearchStrategy):
    def __init__(self, intervals: IntervalSet, force_repr: Optional[str] = None):
        ...

    @classmethod
    def from_characters_args(cls, *, codec: Optional[str] = None, min_codepoint: Optional[int] = None, max_codepoint: Optional[int] = None, categories: Optional[Iterable[str]] = None, exclude_characters: Optional[Iterable[str]] = None, include_characters: Optional[Iterable[str]] = None) -> 'OneCharStringStrategy':
        ...

    @classmethod
    def from_alphabet(cls, alphabet: Union[str, SearchStrategy]) -> 'OneCharStringStrategy':
        ...

    def __repr__(self) -> str:
        ...

    def do_draw(self, data) -> str:
        ...

class TextStrategy(ListStrategy):
    def do_draw(self, data) -> str:
        ...

    def __repr__(self) -> str:
        ...

    def filter(self, condition) -> 'TextStrategy':
        ...

def _string_filter_rewrite(self, kind, condition) -> Optional['TextStrategy']:
    ...

def _identifier_characters() -> Tuple[IntervalSet, IntervalSet]:
    ...

class BytesStrategy(SearchStrategy):
    def __init__(self, min_size: int, max_size: Optional[int]):
        ...

    def do_draw(self, data) -> bytes:
        ...

    def filter(self, condition) -> 'BytesStrategy':
        ...
