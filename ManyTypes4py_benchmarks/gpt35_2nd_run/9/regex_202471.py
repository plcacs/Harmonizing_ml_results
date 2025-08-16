from typing import Tuple

def chars_not_in_alphabet(alphabet: OneCharStringStrategy, string: str) -> Tuple[str, ...]:
    ...

class CharactersBuilder:
    def __init__(self, *, negate: bool = False, flags: int = 0, alphabet: OneCharStringStrategy):
        ...

    @property
    def strategy(self) -> OneCharStringStrategy:
        ...

    def add_category(self, category: int) -> None:
        ...

    def add_char(self, c: str) -> None:
        ...

class BytesBuilder(CharactersBuilder):
    def __init__(self, *, negate: bool = False, flags: int = 0):
        ...

    @property
    def strategy(self) -> st.SearchStrategy:
        ...

    def add_category(self, category: int) -> None:
        ...

def _strategy(codes, context, is_unicode, *, alphabet: OneCharStringStrategy):
    ...
