from typing import Tuple

def chars_not_in_alphabet(alphabet: Any, string: str) -> Tuple[str, ...]:
    ...

class CharactersBuilder:
    def __init__(self, *, negate: bool = False, flags: int = 0, alphabet: Any):
        ...

    @property
    def strategy(self) -> Any:
        ...

    def add_category(self, category: int) -> None:
        ...

    def add_char(self, c: str) -> None:
        ...

class BytesBuilder(CharactersBuilder):
    def __init__(self, *, negate: bool = False, flags: int = 0):
        ...

    @property
    def strategy(self) -> Any:
        ...

    def add_category(self, category: int) -> None:
        ...

def base_regex_strategy(regex: Any, parsed: Any = None, alphabet: Any = None) -> Any:
    ...

def regex_strategy(regex: Any, fullmatch: bool, *, alphabet: Any, _temp_jsonschema_hack_no_end_newline: bool = False) -> Any:
    ...

def _strategy(codes: Any, context: Any, is_unicode: bool, *, alphabet: Any) -> Any:
    ...
