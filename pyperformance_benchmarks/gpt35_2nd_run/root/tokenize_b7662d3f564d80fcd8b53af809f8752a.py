from typing import Callable, Tuple, List, Any, Generator

def generate_tokens(readline: Callable[[], str]) -> Generator[Tuple[int, str, Tuple[int, int], Tuple[int, int], str], None, None]:
    ...

def tokenize(readline: Callable[[], str], tokeneater: Callable[[int, str, Tuple[int, int], Tuple[int, int], str], Any] = printtoken) -> None:
    ...

def tokenize_loop(readline: Callable[[], str], tokeneater: Callable[[int, str, Tuple[int, int], Tuple[int, int], str], Any]) -> None:
    ...

class Untokenizer:
    def __init__(self) -> None:
        ...

    def add_whitespace(self, start: Tuple[int, int]) -> None:
        ...

    def untokenize(self, iterable: List[Tuple[int, str]]) -> str:
        ...

    def compat(self, token: Tuple[int, str], iterable: List[Tuple[int, str]]) -> None:
        ...

def detect_encoding(readline: Callable[[], str]) -> Tuple[str, List[str]]:
    ...

def untokenize(iterable: List[Tuple[int, str]]) -> str:
    ...

