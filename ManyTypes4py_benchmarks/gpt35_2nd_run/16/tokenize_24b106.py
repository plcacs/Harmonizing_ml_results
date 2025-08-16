from typing import Final, Optional, Union, Callable
from blib2to3.pgen2.token import ASYNC, AWAIT, COMMENT, DEDENT, ENDMARKER, ERRORTOKEN, FSTRING_END, FSTRING_MIDDLE, FSTRING_START, INDENT, LBRACE, NAME, NEWLINE, NL, NUMBER, OP, RBRACE, STRING
from typing import Tuple

Coord = Tuple[int, int]
TokenEater = Callable[[int, str, Coord, Coord, str], None]
GoodTokenInfo = Tuple[int, str, Coord, Coord, str]
TokenInfo = Union[Tuple[int, str], GoodTokenInfo]

def tokenize(readline: Callable[[], str], tokeneater: TokenEater = printtoken) -> None:
    ...

def tokenize_loop(readline: Callable[[], str], tokeneater: TokenEater) -> None:
    ...

class Untokenizer:
    def add_whitespace(self, start: Coord) -> None:
        ...

    def untokenize(self, iterable: Iterable) -> str:
        ...

def detect_encoding(readline: Callable[[], str]) -> Tuple[str, list]:
    ...

def untokenize(iterable: Iterable) -> str:
    ...

def is_fstring_start(token: str) -> bool:
    ...

def _split_fstring_start_and_middle(token: str) -> Tuple[str, str]:
    ...

class FStringState:
    def is_in_fstring_expression(self) -> bool:
        ...

    def current(self) -> int:
        ...

    def enter_fstring(self) -> None:
        ...

    def leave_fstring(self) -> None:
        ...

    def consume_lbrace(self) -> None:
        ...

    def consume_rbrace(self) -> None:
        ...

    def consume_colon(self) -> None:
        ...

def generate_tokens(readline: Callable[[], str], grammar: Optional[Grammar] = None) -> Iterable[TokenInfo]:
    ...
