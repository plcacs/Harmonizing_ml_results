from typing import Final, Optional, Union, Callable
from blib2to3.pgen2.grammar import Grammar
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

def untokenize(iterable: Iterable[TokenInfo]) -> str:
    ...

def detect_encoding(readline: Callable[[], str]) -> Tuple[str, list]:
    ...

class FStringState:
    def __init__(self) -> None:
        ...

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

def generate_tokens(readline: Callable[[], str], grammar: Optional[Grammar] = None) -> Iterable[GoodTokenInfo]:
    ...
