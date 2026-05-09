from __future__ import absolute_import
import sys
import re
import itertools as _itertools
from codecs import BOM_UTF8
from typing import NamedTuple, Tuple, Iterator, Iterable, List, Dict, Pattern, Set
from parso.python.token import PythonTokenTypes
from parso.utils import split_lines, PythonVersionInfo, parse_version_string

MAX_UNICODE = '\U0010ffff'
STRING = PythonTokenTypes.STRING
NAME = PythonTokenTypes.NAME
NUMBER = PythonTokenTypes.NUMBER
OP = PythonTokenTypes.OP
NEWLINE = PythonTokenTypes.NEWLINE
INDENT = PythonTokenTypes.INDENT
DEDENT = PythonTokenTypes.DEDENT
ENDMARKER = PythonTokenTypes.ENDMARKER
ERRORTOKEN = PythonTokenTypes.ERRORTOKEN
ERROR_DEDENT = PythonTokenTypes.ERROR_DEDENT
FSTRING_START = PythonTokenTypes.FSTRING_START
FSTRING_STRING = PythonTokenTypes.FSTRING_STRING
FSTRING_END = PythonTokenTypes.FSTRING_END

class TokenCollection(NamedTuple):
    pass

BOM_UTF8_STRING = BOM_UTF8.decode('utf-8')
_token_collection_cache: Dict[Tuple[int, int], TokenCollection] = {}

def group(*choices: Pattern, capture: bool = False, **kwargs) -> str:
    assert not kwargs
    start: str = '('
    if not capture:
        start += '?:'
    return start + '|'.join(map(str, choices)) + ')'

def maybe(*choices: Pattern) -> str:
    return group(*choices) + '?'

def _all_string_prefixes(*, include_fstring: bool = False, only_fstring: bool = False) -> Set[str]:
    ...

def _compile(expr: str) -> re.Pattern:
    return re.compile(expr, re.UNICODE)

class Token(NamedTuple):
    @property
    def end_pos(self) -> Tuple[int, int]:
        ...

class PythonToken(Token):
    def __repr__(self) -> str:
        return 'TokenInfo(type=%s, string=%r, start_pos=%r, prefix=%r)' % self._replace(type=self.type.name)

class FStringNode:
    def __init__(self, quote: str):
        ...

    def open_parentheses(self, character: str) -> None:
        ...

    def close_parentheses(self, character: str) -> None:
        ...

    def allow_multiline(self) -> bool:
        ...

    def is_in_expr(self) -> bool:
        ...

    def is_in_format_spec(self) -> bool:
        ...

def _close_fstring_if_necessary(fstring_stack: List[FStringNode], string: str, line_nr: int, column: int, additional_prefix: str) -> Tuple[PythonToken, str, int]:
    ...

def _find_fstring_string(endpats: Dict[str, re.Pattern], fstring_stack: List[FStringNode], line: str, lnum: int, pos: int) -> Tuple[str, int]:
    ...

def tokenize(code: str, *, version_info: PythonVersionInfo, start_pos: Tuple[int, int] = (1, 0)) -> Iterator[PythonToken]:
    ...

def _print_tokens(func) -> Iterator[PythonToken]:
    ...

def tokenize_lines(lines: Iterable[str], *, version_info: PythonVersionInfo, indents: List[int] = None, start_pos: Tuple[int, int] = (1, 0), is_first_token: bool = True) -> Iterator[PythonToken]:
    ...

def _split_illegal_unicode_name(token: str, start_pos: Tuple[int, int], prefix: str) -> Iterator[PythonToken]:
    ...

if __name__ == '__main__':
    path: str = sys.argv[1]
    with open(path) as f:
        code: str = f.read()
    for token in tokenize(code, version_info=parse_version_string('3.10')):
        print(token)
