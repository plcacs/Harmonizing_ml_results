from typing import NamedTuple, Tuple, Iterator, Iterable, List, Dict, Pattern, Set

MAX_UNICODE: str = '\U0010ffff'
STRING: int = PythonTokenTypes.STRING
NAME: int = PythonTokenTypes.NAME
NUMBER: int = PythonTokenTypes.NUMBER
OP: int = PythonTokenTypes.OP
NEWLINE: int = PythonTokenTypes.NEWLINE
INDENT: int = PythonTokenTypes.INDENT
DEDENT: int = PythonTokenTypes.DEDENT
ENDMARKER: int = PythonTokenTypes.ENDMARKER
ERRORTOKEN: int = PythonTokenTypes.ERRORTOKEN
ERROR_DEDENT: int = PythonTokenTypes.ERROR_DEDENT
FSTRING_START: int = PythonTokenTypes.FSTRING_START
FSTRING_STRING: int = PythonTokenTypes.FSTRING_STRING
FSTRING_END: int = PythonTokenTypes.FSTRING_END

class TokenCollection(NamedTuple):
    pass

BOM_UTF8_STRING: str = BOM_UTF8.decode('utf-8')
_token_collection_cache: Dict[Tuple[int, int], TokenCollection] = {}

def group(*choices, capture: bool = False) -> str:
    assert not kwargs
    start: str = '('
    if not capture:
        start += '?:'
    return start + '|'.join(choices) + ')'

def maybe(*choices) -> str:
    return group(*choices) + '?'

def _all_string_prefixes(include_fstring: bool = False, only_fstring: bool = False) -> Set[str]:
    ...

def _compile(expr: str) -> Pattern:
    return re.compile(expr, re.UNICODE)

def _get_token_collection(version_info: Tuple[int, int]) -> TokenCollection:
    ...

class Token(NamedTuple):
    @property
    def end_pos(self) -> Tuple[int, int]:
        ...

class PythonToken(Token):
    def __repr__(self) -> str:
        ...

class FStringNode:
    def __init__(self, quote: str) -> None:
        ...

def _close_fstring_if_necessary(fstring_stack: List[FStringNode], string: str, line_nr: int, column: int, additional_prefix: str) -> Tuple[PythonToken, str, int]:
    ...

def _find_fstring_string(endpats: Dict[str, Pattern], fstring_stack: List[FStringNode], line: str, lnum: int, pos: int) -> Tuple[str, int]:
    ...

def tokenize(code: str, version_info: Tuple[int, int], start_pos: Tuple[int, int] = (1, 0)) -> Iterator[PythonToken]:
    ...

def _print_tokens(func: Callable) -> Callable:
    ...

def tokenize_lines(lines: Iterable[str], version_info: Tuple[int, int], indents: List[int] = None, start_pos: Tuple[int, int] = (1, 0), is_first_token: bool = True) -> Iterator[PythonToken]:
    ...

def _split_illegal_unicode_name(token: str, start_pos: Tuple[int, int], prefix: str) -> Iterator[PythonToken]:
    ...

if __name__ == '__main__':
    path: str = sys.argv[1]
    with open(path) as f:
        code: str = f.read()
    for token in tokenize(code, version_info=parse_version_string('3.10')):
        print(token)
