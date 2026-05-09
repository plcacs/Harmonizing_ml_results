from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)
from org.transcrypt.stubs.browser import RegExp


T = int
TEMPLATE = int
I = int
IGNORECASE = int
L = int
LOCALE = int
M = int
MULTILINE = int
S = int
DOTALL = int
U = int
UNICODE = int
X = int
VERBOSE = int
DEBUG = int
A = int
ASCII = int
Y = int
STICKY = int
G = int
GLOBAL = int
J = int
JSSTRICT = int


class error(Exception):
    pattern: Optional[str]
    flags: int
    pos: Optional[int]

    def __init__(self, msg: str, error: Any, pattern: Optional[str] = ..., flags: int = ..., pos: Optional[int] = ...) -> None:
        ...


class ReIndexError(IndexError):
    def __init__(self) -> None:
        ...


class Match:
    _obj: List[Optional[str]]
    _pos: int
    _endpos: int
    _re: Any
    _string: str
    _namedGroups: Optional[Dict[str, int]]
    _lastindex: Optional[int]
    _lastgroup: Optional[str]

    def __init__(self, mObj: List[Optional[str]], string: str, pos: int, endpos: int, rObj: Any, namedGroups: Optional[Dict[str, int]] = ...) -> None:
        ...

    @property
    def pos(self) -> int:
        ...

    @property
    def endpos(self) -> int:
        ...

    @property
    def re(self) -> Any:
        ...

    @property
    def string(self) -> str:
        ...

    @property
    def lastgroup(self) -> Optional[str]:
        ...

    @property
    def lastindex(self) -> Optional[int]:
        ...

    def expand(self, template: str) -> None:
        ...

    def group(self, *args: Union[int, str]) -> Union[str, Tuple[str, ...]]:
        ...

    def groups(self, default: Any = ...) -> Tuple[Any, ...]:
        ...

    def groupdict(self, default: Any = ...) -> Dict[str, Any]:
        ...

    def start(self, group: Union[int, str] = ...) -> int:
        ...

    def end(self, group: Union[int, str] = ...) -> int:
        ...

    def span(self, group: Union[int, str] = ...) -> Tuple[int, int]:
        ...


class Regex:
    _flags: int
    _jspattern: str
    _pypattern: str
    _groups: int
    _groupindex: Optional[Dict[str, int]]

    def __init__(self, pattern: str, flags: int) -> None:
        ...

    @property
    def pattern(self) -> str:
        ...

    @property
    def flags(self) -> int:
        ...

    @property
    def groups(self) -> int:
        ...

    @property
    def groupindex(self) -> Optional[Dict[str, int]]:
        ...

    def _compileWrapper(self, pattern: str, flags: int = ...) -> Tuple[str, RegExp]:
        ...

    def _convertFlags(self, flags: int) -> str:
        ...

    def _getTargetStr(self, string: str, pos: int, endpos: Optional[int]) -> str:
        ...

    def _patternHasCaptures(self) -> bool:
        ...

    def search(self, string: str, pos: int = ..., endpos: Optional[int] = ...) -> Optional[Match]:
        ...

    def match(self, string: str, pos: int = ..., endpos: Optional[int] = ...) -> Optional[Match]:
        ...

    def fullmatch(self, string: str, pos: int = ..., endpos: Optional[int] = ...) -> Optional[Match]:
        ...

    def split(self, string: str, maxsplit: int = ...) -> List[str]:
        ...

    def _findAllMatches(self, string: str, pos: int = ..., endpos: Optional[int] = ...) -> Iterable[Any]:
        ...

    def findall(self, string: str, pos: int = ..., endpos: Optional[int] = ...) -> Iterable[Union[str, Tuple[str, ...]]]:
        ...

    def finditer(self, string: str, pos: int, endpos: Optional[int] = ...) -> Iterator[Match]:
        ...

    def sub(self, repl: Union[str, Callable[[Match], str]], string: str, count: int = ...) -> str:
        ...

    def subn(self, repl: Union[str, Callable[[Match], str]], string: str, count: int = ...) -> Tuple[str, int]:
        ...


class PyRegExp(Regex):
    _nsplits: int
    _capgroups: int

    def __init__(self, pyPattern: str, flags: int) -> None:
        ...


def compile(pattern: str, flags: int = ...) -> Regex:
    ...


def search(pattern: str, string: str, flags: int = ...) -> Optional[Match]:
    ...


def match(pattern: str, string: str, flags: int = ...) -> Optional[Match]:
    ...


def fullmatch(pattern: str, string: str, flags: int = ...) -> Optional[Match]:
    ...


def split(pattern: str, string: str, maxsplit: int = ..., flags: int = ...) -> List[str]:
    ...


def findall(pattern: str, string: str, flags: int = ...) -> Iterable[Union[str, Tuple[str, ...]]]:
    ...


def finditer(pattern: str, string: str, flags: int = ...) -> Iterator[Match]:
    ...


def sub(pattern: str, repl: Union[str, Callable[[Match], str]], string: str, count: int = ..., flags: int = ...) -> str:
    ...


def subn(pattern: str, repl: Union[str, Callable[[Match], str]], string: str, count: int = ..., flags: int = ...) -> Tuple[str, int]:
    ...


def escape(string: str) -> str:
    ...


def purge() -> None:
    ...