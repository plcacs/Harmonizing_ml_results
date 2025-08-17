# Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006 Python Software Foundation.
# All rights reserved.

"""Tokenization help for Python programs.

generate_tokens(readline) is a generator that breaks a stream of
text into Python tokens.  It accepts a readline-like method which is called
repeatedly to get the next line of input (or "" for EOF).  It generates
5-tuples with these members:

    the token type (see token.py)
    the token (a string)
    the starting (row, column) indices of the token (a 2-tuple of ints)
    the ending (row, column) indices of the token (a 2-tuple of ints)
    the original line (string)

It is designed to match the working of the Python tokenizer exactly, except
that it produces COMMENT tokens for comments and gives type OP for all
operators

Older entry points
    tokenize_loop(readline, tokeneater)
    tokenize(readline, tokeneater=printtoken)
are the same, except instead of generating tokens, tokeneater is a callback
function to which the 5 fields described above are passed as 5 arguments,
each time a new token is found."""

import builtins
import sys
from collections.abc import Callable, Iterable, Iterator
from re import Pattern
from typing import Final, Optional, Union, Any, Dict, List, Set, Tuple

from blib2to3.pgen2.grammar import Grammar
from blib2to3.pgen2.token import (
    ASYNC,
    AWAIT,
    COMMENT,
    DEDENT,
    ENDMARKER,
    ERRORTOKEN,
    FSTRING_END,
    FSTRING_MIDDLE,
    FSTRING_START,
    INDENT,
    LBRACE,
    NAME,
    NEWLINE,
    NL,
    NUMBER,
    OP,
    RBRACE,
    STRING,
    tok_name,
)

__author__ = "Ka-Ping Yee <ping@lfw.org>"
__credits__ = "GvR, ESR, Tim Peters, Thomas Wouters, Fred Drake, Skip Montanaro"

import re
from codecs import BOM_UTF8, lookup

from . import token

__all__ = [x for x in dir(token) if x[0] != "_"] + [
    "tokenize",
    "generate_tokens",
    "untokenize",
]
del token


def group(*choices: str) -> str:
    return "(" + "|".join(choices) + ")"


def any(*choices: str) -> str:
    return group(*choices) + "*"


def maybe(*choices: str) -> str:
    return group(*choices) + "?"


def _combinations(*l: str) -> Set[str]:
    return {x + y for x in l for y in l + ("",) if x.casefold() != y.casefold()}


Whitespace: Final[str] = r"[ \f\t]*"
Comment: Final[str] = r"#[^\r\n]*"
Ignore: Final[str] = Whitespace + any(r"\\\r?\n" + Whitespace) + maybe(Comment)
Name: Final[str] = r"[^\s#\(\)\[\]\{\}+\-*/!@$%^&=|;:'\",\.<>/?`~\\]+"

Binnumber: Final[str] = r"0[bB]_?[01]+(?:_[01]+)*"
Hexnumber: Final[str] = r"0[xX]_?[\da-fA-F]+(?:_[\da-fA-F]+)*[lL]?"
Octnumber: Final[str] = r"0[oO]?_?[0-7]+(?:_[0-7]+)*[lL]?"
Decnumber: Final[str] = group(r"[1-9]\d*(?:_\d+)*[lL]?", "0[lL]?")
Intnumber: Final[str] = group(Binnumber, Hexnumber, Octnumber, Decnumber)
Exponent: Final[str] = r"[eE][-+]?\d+(?:_\d+)*"
Pointfloat: Final[str] = group(r"\d+(?:_\d+)*\.(?:\d+(?:_\d+)*)?", r"\.\d+(?:_\d+)*") + maybe(
    Exponent
)
Expfloat: Final[str] = r"\d+(?:_\d+)*" + Exponent
Floatnumber: Final[str] = group(Pointfloat, Expfloat)
Imagnumber: Final[str] = group(r"\d+(?:_\d+)*[jJ]", Floatnumber + r"[jJ]")
Number: Final[str] = group(Imagnumber, Floatnumber, Intnumber)

Single: Final[str] = r"(?:\\.|[^'\\])*'"
Double: Final[str] = r'(?:\\.|[^"\\])*"'
Single3: Final[str] = r"(?:\\.|'(?!'')|[^'\\])*'''"
Double3: Final[str] = r'(?:\\.|"(?!"")|[^"\\])*"""'
_litprefix: Final[str] = r"(?:[uUrRbB]|[rR][bB]|[bBuU][rR])?"
_fstringlitprefix: Final[str] = r"(?:rF|FR|Fr|fr|RF|F|rf|f|Rf|fR)"
Triple: Final[str] = group(
    _litprefix + "'''",
    _litprefix + '"""',
    _fstringlitprefix + '"""',
    _fstringlitprefix + "'''",
)

SingleLbrace: Final[str] = r"(?:\\N{|{{|\\'|[^\n'{])*(?<!\\N)({)(?!{)"
DoubleLbrace: Final[str] = r'(?:\\N{|{{|\\"|[^\n"{])*(?<!\\N)({)(?!{)'

Single3Lbrace: Final[str] = r"(?:\\N{|{{|\\'|'(?!'')|[^'{])*(?<!\\N){(?!{)"
Double3Lbrace: Final[str] = r'(?:\\N{|{{|\\"|"(?!"")|[^"{])*(?<!\\N){(?!{)'

Bang: Final[str] = Whitespace + group("!") + r"(?!=)"
bang: Final[Pattern[str]] = re.compile(Bang)
Colon: Final[str] = Whitespace + group(":")
colon: Final[Pattern[str]] = re.compile(Colon)

FstringMiddleAfterColon: Final[str] = group(Whitespace + r".*?") + group("{", "}")
fstring_middle_after_colon: Final[Pattern[str]] = re.compile(FstringMiddleAfterColon)

Operator: Final[str] = group(
    r"\*\*=?",
    r">>=?",
    r"<<=?",
    r"<>",
    r"!=",
    r"//=?",
    r"->",
    r"[+\-*/%&@|^=<>:]=?",
    r"~",
)

Bracket: Final[str] = "[][(){}]"
Special: Final[str] = group(r"\r?\n", r"[:;.,`@]")
Funny: Final[str] = group(Operator, Bracket, Special)

_string_middle_single: Final[str] = r"(?:[^\n'\\]|\\.)*"
_string_middle_double: Final[str] = r'(?:[^\n"\\]|\\.)*'

_fstring_middle_single: Final[str] = SingleLbrace
_fstring_middle_double: Final[str] = DoubleLbrace

ContStr: Final[str] = group(
    _litprefix + "'" + _string_middle_single + group("'", r"\\\r?\n"),
    _litprefix + '"' + _string_middle_double + group('"', r"\\\r?\n"),
    group(_fstringlitprefix + "'") + _fstring_middle_single,
    group(_fstringlitprefix + '"') + _fstring_middle_double,
    group(_fstringlitprefix + "'") + _string_middle_single + group("'", r"\\\r?\n"),
    group(_fstringlitprefix + '"') + _string_middle_double + group('"', r"\\\r?\n"),
)
PseudoExtras: Final[str] = group(r"\\\r?\n", Comment, Triple)
PseudoToken: Final[str] = Whitespace + group(PseudoExtras, Number, Funny, ContStr, Name)

pseudoprog: Final[Pattern[str]] = re.compile(PseudoToken, re.UNICODE)

singleprog: Final[Pattern[str]] = re.compile(Single)
singleprog_plus_lbrace: Final[Pattern[str]] = re.compile(group(SingleLbrace, Single))
doubleprog: Final[Pattern[str]] = re.compile(Double)
doubleprog_plus_lbrace: Final[Pattern[str]] = re.compile(group(DoubleLbrace, Double))

single3prog: Final[Pattern[str]] = re.compile(Single3)
single3prog_plus_lbrace: Final[Pattern[str]] = re.compile(group(Single3Lbrace, Single3))
double3prog: Final[Pattern[str]] = re.compile(Double3)
double3prog_plus_lbrace: Final[Pattern[str]] = re.compile(group(Double3Lbrace, Double3))

_strprefixes: Final[Set[str]] = _combinations("r", "R", "b", "B") | {"u", "U", "ur", "uR", "Ur", "UR"}
_fstring_prefixes: Final[Set[str]] = _combinations("r", "R", "f", "F") - {"r", "R"}

endprogs: Final[Dict[str, Pattern[str]]] = {
    "'": singleprog,
    '"': doubleprog,
    "'''": single3prog,
    '"""': double3prog,
    **{f"{prefix}'": singleprog for prefix in _strprefixes},
    **{f'{prefix}"': doubleprog for prefix in _strprefixes},
    **{f"{prefix}'": singleprog_plus_lbrace for prefix in _fstring_prefixes},
    **{f'{prefix}"': doubleprog_plus_lbrace for prefix in _fstring_prefixes},
    **{f"{prefix}'''": single3prog for prefix in _strprefixes},
    **{f'{prefix}"""': double3prog for prefix in _strprefixes},
    **{f"{prefix}'''": single3prog_plus_lbrace for prefix in _fstring_prefixes},
    **{f'{prefix}"""': double3prog_plus_lbrace for prefix in _fstring_prefixes},
}

triple_quoted: Final[Set[str]] = (
    {"'''", '"""'}
    | {f"{prefix}'''" for prefix in _strprefixes | _fstring_prefixes}
    | {f'{prefix}"""' for prefix in _strprefixes | _fstring_prefixes}
)
single_quoted: Final[Set[str]] = (
    {"'", '"'}
    | {f"{prefix}'" for prefix in _strprefixes | _fstring_prefixes}
    | {f'{prefix}"' for prefix in _strprefixes | _fstring_prefixes}
)
fstring_prefix: Final[Tuple[str, ...]] = tuple(
    {f"{prefix}'" for prefix in _fstring_prefixes}
    | {f'{prefix}"' for prefix in _fstring_prefixes}
    | {f"{prefix}'''" for prefix in _fstring_prefixes}
    | {f'{prefix}"""' for prefix in _fstring_prefixes}
)

tabsize: Final[int] = 8


class TokenError(Exception):
    pass


class StopTokenizing(Exception):
    pass


Coord: Final = Tuple[int, int]


def printtoken(
    type: int, token: str, srow_col: Coord, erow_col: Coord, line: str
) -> None:
    (srow, scol) = srow_col
    (erow, ecol) = erow_col
    print(
        "%d,%d-%d,%d:\t%s\t%s" % (srow, scol, erow, ecol, tok_name[type], repr(token))
    )


TokenEater: Final = Callable[[int, str, Coord, Coord, str], None]


def tokenize(readline: Callable[[], str], tokeneater: TokenEater = printtoken) -> None:
    try:
        tokenize_loop(readline, tokeneater)
    except StopTokenizing:
        pass


def tokenize_loop(readline: Callable[[], str], tokeneater: TokenEater) -> None:
    for token_info in generate_tokens(readline):
        tokeneater(*token_info)


GoodTokenInfo: Final = Tuple[int, str, Coord, Coord, str]
TokenInfo: Final = Union[Tuple[int, str], GoodTokenInfo]


class Untokenizer:
    tokens: List[str]
    prev_row: int
    prev_col: int

    def __init__(self) -> None:
        self.tokens = []
        self.prev_row = 1
        self.prev_col = 0

    def add_whitespace(self, start: Coord) -> None:
        row, col = start
        assert row <= self.prev_row
        col_offset = col - self.prev_col
        if col_offset:
            self.tokens.append(" " * col_offset)

    def untokenize(self, iterable: Iterable[TokenInfo]) -> str:
        for t in iterable:
            if len(t) == 2:
                self.compat(t, iterable)
                break
            tok_type, token, start, end, line = t
            self.add_whitespace(start)
            self.tokens.append(token)
            self.prev_row, self.prev_col = end
            if tok_type in (NEWLINE, NL):
                self.prev_row += 1
                self.prev_col = 0
        return "".join(self.tokens)

    def compat(self, token: Tuple[int, str], iterable: Iterable[TokenInfo]) -> None:
        startline = False
        indents: List[str] = []
        toks_append = self.tokens.append
        toknum, tokval = token
        if toknum in (NAME, NUMBER):
            tokval += " "
        if toknum in (NEWLINE, NL):
            startline = True
        for tok in iterable:
            toknum, tokval = tok[:2]

            if toknum in (NAME, NUMBER, ASYNC, AWAIT):
                tokval += " "

            if toknum == INDENT:
                indents.append(tokval)
                continue
            elif toknum == DEDENT:
                indents.pop()
                continue
            elif toknum in (NEWLINE, NL):
                startline = True
            elif startline and indents:
                toks_append(indents[-1])
                startline = False
            toks_append(tokval)


cookie_re: Final[Pattern[str]] = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)", re.ASCII)
blank_re: Final[Pattern[bytes]] = re.compile(rb"^[ \t\f]*(?:[#\r\n]|$)", re.ASCII)


def _get_normal_name(orig_enc: str) -> str:
    enc = orig_enc[:12].lower().replace("_", "-")
    if enc == "utf-8" or enc.startswith("utf-8-"):
        return "utf-8"
    if enc in ("latin-1", "iso-8859-1", "iso-latin-1") or enc.startswith(
        ("latin-1-", "iso-8859-1-", "iso-latin-1-")
    ):
        return "iso-8859-1"
    return orig_enc


def detect_encoding(readline: Callable[[], bytes]) -> Tuple[str, List[bytes]]:
    bom_found = False
    encoding = None
    default = "utf-8"

    def read_or_stop() -> bytes:
        try:
            return readline()
        except StopIteration:
            return b""

    def find_cookie(line: bytes) -> Optional[str]:
        try:
            line_string = line.decode("ascii")
        except UnicodeDecodeError:
            return None
        match = cookie_re.match(line_string)
        if not match:
            return None
        encoding = _get_normal_name(match.group(1))
        try:
            codec = lookup(encoding)
        except LookupError:
            raise SyntaxError("unknown encoding: " + encoding)

        if bom_found:
            if codec.name != "utf-8":
                raise SyntaxError("encoding problem: utf-8")
            encoding += "-sig"
        return encoding

    first = read_or_stop()
    if first.startswith(BOM_UTF8):
        bom_found = True
        first = first[3:]
        default = "utf-8-sig"
    if not first:
        return default, []

    encoding = find_cookie(first)
    if encoding:
        return encoding, [first]
    if not blank_re.match(first):
        return default, [first]

    second = read_or_stop()
    if not second:
        return default, [first]

    encoding = find_cookie(second)
    if encoding:
        return encoding, [first, second]

    return default, [first, second]


def untokenize(iterable: Iterable[TokenInfo]) -> str:
    ut = Untokenizer()
    return ut.untokenize(iterable)


def is_fstring_start(token: str) -> bool:
    return token.startswith(fstring_prefix)


def _split_fstring_start_and_middle(token: str) -> Tuple[str, str]:
    for prefix in fstring_prefix:
        _, prefix, rest = token.partition(prefix)
        if prefix != "":
            return prefix, rest

    raise ValueError(f"Token {token!r} is not a valid f-string start")


STATE_NOT_FSTRING: Final[int] = 0
STATE_MIDDLE: Final[int] = 1
STATE_IN_BRACES: Final