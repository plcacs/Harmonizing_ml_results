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
from blib2to3.pgen2.token import ASYNC, AWAIT, COMMENT, DEDENT, ENDMARKER, ERRORTOKEN, FSTRING_END, FSTRING_MIDDLE, FSTRING_START, INDENT, LBRACE, NAME, NEWLINE, NL, NUMBER, OP, RBRACE, STRING, tok_name
__author__ = 'Ka-Ping Yee <ping@lfw.org>'
__credits__ = 'GvR, ESR, Tim Peters, Thomas Wouters, Fred Drake, Skip Montanaro'
import re
from codecs import BOM_UTF8, lookup
from . import token
__all__ = [x for x in dir(token) if x[0] != '_'] + ['tokenize', 'generate_tokens', 'untokenize']
del token

def group(*choices: str) -> str:
    return '(' + '|'.join(choices) + ')'

def any(*choices: str) -> str:
    return group(*choices) + '*'

def maybe(*choices: str) -> str:
    return group(*choices) + '?'

def _combinations(*l: str) -> Set[str]:
    return {x + y for x in l for y in l + ('',) if x.casefold() != y.casefold()}

Whitespace: Final[str] = '[ \\f\\t]*'
Comment: Final[str] = '#[^\\r\\n]*'
Ignore: Final[str] = Whitespace + any('\\\\\\r?\\n' + Whitespace) + maybe(Comment)
Name: Final[str] = '[^\\s#\\(\\)\\[\\]\\{\\}+\\-*/!@$%^&=|;:\'\\",\\.<>/?`~\\\\]+'
Binnumber: Final[str] = '0[bB]_?[01]+(?:_[01]+)*'
Hexnumber: Final[str] = '0[xX]_?[\\da-fA-F]+(?:_[\\da-fA-F]+)*[lL]?'
Octnumber: Final[str] = '0[oO]?_?[0-7]+(?:_[0-7]+)*[lL]?'
Decnumber: Final[str] = group('[1-9]\\d*(?:_\\d+)*[lL]?', '0[lL]?')
Intnumber: Final[str] = group(Binnumber, Hexnumber, Octnumber, Decnumber)
Exponent: Final[str] = '[eE][-+]?\\d+(?:_\\d+)*'
Pointfloat: Final[str] = group('\\d+(?:_\\d+)*\\.(?:\\d+(?:_\\d+)*)?', '\\.\\d+(?:_\\d+)*') + maybe(Exponent)
Expfloat: Final[str] = '\\d+(?:_\\d+)*' + Exponent
Floatnumber: Final[str] = group(Pointfloat, Expfloat)
Imagnumber: Final[str] = group('\\d+(?:_\\d+)*[jJ]', Floatnumber + '[jJ]')
Number: Final[str] = group(Imagnumber, Floatnumber, Intnumber)
Single: Final[str] = "(?:\\\\.|[^'\\\\])*'"
Double: Final[str] = '(?:\\\\.|[^"\\\\])*"'
Single3: Final[str] = "(?:\\\\.|'(?!'')|[^'\\\\])*'''"
Double3: Final[str] = '(?:\\\\.|"(?!"")|[^"\\\\])*"""'
_litprefix: Final[str] = '(?:[uUrRbB]|[rR][bB]|[bBuU][rR])?'
_fstringlitprefix: Final[str] = '(?:rF|FR|Fr|fr|RF|F|rf|f|Rf|fR)'
Triple: Final[str] = group(_litprefix + "'''", _litprefix + '"""', _fstringlitprefix + '"""', _fstringlitprefix + "'''")
SingleLbrace: Final[str] = "(?:\\\\N{|{{|\\\\'|[^\\n'{])*(?<!\\\\N)({)(?!{)"
DoubleLbrace: Final[str] = '(?:\\\\N{|{{|\\\\"|[^\\n"{])*(?<!\\\\N)({)(?!{)'
Single3Lbrace: Final[str] = "(?:\\\\N{|{{|\\\\'|'(?!'')|[^'{])*(?<!\\\\N){(?!{)"
Double3Lbrace: Final[str] = '(?:\\\\N{|{{|\\\\"|"(?!"")|[^"{])*(?<!\\\\N){(?!{)'
Bang: Final[str] = Whitespace + group('!') + '(?!=)'
bang: Final[Pattern[str]] = re.compile(Bang)
Colon: Final[str] = Whitespace + group(':')
colon: Final[Pattern[str]] = re.compile(Colon)
FstringMiddleAfterColon: Final[str] = group(Whitespace + '.*?') + group('{', '}')
fstring_middle_after_colon: Final[Pattern[str]] = re.compile(FstringMiddleAfterColon)
Operator: Final[str] = group('\\*\\*=?', '>>=?', '<<=?', '<>', '!=', '//=?', '->', '[+\\-*/%&@|^=<>:]=?', '~')
Bracket: Final[str] = '[][(){}]'
Special: Final[str] = group('\\r?\\n', '[:;.,`@]')
Funny: Final[str] = group(Operator, Bracket, Special)
_string_middle_single: Final[str] = "(?:[^\\n'\\\\]|\\\\.)*"
_string_middle_double: Final[str] = '(?:[^\\n"\\\\]|\\\\.)*'
_fstring_middle_single: Final[str] = SingleLbrace
_fstring_middle_double: Final[str] = DoubleLbrace
ContStr: Final[str] = group(_litprefix + "'" + _string_middle_single + group("'", '\\\\\\r?\\n'), _litprefix + '"' + _string_middle_double + group('"', '\\\\\\r?\\n'), group(_fstringlitprefix + "'") + _fstring_middle_single, group(_fstringlitprefix + '"') + _fstring_middle_double, group(_fstringlitprefix + "'") + _string_middle_single + group("'", '\\\\\\r?\\n'), group(_fstringlitprefix + '"') + _string_middle_double + group('"', '\\\\\\r?\\n'))
PseudoExtras: Final[str] = group('\\\\\\r?\\n', Comment, Triple)
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
_strprefixes: Final[Set[str]] = _combinations('r', 'R', 'b', 'B') | {'u', 'U', 'ur', 'uR', 'Ur', 'UR'}
_fstring_prefixes: Final[Set[str]] = _combinations('r', 'R', 'f', 'F') - {'r', 'R'}
endprogs: Final[Dict[str, Pattern[str]]] = {"'": singleprog, '"': doubleprog, "'''": single3prog, '"""': double3prog, **{f"{prefix}'": singleprog for prefix in _strprefixes}, **{f'{prefix}"': doubleprog for prefix in _strprefixes}, **{f"{prefix}'": singleprog_plus_lbrace for prefix in _fstring_prefixes}, **{f'{prefix}"': doubleprog_plus_lbrace for prefix in _fstring_prefixes}, **{f"{prefix}'''": single3prog for prefix in _strprefixes}, **{f'{prefix}"""': double3prog for prefix in _strprefixes}, **{f"{prefix}'''": single3prog_plus_lbrace for prefix in _fstring_prefixes}, **{f'{prefix}"""': double3prog_plus_lbrace for prefix in _fstring_prefixes}}
triple_quoted: Final[Set[str]] = {"'''", '"""'} | {f"{prefix}'''" for prefix in _strprefixes | _fstring_prefixes} | {f'{prefix}"""' for prefix in _strprefixes | _fstring_prefixes}
single_quoted: Final[Set[str]] = {"'", '"'} | {f"{prefix}'" for prefix in _strprefixes | _fstring_prefixes} | {f'{prefix}"' for prefix in _strprefixes | _fstring_prefixes}
fstring_prefix: Final[Tuple[str, ...]] = tuple({f"{prefix}'" for prefix in _fstring_prefixes} | {f'{prefix}"' for prefix in _fstring_prefixes} | {f"{prefix}'''" for prefix in _fstring_prefixes} | {f'{prefix}"""' for prefix in _fstring_prefixes})
tabsize: Final[int] = 8

class TokenError(Exception):
    pass

class StopTokenizing(Exception):
    pass

Coord = Tuple[int, int]

def printtoken(type: int, token: str, srow_col: Coord, erow_col: Coord, line: str) -> None:
    srow, scol = srow_col
    erow, ecol = erow_col
    print('%d,%d-%d,%d:\t%s\t%s' % (srow, scol, erow, ecol, tok_name[type], repr(token)))

TokenEater = Callable[[int, str, Coord, Coord, str], None]

def tokenize(readline: Callable[[], str], tokeneater: TokenEater = printtoken) -> None:
    """
    The tokenize() function accepts two parameters: one representing the
    input stream, and one providing an output mechanism for tokenize().

    The first parameter, readline, must be a callable object which provides
    the same interface as the readline() method of built-in file objects.
    Each call to the function should return one line of input as a string.

    The second parameter, tokeneater, must also be a callable object. It is
    called once for each token, with five arguments, corresponding to the
    tuples generated by generate_tokens().
    """
    try:
        tokenize_loop(readline, tokeneater)
    except StopTokenizing:
        pass

def tokenize_loop(readline: Callable[[], str], tokeneater: TokenEater) -> None:
    for token_info in generate_tokens(readline):
        tokeneater(*token_info)

GoodTokenInfo = Tuple[int, str, Coord, Coord, str]
TokenInfo = Union[Tuple[int, str], GoodTokenInfo]

class Untokenizer:

    def __init__(self) -> None:
        self.tokens: List[str] = []
        self.prev_row: int = 1
        self.prev_col: int = 0

    def add_whitespace(self, start: Coord) -> None:
        row, col = start
        assert row <= self.prev_row
        col_offset = col - self.prev_col
        if col_offset:
            self.tokens.append(' ' * col_offset)

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
        return ''.join(self.tokens)

    def compat(self, token: TokenInfo, iterable: Iterable[TokenInfo]) -> None:
        startline = False
        indents: List[str] = []
        toks_append = self.tokens.append
        toknum, tokval = token
        if toknum in (NAME, NUMBER):
            tokval += ' '
        if toknum in (NEWLINE, NL):
            startline = True
        for tok in iterable:
            toknum, tokval = tok[:2]
            if toknum in (NAME, NUMBER, ASYNC, AWAIT):
                tokval += ' '
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

cookie_re: Final[Pattern[str]] = re.compile('^[ \\t\\f]*#.*?coding[:=][ \\t]*([-\\w.]+)', re.ASCII)
blank_re: Final[Pattern[bytes]] = re.compile(b'^[ \\t\\f]*(?:[#\\r\\n]|$)', re.ASCII)

def _get_normal_name(orig_enc: str) -> str:
    """Imitates get_normal_name in tokenizer.c."""
    enc = orig_enc[:12].lower().replace('_', '-')
    if enc == 'utf-8' or enc.startswith('utf-8-'):
        return 'utf-8'
    if enc in ('latin-1', 'iso-8859-1', 'iso-latin-1') or enc.startswith(('latin-1-', 'iso-8859-1-', 'iso-latin-1-')):
        return 'iso-8859-1'
    return orig_enc

def detect_encoding(readline: Callable[[], bytes]) -> Tuple[str, List[bytes]]:
    """
    The detect_encoding() function is used to detect the encoding that should
    be used to decode a Python source file. It requires one argument, readline,
    in the same way as the tokenize() generator.

    It will call readline a maximum of twice, and return the encoding used
    (as a string) and a list of any lines (left as bytes) it has read
    in.

    It detects the encoding from the presence of a utf-8 bom or an encoding
    cookie as specified in pep-0263. If both a bom and a cookie are present, but
    disagree, a SyntaxError will be raised. If the encoding cookie is an invalid
    charset, raise a SyntaxError.  Note that if a utf-8 bom is found,
    'utf-8-sig' is returned.

    If no encoding is specified, then the default of 'utf-8' will be returned.
    """
    bom_found = False
    encoding: Optional[str] = None
    default = 'utf-8'

    def read_or_stop() -> bytes:
        try:
            return readline()
        except StopIteration:
            return b''

    def find_cookie(line: bytes) -> Optional[str]:
        try:
            line_string = line.decode('ascii')
        except UnicodeDecodeError:
            return None
        match = cookie_re.match(line_string)
        if not match:
            return None
        encoding = _get_normal_name(match.group(1))
        try:
            codec = lookup(encoding)
        except LookupError:
            raise SyntaxError('unknown encoding: ' + encoding)
        if bom_found:
            if codec.name != 'utf-8':
                raise SyntaxError('encoding problem: utf-8')
            encoding += '-sig'
        return encoding
    first = read_or_stop()
    if first.startswith(BOM_UTF8):
        bom_found = True
        first = first[3:]
        default = 'utf-8-sig'
    if not first:
        return (default, [])
    encoding = find_cookie(first)
    if encoding:
        return (encoding, [first])
    if not blank_re.match(first):
        return (default, [first])
    second = read_or_stop()
    if not second