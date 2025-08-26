'Tokenization help for Python programs.\n\ngenerate_tokens(readline) is a generator that breaks a stream of\ntext into Python tokens.  It accepts a readline-like method which is called\nrepeatedly to get the next line of input (or "" for EOF).  It generates\n5-tuples with these members:\n\n    the token type (see token.py)\n    the token (a string)\n    the starting (row, column) indices of the token (a 2-tuple of ints)\n    the ending (row, column) indices of the token (a 2-tuple of ints)\n    the original line (string)\n\nIt is designed to match the working of the Python tokenizer exactly, except\nthat it produces COMMENT tokens for comments and gives type OP for all\noperators\n\nOlder entry points\n    tokenize_loop(readline, tokeneater)\n    tokenize(readline, tokeneater=printtoken)\nare the same, except instead of generating tokens, tokeneater is a callback\nfunction to which the 5 fields described above are passed as 5 arguments,\neach time a new token is found.'
__author__ = 'Ka-Ping Yee <ping@lfw.org>'
__credits__ = 'GvR, ESR, Tim Peters, Thomas Wouters, Fred Drake, Skip Montanaro'
import string, re
from codecs import BOM_UTF8, lookup
from lib2to3.pgen2.token import *
from . import token
from typing import Any, Callable, Dict, Generator, Iterable, List, Match, Optional, Pattern, Set, Tuple, Union
__all__ = ([x for x in dir(token) if (x[0] != '_')] + ['tokenize', 'generate_tokens', 'untokenize'])
del token
try:
    bytes
except NameError:
    bytes = str

def group(*choices: str) -> str:
    return (('(' + '|'.join(choices)) + ')')

def any(*choices: str) -> str:
    return (group(*choices) + '*')

def maybe(*choices: str) -> str:
    return (group(*choices) + '?')

def _combinations(*l: str) -> Set[str]:
    return set(((x + y) for x in l for y in (l + ('',)) if (x.casefold() != y.casefold())))
Whitespace = '[ \\f\\t]*'
Comment = '#[^\\r\\n]*'
Ignore = ((Whitespace + any(('\\\\\\r?\\n' + Whitespace))) + maybe(Comment))
Name = '\\w+'
Binnumber = '0[bB]_?[01]+(?:_[01]+)*'
Hexnumber = '0[xX]_?[\\da-fA-F]+(?:_[\\da-fA-F]+)*[lL]?'
Octnumber = '0[oO]?_?[0-7]+(?:_[0-7]+)*[lL]?'
Decnumber = group('[1-9]\\d*(?:_\\d+)*[lL]?', '0[lL]?')
Intnumber = group(Binnumber, Hexnumber, Octnumber, Decnumber)
Exponent = '[eE][-+]?\\d+(?:_\\d+)*'
Pointfloat = (group('\\d+(?:_\\d+)*\\.(?:\\d+(?:_\\d+)*)?', '\\.\\d+(?:_\\d+)*') + maybe(Exponent))
Expfloat = ('\\d+(?:_\\d+)*' + Exponent)
Floatnumber = group(Pointfloat, Expfloat)
Imagnumber = group('\\d+(?:_\\d+)*[jJ]', (Floatnumber + '[jJ]'))
Number = group(Imagnumber, Floatnumber, Intnumber)
Single = "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'"
Double = '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"'
Single3 = "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''"
Double3 = '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""'
_litprefix = '(?:[uUrRbBfF]|[rR][fFbB]|[fFbBuU][rR])?'
Triple = group((_litprefix + "'''"), (_litprefix + '"""'))
String = group((_litprefix + "'[^\\n'\\\\]*(?:\\\\.[^\\n'\\\\]*)*'"), (_litprefix + '"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*"'))
Operator = group('\\*\\*=?', '>>=?', '<<=?', '<>', '!=', '//=?', '\\->', '[+\\-*/%&@|^=<>]=?', '~')
Bracket = '[][(){}]'
Special = group('\\r?\\n', ':=', '[:;.,`@]')
Funny = group(Operator, Bracket, Special)
PlainToken = group(Number, Funny, String, Name)
Token = (Ignore + PlainToken)
ContStr = group(((_litprefix + "'[^\\n'\\\\]*(?:\\\\.[^\\n'\\\\]*)*") + group("'", '\\\\\\r?\\n')), ((_litprefix + '"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*') + group('"', '\\\\\\r?\\n')))
PseudoExtras = group('\\\\\\r?\\n', Comment, Triple)
PseudoToken = (Whitespace + group(PseudoExtras, Number, Funny, ContStr, Name))
(tokenprog, pseudoprog, single3prog, double3prog) = map(re.compile, (Token, PseudoToken, Single3, Double3))
_strprefixes: Set[str] = ((_combinations('r', 'R', 'f', 'F') | _combinations('r', 'R', 'b', 'B')) | {'u', 'U', 'ur', 'uR', 'Ur', 'UR'})
endprogs: Dict[str, Optional[Pattern[str]]] = {"'": re.compile(Single), '"': re.compile(Double), "'''": single3prog, '"""': double3prog, **{f"{prefix}'''": single3prog for prefix in _strprefixes}, **{f'{prefix}"""': double3prog for prefix in _strprefixes}, **{prefix: None for prefix in _strprefixes}}
triple_quoted: Set[str] = (({"'''", '"""'} | {f"{prefix}'''" for prefix in _strprefixes}) | {f'{prefix}"""' for prefix in _strprefixes})
single_quoted: Set[str] = (({"'", '"'} | {f"{prefix}'" for prefix in _strprefixes}) | {f'{prefix}"' for prefix in _strprefixes})
tabsize: int = 8

class TokenError(Exception):
    pass

class StopTokenizing(Exception):
    pass

def printtoken(type: int, token: str, xxx_todo_changeme: Tuple[int, int], xxx_todo_changeme1: Tuple[int, int], line: str) -> None:
    (srow, scol) = xxx_todo_changeme
    (erow, ecol) = xxx_todo_changeme1
    print(('%d,%d-%d,%d:\t%s\t%s' % (srow, scol, erow, ecol, tok_name[type], repr(token))))

def tokenize(readline: Callable[[], str], tokeneater: Callable[[int, str, Tuple[int, int], Tuple[int, int], str], None] = printtoken) -> None:
    '\n    The tokenize() function accepts two parameters: one representing the\n    input stream, and one providing an output mechanism for tokenize().\n\n    The first parameter, readline, must be a callable object which provides\n    the same interface as the readline() method of built-in file objects.\n    Each call to the function should return one line of input as a string.\n\n    The second parameter, tokeneater, must also be a callable object. It is\n    called once for each token, with five arguments, corresponding to the\n    tuples generated by generate_tokens().\n    '
    try:
        tokenize_loop(readline, tokeneater)
    except StopTokenizing:
        pass

def tokenize_loop(readline: Callable[[], str], tokeneater: Callable[[int, str, Tuple[int, int], Tuple[int, int], str], None]) -> None:
    for token_info in generate_tokens(readline):
        tokeneater(*token_info)

class Untokenizer():

    def __init__(self) -> None:
        self.tokens: List[str] = []
        self.prev_row: int = 1
        self.prev_col: int = 0

    def add_whitespace(self, start: Tuple[int, int]) -> None:
        (row, col) = start
        assert (row <= self.prev_row)
        col_offset: int = (col - self.prev_col)
        if col_offset:
            self.tokens.append((' ' * col_offset))

    def untokenize(self, iterable: Iterable[Any]) -> str:
        for t in iterable:
            if (len(t) == 2):
                self.compat(t, iterable)
                break
            (tok_type, token, start, end, line) = t
            self.add_whitespace(start)
            self.tokens.append(token)
            (self.prev_row, self.prev_col) = end
            if (tok_type in (NEWLINE, NL)):
                self.prev_row += 1
                self.prev_col = 0
        return ''.join(self.tokens)

    def compat(self, token: Tuple[int, str], iterable: Iterable[Tuple[int, str]]) -> None:
        startline: bool = False
        indents: List[str] = []
        toks_append = self.tokens.append
        (toknum, tokval) = token
        if (toknum in (NAME, NUMBER)):
            tokval += ' '
        if (toknum in (NEWLINE, NL)):
            startline = True
        for tok in iterable:
            (toknum, tokval) = tok[:2]
            if (toknum in (NAME, NUMBER, ASYNC, AWAIT)):
                tokval += ' '
            if (toknum == INDENT):
                indents.append(tokval)
                continue
            elif (toknum == DEDENT):
                indents.pop()
                continue
            elif (toknum in (NEWLINE, NL)):
                startline = True
            elif (startline and indents):
                toks_append(indents[(- 1)])
                startline = False
            toks_append(tokval)
cookie_re: Pattern[str] = re.compile('^[ \\t\\f]*#.*?coding[:=][ \\t]*([-\\w.]+)', re.ASCII)
blank_re: Pattern[bytes] = re.compile(b'^[ \\t\\f]*(?:[#\\r\\n]|$)', re.ASCII)

def _get_normal_name(orig_enc: str) -> str:
    'Imitates get_normal_name in tokenizer.c.'
    enc: str = orig_enc[:12].lower().replace('_', '-')
    if ((enc == 'utf-8') or enc.startswith('utf-8-')):
        return 'utf-8'
    if ((enc in ('latin-1', 'iso-8859-1', 'iso-latin-1')) or enc.startswith(('latin-1-', 'iso-8859-1-', 'iso-latin-1-'))):
        return 'iso-8859-1'
    return orig_enc

def detect_encoding(readline: Callable[[], bytes]) -> Tuple[str, List[bytes]]:
    "\n    The detect_encoding() function is used to detect the encoding that should\n    be used to decode a Python source file. It requires one argument, readline,\n    in the same way as the tokenize() generator.\n\n    It will call readline a maximum of twice, and return the encoding used\n    (as a string) and a list of any lines (left as bytes) it has read\n    in.\n\n    It detects the encoding from the presence of a utf-8 bom or an encoding\n    cookie as specified in pep-0263. If both a bom and a cookie are present, but\n    disagree, a SyntaxError will be raised. If the encoding cookie is an invalid\n    charset, raise a SyntaxError.  Note that if a utf-8 bom is found,\n    'utf-8-sig' is returned.\n\n    If no encoding is specified, then the default of 'utf-8' will be returned.\n    "
    bom_found: bool = False
    encoding: Optional[str] = None
    default: str = 'utf-8'

    def read_or_stop() -> bytes:
        try:
            return readline()
        except StopIteration:
            return bytes()

    def find_cookie(line: bytes) -> Optional[str]:
        try:
            line_string: str = line.decode('ascii')
        except UnicodeDecodeError:
            return None
        match: Optional[Match[str]] = cookie_re.match(line_string)
        if (not match):
            return None
        encoding: str = _get_normal_name(match.group(1))
        try:
            codec = lookup(encoding)
        except LookupError:
            raise SyntaxError(('unknown encoding: ' + encoding))
        if bom_found:
            if (codec.name != 'utf-8'):
                raise SyntaxError('encoding problem: utf-8')
            encoding += '-sig'
        return encoding
    first: bytes = read_or_stop()
    if first.startswith(BOM_UTF8):
        bom_found = True
        first = first[3:]
        default = 'utf-8-sig'
    if (not first):
        return (default, [])
    encoding = find_cookie(first)
    if encoding:
        return (encoding, [first])
    if (not blank_re.match(first)):
        return (default, [first])
    second: bytes = read_or_stop()
    if (not second):
        return (default, [first])
    encoding = find_cookie(second)
    if encoding:
        return (encoding, [first, second])
    return (default, [first, second])

def untokenize(iterable: Iterable[Any]) -> str:
    'Transform tokens back into Python source code.\n\n    Each element returned by the iterable must be a token sequence\n    with at least two elements, a token number and token value.  If\n    only two tokens are passed, the resulting output is poor.\n\n    Round-trip invariant for full input:\n        Untokenized source will match input source exactly\n\n    Round-trip invariant for limited input:\n        # Output text will tokenize the back to the input\n        t1 = [tok[:2] for tok in generate_tokens(f.readline)]\n        newcode = untokenize(t1)\n        readline = iter(newcode.splitlines(1)).next\n        t2 = [tok[:2] for tokin generate_tokens(readline)]\n        assert t1 == t2\n    '
    ut: Untokenizer = Untokenizer()
    return ut.untokenize(iterable)

def generate_tokens(readline: Callable[[], str]) -> Generator[Tuple[int, str, Tuple[int, int], Tuple[int, int], str], None, None]:
    '\n    The generate_tokens() generator requires one argument, readline, which\n    must be a callable object which provides the same interface as the\n    readline() method of built-in file objects. Each call to the function\n    should return one line of input as a string.  Alternately, readline\n    can be a callable function terminating with StopIteration:\n        readline = open(myfile).next    # Example of alternate readline\n\n    The generator produces 5-tuples with these members: the token type; the\n    token string; a 2-tuple (srow, scol) of ints specifying the row and\n    column where the token begins in the source; a 2-tuple (erow, ecol) of\n    ints specifying the row and column where the token ends in the source;\n    and the line on which the token was found. The line passed is the\n    physical line.\n    '
    lnum: int = 0
    parenlev: int = 0
    continued: int = 0
    (contstr, needcont) = ('', 0)
    contline: Optional[str] = None
    indents: List[int] = [0]
    stashed: Optional[Tuple[int, str, Tuple[int, int], Tuple[int, int], str]] = None
    async_def: bool = False
    async_def_indent: int = 0
    async_def_nl: bool = False
    while 1:
        try:
            line: str = readline()
        except StopIteration:
            line = ''
        lnum = (lnum + 1)
        (pos, max) = (0, len(line))
        if contstr:
            if (not line):
                raise TokenError('EOF in multi-line string', strstart)
            endmatch: Optional[Match[str]] = endprog.match(line)
            if endmatch:
                pos = end = endmatch.end(0)
                (yield (STRING, (contstr + line[:end]), strstart, (lnum, end), (contline + line)))
                (contstr, needcont) = ('', 0)
                contline = None
            elif (needcont and (line[(- 2):] != '\\\n') and (line[(- 3):] != '\\\r\n')):
                (yield (ERRORTOKEN, (contstr + line), strstart, (lnum, len(line)), contline))
                contstr = ''
                contline = None
                continue
            else:
                contstr = (contstr