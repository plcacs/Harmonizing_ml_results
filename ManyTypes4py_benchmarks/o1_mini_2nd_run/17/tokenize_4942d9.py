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
__author__ = 'Ka-Ping Yee <ping@lfw.org>'
__credits__ = 'GvR, ESR, Tim Peters, Thomas Wouters, Fred Drake, Skip Montanaro'
import string
import re
from codecs import BOM_UTF8, lookup
from lib2to3.pgen2.token import *
from . import token
from typing import (
    Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union
)

__all__: List[str] = [x for x in dir(token) if x[0] != '_'] + ['tokenize', 'generate_tokens', 'untokenize']
del token
try:
    bytes
except NameError:
    bytes = str

def group(*choices: str) -> str:
    return '(' + '|'.join(choices) + ')'

def any(*choices: str) -> str:
    return group(*choices) + '*'

def maybe(*choices: str) -> str:
    return group(*choices) + '?'

Whitespace: str = '[ \\f\\t]*'
Comment: str = '#[^\\r\\n]*'
Ignore: str = Whitespace + any('\\\\\\r?\\n' + Whitespace) + maybe(Comment)
Name: str = '[a-zA-Z_]\\w*'
Binnumber: str = '0[bB][01]*'
Hexnumber: str = '0[xX][\\da-fA-F]*[lL]?'
Octnumber: str = '0[oO]?[0-7]*[lL]?'
Decnumber: str = '[1-9]\\d*[lL]?'
Intnumber: str = group(Binnumber, Hexnumber, Octnumber, Decnumber)
Exponent: str = '[eE][-+]?\\d+'
Pointfloat: str = group('\\d+\\.\\d*', '\\.\\d+') + maybe(Exponent)
Expfloat: str = '\\d+' + Exponent
Floatnumber: str = group(Pointfloat, Expfloat)
Imagnumber: str = group('\\d+[jJ]', Floatnumber + '[jJ]')
Number: str = group(Imagnumber, Floatnumber, Intnumber)
Single: str = "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'"
Double: str = '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"'
Single3: str = "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''"
Double3: str = '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""'
Triple: str = group("[ubUB]?[rR]?'''", '[ubUB]?[rR]?"""')
String: str = group("[uU]?[rR]?'[^\\n'\\\\]*(?:\\\\.[^\\n'\\\\]*)*'", '[uU]?[rR]?"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*"')
Operator: str = group('\\*\\*=?', '>>=?', '<<=?', '<>', '!=', '//=?', '->', '[+\\-*/%&|^=<>]=?', '~')
Bracket: str = '[][(){}]'
Special: str = group('\\r?\\n', '[:;.,`@]')
Funny: str = group(Operator, Bracket, Special)
PlainToken: str = group(Number, Funny, String, Name)
Token: str = Ignore + PlainToken
ContStr: str = group(
    "[uUbB]?[rR]?'[^\\n'\\\\]*(?:\\\\.[^\\n'\\\\]*)*" + group("'", '\\\\\\r?\\n'),
    '[uUbB]?[rR]?"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*' + group('"', '\\\\\\r?\\n')
)
PseudoExtras: str = group('\\\\\\r?\\n', Comment, Triple)
PseudoToken: str = Whitespace + group(PseudoExtras, Number, Funny, ContStr, Name)
tokenprog, pseudoprog, single3prog, double3prog: Tuple[re.Pattern, re.Pattern, re.Pattern, re.Pattern] = tuple(map(re.compile, (Token, PseudoToken, Single3, Double3)))
endprogs: Dict[str, Optional[re.Pattern]] = {
    "'": re.compile(Single),
    '"': re.compile(Double),
    "'''": single3prog,
    '"""': double3prog,
    "r'''": single3prog,
    'r"""': double3prog,
    "u'''": single3prog,
    'u"""': double3prog,
    "b'''": single3prog,
    'b"""': double3prog,
    "ur'''": single3prog,
    'ur"""': double3prog,
    "br'''": single3prog,
    'br"""': double3prog,
    "R'''": single3prog,
    'R"""': double3prog,
    "U'''": single3prog,
    'U"""': double3prog,
    "B'''": single3prog,
    'B"""': double3prog,
    "uR'''": single3prog,
    'uR"""': double3prog,
    "Ur'''": single3prog,
    'Ur"""': double3prog,
    "UR'''": single3prog,
    'UR"""': double3prog,
    "bR'''": single3prog,
    'bR"""': double3prog,
    "Br'''": single3prog,
    'Br"""': double3prog,
    "BR'''": single3prog,
    'BR"""': double3prog,
    'r': None,
    'R': None,
    'u': None,
    'U': None,
    'b': None,
    'B': None
}
triple_quoted: Dict[str, str] = {}
for t in (
    "'''", '"""', "r'''", 'r"""', "R'''", 'R"""', "u'''", 'u"""',
    "U'''", 'U"""', "b'''", 'b"""', "B'''", 'B"""',
    "ur'''", 'ur"""', "Ur'''", 'Ur"""', "uR'''", 'uR"""',
    "UR'''", 'UR"""', "br'''", 'br"""', "Br'''", 'Br"""',
    "bR'''", 'bR"""', "BR'''", 'BR"""'
):
    triple_quoted[t] = t
single_quoted: Dict[str, str] = {}
for t in (
    "'", '"', "r'", 'r"', "R'", 'R"', "u'", 'u"', "U'", 'U"',
    "b'", 'b"', "B'", 'B"', "ur'", 'ur"', "Ur'", 'Ur"',
    "uR'", 'uR"', "UR'", 'UR"', "br'", 'br"', "Br'", 'Br"',
    "bR'", 'bR"', "BR'", 'BR"'
):
    single_quoted[t] = t
tabsize: int = 8

class TokenError(Exception):
    pass

class StopTokenizing(Exception):
    pass

def printtoken(
    type: int,
    token: str,
    xxx_todo_changeme: Tuple[int, int],
    xxx_todo_changeme1: Tuple[int, int],
    line: str
) -> None:
    srow, scol = xxx_todo_changeme
    erow, ecol = xxx_todo_changeme1
    print('%d,%d-%d,%d:\t%s\t%s' % (srow, scol, erow, ecol, tok_name[type], repr(token)))

def tokenize(
    readline: Callable[[], str],
    tokeneater: Callable[[int, str, Tuple[int, int], Tuple[int, int], str], None] = printtoken
) -> None:
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

def tokenize_loop(
    readline: Callable[[], str],
    tokeneater: Callable[[int, str, Tuple[int, int], Tuple[int, int], str], None]
) -> None:
    for token_info in generate_tokens(readline):
        tokeneater(*token_info)

class Untokenizer:

    tokens: List[str]
    prev_row: int
    prev_col: int

    def __init__(self) -> None:
        self.tokens = []
        self.prev_row = 1
        self.prev_col = 0

    def add_whitespace(self, start: Tuple[int, int]) -> None:
        row, col = start
        assert row <= self.prev_row
        col_offset = col - self.prev_col
        if col_offset:
            self.tokens.append(' ' * col_offset)

    def untokenize(self, iterable: Iterable[Tuple[int, str, Tuple[int, int], Tuple[int, int], str]]) -> str:
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

    def compat(
        self,
        token: Tuple[int, str],
        iterable: Iterable[Tuple[int, str, Tuple[int, int], Tuple[int, int], str]]
    ) -> None:
        startline: bool = False
        indents: List[str] = []
        toks_append = self.tokens.append
        toknum, tokval = token
        if toknum in (NAME, NUMBER):
            tokval += ' '
        if toknum in (NEWLINE, NL):
            startline = True
        for tok in iterable:
            toknum, tokval = tok[:2]
            if toknum in (NAME, NUMBER):
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

cookie_re = re.compile('coding[:=]\\s*([-\\w.]+)')

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
    be used to decode a Python source file. It requires one argment, readline,
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
    default: str = 'utf-8'

    def read_or_stop() -> bytes:
        try:
            return readline()
        except StopIteration:
            return bytes()

    def find_cookie(line: bytes) -> Optional[str]:
        nonlocal bom_found, encoding
        try:
            line_string: str = line.decode('ascii')
        except UnicodeDecodeError:
            return None
        matches: List[str] = cookie_re.findall(line_string)
        if not matches:
            return None
        enc: str = _get_normal_name(matches[0])
        try:
            codec = lookup(enc)
        except LookupError:
            raise SyntaxError('unknown encoding: ' + enc)
        if bom_found:
            if codec.name != 'utf-8':
                raise SyntaxError('encoding problem: utf-8')
            return enc + '-sig'
        return enc

    first: bytes = read_or_stop()
    if first.startswith(BOM_UTF8):
        bom_found = True
        first = first[3:]
        default = 'utf-8-sig'
    if not first:
        return (default, [])
    encoding = find_cookie(first)
    if encoding:
        return (encoding, [first])
    second: bytes = read_or_stop()
    if not second:
        return (default, [first])
    encoding = find_cookie(second)
    if encoding:
        return (encoding, [first, second])
    return (default, [first, second])

def untokenize(iterable: Iterable[Union[Tuple[int, str], Tuple[int, str, Tuple[int, int], Tuple[int, int], str]]]) -> str:
    """Transform tokens back into Python source code.

    Each element returned by the iterable must be a token sequence
    with at least two elements, a token number and token value.  If
    only two tokens are passed, the resulting output is poor.

    Round-trip invariant for full input:
        Untokenized source will match input source exactly

    Round-trip invariant for limited intput:
        # Output text will tokenize the back to the input
        t1 = [tok[:2] for tok in generate_tokens(f.readline)]
        newcode = untokenize(t1)
        readline = iter(newcode.splitlines(1)).next
        t2 = [tok[:2] for tokin generate_tokens(readline)]
        assert t1 == t2
    """
    ut = Untokenizer()
    return ut.untokenize(iterable)

def generate_tokens(
    readline: Callable[[], str]
) -> Iterator[Tuple[int, str, Tuple[int, int], Tuple[int, int], str]]:
    """
    The generate_tokens() generator requires one argment, readline, which
    must be a callable object which provides the same interface as the
    readline() method of built-in file objects. Each call to the function
    should return one line of input as a string.  Alternately, readline
    can be a callable function terminating with StopIteration:
        readline = open(myfile).next    # Example of alternate readline

    The generator produces 5-tuples with these members: the token type; the
    token string; a 2-tuple (srow, scol) of ints specifying the row and
    column where the token begins in the source; a 2-tuple (erow, ecol) of
    ints specifying the row and column where the token ends in the source;
    and the line on which the token was found. The line passed is the
    logical line; continuation lines are included.
    """
    lnum = parenlev = continued = 0
    namechars, numchars = (string.ascii_letters + '_', '0123456789')
    contstr: str = ''
    needcont: int = 0
    contline: Optional[str] = None
    indents: List[int] = [0]
    while True:
        try:
            line: str = readline()
        except StopIteration:
            line = ''
        lnum += 1
        pos, max_pos = (0, len(line))
        if contstr:
            if not line:
                raise TokenError('EOF in multi-line string', strstart)
            endmatch = endprog.match(line)
            if endmatch:
                pos = end = endmatch.end(0)
                yield (
                    STRING,
                    contstr + line[:end],
                    strstart,
                    (lnum, end),
                    contline + line
                )
                contstr, needcont = ('', 0)
                contline = None
            elif needcont and not (line.endswith('\\\n') or line.endswith('\\\r\n')):
                yield (
                    ERRORTOKEN,
                    contstr + line,
                    strstart,
                    (lnum, len(line)),
                    contline
                )
                contstr = ''
                contline = None
                continue
            else:
                contstr += line
                contline += line
                continue
        elif parenlev == 0 and not continued:
            if not line:
                break
            column = 0
            while pos < max_pos:
                char = line[pos]
                if char == ' ':
                    column += 1
                elif char == '\t':
                    column = (column // tabsize + 1) * tabsize
                elif char == '\x0c':
                    column = 0
                else:
                    break
                pos += 1
            if pos == max_pos:
                break
            if line[pos] in '#\r\n':
                if line[pos] == '#':
                    comment_token = line[pos:].rstrip('\r\n')
                    nl_pos = pos + len(comment_token)
                    yield (
                        COMMENT,
                        comment_token,
                        (lnum, pos),
                        (lnum, pos + len(comment_token)),
                        line
                    )
                    yield (
                        NL,
                        line[nl_pos:],
                        (lnum, nl_pos),
                        (lnum, len(line)),
                        line
                    )
                else:
                    yield (
                        NL if line[pos] != '#' else COMMENT,
                        line[pos:],
                        (lnum, pos),
                        (lnum, len(line)),
                        line
                    )
                continue
            if column > indents[-1]:
                indents.append(column)
                yield (
                    INDENT,
                    line[:pos],
                    (lnum, 0),
                    (lnum, pos),
                    line
                )
            while column < indents[-1]:
                if column not in indents:
                    raise IndentationError(
                        'unindent does not match any outer indentation level',
                        ('<tokenize>', lnum, pos, line)
                    )
                indents.pop()
                yield (DEDENT, '', (lnum, pos), (lnum, pos), line)
        else:
            if not line:
                raise TokenError('EOF in multi-line statement', (lnum, 0))
            continued = 0
        while pos < max_pos:
            pseudomatch = pseudoprog.match(line, pos)
            if pseudomatch:
                start, end = pseudomatch.span(1)
                spos = (lnum, start)
                epos = (lnum, end)
                pos = end
                token_str = line[start:end]
                initial = line[start]
                if initial in numchars or (initial == '.' and token_str != '.'):
                    yield (NUMBER, token_str, spos, epos, line)
                elif initial in '\r\n':
                    newline = NEWLINE if parenlev > 0 else NL
                    yield (newline, token_str, spos, epos, line)
                elif initial == '#':
                    assert not token_str.endswith('\n')
                    yield (COMMENT, token_str, spos, epos, line)
                elif token_str in triple_quoted:
                    end_prog = endprogs[token_str]
                    endmatch = end_prog.match(line, pos) if end_prog else None
                    if endmatch:
                        pos = endmatch.end(0)
                        token_str_full = line[start:pos]
                        yield (STRING, token_str_full, spos, (lnum, pos), line)
                    else:
                        strstart = (lnum, start)
                        contstr = line[start:]
                        contline = line
                        break
                elif (
                    initial in single_quoted or
                    token_str[:2] in single_quoted or
                    token_str[:3] in single_quoted
                ):
                    if token_str.endswith('\n'):
                        strstart = (lnum, start)
                        end_prog = endprogs.get(initial) or endprogs.get(token_str[1:]) or endprogs.get(token_str[:3])
                        contstr, needcont = (line[start:], 1)
                        contline = line
                        break
                    else:
                        yield (STRING, token_str, spos, epos, line)
                elif initial in namechars:
                    yield (NAME, token_str, spos, epos, line)
                elif initial == '\\':
                    yield (NL, token_str, spos, (lnum, pos), line)
                    continued = 1
                else:
                    if initial in '([{':
                        parenlev += 1
                    elif initial in ')]}':
                        parenlev -= 1
                    yield (OP, token_str, spos, epos, line)
            else:
                yield (
                    ERRORTOKEN,
                    line[pos],
                    (lnum, pos),
                    (lnum, pos + 1),
                    line
                )
                pos += 1
    for indent in indents[1:]:
        yield (DEDENT, '', (lnum, 0), (lnum, 0), '')
    yield (ENDMARKER, '', (lnum, 0), (lnum, 0), '')
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        tokenize(open(sys.argv[1]).readline)
    else:
        tokenize(sys.stdin.readline)
