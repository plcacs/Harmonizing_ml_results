#!/usr/bin/env python3
"""
Tokenization help for Python programs.

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
each time a new token is found.
"""

__author__ = 'Ka-Ping Yee <ping@lfw.org>'
__credits__ = 'GvR, ESR, Tim Peters, Thomas Wouters, Fred Drake, Skip Montanaro'

import string
import re
from codecs import BOM_UTF8, lookup
from lib2to3.pgen2.token import *
from . import token
from typing import Callable, Iterator, Tuple, List, Optional, Any

__all__ = [x for x in dir(token) if x[0] != '_'] + ['tokenize', 'generate_tokens', 'untokenize']
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
ContStr: str = group("[uUbB]?[rR]?'[^\\n'\\\\]*(?:\\\\.[^\\n'\\\\]*)*" + group("'", '\\\\\\r?\\n'),
                     '[uUbB]?[rR]?"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*' + group('"', '\\\\\\r?\\n'))
PseudoExtras: str = group('\\\\\\r?\\n', Comment, Triple)
PseudoToken: str = Whitespace + group(PseudoExtras, Number, Funny, ContStr, Name)
tokenprog: re.Pattern = re.compile(Token)
pseudoprog: re.Pattern = re.compile(PseudoToken)
single3prog: re.Pattern = re.compile(Single3)
double3prog: re.Pattern = re.compile(Double3)
endprogs: dict[str, Optional[re.Pattern]] = {
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
    'br"""': single3prog,
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
triple_quoted: dict[str, str] = {}
for t in ("'''", '"""', "r'''", 'r"""', "R'''", 'R"""', "u'''", 'u"""', "U'''", 'U"""',
          "b'''", 'b"""', "B'''", 'B"""', "ur'''", 'ur"""', "Ur'''", 'Ur"""', "uR'''",
          'uR"""', "UR'''", 'UR"""', "br'''", 'br"""', "Br'''", 'Br"""', "bR'''", 'bR"""',
          "BR'''", 'BR"""'):
    triple_quoted[t] = t
single_quoted: dict[str, str] = {}
for t in ("'", '"', "r'", 'r"', "R'", 'R"', "u'", 'u"', "U'", 'U"', "b'", 'b"', "B'", 'B"',
          "ur'", 'ur"', "Ur'", 'Ur"', "uR'", 'uR"', "UR'", 'UR"', "br'", 'br"', "Br'", 'Br"',
          "bR'", 'bR"', "BR'", 'BR"'):
    single_quoted[t] = t
tabsize: int = 8

class TokenError(Exception):
    pass

class StopTokenizing(Exception):
    pass

def printtoken(tok_type: int, token: str, start: Tuple[int, int], end: Tuple[int, int], line: str) -> None:
    srow, scol = start
    erow, ecol = end
    print('%d,%d-%d,%d:\t%s\t%s' % (srow, scol, erow, ecol, tok_name[tok_type], repr(token)))

def tokenize(readline: Callable[[], str],
             tokeneater: Callable[..., None] = printtoken) -> None:
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

def tokenize_loop(readline: Callable[[], str],
                  tokeneater: Callable[..., None]) -> None:
    for token_info in generate_tokens(readline):
        tokeneater(*token_info)

class Untokenizer:
    def __init__(self) -> None:
        self.tokens: List[str] = []
        self.prev_row: int = 1
        self.prev_col: int = 0

    def add_whitespace(self, start: Tuple[int, int]) -> None:
        row, col = start
        assert row <= self.prev_row
        col_offset: int = col - self.prev_col
        if col_offset:
            self.tokens.append(' ' * col_offset)

    def untokenize(self, iterable: Iterator[Tuple[int, str, Tuple[int, int], Tuple[int, int], str]]) -> str:
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

    def compat(self, token: Tuple[int, str], iterable: Iterator[Any]) -> None:
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

cookie_re: re.Pattern = re.compile('coding[:=]\\s*([-\\w.]+)')

def _get_normal_name(orig_enc: str) -> str:
    """Imitates get_normal_name in tokenizer.c."""
    enc: str = orig_enc[:12].lower().replace('_', '-')
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
        matches = cookie_re.findall(line_string)
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
            enc += '-sig'
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

def untokenize(iterable: Iterator[Tuple[int, str, Tuple[int, int], Tuple[int, int], str]]) -> str:
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
        readline = iter(newcode.splitlines(1)).__next__
        t2 = [tok[:2] for tok in generate_tokens(readline)]
        assert t1 == t2
    """
    ut = Untokenizer()
    return ut.untokenize(iterable)

def generate_tokens(readline: Callable[[], str]) -> Iterator[Tuple[int, str, Tuple[int, int], Tuple[int, int], str]]:
    """
    The generate_tokens() generator requires one argument, readline, which
    must be a callable object which provides the same interface as the
    readline() method of built-in file objects. Each call to the function
    should return one line of input as a string.  Alternately, readline
    can be a callable function terminating with StopIteration:
        readline = open(myfile).__next__    # Example of alternate readline

    The generator produces 5-tuples with these members: the token type; the
    token string; a 2-tuple (srow, scol) of ints specifying the row and
    column where the token begins in the source; a 2-tuple (erow, ecol) of ints
    specifying the row and column where the token ends in the source; and the line
    on which the token was found. The line passed is the logical line; continuation
    lines are included.
    """
    lnum: int = 0
    parenlev: int = 0
    continued: int = 0
    namechars: str = string.ascii_letters + '_'
    numchars: str = '0123456789'
    contstr: str = ''
    needcont: int = 0
    contline: Optional[str] = None
    indents: List[int] = [0]
    while True:
        try:
            line: str = readline()
        except StopIteration:
            line = ''
        lnum = lnum + 1
        pos: int = 0
        maxpos: int = len(line)
        if contstr:
            if not line:
                raise TokenError('EOF in multi-line string', strstart)
            endmatch = endprog.match(line)
            if endmatch:
                pos = end = endmatch.end(0)
                yield (STRING, contstr + line[:end], strstart, (lnum, end), contline + line)  # type: ignore
                contstr, needcont = ('', 0)
                contline = None
            elif needcont and line[-2:] != '\\\n' and (line[-3:] != '\\\r\n'):
                yield (ERRORTOKEN, contstr + line, strstart, (lnum, len(line)), contline)  # type: ignore
                contstr = ''
                contline = None
                continue
            else:
                contstr = contstr + line
                contline = contline + line  # type: ignore
                continue
        elif parenlev == 0 and (not continued):
            if not line:
                break
            column: int = 0
            while pos < maxpos:
                if line[pos] == ' ':
                    column = column + 1
                elif line[pos] == '\t':
                    column = (column // tabsize + 1) * tabsize
                elif line[pos] == '\x0c':
                    column = 0
                else:
                    break
                pos = pos + 1
            if pos == maxpos:
                break
            if line[pos] in '#\r\n':
                if line[pos] == '#':
                    comment_token: str = line[pos:].rstrip('\r\n')
                    nl_pos: int = pos + len(comment_token)
                    yield (COMMENT, comment_token, (lnum, pos), (lnum, pos + len(comment_token)), line)
                    yield (NL, line[nl_pos:], (lnum, nl_pos), (lnum, len(line)), line)
                else:
                    yield (((NL, COMMENT)[line[pos] == '#']),
                           line[pos:], (lnum, pos), (lnum, len(line)), line)
                continue
            if column > indents[-1]:
                indents.append(column)
                yield (INDENT, line[:pos], (lnum, 0), (lnum, pos), line)
            while column < indents[-1]:
                if column not in indents:
                    raise IndentationError('unindent does not match any outer indentation level', ('<tokenize>', lnum, pos, line))
                indents = indents[:-1]
                yield (DEDENT, '', (lnum, pos), (lnum, pos), line)
        else:
            if not line:
                raise TokenError('EOF in multi-line statement', (lnum, 0))
            continued = 0
        while pos < maxpos:
            pseudomatch = pseudoprog.match(line, pos)
            if pseudomatch:
                start, end = pseudomatch.span(1)
                spos: Tuple[int, int] = (lnum, start)
                epos: Tuple[int, int] = (lnum, end)
                pos = end
                token_text: str = line[start:end]
                initial: str = line[start]
                if initial in numchars or (initial == '.' and token_text != '.'):
                    yield (NUMBER, token_text, spos, epos, line)
                elif initial in '\r\n':
                    newline: int = NEWLINE
                    if parenlev > 0:
                        newline = NL
                    yield (newline, token_text, spos, epos, line)
                elif initial == '#':
                    assert not token_text.endswith('\n')
                    yield (COMMENT, token_text, spos, epos, line)
                elif token_text in triple_quoted:
                    endprog = endprogs[token_text]
                    endmatch = endprog.match(line, pos) if endprog is not None else None
                    if endmatch:
                        pos = endmatch.end(0)
                        token_text = line[start:pos]
                        yield (STRING, token_text, spos, (lnum, pos), line)
                    else:
                        strstart: Tuple[int, int] = (lnum, start)
                        contstr = line[start:]
                        contline = line
                        break
                elif initial in single_quoted or token_text[:2] in single_quoted or token_text[:3] in single_quoted:
                    if token_text[-1] == '\n':
                        strstart = (lnum, start)
                        endprog = (endprogs[initial] or
                                   endprogs.get(token_text[1]) or
                                   endprogs.get(token_text[2]))
                        contstr, needcont = (line[start:], 1)
                        contline = line
                        break
                    else:
                        yield (STRING, token_text, spos, epos, line)
                elif initial in namechars:
                    yield (NAME, token_text, spos, epos, line)
                elif initial == '\\':
                    yield (NL, token_text, spos, (lnum, pos), line)
                    continued = 1
                else:
                    if initial in '([{':
                        parenlev = parenlev + 1
                    elif initial in ')]}':
                        parenlev = parenlev - 1
                    yield (OP, token_text, spos, epos, line)
            else:
                yield (ERRORTOKEN, line[pos], (lnum, pos), (lnum, pos + 1), line)
                pos = pos + 1
    for indent in indents[1:]:
        yield (DEDENT, '', (lnum, 0), (lnum, 0), '')
    yield (ENDMARKER, '', (lnum, 0), (lnum, 0), '')

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        tokenize(open(sys.argv[1]).readline)
    else:
        tokenize(sys.stdin.readline)