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
from typing import Final, Optional, Union, Any, TypeVar, cast, Generator, Match
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

def _combinations(*l: str) -> set[str]:
    return {x + y for x in l for y in l + ('',) if x.casefold() != y.casefold()}
Whitespace: str = '[ \\f\\t]*'
Comment: str = '#[^\\r\\n]*'
Ignore: str = Whitespace + any('\\\\\\r?\\n' + Whitespace) + maybe(Comment)
Name: str = '[^\\s#\\(\\)\\[\\]\\{\\}+\\-*/!@$%^&=|;:\'\\",\\.<>/?`~\\\\]+'
Binnumber: str = '0[bB]_?[01]+(?:_[01]+)*'
Hexnumber: str = '0[xX]_?[\\da-fA-F]+(?:_[\\da-fA-F]+)*[lL]?'
Octnumber: str = '0[oO]?_?[0-7]+(?:_[0-7]+)*[lL]?'
Decnumber: str = group('[1-9]\\d*(?:_\\d+)*[lL]?', '0[lL]?')
Intnumber: str = group(Binnumber, Hexnumber, Octnumber, Decnumber)
Exponent: str = '[eE][-+]?\\d+(?:_\\d+)*'
Pointfloat: str = group('\\d+(?:_\\d+)*\\.(?:\\d+(?:_\\d+)*)?', '\\.\\d+(?:_\\d+)*') + maybe(Exponent)
Expfloat: str = '\\d+(?:_\\d+)*' + Exponent
Floatnumber: str = group(Pointfloat, Expfloat)
Imagnumber: str = group('\\d+(?:_\\d+)*[jJ]', Floatnumber + '[jJ]')
Number: str = group(Imagnumber, Floatnumber, Intnumber)
Single: str = "(?:\\\\.|[^'\\\\])*'"
Double: str = '(?:\\\\.|[^"\\\\])*"'
Single3: str = "(?:\\\\.|'(?!'')|[^'\\\\])*'''"
Double3: str = '(?:\\\\.|"(?!"")|[^"\\\\])*"""'
_litprefix: str = '(?:[uUrRbB]|[rR][bB]|[bBuU][rR])?'
_fstringlitprefix: str = '(?:rF|FR|Fr|fr|RF|F|rf|f|Rf|fR)'
Triple: str = group(_litprefix + "'''", _litprefix + '"""', _fstringlitprefix + '"""', _fstringlitprefix + "'''")
SingleLbrace: str = "(?:\\\\N{|{{|\\\\'|[^\\n'{])*(?<!\\\\N)({)(?!{)"
DoubleLbrace: str = '(?:\\\\N{|{{|\\\\"|[^\\n"{])*(?<!\\\\N)({)(?!{)'
Single3Lbrace: str = "(?:\\\\N{|{{|\\\\'|'(?!'')|[^'{])*(?<!\\\\N){(?!{)"
Double3Lbrace: str = '(?:\\\\N{|{{|\\\\"|"(?!"")|[^"{])*(?<!\\\\N){(?!{)'
Bang: str = Whitespace + group('!') + '(?!=)'
bang: Pattern[str] = re.compile(Bang)
Colon: str = Whitespace + group(':')
colon: Pattern[str] = re.compile(Colon)
FstringMiddleAfterColon: str = group(Whitespace + '.*?') + group('{', '}')
fstring_middle_after_colon: Pattern[str] = re.compile(FstringMiddleAfterColon)
Operator: str = group('\\*\\*=?', '>>=?', '<<=?', '<>', '!=', '//=?', '->', '[+\\-*/%&@|^=<>:]=?', '~')
Bracket: str = '[][(){}]'
Special: str = group('\\r?\\n', '[:;.,`@]')
Funny: str = group(Operator, Bracket, Special)
_string_middle_single: str = "(?:[^\\n'\\\\]|\\\\.)*"
_string_middle_double: str = '(?:[^\\n"\\\\]|\\\\.)*'
_fstring_middle_single: str = SingleLbrace
_fstring_middle_double: str = DoubleLbrace
ContStr: str = group(_litprefix + "'" + _string_middle_single + group("'", '\\\\\\r?\\n'), _litprefix + '"' + _string_middle_double + group('"', '\\\\\\r?\\n'), group(_fstringlitprefix + "'") + _fstring_middle_single, group(_fstringlitprefix + '"') + _fstring_middle_double, group(_fstringlitprefix + "'") + _string_middle_single + group("'", '\\\\\\r?\\n'), group(_fstringlitprefix + '"') + _string_middle_double + group('"', '\\\\\\r?\\n'))
PseudoExtras: str = group('\\\\\\r?\\n', Comment, Triple)
PseudoToken: str = Whitespace + group(PseudoExtras, Number, Funny, ContStr, Name)
pseudoprog: Pattern[str] = re.compile(PseudoToken, re.UNICODE)
singleprog: Pattern[str] = re.compile(Single)
singleprog_plus_lbrace: Pattern[str] = re.compile(group(SingleLbrace, Single))
doubleprog: Pattern[str] = re.compile(Double)
doubleprog_plus_lbrace: Pattern[str] = re.compile(group(DoubleLbrace, Double))
single3prog: Pattern[str] = re.compile(Single3)
single3prog_plus_lbrace: Pattern[str] = re.compile(group(Single3Lbrace, Single3))
double3prog: Pattern[str] = re.compile(Double3)
double3prog_plus_lbrace: Pattern[str] = re.compile(group(Double3Lbrace, Double3))
_strprefixes: set[str] = _combinations('r', 'R', 'b', 'B') | {'u', 'U', 'ur', 'uR', 'Ur', 'UR'}
_fstring_prefixes: set[str] = _combinations('r', 'R', 'f', 'F') - {'r', 'R'}
endprogs: dict[str, Pattern[str]] = {"'": singleprog, '"': doubleprog, "'''": single3prog, '"""': double3prog, **{f"{prefix}'": singleprog for prefix in _strprefixes}, **{f'{prefix}"': doubleprog for prefix in _strprefixes}, **{f"{prefix}'": singleprog_plus_lbrace for prefix in _fstring_prefixes}, **{f'{prefix}"': doubleprog_plus_lbrace for prefix in _fstring_prefixes}, **{f"{prefix}'''": single3prog for prefix in _strprefixes}, **{f'{prefix}"""': double3prog for prefix in _strprefixes}, **{f"{prefix}'''": single3prog_plus_lbrace for prefix in _fstring_prefixes}, **{f'{prefix}"""': double3prog_plus_lbrace for prefix in _fstring_prefixes}}
triple_quoted: set[str] = {"'''", '"""'} | {f"{prefix}'''" for prefix in _strprefixes | _fstring_prefixes} | {f'{prefix}"""' for prefix in _strprefixes | _fstring_prefixes}
single_quoted: set[str] = {"'", '"'} | {f"{prefix}'" for prefix in _strprefixes | _fstring_prefixes} | {f'{prefix}"' for prefix in _strprefixes | _fstring_prefixes}
fstring_prefix: tuple[str, ...] = tuple({f"{prefix}'" for prefix in _fstring_prefixes} | {f'{prefix}"' for prefix in _fstring_prefixes} | {f"{prefix}'''" for prefix in _fstring_prefixes} | {f'{prefix}"""' for prefix in _fstring_prefixes})
tabsize: int = 8

class TokenError(Exception):
    pass

class StopTokenizing(Exception):
    pass
Coord = tuple[int, int]

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
GoodTokenInfo = tuple[int, str, Coord, Coord, str]
TokenInfo = Union[tuple[int, str], GoodTokenInfo]

class Untokenizer:

    def __init__(self) -> None:
        self.tokens: list[str] = []
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

    def compat(self, token: tuple[int, str], iterable: Iterable[tuple[int, str]]) -> None:
        startline = False
        indents: list[str] = []
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
cookie_re: Pattern[str] = re.compile('^[ \\t\\f]*#.*?coding[:=][ \\t]*([-\\w.]+)', re.ASCII)
blank_re: Pattern[bytes] = re.compile(b'^[ \\t\\f]*(?:[#\\r\\n]|$)', re.ASCII)

def _get_normal_name(orig_enc: str) -> str:
    """Imitates get_normal_name in tokenizer.c."""
    enc = orig_enc[:12].lower().replace('_', '-')
    if enc == 'utf-8' or enc.startswith('utf-8-'):
        return 'utf-8'
    if enc in ('latin-1', 'iso-8859-1', 'iso-latin-1') or enc.startswith(('latin-1-', 'iso-8859-1-', 'iso-latin-1-')):
        return 'iso-8859-1'
    return orig_enc

def detect_encoding(readline: Callable[[], bytes]) -> tuple[str, list[bytes]]:
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
    if not second:
        return (default, [first])
    encoding = find_cookie(second)
    if encoding:
        return (encoding, [first, second])
    return (default, [first, second])

def untokenize(iterable: Iterable[TokenInfo]) -> str:
    """Transform tokens back into Python source code.

    Each element returned by the iterable must be a token sequence
    with at least two elements, a token number and token value.  If
    only two tokens are passed, the resulting output is poor.

    Round-trip invariant for full input:
        Untokenized source will match input source exactly

    Round-trip invariant for limited input:
        # Output text will tokenize the back to the input
        t1 = [tok[:2] for tok in generate_tokens(f.readline)]
        newcode = untokenize(t1)
        readline = iter(newcode.splitlines(1)).next
        t2 = [tok[:2] for tokin generate_tokens(readline)]
        assert t1 == t2
    """
    ut = Untokenizer()
    return ut.untokenize(iterable)

def is_fstring_start(token: str) -> bool:
    return token.startswith(fstring_prefix)

def _split_fstring_start_and_middle(token: str) -> tuple[str, str]:
    for prefix in fstring_prefix:
        _, prefix, rest = token.partition(prefix)
        if prefix != '':
            return (prefix, rest)
    raise ValueError(f'Token {token!r} is not a valid f-string start')
STATE_NOT_FSTRING: Final[int] = 0
STATE_MIDDLE: Final[int] = 1
STATE_IN_BRACES: Final[int] = 2
STATE_IN_COLON: Final[int] = 3

class FStringState:
    """Keeps track of state around f-strings.

    The tokenizer should call the appropriate method on this class when
    it transitions to a different part of an f-string. This is needed
    because the tokenization depends on knowing where exactly we are in
    the f-string.

    For example, consider the following f-string:

        f"a{1:b{2}c}d"

    The following is the tokenization of this string and the states
    tracked by this class:

        1,0-1,2:	FSTRING_START	'f"'  # [STATE_NOT_FSTRING, STATE_MIDDLE]
        1,2-1,3:	FSTRING_MIDDLE	'a'
        1,3-1,4:	LBRACE	'{'  # [STATE_NOT_FSTRING, STATE_IN_BRACES]
        1,4-1,5:	NUMBER	'1'
        1,5-1,6:	OP	':'  # [STATE_NOT_FSTRING, STATE_IN_COLON]
        1,6-1,7:	FSTRING_MIDDLE	'b'
        1,7-1,8:	LBRACE	'{'  # [STATE_NOT_FSTRING, STATE_IN_COLON, STATE_IN_BRACES]
        1,8-1,9:	NUMBER	'2'
        1,9-1,10:	RBRACE	'}'  # [STATE_NOT_FSTRING, STATE_IN_COLON]
        1,10-1,11:	FSTRING_MIDDLE	'c'
        1,11-1,12:	RBRACE	'}'  # [STATE_NOT_FSTRING, STATE_MIDDLE]
        1,12-1,13:	FSTRING_MIDDLE	'd'
        1,13-1,14:	FSTRING_END	'"'  # [STATE_NOT_FSTRING]
        1,14-1,15:	NEWLINE	'
'
        2,0-2,0:	ENDMARKER	''

    Notice that the nested braces in the format specifier are represented
    by adding a STATE_IN_BRACES entry to the state stack. The stack is
    also used if there are nested f-strings.

    """

    def __init__(self) -> None:
        self.stack: list[int] = [STATE_NOT_FSTRING]

    def is_in_fstring_expression(self) -> bool:
        return self.stack[-1] not in (STATE_MIDDLE, STATE_NOT_FSTRING)

    def current(self) -> int:
        return self.stack[-1]

    def enter_fstring(self) -> None:
        self.stack.append(STATE_MIDDLE)

    def leave_fstring(self) -> None:
        state = self.stack.pop()
        assert state == STATE_MIDDLE

    def consume_lbrace(self) -> None:
        current_state = self.stack[-1]
        if current_state == STATE_MIDDLE:
            self.stack[-1] = STATE_IN_BRACES
        elif current_state == STATE_IN_COLON:
            self.stack.append(STATE_IN_BRACES)
        else:
            assert False, current_state

    def consume_rbrace(self) -> None:
        current_state = self.stack[-1]
        assert current_state in (STATE_IN_BRACES, STATE_IN_COLON)
        if len(self.stack) > 1 and self.stack[-2] == STATE_IN_COLON:
            self.stack.pop()
        else:
            self.stack[-1] = STATE_MIDDLE

    def consume_colon(self) -> None:
        assert self.stack[-1] == STATE_IN_BRACES, self.stack
        self.stack[-1] = STATE_IN_COLON

def generate_tokens(readline: Callable[[], str], grammar: Optional[Grammar] = None) -> Generator[GoodTokenInfo, None, None]:
    """
    The generate_tokens() generator requires one argument, readline, which
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
    parenlev_stack: list[int] = []
    fstring_state = FStringState()
    formatspec = ''
    formatspec_start: Coord = (0, 0)
    numchars = '0123456789'
    contstr, needcont = ('', 0)
    contline: Optional[str] = None
    indents = [0]
    async_keywords = False if grammar is None else grammar.async_keywords
    stashed: Optional[GoodTokenInfo] = None
    async_def = False
    async_def_indent = 0
    async_def_nl = False
    endprog_stack: list[Pattern[str]] = []
    strstart: Coord = (0, 0)
    while 1:
        try:
            line = readline()
        except StopIteration:
            line = ''
        lnum += 1
        if not contstr and line.rstrip('\n').strip(' \t\x0c') == '\\':
            continue
        pos, max = (0, len(line))
        if contstr:
            assert contline is not None
            if not line:
                raise TokenError('EOF in multi-line string', strstart)
            endprog = endprog_stack[-1]
            endmatch = endprog.match(line)
            if endmatch:
                end = endmatch.end(0)
                token = contstr + line[:end]
                spos = strstart
                epos = (lnum, end)
                tokenline = contline + line
                if fstring_state.current() in (STATE_NOT_FSTRING, STATE_IN_BRACES) and (not is_fstring_start(token)):
                    yield (STRING, token, spos, epos, tokenline)
                    endprog_stack.pop()
                    parenlev = parenlev_stack.pop()
                else:
                    if is_fstring_start(token):
                        fstring_start, token = _split_fstring_start_and_middle(token)
                        fstring_start_epos = (spos[0], spos[1] + len(fstring_start))
                        yield (FSTRING_START, fstring_start, spos, fstring_start_epos, tokenline)
                        fstring_state.enter_fstring()
                        spos = fstring_start_epos
                    if token.endswith('{'):
                        fstring_middle, lbrace = (token[:-1], token[-1])
                        fstring_middle_epos = lbrace_spos = (lnum, end - 1)
                        yield (FSTRING_MIDDLE, fstring_middle, spos, fstring_middle_epos, line)
                        yield (LBRACE, lbrace, lbrace_spos, epos, line)
                        fstring_state.consume_lbrace()
                    else:
                        if token.endswith(('"""', "'''")):
                            fstring_middle, fstring_end = (token[:-3], token[-3:])
                            fstring_middle_epos = end_spos = (lnum, end - 3)
                        else:
                            fstring_middle, fstring_end = (token[:-1], token[-1])
                            fstring_middle_epos = end_spos = (lnum, end - 1)
                        yield (FSTRING_MIDDLE, fstring_middle, spos, fstring_middle_epos, line)
                        yield (FSTRING_END, fstring_end, end_spos, epos, line)
                        fstring_state.leave_fstring()
                        endprog_stack.pop()
                        parenlev = parenlev_stack.pop()
                pos = end
                contstr, needcont = ('', 0)
                contline = None
            elif needcont and line[-2:] != '\\\n' and (line[-3:] != '\\\r\n'):
                yield (ERRORTOKEN, contstr + line, strstart, (lnum, len(line)), contline)
                contstr = ''
                contline = None
                continue
            else:
                contstr = contstr + line
                contline = contline + line
                continue
        elif parenlev == 0 and (not continued) and (not fstring_state.is_in_fstring_expression()):
            if not line:
                break
            column = 0
            while pos < max:
                if line[pos] == ' ':
                    column += 1
                elif line[pos] == '\t':
                    column = (column // tabsize + 1) * tabsize
                elif line[pos] == '\x0c':
                    column = 0
                else:
                    break
                pos += 1
            if pos == max:
                break
            if stashed:
                yield stashed
                stashed = None
            if line[pos] in '\r\n':
                yield (NL, line[pos:], (lnum, pos), (lnum, len(line)), line)
                continue
            if line[pos] == '#':
                comment_token = line[pos:].rstrip('\r\n')
                nl_pos = pos + len(comment_token)
                yield (COMMENT, comment_token, (lnum, pos), (lnum, nl_pos), line)
                yield (NL, line[nl_pos:], (lnum, nl_pos), (lnum, len(line)), line)
                continue
            if column > indents[-1]:
                indents.append(column)
                yield (INDENT, line[:pos], (lnum, 0), (lnum, pos), line)
            while column < indents[-1]:
                if column not in indents:
                    raise IndentationError('unindent does not match any outer indentation level', ('<tokenize>', lnum, pos, line))
                indents = indents[:-1]
                if async_def and async_def_indent >= indents[-1]:
                    async_def = False
                    async_def_nl = False
                    async_def_indent = 0
                yield (DEDENT, '', (lnum, pos), (lnum, pos), line)
            if async_def and async_def_nl and (async_def_indent >= indents[-1]):
                async_def = False
                async_def_nl = False
                async_def_indent = 0
        else:
            if not line:
                raise TokenError('EOF in multi-line statement', (lnum, 0))
            continued = 0
        while pos < max:
            if fstring_state.current() == STATE_MIDDLE:
                endprog = endprog_stack[-1]
                endmatch = endprog.match(line, pos)
                if endmatch:
                    start, end = endmatch.span(0)
                    token = line[start:end]
                    if token.endswith(('"""', "'''")):
                        middle_token, end_token = (token[:-3], token[-3:])
                        middle_epos = end_spos = (lnum, end - 3)
                    else:
                        middle_token, end_token = (token[:-1], token[-1])
                        middle_epos = end_spos = (lnum, end - 1)
                    if stashed:
                        yield stashed
                        stashed = None
                    yield (FSTRING_MIDDLE, middle_token, (lnum, pos), middle_epos, line)
                    if not token.endswith('{'):
                        yield (FSTRING_END, end_token, end_spos, (lnum, end), line)
                        fstring_state.leave_fstring()
                        endprog_stack.pop()
                        parenlev = parenlev_stack.pop()
                    else:
                        yield (LBRACE, '{', (lnum, end - 1), (lnum, end), line)
                        fstring_state.consume_lbrace()
                    pos = end
                    continue
                else:
                    strstart = (lnum, end)
                    contstr = line[end:]
                    contline = line
                    break
            if fstring_state.current() == STATE_IN_COLON:
                match = fstring_middle_after_colon.match(line, pos)
                if match is None:
                    formatspec += line[pos:]
                    pos = max
                    continue
                start, end = match.span(1)
                token = line[start:end]
                formatspec += token
                brace_start, brace_end = match.span(2)
                brace_or_nl = line[brace_start:brace_end]
                if brace_or_nl == '\n':
                    pos = brace_end
                yield (FSTRING_MIDDLE, formatspec, formatspec_start, (lnum, end), line)
                formatspec = ''
                if brace_or_nl == '{':
                    yield (LBRACE, '{', (lnum, brace_start), (lnum, brace_end), line)
                    fstring_state.consume_lbrace()
                    end = brace_end
                elif brace_or_nl == '}':
                    yield (RBRACE, '}', (lnum, brace_start), (lnum, brace_end), line)
                    fstring_state.consume_rbrace()
                    end = brace_end
                    formatspec_start = (lnum, brace_end)
                pos = end
                continue
            if fstring_state.current() == STATE_IN_BRACES and parenlev == 0:
                match = bang.match(line, pos)
                if match:
                    start, end = match.span(1)
                    yield (OP, '!', (lnum, start), (lnum, end), line)
                    pos = end
                    continue
                match = colon.match(line, pos)
                if match:
                    start, end = match.span(1)
                    yield (OP, ':', (lnum, start), (lnum, end), line)
                    fstring_state.consume_colon()
                    formatspec_start = (lnum, end)
                    pos = end
                    continue
            pseudomatch = pseudoprog.match(line, pos)
            if pseudomatch:
                start, end = pseudomatch.span(1)
                spos, epos, pos = ((lnum, start), (lnum, end), end)
                token, initial = (line[start:end], line[start])
                if initial in numchars or (initial == '.' and token != '.'):
                    yield (NUMBER, token, spos, epos, line)
                elif initial in '\r\n':
                    newline = NEWLINE
                    if parenlev > 0 or fstring_state.is_in_fstring_expression():
                        newline = NL
                    elif async_def:
                        async_def_nl = True
                    if stashed:
                        yield stashed
                        stashed = None
                    yield (newline, token, spos, epos, line)
                elif initial == '#':
                    assert not token.endswith('\n')
                    if stashed:
                        yield stashed
                        stashed = None
                    yield (COMMENT, token, spos, epos, line)
                elif token in triple_quoted:
                    endprog = endprogs[token]
                    endprog_stack.append(endprog)
                    parenlev_stack.append(parenlev)
                    parenlev = 0
                    if is_fstring_start(token):
                        yield (FSTRING_START, token, spos, epos, line)
                        fstring_state.enter_fstring()
                    endmatch = endprog.match(line, pos)
                    if endmatch:
                        if stashed:
                            yield stashed
                            stashed = None
                        if not is_fstring_start(token):
                            pos = endmatch.end(0)
                            token = line[start:pos]
                            epos = (lnum, pos)
                            yield (STRING, token, spos, epos, line)
                            endprog_stack.pop()
                            parenlev = parenlev_stack.pop()
                        else:
                            end = endmatch.end(0)
                            token = line[pos:end]
                            spos, epos = ((lnum, pos), (lnum, end))
                            if not token.endswith('{'):
                                fstring_middle, fstring_end = (token[:-3], token[-3:])
                                fstring_middle_epos = fstring_end_spos = (lnum, end - 3)
                                yield (FSTRING_MIDDLE, fstring_middle, spos, fstring_middle_epos, line)
                                yield (FSTRING_END, fstring_end, fstring_end_spos, epos, line)
                                fstring_state.leave_fstring()
                                endprog_stack.pop()
                                parenlev = parenlev_stack.pop()
                            else:
                                fstring_middle, lbrace = (token[:-1], token[-1])
                                fstring_middle_epos = lbrace_spos = (lnum, end - 1)
                                yield (FSTRING_MIDDLE, fstring_middle, spos, fstring_middle_epos, line)
                                yield (LBRACE, lbrace, lbrace_spos, epos, line)
                                fstring_state.consume_lbrace()
                            pos = end
                    else:
                        if is_fstring_start(token):
                            strstart = (lnum, pos)
                            contstr = line[pos:]
                        else:
                            strstart = (lnum, start)
                            contstr = line[start:]
                        contline = line
                        break
                elif initial in single_quoted or token[:2] in single_quoted or token[:3] in single_quoted:
                    maybe_endprog = endprogs.get(initial) or endprogs.get(token[:2]) or endprogs.get(token[:3])
                    assert maybe_endprog is not None, f'endprog not found for {token}'
                    endprog = maybe_endprog
                    if token[-1] == '\n':
                        endprog_stack.append(endprog)
                        parenlev_stack.append(parenlev)
                        parenlev = 0
                        strstart = (lnum, start)
                        contstr, needcont = (line[start:], 1)
                        contline = line
                        break
                    else:
                        if stashed:
                            yield stashed
                            stashed = None
                        if not is_fstring_start(token):
                            yield (STRING, token, spos, epos, line)
                        else:
                            if pseudomatch[20] is not None:
                                fstring_start = pseudomatch[20]
                                offset = pseudomatch.end(20) - pseudomatch.start(1)
                            elif pseudomatch[22] is not None:
                                fstring_start = pseudomatch[22]
                                offset = pseudomatch.end(22) - pseudomatch.start(1)
                            elif pseudomatch[24] is not None:
                                fstring_start = pseudomatch[24]
                                offset = pseudomatch.end(24) - pseudomatch.start(1)
                            else:
                                fstring_start = pseudomatch[26]
                                offset = pseudomatch.end(26) - pseudomatch.start(1)
                            start_epos = (lnum, start + offset)
                            yield (FSTRING_START, fstring_start, spos, start_epos, line)
                            fstring_state.enter_fstring()
                            endprog = endprogs[fstring_start]
                            endprog_stack.append(endprog)
                            parenlev_stack.append(parenlev)
                            parenlev = 0
                            end_offset = pseudomatch.end(1) - 1
                            fstring_middle = line[start + offset:end_offset]
                            middle_spos = (lnum, start + offset)
                            middle_epos = (lnum, end_offset)
                            yield (FSTRING_MIDDLE, fstring_middle, middle_spos, middle_epos, line)
                            if not token.endswith('{'):
                                end_spos = (lnum, end_offset)
                                end_epos = (lnum, end_offset + 1)
                                yield (FSTRING_END, token[-1], end_spos, end_epos, line)
                                fstring_state.leave_fstring()
                                endprog_stack.pop()
                                parenlev = parenlev_stack.pop()
                            else:
                                end_spos = (lnum, end_offset)
                                end_epos = (lnum, end_offset + 1)
                                yield (LBRACE, '{', end_spos, end_epos, line)
                                fstring_state.consume_lbrace()
                elif initial.isidentifier():
                    if token in ('async', 'await'):
                        if async_keywords or async_def:
                            yield (ASYNC if token == 'async' else AWAIT, token, spos, epos, line)
                            continue
                    tok: GoodTokenInfo = (NAME, token, spos, epos, line)
                    if token == 'async' and (not stashed):
                        stashed = tok
                        continue
                    if token in ('def', 'for'):
                        if stashed and stashed[0] == NAME and (stashed[1] == 'async'):
                            if token == 'def':
                                async_def = True
                                async_def_indent = indents[-1]
                            yield (ASYNC, stashed[1], stashed[2], stashed[3], stashed[4])
                            stashed = None
                    if stashed:
                        yield stashed
                        stashed = None
                    yield tok
                elif initial == '\\':
                    if stashed:
                        yield stashed
                        stashed = None
                    yield (NL, token, spos, (lnum, pos), line)
                    continued = 1
                elif initial == '}' and parenlev == 0 and fstring_state.is_in_fstring_expression():
                    yield (RBRACE, token, spos, epos, line)
                    fstring_state.consume_rbrace()
                    formatspec_start = epos
                else:
                    if initial in '([{':
                        parenlev += 1
                    elif initial in ')]}':
                        parenlev -= 1
                    if stashed:
                        yield stashed
                        stashed = None
                    yield (OP, token, spos, epos, line)
            else:
                yield (ERRORTOKEN, line[pos], (lnum, pos), (lnum, pos + 1), line)
                pos += 1
    if stashed:
        yield stashed
        stashed = None
    for _indent in indents[1:]:
        yield (DEDENT, '', (lnum, 0), (lnum, 0), '')
    yield (ENDMARKER, '', (lnum, 0), (lnum, 0), '')
    assert len(endprog_stack) == 0
    assert len(parenlev_stack) == 0
if __name__ == '__main__':
    if len(sys.argv) > 1:
        tokenize(open(sys.argv[1]).readline)
    else:
        tokenize(sys.stdin.readline)
