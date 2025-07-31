"""
This tokenizer has been copied from the ``tokenize.py`` standard library
tokenizer. The reason was simple: The standard library tokenizer fails
if the indentation is not right. To make it possible to do error recovery the
    tokenizer needed to be rewritten.

Basically this is a stripped down version of the standard library module, so
you can read the documentation there. Additionally we included some speed and
memory optimizations here.
"""
from __future__ import absolute_import
import sys
import re
import itertools as _itertools
from codecs import BOM_UTF8
from typing import NamedTuple, Tuple, Iterator, Iterable, List, Dict, Pattern, Set, Optional, Union, Any, Callable
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
    pseudo_token_compiled: Pattern[str]
    single_quoted: Set[str]
    triple_quoted: Set[str]
    endpats: Dict[str, Pattern[str]]
    whitespace: Pattern[str]
    fstring_pattern_map: Dict[str, str]
    always_break_tokens: Set[str]
BOM_UTF8_STRING = BOM_UTF8.decode('utf-8')
_token_collection_cache: Dict[Tuple[int, ...], TokenCollection] = {}

def group(*choices: str, capture: bool = False, **kwargs: Any) -> str:
    assert not kwargs
    start = '('
    if not capture:
        start += '?:'
    return start + '|'.join(choices) + ')'

def maybe(*choices: str) -> str:
    return group(*choices) + '?'

def _all_string_prefixes(*, include_fstring: bool = False, only_fstring: bool = False) -> Set[str]:

    def different_case_versions(prefix: str) -> Iterator[str]:
        for s in _itertools.product(*[(c, c.upper()) for c in prefix]):
            yield ''.join(s)
    valid_string_prefixes: List[str] = ['b', 'r', 'u', 'br']
    result: Set[str] = {''}
    if include_fstring:
        f = ['f', 'fr']
        if only_fstring:
            valid_string_prefixes = f
            result = set()
        else:
            valid_string_prefixes += f
    elif only_fstring:
        return set()
    for prefix in valid_string_prefixes:
        for t in _itertools.permutations(prefix):
            result.update(different_case_versions(t))
    return result

def _compile(expr: str) -> Pattern[str]:
    return re.compile(expr, re.UNICODE)

def _get_token_collection(version_info: PythonVersionInfo) -> TokenCollection:
    try:
        return _token_collection_cache[tuple(version_info)]
    except KeyError:
        _token_collection_cache[tuple(version_info)] = result = _create_token_collection(version_info)
        return result
unicode_character_name = '[A-Za-z0-9\\-]+(?: [A-Za-z0-9\\-]+)*'
fstring_string_single_line = _compile('(?:\\{\\{|\\}\\}|\\\\N\\{' + unicode_character_name + '\\}|\\\\(?:\\r\\n?|\\n)|\\\\[^\\r\\nN]|[^{}\\r\\n\\\\])+')
fstring_string_multi_line = _compile('(?:\\{\\{|\\}\\}|\\\\N\\{' + unicode_character_name + '\\}|\\\\[^N]|[^{}\\\\])+')
fstring_format_spec_single_line = _compile('(?:\\\\(?:\\r\\n?|\\n)|[^{}\\r\\n])+')
fstring_format_spec_multi_line = _compile('[^{}]+')

def _create_token_collection(version_info: PythonVersionInfo) -> TokenCollection:
    Whitespace = '[ \\f\\t]*'
    whitespace = _compile(Whitespace)
    Comment = '#[^\\r\\n]*'
    Name = '([A-Za-z_0-9\x80-' + MAX_UNICODE + ']+)'
    Hexnumber = '0[xX](?:_?[0-9a-fA-F])+'
    Binnumber = '0[bB](?:_?[01])+'
    Octnumber = '0[oO](?:_?[0-7])+'
    Decnumber = '(?:0(?:_?0)*|[1-9](?:_?[0-9])*)'
    Intnumber = group(Hexnumber, Binnumber, Octnumber, Decnumber)
    Exponent = '[eE][-+]?[0-9](?:_?[0-9])*'
    Pointfloat = group('[0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?', '\\.[0-9](?:_?[0-9])*') + maybe(Exponent)
    Expfloat = '[0-9](?:_?[0-9])*' + Exponent
    Floatnumber = group(Pointfloat, Expfloat)
    Imagnumber = group('[0-9](?:_?[0-9])*[jJ]', Floatnumber + '[jJ]')
    Number = group(Imagnumber, Floatnumber, Intnumber)
    possible_prefixes = _all_string_prefixes()
    StringPrefix = group(*possible_prefixes)
    StringPrefixWithF = group(*_all_string_prefixes(include_fstring=True))
    fstring_prefixes = _all_string_prefixes(include_fstring=True, only_fstring=True)
    FStringStart = group(*fstring_prefixes)
    Single = "(?:\\\\.|[^'\\\\])*'"
    Double = '(?:\\\\.|[^"\\\\])*"'
    Single3 = "(?:\\\\.|'(?!'')|[^'\\\\])*'''"
    Double3 = '(?:\\\\.|"(?!"")|[^"\\\\])*"""'
    Triple = group(StringPrefixWithF + "'''", StringPrefixWithF + '"""')
    Operator = group('\\*\\*=?', '>>=?', '<<=?', '//=?', '->', '[+\\-*/%&@`|^!=<>]=?', '~')
    Bracket = '[][(){}]'
    special_args: List[str] = ['\\.\\.\\.', '\\r\\n?', '\\n', '[;.,@]']
    if version_info >= (3, 8):
        special_args.insert(0, ':=?')
    else:
        special_args.insert(0, ':')
    Special = group(*special_args)
    Funny = group(Operator, Bracket, Special)
    ContStr = group(StringPrefix + "'[^\\r\\n'\\\\]*(?:\\\\.[^\\r\\n'\\\\]*)*" + group("'", '\\\\(?:\\r\\n?|\\n)'), StringPrefix + '"[^\\r\\n"\\\\]*(?:\\\\.[^\\r\\n"\\\\]*)*' + group('"', '\\\\(?:\\r\\n?|\\n)'))
    pseudo_extra_pool: List[str] = [Comment, Triple]
    all_quotes = ('"', "'", '"""', "'''")
    if fstring_prefixes:
        pseudo_extra_pool.append(FStringStart + group(*all_quotes))
    PseudoExtras = group('\\\\(?:\\r\\n?|\\n)|\\Z', *pseudo_extra_pool)
    PseudoToken = group(Whitespace, capture=True) + group(PseudoExtras, Number, Funny, ContStr, Name, capture=True)
    endpats: Dict[str, Pattern[str]] = {}
    for _prefix in possible_prefixes:
        endpats[_prefix + "'"] = _compile(Single)
        endpats[_prefix + '"'] = _compile(Double)
        endpats[_prefix + "'''"] = _compile(Single3)
        endpats[_prefix + '"""'] = _compile(Double3)
    single_quoted: Set[str] = set()
    triple_quoted: Set[str] = set()
    fstring_pattern_map: Dict[str, str] = {}
    for t in possible_prefixes:
        for quote in ('"', "'"):
            single_quoted.add(t + quote)
        for quote in ('"""', "'''"):
            triple_quoted.add(t + quote)
    for t in fstring_prefixes:
        for quote in all_quotes:
            fstring_pattern_map[t + quote] = quote
    ALWAYS_BREAK_TOKENS = (';', 'import', 'class', 'def', 'try', 'except', 'finally', 'while', 'with', 'return', 'continue', 'break', 'del', 'pass', 'global', 'assert', 'nonlocal')
    pseudo_token_compiled = _compile(PseudoToken)
    return TokenCollection(pseudo_token_compiled, single_quoted, triple_quoted, endpats, whitespace, fstring_pattern_map, set(ALWAYS_BREAK_TOKENS))

class Token(NamedTuple):
    type: PythonTokenTypes
    string: str
    start_pos: Tuple[int, int]
    prefix: str

    @property
    def end_pos(self) -> Tuple[int, int]:
        lines = split_lines(self.string)
        if len(lines) > 1:
            return (self.start_pos[0] + len(lines) - 1, 0)
        else:
            return (self.start_pos[0], self.start_pos[1] + len(self.string))

class PythonToken(Token):

    def __repr__(self) -> str:
        return 'TokenInfo(type=%s, string=%r, start_pos=%r, prefix=%r)' % self._replace(type=self.type.name)

class FStringNode:

    def __init__(self, quote: str) -> None:
        self.quote = quote
        self.parentheses_count = 0
        self.previous_lines = ''
        self.last_string_start_pos: Optional[Tuple[int, int]] = None
        self.format_spec_count = 0

    def open_parentheses(self, character: str) -> None:
        self.parentheses_count += 1

    def close_parentheses(self, character: str) -> None:
        self.parentheses_count -= 1
        if self.parentheses_count == 0:
            self.format_spec_count = 0

    def allow_multiline(self) -> bool:
        return len(self.quote) == 3

    def is_in_expr(self) -> bool:
        return self.parentheses_count > self.format_spec_count

    def is_in_format_spec(self) -> bool:
        return not self.is_in_expr() and self.format_spec_count

def _close_fstring_if_necessary(fstring_stack: List[FStringNode], string: str, line_nr: int, column: int, additional_prefix: str) -> Tuple[Optional[PythonToken], str, int]:
    for fstring_stack_index, node in enumerate(fstring_stack):
        lstripped_string = string.lstrip()
        len_lstrip = len(string) - len(lstripped_string)
        if lstripped_string.startswith(node.quote):
            token = PythonToken(FSTRING_END, node.quote, (line_nr, column + len_lstrip), prefix=additional_prefix + string[:len_lstrip])
            additional_prefix = ''
            assert not node.previous_lines
            del fstring_stack[fstring_stack_index:]
            return (token, '', len(node.quote) + len_lstrip)
    return (None, additional_prefix, 0)

def _find_fstring_string(endpats: Dict[str, Pattern[str]], fstring_stack: List[FStringNode], line: str, lnum: int, pos: int) -> Tuple[str, int]:
    tos = fstring_stack[-1]
    allow_multiline = tos.allow_multiline()
    if tos.is_in_format_spec():
        if allow_multiline:
            regex = fstring_format_spec_multi_line
        else:
            regex = fstring_format_spec_single_line
    elif allow_multiline:
        regex = fstring_string_multi_line
    else:
        regex = fstring_string_single_line
    match = regex.match(line, pos)
    if match is None:
        return (tos.previous_lines, pos)
    if not tos.previous_lines:
        tos.last_string_start_pos = (lnum, pos)
    string = match.group(0)
    for fstring_stack_node in fstring_stack:
        end_match = endpats[fstring_stack_node.quote].match(string)
        if end_match is not None:
            string = end_match.group(0)[:-len(fstring_stack_node.quote)]
    new_pos = pos
    new_pos += len(string)
    if string.endswith('\n') or string.endswith('\r'):
        tos.previous_lines += string
        string = ''
    else:
        string = tos.previous_lines + string
    return (string, new_pos)

def tokenize(code: str, *, version_info: PythonVersionInfo, start_pos: Tuple[int, int] = (1, 0)) -> Iterator[PythonToken]:
    """Generate tokens from a the source code (string)."""
    lines = split_lines(code, keepends=True)
    return tokenize_lines(lines, version_info=version_info, start_pos=start_pos)

def _print_tokens(func: Callable[..., Iterator[PythonToken]]) -> Callable[..., Iterator[PythonToken]]:
    """
    A small helper function to help debug the tokenize_lines function.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Iterator[PythonToken]:
        for token in func(*args, **kwargs):
            print(token)
            yield token
    return wrapper

def tokenize_lines(lines: Iterable[str], *, version_info: PythonVersionInfo, indents: Optional[List[int]] = None, start_pos: Tuple[int, int] = (1, 0), is_first_token: bool = True) -> Iterator[PythonToken]:
    """
    A heavily modified Python standard library tokenizer.

    Additionally to the default information, yields also the prefix of each
    token. This idea comes from lib2to3. The prefix contains all information
    that is irrelevant for the parser like newlines in parentheses or comments.
    """

    def dedent_if_necessary(start: int) -> Iterator[PythonToken]:
        while start < indents[-1]:
            if start > indents[-2]:
                yield PythonToken(ERROR_DEDENT, '', (lnum, start), '')
                indents[-1] = start
                break
            indents.pop()
            yield PythonToken(DEDENT, '', spos, '')
    pseudo_token, single_quoted, triple_quoted, endpats, whitespace, fstring_pattern_map, always_break_tokens = _get_token_collection(version_info)
    paren_level = 0
    if indents is None:
        indents = [0]
    max_ = 0
    numchars = '0123456789'
    contstr = ''
    new_line = True
    prefix = ''
    additional_prefix = ''
    lnum = start_pos[0] - 1
    fstring_stack: List[FStringNode] = []
    contstr_start: Tuple[int, int] = (0, 0)
    endprog: Optional[Pattern[str]] = None
    contline = ''
    for line in lines:
        lnum += 1
        pos = 0
        max_ = len(line)
        if is_first_token:
            if line.startswith(BOM_UTF8_STRING):
                additional_prefix = BOM_UTF8_STRING
                line = line[1:]
                max_ = len(line)
            line = '^' * start_pos[1] + line
            pos = start_pos[1]
            max_ += start_pos[1]
            is_first_token = False
        if contstr:
            assert endprog is not None
            endmatch = endprog.match(line)
            if endmatch:
                pos = endmatch.end(0)
                yield PythonToken(STRING, contstr + line[:pos], contstr_start, prefix)
                contstr = ''
                contline = ''
            else:
                contstr = contstr + line
                contline = contline + line
                continue
        while pos < max_:
            if fstring_stack:
                tos = fstring_stack[-1]
                if not tos.is_in_expr():
                    string, pos = _find_fstring_string(endpats, fstring_stack, line, lnum, pos)
                    if string:
                        yield PythonToken(FSTRING_STRING, string, tos.last_string_start_pos, prefix='')
                        tos.previous_lines = ''
                        continue
                    if pos == max_:
                        break
                rest = line[pos:]
                fstring_end_token, additional_prefix, quote_length = _close_fstring_if_necessary(fstring_stack, rest, lnum, pos, additional_prefix)
                pos += quote_length
                if fstring_end_token is not None:
                    yield fstring_end_token
                    continue
            if fstring_stack:
                string_line = line
                for fstring_stack_node in fstring_stack:
                    quote = fstring_stack_node.quote
                    end_match = endpats[quote].match(line, pos)
                    if end_match is not None:
                        end_match_string = end_match.group(0)
                        if len(end_match_string) - len(quote) + pos < len(string_line):
                            string_line = line[:pos] + end_match_string[:-len(quote)]
                pseudomatch = pseudo_token.match(string_line, pos)
            else:
                pseudomatch = pseudo_token.match(line, pos)
            if pseudomatch:
                prefix = additional_prefix + pseudomatch.group(1)
                additional_prefix = ''
                start, pos = pseudomatch.span(2)
                spos = (lnum, start)
                token = pseudomatch.group(2)
                if token == '':
                    assert prefix
                    additional_prefix = prefix
                    break
                initial = token[0]
            else:
                match = whitespace.match(line, pos)
                initial = line[match.end()]
                start = match.end()
                spos = (lnum, start)
            if new_line and initial not in '\r\n#' and (initial != '\\' or pseudomatch is None):
                new_line = False
                if paren_level == 0 and (not fstring_stack):
                    indent_start = start
                    if indent_start > indents[-1]:
                        yield PythonToken(INDENT, '', spos, '')
                        indents.append(indent_start)
                    yield from dedent_if_necessary(indent_start)
            if not pseudomatch:
                match = whitespace.match(line, pos)
                if new_line and paren_level == 0 and (not fstring_stack):
                    yield from dedent_if_necessary(match.end())
                pos = match.end()
                new_line = False
                yield PythonToken(ERRORTOKEN, line[pos], (lnum, pos), additional_prefix + match.group(0))
                additional_prefix = ''
                pos += 1
                continue
            if initial in numchars or (initial == '.' and token != '.' and (token != '...')):
                yield PythonToken(NUMBER, token, spos, prefix)
            elif pseudomatch.group(3) is not None:
                if token in always_break_tokens and (fstring_stack or paren_level):
                    fstring_stack[:] = []
                    paren_level = 0
                    m = re.match('[ \\f\\t]*$', line[:start])
                    if m is not None:
                        yield from dedent_if_necessary(m.end())
                if token.isidentifier():
                    yield PythonToken(NAME, token, spos, prefix)
                else:
                    yield from _split_illegal_unicode_name(token, spos, prefix)
            elif initial in '\r\n':
                if any((not f.allow_multiline() for f in fstring_stack)):
                    fstring_stack.clear()
                if not new_line and paren_level == 0 and (not fstring_stack):
                    yield PythonToken(NEWLINE, token, spos, prefix)
                else:
                    additional_prefix = prefix + token
                new_line = True
            elif initial == '#':
                assert not token.endswith('\n') and (not token.endswith('\r'))
                if fstring_stack and fstring_stack[-1].is_in_expr():
                    yield PythonToken(ERRORTOKEN, initial, spos, prefix)
                    pos = start + 1
                else:
                    additional_prefix = prefix + token
            elif token in triple_quoted:
                endprog = endpats[token]
                endmatch = endprog.match(line, pos)
                if endmatch:
                    pos = endmatch.end(0)
                    token = line[start:pos]
                    yield PythonToken(STRING, token, spos, prefix)
                else:
                    contstr_start = spos
                    contstr = line[start:]
                    contline = line
                    break
            elif initial in single_quoted or token[:2] in single_quoted or token[:3] in single_quoted:
                if token[-1] in '\r\n':
                    contstr_start = (lnum, start)
                    endprog = endpats.get(initial) or endpats.get(token[1]) or endpats.get(token[2])
                    contstr = line[start:]
                    contline = line
                    break
                else:
                    yield PythonToken(STRING, token, spos, prefix)
            elif token in fstring_pattern_map:
                fstring_stack.append(FStringNode(fstring_pattern_map[token]))
                yield PythonToken(FSTRING_START, token, spos, prefix)
            elif initial == '\\' and line[start:] in ('\\\n', '\\\r\n', '\\\r'):
                additional_prefix += prefix + line[start:]
                break
            else:
                if token in '([{':
                    if fstring_stack:
                        fstring_stack[-1].open_parentheses(token)
                    else:
                        paren_level += 1
                elif token in ')]}':
                    if fstring_stack:
                        fstring_stack[-1].close_parentheses(token)
                    elif paren_level:
                        paren_level -= 1
                elif token.startswith(':') and fstring_stack and (fstring_stack[-1].parentheses_count - fstring_stack[-1].format_spec_count == 1):
                    fstring_stack[-1].format_spec_count += 1
                    token = ':'
                    pos = start + 1
                yield PythonToken(OP, token, spos, prefix)
    if contstr:
        yield PythonToken(ERRORTOKEN, contstr, contstr_start, prefix)
        if contstr.endswith('\n') or contstr.endswith('\r'):
            new_line = True
    if fstring_stack:
        tos = fstring_stack[-1]
        if tos.previous_lines:
            yield PythonToken(FSTRING_STRING, tos.previous_lines, tos.last_string_start_pos, prefix='')
    end_pos = (lnum, max_)
    for indent in indents[1:]:
        indents.pop()
        yield PythonToken(DEDENT, '', end_pos, '')
    yield PythonToken(ENDMARKER, '', end_pos, additional_prefix)

def _split_illegal_unicode_name(token: str, start_pos: Tuple[int, int], prefix: str) -> Iterator[PythonToken]:

    def create_token() -> PythonToken:
        return PythonToken(ERRORTOKEN if is_illegal else NAME, found, pos, prefix)
    found = ''
    is_illegal = False
    pos = start_pos
    for i, char in enumerate(token):
        if is_illegal:
            if char.isidentifier():
                yield create_token()
                found = char
                is_illegal = False
                prefix = ''
                pos = (start_pos[0], start_pos[1] + i)
            else:
                found += char
        else:
            new_found = found + char
            if new_found.isidentifier():
                found = new_found
            else:
                if found:
                    yield create_token()
                    prefix = ''
                    pos = (start_pos[0], start_pos[1] + i)
                found = char
                is_illegal = True
    if found:
        yield create_token()
if __name__ == '__main__':
    path = sys.argv[1]
    with open(path) as f:
        code = f.read()
    for token in tokenize(code, version_info=parse_version_string('3.10')):
        print(token)
