#!/usr/bin/env python
"""
Check Python source code formatting, according to PEP 8.

For usage and a list of options, try this:
$ python pycodestyle.py -h

This program and its regression test suite live here:
https://github.com/pycqa/pycodestyle

Groups of errors and warnings:
E errors
W warnings
100 indentation
200 whitespace
300 blank lines
400 imports
500 line length
600 deprecation
700 statements
900 syntax error
"""

import bisect
import inspect
import keyword
import os
import re
import sys
import time
import tokenize
import warnings
from fnmatch import fnmatch
from functools import lru_cache
from optparse import OptionParser
from io import TextIOWrapper
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Set, Tuple

try:
    from configparser import RawConfigParser
except ImportError:
    from ConfigParser import RawConfigParser

if sys.version_info < (3, 10) and callable(getattr(tokenize, '_compile', None)):
    tokenize._compile = lru_cache()(tokenize._compile)
    
__version__ = '2.8.0'
DEFAULT_EXCLUDE: str = '.svn,CVS,.bzr,.hg,.git,__pycache__,.tox'
DEFAULT_IGNORE: str = 'E121,E123,E126,E226,E24,E704,W503,W504'
try:
    if sys.platform == 'win32':
        USER_CONFIG: str = os.path.expanduser('~\\.pycodestyle')
    else:
        USER_CONFIG = os.path.join(os.getenv('XDG_CONFIG_HOME') or os.path.expanduser('~/.config'), 'pycodestyle')
except ImportError:
    USER_CONFIG = None
PROJECT_CONFIG: Tuple[str, str] = ('setup.cfg', 'tox.ini')
TESTSUITE_PATH: str = os.path.join(os.path.dirname(__file__), 'testsuite')
MAX_LINE_LENGTH: int = 79
BLANK_LINES_CONFIG: Dict[str, int] = {'top_level': 2, 'method': 1}
MAX_DOC_LENGTH: int = 72
INDENT_SIZE: int = 4
REPORT_FORMAT: Dict[str, str] = {
    'default': '%(path)s:%(row)d:%(col)d: %(code)s %(text)s',
    'pylint': '%(path)s:%(row)d: [%(code)s] %(text)s'
}
PyCF_ONLY_AST: int = 1024
SINGLETONS: Set[str] = frozenset(['False', 'None', 'True'])
KEYWORDS: Set[str] = frozenset(keyword.kwlist + ['print', 'async']) - SINGLETONS
UNARY_OPERATORS: Set[str] = frozenset(['>>', '**', '*', '+', '-'])
ARITHMETIC_OP: Set[str] = frozenset(['**', '*', '/', '//', '+', '-', '@'])
WS_OPTIONAL_OPERATORS: Set[str] = ARITHMETIC_OP.union(['^', '&', '|', '<<', '>>', '%'])
ASSIGNMENT_EXPRESSION_OP: List[str] = [':='] if sys.version_info >= (3, 8) else []
WS_NEEDED_OPERATORS: Set[str] = frozenset(['**=', '*=', '/=', '//=', '+=', '-=', '!=', '<>', '<', '>', '%=', '^=', '&=', '|=', '==', '<=', '>=', '<<=', '>>=', '=', 'and', 'in', 'is', 'or', '->'] + ASSIGNMENT_EXPRESSION_OP)
WHITESPACE: Set[str] = frozenset(' \t\xa0')
NEWLINE: Set[int] = frozenset([tokenize.NL, tokenize.NEWLINE])
SKIP_TOKENS: Set[Any] = NEWLINE.union([tokenize.INDENT, tokenize.DEDENT])
SKIP_COMMENTS: Set[Any] = SKIP_TOKENS.union([tokenize.COMMENT, tokenize.ERRORTOKEN])
BENCHMARK_KEYS: List[str] = ['directories', 'files', 'logical lines', 'physical lines']
INDENT_REGEX: Any = re.compile('([ \\t]*)')
RAISE_COMMA_REGEX: Any = re.compile('raise\\s+\\w+\\s*,')
RERAISE_COMMA_REGEX: Any = re.compile('raise\\s+\\w+\\s*,.*,\\s*\\w+\\s*$')
ERRORCODE_REGEX: Any = re.compile('\\b[A-Z]\\d{3}\\b')
DOCSTRING_REGEX: Any = re.compile('u?r?["\\\']')
EXTRANEOUS_WHITESPACE_REGEX: Any = re.compile('[\\[({][ \\t]|[ \\t][\\]}),;:](?!=)')
WHITESPACE_AFTER_COMMA_REGEX: Any = re.compile('[,;:]\\s*(?:  |\\t)')
COMPARE_SINGLETON_REGEX: Any = re.compile('(\\bNone|\\bFalse|\\bTrue)?\\s*([=!]=)\\s*(?(1)|(None|False|True))\\b')
COMPARE_NEGATIVE_REGEX: Any = re.compile('\\b(?<!is\\s)(not)\\s+[^][)(}{ ]+\\s+(in|is)\\s')
COMPARE_TYPE_REGEX: Any = re.compile('(?:[=!]=|is(?:\\s+not)?)\\s+type(?:\\s*\\(\\s*([^)]*[^ )])\\s*\\))' + '|\\btype(?:\\s*\\(\\s*([^)]*[^ )])\\s*\\))\\s+(?:[=!]=|is(?:\\s+not)?)')
KEYWORD_REGEX: Any = re.compile('(\\s*)\\b(?:%s)\\b(\\s*)' % '|'.join(KEYWORDS))
OPERATOR_REGEX: Any = re.compile('(?:[^,\\s])(\\s*)(?:[-+*/|!<=>%&^]+|:=)(\\s*)')
LAMBDA_REGEX: Any = re.compile('\\blambda\\b')
HUNK_REGEX: Any = re.compile('^@@ -\\d+(?:,\\d+)? \\+(\\d+)(?:,(\\d+))? @@.*$')
STARTSWITH_DEF_REGEX: Any = re.compile('^(async\\s+def|def)\\b')
STARTSWITH_TOP_LEVEL_REGEX: Any = re.compile('^(async\\s+def\\s+|def\\s+|class\\s+|@)')
STARTSWITH_INDENT_STATEMENT_REGEX: Any = re.compile('^\\s*({})\\b'.format('|'.join((s.replace(' ', '\\s+') for s in ('def', 'async def', 'for', 'async for', 'if', 'elif', 'else', 'try', 'except', 'finally', 'with', 'async with', 'class', 'while')))))
DUNDER_REGEX: Any = re.compile('^__([^\\s]+)__(?::\\s*[a-zA-Z.0-9_\\[\\]\\"]+)? = ')
BLANK_EXCEPT_REGEX: Any = re.compile('except\\s*:')

_checks: Dict[str, Dict[Any, Tuple[List[str], Optional[List[str]]]]] = {'physical_line': {}, 'logical_line': {}, 'tree': {}}


def _get_parameters(function: Callable[..., Any]) -> List[str]:
    return [parameter.name for parameter in inspect.signature(function).parameters.values() if parameter.kind == parameter.POSITIONAL_OR_KEYWORD]


def register_check(check: Any, codes: Optional[List[str]] = None) -> Any:
    """Register a new check object."""
    def _add_check(check: Any, kind: str, codes: List[str], args: Optional[List[str]]) -> None:
        if check in _checks[kind]:
            _checks[kind][check][0].extend(codes or [])
        else:
            _checks[kind][check] = (codes or [''], args)
    if inspect.isfunction(check):
        args: List[str] = _get_parameters(check)
        if args and args[0] in ('physical_line', 'logical_line'):
            if codes is None:
                codes = ERRORCODE_REGEX.findall(check.__doc__ or '')
            _add_check(check, args[0], codes, args)
    elif inspect.isclass(check):
        if _get_parameters(check.__init__)[:2] == ['self', 'tree']:
            _add_check(check, 'tree', codes, None)
    return check


@register_check
def tabs_or_spaces(physical_line: str, indent_char: str) -> Optional[Tuple[int, str]]:
    """Never mix tabs and spaces.

    The most popular way of indenting Python is with spaces only.  The
    second-most popular way is with tabs only.  Code indented with a
    mixture of tabs and spaces should be converted to using spaces
    exclusively.  When invoking the Python command line interpreter with
    the -t option, it issues warnings about code that illegally mixes
    tabs and spaces.  When using -tt these warnings become errors.
    These options are highly recommended!

    Okay: if a == 0:\n    a = 1\n    b = 1
    E101: if a == 0:\n        a = 1\n\tb = 1
    """
    indent = INDENT_REGEX.match(physical_line).group(1)
    for offset, char in enumerate(indent):
        if char != indent_char:
            return (offset, 'E101 indentation contains mixed spaces and tabs')
    return None


@register_check
def tabs_obsolete(physical_line: str) -> Optional[Tuple[int, str]]:
    """On new projects, spaces-only are strongly recommended over tabs.

    Okay: if True:\n    return
    W191: if True:\n\treturn
    """
    indent = INDENT_REGEX.match(physical_line).group(1)
    if '\t' in indent:
        return (indent.index('\t'), 'W191 indentation contains tabs')
    return None


@register_check
def trailing_whitespace(physical_line: str) -> Optional[Tuple[int, str]]:
    """Trailing whitespace is superfluous.

    The warning returned varies on whether the line itself is blank,
    for easier filtering for those who want to indent their blank lines.

    Okay: spam(1)\n#
    W291: spam(1) \n#
    W293: class Foo(object):\n    \n    bang = 12
    """
    physical_line = physical_line.rstrip('\n')
    physical_line = physical_line.rstrip('\r')
    physical_line = physical_line.rstrip('\x0c')
    stripped = physical_line.rstrip(' \t\x0b')
    if physical_line != stripped:
        if stripped:
            return (len(stripped), 'W291 trailing whitespace')
        else:
            return (0, 'W293 blank line contains whitespace')
    return None


@register_check
def trailing_blank_lines(physical_line: str, lines: List[str], line_number: int, total_lines: int) -> Optional[Tuple[int, str]]:
    """Trailing blank lines are superfluous.

    Okay: spam(1)
    W391: spam(1)\n

    However the last line should end with a new line (warning W292).
    """
    if line_number == total_lines:
        stripped_last_line = physical_line.rstrip('\r\n')
        if physical_line and (not stripped_last_line):
            return (0, 'W391 blank line at end of file')
        if stripped_last_line == physical_line:
            return (len(lines[-1]), 'W292 no newline at end of file')
    return None


@register_check
def maximum_line_length(physical_line: str, max_line_length: int, multiline: bool, line_number: int, noqa: bool) -> Optional[Tuple[int, str]]:
    """Limit all lines to a maximum of 79 characters.

    There are still many devices around that are limited to 80 character
    lines; plus, limiting windows to 80 characters makes it possible to
    have several windows side-by-side.  The default wrapping on such
    devices looks ugly.  Therefore, please limit all lines to a maximum
    of 79 characters. For flowing long blocks of text (docstrings or
    comments), limiting the length to 72 characters is recommended.

    Reports error E501.
    """
    line = physical_line.rstrip()
    length = len(line)
    if length > max_line_length and (not noqa):
        if line_number == 1 and line.startswith('#!'):
            return None
        chunks = line.split()
        if (len(chunks) == 1 and multiline or (len(chunks) == 2 and chunks[0] == '#')) and len(line) - len(chunks[-1]) < max_line_length - 7:
            return None
        if length > max_line_length:
            return (max_line_length, 'E501 line too long (%d > %d characters)' % (length, max_line_length))
    return None


def _is_one_liner(logical_line: str, indent_level: int, lines: List[str], line_number: int) -> bool:
    if not STARTSWITH_TOP_LEVEL_REGEX.match(logical_line):
        return False
    line_idx: int = line_number - 1
    if line_idx < 1:
        prev_indent: int = 0
    else:
        prev_indent = expand_indent(lines[line_idx - 1])
    if prev_indent > indent_level:
        return False
    while line_idx < len(lines):
        line = lines[line_idx].strip()
        if not line.startswith('@') and STARTSWITH_TOP_LEVEL_REGEX.match(line):
            break
        else:
            line_idx += 1
    else:
        return False
    next_idx: int = line_idx + 1
    while next_idx < len(lines):
        if lines[next_idx].strip():
            break
        else:
            next_idx += 1
    else:
        return True
    return expand_indent(lines[next_idx]) <= indent_level


@register_check
def blank_lines(logical_line: str, blank_lines: int, indent_level: int, line_number: int, blank_before: int,
                  previous_logical: str, previous_unindented_logical_line: str, previous_indent_level: int,
                  lines: List[str]) -> Generator[Tuple[int, str], None, None]:
    """Separate top-level function and class definitions with two blank
    lines.

    Method definitions inside a class are separated by a single blank
    line.

    Extra blank lines may be used (sparingly) to separate groups of
    related functions.  Blank lines may be omitted between a bunch of
    related one-liners (e.g. a set of dummy implementations).

    Use blank lines in functions, sparingly, to indicate logical
    sections.

    Okay: def a():\n    pass\n\n\ndef b():\n    pass
    Okay: def a():\n    pass\n\n\nasync def b():\n    pass
    Okay: def a():\n    pass\n\n\n# Foo\n# Bar\n\ndef b():\n    pass
    Okay: default = 1\nfoo = 1
    Okay: classify = 1\nfoo = 1

    E301: class Foo:\n    b = 0\n    def bar():\n        pass
    E302: def a():\n    pass\n\n\ndef b(n):\n    pass
    E302: def a():\n    pass\n\nasync def b(n):\n    pass
    E303: def a():\n    pass\n\n\n\ndef b(n):\n    pass
    E303: def a():\n\n\n\ndef b(n):\n    pass
    E304: @decorator\ndef a():\n    pass
    E305: def a():\n    pass\na()
    E306: def a():\n    def b():\n        pass\n    def c():\n        pass
    """
    top_level_lines: int = BLANK_LINES_CONFIG['top_level']
    method_lines: int = BLANK_LINES_CONFIG['method']
    if not previous_logical and blank_before < top_level_lines:
        return
    if previous_logical.startswith('@'):
        if blank_lines:
            yield (0, 'E304 blank lines found after function decorator')
    elif blank_lines > top_level_lines or (indent_level and blank_lines == method_lines + 1):
        yield (0, 'E303 too many blank lines (%d)' % blank_lines)
    elif STARTSWITH_TOP_LEVEL_REGEX.match(logical_line):
        if _is_one_liner(logical_line, indent_level, lines, line_number) and blank_before == 0:
            return
        if indent_level:
            if not (blank_before == method_lines or previous_indent_level < indent_level or DOCSTRING_REGEX.match(previous_logical)):
                ancestor_level: int = indent_level
                nested: bool = False
                for line in lines[line_number - top_level_lines::-1]:
                    if line.strip() and expand_indent(line) < ancestor_level:
                        ancestor_level = expand_indent(line)
                        nested = bool(STARTSWITH_DEF_REGEX.match(line.lstrip()))
                        if nested or ancestor_level == 0:
                            break
                if nested:
                    yield (0, 'E306 expected %s blank line before a nested definition, found 0' % (method_lines,))
                else:
                    yield (0, 'E301 expected {} blank line, found 0'.format(method_lines))
        elif blank_before != top_level_lines:
            yield (0, 'E302 expected %s blank lines, found %d' % (top_level_lines, blank_before))
    elif logical_line and (not indent_level) and (blank_before != top_level_lines) and previous_unindented_logical_line.startswith(('def ', 'class ')):
        yield (0, 'E305 expected %s blank lines after class or function definition, found %d' % (top_level_lines, blank_before))


@register_check
def extraneous_whitespace(logical_line: str) -> Generator[Tuple[int, str], None, None]:
    """Avoid extraneous whitespace.

    Avoid extraneous whitespace in these situations:
    - Immediately inside parentheses, brackets or braces.
    - Immediately before a comma, semicolon, or colon.

    Okay: spam(ham[1], {eggs: 2})
    E201: spam( ham[1], {eggs: 2})
    E201: spam(ham[ 1], {eggs: 2})
    E201: spam(ham[1], { eggs: 2})
    E202: spam(ham[1], {eggs: 2} )
    E202: spam(ham[1 ], {eggs: 2})
    E202: spam(ham[1], {eggs: 2 })

    E203: if x == 4: print x, y; x, y = y , x
    E203: if x == 4: print x, y ; x, y = y, x
    E203: if x == 4 : print x, y; x, y = y, x
    """
    line: str = logical_line
    for match in EXTRANEOUS_WHITESPACE_REGEX.finditer(line):
        text: str = match.group()
        char: str = text.strip()
        found: int = match.start()
        if text[-1].isspace():
            yield (found + 1, "E201 whitespace after '%s'" % char)
        elif line[found - 1] != ',':
            code: str = 'E202' if char in '}])' else 'E203'
            yield (found, f"{code} whitespace before '{char}'")


@register_check
def whitespace_around_keywords(logical_line: str) -> Generator[Tuple[int, str], None, None]:
    """Avoid extraneous whitespace around keywords.

    Okay: True and False
    E271: True and  False
    E272: True  and False
    E273: True and\tFalse
    E274: True\tand False
    """
    for match in KEYWORD_REGEX.finditer(logical_line):
        before, after = match.groups()
        if '\t' in before:
            yield (match.start(1), 'E274 tab before keyword')
        elif len(before) > 1:
            yield (match.start(1), 'E272 multiple spaces before keyword')
        if '\t' in after:
            yield (match.start(2), 'E273 tab after keyword')
        elif len(after) > 1:
            yield (match.start(2), 'E271 multiple spaces after keyword')


@register_check
def missing_whitespace_after_import_keyword(logical_line: str) -> Generator[Tuple[int, str], None, None]:
    """Multiple imports in form from x import (a, b, c) should have
    space between import statement and parenthesised name list.

    Okay: from foo import (bar, baz)
    E275: from foo import(bar, baz)
    E275: from importable.module import(bar, baz)
    """
    line: str = logical_line
    indicator: str = ' import('
    if line.startswith('from '):
        found: int = line.find(indicator)
        if -1 < found:
            pos: int = found + len(indicator) - 1
            yield (pos, 'E275 missing whitespace after keyword')


@register_check
def missing_whitespace(logical_line: str) -> Generator[Tuple[int, str], None, None]:
    """Each comma, semicolon or colon should be followed by whitespace.

    Okay: [a, b]
    Okay: (3,)
    Okay: a[1:4]
    Okay: a[:4]
    Okay: a[1:]
    Okay: a[1:4:2]
    E231: ['a','b']
    E231: foo(bar,baz)
    E231: [{'a':'b'}]
    """
    line: str = logical_line
    for index in range(len(line) - 1):
        char: str = line[index]
        next_char: str = line[index + 1]
        if char in ',;:' and next_char not in WHITESPACE:
            before = line[:index]
            if char == ':' and before.count('[') > before.count(']') and (before.rfind('{') < before.rfind('[')):
                continue
            if char == ',' and next_char == ')':
                continue
            if char == ':' and next_char == '=' and (sys.version_info >= (3, 8)):
                continue
            yield (index, "E231 missing whitespace after '%s'" % char)


@register_check
def indentation(logical_line: str, previous_logical: str, indent_char: str, indent_level: int,
                  previous_indent_level: int, indent_size: int) -> Generator[Tuple[int, str], None, None]:
    """Use indent_size (PEP8 says 4) spaces per indentation level.

    For really old code that you don't want to mess up, you can continue
    to use 8-space tabs.

    Okay: a = 1
    Okay: if a == 0:\n    a = 1
    E111:   a = 1
    E114:   # a = 1

    Okay: for item in items:\n    pass
    E112: for item in items:\npass
    E115: for item in items:\n# Hi\n    pass

    Okay: a = 1\nb = 2
    E113: a = 1\n    b = 2
    E116: a = 1\n    # b = 2
    """
    c: int = 0 if logical_line else 3
    tmpl: str = 'E11%d %s' if logical_line else 'E11%d %s (comment)'
    if indent_level % indent_size:
        yield (0, tmpl % (1 + c, 'indentation is not a multiple of ' + str(indent_size)))
    indent_expect: bool = previous_logical.endswith(':')
    if indent_expect and indent_level <= previous_indent_level:
        yield (0, tmpl % (2 + c, 'expected an indented block'))
    elif not indent_expect and indent_level > previous_indent_level:
        yield (0, tmpl % (3 + c, 'unexpected indentation'))
    if indent_expect:
        expected_indent_amount: int = 8 if indent_char == '\t' else 4
        expected_indent_level: int = previous_indent_level + expected_indent_amount
        if indent_level > expected_indent_level:
            yield (0, tmpl % (7, 'over-indented'))


@register_check
def continued_indentation(logical_line: str, tokens: List[Tuple[Any, ...]], indent_level: int, hang_closing: bool,
                          indent_char: str, indent_size: int, noqa: bool, verbose: int) -> Generator[Tuple[Any, str], None, None]:
    """Continuation lines indentation.

    Continuation lines should align wrapped elements either vertically
    using Python's implicit line joining inside parentheses, brackets
    and braces, or using a hanging indent.

    When using a hanging indent these considerations should be applied:
    - there should be no arguments on the first line, and
    - further indentation should be used to clearly distinguish itself
      as a continuation line.

    Okay: a = (\n)
    E123: a = (\n    )

    Okay: a = (\n    42)
    E121: a = (\n   42)
    E122: a = (\n42)
    E123: a = (\n    42\n    )
    E124: a = (24,\n     42\n)
    E125: if (\n    b):\n    pass
    E126: a = (\n        42)
    E127: a = (24,\n      42)
    E128: a = (24,\n    42)
    E129: if (a or\n    b):\n    pass
    E131: a = (\n    42\n 24)
    """
    first_row: int = tokens[0][2][0]
    nrows: int = 1 + tokens[-1][2][0] - first_row
    if noqa or nrows == 1:
        return
    indent_next: bool = logical_line.endswith(':')
    row: int = 0
    depth: int = 0
    valid_hangs: Tuple[int, ...] = (indent_size,) if indent_char != '\t' else (indent_size, indent_size * 2)
    parens: List[int] = [0] * nrows
    rel_indent: List[int] = [0] * nrows
    open_rows: List[List[int]] = [[0]]
    hangs: List[Optional[int]] = [None]
    indent_chances: Dict[int, Any] = {}
    last_indent: Tuple[int, int] = tokens[0][2]
    visual_indent: Optional[bool] = None
    last_token_multiline: bool = False
    indent: List[int] = [last_indent[1]]
    if verbose >= 3:
        print('>>> ' + tokens[0][4].rstrip())
    for token_type, text, start, end, line in tokens:
        newline: bool = row < start[0] - first_row
        if newline:
            row = start[0] - first_row
            newline = not last_token_multiline and token_type not in NEWLINE
        if newline:
            last_indent = start
            if verbose >= 3:
                print('... ' + line.rstrip())
            rel_indent[row] = expand_indent(line) - indent_level
            close_bracket: bool = token_type == tokenize.OP and text in ']})'
            for open_row in reversed(open_rows[depth]):
                hang: int = rel_indent[row] - rel_indent[open_row]
                hanging_indent: bool = hang in valid_hangs
                if hanging_indent:
                    break
            if hangs[depth]:
                hanging_indent = hang == hangs[depth]
            visual_indent = (not close_bracket) and (hang > 0) and indent_chances.get(start[1])
            if close_bracket and indent[depth]:
                if start[1] != indent[depth]:
                    yield (start, 'E124 closing bracket does not match visual indentation')
            elif close_bracket and (not hang):
                if hang_closing:
                    yield (start, 'E133 closing bracket is missing indentation')
            elif indent[depth] and start[1] < indent[depth]:
                if visual_indent is not True:
                    yield (start, 'E128 continuation line under-indented for visual indent')
            elif hanging_indent or (indent_next and rel_indent[row] == 2 * indent_size):
                if close_bracket and (not hang_closing):
                    yield (start, "E123 closing bracket does not match indentation of opening bracket's line")
                hangs[depth] = hang
            elif visual_indent is True:
                indent[depth] = start[1]
            elif visual_indent in (text, str):
                pass
            else:
                if hang <= 0:
                    error = ('E122', 'missing indentation or outdented')
                elif indent[depth]:
                    error = ('E127', 'over-indented for visual indent')
                elif not close_bracket and hangs[depth]:
                    error = ('E131', 'unaligned for hanging indent')
                else:
                    hangs[depth] = hang
                    if hang > indent_size:
                        error = ('E126', 'over-indented for hanging indent')
                    else:
                        error = ('E121', 'under-indented for hanging indent')
                yield (start, '%s continuation line %s' % error)
        if parens[row] and token_type not in (tokenize.NL, tokenize.COMMENT) and (not indent[depth]):
            indent[depth] = start[1]
            indent_chances[start[1]] = True
            if verbose >= 4:
                print(f'bracket depth {depth} indent to {start[1]}')
        elif token_type in (tokenize.STRING, tokenize.COMMENT) or text in ('u', 'ur', 'b', 'br'):
            indent_chances[start[1]] = str
        elif not row and (not depth) and (text in ['assert', 'raise', 'with']):
            indent_chances[end[1] + 1] = True
        elif not indent_chances and (not row) and (not depth) and (text == 'if'):
            indent_chances[end[1] + 1] = True
        elif text == ':' and line[end[1]:].isspace():
            open_rows[depth].append(row)
        if token_type == tokenize.OP:
            if text in '([{':
                depth += 1
                indent.append(0)
                hangs.append(None)
                if len(open_rows) == depth:
                    open_rows.append([])
                open_rows[depth].append(row)
                parens[row] += 1
                if verbose >= 4:
                    print('bracket depth %s seen, col %s, visual min = %s' % (depth, start[1], indent[depth]))
            elif text in ')]}' and depth > 0:
                prev_indent: int = indent.pop() or last_indent[1]
                hangs.pop()
                for d in range(depth):
                    if indent[d] > prev_indent:
                        indent[d] = 0
                for ind in list(indent_chances):
                    if ind >= prev_indent:
                        del indent_chances[ind]
                del open_rows[depth + 1:]
                depth -= 1
                if depth:
                    indent_chances[indent[depth]] = True
                for idx in range(row, -1, -1):
                    if parens[idx]:
                        parens[idx] -= 1
                        break
            assert len(indent) == depth + 1
            if start[1] not in indent_chances:
                indent_chances[start[1]] = text
        last_token_multiline = start[0] != end[0]
        if last_token_multiline:
            rel_indent[end[0] - first_row] = rel_indent[row]
    if indent_next and expand_indent(line) == indent_level + indent_size:
        pos = (start[0], indent[0] + indent_size)
        if visual_indent:
            code = 'E129 visually indented line'
        else:
            code = 'E125 continuation line'
        yield (pos, '%s with same indent as next logical line' % code)


@register_check
def whitespace_before_parameters(logical_line: str, tokens: List[Tuple[Any, ...]]) -> Generator[Tuple[Tuple[int, int], str], None, None]:
    """Avoid extraneous whitespace.

    Avoid extraneous whitespace in the following situations:
    - before the open parenthesis that starts the argument list of a
      function call.
    - before the open parenthesis that starts an indexing or slicing.

    Okay: spam(1)
    E211: spam (1)

    Okay: dict['key'] = list[index]
    E211: dict ['key'] = list[index]
    E211: dict['key'] = list [index]
    """
    prev_type, prev_text, __, prev_end, __ = tokens[0]
    for index in range(1, len(tokens)):
        token_type, text, start, end, __ = tokens[index]
        if token_type == tokenize.OP and text in '([' and (start != prev_end) and (prev_type == tokenize.NAME or prev_text in '}])') and (index < 2 or tokens[index - 2][1] != 'class') and (not keyword.iskeyword(prev_text)) and (sys.version_info < (3, 9) or not keyword.issoftkeyword(prev_text)):
            yield (prev_end, "E211 whitespace before '%s'" % text)
        prev_type = token_type
        prev_text = text
        prev_end = end


@register_check
def whitespace_around_operator(logical_line: str) -> Generator[Tuple[int, str], None, None]:
    """Avoid extraneous whitespace around an operator.

    Okay: a = 12 + 3
    E221: a = 4  + 5
    E222: a = 4 +  5
    E223: a = 4\t+ 5
    E224: a = 4 +\t5
    """
    for match in OPERATOR_REGEX.finditer(logical_line):
        before, after = match.groups()
        if '\t' in before:
            yield (match.start(1), 'E223 tab before operator')
        elif len(before) > 1:
            yield (match.start(1), 'E221 multiple spaces before operator')
        if '\t' in after:
            yield (match.start(2), 'E224 tab after operator')
        elif len(after) > 1:
            yield (match.start(2), 'E222 multiple spaces after operator')


@register_check
def missing_whitespace_around_operator(logical_line: str, tokens: List[Tuple[Any, ...]]) -> Generator[Tuple[Any, str], None, None]:
    """Surround operators with a single space on either side.

    - Always surround these binary operators with a single space on
      either side: assignment (=), augmented assignment (+=, -= etc.),
      comparisons (==, <, >, !=, <=, >=, in, not in, is, is not),
      Booleans (and, or, not).

    - If operators with different priorities are used, consider adding
      whitespace around the operators with the lowest priorities.

    Okay: i = i + 1
    Okay: submitted += 1
    Okay: x = x * 2 - 1
    Okay: hypot2 = x * x + y * y
    Okay: c = (a + b) * (a - b)
    Okay: foo(bar, key='word', *args, **kwargs)
    Okay: alpha[:-i]

    E225: i=i+1
    E225: submitted +=1
    E225: x = x /2 - 1
    E225: z = x **y
    E225: z = 1and 1
    E226: c = (a+b) * (a-b)
    E226: hypot2 = x*x + y*y
    E227: c = a|b
    E228: msg = fmt%(errno, errmsg)
    """
    parens: int = 0
    need_space: Any = False
    prev_type: Any = tokenize.OP
    prev_text: Any = None
    prev_end: Any = None
    operator_types: Tuple[Any, ...] = (tokenize.OP, tokenize.NAME)
    for token_type, text, start, end, line in tokens:
        if token_type in SKIP_COMMENTS:
            continue
        if text in ('(', 'lambda'):
            parens += 1
        elif text == ')':
            parens -= 1
        if need_space:
            if start != prev_end:
                if need_space is not True and (not need_space[1]):
                    yield (need_space[0], 'E225 missing whitespace around operator')
                need_space = False
            elif text == '>' and prev_text in ('<', '-'):
                pass
            elif prev_text == '/' and text in {',', ')', ':'} or (prev_text == ')' and text == ':'):
                pass
            else:
                if need_space is True or need_space[1]:
                    yield (prev_end, 'E225 missing whitespace around operator')
                elif prev_text != '**':
                    code, optype = ('E226', 'arithmetic')
                    if prev_text == '%':
                        code, optype = ('E228', 'modulo')
                    elif prev_text not in ARITHMETIC_OP:
                        code, optype = ('E227', 'bitwise or shift')
                    yield (need_space[0], '%s missing whitespace around %s operator' % (code, optype))
                need_space = False
        elif token_type in operator_types and prev_end is not None:
            if text == '=' and parens:
                pass
            elif text in WS_NEEDED_OPERATORS:
                need_space = True
            elif text in UNARY_OPERATORS:
                if prev_type == tokenize.OP and prev_text in '}])' or (prev_type != tokenize.OP and prev_text not in KEYWORDS and (sys.version_info < (3, 9) or not keyword.issoftkeyword(prev_text))):
                    need_space = None
            elif text in WS_OPTIONAL_OPERATORS:
                need_space = None
            if need_space is None:
                need_space = (prev_end, start != prev_end)
            elif need_space and start == prev_end:
                yield (prev_end, 'E225 missing whitespace around operator')
                need_space = False
        prev_type = token_type
        prev_text = text
        prev_end = end


@register_check
def whitespace_around_comma(logical_line: str) -> Generator[Tuple[int, str], None, None]:
    """Avoid extraneous whitespace after a comma or a colon.

    Note: these checks are disabled by default

    Okay: a = (1, 2)
    E241: a = (1,  2)
    E242: a = (1,\t2)
    """
    line: str = logical_line
    for m in WHITESPACE_AFTER_COMMA_REGEX.finditer(line):
        found: int = m.start() + 1
        if '\t' in m.group():
            yield (found, "E242 tab after '%s'" % m.group()[0])
        else:
            yield (found, "E241 multiple spaces after '%s'" % m.group()[0])


@register_check
def whitespace_around_named_parameter_equals(logical_line: str, tokens: List[Tuple[Any, ...]]) -> Generator[Tuple[int, str], None, None]:
    """Don't use spaces around the '=' sign in function arguments.

    Don't use spaces around the '=' sign when used to indicate a
    keyword argument or a default parameter value, except when
    using a type annotation.

    Okay: def complex(real, imag=0.0):
    Okay: return magic(r=real, i=imag)
    Okay: boolean(a == b)
    Okay: boolean(a != b)
    Okay: boolean(a <= b)
    Okay: boolean(a >= b)
    Okay: def foo(arg: int = 42):
    Okay: async def foo(arg: int = 42):

    E251: def complex(real, imag = 0.0):
    E251: return magic(r = real, i = imag)
    E252: def complex(real, image: float=0.0):
    """
    parens: int = 0
    no_space: bool = False
    require_space: bool = False
    prev_end: Any = None
    annotated_func_arg: bool = False
    in_def: bool = bool(STARTSWITH_DEF_REGEX.match(logical_line))
    message: str = 'E251 unexpected spaces around keyword / parameter equals'
    missing_message: str = 'E252 missing whitespace around parameter equals'
    for token_type, text, start, end, line in tokens:
        if token_type == tokenize.NL:
            continue
        if no_space:
            no_space = False
            if start != prev_end:
                yield (prev_end, message)
        if require_space:
            require_space = False
            if start == prev_end:
                yield (prev_end, missing_message)
        if token_type == tokenize.OP:
            if text in '([':
                parens += 1
            elif text in ')]':
                parens -= 1
            elif in_def and text == ':' and (parens == 1):
                annotated_func_arg = True
            elif parens == 1 and text == ',':
                annotated_func_arg = False
            elif parens and text == '=':
                if annotated_func_arg and parens == 1:
                    require_space = True
                    if start == prev_end:
                        yield (prev_end, missing_message)
                else:
                    no_space = True
                    if start != prev_end:
                        yield (prev_end, message)
            if not parens:
                annotated_func_arg = False
        prev_end = end


@register_check
def whitespace_before_comment(logical_line: str, tokens: List[Tuple[Any, ...]]) -> Generator[Tuple[Any, str], None, None]:
    """Separate inline comments by at least two spaces.

    An inline comment is a comment on the same line as a statement.
    Inline comments should be separated by at least two spaces from the
    statement. They should start with a # and a single space.

    Each line of a block comment starts with a # and one or multiple
    spaces as there can be indented text inside the comment.

    Okay: x = x + 1  # Increment x
    Okay: x = x + 1    # Increment x
    Okay: # Block comments:
    Okay: #  - Block comment list
    Okay: # \xa0- Block comment list
    E261: x = x + 1 # Increment x
    E262: x = x + 1  #Increment x
    E262: x = x + 1  #  Increment x
    E262: x = x + 1  # \xa0Increment x
    E265: #Block comment
    E266: ### Block comment
    """
    prev_end: Tuple[int, int] = (0, 0)
    for token_type, text, start, end, line in tokens:
        if token_type == tokenize.COMMENT:
            inline_comment: str = line[:start[1]].strip()
            if inline_comment:
                if prev_end[0] == start[0] and start[1] < prev_end[1] + 2:
                    yield (prev_end, 'E261 at least two spaces before inline comment')
            symbol, sp, comment = text.partition(' ')
            bad_prefix: str = symbol if (symbol not in '#:' and (symbol.lstrip('#')[:1] or '#')) else ''
            if inline_comment:
                if bad_prefix or comment[:1] in WHITESPACE:
                    yield (start, "E262 inline comment should start with '# '")
            elif bad_prefix and (bad_prefix != '!' or start[0] > 1):
                if bad_prefix != '#':
                    yield (start, "E265 block comment should start with '# '")
                elif comment:
                    yield (start, "E266 too many leading '#' for block comment")
        elif token_type != tokenize.NL:
            prev_end = end


@register_check
def imports_on_separate_lines(logical_line: str) -> Generator[Tuple[int, str], None, None]:
    """Place imports on separate lines.

    Okay: import os\nimport sys
    E401: import sys, os

    Okay: from subprocess import Popen, PIPE
    Okay: from myclas import MyClass
    Okay: from foo.bar.yourclass import YourClass
    Okay: import myclass
    Okay: import foo.bar.yourclass
    """
    line: str = logical_line
    if line.startswith('import '):
        found: int = line.find(',')
        if -1 < found and ';' not in line[:found]:
            yield (found, 'E401 multiple imports on one line')


@register_check
def module_imports_on_top_of_file(logical_line: str, indent_level: int, checker_state: Dict[str, Any], noqa: bool) -> Generator[Tuple[int, str], None, None]:
    """Place imports at the top of the file.

    Always put imports at the top of the file, just after any module
    comments and docstrings, and before module globals and constants.

    Okay: import os
    Okay: # this is a comment\nimport os
    Okay: '''this is a module docstring'''\nimport os
    Okay: r'''this is a module docstring'''\nimport os
    Okay:
    try:\n\timport x\nexcept ImportError:\n\tpass\nelse:\n\tpass\nimport y
    Okay:
    try:\n\timport x\nexcept ImportError:\n\tpass\nfinally:\n\tpass\nimport y
    E402: a=1\nimport os
    E402: 'One string'\n"Two string"\nimport os
    E402: a=1\nfrom sys import x

    Okay: if x:\n    import os
    """

    def is_string_literal(line: str) -> bool:
        if line[0] in 'uUbB':
            line = line[1:]
        if line and line[0] in 'rR':
            line = line[1:]
        return bool(line) and (line[0] == '"' or line[0] == "'")
    allowed_keywords: Tuple[str, ...] = ('try', 'except', 'else', 'finally', 'with', 'if', 'elif')
    if indent_level:
        return
    if not logical_line:
        return
    if noqa:
        return
    line: str = logical_line
    if line.startswith('import ') or line.startswith('from '):
        if checker_state.get('seen_non_imports', False):
            yield (0, 'E402 module level import not at top of file')
    elif re.match(DUNDER_REGEX, line):
        return
    elif any((line.startswith(kw) for kw in allowed_keywords)):
        return
    elif is_string_literal(line):
        if checker_state.get('seen_docstring', False):
            checker_state['seen_non_imports'] = True
        else:
            checker_state['seen_docstring'] = True
    else:
        checker_state['seen_non_imports'] = True


@register_check
def compound_statements(logical_line: str) -> Generator[Tuple[int, str], None, None]:
    """Compound statements (on the same line) are generally
    discouraged.

    While sometimes it's okay to put an if/for/while with a small body
    on the same line, never do this for multi-clause statements.
    Also avoid folding such long lines!

    Always use a def statement instead of an assignment statement that
    binds a lambda expression directly to a name.

    Okay: if foo == 'blah':\n    do_blah_thing()
    Okay: do_one()
    Okay: do_two()
    Okay: do_three()

    E701: if foo == 'blah': do_blah_thing()
    E701: for x in lst: total += x
    E701: while t < 10: t = delay()
    E701: if foo == 'blah': do_blah_thing()
    E701: else: do_non_blah_thing()
    E701: if foo == 'blah': one(); two(); three()
    E702: do_one(); do_two(); do_three()
    E703: do_four();  # useless semicolon
    E704: def f(x): return 2*x
    E731: f = lambda x: 2*x
    """
    line: str = logical_line
    last_char: int = len(line) - 1
    found: int = line.find(':')
    prev_found: int = 0
    counts: Dict[str, int] = {char: 0 for char in '{}[]()'}
    while -1 < found < last_char:
        update_counts(line[prev_found:found], counts)
        if (counts['{'] <= counts['}'] and counts['['] <= counts[']'] and (counts['('] <= counts[')'])) and (not (sys.version_info >= (3, 8) and line[found + 1] == '=')):
            lambda_kw = LAMBDA_REGEX.search(line, 0, found)
            if lambda_kw:
                before: str = line[:lambda_kw.start()].rstrip()
                if before[-1:] == '=' and before[:-1].strip().isidentifier():
                    yield (0, 'E731 do not assign a lambda expression, use a def')
                break
            if STARTSWITH_DEF_REGEX.match(line):
                yield (0, 'E704 multiple statements on one line (def)')
            elif STARTSWITH_INDENT_STATEMENT_REGEX.match(line):
                yield (found, 'E701 multiple statements on one line (colon)')
        prev_found = found
        found = line.find(':', found + 1)
    found = line.find(';')
    while -1 < found:
        if found < last_char:
            yield (found, 'E702 multiple statements on one line (semicolon)')
        else:
            yield (found, 'E703 statement ends with a semicolon')
        found = line.find(';', found + 1)


@register_check
def explicit_line_join(logical_line: str, tokens: List[Tuple[Any, ...]]) -> Generator[Tuple[int, str], None, None]:
    """Avoid explicit line join between brackets.

    The preferred way of wrapping long lines is by using Python's
    implied line continuation inside parentheses, brackets and braces.
    Long lines can be broken over multiple lines by wrapping expressions
    in parentheses.  These should be used in preference to using a
    backslash for line continuation.

    E502: aaa = [123, \\\n       123]
    E502: aaa = ("bbb " \\\n       "ccc")

    Okay: aaa = [123,\n       123]
    Okay: aaa = ("bbb "\n       "ccc")
    Okay: aaa = "bbb " \\\n    "ccc"
    Okay: aaa = 123  # \\
    """
    prev_start: int = 0
    prev_end: int = 0
    parens: int = 0
    comment: bool = False
    backslash: Optional[int] = None
    for token_type, text, start, end, line in tokens:
        if token_type == tokenize.COMMENT:
            comment = True
        if start[0] != prev_start and parens and backslash and (not comment):
            yield (backslash, 'E502 the backslash is redundant between brackets')
        if end[0] != prev_end:
            if line.rstrip('\r\n').endswith('\\'):
                backslash = (end[0], len(line.splitlines()[-1]) - 1)[1]
            else:
                backslash = None
            prev_start = end[0]
            prev_end = end[0]
        else:
            prev_start = start[0]
        if token_type == tokenize.OP:
            if text in '([{':
                parens += 1
            elif text in ')]}':
                parens -= 1


_SYMBOLIC_OPS: Set[str] = frozenset('()[]{},:.;@=%~') | frozenset(('...',))


def _is_binary_operator(token_type: Any, text: str) -> bool:
    return (token_type == tokenize.OP or text in {'and', 'or'}) and text not in _SYMBOLIC_OPS


def _break_around_binary_operators(tokens: List[Tuple[Any, ...]]) -> Generator[Tuple[Any, ...], None, None]:
    """Private function to reduce duplication.

    This factors out the shared details between
    :func:`break_before_binary_operator` and
    :func:`break_after_binary_operator`.
    """
    line_break: bool = False
    unary_context: bool = True
    previous_token_type: Optional[Any] = None
    previous_text: Optional[str] = None
    for token_type, text, start, end, line in tokens:
        if token_type == tokenize.COMMENT:
            continue
        if ('\n' in text or '\r' in text) and token_type != tokenize.STRING:
            line_break = True
        else:
            yield (token_type, text, previous_token_type, previous_text, line_break, unary_context, start)
            unary_context = text in '([{,;'
            line_break = False
            previous_token_type = token_type
            previous_text = text


@register_check
def break_before_binary_operator(logical_line: str, tokens: List[Tuple[Any, ...]]) -> Generator[Tuple[Any, str], None, None]:
    """
    Avoid breaks before binary operators.

    The preferred place to break around a binary operator is after the
    operator, not before it.

    W503: (width == 0\n + height == 0)
    W503: (width == 0\n and height == 0)
    W503: var = (1\n       & ~2)
    W503: var = (1\n       / -2)
    W503: var = (1\n       + -1\n       + -2)

    Okay: foo(\n    -x)
    Okay: foo(x\n    [])
    Okay: x = '''\n''' + ''
    Okay: foo(x,\n    -y)
    Okay: foo(x,  # comment\n    -y)
    """
    for context in _break_around_binary_operators(tokens):
        token_type, text, previous_token_type, previous_text, line_break, unary_context, start = context
        if _is_binary_operator(token_type, text) and line_break and (not unary_context) and (not _is_binary_operator(previous_token_type, previous_text)):
            yield (start, 'W503 line break before binary operator')


@register_check
def break_after_binary_operator(logical_line: str, tokens: List[Tuple[Any, ...]]) -> Generator[Tuple[Any, str], None, None]:
    """
    Avoid breaks after binary operators.

    The preferred place to break around a binary operator is before the
    operator, not after it.

    W504: (width == 0 +\n height == 0)
    W504: (width == 0 and\n height == 0)
    W504: var = (1 &\n       ~2)

    Okay: foo(\n    -x)
    Okay: foo(x\n    [])
    Okay: x = '''\n''' + ''
    Okay: x = '' + '''\n'''
    Okay: foo(x,\n    -y)
    Okay: foo(x,  # comment\n    -y)

    The following should be W504 but unary_context is tricky with these
    Okay: var = (1 /\n       -2)
    Okay: var = (1 +\n       -1 +\n       -2)
    """
    prev_start: Any = None
    for context in _break_around_binary_operators(tokens):
        token_type, text, previous_token_type, previous_text, line_break, unary_context, start = context
        if _is_binary_operator(previous_token_type, previous_text) and line_break and (not unary_context) and (not _is_binary_operator(token_type, text)):
            yield (prev_start, 'W504 line break after binary operator')
        prev_start = start


@register_check
def comparison_to_singleton(logical_line: str, noqa: bool) -> Generator[Tuple[int, str], None, None]:
    """Comparison to singletons should use "is" or "is not".

    Comparisons to singletons like None should always be done
    with "is" or "is not", never the equality operators.

    Okay: if arg is not None:
    E711: if arg != None:
    E711: if None == arg:
    E712: if arg == True:
    E712: if False == arg:

    Also, beware of writing if x when you really mean if x is not None
    -- e.g. when testing whether a variable or argument that defaults to
    None was set to some other value.  The other value might have a type
    (such as a container) that could be false in a boolean context!
    """
    if noqa:
        return
    for match in COMPARE_SINGLETON_REGEX.finditer(logical_line):
        singleton: str = match.group(1) or match.group(3)
        same: bool = match.group(2) == '=='
        msg: str = "'if cond is %s:'" % (('' if same else 'not ') + singleton)
        if singleton in ('None',):
            code: str = 'E711'
        else:
            code = 'E712'
            nonzero: bool = singleton == 'True' and same or (singleton == 'False' and (not same))
            msg += " or 'if %scond:'" % ('' if nonzero else 'not ')
        yield (match.start(2), '%s comparison to %s should be %s' % (code, singleton, msg))


@register_check
def comparison_negative(logical_line: str) -> Generator[Tuple[int, str], None, None]:
    """Negative comparison should be done using "not in" and "is not".

    Okay: if x not in y:\n    pass
    Okay: assert (X in Y or X is Z)
    Okay: if not (X in Y):\n    pass
    Okay: zz = x is not y
    E713: Z = not X in Y
    E713: if not X.B in Y:\n    pass
    E714: if not X is Y:\n    pass
    E714: Z = not X.B is Y
    """
    match = COMPARE_NEGATIVE_REGEX.search(logical_line)
    if match:
        pos: int = match.start(1)
        if match.group(2) == 'in':
            yield (pos, "E713 test for membership should be 'not in'")
        else:
            yield (pos, "E714 test for object identity should be 'is not'")


@register_check
def comparison_type(logical_line: str, noqa: bool) -> Generator[Tuple[int, str], None, None]:
    """Object type comparisons should always use isinstance().

    Do not compare types directly.

    Okay: if isinstance(obj, int):
    E721: if type(obj) is type(1):
    """
    match = COMPARE_TYPE_REGEX.search(logical_line)
    if match and (not noqa):
        inst: Optional[str] = match.group(1)
        if inst and inst.isidentifier() and (inst not in SINGLETONS):
            return
        yield (match.start(), "E721 do not compare types, use 'isinstance()'")


@register_check
def bare_except(logical_line: str, noqa: bool) -> Generator[Tuple[int, str], None, None]:
    """When catching exceptions, mention specific exceptions when
    possible.

    Okay: except Exception:
    Okay: except BaseException:
    E722: except:
    """
    if noqa:
        return
    match = BLANK_EXCEPT_REGEX.match(logical_line)
    if match:
        yield (match.start(), "E722 do not use bare 'except'")


@register_check
def ambiguous_identifier(logical_line: str, tokens: List[Tuple[Any, ...]]) -> Generator[Tuple[Any, str], None, None]:
    """Never use the characters 'l', 'O', or 'I' as variable names.

    In some fonts, these characters are indistinguishable from the
    numerals one and zero. When tempted to use 'l', use 'L' instead.

    Okay: L = 0
    Okay: o = 123
    Okay: i = 42
    E741: l = 0
    E741: O = 123
    E741: I = 42

    Variables can be bound in several other contexts, including class
    and function definitions, 'global' and 'nonlocal' statements,
    exception handlers, and 'with' and 'for' statements.
    In addition, we have a special handling for function parameters.

    Okay: except AttributeError as o:
    Okay: with lock as L:
    Okay: foo(l=12)
    Okay: for a in foo(l=12):
    E741: except AttributeError as O:
    E741: with lock as l:
    E741: global I
    E741: nonlocal l
    E741: def foo(l):
    E741: def foo(l=12):
    E741: l = foo(l=12)
    E741: for l in range(10):
    E742: class I(object):
    E743: def l(x):
    """
    is_func_def: bool = False
    parameter_parentheses_level: int = 0
    idents_to_avoid: Tuple[str, ...] = ('l', 'O', 'I')
    prev_type, prev_text, prev_start, prev_end, __ = tokens[0]
    for token_type, text, start, end, line in tokens[1:]:
        ident: Optional[str] = None
        pos: Any = None
        if prev_text == 'def':
            is_func_def = True
        if parameter_parentheses_level == 0 and prev_type == tokenize.NAME and (token_type == tokenize.OP) and (text == '('):
            parameter_parentheses_level = 1
        elif parameter_parentheses_level > 0 and token_type == tokenize.OP:
            if text == '(':
                parameter_parentheses_level += 1
            elif text == ')':
                parameter_parentheses_level -= 1
        if token_type == tokenize.OP and '=' in text and (parameter_parentheses_level == 0):
            if prev_text in idents_to_avoid:
                ident = prev_text
                pos = prev_start
        if prev_text in ('as', 'for', 'global', 'nonlocal'):
            if text in idents_to_avoid:
                ident = text
                pos = start
        if is_func_def:
            if text in idents_to_avoid:
                ident = text
                pos = start
        if prev_text == 'class':
            if text in idents_to_avoid:
                yield (start, "E742 ambiguous class definition '%s'" % text)
        if prev_text == 'def':
            if text in idents_to_avoid:
                yield (start, "E743 ambiguous function definition '%s'" % text)
        if ident:
            yield (pos, "E741 ambiguous variable name '%s'" % ident)
        prev_type = token_type
        prev_text = text
        prev_start = start


@register_check
def python_3000_has_key(logical_line: str) -> Generator[Tuple[int, str], None, None]:
    """The {}.has_key() method is removed in Python 3: use the 'in'
    operator.

    Okay: if "alph" in d:\n    print d["alph"]
    W601: assert d.has_key('alph')
    """
    pos: int = logical_line.find('.has_key(')
    if pos > -1 and (not noqa(logical_line)):
        yield (pos, "W601 .has_key() is deprecated, use 'in'")


@register_check
def python_3000_raise_comma(logical_line: str) -> Generator[Tuple[int, str], None, None]:
    """When raising an exception, use "raise ValueError('message')".

    The older form is removed in Python 3.

    Okay: raise DummyError("Message")
    W602: raise DummyError, "Message"
    """
    match = RAISE_COMMA_REGEX.match(logical_line)
    if match and (not RERAISE_COMMA_REGEX.match(logical_line)):
        yield (match.end() - 1, 'W602 deprecated form of raising exception')


@register_check
def python_3000_not_equal(logical_line: str) -> Optional[Tuple[int, str]]:
    """New code should always use != instead of <>.

    The older syntax is removed in Python 3.

    Okay: if a != 'no':
    W603: if a <> 'no':
    """
    pos: int = logical_line.find('<>')
    if pos > -1:
        return (pos, "W603 '<>' is deprecated, use '!='")
    return None


@register_check
def python_3000_backticks(logical_line: str) -> Optional[Tuple[int, str]]:
    """Use repr() instead of backticks in Python 3.

    Okay: val = repr(1 + 2)
    W604: val = `1 + 2`
    """
    pos: int = logical_line.find('`')
    if pos > -1:
        return (pos, "W604 backticks are deprecated, use 'repr()'")
    return None


@register_check
def python_3000_invalid_escape_sequence(logical_line: str, tokens: List[Tuple[Any, ...]], noqa: bool) -> Generator[Tuple[Tuple[int, int], str], None, None]:
    """Invalid escape sequences are deprecated in Python 3.6.

    Okay: regex = r'\.png$'
    W605: regex = '\\.png$'
    """
    if noqa:
        return
    valid: List[str] = ['\n', '\\', "'", '"', 'a', 'b', 'f', 'n', 'r', 't', 'v', '0', '1', '2', '3', '4', '5', '6', '7', 'x', 'N', 'u', 'U']
    for token_type, text, start, end, line in tokens:
        if token_type == tokenize.STRING:
            start_line, start_col = start
            quote = text[-3:] if text[-3:] in ('"""', "'''") else text[-1]
            quote_pos = text.index(quote)
            prefix = text[:quote_pos].lower()
            sindex = quote_pos + len(quote)
            string = text[sindex:-len(quote)]
            if 'r' not in prefix:
                pos = string.find('\\')
                while pos >= 0:
                    pos += 1
                    if pos < len(string) and string[pos] not in valid:
                        line_num = start_line + string.count('\n', 0, pos)
                        if line_num == start_line:
                            col: int = start_col + len(prefix) + len(quote) + pos
                        else:
                            col = pos - string.rfind('\n', 0, pos) - 1
                        yield ((line_num, col - 1), "W605 invalid escape sequence '\\%s'" % string[pos])
                    pos = string.find('\\', pos + 1)


@register_check
def python_3000_async_await_keywords(logical_line: str, tokens: List[Tuple[Any, ...]]) -> Generator[Tuple[Any, str], None, None]:
    """'async' and 'await' are reserved keywords starting at Python 3.7.

    W606: async = 42
    W606: await = 42
    Okay: async def read(db):
    data = await db.fetch('SELECT ...')
    """
    state: Optional[Tuple[str, Any]] = None
    for token_type, text, start, end, line in tokens:
        error: bool = False
        if token_type == tokenize.NL:
            continue
        if state is None:
            if token_type == tokenize.NAME:
                if text == 'async':
                    state = ('async_stmt', start)
                elif text == 'await':
                    state = ('await', start)
                elif token_type == tokenize.NAME and text in ('def', 'for'):
                    state = ('define', start)
        elif state[0] == 'async_stmt':
            if token_type == tokenize.NAME and text in ('def', 'with', 'for'):
                state = None
            else:
                error = True
        elif state[0] == 'await':
            if token_type == tokenize.NAME:
                state = None
            elif token_type == tokenize.OP and text == '(':
                state = None
            else:
                error = True
        elif state[0] == 'define':
            if token_type == tokenize.NAME and text in ('async', 'await'):
                error = True
            else:
                state = None
        if error:
            yield (state[1], "W606 'async' and 'await' are reserved keywords starting with Python 3.7")
            state = None
    if state is not None:
        yield (state[1], "W606 'async' and 'await' are reserved keywords starting with Python 3.7")


@register_check
def maximum_doc_length(logical_line: str, max_doc_length: Optional[int], noqa: bool, tokens: List[Tuple[Any, ...]]) -> Generator[Tuple[Tuple[int, int], str], None, None]:
    """Limit all doc lines to a maximum of 72 characters.

    For flowing long blocks of text (docstrings or comments), limiting
    the length to 72 characters is recommended.

    Reports warning W505
    """
    if max_doc_length is None or noqa:
        return
    prev_token: Optional[Any] = None
    skip_lines: Set[str] = set()
    for token_type, text, start, end, line in tokens:
        if token_type not in SKIP_COMMENTS.union([tokenize.STRING]):
            skip_lines.add(line)
    for token_type, text, start, end, line in tokens:
        if token_type == tokenize.STRING and skip_lines:
            continue
        if token_type in (tokenize.STRING, tokenize.COMMENT):
            if prev_token is None or prev_token in SKIP_TOKENS:
                lines_split: List[str] = line.splitlines()
                for line_num, physical_line in enumerate(lines_split):
                    if start[0] + line_num == 1 and line.startswith('#!'):
                        return
                    length: int = len(physical_line)
                    chunks = physical_line.split()
                    if token_type == tokenize.COMMENT:
                        if len(chunks) == 2 and length - len(chunks[-1]) < MAX_DOC_LENGTH:
                            continue
                    if len(chunks) == 1 and line_num + 1 < len(lines_split):
                        if len(chunks) == 1 and length - len(chunks[-1]) < MAX_DOC_LENGTH:
                            continue
                    if length > max_doc_length:
                        doc_error: Tuple[int, int] = (start[0] + line_num, max_doc_length)
                        yield (doc_error, 'W505 doc line too long (%d > %d characters)' % (length, max_doc_length))
        prev_token = token_type


def readlines(filename: str) -> List[str]:
    """Read the source code."""
    try:
        with tokenize.open(filename) as f:
            return f.readlines()
    except (LookupError, SyntaxError, UnicodeError):
        with open(filename, encoding='latin-1') as f:
            return f.readlines()


def stdin_get_value() -> str:
    """Read the value from stdin."""
    return TextIOWrapper(sys.stdin.buffer, errors='ignore').read()


noqa: Callable[[str], Optional[Any]] = lru_cache(512)(re.compile('# no(?:qa|pep8)\\b', re.I).search)


def expand_indent(line: str) -> int:
    """Return the amount of indentation.

    Tabs are expanded to the next multiple of 8.

    >>> expand_indent('    ')
    4
    >>> expand_indent('\t')
    8
    >>> expand_indent('       \t')
    8
    >>> expand_indent('        \t')
    16
    """
    line = line.rstrip('\n\r')
    if '\t' not in line:
        return len(line) - len(line.lstrip())
    result: int = 0
    for char in line:
        if char == '\t':
            result = result // 8 * 8 + 8
        elif char == ' ':
            result += 1
        else:
            break
    return result


def mute_string(text: str) -> str:
    """Replace contents with 'xxx' to prevent syntax matching.

    >>> mute_string('"abc"')
    '"xxx"'
    >>> mute_string("'''abc'''")
    "'''xxx'''"
    >>> mute_string("r'abc'")
    "r'xxx'"
    """
    start: int = text.index(text[-1]) + 1
    end: int = len(text) - 1
    if text[-3:] in ('"""', "'''"):
        start += 2
        end -= 2
    return text[:start] + 'x' * (end - start) + text[end:]


def parse_udiff(diff: str, patterns: Optional[Any] = None, parent: str = '.') -> Dict[str, Set[int]]:
    """Return a dictionary of matching lines."""
    rv: Dict[str, Set[int]] = {}
    path: Optional[str] = None
    nrows: Optional[int] = None
    for line in diff.splitlines():
        if nrows:
            if line[:1] != '-':
                nrows -= 1
            continue
        if line[:3] == '@@ ':
            hunk_match = HUNK_REGEX.match(line)
            row, nrows = (int(g or '1') for g in hunk_match.groups())
            rv[path].update(range(row, row + nrows))  # type: ignore
        elif line[:3] == '+++':
            path = line[4:].split('\t', 1)[0]
            if path[:2] in ('b/', 'w/', 'i/'):
                path = path[2:]
            rv[path] = set()
    return {os.path.join(parent, filepath): rows for filepath, rows in rv.items() if rows and filename_match(filepath, patterns)}


def normalize_paths(value: Any, parent: str = os.curdir) -> List[str]:
    """Parse a comma-separated list of paths.

    Return a list of absolute paths.
    """
    if not value:
        return []
    if isinstance(value, list):
        return value
    paths: List[str] = []
    for path in value.split(','):
        path = path.strip()
        if '/' in path:
            path = os.path.abspath(os.path.join(parent, path))
        paths.append(path.rstrip('/'))
    return paths


def filename_match(filename: str, patterns: Optional[Any], default: bool = True) -> bool:
    """Check if patterns contains a pattern that matches filename.

    If patterns is unspecified, this always returns True.
    """
    if not patterns:
        return default
    return any((fnmatch(filename, pattern) for pattern in patterns))


def update_counts(s: str, counts: Dict[str, int]) -> None:
    """Adds one to the counts of each appearance of characters in s,
        for characters in counts"""
    for char in s:
        if char in counts:
            counts[char] += 1


def _is_eol_token(token: Tuple[Any, ...]) -> bool:
    return token[0] in NEWLINE or token[4][token[3][1]:].lstrip() == '\\\n'


class Checker:
    """Load a Python source file, tokenize it, check coding style."""

    def __init__(self, filename: Optional[str] = None, lines: Optional[List[str]] = None, options: Any = None, report: Optional[Any] = None, **kwargs: Any) -> None:
        if options is None:
            options = StyleGuide(kwargs).options
        else:
            assert not kwargs
        self._io_error: Optional[str] = None
        self._physical_checks: Any = options.physical_checks
        self._logical_checks: Any = options.logical_checks
        self._ast_checks: Any = options.ast_checks
        self.max_line_length: int = options.max_line_length
        self.max_doc_length: Optional[int] = options.max_doc_length
        self.indent_size: int = options.indent_size
        self.multiline: bool = False
        self.hang_closing: bool = options.hang_closing
        self.indent_size = options.indent_size
        self.verbose: int = options.verbose
        self.filename: str = filename if filename is not None else ''
        self._checker_states: Dict[str, Any] = {}
        if filename is None:
            self.filename = 'stdin'
            self.lines: List[str] = lines or []
        elif filename == '-':
            self.filename = 'stdin'
            self.lines = stdin_get_value().splitlines(True)
        elif lines is None:
            try:
                self.lines = readlines(filename)
            except OSError:
                exc_type, exc = sys.exc_info()[:2]
                self._io_error = f'{exc_type.__name__}: {exc}'
                self.lines = []
        else:
            self.lines = lines
        if self.lines:
            ord0: int = ord(self.lines[0][0])
            if ord0 in (239, 65279):
                if ord0 == 65279:
                    self.lines[0] = self.lines[0][1:]
                elif self.lines[0][:3] == 'ï»¿':
                    self.lines[0] = self.lines[0][3:]
        self.report = report or options.report
        self.report_error: Any = self.report.error
        self.noqa: bool = False

    def report_invalid_syntax(self) -> None:
        """Check if the syntax is valid."""
        exc_type, exc = sys.exc_info()[:2]
        if len(exc.args) > 1:
            offset = exc.args[1]
            if len(offset) > 2:
                offset = offset[1:3]
        else:
            offset = (1, 0)
        self.report_error(offset[0], offset[1] or 0, f'E901 {exc_type.__name__}: {exc.args[0]}', self.report_invalid_syntax)

    def readline(self) -> str:
        """Get the next line from the input buffer."""
        if self.line_number >= self.total_lines:
            return ''
        line: str = self.lines[self.line_number]
        self.line_number += 1
        if self.indent_char is None and line[:1] in WHITESPACE:
            self.indent_char = line[0]
        return line

    def run_check(self, check: Callable, argument_names: List[str]) -> Any:
        """Run a check plugin."""
        arguments: List[Any] = []
        for name in argument_names:
            arguments.append(getattr(self, name))
        return check(*arguments)

    def init_checker_state(self, name: str, argument_names: List[str]) -> None:
        """Prepare custom state for the specific checker plugin."""
        if 'checker_state' in argument_names:
            self.checker_state = self._checker_states.setdefault(name, {})

    def check_physical(self, line: str) -> None:
        """Run all physical checks on a raw input line."""
        self.physical_line = line
        for name, check, argument_names in self._physical_checks:
            self.init_checker_state(name, argument_names)
            result: Any = self.run_check(check, argument_names)
            if result is not None:
                offset, text = result
                self.report_error(self.line_number, offset, text, check)
                if text[:4] == 'E101':
                    self.indent_char = line[0]

    def build_tokens_line(self) -> List[Tuple[int, Tuple[int, int]]]:
        """Build a logical line from tokens."""
        logical: List[str] = []
        comments: List[str] = []
        length: int = 0
        prev_row: Optional[int] = None
        prev_col: Optional[int] = None
        mapping: Optional[List[Tuple[int, Tuple[int, int]]]] = None
        for token_type, text, start, end, line in self.tokens:
            if token_type in SKIP_TOKENS:
                continue
            if not mapping:
                mapping = [(0, start)]
            if token_type == tokenize.COMMENT:
                comments.append(text)
                continue
            if token_type == tokenize.STRING:
                text = mute_string(text)
            if prev_row is not None:
                start_row, start_col = start
                if prev_row != start_row:
                    prev_text = self.lines[prev_row - 1][prev_col - 1]
                    if prev_text == ',' or (prev_text not in '{[(' and text not in '}])'):
                        text = ' ' + text
                elif prev_col != start_col:
                    text = line[prev_col:start_col] + text
            logical.append(text)
            length += len(text)
            mapping.append((length, end))
            prev_row, prev_col = end
        self.logical_line = ''.join(logical)
        self.noqa = bool(comments and noqa(''.join(comments)))
        return mapping if mapping is not None else []

    def check_logical(self) -> None:
        """Build a line from tokens and run all logical checks on it."""
        self.report.increment_logical_line()
        mapping: List[Tuple[int, Tuple[int, int]]] = self.build_tokens_line()
        if not mapping:
            return
        mapping_offsets: List[int] = [offset for offset, _ in mapping]
        start_row, start_col = mapping[0][1]
        start_line: str = self.lines[start_row - 1]
        self.indent_level = expand_indent(start_line[:start_col])
        if self.blank_before < self.blank_lines:
            self.blank_before = self.blank_lines
        if self.verbose >= 2:
            print(self.logical_line[:80].rstrip())
        for name, check, argument_names in self._logical_checks:
            if self.verbose >= 4:
                print('   ' + name)
            self.init_checker_state(name, argument_names)
            for offset, text in self.run_check(check, argument_names) or ():
                if not isinstance(offset, tuple):
                    token_offset, pos = mapping[bisect.bisect_left(mapping_offsets, offset)]
                    offset = (pos[0], pos[1] + offset - token_offset)
                self.report_error(offset[0], offset[1], text, check)
        if self.logical_line:
            self.previous_indent_level = self.indent_level
            self.previous_logical = self.logical_line
            if not self.indent_level:
                self.previous_unindented_logical_line = self.logical_line
        self.blank_lines = 0
        self.tokens = []

    def check_ast(self) -> None:
        """Build the file's AST and run all AST checks."""
        try:
            tree = compile(''.join(self.lines), '', 'exec', PyCF_ONLY_AST)
        except (ValueError, SyntaxError, TypeError):
            return self.report_invalid_syntax()
        for name, cls, __ in self._ast_checks:
            checker = cls(tree, self.filename)
            for lineno, offset, text, check in checker.run():
                if not self.lines or not noqa(self.lines[lineno - 1]):
                    self.report_error(lineno, offset, text, check)

    def generate_tokens(self) -> Iterable[Tuple[Any, ...]]:
        """Tokenize file, run physical line checks and yield tokens."""
        if self._io_error:
            self.report_error(1, 0, 'E902 %s' % self._io_error, readlines)
        tokengen = tokenize.generate_tokens(self.readline)
        try:
            prev_physical: str = ''
            for token in tokengen:
                if token[2][0] > self.total_lines:
                    return
                self.noqa = bool(token[4] and noqa(token[4]))
                self.maybe_check_physical(token, prev_physical)
                yield token
                prev_physical = token[4]
        except (SyntaxError, tokenize.TokenError):
            self.report_invalid_syntax()

    def maybe_check_physical(self, token: Tuple[Any, ...], prev_physical: str) -> None:
        """If appropriate for token, check current physical line(s)."""
        if _is_eol_token(token):
            if token[4] == '':
                self.check_physical(prev_physical)
            else:
                self.check_physical(token[4])
        elif token[0] == tokenize.STRING and '\n' in token[1]:
            if noqa(token[4]):
                return
            self.multiline = True
            self.line_number = token[2][0]
            _, src, (_, offset), _, _ = token
            src = self.lines[self.line_number - 1][:offset] + src
            for line in src.split('\n')[:-1]:
                self.check_physical(line + '\n')
                self.line_number += 1
            self.multiline = False

    def check_all(self, expected: Optional[Any] = None, line_offset: int = 0) -> Any:
        """Run all checks on the input file."""
        self.report.init_file(self.filename, self.lines, expected, line_offset)
        self.total_lines = len(self.lines)
        if self._ast_checks:
            self.check_ast()
        self.line_number = 0
        self.indent_char = None
        self.indent_level = self.previous_indent_level = 0
        self.previous_logical = ''
        self.previous_unindented_logical_line = ''
        self.tokens = []
        self.blank_lines = self.blank_before = 0
        parens: int = 0
        for token in self.generate_tokens():
            self.tokens.append(token)
            token_type, text = token[0:2]
            if self.verbose >= 3:
                if token[2][0] == token[3][0]:
                    pos = '[{}:{}]'.format(token[2][1] or '', token[3][1])
                else:
                    pos = 'l.%s' % token[3][0]
                print('l.%s\t%s\t%s\t%r' % (token[2][0], pos, tokenize.tok_name[token[0]], text))
            if token_type == tokenize.OP:
                if text in '([{':
                    parens += 1
                elif text in '}])':
                    parens -= 1
            elif not parens:
                if token_type in NEWLINE:
                    if token_type == tokenize.NEWLINE:
                        self.check_logical()
                        self.blank_before = 0
                    elif len(self.tokens) == 1:
                        self.blank_lines += 1
                        del self.tokens[0]
                    else:
                        self.check_logical()
        if self.tokens:
            self.check_physical(self.lines[-1])
            self.check_logical()
        return self.report.get_file_results()


class BaseReport:
    """Collect the results of the checks."""
    print_filename: bool = False

    def __init__(self, options: Any) -> None:
        self._benchmark_keys: List[str] = options.benchmark_keys
        self._ignore_code: Any = options.ignore_code
        self.elapsed: float = 0
        self.total_errors: int = 0
        self.counters: Dict[str, int] = dict.fromkeys(self._benchmark_keys, 0)
        self.messages: Dict[str, str] = {}

    def start(self) -> None:
        """Start the timer."""
        self._start_time = time.time()

    def stop(self) -> None:
        """Stop the timer."""
        self.elapsed = time.time() - self._start_time

    def init_file(self, filename: str, lines: List[str], expected: Optional[Any], line_offset: int) -> None:
        """Signal a new file."""
        self.filename: str = filename
        self.lines: List[str] = lines
        self.expected = expected or ()
        self.line_offset: int = line_offset
        self.file_errors: int = 0
        self.counters['files'] += 1
        self.counters['physical lines'] += len(lines)

    def increment_logical_line(self) -> None:
        """Signal a new logical line."""
        self.counters['logical lines'] += 1

    def error(self, line_number: int, offset: int, text: str, check: Any) -> Optional[str]:
        """Report an error, according to options."""
        code: str = text[:4]
        if self._ignore_code(code):
            return None
        if code in self.counters:
            self.counters[code] += 1
        else:
            self.counters[code] = 1
            self.messages[code] = text[5:]
        if code in self.expected:
            return None
        if self.print_filename and (not self.file_errors):
            print(self.filename)
        self.file_errors += 1
        self.total_errors += 1
        return code

    def get_file_results(self) -> int:
        """Return the count of errors and warnings for this file."""
        return self.file_errors

    def get_count(self, prefix: str = '') -> int:
        """Return the total count of errors and warnings."""
        return sum((self.counters[key] for key in self.messages if key.startswith(prefix)))

    def get_statistics(self, prefix: str = '') -> List[str]:
        """Get statistics for message codes that start with the prefix.

        prefix='' matches all errors and warnings
        prefix='E' matches all errors
        prefix='W' matches all warnings
        prefix='E4' matches all errors that have to do with imports
        """
        return ['%-7s %s %s' % (self.counters[key], key, self.messages[key]) for key in sorted(self.messages) if key.startswith(prefix)]

    def print_statistics(self, prefix: str = '') -> None:
        """Print overall statistics (number of errors and warnings)."""
        for line in self.get_statistics(prefix):
            print(line)

    def print_benchmark(self) -> None:
        """Print benchmark numbers."""
        print('{:<7.2f} {}'.format(self.elapsed, 'seconds elapsed'))
        if self.elapsed:
            for key in self._benchmark_keys:
                print('%-7d %s per second (%d total)' % (self.counters[key] / self.elapsed, key, self.counters[key]))

        
class FileReport(BaseReport):
    """Collect the results of the checks and print the filenames."""
    print_filename: bool = True


class StandardReport(BaseReport):
    """Collect and print the results of the checks."""

    def __init__(self, options: Any) -> None:
        super().__init__(options)
        self._fmt: str = REPORT_FORMAT.get(options.format.lower(), options.format)
        self._repeat: bool = options.repeat
        self._show_source: bool = options.show_source
        self._show_pep8: bool = options.show_pep8

    def init_file(self, filename: str, lines: List[str], expected: Optional[Any], line_offset: int) -> None:
        """Signal a new file."""
        self._deferred_print: List[Tuple[int, int, str, str, Optional[str]]] = []
        super().init_file(filename, lines, expected, line_offset)

    def error(self, line_number: int, offset: int, text: str, check: Any) -> Optional[str]:
        """Report an error, according to options."""
        code: Optional[str] = super().error(line_number, offset, text, check)
        if code and (self.counters[code] == 1 or self._repeat):
            self._deferred_print.append((line_number, offset, code, text[5:], check.__doc__))
        return code

    def get_file_results(self) -> int:
        """Print results and return the overall count for this file."""
        self._deferred_print.sort()
        for line_number, offset, code, text, doc in self._deferred_print:
            print(self._fmt % {'path': self.filename, 'row': self.line_offset + line_number, 'col': offset + 1, 'code': code, 'text': text})
            if self._show_source:
                if line_number > len(self.lines):
                    line = ''
                else:
                    line = self.lines[line_number - 1]
                print(line.rstrip())
                print(re.sub('\\S', ' ', line[:offset]) + '^')
            if self._show_pep8 and doc:
                print('    ' + doc.strip())
            sys.stdout.flush()
        return self.file_errors


class DiffReport(StandardReport):
    """Collect and print the results for the changed lines only."""

    def __init__(self, options: Any) -> None:
        super().__init__(options)
        self._selected: Any = options.selected_lines

    def error(self, line_number: int, offset: int, text: str, check: Any) -> Optional[str]:
        if line_number not in self._selected[self.filename]:
            return None
        return super().error(line_number, offset, text, check)


class StyleGuide:
    """Initialize a PEP-8 instance with few options."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.checker_class: Any = kwargs.pop('checker_class', Checker)
        parse_argv: bool = kwargs.pop('parse_argv', False)
        config_file: bool = kwargs.pop('config_file', False)
        parser: Optional[Any] = kwargs.pop('parser', None)
        options_dict: Dict[str, Any] = dict(*args, **kwargs)
        arglist: Optional[List[str]] = None if parse_argv else options_dict.get('paths', None)
        verbose: Optional[int] = options_dict.get('verbose', None)
        options, self.paths = process_options(arglist, parse_argv, config_file, parser, verbose)
        if options_dict:
            options.__dict__.update(options_dict)
            if 'paths' in options_dict:
                self.paths = options_dict['paths']
        self.runner: Callable[[str], Any] = self.input_file
        self.options = options
        if not options.reporter:
            options.reporter = BaseReport if options.quiet else StandardReport
        options.select = tuple(options.select or ())
        if not (options.select or options.ignore or options.testsuite or options.doctest) and DEFAULT_IGNORE:
            options.ignore = tuple(DEFAULT_IGNORE.split(','))
        else:
            options.ignore = ('',) if options.select else tuple(options.ignore)
        options.benchmark_keys = BENCHMARK_KEYS[:]
        options.ignore_code = self.ignore_code
        options.physical_checks = self.get_checks('physical_line')
        options.logical_checks = self.get_checks('logical_line')
        options.ast_checks = self.get_checks('tree')
        self.init_report()

    def init_report(self, reporter: Optional[Any] = None) -> Any:
        """Initialize the report instance."""
        self.options.report = (reporter or self.options.reporter)(self.options)
        return self.options.report

    def check_files(self, paths: Optional[List[str]] = None) -> Any:
        """Run all checks on the paths."""
        if paths is None:
            paths = self.paths
        report = self.options.report
        runner = self.runner
        report.start()
        try:
            for path in paths:
                if os.path.isdir(path):
                    self.input_dir(path)
                elif not self.excluded(path):
                    runner(path)
        except KeyboardInterrupt:
            print('... stopped')
        report.stop()
        return report

    def input_file(self, filename: str, lines: Optional[List[str]] = None, expected: Optional[Any] = None, line_offset: int = 0) -> Any:
        """Run all checks on a Python source file."""
        if self.options.verbose:
            print('checking %s' % filename)
        fchecker = self.checker_class(filename, lines=lines, options=self.options)
        return fchecker.check_all(expected=expected, line_offset=line_offset)

    def input_dir(self, dirname: str) -> None:
        """Check all files in this directory and all subdirectories."""
        dirname = dirname.rstrip('/')
        if self.excluded(dirname):
            return
        counters = self.options.report.counters
        verbose: Optional[int] = self.options.verbose
        filepatterns = self.options.filename
        runner = self.runner
        for root, dirs, files in os.walk(dirname):
            if verbose:
                print('directory ' + root)
            counters['directories'] += 1
            for subdir in sorted(dirs):
                if self.excluded(subdir, root):
                    dirs.remove(subdir)
            for filename in sorted(files):
                if filename_match(filename, filepatterns) and (not self.excluded(filename, root)):
                    runner(os.path.join(root, filename))

    def excluded(self, filename: str, parent: Optional[str] = None) -> bool:
        """Check if the file should be excluded.

        Check if 'options.exclude' contains a pattern matching filename.
        """
        if not self.options.exclude:
            return False
        basename: str = os.path.basename(filename)
        if filename_match(basename, self.options.exclude):
            return True
        if parent:
            filename = os.path.join(parent, filename)
        filename = os.path.abspath(filename)
        return filename_match(filename, self.options.exclude)

    def ignore_code(self, code: str) -> bool:
        """Check if the error code should be ignored.

        If 'options.select' contains a prefix of the error code,
        return False.  Else, if 'options.ignore' contains a prefix of
        the error code, return True.
        """
        if len(code) < 4 and any((s.startswith(code) for s in self.options.select)):
            return False
        return code.startswith(self.options.ignore) and (not code.startswith(self.options.select))

    def get_checks(self, argument_name: str) -> List[Tuple[str, Any, Optional[List[str]]]]:
        """Get all the checks for this category.

        Find all globally visible functions where the first argument
        name starts with argument_name and which contain selected tests.
        """
        checks: List[Tuple[str, Any, Optional[List[str]]]] = []
        for check, attrs in _checks[argument_name].items():
            codes, args = attrs
            if any((not (code and self.ignore_code(code)) for code in codes)):
                checks.append((check.__name__, check, args))
        return sorted(checks)

def get_parser(prog: str = 'pycodestyle', version: str = __version__) -> OptionParser:
    """Create the parser for the program."""
    parser = OptionParser(prog=prog, version=version, usage='%prog [options] input ...')
    parser.config_options = ['exclude', 'filename', 'select', 'ignore', 'max-line-length', 'max-doc-length', 'indent-size', 'hang-closing', 'count', 'format', 'quiet', 'show-pep8', 'show-source', 'statistics', 'verbose']
    parser.add_option('-v', '--verbose', default=0, action='count', help='print status messages, or debug with -vv')
    parser.add_option('-q', '--quiet', default=0, action='count', help='report only file names, or nothing with -qq')
    parser.add_option('-r', '--repeat', default=True, action='store_true', help='(obsolete) show all occurrences of the same error')
    parser.add_option('--first', action='store_false', dest='repeat', help='show first occurrence of each error')
    parser.add_option('--exclude', metavar='patterns', default=DEFAULT_EXCLUDE, help='exclude files or directories which match these comma separated patterns (default: %default)')
    parser.add_option('--filename', metavar='patterns', default='*.py', help='when parsing directories, only check filenames matching these comma separated patterns (default: %default)')
    parser.add_option('--select', metavar='errors', default='', help='select errors and warnings (e.g. E,W6)')
    parser.add_option('--ignore', metavar='errors', default='', help='skip errors and warnings (e.g. E4,W) (default: %s)' % DEFAULT_IGNORE)
    parser.add_option('--show-source', action='store_true', help='show source code for each error')
    parser.add_option('--show-pep8', action='store_true', help='show text of PEP 8 for each error (implies --first)')
    parser.add_option('--statistics', action='store_true', help='count errors and warnings')
    parser.add_option('--count', action='store_true', help='print total number of errors and warnings to standard error and set exit code to 1 if total is not null')
    parser.add_option('--max-line-length', type='int', metavar='n', default=MAX_LINE_LENGTH, help='set maximum allowed line length (default: %default)')
    parser.add_option('--max-doc-length', type='int', metavar='n', default=None, help='set maximum allowed doc line length and perform these checks (unchecked if not set)')
    parser.add_option('--indent-size', type='int', metavar='n', default=INDENT_SIZE, help='set how many spaces make up an indent (default: %default)')
    parser.add_option('--hang-closing', action='store_true', help="hang closing bracket instead of matching indentation of opening bracket's line")
    parser.add_option('--format', metavar='format', default='default', help='set the error format [default|pylint|<custom>]')
    parser.add_option('--diff', action='store_true', help='report changes only within line number ranges in the unified diff received on STDIN')
    group = parser.add_option_group('Testing Options')
    if os.path.exists(TESTSUITE_PATH):
        group.add_option('--testsuite', metavar='dir', help='run regression tests from dir')
        group.add_option('--doctest', action='store_true', help='run doctest on myself')
    group.add_option('--benchmark', action='store_true', help='measure processing speed')
    return parser


def read_config(options: Any, args: List[str], arglist: Any, parser: OptionParser) -> Any:
    """Read and parse configurations.

    If a config file is specified on the command line with the
    "--config" option, then only it is used for configuration.

    Otherwise, the user configuration (~/.config/pycodestyle) and any
    local configurations in the current directory or above will be
    merged together (in that order) using the read method of
    ConfigParser.
    """
    config: Any = RawConfigParser()
    cli_conf: Any = options.config
    local_dir: str = os.curdir
    if USER_CONFIG and os.path.isfile(USER_CONFIG):
        if options.verbose:
            print('user configuration: %s' % USER_CONFIG)
        config.read(USER_CONFIG)
    parent: Optional[str] = args and os.path.abspath(os.path.commonprefix(args))
    tail: Any = parent
    while tail:
        if config.read((os.path.join(parent, fn) for fn in PROJECT_CONFIG)):
            local_dir = parent
            if options.verbose:
                print('local configuration: in %s' % parent)
            break
        parent, tail = os.path.split(parent)
    if cli_conf and os.path.isfile(cli_conf):
        if options.verbose:
            print('cli configuration: %s' % cli_conf)
        config.read(cli_conf)
    pycodestyle_section: Optional[str] = None
    if config.has_section(parser.prog):
        pycodestyle_section = parser.prog
    elif config.has_section('pep8'):
        pycodestyle_section = 'pep8'
        warnings.warn('[pep8] section is deprecated. Use [pycodestyle].')
    if pycodestyle_section:
        option_list: Dict[str, Any] = {o.dest: o.type or o.action for o in parser.option_list}
        new_options, __ = parser.parse_args([])
        for opt in config.options(pycodestyle_section):
            if opt.replace('_', '-') not in parser.config_options:
                print("  unknown option '%s' ignored" % opt)
                continue
            if options.verbose > 1:
                print('  {} = {}'.format(opt, config.get(pycodestyle_section, opt)))
            normalized_opt: str = opt.replace('-', '_')
            opt_type: Any = option_list[normalized_opt]
            if opt_type in ('int', 'count'):
                value = config.getint(pycodestyle_section, opt)
            elif opt_type in ('store_true', 'store_false'):
                value = config.getboolean(pycodestyle_section, opt)
            else:
                value = config.get(pycodestyle_section, opt)
                if normalized_opt == 'exclude':
                    value = normalize_paths(value, local_dir)
            setattr(new_options, normalized_opt, value)
        options, __ = parser.parse_args(arglist, values=new_options)
    options.doctest = options.testsuite = False
    return options


def process_options(arglist: Optional[List[str]] = None, parse_argv: bool = False, config_file: Optional[str] = None, parser: Optional[OptionParser] = None, verbose: Optional[int] = None) -> Tuple[Any, List[str]]:
    """Process options passed either via arglist or command line args.

    Passing in the ``config_file`` parameter allows other tools, such as
    flake8 to specify their own options to be processed in pycodestyle.
    """
    if not parser:
        parser = get_parser()
    if not parser.has_option('--config'):
        group = parser.add_option_group('Configuration', description='The project options are read from the [%s] section of the tox.ini file or the setup.cfg file located in any parent folder of the path(s) being processed.  Allowed options are: %s.' % (parser.prog, ', '.join(parser.config_options)))
        group.add_option('--config', metavar='path', default=config_file, help='user config file location')
    if not arglist and (not parse_argv):
        arglist = []
    options, args = parser.parse_args(arglist)
    options.reporter = None
    if verbose is not None:
        options.verbose = verbose
    if options.ensure_value('testsuite', False):
        args.append(options.testsuite)
    elif not options.ensure_value('doctest', False):
        if parse_argv and (not args):
            if options.diff or any((os.path.exists(name) for name in PROJECT_CONFIG)):
                args = ['.']
            else:
                parser.error('input not specified')
        options = read_config(options, args, arglist, parser)
        options.reporter = parse_argv and options.quiet == 1 and FileReport
    options.filename = _parse_multi_options(options.filename)
    options.exclude = normalize_paths(options.exclude)
    options.select = _parse_multi_options(options.select)
    options.ignore = _parse_multi_options(options.ignore)
    if options.diff:
        options.reporter = DiffReport
        stdin = stdin_get_value()
        options.selected_lines = parse_udiff(stdin, options.filename, args[0])
        args = sorted(options.selected_lines)
    return (options, args)


def _parse_multi_options(options: Optional[str], split_token: str = ',') -> Optional[List[str]]:
    """Split and strip and discard empties.

    Turns the following:

    A,
    B,

    into ["A", "B"]
    """
    if options:
        return [o.strip() for o in options.split(split_token) if o.strip()]
    else:
        return options


def _main() -> None:
    """Parse options and run checks on Python source."""
    import signal
    try:
        signal.signal(signal.SIGPIPE, lambda signum, frame: sys.exit(1))
    except AttributeError:
        pass
    style_guide = StyleGuide(parse_argv=True)
    options = style_guide.options
    if options.doctest or options.testsuite:
        from testsuite.support import run_tests
        report = run_tests(style_guide)
    else:
        report = style_guide.check_files()
    if options.statistics:
        report.print_statistics()
    if options.benchmark:
        report.print_benchmark()
    if options.testsuite and (not options.quiet):
        report.print_results()
    if report.total_errors:
        if options.count:
            sys.stderr.write(str(report.total_errors) + '\n')
        sys.exit(1)


if __name__ == '__main__':
    _main()