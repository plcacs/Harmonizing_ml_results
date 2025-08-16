#!/usr/bin/env python
# pycodestyle.py - Check Python source code formatting, according to
# PEP 8
#
# Copyright (C) 2006-2009 Johann C. Rocholl <johann@rocholl.net>
# Copyright (C) 2009-2014 Florent Xicluna <florent.xicluna@gmail.com>
# Copyright (C) 2014-2016 Ian Lee <ianlee1521@gmail.com>
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import annotations

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
from optparse import OptionParser, Option
from typing import (Any, Callable, Dict, FrozenSet, Iterable, Iterator, List, 
                    Optional, Pattern, Set, Tuple, Type, TypeVar, Union, cast)
from configparser import RawConfigParser
from io import TextIOWrapper

try:
    from configparser import RawConfigParser
    from io import TextIOWrapper
except ImportError:
    from ConfigParser import RawConfigParser  # type: ignore

T = TypeVar('T')
CheckFunction = Callable[..., Optional[Tuple[int, str]]]
CheckTreeFunction = Type[Any]
Token = Tuple[int, str, Tuple[int, int], Tuple[int, int], str]
Tokens = List[Token]
LogicalLine = str
PhysicalLine = str
Lines = List[str]
Error = Tuple[int, int, str, Any]
CheckerState = Dict[str, Any]

# this is a performance hack.  see https://bugs.python.org/issue43014
if (
        sys.version_info < (3, 10) and
        callable(getattr(tokenize, '_compile', None))
):  # pragma: no cover (<py310)
    tokenize._compile = lru_cache()(tokenize._compile)  # type: ignore

__version__: str = '2.8.0'

DEFAULT_EXCLUDE: str = '.svn,CVS,.bzr,.hg,.git,__pycache__,.tox'
DEFAULT_IGNORE: str = 'E121,E123,E126,E226,E24,E704,W503,W504'
USER_CONFIG: Optional[str] = None
try:
    if sys.platform == 'win32':
        USER_CONFIG = os.path.expanduser(r'~\.pycodestyle')
    else:
        USER_CONFIG = os.path.join(
            os.getenv('XDG_CONFIG_HOME') or os.path.expanduser('~/.config'),
            'pycodestyle'
        )
except ImportError:
    pass

PROJECT_CONFIG: Tuple[str, str] = ('setup.cfg', 'tox.ini')
TESTSUITE_PATH: str = os.path.join(os.path.dirname(__file__), 'testsuite')
MAX_LINE_LENGTH: int = 79
BLANK_LINES_CONFIG: Dict[str, int] = {
    'top_level': 2,
    'method': 1,
}
MAX_DOC_LENGTH: int = 72
INDENT_SIZE: int = 4
REPORT_FORMAT: Dict[str, str] = {
    'default': '%(path)s:%(row)d:%(col)d: %(code)s %(text)s',
    'pylint': '%(path)s:%(row)d: [%(code)s] %(text)s',
}

PyCF_ONLY_AST: int = 1024
SINGLETONS: FrozenSet[str] = frozenset(['False', 'None', 'True'])
KEYWORDS: FrozenSet[str] = frozenset(keyword.kwlist + ['print', 'async']) - SINGLETONS
UNARY_OPERATORS: FrozenSet[str] = frozenset(['>>', '**', '*', '+', '-'])
ARITHMETIC_OP: FrozenSet[str] = frozenset(['**', '*', '/', '//', '+', '-', '@'])
WS_OPTIONAL_OPERATORS: FrozenSet[str] = ARITHMETIC_OP.union(['^', '&', '|', '<<', '>>', '%'])
ASSIGNMENT_EXPRESSION_OP: List[str] = [':='] if sys.version_info >= (3, 8) else []
WS_NEEDED_OPERATORS: FrozenSet[str] = frozenset([
    '**=', '*=', '/=', '//=', '+=', '-=', '!=', '<>', '<', '>',
    '%=', '^=', '&=', '|=', '==', '<=', '>=', '<<=', '>>=', '=',
    'and', 'in', 'is', 'or', '->'] +
    ASSIGNMENT_EXPRESSION_OP)
WHITESPACE: FrozenSet[str] = frozenset(' \t\xa0')
NEWLINE: FrozenSet[int] = frozenset([tokenize.NL, tokenize.NEWLINE])
SKIP_TOKENS: FrozenSet[int] = NEWLINE.union([tokenize.INDENT, tokenize.DEDENT])
SKIP_COMMENTS: FrozenSet[int] = SKIP_TOKENS.union([tokenize.COMMENT, tokenize.ERRORTOKEN])
BENCHMARK_KEYS: List[str] = ['directories', 'files', 'logical lines', 'physical lines']

INDENT_REGEX: Pattern[str] = re.compile(r'([ \t]*)')
RAISE_COMMA_REGEX: Pattern[str] = re.compile(r'raise\s+\w+\s*,')
RERAISE_COMMA_REGEX: Pattern[str] = re.compile(r'raise\s+\w+\s*,.*,\s*\w+\s*$')
ERRORCODE_REGEX: Pattern[str] = re.compile(r'\b[A-Z]\d{3}\b')
DOCSTRING_REGEX: Pattern[str] = re.compile(r'u?r?["\']')
EXTRANEOUS_WHITESPACE_REGEX: Pattern[str] = re.compile(r'[\[({][ \t]|[ \t][\]}),;:](?!=)')
WHITESPACE_AFTER_COMMA_REGEX: Pattern[str] = re.compile(r'[,;:]\s*(?:  |\t)')
COMPARE_SINGLETON_REGEX: Pattern[str] = re.compile(r'(\bNone|\bFalse|\bTrue)?\s*([=!]=)'
                                     r'\s*(?(1)|(None|False|True))\b')
COMPARE_NEGATIVE_REGEX: Pattern[str] = re.compile(r'\b(?<!is\s)(not)\s+[^][)(}{ ]+\s+'
                                    r'(in|is)\s')
COMPARE_TYPE_REGEX: Pattern[str] = re.compile(
    r'(?:[=!]=|is(?:\s+not)?)\s+type(?:\s*\(\s*([^)]*[^ )])\s*\))' +
    r'|\btype(?:\s*\(\s*([^)]*[^ )])\s*\))\s+(?:[=!]=|is(?:\s+not)?)'
)
KEYWORD_REGEX: Pattern[str] = re.compile(r'(\s*)\b(?:%s)\b(\s*)' % r'|'.join(KEYWORDS))
OPERATOR_REGEX: Pattern[str] = re.compile(r'(?:[^,\s])(\s*)(?:[-+*/|!<=>%&^]+|:=)(\s*)')
LAMBDA_REGEX: Pattern[str] = re.compile(r'\blambda\b')
HUNK_REGEX: Pattern[str] = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@.*$')
STARTSWITH_DEF_REGEX: Pattern[str] = re.compile(r'^(async\s+def|def)\b')
STARTSWITH_TOP_LEVEL_REGEX: Pattern[str] = re.compile(r'^(async\s+def\s+|def\s+|class\s+|@)')
STARTSWITH_INDENT_STATEMENT_REGEX: Pattern[str] = re.compile(
    r'^\s*({})\b'.format('|'.join(s.replace(' ', r'\s+') for s in (
        'def', 'async def',
        'for', 'async for',
        'if', 'elif', 'else',
        'try', 'except', 'finally',
        'with', 'async with',
        'class',
        'while',
    )))
)
DUNDER_REGEX: Pattern[str] = re.compile(r"^__([^\s]+)__(?::\s*[a-zA-Z.0-9_\[\]\"]+)? = ")
BLANK_EXCEPT_REGEX: Pattern[str] = re.compile(r"except\s*:")

_checks: Dict[str, Dict[Union[CheckFunction, CheckTreeFunction], Tuple[List[str], Optional[List[str]]]] = {
    'physical_line': {},
    'logical_line': {},
    'tree': {}
}

def _get_parameters(function: Callable[..., Any]) -> List[str]:
    return [parameter.name
            for parameter
            in inspect.signature(function).parameters.values()
            if parameter.kind == parameter.POSITIONAL_OR_KEYWORD]

def register_check(check: Union[CheckFunction, CheckTreeFunction], 
                  codes: Optional[List[str]] = None) -> Union[CheckFunction, CheckTreeFunction]:
    """Register a new check object."""
    def _add_check(check: Union[CheckFunction, CheckTreeFunction], 
                  kind: str, 
                  codes: Optional[List[str]], 
                  args: Optional[List[str]]) -> None:
        if check in _checks[kind]:
            _checks[kind][check][0].extend(codes or [])
        else:
            _checks[kind][check] = (codes or [''], args)
            
    if inspect.isfunction(check):
        args = _get_parameters(check)
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
    r"""Never mix tabs and spaces."""
    indent = INDENT_REGEX.match(physical_line).group(1)
    for offset, char in enumerate(indent):
        if char != indent_char:
            return offset, "E101 indentation contains mixed spaces and tabs"
    return None

@register_check
def tabs_obsolete(physical_line: str) -> Optional[Tuple[int, str]]:
    r"""On new projects, spaces-only are strongly recommended over tabs."""
    indent = INDENT_REGEX.match(physical_line).group(1)
    if '\t' in indent:
        return indent.index('\t'), "W191 indentation contains tabs"
    return None

@register_check
def trailing_whitespace(physical_line: str) -> Optional[Tuple[int, str]]:
    r"""Trailing whitespace is superfluous."""
    physical_line = physical_line.rstrip('\n')    # chr(10), newline
    physical_line = physical_line.rstrip('\r')    # chr(13), carriage return
    physical_line = physical_line.rstrip('\x0c')  # chr(12), form feed, ^L
    stripped = physical_line.rstrip(' \t\v')
    if physical_line != stripped:
        if stripped:
            return len(stripped), "W291 trailing whitespace"
        else:
            return 0, "W293 blank line contains whitespace"
    return None

@register_check
def trailing_blank_lines(physical_line: str, 
                        lines: List[str], 
                        line_number: int, 
                        total_lines: int) -> Optional[Tuple[int, str]]:
    r"""Trailing blank lines are superfluous."""
    if line_number == total_lines:
        stripped_last_line = physical_line.rstrip('\r\n')
        if physical_line and not stripped_last_line:
            return 0, "W391 blank line at end of file"
        if stripped_last_line == physical_line:
            return len(lines[-1]), "W292 no newline at end of file"
    return None

@register_check
def maximum_line_length(physical_line: str, 
                       max_line_length: int, 
                       multiline: bool,
                       line_number: int, 
                       noqa: bool) -> Optional[Tuple[int, str]]:
    r"""Limit all lines to a maximum of 79 characters."""
    line = physical_line.rstrip()
    length = len(line)
    if length > max_line_length and not noqa:
        if line_number == 1 and line.startswith('#!'):
            return None
        chunks = line.split()
        if ((len(chunks) == 1 and multiline) or
            (len(chunks) == 2 and chunks[0] == '#')) and \
                len(line) - len(chunks[-1]) < max_line_length - 7:
            return None
        if length > max_line_length:
            return (max_line_length, "E501 line too long "
                    "(%d > %d characters)" % (length, max_line_length))
    return None

def _is_one_liner(logical_line: str, 
                 indent_level: int, 
                 lines: List[str], 
                 line_number: int) -> bool:
    if not STARTSWITH_TOP_LEVEL_REGEX.match(logical_line):
        return False

    line_idx = line_number - 1

    if line_idx < 1:
        prev_indent = 0
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

    next_idx = line_idx + 1
    while next_idx < len(lines):
        if lines[next_idx].strip():
            break
        else:
            next_idx += 1
    else:
        return True

    return expand_indent(lines[next_idx]) <= indent_level

@register_check
def blank_lines(logical_line: str, 
               blank_lines: int, 
               indent_level: int, 
               line_number: int,
               blank_before: int, 
               previous_logical: str,
               previous_unindented_logical_line: str, 
               previous_indent_level: int,
               lines: List[str]) -> Iterator[Tuple[int, str]]:
    r"""Separate top-level function and class definitions with two blank lines."""
    top_level_lines = BLANK_LINES_CONFIG['top_level']
    method_lines = BLANK_LINES_CONFIG['method']

    if not previous_logical and blank_before < top_level_lines:
        return
    if previous_logical.startswith('@'):
        if blank_lines:
            yield 0, "E304 blank lines found after function decorator"
    elif (blank_lines > top_level_lines or
            (indent_level and blank_lines == method_lines + 1)
          ):
        yield 0, "E303 too many blank lines (%d)" % blank_lines
    elif STARTSWITH_TOP_LEVEL_REGEX.match(logical_line):
        if (_is_one_liner(logical_line, indent_level, lines, line_number) and
                blank_before == 0):
            return
        if indent_level:
            if not (blank_before == method_lines or
                    previous_indent_level < indent_level or
                    DOCSTRING_REGEX.match(previous_logical)
                    ):
                ancestor_level = indent_level
                nested = False
                for line in lines[line_number - top_level_lines::-1]:
                    if line.strip() and expand_indent(line) < ancestor_level:
                        ancestor_level = expand_indent(line)
                        nested = STARTSWITH_DEF_REGEX.match(line.lstrip())
                        if nested or ancestor_level == 0:
                            break
                if nested:
                    yield 0, "E306 expected %s blank line before a " \
                        "nested definition, found 0" % (method_lines,)
                else:
                    yield 0, "E301 expected {} blank line, found 0".format(
                        method_lines)
        elif blank_before != top_level_lines:
            yield 0, "E302 expected %s blank lines, found %d" % (
                top_level_lines, blank_before)
    elif (logical_line and
            not indent_level and
            blank_before != top_level_lines and
            previous_unindented_logical_line.startswith(('def ', 'class '))
          ):
        yield 0, "E305 expected %s blank lines after " \
            "class or function definition, found %d" % (
                top_level_lines, blank_before)

@register_check
def extraneous_whitespace(logical_line: str) -> Iterator[Tuple[int, str]]:
    r"""Avoid extraneous whitespace."""
    line = logical_line
    for match in EXTRANEOUS_WHITESPACE