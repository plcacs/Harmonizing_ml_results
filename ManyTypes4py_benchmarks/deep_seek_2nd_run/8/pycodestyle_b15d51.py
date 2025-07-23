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
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Match, Optional, Pattern, Set, Tuple, Union

try:
    from configparser import RawConfigParser
    from io import TextIOWrapper
except ImportError:
    from ConfigParser import RawConfigParser

if sys.version_info < (3, 10) and callable(getattr(tokenize, '_compile', None)):
    tokenize._compile = lru_cache()(tokenize._compile)

__version__: str = '2.8.0'
DEFAULT_EXCLUDE: str = '.svn,CVS,.bzr,.hg,.git,__pycache__,.tox'
DEFAULT_IGNORE: str = 'E121,E123,E126,E226,E24,E704,W503,W504'

try:
    if sys.platform == 'win32':
        USER_CONFIG: Optional[str] = os.path.expanduser('~\\.pycodestyle')
    else:
        USER_CONFIG = os.path.join(os.getenv('XDG_CONFIG_HOME') or os.path.expanduser('~/.config'), 'pycodestyle')
except ImportError:
    USER_CONFIG: Optional[str] = None

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
SINGLETONS: FrozenSet[str] = frozenset(['False', 'None', 'True'])
KEYWORDS: FrozenSet[str] = frozenset(keyword.kwlist + ['print', 'async']) - SINGLETONS
UNARY_OPERATORS: FrozenSet[str] = frozenset(['>>', '**', '*', '+', '-'])
ARITHMETIC_OP: FrozenSet[str] = frozenset(['**', '*', '/', '//', '+', '-', '@'])
WS_OPTIONAL_OPERATORS: FrozenSet[str] = ARITHMETIC_OP.union(['^', '&', '|', '<<', '>>', '%'])
ASSIGNMENT_EXPRESSION_OP: List[str] = [':='] if sys.version_info >= (3, 8) else []
WS_NEEDED_OPERATORS: FrozenSet[str] = frozenset([
    '**=', '*=', '/=', '//=', '+=', '-=', '!=', '<>', '<', '>', '%=', '^=', '&=', '|=',
    '==', '<=', '>=', '<<=', '>>=', '=', 'and', 'in', 'is', 'or', '->'
] + ASSIGNMENT_EXPRESSION_OP)
WHITESPACE: FrozenSet[str] = frozenset(' \t\xa0')
NEWLINE: FrozenSet[int] = frozenset([tokenize.NL, tokenize.NEWLINE])
SKIP_TOKENS: FrozenSet[int] = NEWLINE.union([tokenize.INDENT, tokenize.DEDENT])
SKIP_COMMENTS: FrozenSet[int] = SKIP_TOKENS.union([tokenize.COMMENT, tokenize.ERRORTOKEN])
BENCHMARK_KEYS: List[str] = ['directories', 'files', 'logical lines', 'physical lines']

INDENT_REGEX: Pattern[str] = re.compile('([ \\t]*)')
RAISE_COMMA_REGEX: Pattern[str] = re.compile('raise\\s+\\w+\\s*,')
RERAISE_COMMA_REGEX: Pattern[str] = re.compile('raise\\s+\\w+\\s*,.*,\\s*\\w+\\s*$')
ERRORCODE_REGEX: Pattern[str] = re.compile('\\b[A-Z]\\d{3}\\b')
DOCSTRING_REGEX: Pattern[str] = re.compile('u?r?["\\\']')
EXTRANEOUS_WHITESPACE_REGEX: Pattern[str] = re.compile('[\\[({][ \\t]|[ \\t][\\]}),;:](?!=)')
WHITESPACE_AFTER_COMMA_REGEX: Pattern[str] = re.compile('[,;:]\\s*(?:  |\\t)')
COMPARE_SINGLETON_REGEX: Pattern[str] = re.compile('(\\bNone|\\bFalse|\\bTrue)?\\s*([=!]=)\\s*(?(1)|(None|False|True))\\b')
COMPARE_NEGATIVE_REGEX: Pattern[str] = re.compile('\\b(?<!is\\s)(not)\\s+[^][)(}{ ]+\\s+(in|is)\\s')
COMPARE_TYPE_REGEX: Pattern[str] = re.compile('(?:[=!]=|is(?:\\s+not)?)\\s+type(?:\\s*\\(\\s*([^)]*[^ )])\\s*\\))' + '|\\btype(?:\\s*\\(\\s*([^)]*[^ )])\\s*\\))\\s+(?:[=!]=|is(?:\\s+not)?)')
KEYWORD_REGEX: Pattern[str] = re.compile('(\\s*)\\b(?:%s)\\b(\\s*)' % '|'.join(KEYWORDS))
OPERATOR_REGEX: Pattern[str] = re.compile('(?:[^,\\s])(\\s*)(?:[-+*/|!<=>%&^]+|:=)(\\s*)')
LAMBDA_REGEX: Pattern[str] = re.compile('\\blambda\\b')
HUNK_REGEX: Pattern[str] = re.compile('^@@ -\\d+(?:,\\d+)? \\+(\\d+)(?:,(\\d+))? @@.*$')
STARTSWITH_DEF_REGEX: Pattern[str] = re.compile('^(async\\s+def|def)\\b')
STARTSWITH_TOP_LEVEL_REGEX: Pattern[str] = re.compile('^(async\\s+def\\s+|def\\s+|class\\s+|@)')
STARTSWITH_INDENT_STATEMENT_REGEX: Pattern[str] = re.compile('^\\s*({})\\b'.format('|'.join((s.replace(' ', '\\s+') for s in ('def', 'async def', 'for', 'async for', 'if', 'elif', 'else', 'try', 'except', 'finally', 'with', 'async with', 'class', 'while')))))
DUNDER_REGEX: Pattern[str] = re.compile('^__([^\\s]+)__(?::\\s*[a-zA-Z.0-9_\\[\\]\\"]+)? = ')
BLANK_EXCEPT_REGEX: Pattern[str] = re.compile('except\\s*:')

_checks: Dict[str, Dict[Callable, Tuple[List[str], Optional[List[str]]]] = {
    'physical_line': {},
    'logical_line': {},
    'tree': {}
}

def _get_parameters(function: Callable) -> List[str]:
    return [parameter.name for parameter in inspect.signature(function).parameters.values() 
            if parameter.kind == parameter.POSITIONAL_OR_KEYWORD]

def register_check(check: Callable, codes: Optional[List[str]] = None) -> Callable:
    """Register a new check object."""

    def _add_check(check: Callable, kind: str, codes: Optional[List[str]], args: Optional[List[str]]) -> None:
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
    """Never mix tabs and spaces."""
    indent = INDENT_REGEX.match(physical_line).group(1)
    for offset, char in enumerate(indent):
        if char != indent_char:
            return (offset, 'E101 indentation contains mixed spaces and tabs')
    return None

@register_check
def tabs_obsolete(physical_line: str) -> Optional[Tuple[int, str]]:
    """On new projects, spaces-only are strongly recommended over tabs."""
    indent = INDENT_REGEX.match(physical_line).group(1)
    if '\t' in indent:
        return (indent.index('\t'), 'W191 indentation contains tabs')
    return None

@register_check
def trailing_whitespace(physical_line: str) -> Optional[Tuple[int, str]]:
    """Trailing whitespace is superfluous."""
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
    """Trailing blank lines are superfluous."""
    if line_number == total_lines:
        stripped_last_line = physical_line.rstrip('\r\n')
        if physical_line and (not stripped_last_line):
            return (0, 'W391 blank line at end of file')
        if stripped_last_line == physical_line:
            return (len(lines[-1]), 'W292 no newline at end of file')
    return None

@register_check
def maximum_line_length(physical_line: str, max_line_length: int, multiline: bool, line_number: int, noqa: bool) -> Optional[Tuple[int, str]]:
    """Limit all lines to a maximum of 79 characters."""
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
def blank_lines(logical_line: str, blank_lines: int, indent_level: int, line_number: int, blank_before: int, 
               previous_logical: str, previous_unindented_logical_line: str, previous_indent_level: int, 
               lines: List[str]) -> Iterator[Tuple[int, str]]:
    """Separate top-level function and class definitions with two blank lines."""
    top_level_lines = BLANK_LINES_CONFIG['top_level']
    method_lines = BLANK_LINES_CONFIG['method']
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
                ancestor_level = indent_level
                nested = False
                for line in lines[line_number - top_level_lines::-1]:
                    if line.strip() and expand_indent(line) < ancestor_level:
                        ancestor_level = expand_indent(line)
                        nested = STARTSWITH_DEF_REGEX.match(line.lstrip())
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
def extraneous_whitespace(logical_line: str) -> Iterator[Tuple[int, str]]:
    """Avoid extraneous whitespace."""
    line = logical_line
    for match in EXTRANEOUS_WHITESPACE_REGEX.finditer(line):
        text = match.group()
        char = text.strip()
        found = match.start()
        if text[-1].isspace():
            yield (found + 1, "E201 whitespace after '%s'" % char)
        elif line[found - 1] != ',':
            code = 'E202' if char in '}])' else 'E203'
            yield (found, f"{code} whitespace before '{char}'")

@register_check
def whitespace_around_keywords(logical_line: str) -> Iterator[Tuple[int, str]]:
    """Avoid extraneous whitespace around keywords."""
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
def missing_whitespace_after_import_keyword(logical_line: str) -> Iterator[Tuple[int, str]]:
    """Multiple imports in form from x import (a, b, c) should have space."""
    line = logical_line
    indicator = ' import('
    if line.startswith('from '):
        found = line.find(indicator)
        if -1 < found:
            pos = found + len(indicator) - 1
            yield (pos, 'E275 missing whitespace after keyword')

@register_check
def missing_whitespace(logical_line: str) -> Iterator[Tuple[int, str]]:
    """Each comma, semicolon or colon should be followed by whitespace."""
    line = logical_line
    for index in range(len(line) - 1):
        char = line[index]
        next_char = line[index + 1]
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
def indentation