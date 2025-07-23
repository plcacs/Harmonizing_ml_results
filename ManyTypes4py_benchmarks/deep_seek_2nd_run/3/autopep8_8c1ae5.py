"""Automatically formats Python code to conform to the PEP 8 style guide."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import codecs
import collections
import copy
import difflib
import fnmatch
import inspect
import io
import keyword
import locale
import os
import re
import signal
import sys
import textwrap
import token
import tokenize
import pycodestyle
from typing import (
    Any, Callable, Dict, Generator, IO, Iterable, List, Optional, Pattern, Set,
    Tuple, Union, NamedTuple, Text, TypeVar, cast
)

try:
    unicode
except NameError:
    unicode = str

__version__ = '1.3.2'

CR = '\r'
LF = '\n'
CRLF = '\r\n'
PYTHON_SHEBANG_REGEX: Pattern[str] = re.compile(r'^#!.*\bpython[23]?\b\s*$')
LAMBDA_REGEX: Pattern[str] = re.compile(r'([\w.]+)\s=\slambda\s*([\(\)\w,\s.]*):')
COMPARE_NEGATIVE_REGEX: Pattern[str] = re.compile(r'\b(not)\s+([^][)(}{]+)\s+(in|is)\s')
COMPARE_NEGATIVE_REGEX_THROUGH: Pattern[str] = re.compile(r'\b(not\s+in)\s')
BARE_EXCEPT_REGEX: Pattern[str] = re.compile(r'except\s*:')
STARTSWITH_DEF_REGEX: Pattern[str] = re.compile(r'^(async\s+def|def)\s.*\):')
SHORTEN_OPERATOR_GROUPS: Set[frozenset] = frozenset([
    frozenset([',']), frozenset(['%']), frozenset([',', '(', '[', '{']),
    frozenset(['%', '(', '[', '{']), frozenset([',', '(', '[', '{', '%', '+', '-', '*', '/', '//']),
    frozenset(['%', '+', '-', '*', '/', '//'])
])
DEFAULT_IGNORE: str = 'E24,W503'
DEFAULT_INDENT_SIZE: int = 4
CODE_TO_2TO3: Dict[str, List[str]] = {
    'E231': ['ws_comma'], 'E721': ['idioms'], 'W601': ['has_key'],
    'W603': ['ne'], 'W604': ['repr'], 'W690': [
        'apply', 'except', 'exitfunc', 'numliterals', 'operator', 'paren',
        'reduce', 'renames', 'standarderror', 'sys_exc', 'throw',
        'tuple_params', 'xreadlines'
    ]
}

if sys.platform == 'win32':
    DEFAULT_CONFIG: str = os.path.expanduser(r'~\.pep8')
else:
    DEFAULT_CONFIG: str = os.path.join(
        os.getenv('XDG_CONFIG_HOME') or os.path.expanduser('~/.config'), 'pep8'
    )

PROJECT_CONFIG: Tuple[str, str, str] = ('setup.cfg', 'tox.ini', '.pep8')
MAX_PYTHON_FILE_DETECTION_BYTES: int = 1024

T = TypeVar('T')
TokenInfo = Tuple[int, str, Tuple[int, int], Tuple[int, int], str]
Token = NamedTuple('Token', [
    ('token_type', int),
    ('token_string', str),
    ('spos', Tuple[int, int]),
    ('epos', Tuple[int, int]),
    ('line', str)
])

def open_with_encoding(
    filename: str,
    encoding: Optional[str] = None,
    mode: str = 'r',
    limit_byte_check: int = -1
) -> IO[Any]:
    """Return opened file with a specific encoding."""
    if not encoding:
        encoding = detect_encoding(filename, limit_byte_check=limit_byte_check)
    return io.open(filename, mode=mode, encoding=encoding, newline='')

def detect_encoding(filename: str, limit_byte_check: int = -1) -> str:
    """Return file encoding."""
    try:
        with open(filename, 'rb') as input_file:
            from lib2to3.pgen2 import tokenize as lib2to3_tokenize
            encoding = lib2to3_tokenize.detect_encoding(input_file.readline)[0]
        with open_with_encoding(filename, encoding) as test_file:
            test_file.read(limit_byte_check)
        return encoding
    except (LookupError, SyntaxError, UnicodeDecodeError):
        return 'latin-1'

def readlines_from_file(filename: str) -> List[str]:
    """Return contents of file."""
    with open_with_encoding(filename) as input_file:
        return input_file.readlines()

def extended_blank_lines(
    logical_line: str,
    blank_lines: int,
    blank_before: int,
    indent_level: int,
    previous_logical: str
) -> Generator[Tuple[int, str], None, None]:
    """Check for missing blank lines after class declaration."""
    if previous_logical.startswith('def '):
        if blank_lines and pycodestyle.DOCSTRING_REGEX.match(logical_line):
            yield (0, 'E303 too many blank lines ({0})'.format(blank_lines))
    elif pycodestyle.DOCSTRING_REGEX.match(previous_logical):
        if indent_level and (not blank_lines) and (not blank_before) and logical_line.startswith('def ') and ('(self' in logical_line):
            yield (0, 'E301 expected 1 blank line, found 0')

pycodestyle.register_check(extended_blank_lines)

def continued_indentation(
    logical_line: str,
    tokens: List[TokenInfo],
    indent_level: int,
    indent_char: str,
    noqa: bool
) -> Generator[Tuple[Tuple[int, int], str], None, None]:
    """Override pycodestyle's function to provide indentation information."""
    first_row = tokens[0][2][0]
    nrows = 1 + tokens[-1][2][0] - first_row
    if noqa or nrows == 1:
        return
    indent_next = logical_line.endswith(':')
    row = depth = 0
    valid_hangs = (DEFAULT_INDENT_SIZE,) if indent_char != '\t' else (DEFAULT_INDENT_SIZE, 2 * DEFAULT_INDENT_SIZE)
    parens = [0] * nrows
    rel_indent = [0] * nrows
    open_rows = [[0]]
    hangs = [None]
    indent_chances = {}
    last_indent = tokens[0][2]
    indent = [last_indent[1]]
    last_token_multiline = None
    line = None
    last_line = ''
    last_line_begins_with_multiline = False
    for token_type, text, start, end, line in tokens:
        newline = row < start[0] - first_row
        if newline:
            row = start[0] - first_row
            newline = not last_token_multiline and token_type not in (tokenize.NL, tokenize.NEWLINE)
            last_line_begins_with_multiline = last_token_multiline
        if newline:
            last_indent = start
            rel_indent[row] = pycodestyle.expand_indent(line) - indent_level
            close_bracket = token_type == tokenize.OP and text in ']})'
            for open_row in reversed(open_rows[depth]):
                hang = rel_indent[row] - rel_indent[open_row]
                hanging_indent = hang in valid_hangs
                if hanging_indent:
                    break
            if hangs[depth]:
                hanging_indent = hang == hangs[depth]
            visual_indent = not close_bracket and hang > 0 and indent_chances.get(start[1])
            if close_bracket and indent[depth]:
                if start[1] != indent[depth]:
                    yield (start, 'E124 {0}'.format(indent[depth]))
            elif close_bracket and (not hang):
                pass
            elif indent[depth] and start[1] < indent[depth]:
                yield (start, 'E128 {0}'.format(indent[depth]))
            elif hanging_indent or (indent_next and rel_indent[row] == 2 * DEFAULT_INDENT_SIZE):
                if close_bracket:
                    yield (start, 'E123 {0}'.format(indent_level + rel_indent[open_row]))
                hangs[depth] = hang
            elif visual_indent is True:
                indent[depth] = start[1]
            elif visual_indent in (text, unicode):
                pass
            else:
                one_indented = indent_level + rel_indent[open_row] + DEFAULT_INDENT_SIZE
                if hang <= 0:
                    error = ('E122', one_indented)
                elif indent[depth]:
                    error = ('E127', indent[depth])
                elif not close_bracket and hangs[depth]:
                    error = ('E131', one_indented)
                elif hang > DEFAULT_INDENT_SIZE:
                    error = ('E126', one_indented)
                else:
                    hangs[depth] = hang
                    error = ('E121', one_indented)
                yield (start, '{0} {1}'.format(*error))
        if parens[row] and token_type not in (tokenize.NL, tokenize.COMMENT) and (not indent[depth]):
            indent[depth] = start[1]
            indent_chances[start[1]] = True
        elif token_type in (tokenize.STRING, tokenize.COMMENT) or text in ('u', 'ur', 'b', 'br'):
            indent_chances[start[1]] = unicode
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
            elif text in ')]}' and depth > 0:
                prev_indent = indent.pop() or last_indent[1]
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
            if start[1] not in indent_chances and (not last_line.rstrip().endswith(',')):
                indent_chances[start[1]] = text
        last_token_multiline = start[0] != end[0]
        if last_token_multiline:
            rel_indent[end[0] - first_row] = rel_indent[row]
        last_line = line
    if indent_next and (not last_line_begins_with_multiline) and (pycodestyle.expand_indent(line) == indent_level + DEFAULT_INDENT_SIZE):
        pos = (start[0], indent[0] + 4)
        desired_indent = indent_level + 2 * DEFAULT_INDENT_SIZE
        if visual_indent:
            yield (pos, 'E129 {0}'.format(desired_indent))
        else:
            yield (pos, 'E125 {0}'.format(desired_indent))

del pycodestyle._checks['logical_line'][pycodestyle.continued_indentation]
pycodestyle.register_check(continued_indentation)

class FixPEP8(object):
    """Fix invalid code."""
    def __init__(
        self,
        filename: Optional[str],
        options: Any,
        contents: Optional[str] = None,
        long_line_ignore_cache: Optional[Set[Tuple[str, str, str]] = None
    ) -> None:
        self.filename = filename
        if contents is None:
            self.source = readlines_from_file(filename) if filename else []
        else:
            sio = io.StringIO(contents)
            self.source = sio.readlines()
        self.options = options
        self.indent_word = _get_indentword(''.join(self.source))
        self.long_line_ignore_cache = set() if long_line_ignore_cache is None else long_line_ignore_cache
        self.fix_e115 = self.fix_e112
        self.fix_e116 = self.fix_e113
        self.fix_e121 = self._fix_reindent
        self.fix_e122 = self._fix_reindent
        self.fix_e123 = self._fix_reindent
        self.fix_e124 = self._fix_reindent
        self.fix_e126 = self._fix_reindent
        self.fix_e127 = self._fix_reindent
        self.fix_e128 = self._fix_reindent
        self.fix_e129 = self._fix_reindent
        self.fix_e202 = self.fix_e201
        self.fix_e203 = self.fix_e201
        self.fix_e211 = self.fix_e201
        self.fix_e221 = self.fix_e271
        self.fix_e222 = self.fix_e271
        self.fix_e223 = self.fix_e271
        self.fix_e226 = self.fix_e225
        self.fix_e227 = self.fix_e225
        self.fix_e228 = self.fix_e225
        self.fix_e241 = self.fix_e271
        self.fix_e242 = self.fix_e224
        self.fix_e261 = self.fix_e262
        self.fix_e272 = self.fix_e271
        self.fix_e273 = self.fix_e271
        self.fix_e274 = self.fix_e271
        self.fix_e306 = self.fix_e301
        self.fix_e501 = self.fix_long_line_logically if options and (options.aggressive >= 2 or options.experimental) else self.fix_long_line_physically
        self.fix_e703 = self.fix_e702
        self.fix_w293 = self.fix_w291

    def _fix_source(self, results: Iterable[Dict[str, Any]]) -> None:
        try:
            logical_start, logical_end = _find_logical(self.source)
            logical_support = True
        except (SyntaxError, tokenize.TokenError):
            logical_support = False
        completed_lines = set()
        for result in sorted(results, key=_priority_key):
            if result['line'] in completed_lines:
                continue
            fixed_methodname = 'fix_' + result['id'].lower()
            if hasattr(self, fixed_methodname):
                fix = getattr(self, fixed_methodname)
                line_index = result['line'] - 1
                original_line = self.source[line_index]
                is_logical_fix = len(_get_parameters(fix)) > 2
                if is_logical_fix:
                    logical = None
                    if logical_support:
                        logical = _get_logical(self.source, result, logical_start, logical_end)
                        if logical and set(range(logical[0][0] + 1, logical[1][0] + 1)).intersection(completed_lines):
                            continue
                    modified_lines = fix(result, logical)
                else:
                    modified_lines = fix(result)
                if modified_lines is None:
                    assert not is_logical_fix
                    if self.source[line_index] == original_line:
                        modified_lines = []
                if modified_lines:
                    completed_lines.update(modified_lines)
                elif modified_lines == []:
                    if self.options.verbose >= 2:
                        print('--->  Not fixing {error} on line {line}'.format(error=result['id'], line=result['line']), file=sys.stderr)
                else:
                    completed_lines.add(result['line'])
            elif self.options.verbose >= 3:
                print("--->  '{0}' is not defined.".format(fixed_methodname), file=sys.stderr)
                info = result['info'].strip()
                print('--->  {0}:{1}:{2}:{3}'.format(self.filename, result['line'], result['column'], info), file=sys.stderr)

    def fix(self) -> str:
        """Return a version of the source code with PEP 8 violations fixed."""
        pep8_options = {'ignore': self.options.ignore, 'select': self.options.select, 'max_line_length': self.options.max_line_length}
        results = _execute_pep8(pep8_options, self.source)
        if self.options.verbose:
            progress = {}
            for r in results:
                if r['id'] not in progress:
                    progress[r['id']] = set()
                progress[r['id']].add(r['line'])
            print('--->  {n} issue(s) to fix {progress}'.format(n=len(results), progress=progress), file=sys.stderr)
        if self.options.line_range:
            start, end = self.options.line_range
            results = [r for r in results if start <= r['line'] <= end]
        self._fix_source(filter_results(source=''.join(self.source), results=results, aggressive=self.options.aggressive))
        if self.options.line_range:
            count = sum((sline.count('\n') for sline in self.source[start - 1:end]))
            self.options.line_range[1] = start + count - 1
        return ''.join(self.source)

    def _fix_reindent(self, result: Dict[str, Any]) -> None:
        """Fix a badly indented line."""
        num_indent_spaces = int(result['info'].split()[1])
        line_index = result['line'] - 1
        target = self.source[line