#!/usr/bin/env python
# Copyright (C) 2010-2011 Hideo Hattori
# Copyright (C) 2011-2013 Hideo Hattori, Steven Myint
# Copyright (C) 2013-2016 Hideo Hattori, Steven Myint, Bill Wendling
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
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

"""Automatically formats Python code to conform to the PEP 8 style guide.

Fixes that only need be done once can be added by adding a function of the form
"fix_<code>(source)" to this module. They should return the fixed source code.
These fixes are picked up by apply_global_fixes().

Fixes that depend on pycodestyle should be added as methods to FixPEP8. See the
class documentation for more information.

"""

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

from typing import Callable, Dict, List, Optional, Set, Tuple, Union, Any, Iterator

try:
    unicode
except NameError:
    unicode = str

__version__: str = '1.3.2'

CR: str = '\r'
LF: str = '\n'
CRLF: str = '\r\n'

PYTHON_SHEBANG_REGEX: re.Pattern = re.compile(r'^#!.*\bpython[23]?\b\s*$')
LAMBDA_REGEX: re.Pattern = re.compile(r'([\w.]+)\s=\slambda\s*([\(\)\w,\s.]*):')
COMPARE_NEGATIVE_REGEX: re.Pattern = re.compile(r'\b(not)\s+([^][)(}{]+)\s+(in|is)\s')
COMPARE_NEGATIVE_REGEX_THROUGH: re.Pattern = re.compile(r'\b(not\s+in)\s')
BARE_EXCEPT_REGEX: re.Pattern = re.compile(r'except\s*:')
STARTSWITH_DEF_REGEX: re.Pattern = re.compile(r'^(async\s+def|def)\s.*\):')

# For generating line shortening candidates.
SHORTEN_OPERATOR_GROUPS: Set[FrozenSet[str]] = frozenset([
    frozenset([',']),
    frozenset(['%']),
    frozenset([',', '(', '[', '{']),
    frozenset(['%', '(', '[', '{']),
    frozenset([',', '(', '[', '{', '%', '+', '-', '*', '/', '//']),
    frozenset(['%', '+', '-', '*', '/', '//']),
])

DEFAULT_IGNORE: str = 'E24,W503'
DEFAULT_INDENT_SIZE: int = 4

# W602 is handled separately due to the need to avoid "with_traceback".
CODE_TO_2TO3: Dict[str, List[str]] = {
    'E231': ['ws_comma'],
    'E721': ['idioms'],
    'W601': ['has_key'],
    'W603': ['ne'],
    'W604': ['repr'],
    'W690': ['apply',
             'except',
             'exitfunc',
             'numliterals',
             'operator',
             'paren',
             'reduce',
             'renames',
             'standarderror',
             'sys_exc',
             'throw',
             'tuple_params',
             'xreadlines']
}

if sys.platform == 'win32':  # pragma: no cover
    DEFAULT_CONFIG: str = os.path.expanduser(r'~\.pep8')
else:
    DEFAULT_CONFIG: str = os.path.join(os.getenv('XDG_CONFIG_HOME') or
                                      os.path.expanduser('~/.config'), 'pep8')
PROJECT_CONFIG: Tuple[str, ...] = ('setup.cfg', 'tox.ini', '.pep8')

MAX_PYTHON_FILE_DETECTION_BYTES: int = 1024

def open_with_encoding(filename: str,
                       encoding: Optional[str] = None, mode: str = 'r', limit_byte_check: int = -1) -> io.TextIOBase:
    """Return opened file with a specific encoding."""
    if not encoding:
        encoding = detect_encoding(filename, limit_byte_check=limit_byte_check)

    return io.open(filename, mode=mode, encoding=encoding,
                   newline='')  # Preserve line endings

def detect_encoding(filename: str, limit_byte_check: int = -1) -> str:
    """Return file encoding."""
    try:
        with open(filename, 'rb') as input_file:
            from lib2to3.pgen2 import tokenize as lib2to3_tokenize
            encoding: str = lib2to3_tokenize.detect_encoding(input_file.readline)[0]

        with open_with_encoding(filename, encoding) as test_file:
            test_file.read(limit_byte_check)

        return encoding
    except (LookupError, SyntaxError, UnicodeDecodeError):
        return 'latin-1'

def readlines_from_file(filename: str) -> List[str]:
    """Return contents of file."""
    with open_with_encoding(filename) as input_file:
        return input_file.readlines()

def extended_blank_lines(logical_line: str,
                         blank_lines: int,
                         blank_before: bool,
                         indent_level: int,
                         previous_logical: str) -> Iterator[Tuple[int, str]]:
    """Check for missing blank lines after class declaration."""
    if previous_logical.startswith('def '):
        if blank_lines and pycodestyle.DOCSTRING_REGEX.match(logical_line):
            yield (0, f'E303 too many blank lines ({blank_lines})')
    elif pycodestyle.DOCSTRING_REGEX.match(previous_logical):
        # Missing blank line between class docstring and method declaration.
        if (
            indent_level and
            not blank_lines and
            not blank_before and
            logical_line.startswith(('def ')) and
            '(self' in logical_line
        ):
            yield (0, 'E301 expected 1 blank line, found 0')

pycodestyle.register_check(extended_blank_lines)

def continued_indentation(logical_line: str, tokens: List[tokenize.TokenInfo], indent_level: int, indent_char: str,
                          noqa: bool) -> Iterator[Tuple[int, str]]:
    """Override pycodestyle's function to provide indentation information."""
    first_row: int = tokens[0][2][0]
    nrows: int = 1 + tokens[-1][2][0] - first_row
    if noqa or nrows == 1:
        return

    # indent_next tells us whether the next block is indented. Assuming
    # that it is indented by 4 spaces, then we should not allow 4-space
    # indents on the final continuation line. In turn, some other
    # indents are allowed to have an extra 4 spaces.
    indent_next: bool = logical_line.endswith(':')

    row: int = 0
    depth: int = 0
    if noqa or nrows == 1:
        return

    # Remember how many brackets were opened on each line.
    parens: List[int] = [0] * nrows

    # Relative indents of physical lines.
    rel_indent: List[int] = [0] * nrows

    # For each depth, collect a list of opening rows.
    open_rows: List[List[int]] = [[0]]
    # For each depth, memorize the hanging indentation.
    hangs: List[Optional[int]] = [None]

    # Visual indents.
    indent_chances: Dict[int, Any] = {}
    last_indent: Tuple[int, int] = tokens[0][2]
    indent: List[int] = [last_indent[1]]

    last_token_multiline: Optional[bool] = None
    line: Optional[str] = None
    last_line: str = ''
    last_line_begins_with_multiline: bool = False
    for token_type, text, start, end, line in tokens:

        newline: bool = row < start[0] - first_row
        if newline:
            row = start[0] - first_row
            newline = (not last_token_multiline and
                       token_type not in (tokenize.NL, tokenize.NEWLINE))
            last_line_begins_with_multiline = last_token_multiline

        if newline:
            # This is the beginning of a continuation line.
            last_indent = start

            # Record the initial indent.
            rel_indent[row] = pycodestyle.expand_indent(line) - indent_level

            # Identify closing bracket.
            close_bracket: bool = (token_type == tokenize.OP and text in ']})')

            # Is the indent relative to an opening bracket line?
            for open_row in reversed(open_rows[depth]):
                hang: int = rel_indent[row] - rel_indent[open_row]
                hanging_indent: bool = (hang in pycodestyle.valid_hangs)
                if hanging_indent:
                    break
            if hangs[depth]:
                hanging_indent = (hang == hangs[depth])

            visual_indent: bool = (not close_bracket and hang > 0 and
                                   indent_chances.get(start[1]))

            if close_bracket and indent[depth]:
                # Closing bracket for visual indent.
                if start[1] != indent[depth]:
                    yield (start, f'E124 {indent[depth]}')
            elif close_bracket and not hang:
                pass
            elif indent[depth] and start[1] < indent[depth]:
                # Visual indent is broken.
                yield (start, f'E128 {indent[depth]}')
            elif (hanging_indent or
                  (indent_next and
                   rel_indent[row] == 2 * DEFAULT_INDENT_SIZE)):
                # Hanging indent is verified.
                if close_bracket:
                    yield (start, f'E123 {indent_level + rel_indent[open_row]}')
                hangs[depth] = hang
            elif visual_indent:
                # Visual indent is verified.
                if close_bracket:
                    yield (start, f'E123 {indent_level + rel_indent[open_row]}')
                hangs[depth] = hang
            elif visual_indent is True:
                # Visual indent is verified.
                indent[depth] = start[1]
            elif visual_indent in (text, unicode):
                # Ignore token lined up with matching one from a previous line.
                pass
            else:
                one_indented: int = (indent_level + rel_indent[open_row] +
                                      DEFAULT_INDENT_SIZE)
                # Indent is broken.
                if hang <= 0:
                    error: Tuple[str, int] = ('E122', one_indented)
                elif indent[depth]:
                    error = ('E127', indent[depth])
                elif not close_bracket and hangs[depth]:
                    error = ('E131', one_indented)
                elif hang > DEFAULT_INDENT_SIZE:
                    error = ('E126', one_indented)
                else:
                    hangs[depth] = hang
                    error = ('E121', one_indented)

                yield (start, f'{error[0]} {error[1]}')

        # Look for visual indenting.
        if (
            parens[row] and
            token_type not in (tokenize.NL, tokenize.COMMENT) and
            not indent[depth]
        ):
            indent[depth] = start[1]
            indent_chances[start[1]] = True
        # Deal with implicit string concatenation.
        elif (token_type in (tokenize.STRING, tokenize.COMMENT) or
              text in ('u', 'ur', 'b', 'br')):
            indent_chances[start[1]] = unicode
        # Special case for the "if" statement because len("if (") is equal to
        # 4.
        elif not indent_chances and not row and not depth and text == 'if':
            indent_chances[end[1] + 1] = True
        elif text == ':' and line[end[1]:].isspace():
            open_rows[depth].append(row)

        # Keep track of bracket depth.
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
                # Parent indents should not be more than this one.
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
            if (
                start[1] not in indent_chances and
                # This is for purposes of speeding up E121 (GitHub #90).
                not last_line.rstrip().endswith(',')
            ):
                # Allow to line up tokens.
                indent_chances[start[1]] = text

        last_token_multiline = (start[0] != end[0])
        if last_token_multiline:
            rel_indent[end[0] - first_row] = rel_indent[row]

        last_line = line

    if (
        indent_next and
        not last_line_begins_with_multiline and
        pycodestyle.expand_indent(line) == indent_level + DEFAULT_INDENT_SIZE
    ):
        pos: Tuple[int, int] = (start[0], indent[0] + 4)
        desired_indent: int = indent_level + 2 * DEFAULT_INDENT_SIZE
        if visual_indent:
            yield (pos, f'E129 {desired_indent}')
        else:
            yield (pos, f'E125 {desired_indent}')

del pycodestyle._checks['logical_line'][pycodestyle.continued_indentation]
pycodestyle.register_check(continued_indentation)

class FixPEP8(object):

    """Fix invalid code.

    Fixer methods are prefixed "fix_". The _fix_source() method looks for these
    automatically.

    The fixer method can take either one or two arguments (in addition to
    self). The first argument is "result", which is the error information from
    pycodestyle. The second argument, "logical", is required only for
    logical-line fixes.

    The fixer method can return the list of modified lines or None. An empty
    list would mean that no changes were made. None would mean that only the
    line reported in the pycodestyle error was modified. Note that the modified
    line numbers that are returned are indexed at 1. This typically would
    correspond with the line number reported in the pycodestyle error
    information.

    [fixed method list]
        - e111,e114,e115,e116
        - e121,e122,e123,e124,e125,e126,e127,e128,e129
        - e201,e202,e203
        - e211
        - e221,e222,e223,e224,e225
        - e231
        - e251
        - e261,e262
        - e271,e272,e273,e274
        - e301,e302,e303,e304,e306
        - e401
        - e502
        - e701,e702,e703,e704
        - e711,e712,e713,e714
        - e722
        - e731
        - w291
        - w503

    """

    def __init__(self, filename: Optional[str],
                 options: Any,
                 contents: Optional[str] = None,
                 long_line_ignore_cache: Optional[Set[Tuple[str, Tuple[str, str, str, str]]]] = None):
        self.filename: Optional[str] = filename
        if contents is None:
            self.source: List[str] = readlines_from_file(filename)
        else:
            sio: io.StringIO = io.StringIO(contents)
            self.source: List[str] = sio.readlines()
        self.options: Any = options
        self.indent_word: str = _get_indentword(''.join(self.source))

        self.long_line_ignore_cache: Set[Tuple[str, Tuple[str, str, str, str]]] = (
            set() if long_line_ignore_cache is None
            else long_line_ignore_cache
        )

        # Many fixers are the same even though pycodestyle categorizes them
        # differently.
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
        self.fix_e501 = (
            self.fix_long_line_logically if
            options and (options.aggressive >= 2 or options.experimental) else
            self.fix_long_line_physically
        )
        self.fix_e703 = self.fix_e702
        self.fix_w293 = self.fix_w291

    def _fix_source(self, results: Iterator[Dict[str, Any]]) -> None:
        try:
            (logical_start, logical_end) = _find_logical(self.source)
            logical_support: bool = True
        except (SyntaxError, tokenize.TokenError):  # pragma: no cover
            logical_support = False

        completed_lines: Set[int] = set()
        for result in sorted(results, key=_priority_key):
            if result['line'] in completed_lines:
                continue

            fixed_methodname: str = 'fix_' + result['id'].lower()
            if hasattr(self, fixed_methodname):
                fix: Callable[..., Optional[List[int]]] = getattr(self, fixed_methodname)

                line_index: int = result['line'] - 1
                original_line: str = self.source[line_index]

                is_logical_fix: bool = len(inspect.signature(fix).parameters) > 2
                if is_logical_fix:
                    logical: Optional[Tuple[Tuple[int, int], Tuple[int, int], List[str]]] = None
                    if logical_support:
                        logical = _get_logical(self.source,
                                               result,
                                               logical_start,
                                               logical_end)
                        if logical and set(range(
                            logical[0][0] + 1,
                            logical[1][0] + 1
                        )).intersection(
                            completed_lines
                        ):
                            continue

                    modified_lines: Optional[List[int]] = fix(result, logical)
                else:
                    modified_lines: Optional[List[int]] = fix(result)

                if modified_lines is None:
                    # Force logical fixes to report what they modified.
                    assert not is_logical_fix

                    if self.source[line_index] == original_line:
                        modified_lines = []

                if modified_lines:
                    completed_lines.update(modified_lines)
                elif modified_lines == []:  # Empty list means no fix
                    if self.options.verbose >= 2:
                        print(
                            f'--->  Not fixing {result["id"]} on line {result["line"]}',
                            file=sys.stderr
                        )
                else:  # We assume one-line fix when None.
                    completed_lines.add(result['line'])
            else:
                if self.options.verbose >= 3:
                    print(
                        f"--->  '{fixed_methodname}' is not defined.",
                        file=sys.stderr
                    )

                    info: str = result['info'].strip()
                    print(f'--->  {self.filename}:{result["line"]}:{result["column"]}:{info}',
                          file=sys.stderr)

    def fix(self) -> str:
        """Return a version of the source code with PEP 8 violations fixed."""
        pep8_options: Dict[str, Any] = {
            'ignore': self.options.ignore,
            'select': self.options.select,
            'max_line_length': self.options.max_line_length,
        }
        results: List[Dict[str, Any]] = _execute_pep8(pep8_options, self.source)

        if self.options.verbose:
            progress: Dict[str, Set[int]] = {}
            for r in results:
                if r['id'] not in progress:
                    progress[r['id']] = set()
                progress[r['id']].add(r['line'])
            print(f'--->  {len(results)} issue(s) to fix {progress}', file=sys.stderr)

        if self.options.line_range:
            start, end = self.options.line_range
            results = [r for r in results
                       if start <= r['line'] <= end]

        self._fix_source(filter_results(source=''.join(self.source),
                                        results=results,
                                        aggressive=self.options.aggressive))

        if self.options.line_range:
            # If number of lines has changed then change line_range.
            count: int = sum(sline.count('\n')
                             for sline in self.source[start - 1:end])
            self.options.line_range[1] = start + count - 1

        return ''.join(self.source)

    def _fix_reindent(self, result: Dict[str, Any]) -> None:
        """Fix a badly indented line.

        This is done by adding or removing from its initial indent only.

        """
        num_indent_spaces: int = int(result['info'].split()[1])
        line_index: int = result['line'] - 1
        target: str = self.source[line_index]

        self.source[line_index] = ' ' * num_indent_spaces + target.lstrip()

    def fix_e112(self, result: Dict[str, Any]) -> List[int]:
        """Fix under-indented comments."""
        line_index: int = result['line'] - 1
        target: str = self.source[line_index]

        if not target.lstrip().startswith('#'):
            # Don't screw with invalid syntax.
            return []

        self.source[line_index] = self.indent_word + target
        return []

    def fix_e113(self, result: Dict[str, Any]) -> List[int]:
        """Fix over-indented comments."""
        line_index: int = result['line'] - 1
        target: str = self.source[line_index]

        indent: str = _get_indentation(target)
        stripped: str = target.lstrip()

        if not stripped.startswith('#'):
            # Don't screw with invalid syntax.
            return []

        self.source[line_index] = indent[1:] + stripped
        return []

    def fix_e125(self, result: Dict[str, Any]) -> List[int]:
        """Fix indentation undistinguish from the next logical line."""
        num_indent_spaces: int = int(result['info'].split()[1])
        line_index: int = result['line'] - 1
        target: str = self.source[line_index]

        spaces_to_add: int = num_indent_spaces - len(_get_indentation(target))
        indent: int = len(_get_indentation(target))
        modified_lines: List[int] = []

        while len(_get_indentation(self.source[line_index])) >= indent:
            self.source[line_index] = (' ' * spaces_to_add +
                                       self.source[line_index])
            modified_lines.append(1 + line_index)  # Line indexed at 1.
            line_index -= 1

        return modified_lines

    def fix_e131(self, result: Dict[str, Any]) -> None:
        """Fix indentation undistinguish from the next logical line."""
        num_indent_spaces: int = int(result['info'].split()[1])
        line_index: int = result['line'] - 1
        target: str = self.source[line_index]

        spaces_to_add: int = num_indent_spaces - len(_get_indentation(target))

        if spaces_to_add >= 0:
            self.source[line_index] = (' ' * spaces_to_add +
                                       self.source[line_index])
        else:
            offset: int = abs(spaces_to_add)
            self.source[line_index] = self.source[line_index][offset:]

    def fix_e201(self, result: Dict[str, Any]) -> List[int]:
        """Remove extraneous whitespace."""
        line_index: int = result['line'] - 1
        target: str = self.source[line_index]
        offset: int = result['column'] - 1

        fixed: str = fix_whitespace(target,
                                     offset=offset,
                                     replacement='')

        self.source[line_index] = fixed
        return [line_index + 1]

    def fix_e224(self, result: Dict[str, Any]) -> List[int]:
        """Remove extraneous whitespace around operator."""
        target: str = self.source[result['line'] - 1]
        offset: int = result['column'] - 1
        fixed: str = target[:offset] + target[offset:].replace('\t', ' ')
        self.source[result['line'] - 1] = fixed
        return [result['line']]

    def fix_e225(self, result: Dict[str, Any]) -> List[int]:
        """Fix missing whitespace around operator."""
        target: str = self.source[result['line'] - 1]
        offset: int = result['column'] - 1
        fixed: str = target[:offset] + ' ' + target[offset:]

        # Only proceed if non-whitespace characters match.
        # And make sure we don't break the indentation.
        if (
            fixed.replace(' ', '') == target.replace(' ', '') and
            _get_indentation(fixed) == _get_indentation(target)
        ):
            self.source[result['line'] - 1] = fixed
            error_code: str = result.get('id', 'E225')
            try:
                ts: List[tokenize.TokenInfo] = generate_tokens(fixed)
            except tokenize.TokenError:
                return []
            if not pycodestyle.check_syntax(fixed.lstrip()):
                return []
            errors: List[Tuple[tokenize.TokenInfo, str]] = list(
                pycodestyle.missing_whitespace_around_operator(fixed, ts))
            for e in reversed(errors):
                if error_code != e[1].split()[0]:
                    continue
                offset_new: int = e[0][1]
                fixed = fixed[:offset_new] + ' ' + fixed[offset_new:]
            self.source[result['line'] - 1] = fixed
        else:
            return []

    def fix_e231(self, result: Dict[str, Any]) -> List[int]:
        """Add missing whitespace."""
        line_index: int = result['line'] - 1
        target: str = self.source[line_index]
        offset: int = result['column']

        fixed: str = target[:offset].rstrip() + ' ' + target[offset:].lstrip()
        self.source[line_index] = fixed
        return [line_index + 1]

    def fix_e251(self, result: Dict[str, Any]) -> List[int]:
        """Remove whitespace around parameter '=' sign."""
        line_index: int = result['line'] - 1
        target: str = self.source[line_index]

        # This is necessary since pycodestyle sometimes reports columns that
        # goes past the end of the physical line. This happens in cases like,
        # foo(bar\n=None)
        c: int = min(result['column'] - 1,
                    len(target) - 1)

        if target[c].strip():
            fixed: str = target
        else:
            fixed = target[:c].rstrip() + target[c:].lstrip()

        # There could be an escaped newline
        #
        #     def foo(a=\
        #             1)
        if fixed.endswith(('=\\\n', '=\\\r\n', '=\\\r')):
            self.source[line_index] = fixed.rstrip('\n\r \t\\')
            self.source[line_index + 1] = self.source[line_index + 1].lstrip()
            return [line_index + 1, line_index + 2]  # Line indexed at 1

        self.source[result['line'] - 1] = fixed
        return [line_index + 1]

    def fix_e262(self, result: Dict[str, Any]) -> List[int]:
        """Fix spacing after comment hash."""
        target: str = self.source[result['line'] - 1]
        offset: int = result['column']

        code: str = target[:offset].rstrip(' \t#')
        comment: str = target[offset:].lstrip(' \t#')

        fixed: str = code + ('  # ' + comment if comment.strip() else '\n')

        self.source[result['line'] - 1] = fixed
        return [result['line']]

    def fix_e271(self, result: Dict[str, Any]) -> List[int]:
        """Fix extraneous whitespace around keywords."""
        line_index: int = result['line'] - 1
        target: str = self.source[line_index]
        offset: int = result['column'] - 1

        fixed: str = fix_whitespace(target,
                                     offset=offset,
                                     replacement=' ')

        if fixed == target:
            return []
        else:
            self.source[line_index] = fixed
            return [line_index + 1]

    def fix_e301(self, result: Dict[str, Any]) -> List[int]:
        """Add missing blank line."""
        cr: str = '\n'
        self.source[result['line'] - 1] = cr + self.source[result['line'] - 1]
        return [result['line']]

    def fix_e302(self, result: Dict[str, Any]) -> List[int]:
        """Add missing 2 blank lines."""
        add_linenum: int = 2 - int(result['info'].split()[-1])
        cr: str = '\n' * add_linenum
        self.source[result['line'] - 1] = cr + self.source[result['line'] - 1]
        return [result['line']]

    def fix_e303(self, result: Dict[str, Any]) -> List[int]:
        """Remove extra blank lines."""
        delete_linenum: int = int(result['info'].split('(')[1].split(')')[0]) - 2
        delete_linenum = max(1, delete_linenum)

        # We need to count because pycodestyle reports an offset line number if
        # there are comments.
        cnt: int = 0
        line: int = result['line'] - 2
        modified_lines: List[int] = []
        while cnt < delete_linenum and line >= 0:
            if not self.source[line].strip():
                self.source[line] = ''
                modified_lines.append(1 + line)  # Line indexed at 1
                cnt += 1
            line -= 1

        return modified_lines

    def fix_e304(self, result: Dict[str, Any]) -> List[int]:
        """Remove blank line following function decorator."""
        line: int = result['line'] - 2
        if not self.source[line].strip():
            self.source[line] = ''
            return [line + 1]
        return []

    def fix_e305(self, result: Dict[str, Any]) -> List[int]:
        """Add missing 2 blank lines after end of function or class."""
        cr: str = '\n'
        # check comment line
        offset: int = result['line'] - 2
        while True:
            if offset < 0:
                break
            line: str = self.source[offset].lstrip()
            if not line:
                break
            if line[0] != '#':
                break
            offset -= 1
        offset += 1
        self.source[offset] = cr + self.source[offset]
        return [offset + 1]

    def fix_e401(self, result: Dict[str, Any]) -> List[int]:
        """Put imports on separate lines."""
        line_index: int = result['line'] - 1
        target: str = self.source[line_index]
        offset: int = result['column'] - 1

        if not target.lstrip().startswith('import'):
            return []

        indentation: str = re.split(pattern=r'\bimport\b',
                                   string=target, maxsplit=1)[0]
        fixed: str = (target[:offset].rstrip('\t ,') + '\n' +
                     indentation + 'import ' + target[offset:].lstrip('\t ,'))
        self.source[line_index] = fixed
        return [line_index + 1]

    def fix_long_line_logically(self, result: Dict[str, Any], logical: Optional[Tuple[Tuple[int, int], Tuple[int, int], List[str]]]) -> List[int]:
        """Try to make lines fit within --max-line-length characters."""
        if (
            not logical or
            len(logical[2]) == 1 or
            self.source[result['line'] - 1].lstrip().startswith('#')
        ):
            return self.fix_long_line_physically(result)

        start_line_index: int = logical[0][0]
        end_line_index: int = logical[1][0]
        logical_lines: List[str] = logical[2]

        previous_line: str = get_item(self.source, start_line_index - 1, default='')
        next_line: str = get_item(self.source, end_line_index + 1, default='')

        single_line: str = join_logical_line(''.join(logical_lines))

        try:
            fixed: Optional[str] = self.fix_long_line(
                target=single_line,
                previous_line=previous_line,
                next_line=next_line,
                original=''.join(logical_lines)
            )
        except (SyntaxError, tokenize.TokenError):
            return self.fix_long_line_physically(result)

        if fixed:
            for line_index in range(start_line_index, end_line_index + 1):
                self.source[line_index] = ''
            self.source[start_line_index] = fixed
            return list(range(start_line_index + 1, end_line_index + 1))

        return []

    def fix_long_line_physically(self, result: Dict[str, Any]) -> List[int]:
        """Try to make lines fit within --max-line-length characters."""
        line_index: int = result['line'] - 1
        target: str = self.source[line_index]

        previous_line: str = get_item(self.source, line_index - 1, default='')
        next_line: str = get_item(self.source, line_index + 1, default='')

        try:
            fixed: Optional[str] = self.fix_long_line(
                target=target,
                previous_line=previous_line,
                next_line=next_line,
                original=target
            )
        except (SyntaxError, tokenize.TokenError):
            return []

        if fixed:
            self.source[line_index] = fixed
            return [line_index + 1]

        return []

    def fix_long_line(self, target: str, previous_line: str,
                      next_line: str, original: str) -> Optional[str]:
        cache_entry: Tuple[str, str, str] = (target, previous_line, next_line)
        if cache_entry in self.long_line_ignore_cache:
            return None

        if target.lstrip().startswith('#'):
            if self.options.aggressive:
                # Wrap commented lines.
                return shorten_comment(
                    line=target,
                    max_line_length=self.options.max_line_length,
                    last_comment=not next_line.lstrip().startswith('#')
                )
            else:
                return None

        fixed: Optional[str] = get_fixed_long_line(
            target=target,
            previous_line=previous_line,
            original=original,
            indent_word=self.indent_word,
            max_line_length=self.options.max_line_length,
            aggressive=self.options.aggressive,
            experimental=self.options.experimental,
            verbose=self.options.verbose
        )

        if fixed and not code_almost_equal(original, fixed):
            return fixed

        self.long_line_ignore_cache.add(cache_entry)
        return None

    def fix_e502(self, result: Dict[str, Any]) -> List[int]:
        """Remove extraneous escape of newline."""
        (line_index, _, target) = get_index_offset_contents(result,
                                                            self.source)
        self.source[line_index] = target.rstrip('\n\r \t\\') + '\n'
        return [line_index + 1]

    def fix_e701(self, result: Dict[str, Any]) -> List[int]:
        """Put colon-separated compound statement on separate lines."""
        line_index: int = result['line'] - 1
        target: str = self.source[line_index]
        c: int = result['column']

        fixed_source: str = (target[:c] + '\n' +
                            _get_indentation(target) + self.indent_word +
                            target[c:].lstrip('\n\r \t\\'))
        self.source[result['line'] - 1] = fixed_source
        return [result['line'], result['line'] + 1]

    def fix_e702(self, result: Dict[str, Any], logical: Optional[Tuple[Tuple[int, int], Tuple[int, int], List[str]]]) -> List[int]:
        """Put semicolon-separated compound statement on separate lines."""
        if not logical:
            return []  # pragma: no cover
        logical_lines: List[str] = logical[2]

        # Avoid applying this when indented.
        # https://docs.python.org/reference/compound_stmts.html
        for line in logical_lines:
            if ':' in line:
                return []

        line_index: int = result['line'] - 1
        target: str = self.source[line_index]

        if target.rstrip().endswith('\\'):
            # Normalize '1; \\\n2' into '1; 2'.
            self.source[line_index] = target.rstrip('\n \r\t\\')
            self.source[line_index + 1] = self.source[line_index + 1].lstrip()
            return [line_index + 1, line_index + 2]

        if target.rstrip().endswith(';'):
            self.source[line_index] = target.rstrip('\n \r\t;') + '\n'
            return [line_index + 1]

        offset: int = result['column'] - 1
        first: str = target[:offset].rstrip(';').rstrip()
        second: str = (_get_indentation(logical_lines[0]) +
                      target[offset:].lstrip(';').lstrip())

        # Find inline comment.
        inline_comment: Optional[str] = None
        if target[offset:].lstrip(';').lstrip()[:2] == '# ':
            inline_comment = target[offset:].lstrip(';')

        if inline_comment:
            self.source[line_index] = first + inline_comment
        else:
            self.source[line_index] = first + '\n' + second
        return [line_index + 1]

    def fix_e704(self, result: Dict[str, Any]) -> List[int]:
        """Fix multiple statements on one line def"""
        (line_index, _, target) = get_index_offset_contents(result,
                                                            self.source)
        match: Optional[re.Match] = STARTSWITH_DEF_REGEX.match(target)
        if match:
            self.source[line_index] = f'{match.group(0)}\n{_get_indentation(target) + self.indent_word}{target[match.end(0):].lstrip()}'
            return [line_index + 1]
        return []

    def fix_e711(self, result: Dict[str, Any]) -> List[int]:
        """Fix comparison with None."""
        (line_index, offset, target) = get_index_offset_contents(result,
                                                                 self.source)

        right_offset: int = offset + 2
        if right_offset >= len(target):
            return []

        left: str = target[:offset].rstrip()
        center: str = target[offset:right_offset]
        right: str = target[right_offset:].lstrip()

        if not right.startswith('None'):
            return []

        if center.strip() == '==':
            new_center: str = 'is'
        elif center.strip() == '!=':
            new_center: str = 'is not'
        else:
            return []

        self.source[line_index] = ' '.join([left, new_center, right])
        return [line_index + 1]

    def fix_e712(self, result: Dict[str, Any]) -> List[int]:
        """Fix (trivial case of) comparison with boolean."""
        (line_index, offset, target) = get_index_offset_contents(result,
                                                                 self.source)

        # Handle very easy "not" special cases.
        if re.match(r'^\s*if [\w.]+ == False:$', target):
            self.source[line_index] = re.sub(r'if ([\w.]+) == False:',
                                             r'if not \1:', target, count=1)
            return [line_index + 1]
        elif re.match(r'^\s*if [\w.]+ != True:$', target):
            self.source[line_index] = re.sub(r'if ([\w.]+) != True:',
                                             r'if not \1:', target, count=1)
            return [line_index + 1]
        else:
            right_offset: int = offset + 2
            if right_offset >= len(target):
                return []

            left: str = target[:offset].rstrip()
            center: str = target[offset:right_offset]
            right: str = target[right_offset:].lstrip()

            # Handle simple cases only.
            new_right: Optional[str] = None
            if center.strip() == '==':
                if re.match(r'\bTrue\b', right):
                    new_right = re.sub(r'\bTrue\b *', '', right, count=1)
            elif center.strip() == '!=':
                if re.match(r'\bFalse\b', right):
                    new_right = re.sub(r'\bFalse\b *', '', right, count=1)

            if new_right is None:
                return []

            if new_right[0].isalnum():
                new_right = ' ' + new_right

            self.source[line_index] = left + new_right
            return [line_index + 1]

    def fix_e713(self, result: Dict[str, Any]) -> None:
        """Fix (trivial case of) non-membership check."""
        (line_index, offset, target) = get_index_offset_contents(result,
                                                                 self.source)

        # to convert once 'not in' -> 'in'
        before_target: str = target[:offset]
        target = target[offset:]
        match_notin: Optional[re.Match] = COMPARE_NEGATIVE_REGEX_THROUGH.search(target)
        notin_pos_start: int = 0
        notin_pos_end: int = 0
        if match_notin:
            notin_pos_start = match_notin.start(1)
            notin_pos_end = match_notin.end()
            target = f'{target[:notin_pos_start]} in {target[notin_pos_end:]}'

        # fix 'not in'
        match: Optional[re.Match] = COMPARE_NEGATIVE_REGEX.search(target)
        if match:
            if match.group(3) == 'in':
                pos_start: int = match.start(1)
                new_target: str = f'{target[:pos_start]}{match.group(2)} {match.group(1)} {match.group(3)} {target[match.end():]}'
                if match_notin:
                    # revert 'in' -> 'not in'
                    pos_start = notin_pos_start + offset
                    pos_end = notin_pos_end + offset - 4     # len('not ')
                    new_target = f'{new_target[:pos_start]} not in {new_target[pos_end:]}'
                self.source[line_index] = new_target

    def fix_e714(self, result: Dict[str, Any]) -> None:
        """Fix object identity should be 'is not' case."""
        (line_index, _, target) = get_index_offset_contents(result,
                                                            self.source)

        match: Optional[re.Match] = COMPARE_NEGATIVE_REGEX.search(target)
        if match:
            if match.group(3) == 'is':
                pos_start: int = match.start(1)
                self.source[line_index] = f'{target[:pos_start]}{match.group(2)} {match.group(3)} {match.group(1)} {target[match.end():]}'

    def fix_e722(self, result: Dict[str, Any]) -> None:
        """fix bare except"""
        (line_index, _, target) = get_index_offset_contents(result,
                                                            self.source)
        match: Optional[re.Match] = BARE_EXCEPT_REGEX.search(target)
        if match:
            self.source[line_index] = f'{target[:result["column"] - 1]}except BaseException:{target[match.end():]}'
        return [line_index + 1]

    def fix_e731(self, result: Dict[str, Any]) -> None:
        """Fix do not assign a lambda expression check."""
        (line_index, _, target) = get_index_offset_contents(result,
                                                            self.source)
        match: Optional[re.Match] = LAMBDA_REGEX.search(target)
        if match:
            end: int = match.end()
            self.source[line_index] = f'def {match.group(1)}({match.group(2)}): return {target[end:].lstrip()}'
            return [line_index + 1]
        return []

    def fix_w291(self, result: Dict[str, Any]) -> List[int]:
        """Remove trailing whitespace."""
        fixed_line: str = self.source[result['line'] - 1].rstrip()
        self.source[result['line'] - 1] = fixed_line + '\n'
        return [result['line']]

    def fix_w391(self, _: Dict[str, Any]) -> List[int]:
        """Remove trailing blank lines."""
        blank_count: int = 0
        for line in reversed(self.source):
            line = line.rstrip()
            if line:
                break
            else:
                blank_count += 1

        original_length: int = len(self.source)
        self.source = self.source[:original_length - blank_count]
        return list(range(1, 1 + original_length))

    def fix_w503(self, result: Dict[str, Any]) -> List[int]:
        (line_index, _, target) = get_index_offset_contents(result,
                                                            self.source)
        one_string_token: str = target.split()[0]
        try:
            ts: List[tokenize.TokenInfo] = generate_tokens(one_string_token)
        except tokenize.TokenError:
            return []
        if not _is_binary_operator(ts[0][0], one_string_token):
            return []
        # find comment
        comment_index: Optional[int] = None
        for i in range(5):
            # NOTE: try to parse code in 5 times
            if (line_index - i) < 0:
                break
            from_index: int = line_index - i - 1
            to_index: int = line_index + 1
            try:
                ts = generate_tokens("".join(self.source[from_index:to_index]))
            except Exception:
                continue
            newline_count: int = 0
            newline_index: List[int] = []
            for i, t in enumerate(ts):
                if t[0] in (tokenize.NEWLINE, tokenize.NL):
                    newline_index.append(i)
                    newline_count += 1
            if newline_count > 2:
                tts: List[tokenize.TokenInfo] = ts[newline_index[-3]:]
            else:
                tts = ts
            old: Optional[tokenize.TokenInfo] = None
            for t in tts:
                if tokenize.COMMENT == t[0]:
                    if old is None:
                        comment_index = 0
                    else:
                        comment_index = old[3][1]
                    break
                old = t
            break
        i = target.index(one_string_token)
        self.source[line_index] = f'{target[:i]}{target[i + len(one_string_token):]}'
        nl: str = find_newline(self.source[line_index - 1:line_index])
        before_line: str = self.source[line_index - 1]
        bl: int = before_line.index(nl)
        if comment_index:
            self.source[line_index - 1] = f'{before_line[:comment_index]} {one_string_token} {before_line[comment_index + 1:]}'
        else:
            self.source[line_index - 1] = f'{before_line[:bl]} {one_string_token}{before_line[bl:]}'
        return [line_index + 1]

def get_index_offset_contents(result: Dict[str, Any], source: List[str]) -> Tuple[int, int, str]:
    """Return (line_index, column_offset, line_contents)."""
    line_index: int = result['line'] - 1
    return (line_index,
            result['column'] - 1,
            source[line_index])

def get_fixed_long_line(target: str, previous_line: str, original: str,
                       indent_word: str = '    ', max_line_length: int = 79,
                       aggressive: bool = False, experimental: bool = False, verbose: int = 0) -> Optional[str]:
    """Break up long line and return result.

    Do this by generating multiple reformatted candidates and then
    ranking the candidates to heuristically select the best option.

    """
    indent: str = _get_indentation(target)
    source: str = target[len(indent):]
    assert source.lstrip() == source
    assert not target.lstrip().startswith('#')

    # Check for partial multiline.
    tokens: List[tokenize.TokenInfo] = list(generate_tokens(source))

    candidates: Iterator[str] = shorten_line(
        tokens=tokens,
        source=source,
        indentation=indent,
        indent_word=indent_word,
        max_line_length=max_line_length,
        aggressive=aggressive,
        experimental=experimental,
        previous_line=previous_line
    )

    # Also sort alphabetically as a tie breaker (for determinism).
    candidates = sorted(
        sorted(set(candidates).union([target, original])),
        key=lambda x: line_shortening_rank(
            x,
            indent_word,
            max_line_length,
            experimental=experimental
        )
    )

    if verbose >= 4:
        print(('-' * 79 + '\n').join([''] + list(candidates) + ['']),
              file=wrap_output(sys.stderr, 'utf-8'))

    if candidates:
        best_candidate: str = candidates[0]

        # Don't allow things to get longer.
        if longest_line_length(best_candidate) > longest_line_length(original):
            return None

        return best_candidate

    return None

def longest_line_length(code: str) -> int:
    """Return length of longest line."""
    return max(len(line) for line in code.splitlines())

def join_logical_line(logical_line: str) -> str:
    """Return single line based on logical line input."""
    indentation: str = _get_indentation(logical_line)

    return indentation + untokenize_without_newlines(
        generate_tokens(logical_line.lstrip())) + '\n'

def untokenize_without_newlines(tokens: List[tokenize.TokenInfo]) -> str:
    """Return source code based on tokens."""
    text: str = ''
    last_row: int = 0
    last_column: int = -1

    for t in tokens:
        token_string: str = t[1]
        (start_row, start_column) = t[2]
        (end_row, end_column) = t[3]

        if start_row > last_row:
            last_column = 0
        if (
            (start_column > last_column or token_string == '\n') and
            not text.endswith(' ')
        ):
            text += ' '

        if token_string != '\n':
            text += token_string

        last_row = end_row
        last_column = end_column

    return text.rstrip()

def _find_logical(source_lines: List[str]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    # Make a variable which is the index of all the starts of lines.
    logical_start: List[Tuple[int, int]] = []
    logical_end: List[Tuple[int, int]] = []
    last_newline: bool = True
    parens: int = 0
    for t in generate_tokens(''.join(source_lines)):
        if t[0] in [tokenize.COMMENT, tokenize.DEDENT,
                   tokenize.INDENT, tokenize.NL,
                   tokenize.ENDMARKER]:
            continue
        if not parens and t[0] in [tokenize.NEWLINE, tokenize.SEMI]:
            last_newline = True
            logical_end.append((t[3][0] - 1, t[2][1]))
            continue
        if last_newline and not parens:
            logical_start.append((t[2][0] - 1, t[2][1]))
            last_newline = False
        if t[0] == tokenize.OP:
            if t[1] in '([{':
                parens += 1
            elif t[1] in '}])':
                parens -= 1
        assert parens >= 0, "Unbalanced parentheses in _find_logical"
    return (logical_start, logical_end)

def _get_logical(source_lines: List[str], result: Dict[str, Any], logical_start: List[Tuple[int, int]], logical_end: List[Tuple[int, int]]) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], List[str]]]:
    """Return the logical line corresponding to the result.

    Assumes input is already E702-clean.

    """
    row: int = result['line'] - 1
    col: int = result['column'] - 1
    ls: Optional[Tuple[int, int]] = None
    le: Optional[Tuple[int, int]] = None
    for i in range(0, len(logical_start), 1):
        assert logical_end
        x: Tuple[int, int] = logical_end[i]
        if x[0] > row or (x[0] == row and x[1] > col):
            le = x
            ls = logical_start[i]
            break
    if ls is None:
        return None
    original: List[str] = source_lines[ls[0]:le[0] + 1]
    return ls, le, original

def get_item(items: List[Any], index: int, default: Any = None) -> Any:
    if 0 <= index < len(items):
        return items[index]

    return default

def reindent(source: str, indent_size: int) -> str:
    """Reindent all lines."""
    reindenter = Reindenter(source)
    return reindenter.run(indent_size)

def code_almost_equal(a: str, b: str) -> bool:
    """Return True if code is similar.

    Ignore whitespace when comparing specific line.

    """
    split_a: List[str] = split_and_strip_non_empty_lines(a)
    split_b: List[str] = split_and_strip_non_empty_lines(b)

    if len(split_a) != len(split_b):
        return False

    for (index, _) in enumerate(split_a):
        if ''.join(split_a[index].split()) != ''.join(split_b[index].split()):
            return False

    return True

def split_and_strip_non_empty_lines(text: str) -> List[str]:
    """Return lines split by newline.

    Ignore empty lines.

    """
    return [line.strip() for line in text.splitlines() if line.strip()]

def fix_e265(source: str, aggressive: bool = False) -> str:
    """Format block comments."""
    if '#' not in source:
        # Optimization.
        return source

    ignored_line_numbers: Set[int] = (
        multiline_string_lines(source, include_docstrings=True)
        | set(commented_out_code_lines(source))
    )

    fixed_lines: List[str] = []
    sio: io.StringIO = io.StringIO(source)
    for (line_number, line) in enumerate(sio.readlines(), start=1):
        if (
            line.lstrip().startswith('#') and
            line_number not in ignored_line_numbers and
            not pycodestyle.noqa(line)
        ):
            indentation: str = _get_indentation(line)
            line = line.lstrip()

            # Normalize beginning if not a shebang.
            if len(line) > 1:
                pos: int = next((index for index, c in enumerate(line)
                                if c != '#'), 1)
                if (
                    # Leave multiple spaces like '#    ' alone.
                    (line[:pos].count('#') > 1 or line[1].isalnum()) and
                    # Leave stylistic outlined blocks alone.
                    not line.rstrip().endswith('#')
                ):
                    line = '# ' + line.lstrip('# \t')

            fixed_lines.append(indentation + line)
        else:
            fixed_lines.append(line)

    return ''.join(fixed_lines)

def refactor(source: str, fixer_names: List[str], ignore: Optional[str] = None, filename: str = '') -> str:
    """Return refactored code using lib2to3.

    Skip if ignore string is produced in the refactored code.

    """
    from lib2to3 import pgen2
    try:
        new_text: str = refactor_with_2to3(source,
                                          fixer_names=fixer_names,
                                          filename=filename)
    except (pgen2.parse.ParseError,
            SyntaxError,
            UnicodeDecodeError,
            UnicodeEncodeError):
        return source

    if ignore:
        if ignore in new_text and ignore not in source:
            return source

    return new_text

def code_to_2to3(select: List[str], ignore: str) -> Set[str]:
    """Return set of 2to3 fix names based on PEP8 codes."""
    fixes: Set[str] = set()
    for code, fix in CODE_TO_2TO3.items():
        if code_match(code, select=select, ignore=ignore):
            fixes |= set(fix)
    return fixes

def fix_2to3(source: str,
             aggressive: bool = True, select: Optional[List[str]] = None, ignore: Optional[str] = None, filename: str = '') -> str:
    """Fix various deprecated code (via lib2to3)."""
    if not aggressive:
        return source

    select = select or []
    ignore = ignore or ''

    return refactor(source,
                    fixer_names=list(code_to_2to3(select=select,
                                                 ignore=ignore)),
                    filename=filename)

def fix_w602(source: str, aggressive: bool = True) -> str:
    """Fix deprecated form of raising exception."""
    if not aggressive:
        return source

    return refactor(source, ['raise'],
                    ignore='with_traceback')

def find_newline(source: List[str]) -> str:
    """Return type of newline used in source.

    Input is a list of lines.

    """
    assert not isinstance(source, unicode)

    counter: collections.defaultdict = collections.defaultdict(int)
    for line in source:
        if line.endswith(CRLF):
            counter[CRLF] += 1
        elif line.endswith(CR):
            counter[CR] += 1
        elif line.endswith(LF):
            counter[LF] += 1

    return (sorted(counter, key=counter.get, reverse=True) or [LF])[0]

def _get_indentword(source: str) -> str:
    """Return indentation type."""
    indent_word: str = '    '  # Default in case source has no indentation
    try:
        for t in generate_tokens(source):
            if t[0] == token.INDENT:
                indent_word = t[1]
                break
    except (SyntaxError, tokenize.TokenError):
        pass
    return indent_word

def _get_indentation(line: str) -> str:
    """Return leading whitespace."""
    if line.strip():
        non_whitespace_index: int = len(line) - len(line.lstrip())
        return line[:non_whitespace_index]

    return ''

def get_diff_text(old: List[str], new: List[str], filename: str) -> str:
    """Return text of unified diff between old and new."""
    newline: str = '\n'
    diff: Iterator[str] = difflib.unified_diff(
        old, new,
        'original/' + filename,
        'fixed/' + filename,
        lineterm=newline
    )

    text: str = ''
    for line in diff:
        text += line

        # Work around missing newline (http://bugs.python.org/issue2142).
        if text and not line.endswith(newline):
            text += newline + r'\ No newline at end of file' + newline

    return text

def _priority_key(pep8_result: Dict[str, Any]) -> int:
    """Key for sorting PEP8 results.

    Global fixes should be done first. This is important for things like
    indentation.

    """
    priority: List[str] = [
        # Fix multiline colon-based before semicolon based.
        'e701',
        # Break multiline statements early.
        'e702',
        # Things that make lines longer.
        'e225', 'e231',
        # Remove extraneous whitespace before breaking lines.
        'e201',
        # Shorten whitespace in comment before resorting to wrapping.
        'e262'
    ]
    middle_index: int = 10000
    lowest_priority: List[str] = [
        # We need to shorten lines last since the logical fixer can get in a
        # loop, which causes us to exit early.
        'e501',
        'w503'
    ]
    key: str = pep8_result['id'].lower()
    try:
        return priority.index(key)
    except ValueError:
        try:
            return middle_index + lowest_priority.index(key) + 1
        except ValueError:
            return middle_index

def shorten_line(tokens: List[tokenize.TokenInfo], source: str, indentation: str, indent_word: str, max_line_length: int,
                 aggressive: bool = False, experimental: bool = False, previous_line: str = '') -> Iterator[str]:
    """Separate line at OPERATOR.

    Multiple candidates will be yielded.

    """
    for candidate in _shorten_line(tokens=tokens,
                                   source=source,
                                   indentation=indentation,
                                   indent_word=indent_word,
                                   aggressive=aggressive,
                                   previous_line=previous_line):
        yield candidate

    if aggressive:
        for key_token_strings in SHORTEN_OPERATOR_GROUPS:
            shortened: Optional[str] = _shorten_line_at_tokens(
                tokens=tokens,
                source=source,
                indentation=indentation,
                indent_word=indent_word,
                key_token_strings=key_token_strings,
                aggressive=aggressive
            )

            if shortened is not None and shortened != source:
                yield shortened

    if experimental:
        for shortened in _shorten_line_at_tokens_new(
                tokens=tokens,
                source=source,
                indentation=indentation,
                max_line_length=max_line_length):

            yield shortened

def _shorten_line(tokens: List[tokenize.TokenInfo], source: str, indentation: str, indent_word: str,
                 aggressive: bool = False, previous_line: str = '') -> Iterator[str]:
    """Separate line at OPERATOR.

    The input is expected to be free of newlines except for inside multiline
    strings and at the end.

    Multiple candidates will be yielded.

    """
    for (token_type,
         token_string,
         start_offset,
         end_offset) in token_offsets(tokens):

        if (
            token_type == tokenize.COMMENT and
            not is_probably_part_of_multiline(previous_line) and
            not is_probably_part_of_multiline(source) and
            not source[start_offset + 1:].strip().lower().startswith(
                ('noqa', 'pragma:', 'pylint:'))
        ):
            # Move inline comments to previous line.
            first: str = source[:start_offset]
            second: str = source[start_offset:]
            yield (indentation + second.strip() + '\n' +
                   indentation + first.strip() + '\n')
        elif token_type == token.OP and token_string != '=':
            # Don't break on '=' after keyword as this violates PEP 8.

            assert token_type != token.INDENT

            first: str = source[:end_offset]

            second_indent: str = indentation
            if first.rstrip().endswith('('):
                second_indent += indent_word
            elif '(' in first:
                second_indent += ' ' * (1 + first.find('('))
            else:
                second_indent += indent_word

            second: str = (second_indent + source[end_offset:].lstrip())
            if (
                not second.strip() or
                second.lstrip().startswith('#')
            ):
                continue

            # Do not begin a line with a comma
            if second.lstrip().startswith(','):
                continue
            # Do end a line with a dot
            if first.rstrip().endswith('.'):
                continue
            if token_string in '+-*/':
                fixed: str = first + ' \\' + '\n' + second
            else:
                fixed = first + '\n' + second

            # Only fix if syntax is okay.
            if check_syntax(normalize_multiline(fixed.lstrip())
                            if aggressive else fixed):
                yield indentation + fixed

def _is_binary_operator(token_type: int, text: str) -> bool:
    return ((token_type == tokenize.OP or text in ['and', 'or']) and
            text not in '()[]{},:.;@=%~')

# A convenient way to handle tokens.
Token = collections.namedtuple('Token', ['token_type', 'token_string',
                                         'spos', 'epos', 'line'])

class ReformattedLines(object):

    """The reflowed lines of atoms.

    Each part of the line is represented as an "atom." They can be moved
    around when need be to get the optimal formatting.

    """

    ###########################################################################
    # Private Classes

    class _Indent(object):

        """Represent an indentation in the atom stream."""

        def __init__(self, indent_amt: int):
            self._indent_amt: int = indent_amt

        def emit(self) -> str:
            return ' ' * self._indent_amt

        @property
        def size(self) -> int:
            return self._indent_amt

    class _Space(object):

        """Represent a space in the atom stream."""

        def emit(self) -> str:
            return ' '

        @property
        def size(self) -> int:
            return 1

    class _LineBreak(object):

        """Represent a line break in the atom stream."""

        def emit(self) -> str:
            return '\n'

        @property
        def size(self) -> int:
            return 0

    def __init__(self, max_line_length: int):
        self._max_line_length: int = max_line_length
        self._lines: List[Union[_Indent, _Space, _LineBreak, 'Atom']] = []
        self._bracket_depth: int = 0
        self._prev_item: Optional['Atom'] = None
        self._prev_prev_item: Optional['Atom'] = None

    def __repr__(self) -> str:
        return self.emit()

    ###########################################################################
    # Public Methods

    def add(self, obj: Union['Atom', 'Container'], indent_amt: int, break_after_open_bracket: bool) -> None:
        if isinstance(obj, Atom):
            self._add_item(obj, indent_amt)
            return

        self._add_container(obj, indent_amt, break_after_open_bracket)

    def add_comment(self, item: 'Atom') -> None:
        num_spaces: int = 2
        if len(self._lines) > 1:
            if isinstance(self._lines[-1], self._Space):
                num_spaces -= 1
            if len(self._lines) > 2:
                if isinstance(self._lines[-2], self._Space):
                    num_spaces -= 1

        while num_spaces > 0:
            self._lines.append(self._Space())
            num_spaces -= 1
        self._lines.append(item)

    def add_indent(self, indent_amt: int) -> None:
        self._lines.append(self._Indent(indent_amt))

    def add_line_break(self, indent: str) -> None:
        self._lines.append(self._LineBreak())
        self.add_indent(len(indent))

    def add_line_break_at(self, index: int, indent_amt: int) -> None:
        self._lines.insert(index, self._LineBreak())
        self._lines.insert(index + 1, self._Indent(indent_amt))

    def add_space_if_needed(self, curr_text: str, equal: bool = False) -> None:
        if (
            not self._lines or isinstance(
                self._lines[-1], (self._LineBreak, self._Indent, self._Space))
        ):
            return

        prev_text: str = unicode(self._prev_item) if self._prev_item else ''
        prev_prev_text: str = (
            unicode(self._prev_prev_item) if self._prev_prev_item else ''
        )

        if (
            # The previous item was a keyword or identifier and the current
            # item isn't an operator that doesn't require a space.
            ((self._prev_item.is_keyword or
              self._prev_item.is_string or
              self._prev_item.is_name or
              self._prev_item.is_number) and
             (curr_text[0] not in '([{.,:}])' or
              (curr_text[0] == '=' and equal))) or

            # Don't place spaces around a '.', unless it's in an 'import'
            # statement.
            ((prev_prev_text != 'from' and prev_text[-1] != '.' and
              curr_text != 'import') and

             # Don't place a space before a colon.
             curr_text[0] != ':' and

             # Don't split up ending brackets by spaces.
             ((prev_text[-1] in '}])' and curr_text[0] not in '.,}])') or

              # Put a space after a colon or comma.
              prev_text[-1] in ':,' or

              # Put space around '=' if asked to.
              (equal and prev_text == '=') or

              # Put spaces around non-unary arithmetic operators.
              ((self._prev_prev_item and
                (prev_text not in '+-' and
                 (self._prev_prev_item.is_name or
                  self._prev_prev_item.is_number or
                  self._prev_prev_item.is_string)) and
                prev_text in ('+', '-', '%', '*', '/', '//', '**', 'in'))))
        ):
            self._lines.append(self._Space())

    def previous_item(self) -> Optional['Atom']:
        """Return the previous non-whitespace item."""
        return self._prev_item

    def fits_on_current_line(self, item_extent: int) -> bool:
        return self.current_size() + item_extent <= self._max_line_length

    def current_size(self) -> int:
        """The size of the current line minus the indentation."""
        size: int = 0
        for item in reversed(self._lines):
            size += item.size
            if isinstance(item, self._LineBreak):
                break

        return size

    def line_empty(self) -> bool:
        return (self._lines and
                isinstance(self._lines[-1],
                           (self._LineBreak, self._Indent)))

    def emit(self) -> str:
        string: str = ''
        for item in self._lines:
            if isinstance(item, self._LineBreak):
                string = string.rstrip()
            string += item.emit()

        return string.rstrip() + '\n'

    ###########################################################################
    # Private Methods

    def _add_item(self, item: 'Atom', indent_amt: int) -> None:
        """Add an item to the line.

        Reflow the line to get the best formatting after the item is
        inserted. The bracket depth indicates if the item is being
        inserted inside of a container or not.

        """
        if self._prev_item and self._prev_item.is_string and item.is_string:
            # Place consecutive string literals on separate lines.
            self._lines.append(self._LineBreak())
            self._lines.append(self._Indent(indent_amt))

        item_text: str = unicode(item)
        if self._lines and self._bracket_depth:
            # Adding the item into a container.
            self._prevent_default_initializer_splitting(item, indent_amt)

            if item_text in '.,)]}':
                self._split_after_delimiter(item, indent_amt)

        elif self._lines and not self.line_empty():
            # Adding the item outside of a container.
            if self.fits_on_current_line(len(item_text)):
                self._enforce_space(item)

            else:
                # Line break for the new item.
                self._lines.append(self._LineBreak())
                self._lines.append(self._Indent(indent_amt))

        self._lines.append(item)
        self._prev_item, self._prev_prev_item = item, self._prev_item

        if item_text in '([{':
            self._bracket_depth += 1

        elif item_text in '}])':
            self._bracket_depth -= 1
            assert self._bracket_depth >= 0

    def _add_container(self, container: 'Container', indent_amt: int, break_after_open_bracket: bool) -> None:
        actual_indent: int = indent_amt + 1

        if (
            unicode(container)[0] == '(' and self._prev_item.is_name and
            self._prev_item.is_name
        ):
            # Specific condition can be handled here if needed
            pass

        if (
            not isinstance(container, (ListComprehension, IfExpression)) and
            not self.fits_on_current_line(container.size + self._bracket_depth + 2)
        ):
            self._lines.append(self._LineBreak())
            self._lines.append(self._Indent(indent_amt))
            break_after_open_bracket = False

        container.reflow(self, ' ' * actual_indent,
                        break_after_open_bracket=break_after_open_bracket)

    def _prevent_default_initializer_splitting(self, item: 'Atom', indent_amt: int) -> None:
        """Prevent splitting between a default initializer.

        When there is a default initializer, it's best to keep it all on
        the same line. It's nicer and more readable, even if it goes
        over the maximum allowable line length. This goes back along the
        current line to determine if we have a default initializer, and,
        if so, to remove extraneous whitespaces and add a line
        break/indent before it if needed.

        """
        if unicode(item) == '=':
            # This is the assignment in the initializer. Just remove spaces for
            # now.
            self._delete_whitespace()
            return

        if (not self._prev_item or not self._prev_prev_item or
                unicode(self._prev_item) != '='):
            return

        self._delete_whitespace()
        prev_prev_index: int = self._lines.index(self._prev_prev_item)

        if (
            isinstance(self._lines[prev_prev_index - 1], self._Indent) or
            self.fits_on_current_line(self._get_extent(prev_prev_index) + 2)
        ):
            # The default initializer is already the only item on this line.
            # Don't insert a newline here.
            return

        # Replace the space with a newline/indent combo.
        if isinstance(self._lines[prev_prev_index - 1], self._Space):
            del self._lines[prev_prev_index - 1]

        self.add_line_break_at(self._lines.index(self._prev_prev_item),
                               indent_amt)

    def _split_after_delimiter(self, item: 'Atom', indent_amt: int) -> None:
        """Split the line only after a delimiter."""
        self._delete_whitespace()

        if self.fits_on_current_line(item.size):
            return

        last_space: Optional['ReformattedLines._Space'] = None
        for current_item in reversed(self._lines):
            if (
                last_space and
                (not isinstance(current_item, Atom) or
                 not current_item.is_colon)
            ):
                break
            else:
                last_space = None
            if isinstance(current_item, self._Space):
                last_space = current_item
            if isinstance(current_item, (self._LineBreak, self._Indent)):
                return

        if not last_space:
            return

        self.add_line_break_at(self._lines.index(last_space), indent_amt)

    def _enforce_space(self, item: 'Atom') -> None:
        """Enforce a space in certain situations.

        There are cases where we will want a space where normally we
        wouldn't put one. This just enforces the addition of a space.

        """
        if isinstance(self._lines[-1],
                      (self._Space, self._LineBreak, self._Indent)):
            return

        if not self._prev_item:
            return

        item_text: str = unicode(item)
        prev_text: str = unicode(self._prev_item)

        # Prefer a space around a '.' in an import statement, and between the
        # 'import' and '('.
        if (
            (item_text == '.' and prev_text == 'from') or
            (item_text == 'import' and prev_text == '.') or
            (item_text == '(' and prev_text == 'import')
        ):
            self._lines.append(self._Space())

    def _delete_whitespace(self) -> None:
        """Delete all whitespace from the end of the line."""
        while isinstance(self._lines[-1], (self._Space, self._LineBreak, self._Indent)):
            del self._lines[-1]

class Atom(object):

    """The smallest unbreakable unit that can be reflowed."""

    def __init__(self, atom: Token):
        self._atom: Token = atom

    def __repr__(self) -> str:
        return self._atom.token_string

    def __len__(self) -> int:
        return self.size

    def reflow(
        self, reflowed_lines: ReformattedLines, continued_indent: str, extent: Optional[int],
        break_after_open_bracket: bool = False,
        is_list_comp_or_if_expr: bool = False,
        next_is_dot: bool = False
    ) -> None:
        if self._atom.token_type == tokenize.COMMENT:
            reflowed_lines.add_comment(self)
            return

        total_size: int = extent if extent else self.size

        if self._atom.token_string not in ',:([{}])':
            # Some atoms will need an extra 1-sized space token after them.
            total_size += 1

        prev_item: Optional[Atom] = reflowed_lines.previous_item()
        if self._atom.token_string not in ',:([{}])':
            reflowed_lines.add_space_if_needed(unicode(self), equal=True)

        reflowed_lines.add(self, len(continued_indent),
                           break_after_open_bracket)

    def emit(self) -> str:
        return self.__repr__()

    @property
    def is_keyword(self) -> bool:
        return keyword.iskeyword(self._atom.token_string)

    @property
    def is_string(self) -> bool:
        return self._atom.token_type == tokenize.STRING

    @property
    def is_name(self) -> bool:
        return self._atom.token_type == tokenize.NAME

    @property
    def is_number(self) -> bool:
        return self._atom.token_type == tokenize.NUMBER

    @property
    def is_comma(self) -> bool:
        return self._atom.token_string == ','

    @property
    def is_colon(self) -> bool:
        return self._atom.token_string == ':'

    @property
    def size(self) -> int:
        return len(self._atom.token_string)

class Container(object):

    """Base class for all container types."""

    def __init__(self, items: List[Union[Atom, 'Container']]):
        self._items: List[Union[Atom, 'Container']] = items

    def __repr__(self) -> str:
        string: str = ''
        last_was_keyword: bool = False

        for item in self._items:
            if item.is_comma:
                string += ', '
            elif item.is_colon:
                string += ': '
            else:
                item_string: str = unicode(item)
                if (
                    string and
                    (last_was_keyword or
                     (not string.endswith(tuple('([{,.:}]) ')) and
                      not item_string.startswith(tuple('([{,.:}])'))))
                ):
                    string += ' '
                string += item_string

            last_was_keyword = item.is_keyword
        return string

    def __iter__(self) -> Iterator[Union[Atom, 'Container']]:
        for element in self._items:
            yield element

    def __getitem__(self, idx: int) -> Union[Atom, 'Container']:
        return self._items[idx]

    def reflow(self, reflowed_lines: ReformattedLines, continued_indent: str,
               break_after_open_bracket: bool = False) -> None:
        last_was_container: bool = False
        for (index, item) in enumerate(self._items):
            next_item: Optional[Union[Atom, 'Container']] = get_item(self._items, index + 1)

            if isinstance(item, Atom):
                is_list_comp_or_if_expr: bool = isinstance(self, (ListComprehension, IfExpression))
                item.reflow(reflowed_lines, continued_indent,
                            self._get_extent(index),
                            is_list_comp_or_if_expr=is_list_comp_or_if_expr,
                            next_is_dot=(next_item and
                                         unicode(next_item) == '.'))
                if last_was_container and item.is_comma:
                    reflowed_lines.add_line_break(continued_indent)
                last_was_container = False
            else:  # isinstance(item, Container)
                reflowed_lines.add(item, len(continued_indent),
                                   break_after_open_bracket)
                last_was_container = not isinstance(item, (ListComprehension,
                                                           IfExpression))

            if (
                break_after_open_bracket and index == 0 and
                # Prefer to keep empty containers together instead of
                # separating them.
                unicode(item) == self.open_bracket and
                (not next_item or unicode(next_item) != self.close_bracket) and
                (len(self._items) != 3 or not isinstance(next_item, Atom))
            ):
                reflowed_lines.add_line_break(continued_indent)
                break_after_open_bracket = False
            else:
                next_next_item: Optional[Union[Atom, 'Container']] = get_item(self._items, index + 2)
                if (
                    unicode(item) not in ['.', '%', '+', '-', '/'] and
                    next_item and not isinstance(next_item, Container) and
                    unicode(next_item) != ':' and
                    next_next_item and (not isinstance(next_next_item, Atom) or
                                        unicode(next_item) == 'not') and
                    not reflowed_lines.line_empty() and
                    not reflowed_lines.fits_on_current_line(
                        self._get_extent(index + 1) + 2)
                ):
                    reflowed_lines.add_line_break(continued_indent)

    def _get_extent(self, index: int) -> int:
        """The extent of the full element.

        E.g., the length of a function call or keyword.

        """
        extent: int = 0
        prev_item: Optional[Union[Atom, 'Container']] = get_item(self._items, index - 1)
        seen_dot: bool = prev_item and unicode(prev_item) == '.'
        while index < len(self._items):
            item: Optional[Union[Atom, 'Container']] = get_item(self._items, index)
            if not item:
                break

            if isinstance(item, (ListComprehension, IfExpression)):
                break

            if isinstance(item, Atom):
                if prev_item and prev_item.is_comma:
                    pass
                else:
                    if seen_dot:
                        extent += 1
            else:
                if prev_item and prev_item.is_name:
                    if seen_dot:
                        extent += 1
                    else:
                        extent += item.size
                if not isinstance(item, Atom):
                    extent += item.size

            if isinstance(item, Container):
                if prev_item and prev_item.is_name:
                    if seen_dot:
                        extent += 1
                    else:
                        extent += item.size

            elif isinstance(item, Atom):
                if item.is_comma or item.is_colon:
                    pass
                else:
                    extent += item.size

            if isinstance(item, Atom):
                if item.token_string == '.':
                    seen_dot = True
                else:
                    seen_dot = False

            index += 1

        return extent

    @property
    def is_string(self) -> bool:
        return False

    @property
    def size(self) -> int:
        return len(self.__repr__())

    @property
    def is_keyword(self) -> bool:
        return False

    @property
    def is_name(self) -> bool:
        return False

    @property
    def is_comma(self) -> bool:
        return False

    @property
    def is_colon(self) -> bool:
        return False

    @property
    def open_bracket(self) -> Optional[str]:
        return None

    @property
    def close_bracket(self) -> Optional[str]:
        return None

class Tuple(Container):

    """A high-level representation of a tuple."""

    @property
    def open_bracket(self) -> str:
        return '('

    @property
    def close_bracket(self) -> str:
        return ')'

class List(Container):

    """A high-level representation of a list."""

    @property
    def open_bracket(self) -> str:
        return '['

    @property
    def close_bracket(self) -> str:
        return ']'

class DictOrSet(Container):

    """A high-level representation of a dictionary or set."""

    @property
    def open_bracket(self) -> str:
        return '{'

    @property
    def close_bracket(self) -> str:
        return '}'

class ListComprehension(Container):

    """A high-level representation of a list comprehension."""

    @property
    def size(self) -> int:
        length: int = 0
        for item in self._items:
            if isinstance(item, IfExpression):
                break
            length += item.size
        return length

class IfExpression(Container):

    """A high-level representation of an if-expression."""

def _parse_container(tokens: List[tokenize.TokenInfo], index: int, for_or_if: Optional[str] = None) -> Tuple[Optional[Container], Optional[int]]:
    """Parse a high-level container, such as a list, tuple, etc."""

    # Store the opening bracket.
    items: List[Union[Atom, Container]] = [Atom(Token(*tokens[index]))]
    index += 1

    num_tokens: int = len(tokens)
    while index < num_tokens:
        tok: tokenize.TokenInfo = tokens[index]

        if tok.token_string in ',)]}':
            # First check if we're at the end of a list comprehension or
            # if-expression. Don't add the ending token as part of the list
            # comprehension or if-expression, because they aren't part of those
            # constructs.
            if for_or_if == 'for':
                return (ListComprehension(items), index - 1)

            elif for_or_if == 'if':
                return (IfExpression(items), index - 1)

            # We've reached the end of a container.
            items.append(Atom(tok))

            # If not, then we are at the end of a container.
            if tok.token_string == ')':
                # The end of a tuple.
                return (Tuple(items), index)

            elif tok.token_string == ']':
                # The end of a list.
                return (List(items), index)

            elif tok.token_string == '}':
                # The end of a dictionary or set.
                return (DictOrSet(items), index)

        elif tok.token_string in '([{':
            # A sub-container is being defined.
            container, new_index = _parse_container(tokens, index)
            if not container:
                return (None, None)
            items.append(container)
            index = new_index

        elif tok.token_string == 'for':
            container, new_index = _parse_container(tokens, index, 'for')
            items.append(container)
            index = new_index

        elif tok.token_string == 'if':
            container, new_index = _parse_container(tokens, index, 'if')
            items.append(container)
            index = new_index

        else:
            items.append(Atom(tok))

        index += 1

    return (None, None)

def _parse_tokens(tokens: List[tokenize.TokenInfo]) -> Optional[List[Union[Atom, Container]]]:
    """Parse the tokens.

    This converts the tokens into a form where we can manipulate them
    more easily.

    """

    index: int = 0
    parsed_tokens: List[Union[Atom, Container]] = []

    num_tokens: int = len(tokens)
    while index < num_tokens:
        tok: tokenize.TokenInfo = tokens[index]

        assert tok.type != token.INDENT
        if tok.type == tokenize.NEWLINE:
            # There's only one newline and it's at the end.
            break

        if tok.string in '([{':
            container, new_index = _parse_container(tokens, index)
            if not container:
                return None
            parsed_tokens.append(container)
            index = new_index
        else:
            parsed_tokens.append(Atom(tok))

        index += 1

    return parsed_tokens

def _reflow_lines(parsed_tokens: List[Union[Atom, Container]], indentation: str, max_line_length: int,
                 start_on_prefix_line: bool) -> Optional[str]:
    """Reflow the lines so that it looks nice."""

    if unicode(parsed_tokens[0]) == 'def':
        # A function definition gets indented a bit more.
        continued_indent: str = indentation + ' ' * 2 * DEFAULT_INDENT_SIZE
    else:
        continued_indent: str = indentation + ' ' * DEFAULT_INDENT_SIZE

    break_after_open_bracket: bool = not start_on_prefix_line

    lines: ReformattedLines = ReformattedLines(max_line_length)
    lines.add_indent(len(indentation.lstrip('\r\n')))

    if not start_on_prefix_line:
        # If splitting after the opening bracket will cause the first element
        # to be aligned weirdly, don't try it.
        first_token: Optional[Union[Atom, Container]] = get_item(parsed_tokens, 0)
        second_token: Optional[Union[Atom, Container]] = get_item(parsed_tokens, 1)

        if (
            first_token and second_token and
            unicode(second_token)[0] == '(' and
            len(indentation) + len(first_token) + 1 == len(continued_indent)
        ):
            return None

    for item in parsed_tokens:
        lines.add_space_if_needed(unicode(item), equal=True)

        save_continued_indent: str = continued_indent
        if start_on_prefix_line and isinstance(item, Container):
            start_on_prefix_line = False
            continued_indent = ' ' * (lines.current_size() + 1)

        item.reflow(lines, continued_indent, break_after_open_bracket)
        continued_indent = save_continued_indent

    return lines.emit()

def _shorten_line_at_tokens_new(tokens: List[tokenize.TokenInfo], source: str, indentation: str,
                                max_line_length: int) -> Optional[str]:
    """Shorten the line taking its length into account.

    The input is expected to be free of newlines except for inside
    multiline strings and at the end.

    """
    # Yield the original source so to see if it's a better choice than the
    # shortened candidate lines we generate here.
    yield indentation + source

    parsed_tokens: Optional[List[Union[Atom, Container]]] = _parse_tokens(tokens)

    if parsed_tokens:
        # Perform two reflows. The first one starts on the same line as the
        # prefix. The second starts on the line after the prefix.
        fixed: Optional[str] = _reflow_lines(parsed_tokens, indentation, max_line_length,
                                            start_on_prefix_line=True)
        if fixed and check_syntax(normalize_multiline(fixed.lstrip())):
            yield fixed

        fixed = _reflow_lines(parsed_tokens, indentation, max_line_length,
                              start_on_prefix_line=False)
        if fixed and check_syntax(normalize_multiline(fixed.lstrip())):
            yield fixed

def _shorten_line_at_tokens(tokens: List[tokenize.TokenInfo], source: str, indentation: str, indent_word: str,
                           key_token_strings: FrozenSet[str], aggressive: bool) -> Optional[str]:
    """Separate line by breaking at tokens in key_token_strings.

    The input is expected to be free of newlines except for inside
    multiline strings and at the end.

    """
    offsets: List[int] = []
    for (index, _) in enumerate(token_offsets(tokens)):
        (token_type,
         token_string,
         start_offset,
         end_offset) = _

        if token_string in key_token_strings:
            # Do not break in containers with zero or one items.
            unwanted_next_token: Optional[str] = {
                '(': ')',
                '[': ']',
                '{': '}'
            }.get(token_string)
            if unwanted_next_token:
                if (
                    get_item(tokens,
                             index + 1,
                             default=[None, None])[1] == unwanted_next_token or
                    get_item(tokens,
                             index + 2,
                             default=[None, None])[1] == unwanted_next_token
                ):
                    continue

            if (
                index > 2 and token_string == '(' and
                tokens[index - 1].string in ',(%['
            ):
                # Don't split after a tuple start, or before a tuple start if
                # the tuple is in a list.
                continue

            if (
                end_offset < len(source) - 1
            ):
                # Don't split right before newline.
                offsets.append(end_offset)
        else:
            # Break at adjacent strings. These were probably meant to be on
            # separate lines in the first place.
            previous_token: Optional[tokenize.TokenInfo] = get_item(tokens, index - 1)
            if (
                token_type == tokenize.STRING and
                previous_token and previous_token.type == tokenize.STRING
            ):
                offsets.append(start_offset)

    current_indent: Optional[int] = None
    fixed: Optional[str] = None
    for line in split_at_offsets(source, offsets):
        if fixed:
            fixed += '\n' + current_indent + line

            for symbol in '([{':
                if line.endswith(symbol):
                    current_indent += indent_word
        else:
            # First line.
            fixed = line
            assert not current_indent
            current_indent = indent_word

    assert fixed is not None

    if check_syntax(normalize_multiline(fixed.lstrip()) if aggressive > 1 else fixed):
        return indentation + fixed

    return None

def token_offsets(tokens: List[tokenize.TokenInfo]) -> Iterator[Tuple[int, str, int, int]]:
    """Yield tokens and offsets."""
    end_offset: int = 0
    previous_end_row: int = 0
    previous_end_column: int = 0
    for t in tokens:
        token_type: int = t.type
        token_string: str = t.string
        (start_row, start_column) = t.start
        (end_row, end_column) = t.end

        # Account for the whitespace between tokens.
        end_offset += start_column
        if previous_end_row == start_row:
            end_offset -= previous_end_column

        # Record the start offset of the token.
        start_offset: int = end_offset

        # Account for the length of the token itself.
        end_offset += len(token_string)

        yield (token_type,
               token_string,
               start_offset,
               end_offset)

        previous_end_row = end_row
        previous_end_column = end_column

class ReformattedLines(object):

    """The reflowed lines of atoms.

    Each part of the line is represented as an "atom." They can be moved
    around when need be to get the optimal formatting.

    """

    ###########################################################################
    # Private Classes

    class _Indent(object):

        """Represent an indentation in the atom stream."""

        def __init__(self, indent_amt: int):
            self._indent_amt: int = indent_amt

        def emit(self) -> str:
            return ' ' * self._indent_amt

        @property
        def size(self) -> int:
            return self._indent_amt

    class _Space(object):

        """Represent a space in the atom stream."""

        def emit(self) -> str:
            return ' '

        @property
        def size(self) -> int:
            return 1

    class _LineBreak(object):

        """Represent a line break in the atom stream."""

        def emit(self) -> str:
            return '\n'

        @property
        def size(self) -> int:
            return 0

    def __init__(self, max_line_length: int):
        self._max_line_length: int = max_line_length
        self._lines: List[Union[_Indent, _Space, _LineBreak, Atom]] = []
        self._bracket_depth: int = 0
        self._prev_item: Optional[Atom] = None
        self._prev_prev_item: Optional[Atom] = None

    def __repr__(self) -> str:
        return self.emit()

    ###########################################################################
    # Public Methods

    def add(self, obj: Union[Atom, Container], indent_amt: int, break_after_open_bracket: bool) -> None:
        if isinstance(obj, Atom):
            self._add_item(obj, indent_amt)
            return

        self._add_container(obj, indent_amt, break_after_open_bracket)

    def add_comment(self, item: Atom) -> None:
        num_spaces: int = 2
        if len(self._lines) > 1:
            if isinstance(self._lines[-1], self._Space):
                num_spaces -= 1
            if len(self._lines) > 2:
                if isinstance(self._lines[-2], self._Space):
                    num_spaces -= 1

        while num_spaces > 0:
            self._lines.append(ReformattedLines._Space())
            num_spaces -= 1
        self._lines.append(item)

    def add_indent(self, indent_amt: int) -> None:
        self._lines.append(ReformattedLines._Indent(indent_amt))

    def add_line_break(self, indent: str) -> None:
        self._lines.append(ReformattedLines._LineBreak())
        self.add_indent(len(indent))

    def add_line_break_at(self, index: int, indent_amt: int) -> None:
        self._lines.insert(index, ReformattedLines._LineBreak())
        self._lines.insert(index + 1, ReformattedLines._Indent(indent_amt))

    def add_space_if_needed(self, curr_text: str, equal: bool = False) -> None:
        if (
            not self._lines or isinstance(
                self._lines[-1], (ReformattedLines._LineBreak, ReformattedLines._Indent, ReformattedLines._Space))
        ):
            return

        prev_text: str = unicode(self._prev_item) if self._prev_item else ''
        prev_prev_text: str = (
            unicode(self._prev_prev_item) if self._prev_prev_item else ''
        )

        if (
            # The previous item was a keyword or identifier and the current
            # item isn't an operator that doesn't require a space.
            ((self._prev_item.is_keyword or
              self._prev_item.is_string or
              self._prev_item.is_name or
              self._prev_item.is_number) and
             (curr_text[0] not in '([{.,:}])' or
              (curr_text[0] == '=' and equal))) or

            # Don't place spaces around a '.', unless it's in an 'import'
            # statement.
            ((prev_prev_text != 'from' and prev_text[-1] != '.' and
              curr_text != 'import') and

             # Don't place a space before a colon.
             curr_text[0] != ':' and

             # Don't split up ending brackets by spaces.
             ((prev_text[-1] in '}])' and curr_text[0] not in '.,}])') or

              # Put a space after a colon or comma.
              prev_text[-1] in ':,' or

              # Put space around '=' if asked to.
              (equal and prev_text == '=') or

              # Put spaces around non-unary arithmetic operators.
              ((self._prev_prev_item and
                (prev_text not in '+-' and
                 (self._prev_prev_item.is_name or
                  self._prev_prev_item.is_number or
                  self._prev_prev_item.is_string)) and
                prev_text in ('+', '-', '%', '*', '/', '//', '**', 'in'))))
        ):
            self._lines.append(ReformattedLines._Space())

    def previous_item(self) -> Optional[Atom]:
        """Return the previous non-whitespace item."""
        return self._prev_item

    def fits_on_current_line(self, item_extent: int) -> bool:
        return self.current_size() + item_extent <= self._max_line_length

    def current_size(self) -> int:
        """The size of the current line minus the indentation."""
        size: int = 0
        for item in reversed(self._lines):
            size += item.size
            if isinstance(item, ReformattedLines._LineBreak):
                break

        return size

    def line_empty(self) -> bool:
        return (self._lines and
                isinstance(self._lines[-1],
                           (ReformattedLines._LineBreak, ReformattedLines._Indent)))

    def emit(self) -> str:
        string: str = ''
        for item in self._lines:
            if isinstance(item, ReformattedLines._LineBreak):
                string = string.rstrip()
            string += item.emit()

        return string.rstrip() + '\n'

    ###########################################################################
    # Private Methods

    def _add_item(self, item: Atom, indent_amt: int) -> None:
        """Add an item to the line.

        Reflow the line to get the best formatting after the item is
        inserted. The bracket depth indicates if the item is being
        inserted inside of a container or not.

        """
        if self._prev_item and self._prev_item.is_string and item.is_string:
            # Place consecutive string literals on separate lines.
            self._lines.append(ReformattedLines._LineBreak())
            self._lines.append(ReformattedLines._Indent(indent_amt))

        item_text: str = unicode(item)
        if self._lines and self._bracket_depth:
            # Adding the item into a container.
            self._prevent_default_initializer_splitting(item, indent_amt)

            if item_text in '.,)]}':
                self._split_after_delimiter(item, indent_amt)

        elif self._lines and not self.line_empty():
            # Adding the item outside of a container.
            if self.fits_on_current_line(len(item_text)):
                self._enforce_space(item)

            else:
                # Line break for the new item.
                self._lines.append(ReformattedLines._LineBreak())
                self._lines.append(ReformattedLines._Indent(indent_amt))

        self._lines.append(item)
        self._prev_item, self._prev_prev_item = item, self._prev_item

        if item_text in '([{':
            self._bracket_depth += 1

        elif item_text in '}])':
            self._bracket_depth -= 1
            assert self._bracket_depth >= 0

    def _add_container(self, container: Container, indent_amt: int, break_after_open_bracket: bool) -> None:
        actual_indent: int = indent_amt + 1

        if (
            not isinstance(container, (ListComprehension, IfExpression)) and
            not self.fits_on_current_line(container.size + self._bracket_depth + 2)
        ):

            if isinstance(container, Container):
                self._lines.append(ReformattedLines._LineBreak())
                self._lines.append(ReformattedLines._Indent(indent_amt))
                break_after_open_bracket = False

        container.reflow(self, ' ' * actual_indent,
                        break_after_open_bracket=break_after_open_bracket)

    def _prevent_default_initializer_splitting(self, item: Atom, indent_amt: int) -> None:
        """Prevent splitting between a default initializer.

        When there is a default initializer, it's best to keep it all on
        the same line. It's nicer and more readable, even if it goes
        over the maximum allowable line length. This goes back along the
        current line to determine if we have a default initializer, and,
        if so, to remove extraneous whitespaces and add a line
        break/indent before it if needed.

        """
        if unicode(item) == '=':
            # This is the assignment in the initializer. Just remove spaces for
            # now.
            self._delete_whitespace()
            return

        if (not self._prev_item or not self._prev_prev_item or
                unicode(self._prev_item) != '='):
            return

        self._delete_whitespace()
        prev_prev_index: int = self._lines.index(self._prev_prev_item)

        if (
            isinstance(self._lines[prev_prev_index - 1], ReformattedLines._Indent) or
            self.fits_on_current_line(self._get_extent(prev_prev_index) + 2)
        ):
            # The default initializer is already the only item on this line.
            # Don't insert a newline here.
            return

        # Replace the space with a newline/indent combo.
        if isinstance(self._lines[prev_prev_index - 1], ReformattedLines._Space):
            del self._lines[prev_prev_index - 1]

        self.add_line_break_at(self._lines.index(self._prev_prev_item),
                               indent_amt)

    def _split_after_delimiter(self, item: Atom, indent_amt: int) -> None:
        """Split the line only after a delimiter."""
        self._delete_whitespace()

        if self.fits_on_current_line(item.size):
            return

        last_space: Optional[ReformattedLines._Space] = None
        for current_item in reversed(self._lines):
            if (
                last_space and
                (not isinstance(current_item, Atom) or
                 not current_item.is_colon)
            ):
                break
            else:
                last_space = None
            if isinstance(current_item, ReformattedLines._Space):
                last_space = current_item
            if isinstance(current_item, (ReformattedLines._LineBreak, ReformattedLines._Indent)):
                return

        if not last_space:
            return

        self.add_line_break_at(self._lines.index(last_space), indent_amt)

    def _enforce_space(self, item: Atom) -> None:
        """Enforce a space in certain situations.

        There are cases where we will want a space where normally we
        wouldn't put one. This just enforces the addition of a space.

        """
        if isinstance(self._lines[-1],
                      (ReformattedLines._Space, ReformattedLines._LineBreak, ReformattedLines._Indent)):
            return

        if not self._prev_item:
            return

        item_text: str = unicode(item)
        prev_text: str = unicode(self._prev_item)

        # Prefer a space around a '.' in an import statement, and between the
        # 'import' and '('.
        if (
            (item_text == '.' and prev_text == 'from') or
            (item_text == 'import' and prev_text == '.') or
            (item_text == '(' and prev_text == 'import')
        ):
            self._lines.append(ReformattedLines._Space())

    def _delete_whitespace(self) -> None:
        """Delete all whitespace from the end of the line."""
        while isinstance(self._lines[-1], (ReformattedLines._Space, ReformattedLines._LineBreak, ReformattedLines._Indent)):
            del self._lines[-1]

def has_arithmetic_operator(line: str) -> bool:
    """Return True if line contains any arithmetic operators."""
    for operator in pycodestyle.ARITHMETIC_OP:
        if operator in line:
            return True

    return False

def count_unbalanced_brackets(line: str) -> int:
    """Return number of unmatched open/close brackets."""
    count: int = 0
    for opening, closing in ['()', '[]', '{}']:
        count += abs(line.count(opening) - line.count(closing))

    return count

def split_at_offsets(line: str, offsets: List[int]) -> List[str]:
    """Split line at offsets.

    Return list of strings.

    """
    result: List[str] = []

    previous_offset: int = 0
    current_offset: int = 0
    for current_offset in sorted(offsets):
        if current_offset < len(line) and previous_offset != current_offset:
            result.append(line[previous_offset:current_offset].strip())
        previous_offset = current_offset

    result.append(line[current_offset:])

    return result

class LineEndingWrapper(object):

    """Replace line endings to work with sys.stdout.

    It seems that sys.stdout expects only '\n' as the line ending, no matter
    the platform. Otherwise, we get repeated line endings.

    """

    def __init__(self, output: io.TextIOBase):
        self.__output: io.TextIOBase = output

    def write(self, s: str) -> None:
        self.__output.write(s.replace('\r\n', '\n').replace('\r', '\n'))

    def flush(self) -> None:
        self.__output.flush()

def match_file(filename: str, exclude: Set[str]) -> bool:
    """Return True if file is okay for modifying/recursing."""
    base_name: str = os.path.basename(filename)

    if base_name.startswith('.'):
        return False

    for pattern in exclude:
        if fnmatch.fnmatch(base_name, pattern):
            return False
        if fnmatch.fnmatch(filename, pattern):
            return False

    if not os.path.isdir(filename) and not is_python_file(filename):
        return False

    return True

def find_files(filenames: List[str], recursive: bool, exclude: Set[str]) -> Iterator[str]:
    """Yield filenames."""
    while filenames:
        name: str = filenames.pop(0)
        if recursive and os.path.isdir(name):
            for root, directories, children in os.walk(name):
                filenames += [os.path.join(root, f) for f in children
                              if match_file(os.path.join(root, f),
                                            exclude)]
                directories[:] = [d for d in directories
                                  if match_file(os.path.join(root, d),
                                                exclude)]
        else:
            yield name

def _fix_file(parameters: Tuple[str, Any, Optional[io.TextIOBase]]) -> None:
    """Helper function for optionally running fix_file() in parallel."""
    if parameters[1].verbose:
        print(f'[file:{parameters[0]}]', file=sys.stderr)
    try:
        fix_file(*parameters)
    except IOError as error:
        print(unicode(error), file=sys.stderr)

def fix_multiple_files(filenames: List[str], options: Any, output: Optional[io.TextIOBase] = None) -> None:
    """Fix list of files.

    Optionally fix files recursively.

    """
    filenames_iterator: Iterator[str] = find_files(filenames, options.recursive, options.exclude)
    if options.jobs > 1:
        import multiprocessing
        pool: multiprocessing.pool.Pool = multiprocessing.Pool(options.jobs)
        pool.map(_fix_file,
                 [(name, options) for name in filenames_iterator])
        pool.close()
        pool.join()
    else:
        for name in filenames_iterator:
            _fix_file((name, options, output))

def is_python_file(filename: str) -> bool:
    """Return True if filename is Python file."""
    if filename.endswith('.py'):
        return True

    try:
        with open_with_encoding(
                filename,
                limit_byte_check=MAX_PYTHON_FILE_DETECTION_BYTES) as f:
            text: str = f.read(MAX_PYTHON_FILE_DETECTION_BYTES)
            if not text:
                return False
            first_line: str = text.splitlines()[0]
    except (IOError, IndexError):
        return False

    if not PYTHON_SHEBANG_REGEX.match(first_line):
        return False

    return True

def is_probably_part_of_multiline(line: str) -> bool:
    """Return True if line is likely part of a multiline string.

    When multiline strings are involved, pep8 reports the error as being
    at the start of the multiline string, which doesn't work for us.

    """
    return (
        '"""' in line or
        "'''" in line or
        line.rstrip().endswith('\\')
    )

def wrap_output(output: io.TextIOBase, encoding: str) -> LineEndingWrapper:
    """Return output with specified encoding."""
    return LineEndingWrapper(codecs.getwriter(encoding)(output.buffer
                                      if hasattr(output, 'buffer')
                                      else output))

def get_encoding() -> str:
    """Return preferred encoding."""
    return locale.getpreferredencoding() or sys.getdefaultencoding()

def create_parser() -> argparse.ArgumentParser:
    """Return command-line parser."""
    # Do import locally to be friendly to those who use autopep8 as a library
    # and are supporting Python 2.6.
    import argparse

    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=docstring_summary(__doc__),
                                     prog='autopep8')
    parser.add_argument('--version', action='version',
                        version=f'%(prog)s {__version__} ({_get_package_version()})')
    parser.add_argument('-v', '--verbose', action='count',
                        default=0,
                        help='print verbose messages; '
                             'multiple -v result in more verbose messages')
    parser.add_argument('-d', '--diff', action='store_true',
                        help='print the diff for the fixed source')
    parser.add_argument('-i', '--in-place', action='store_true',
                        help='make changes to files in place')
    parser.add_argument('--global-config', metavar='filename',
                        default=DEFAULT_CONFIG,
                        help='path to a global pep8 config file; if this file '
                             'does not exist then this is ignored '
                             f'(default: {DEFAULT_CONFIG})')
    parser.add_argument('--ignore-local-config', action='store_true',
                        help="don't look for and apply local config files; "
                             'if not passed, defaults are updated with any '
                             "config files in the project's root directory")
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='run recursively over directories; '
                             'must be used with --in-place or --diff')
    parser.add_argument('-j', '--jobs', type=int, metavar='n', default=1,
                        help='number of parallel jobs; '
                             'match CPU count if value is less than 1')
    parser.add_argument('-p', '--pep8-passes', metavar='n',
                        default=-1, type=int,
                        help='maximum number of additional pep8 passes '
                             '(default: infinite)')
    parser.add_argument('-a', '--aggressive', action='count', default=0,
                        help='enable non-whitespace changes; '
                             'multiple -a result in more aggressive changes')
    parser.add_argument('--experimental', action='store_true',
                        help='enable experimental fixes')
    parser.add_argument('--exclude', metavar='globs',
                        help='exclude file/directory names that match these '
                             'comma-separated globs')
    parser.add_argument('--list-fixes', action='store_true',
                        help='list codes for fixes; '
                        'used by --ignore and --select')
    parser.add_argument('--ignore', metavar='errors', default='',
                        help='do not fix these errors/warnings '
                             f'(default: {DEFAULT_IGNORE})')
    parser.add_argument('--select', metavar='errors', default='',
                        help='fix only these errors/warnings (e.g. E4,W)')
    parser.add_argument('--max-line-length', metavar='n', default=79, type=int,
                        help='set maximum allowed line length '
                             '(default: %(default)s)')
    parser.add_argument('--line-range', '--range', metavar='line',
                        default=None, type=int, nargs=2,
                        help='only fix errors found within this inclusive '
                             'range of line numbers (e.g. 1 99); '
                             'line numbers are indexed at 1')
    parser.add_argument('--indent-size', default=DEFAULT_INDENT_SIZE,
                        type=int, help=argparse.SUPPRESS)
    parser.add_argument('files', nargs='*',
                        help="files to format or '-' for standard in")

    return parser

def parse_args(arguments: List[str], apply_config: bool = False) -> argparse.Namespace:
    """Parse command-line options."""
    parser: argparse.ArgumentParser = create_parser()
    args: argparse.Namespace = parser.parse_args(arguments)

    if not args.files and not args.list_fixes:
        parser.error('incorrect number of arguments')

    args.files = [decode_filename(name) for name in args.files]

    if apply_config:
        parser = read_config(args, parser)
        args = parser.parse_args(arguments)
        args.files = [decode_filename(name) for name in args.files]

    if '-' in args.files:
        if len(args.files) > 1:
            parser.error('cannot mix stdin and regular files')

        if args.diff:
            parser.error('--diff cannot be used with standard input')

        if args.in_place:
            parser.error('--in-place cannot be used with standard input')

        if args.recursive:
            parser.error('--recursive cannot be used with standard input')

    if len(args.files) > 1 and not (args.in_place or args.diff):
        parser.error('autopep8 only takes one filename as argument '
                     'unless the "--in-place" or "--diff" args are '
                     'used')

    if args.recursive and not (args.in_place or args.diff):
        parser.error('--recursive must be used with --in-place or --diff')

    if args.in_place and args.diff:
        parser.error('--in-place and --diff are mutually exclusive')

    if args.max_line_length <= 0:
        parser.error('--max-line-length must be greater than 0')

    if args.select:
        args.select = _split_comma_separated(args.select)

    if args.ignore:
        args.ignore = _split_comma_separated(args.ignore)
    elif not args.select:
        if args.aggressive:
            # Enable everything by default if aggressive.
            args.select = set(['E', 'W'])
        else:
            args.ignore = _split_comma_separated(DEFAULT_IGNORE)

    if args.exclude:
        args.exclude = _split_comma_separated(args.exclude)
    else:
        args.exclude = set([])

    if args.jobs < 1:
        # Do not import multiprocessing globally in case it is not supported
        # on the platform.
        import multiprocessing
        args.jobs = multiprocessing.cpu_count()

    if args.jobs > 1 and not args.in_place:
        parser.error('parallel jobs requires --in-place')

    if args.line_range:
        if args.line_range[0] <= 0:
            parser.error('--range must be positive numbers')
        if args.line_range[0] > args.line_range[1]:
            parser.error('First value of --range should be less than or equal '
                         'to the second')

    return args

def read_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Read both user configuration and local configuration."""
    try:
        from configparser import ConfigParser as SafeConfigParser
        from configparser import Error
    except ImportError:
        from ConfigParser import SafeConfigParser
        from ConfigParser import Error

    config: SafeConfigParser = SafeConfigParser()

    try:
        config.read(args.global_config)

        if not args.ignore_local_config:
            parent: str = tail: str = args.files and os.path.abspath(
                os.path.commonprefix(args.files))
            while tail:
                if config.read([os.path.join(parent, fn)
                                for fn in PROJECT_CONFIG]):
                    break
                (parent, tail) = os.path.split(parent)

        defaults: Dict[str, Any] = dict()
        option_list: Dict[str, Union[type, type]] = dict([(o.dest, o.type or type(o.default))
                                for o in parser._actions])

        for section in ['pep8', 'pycodestyle']:
            if not config.has_section(section):
                continue
            for (k, _) in config.items(section):
                norm_opt: str = k.lstrip('-').replace('-', '_')
                opt_type: Union[type, type] = option_list.get(norm_opt, str)
                if opt_type is int:
                    value: int = config.getint(section, k)
                elif opt_type is bool:
                    value: bool = config.getboolean(section, k)
                else:
                    value: str = config.get(section, k)
                defaults[norm_opt] = value

        parser.set_defaults(**defaults)
    except Error:
        # Ignore for now.
        pass

    return parser

def _split_comma_separated(string: str) -> Set[str]:
    """Return a set of strings."""
    return set(text.strip() for text in string.split(',') if text.strip())

def decode_filename(filename: Union[str, bytes]) -> str:
    """Return Unicode filename."""
    if isinstance(filename, str):
        return filename

    return filename.decode(sys.getfilesystemencoding())

def supported_fixes() -> Iterator[Tuple[str, str]]:
    """Yield pep8 error codes that autopep8 fixes.

    Each item we yield is a tuple of the code followed by its
    description.

    """
    yield ('E101', docstring_summary(reindent.__doc__))

    instance: FixPEP8 = FixPEP8(filename=None, options=None, contents='')
    for attribute in dir(instance):
        code_match_obj: Optional[re.Match] = re.match('fix_([ew][0-9][0-9][0-9])', attribute)
        if code_match_obj:
            code: str = code_match_obj.group(1).upper()
            description: str = re.sub(r'\s+', ' ',
                                       docstring_summary(getattr(instance, attribute).__doc__))
            yield (code.ljust(4), description)

    for (code, function) in sorted(global_fixes()):
        yield (code.upper().ljust(4),
               re.sub(r'\s+', ' ', docstring_summary(function.__doc__)))

    for code in sorted(CODE_TO_2TO3):
        yield (code.upper().ljust(4),
               re.sub(r'\s+', ' ', docstring_summary(fix_2to3.__doc__)))

def docstring_summary(docstring: Optional[str]) -> str:
    """Return summary of docstring."""
    return docstring.split('\n')[0] if docstring else ''

def filter_results(source: str, results: List[Dict[str, Any]], aggressive: int) -> Iterator[Dict[str, Any]]:
    """Filter out spurious reports from pycodestyle.

    If aggressive is True, we allow possibly unsafe fixes (E711, E712).

    """
    non_docstring_string_line_numbers: Set[int] = multiline_string_lines(
        source,
        include_docstrings=False
    )
    all_string_line_numbers: Set[int] = multiline_string_lines(
        source,
        include_docstrings=True
    )

    commented_out_code_line_numbers: Set[int] = set(commented_out_code_lines(source))

    has_e901: bool = any(result['id'].lower() == 'e901' for result in results)

    for r in results:
        issue_id: str = r['id'].lower()

        if r['line'] in non_docstring_string_line_numbers:
            if issue_id.startswith(('e1', 'e501', 'w191')):
                continue

        if r['line'] in all_string_line_numbers:
            if issue_id in ['e501']:
                continue

        # We must offset by 1 for lines that contain the trailing contents of
        # multiline strings.
        if not aggressive and (r['line'] + 1) in all_string_line_numbers:
            # Do not modify multiline strings in non-aggressive mode. Remove
            # trailing whitespace could break doctests.
            if issue_id.startswith(('w29', 'w39')):
                continue

        if aggressive <= 0:
            if issue_id.startswith(('e711', 'e72', 'w6')):
                continue

        if aggressive <= 1:
            if issue_id.startswith(('e712', 'e713', 'e714', 'w5')):
                continue

        if aggressive <= 2:
            if issue_id.startswith(('e704', 'w5')):
                continue

        if r['line'] in commented_out_code_line_numbers:
            if issue_id.startswith(('e26', 'e501')):
                continue

        # Do not touch indentation if there is a token error caused by
        # incomplete multi-line statement. Otherwise, we risk screwing up the
        # indentation.
        if has_e901:
            if issue_id.startswith(('e1', 'e7')):
                continue

        yield r

def multiline_string_lines(source: str, include_docstrings: bool = False) -> Set[int]:
    """Return line numbers that are within multiline strings.

    The line numbers are indexed at 1.

    Docstrings are ignored.

    """
    line_numbers: Set[int] = set()
    previous_token_type: int = 0
    try:
        for t in generate_tokens(source):
            token_type: int = t.type
            start_row: int = t.start[0]
            end_row: int = t.end[0]

            if token_type == tokenize.STRING and start_row != end_row:
                if (
                    include_docstrings or
                    previous_token_type != tokenize.INDENT
                ):
                    # We increment by one since we want the contents of the
                    # string.
                    line_numbers |= set(range(1 + start_row, 1 + end_row))

            previous_token_type = token_type
    except (SyntaxError, tokenize.TokenError):
        pass

    return line_numbers

def commented_out_code_lines(source: str) -> List[int]:
    """Return line numbers of comments that are likely code.

    Commented-out code is bad practice, but modifying it just adds even
    more clutter.

    """
    line_numbers: List[int] = []
    try:
        for t in generate_tokens(source):
            token_type: int = t.type
            token_string: str = t.string
            start_row: int = t.start[0]
            line: str = t.line

            # Ignore inline comments.
            if not line.lstrip().startswith('#'):
                continue

            if token_type == tokenize.COMMENT:
                stripped_line: str = token_string.lstrip('#').strip()
                if (
                    ' ' in stripped_line and
                    '#' not in stripped_line and
                    check_syntax(stripped_line)
                ):
                    line_numbers.append(start_row)
    except (SyntaxError, tokenize.TokenError):
        pass

    return line_numbers

def shorten_comment(line: str, max_line_length: int, last_comment: bool = False) -> List[str]:
    """Return trimmed or split long comment line.

    If there are no comments immediately following it, do a text wrap.
    Doing this wrapping on all comments in general would lead to jagged
    comment text.

    """
    assert len(line) > max_line_length
    line = line.rstrip()

    # PEP 8 recommends 72 characters for comment text.
    indentation: str = _get_indentation(line) + '# '
    max_line_length = min(max_line_length,
                          len(indentation) + 72)

    MIN_CHARACTER_REPEAT: int = 5
    if (
        len(line) - len(line.rstrip(line[-1])) >= MIN_CHARACTER_REPEAT and
        not line[-1].isalnum()
    ):
        # Trim comments that end with things like ---------
        return [line[:max_line_length] + '\n']
    elif last_comment and re.match(r'\s*#+\s*\w+', line):
        split_lines: List[str] = textwrap.wrap(line.lstrip(' \t#'),
                                        initial_indent=indentation,
                                        subsequent_indent=indentation,
                                        width=max_line_length,
                                        break_long_words=False,
                                        break_on_hyphens=False)
        return [s + '\n' for s in split_lines]
    
    return [line + '\n']

def normalize_multiline(line: str) -> str:
    """Normalize multiline-related code that will cause syntax error.

    This is for purposes of checking syntax.

    """
    if line.startswith('def ') and line.rstrip().endswith(':'):
        return line + ' pass'
    elif line.startswith('return '):
        return 'def _(): ' + line
    elif line.startswith('@'):
        return line + 'def _(): pass'
    elif line.startswith('class '):
        return line + ' pass'
    elif line.startswith(('if ', 'elif ', 'for ', 'while ')):
        return line + ' pass'

    return line

def fix_whitespace(line: str, offset: int, replacement: str) -> str:
    """Replace whitespace at offset and return fixed line."""
    # Replace escaped newlines too
    left: str = line[:offset].rstrip('\n\r \t\\')
    right: str = line[offset:].lstrip('\n\r \t\\')
    if right.startswith('#'):
        return line

    return left + replacement + right

def _execute_pep8(pep8_options: Dict[str, Any], source: List[str]) -> List[Dict[str, Any]]:
    """Execute pycodestyle via python method calls."""
    class QuietReport(pycodestyle.BaseReport):

        """Version of checker that does not print."""

        def __init__(self, options: Dict[str, Any]):
            super().__init__(options)
            self.__full_error_results: List[Dict[str, Any]] = []

        def error(self, line_number: int, offset: int, text: str, check: Callable) -> Optional[str]:
            """Collect errors."""
            code: Optional[str] = super().error(line_number,
                                                offset,
                                                text,
                                                check)
            if code:
                self.__full_error_results.append(
                    {'id': code,
                     'line': line_number,
                     'column': offset + 1,
                     'info': text})
            return code

        def full_error_results(self) -> List[Dict[str, Any]]:
            """Return error results in detail.

            Results are in the form of a list of dictionaries. Each
            dictionary contains 'id', 'line', 'column', and 'info'.

            """
            return self.__full_error_results

    checker: pycodestyle.Checker = pycodestyle.Checker('', lines=source, reporter=QuietReport,
                                      **pep8_options)
    checker.check_all()
    return checker.report.full_error_results()

def _get_package_version() -> str:
    packages: List[str] = [f"pycodestyle: {pycodestyle.__version__}"]
    return ", ".join(packages)

def get_fixed_long_line(target: str, previous_line: str, original: str,
                       indent_word: str = '    ', max_line_length: int = 79,
                       aggressive: bool = False, experimental: bool = False, verbose: int = 0) -> Optional[str]:
    """Break up long line and return result.

    Do this by generating multiple reformatted candidates and then
    ranking the candidates to heuristically select the best option.

    """
    indent: str = _get_indentation(target)
    source: str = target[len(indent):]
    assert source.lstrip() == source
    assert not target.lstrip().startswith('#')

    # Check for partial multiline.
    tokens: List[tokenize.TokenInfo] = list(generate_tokens(source))

    candidates: Iterator[str] = shorten_line(
        tokens=tokens,
        source=source,
        indentation=indent,
        indent_word=indent_word,
        max_line_length=max_line_length,
        aggressive=aggressive,
        experimental=experimental,
        previous_line=previous_line
    )

    # Also sort alphabetically as a tie breaker (for determinism).
    candidates = sorted(
        sorted(set(candidates).union([target, original])),
        key=lambda x: line_shortening_rank(
            x,
            indent_word,
            max_line_length,
            experimental=experimental)
    )

    if verbose >= 4:
        print(('-' * 79 + '\n').join([''] + list(candidates) + ['']),
              file=wrap_output(sys.stderr, 'utf-8'))

    if candidates:
        best_candidate: str = candidates[0]

        # Don't allow things to get longer.
        if longest_line_length(best_candidate) > longest_line_length(original):
            return None

        return best_candidate

    return None

def join_logical_line(logical_line: str) -> str:
    """Return single line based on logical line input."""
    indentation: str = _get_indentation(logical_line)

    return indentation + untokenize_without_newlines(
        generate_tokens(logical_line.lstrip())) + '\n'

def find_newline(source: List[str]) -> str:
    """Return type of newline used in source.

    Input is a list of lines.

    """
    assert not isinstance(source, unicode)

    counter: collections.defaultdict = collections.defaultdict(int)
    for line in source:
        if line.endswith(CRLF):
            counter[CRLF] += 1
        elif line.endswith(CR):
            counter[CR] += 1
        elif line.endswith(LF):
            counter[LF] += 1

    return (sorted(counter, key=counter.get, reverse=True) or [LF])[0]

def _get_indentword(source: str) -> str:
    """Return indentation type."""
    indent_word: str = '    '  # Default in case source has no indentation
    try:
        for t in generate_tokens(source):
            if t[0] == token.INDENT:
                indent_word = t[1]
                break
    except (SyntaxError, tokenize.TokenError):
        pass
    return indent_word

def _get_indentation(line: str) -> str:
    """Return leading whitespace."""
    if line.strip():
        non_whitespace_index: int = len(line) - len(line.lstrip())
        return line[:non_whitespace_index]

    return ''

def get_diff_text(old: List[str], new: List[str], filename: str) -> str:
    """Return text of unified diff between old and new."""
    newline: str = '\n'
    diff: Iterator[str] = difflib.unified_diff(
        old, new,
        'original/' + filename,
        'fixed/' + filename,
        lineterm=newline
    )

    text: str = ''
    for line in diff:
        text += line

        # Work around missing newline (http://bugs.python.org/issue2142).
        if text and not line.endswith(newline):
            text += newline + r'\ No newline at end of file' + newline

    return text

def _priority_key(pep8_result: Dict[str, Any]) -> int:
    """Key for sorting PEP8 results.

    Global fixes should be done first. This is important for things like
    indentation.

    """
    priority: List[str] = [
        # Fix multiline colon-based before semicolon based.
        'e701',
        # Break multiline statements early.
        'e702',
        # Things that make lines longer.
        'e225', 'e231',
        # Remove extraneous whitespace before breaking lines.
        'e201',
        # Shorten whitespace in comment before resorting to wrapping.
        'e262'
    ]
    middle_index: int = 10000
    lowest_priority: List[str] = [
        # We need to shorten lines last since the logical fixer can get in a
        # loop, which causes us to exit early.
        'e501',
        'w503'
    ]
    key: str = pep8_result['id'].lower()
    try:
        return priority.index(key)
    except ValueError:
        try:
            return middle_index + lowest_priority.index(key) + 1
        except ValueError:
            return middle_index

def shorten_line(tokens: List[tokenize.TokenInfo], source: str, indentation: str, indent_word: str, max_line_length: int,
                 aggressive: bool = False, experimental: bool = False, previous_line: str = '') -> Iterator[str]:
    """Separate line at OPERATOR.

    Multiple candidates will be yielded.

    """
    for candidate in _shorten_line(tokens=tokens,
                                   source=source,
                                   indentation=indentation,
                                   indent_word=indent_word,
                                   aggressive=aggressive,
                                   previous_line=previous_line):
        yield candidate

    if aggressive:
        for key_token_strings in SHORTEN_OPERATOR_GROUPS:
            shortened: Optional[str] = _shorten_line_at_tokens(
                tokens=tokens,
                source=source,
                indentation=indentation,
                indent_word=indent_word,
                key_token_strings=key_token_strings,
                aggressive=aggressive
            )

            if shortened is not None and shortened != source:
                yield shortened

    if experimental:
        for shortened in _shorten_line_at_tokens_new(
                tokens=tokens,
                source=source,
                indentation=indentation,
                max_line_length=max_line_length):

            yield shortened

def generate_tokens(text: str) -> Iterator[tokenize.TokenInfo]:
    """Generate tokens with caching."""
    return _cached_tokenizer.generate_tokens(text)

def _shorten_line_at_tokens_new(tokens: List[tokenize.TokenInfo], source: str, indentation: str,
                                max_line_length: int) -> Optional[str]:
    """Shorten the line taking its length into account.

    The input is expected to be free of newlines except for inside
    multiline strings and at the end.

    """
    # Yield the original source so to see if it's a better choice than the
    # shortened candidate lines we generate here.
    yield indentation + source

    parsed_tokens: Optional[List[Union[Atom, Container]]] = _parse_tokens(tokens)

    if parsed_tokens:
        # Perform two reflows. The first one starts on the same line as the
        # prefix. The second starts on the line after the prefix.
        fixed: Optional[str] = _reflow_lines(parsed_tokens, indentation, max_line_length,
                                            start_on_prefix_line=True)
        if fixed and check_syntax(normalize_multiline(fixed.lstrip())):
            yield fixed

        fixed = _reflow_lines(parsed_tokens, indentation, max_line_length,
                              start_on_prefix_line=False)
        if fixed and check_syntax(normalize_multiline(fixed.lstrip())):
            yield fixed

def _shorten_line_at_tokens(tokens: List[tokenize.TokenInfo], source: str, indentation: str, indent_word: str,
                           key_token_strings: FrozenSet[str], aggressive: bool) -> Optional[str]:
    """Separate line by breaking at tokens in key_token_strings.

    The input is expected to be free of newlines except for inside
    multiline strings and at the end.

    """
    offsets: List[int] = []
    for (index, _) in enumerate(token_offsets(tokens)):
        (token_type,
         token_string,
         start_offset,
         end_offset) = _

        if token_string in key_token_strings:
            # Do not break in containers with zero or one items.
            unwanted_next_token: Optional[str] = {
                '(': ')',
                '[': ']',
                '{': '}'
            }.get(token_string)
            if unwanted_next_token:
                if (
                    get_item(tokens,
                             index + 1,
                             default=[None, None])[1] == unwanted_next_token or
                    get_item(tokens,
                             index + 2,
                             default=[None, None])[1] == unwanted_next_token
                ):
                    continue

            if (
                index > 2 and token_string == '(' and
                tokens[index - 1].string in ',(%['
            ):
                # Don't split after a tuple start, or before a tuple start if
                # the tuple is in a list.
                continue

            if (
                end_offset < len(source) - 1
            ):
                # Don't split right before newline.
                offsets.append(end_offset)
        else:
            # Break at adjacent strings. These were probably meant to be on
            # separate lines in the first place.
            previous_token: Optional[tokenize.TokenInfo] = get_item(tokens, index - 1)
            if (
                token_type == tokenize.STRING and
                previous_token and previous_token.type == tokenize.STRING
            ):
                offsets.append(start_offset)

    current_indent: Optional[int] = None
    fixed: Optional[str] = None
    for line in split_at_offsets(source, offsets):
        if fixed:
            fixed += '\n' + current_indent + line

            for symbol in '([{':
                if line.endswith(symbol):
                    current_indent += indent_word
        else:
            # First line.
            fixed = line
            assert not current_indent
            current_indent = indent_word

    assert fixed is not None

    if check_syntax(normalize_multiline(fixed.lstrip()) if aggressive > 1 else fixed):
        return indentation + fixed

    return None

def token_offsets(tokens: List[tokenize.TokenInfo]) -> Iterator[Tuple[int, str, int, int]]:
    """Yield tokens and offsets."""
    end_offset: int = 0
    previous_end_row: int = 0
    previous_end_column: int = 0
    for t in tokens:
        token_type: int = t.type
        token_string: str = t.string
        (start_row, start_column) = t.start
        (end_row, end_column) = t.end

        # Account for the whitespace between tokens.
        end_offset += start_column
        if previous_end_row == start_row:
            end_offset -= previous_end_column

        # Record the start offset of the token.
        start_offset: int = end_offset

        # Account for the length of the token itself.
        end_offset += len(token_string)

        yield (token_type,
               token_string,
               start_offset,
               end_offset)

        previous_end_row = end_row
        previous_end_column = end_column

def _shorten_line_at_tokens_new(tokens: List[tokenize.TokenInfo], source: str, indentation: str,
                                max_line_length: int) -> Optional[str]:
    """Shorten the line taking its length into account.

    The input is expected to be free of newlines except for inside
    multiline strings and at the end.

    """
    # Yield the original source so to see if it's a better choice than the
    # shortened candidate lines we generate here.
    yield indentation + source

    parsed_tokens: Optional[List[Union[Atom, Container]]] = _parse_tokens(tokens)

    if parsed_tokens:
        # Perform two reflows. The first one starts on the same line as the
        # prefix. The second starts on the line after the prefix.
        fixed: Optional[str] = _reflow_lines(parsed_tokens, indentation, max_line_length,
                                            start_on_prefix_line=True)
        if fixed and check_syntax(normalize_multiline(fixed.lstrip())):
            yield fixed

        fixed = _reflow_lines(parsed_tokens, indentation, max_line_length,
                              start_on_prefix_line=False)
        if fixed and check_syntax(normalize_multiline(fixed.lstrip())):
            yield fixed

def longest_line_length(code: str) -> int:
    """Return length of longest line."""
    return max(len(line) for line in code.splitlines())

def normalize_multiline(line: str) -> str:
    """Normalize multiline-related code that will cause syntax error.

    This is for purposes of checking syntax.

    """
    if line.startswith('def ') and line.rstrip().endswith(':'):
        return line + ' pass'
    elif line.startswith('return '):
        return 'def _(): ' + line
    elif line.startswith('@'):
        return line + 'def _(): pass'
    elif line.startswith('class '):
        return line + ' pass'
    elif line.startswith(('if ', 'elif ', 'for ', 'while ')):
        return line + ' pass'

    return line

def check_syntax(code: str) -> bool:
    """Return True if syntax is okay."""
    try:
        compile(code, '<string>', 'exec')
        return True
    except (SyntaxError, TypeError, UnicodeDecodeError):
        return False

def filter_results(source: str, results: List[Dict[str, Any]], aggressive: int) -> Iterator[Dict[str, Any]]:
    """Filter out spurious reports from pycodestyle.

    If aggressive is True, we allow possibly unsafe fixes (E711, E712).

    """
    non_docstring_string_line_numbers: Set[int] = multiline_string_lines(
        source,
        include_docstrings=False
    )
    all_string_line_numbers: Set[int] = multiline_string_lines(
        source,
        include_docstrings=True
    )

    commented_out_code_line_numbers: Set[int] = set(commented_out_code_lines(source))

    has_e901: bool = any(result['id'].lower() == 'e901' for result in results)

    for r in results:
        issue_id: str = r['id'].lower()

        if r['line'] in non_docstring_string_line_numbers:
            if issue_id.startswith(('e1', 'e501', 'w191')):
                continue

        if r['line'] in all_string_line_numbers:
            if issue_id in ['e501']:
                continue

        # We must offset by 1 for lines that contain the trailing contents of
        # multiline strings.
        if not aggressive and (r['line'] + 1) in all_string_line_numbers:
            # Do not modify multiline strings in non-aggressive mode. Remove
            # trailing whitespace could break doctests.
            if issue_id.startswith(('w29', 'w39')):
                continue

        if aggressive <= 0:
            if issue_id.startswith(('e711', 'e72', 'w6')):
                continue

        if aggressive <= 1:
            if issue_id.startswith(('e712', 'e713', 'e714', 'w5')):
                continue

        if aggressive <= 2:
            if issue_id.startswith(('e704', 'w5')):
                continue

        if r['line'] in commented_out_code_line_numbers:
            if issue_id.startswith(('e26', 'e501')):
                continue

        # Do not touch indentation if there is a token error caused by
        # incomplete multi-line statement. Otherwise, we risk screwing up the
        # indentation.
        if has_e901:
            if issue_id.startswith(('e1', 'e7')):
                continue

        yield r

def multiline_string_lines(source: str, include_docstrings: bool = False) -> Set[int]:
    """Return line numbers that are within multiline strings.

    The line numbers are indexed at 1.

    Docstrings are ignored.

    """
    line_numbers: Set[int] = set()
    previous_token_type: int = 0
    try:
        for t in generate_tokens(source):
            token_type: int = t.type
            start_row: int = t.start[0]
            end_row: int = t.end[0]

            if token_type == tokenize.STRING and start_row != end_row:
                if (
                    include_docstrings or
                    previous_token_type != tokenize.INDENT
                ):
                    # We increment by one since we want the contents of the
                    # string.
                    line_numbers |= set(range(1 + start_row, 1 + end_row))

            previous_token_type = token_type
    except (SyntaxError, tokenize.TokenError):
        pass

    return line_numbers

def commented_out_code_lines(source: str) -> List[int]:
    """Return line numbers of comments that are likely code.

    Commented-out code is bad practice, but modifying it just adds even
    more clutter.

    """
    line_numbers: List[int] = []
    try:
        for t in generate_tokens(source):
            token_type: int = t.type
            token_string: str = t.string
            start_row: int = t.start[0]
            line: str = t.line

            # Ignore inline comments.
            if not line.lstrip().startswith('#'):
                continue

            if token_type == tokenize.COMMENT:
                stripped_line: str = token_string.lstrip('#').strip()
                if (
                    ' ' in stripped_line and
                    '#' not in stripped_line and
                    check_syntax(stripped_line)
                ):
                    line_numbers.append(start_row)
    except (SyntaxError, tokenize.TokenError):
        pass

    return line_numbers

def shorten_comment(line: str, max_line_length: int, last_comment: bool = False) -> List[str]:
    """Return trimmed or split long comment line.

    If there are no comments immediately following it, do a text wrap.
    Doing this wrapping on all comments in general would lead to jagged
    comment text.

    """
    assert len(line) > max_line_length
    line = line.rstrip()

    # PEP 8 recommends 72 characters for comment text.
    indentation: str = _get_indentation(line) + '# '
    max_line_length: int = min(max_line_length,
                              len(indentation) + 72)

    MIN_CHARACTER_REPEAT: int = 5
    if (
        len(line) - len(line.rstrip(line[-1])) >= MIN_CHARACTER_REPEAT and
        not line[-1].isalnum()
    ):
        # Trim comments that end with things like ---------
        return [line[:max_line_length] + '\n']
    elif last_comment and re.match(r'\s*#+\s*\w+', line):
        split_lines: List[str] = textwrap.wrap(line.lstrip(' \t#'),
                                        initial_indent=indentation,
                                        subsequent_indent=indentation,
                                        width=max_line_length,
                                        break_long_words=False,
                                        break_on_hyphens=False)
        return [s + '\n' for s in split_lines]
    
    return [line + '\n']

def refactor_with_2to3(source_text: str, fixer_names: List[str], filename: str = '') -> str:
    """Use lib2to3 to refactor the source.

    Return the refactored source code.

    """
    from lib2to3.refactor import RefactoringTool
    fixers: List[str] = ['lib2to3.fixes.fix_' + name for name in fixer_names]
    tool: RefactoringTool = RefactoringTool(fixer_names=fixers, explicit=fixers)

    from lib2to3.pgen2 import tokenize as lib2to3_tokenize
    try:
        # The name parameter is necessary particularly for the "import" fixer.
        return unicode(tool.refactor_string(source_text, name=filename))
    except lib2to3_tokenize.TokenError:
        return source_text

def check_syntax(code: str) -> bool:
    """Return True if syntax is okay."""
    try:
        compile(code, '<string>', 'exec')
        return True
    except (SyntaxError, TypeError, UnicodeDecodeError):
        return False

def filter_results(source: str, results: List[Dict[str, Any]], aggressive: int) -> Iterator[Dict[str, Any]]:
    """Filter out spurious reports from pycodestyle.

    If aggressive is True, we allow possibly unsafe fixes (E711, E712).

    """
    non_docstring_string_line_numbers: Set[int] = multiline_string_lines(
        source,
        include_docstrings=False
    )
    all_string_line_numbers: Set[int] = multiline_string_lines(
        source,
        include_docstrings=True
    )

    commented_out_code_line_numbers: Set[int] = set(commented_out_code_lines(source))

    has_e901: bool = any(result['id'].lower() == 'e901' for result in results)

    for r in results:
        issue_id: str = r['id'].lower()

        if r['line'] in non_docstring_string_line_numbers:
            if issue_id.startswith(('e1', 'e501', 'w191')):
                continue

        if r['line'] in all_string_line_numbers:
            if issue_id in ['e501']:
                continue

        # We must offset by 1 for lines that contain the trailing contents of
        # multiline strings.
        if not aggressive and (r['line'] + 1) in all_string_line_numbers:
            # Do not modify multiline strings in non-aggressive mode. Remove
            # trailing whitespace could break doctests.
            if issue_id.startswith(('w29', 'w39')):
                continue

        if aggressive <= 0:
            if issue_id.startswith(('e711', 'e72', 'w6')):
                continue

        if aggressive <= 1:
            if issue_id.startswith(('e712', 'e713', 'e714', 'w5')):
                continue

        if aggressive <= 2:
            if issue_id.startswith(('e704', 'w5')):
                continue

        if r['line'] in commented_out_code_line_numbers:
            if issue_id.startswith(('e26', 'e501')):
                continue

        # Do not touch indentation if there is a token error caused by
        # incomplete multi-line statement. Otherwise, we risk screwing up the
        # indentation.
        if has_e901:
            if issue_id.startswith(('e1', 'e7')):
                continue

        yield r

def refactor(source: str, fixer_names: List[str], ignore: Optional[str] = None, filename: str = '') -> str:
    """Return refactored code using lib2to3.

    Skip if ignore string is produced in the refactored code.

    """
    from lib2to3 import pgen2
    try:
        new_text: str = refactor_with_2to3(source,
                                          fixer_names=fixer_names,
                                          filename=filename)
    except (pgen2.parse.ParseError,
            SyntaxError,
            UnicodeDecodeError,
            UnicodeEncodeError):
        return source

    if ignore:
        if ignore in new_text and ignore not in source:
            return source

    return new_text

def code_match(code: str, select: Set[str], ignore: Set[str]) -> bool:
    """Return True if code matches select/ignore criteria."""
    if ignore:
        assert not isinstance(ignore, unicode)
        for ignored_code in [c.strip() for c in ignore]:
            if mutual_startswith(code.lower(), ignored_code.lower()):
                return False

    if select:
        assert not isinstance(select, unicode)
        for selected_code in [c.strip() for c in select]:
            if mutual_startswith(code.lower(), selected_code.lower()):
                return True
        return False

    return True

def mutual_startswith(a: str, b: str) -> bool:
    return b.startswith(a) or a.startswith(b)

def code_to_2to3(select: Set[str], ignore: str) -> Set[str]:
    """Return set of 2to3 fix names based on PEP8 codes."""
    fixes: Set[str] = set()
    for code, fix in CODE_TO_2TO3.items():
        if code_match(code, select=select, ignore=ignore):
            fixes |= set(fix)
    return fixes

def fix_code(source: Union[str, bytes], options: Optional[Any] = None, encoding: Optional[str] = None, apply_config: bool = False) -> str:
    """Return fixed source code.

    "encoding" will be used to decode "source" if it is a byte string.

    """
    options = _get_options(options, apply_config)

    if not isinstance(source, str):
        source = source.decode(encoding or get_encoding())

    sio: io.StringIO = io.StringIO(source)
    return fix_lines(sio.readlines(), options=options)

def _get_options(raw_options: Optional[Any], apply_config: bool) -> Any:
    """Return parsed options."""
    if not raw_options:
        return parse_args([''], apply_config=apply_config)

    if isinstance(raw_options, dict):
        options: argparse.Namespace = parse_args([''], apply_config=apply_config)
        for name, value in raw_options.items():
            if not hasattr(options, name):
                raise ValueError(f"No such option '{name}'")

            # Check for very basic type errors.
            expected_type: type = type(getattr(options, name))
            if not isinstance(expected_type, (str, type)):
                if isinstance(value, (str, bytes)):
                    raise ValueError(
                        f"Option '{name}' should not be a string"
                    )
            setattr(options, name, value)
    else:
        options = raw_options

    return options

def fix_lines(source_lines: List[str], options: Any, filename: str = '') -> str:
    """Return fixed source code."""
    # Transform everything to line feed. Then change them back to original
    # before returning fixed source code.
    original_newline: str = find_newline(source_lines)
    tmp_source: str = ''.join(normalize_line_endings(source_lines, '\n'))

    # Keep a history to break out of cycles.
    previous_hashes: Set[int] = set()

    if options.line_range:
        # Disable "apply_local_fixes()" for now due to issue #175.
        fixed_source: str = tmp_source
    else:
        # Apply global fixes only once (for efficiency).
        fixed_source: str = apply_global_fixes(tmp_source,
                                              options,
                                              filename=filename)

    passes: int = 0
    long_line_ignore_cache: Set[Tuple[str, Tuple[str, str, str, str]]] = set()
    while hash(fixed_source) not in previous_hashes:
        if options.pep8_passes >= 0 and passes > options.pep8_passes:
            break
        passes += 1

        previous_hashes.add(hash(fixed_source))

        tmp_source = copy.copy(fixed_source)

        fix: FixPEP8 = FixPEP8(
            filename,
            options,
            contents=tmp_source,
            long_line_ignore_cache=long_line_ignore_cache
        )

        fixed_source = fix.fix()

    sio: io.StringIO = io.StringIO(fixed_source)
    return ''.join(normalize_line_endings(sio.readlines(), original_newline))

def fix_file(filename: str, options: Optional[Any] = None, output: Optional[io.TextIOBase] = None, apply_config: bool = False) -> Optional[str]:
    """Fix a single file."""
    if not options:
        options = parse_args([filename], apply_config=apply_config)

    original_source: List[str] = readlines_from_file(filename)

    fixed_source: List[str] = original_source

    if options.in_place or output:
        encoding: str = detect_encoding(filename)

    if output:
        output = LineEndingWrapper(wrap_output(output, encoding=encoding))

    fixed_source = fix_lines(fixed_source, options, filename=filename)

    if options.diff:
        new: io.StringIO = io.StringIO(fixed_source)
        new = new.readlines()
        diff: str = get_diff_text(original_source, new, filename)
        if output:
            output.write(diff)
            output.flush()
        else:
            return diff
    elif options.in_place:
        with open_with_encoding(filename, encoding=encoding, mode='w') as fp:
            fp.write(fixed_source)
    else:
        if output:
            output.write(fixed_source)
            output.flush()
        else:
            return fixed_source

def global_fixes() -> Iterator[Tuple[str, Callable[..., Optional[List[int]]]]]:
    """Yield multiple (code, function) tuples."""
    for function in list(globals().values()):
        if inspect.isfunction(function):
            arguments: List[str] = _get_parameters(function)
            if arguments[:1] != ['source']:
                continue

            code: Optional[str] = extract_code_from_function(function)
            if code:
                yield (code, function)

def _get_parameters(function: Callable[..., Any]) -> List[str]:
    # pylint: disable=deprecated-method
    if sys.version_info >= (3, 3):
        # We need to match "getargspec()", which includes "self" as the first
        # value for methods.
        # https://bugs.python.org/issue17481#msg209469
        if inspect.ismethod(function):
            function = function.__func__

        return list(inspect.signature(function).parameters)
    else:
        return inspect.getargspec(function)[0]

def apply_global_fixes(source: str,
                       options: Any,
                       where: str = 'global',
                       filename: str = '') -> str:
    """Run global fixes on source code.

    These are fixes that only need be done once (unlike those in
    FixPEP8, which are dependent on pycodestyle).

    """
    if any(code_match(code, select=options.select, ignore=options.ignore)
           for code in ['E101', 'E111']):
        source = reindent(source,
                          indent_size=options.indent_size)

    for (code, function) in global_fixes():
        if code_match(code, select=options.select, ignore=options.ignore):
            if options.verbose:
                print(f'--->  Applying {where} fix for {code.upper()}', file=sys.stderr)
            source = function(source,
                              aggressive=options.aggressive)

    source = fix_2to3(source,
                      aggressive=options.aggressive,
                      select=options.select,
                      ignore=options.ignore,
                      filename=filename)

    return source

def extract_code_from_function(function: Callable[..., Any]) -> Optional[str]:
    """Return code handled by function."""
    if not function.__name__.startswith('fix_'):
        return None

    code: str = re.sub('^fix_', '', function.__name__)
    if not code:
        return None

    try:
        int(code[1:])
    except ValueError:
        return None

    return code

def get_fixed_long_line(target: str, previous_line: str, original: str,
                       indent_word: str = '    ', max_line_length: int = 79,
                       aggressive: bool = False, experimental: bool = False, verbose: int = 0) -> Optional[str]:
    """Break up long line and return result.

    Do this by generating multiple reformatted candidates and then
    ranking the candidates to heuristically select the best option.

    """
    indent: str = _get_indentation(target)
    source: str = target[len(indent):]
    assert source.lstrip() == source
    assert not target.lstrip().startswith('#')

    # Check for partial multiline.
    tokens: List[tokenize.TokenInfo] = list(generate_tokens(source))

    candidates: Iterator[str] = shorten_line(
        tokens=tokens,
        source=source,
        indentation=indent,
        indent_word=indent_word,
        max_line_length=max_line_length,
        aggressive=aggressive,
        experimental=experimental,
        previous_line=previous_line
    )

    # Also sort alphabetically as a tie breaker (for determinism).
    candidates = sorted(
        sorted(set(candidates).union([target, original])),
        key=lambda x: line_shortening_rank(
            x,
            indent_word,
            max_line_length,
            experimental=experimental)
    )

    if verbose >= 4:
        print(('-' * 79 + '\n').join([''] + list(candidates) + ['']),
              file=wrap_output(sys.stderr, 'utf-8'))

    if candidates:
        best_candidate: str = candidates[0]

        # Don't allow things to get longer.
        if longest_line_length(best_candidate) > longest_line_length(original):
            return None

        return best_candidate

    return None

def join_logical_line(logical_line: str) -> str:
    """Return single line based on logical line input."""
    indentation: str = _get_indentation(logical_line)

    return indentation + untokenize_without_newlines(
        generate_tokens(logical_line.lstrip())) + '\n'

class Reindenter(object):

    """Reindents badly-indented code to uniformly use four-space indentation.

    Released to the public domain, by Tim Peters, 03 October 2000.

    """

    def __init__(self, input_text: str):
        sio: io.StringIO = io.StringIO(input_text)
        source_lines: List[str] = sio.readlines()

        self.string_content_line_numbers: Set[int] = multiline_string_lines(input_text)

        # File lines, rstripped & tab-expanded. Dummy at start is so
        # that we can use tokenize's 1-based line numbering easily.
        # Note that a line is all-blank iff it is a newline.
        self.lines: List[str] = []
        for line_number, line in enumerate(source_lines, start=1):
            # Do not modify if inside a multiline string.
            if line_number in self.string_content_line_numbers:
                self.lines.append(line)
            else:
                # Only expand leading tabs.
                self.lines.append(_get_indentation(line).expandtabs() +
                                  _remove_leading_and_normalize(line))

        self.lines.insert(0, None)
        self.index: int = 1  # index into self.lines of next line
        self.input_text: str = input_text

    def run(self, indent_size: int = DEFAULT_INDENT_SIZE) -> str:
        """Fix indentation and return modified line numbers.

        Line numbers are indexed at 1.

        """
        if indent_size < 1:
            return self.input_text

        try:
            stats: List[Tuple[int, int]] = _reindent_stats(tokenize.generate_tokens(self.getline))
        except (SyntaxError, tokenize.TokenError):
            return self.input_text
        # Remove trailing empty lines.
        lines: List[str] = self.lines
        # Sentinel.
        stats.append((len(lines), 0))
        # Map count of leading spaces to # we want.
        have2want: Dict[int, int] = {}
        # Program after transformation.
        after: List[str] = []
        # Copy over initial empty lines -- there's nothing to do until
        # we see a line with *something* on it.
        i: int = stats[0][0]
        after.extend(lines[1:i])
        for i in range(len(stats) - 1):
            thisstmt, thislevel = stats[i]
            nextstmt: int = stats[i + 1][0]
            have: int = _leading_space_count(lines[thisstmt])
            want: int = thislevel * indent_size
            if want < 0:
                # A comment line.
                if have:
                    # An indented comment line. If we saw the same
                    # indentation before, reuse what it most recently
                    # mapped to.
                    want = have2want.get(have, -1)
                    if want < 0:
                        # Then it probably belongs to the next real stmt.
                        for j in range(i + 1, len(stats) - 1):
                            jline, jlevel = stats[j]
                            if jlevel >= 0:
                                if have == _leading_space_count(lines[jline]):
                                    want = jlevel * indent_size
                                break
                    if want < 0:
                        # Maybe it's a hanging
                        # comment like this one,
                        want = have + _leading_space_count(
                            after[jline - 1]) - _leading_space_count(lines[jline])

                    if want < 0:
                        # Still no luck -- leave it alone.
                        want = have
                else:
                    want = 0
            have2want[have] = want
            diff: int = want - have
            if diff == 0 or have == 0:
                after.extend(lines[thisstmt:nextstmt])
            else:
                for line_number, line in enumerate(lines[thisstmt:nextstmt],
                                                   start=thisstmt):
                    if line_number in self.string_content_line_numbers:
                        after.append(line)
                    elif diff > 0:
                        if line == '\n':
                            after.append(line)
                        else:
                            after.append(' ' * diff + line)
                    else:
                        remove: int = min(_leading_space_count(line), -diff)
                        after.append(line[remove:])

        return ''.join(after)

    def getline(self) -> str:
        """Line-getter for tokenize."""
        if self.index >= len(self.lines):
            line: str = ''
        else:
            line: str = self.lines[self.index]
            self.index += 1
        return line

def _reindent_stats(tokens: Iterator[tokenize.TokenInfo]) -> List[Tuple[int, int]]:
    """Return list of (lineno, indentlevel) pairs.

    One for each stmt and comment line. indentlevel is -1 for comment
    lines, as a signal that tokenize doesn't know what to do about them;
    indeed, they're our headache!

    """
    find_stmt: int = 1  # Next token begins a fresh stmt?
    level: int = 0  # Current indent level.
    stats: List[Tuple[int, int]] = []

    for t in tokens:
        token_type: int = t.type
        sline: int = t.start[0]
        line: str = t.line

        if token_type == tokenize.NEWLINE:
            # A program statement, or ENDMARKER, will eventually follow,
            # after some (possibly empty) run of tokens of the form
            #     (NL | COMMENT)* (INDENT | DEDENT+)?
            find_stmt = 1

        elif token_type == tokenize.INDENT:
            find_stmt = 1
            level += 1

        elif token_type == tokenize.DEDENT:
            find_stmt = 1
            level -= 1

        elif token_type == tokenize.COMMENT:
            if find_stmt:
                stats.append((sline, -1))
                # But we're still looking for a new stmt, so leave
                # find_stmt alone.

        elif token_type == tokenize.NL:
            pass

        elif find_stmt:
            # This is the first "real token" following a NEWLINE, so it
            # must be the first token of the next program statement, or an
            # ENDMARKER.
            find_stmt = 0
            if line:   # Not endmarker.
                stats.append((sline, level))

    return stats

def fix_code(source: Union[str, bytes], options: Optional[Any] = None, encoding: Optional[str] = None, apply_config: bool = False) -> str:
    """Return fixed source code.

    "encoding" will be used to decode "source" if it is a byte string.

    """
    options = _get_options(options, apply_config)

    if not isinstance(source, str):
        source = source.decode(encoding or get_encoding())

    sio: io.StringIO = io.StringIO(source)
    return fix_lines(sio.readlines(), options=options)

def create_parser() -> argparse.ArgumentParser:
    """Return command-line parser."""
    import argparse

    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=docstring_summary(__doc__),
                                     prog='autopep8')
    parser.add_argument('--version', action='version',
                        version=f'%(prog)s {__version__} ({_get_package_version()})')
    parser.add_argument('-v', '--verbose', action='count',
                        default=0,
                        help='print verbose messages; '
                             'multiple -v result in more verbose messages')
    parser.add_argument('-d', '--diff', action='store_true',
                        help='print the diff for the fixed source')
    parser.add_argument('-i', '--in-place', action='store_true',
                        help='make changes to files in place')
    parser.add_argument('--global-config', metavar='filename',
                        default=DEFAULT_CONFIG,
                        help='path to a global pep8 config file; if this file '
                             'does not exist then this is ignored '
                             f'(default: {DEFAULT_CONFIG})')
    parser.add_argument('--ignore-local-config', action='store_true',
                        help="don't look for and apply local config files; "
                             'if not passed, defaults are updated with any '
                             "config files in the project's root directory")
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='run recursively over directories; '
                             'must be used with --in-place or --diff')
    parser.add_argument('-j', '--jobs', type=int, metavar='n', default=1,
                        help='number of parallel jobs; '
                             'match CPU count if value is less than 1')
    parser.add_argument('-p', '--pep8-passes', metavar='n',
                        default=-1, type=int,
                        help='maximum number of additional pep8 passes '
                             '(default: infinite)')
    parser.add_argument('-a', '--aggressive', action='count', default=0,
                        help='enable non-whitespace changes; '
                             'multiple -a result in more aggressive changes')
    parser.add_argument('--experimental', action='store_true',
                        help='enable experimental fixes')
    parser.add_argument('--exclude', metavar='globs',
                        help='exclude file/directory names that match these '
                             'comma-separated globs')
    parser.add_argument('--list-fixes', action='store_true',
                        help='list codes for fixes; '
                        'used by --ignore and --select')
    parser.add_argument('--ignore', metavar='errors', default='',
                        help='do not fix these errors/warnings '
                             f'(default: {DEFAULT_IGNORE})')
    parser.add_argument('--select', metavar='errors', default='',
                        help='fix only these errors/warnings (e.g. E4,W)')
    parser.add_argument('--max-line-length', metavar='n', default=79, type=int,
                        help='set maximum allowed line length '
                             '(default: %(default)s)')
    parser.add_argument('--line-range', '--range', metavar='line',
                        default=None, type=int, nargs=2,
                        help='only fix errors found within this inclusive '
                             'range of line numbers (e.g. 1 99); '
                             'line numbers are indexed at 1')
    parser.add_argument('--indent-size', default=DEFAULT_INDENT_SIZE,
                        type=int, help=argparse.SUPPRESS)
    parser.add_argument('files', nargs='*',
                        help="files to format or '-' for standard in")

    return parser

def parse_args(arguments: List[str], apply_config: bool = True) -> argparse.Namespace:
    """Parse command-line options."""
    parser: argparse.ArgumentParser = create_parser()
    args: argparse.Namespace = parser.parse_args(arguments)

    if not args.files and not args.list_fixes:
        parser.error('incorrect number of arguments')

    args.files = [decode_filename(name) for name in args.files]

    if apply_config:
        parser = read_config(args, parser)
        args = parser.parse_args(arguments)
        args.files = [decode_filename(name) for name in args.files]

    if '-' in args.files:
        if len(args.files) > 1:
            parser.error('cannot mix stdin and regular files')

        if args.diff:
            parser.error('--diff cannot be used with standard input')

        if args.in_place:
            parser.error('--in-place cannot be used with standard input')

        if args.recursive:
            parser.error('--recursive cannot be used with standard input')

    if len(args.files) > 1 and not (args.in_place or args.diff):
        parser.error('autopep8 only takes one filename as argument '
                     'unless the "--in-place" or "--diff" args are '
                     'used')

    if args.recursive and not (args.in_place or args.diff):
        parser.error('--recursive must be used with --in-place or --diff')

    if args.in_place and args.diff:
        parser.error('--in-place and --diff are mutually exclusive')

    if args.max_line_length <= 0:
        parser.error('--max-line-length must be greater than 0')

    if args.select:
        args.select = _split_comma_separated(args.select)

    if args.ignore:
        args.ignore = _split_comma_separated(args.ignore)
    elif not args.select:
        if args.aggressive:
            # Enable everything by default if aggressive.
            args.select = set(['E', 'W'])
        else:
            args.ignore = _split_comma_separated(DEFAULT_IGNORE)

    if args.exclude:
        args.exclude = _split_comma_separated(args.exclude)
    else:
        args.exclude = set([])

    if args.jobs < 1:
        # Do not import multiprocessing globally in case it is not supported
        # on the platform.
        import multiprocessing
        args.jobs = multiprocessing.cpu_count()

    if args.jobs > 1 and not args.in_place:
        parser.error('parallel jobs requires --in-place')

    if args.line_range:
        if args.line_range[0] <= 0:
            parser.error('--range must be positive numbers')
        if args.line_range[0] > args.line_range[1]:
            parser.error('First value of --range should be less than or equal '
                         'to the second')

    return args

def read_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Read both user configuration and local configuration."""
    try:
        from configparser import ConfigParser as SafeConfigParser
        from configparser import Error
    except ImportError:
        from ConfigParser import SafeConfigParser
        from ConfigParser import Error

    config: SafeConfigParser = SafeConfigParser()

    try:
        config.read(args.global_config)

        if not args.ignore_local_config:
            parent: str = tail: str = args.files and os.path.abspath(
                os.path.commonprefix(args.files))
            while tail:
                if config.read([os.path.join(parent, fn)
                                for fn in PROJECT_CONFIG]):
                    break
                (parent, tail) = os.path.split(parent)

        defaults: Dict[str, Any] = dict()
        option_list: Dict[str, Union[type, type]] = dict([(o.dest, o.type or type(o.default))
                                for o in parser._actions])

        for section in ['pep8', 'pycodestyle']:
            if not config.has_section(section):
                continue
            for (k, _) in config.items(section):
                norm_opt: str = k.lstrip('-').replace('-', '_')
                opt_type: Union[type, type] = option_list.get(norm_opt, str)
                if opt_type is int:
                    value: int = config.getint(section, k)
                elif opt_type is bool:
                    value: bool = config.getboolean(section, k)
                else:
                    value: str = config.get(section, k)
                defaults[norm_opt] = value

        parser.set_defaults(**defaults)
    except Error:
        # Ignore for now.
        pass

    return parser

def _split_comma_separated(string: str) -> Set[str]:
    """Return a set of strings."""
    return set(text.strip() for text in string.split(',') if text.strip())

def decode_filename(filename: Union[str, bytes]) -> str:
    """Return Unicode filename."""
    if isinstance(filename, str):
        return filename

    return filename.decode(sys.getfilesystemencoding())

def supported_fixes() -> Iterator[Tuple[str, str]]:
    """Yield pep8 error codes that autopep8 fixes.

    Each item we yield is a tuple of the code followed by its
    description.

    """
    yield ('E101', docstring_summary(reindent.__doc__))

    instance: FixPEP8 = FixPEP8(filename=None, options=None, contents='')
    for attribute in dir(instance):
        code_match_obj: Optional[re.Match] = re.match('fix_([ew][0-9][0-9][0-9])', attribute)
        if code_match_obj:
            code: str = code_match_obj.group(1).upper()
            description: str = re.sub(r'\s+', ' ',
                                       docstring_summary(getattr(instance, attribute).__doc__))
            yield (code.ljust(4), description)

    for (code, function) in sorted(global_fixes()):
        yield (code.upper().ljust(4),
               re.sub(r'\s+', ' ', docstring_summary(function.__doc__)))

    for code in sorted(CODE_TO_2TO3):
        yield (code.upper().ljust(4),
               re.sub(r'\s+', ' ', docstring_summary(fix_2to3.__doc__)))

def fix_code(source: Union[str, bytes], options: Optional[Any] = None, encoding: Optional[str] = None, apply_config: bool = False) -> str:
    """Return fixed source code.

    "encoding" will be used to decode "source" if it is a byte string.

    """
    options = _get_options(options, apply_config)

    if not isinstance(source, str):
        source = source.decode(encoding or get_encoding())

    sio: io.StringIO = io.StringIO(source)
    return fix_lines(sio.readlines(), options=options)

def main(argv: Optional[List[str]] = None, apply_config: bool = True) -> int:
    """Command-line entry."""
    if argv is None:
        argv = sys.argv

    try:
        # Exit on broken pipe.
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except AttributeError:  # pragma: no cover
        # SIGPIPE is not available on Windows.
        pass

    try:
        args: argparse.Namespace = parse_args(argv[1:], apply_config=apply_config)

        if args.list_fixes:
            for code, description in sorted(supported_fixes()):
                print(f'{code} - {description}')
            return 0

        if args.files == ['-']:
            assert not args.in_place

            encoding: str = sys.stdin.encoding or get_encoding()

            # LineEndingWrapper is unnecessary here due to the symmetry between
            # standard in and standard out.
            wrap_output(sys.stdout, encoding=encoding).write(
                fix_code(sys.stdin.read(), args, encoding=encoding))
        else:
            if args.in_place or args.diff:
                args.files = list(set(args.files))
            else:
                assert len(args.files) == 1
                assert not args.recursive

            fix_multiple_files(args.files, args, sys.stdout)
    except KeyboardInterrupt:
        return 1  # pragma: no cover

    return 0

class CachedTokenizer(object):

    """A one-element cache around tokenize.generate_tokens().

    Original code written by Ned Batchelder, in coverage.py.

    """

    def __init__(self) -> None:
        self.last_text: Optional[str] = None
        self.last_tokens: Optional[List[tokenize.TokenInfo]] = None

    def generate_tokens(self, text: str) -> List[tokenize.TokenInfo]:
        """A stand-in for tokenize.generate_tokens()."""
        if text != self.last_text:
            string_io: io.StringIO = io.StringIO(text)
            self.last_tokens = list(
                tokenize.generate_tokens(string_io.readline)
            )
            self.last_text = text
        return self.last_tokens

_cached_tokenizer: CachedTokenizer = CachedTokenizer()
generate_tokens = _cached_tokenizer.generate_tokens

def main(argv: Optional[List[str]] = None, apply_config: bool = True) -> int:
    """Command-line entry."""
    if argv is None:
        argv = sys.argv

    try:
        # Exit on broken pipe.
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except AttributeError:  # pragma: no cover
        # SIGPIPE is not available on Windows.
        pass

    try:
        args: argparse.Namespace = parse_args(argv[1:], apply_config=apply_config)

        if args.list_fixes:
            for code, description in sorted(supported_fixes()):
                print(f'{code} - {description}')
            return 0

        if args.files == ['-']:
            assert not args.in_place

            encoding: str = sys.stdin.encoding or get_encoding()

            # LineEndingWrapper is unnecessary here due to the symmetry between
            # standard in and standard out.
            wrap_output(sys.stdout, encoding=encoding).write(
                fix_code(sys.stdin.read(), args, encoding=encoding))
        else:
            if args.in_place or args.diff:
                args.files = list(set(args.files))
            else:
                assert len(args.files) == 1
                assert not args.recursive

            fix_multiple_files(args.files, args, sys.stdout)
    except KeyboardInterrupt:
        return 1  # pragma: no cover

    return 0

if __name__ == '__main__':
    sys.exit(main())
