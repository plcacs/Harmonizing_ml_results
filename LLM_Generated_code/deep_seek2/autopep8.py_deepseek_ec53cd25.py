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


try:
    unicode
except NameError:
    unicode = str


__version__ = '1.3.2'


CR = '\r'
LF = '\n'
CRLF = '\r\n'


PYTHON_SHEBANG_REGEX = re.compile(r'^#!.*\bpython[23]?\b\s*$')
LAMBDA_REGEX = re.compile(r'([\w.]+)\s=\slambda\s*([\(\)\w,\s.]*):')
COMPARE_NEGATIVE_REGEX = re.compile(r'\b(not)\s+([^][)(}{]+)\s+(in|is)\s')
COMPARE_NEGATIVE_REGEX_THROUGH = re.compile(r'\b(not\s+in)\s')
BARE_EXCEPT_REGEX = re.compile(r'except\s*:')
STARTSWITH_DEF_REGEX = re.compile(r'^(async\s+def|def)\s.*\):')


# For generating line shortening candidates.
SHORTEN_OPERATOR_GROUPS = frozenset([
    frozenset([',']),
    frozenset(['%']),
    frozenset([',', '(', '[', '{']),
    frozenset(['%', '(', '[', '{']),
    frozenset([',', '(', '[', '{', '%', '+', '-', '*', '/', '//']),
    frozenset(['%', '+', '-', '*', '/', '//']),
])


DEFAULT_IGNORE = 'E24,W503'
DEFAULT_INDENT_SIZE = 4


# W602 is handled separately due to the need to avoid "with_traceback".
CODE_TO_2TO3 = {
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
             'xreadlines']}


if sys.platform == 'win32':  # pragma: no cover
    DEFAULT_CONFIG = os.path.expanduser(r'~\.pep8')
else:
    DEFAULT_CONFIG = os.path.join(os.getenv('XDG_CONFIG_HOME') or
                                  os.path.expanduser('~/.config'), 'pep8')
PROJECT_CONFIG = ('setup.cfg', 'tox.ini', '.pep8')


MAX_PYTHON_FILE_DETECTION_BYTES = 1024


def open_with_encoding(filename: str,
                       encoding: str = None, mode: str = 'r', limit_byte_check: int = -1) -> io.TextIOWrapper:
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
            encoding = lib2to3_tokenize.detect_encoding(input_file.readline)[0]

        with open_with_encoding(filename, encoding) as test_file:
            test_file.read(limit_byte_check)

        return encoding
    except (LookupError, SyntaxError, UnicodeDecodeError):
        return 'latin-1'


def readlines_from_file(filename: str) -> list[str]:
    """Return contents of file."""
    with open_with_encoding(filename) as input_file:
        return input_file.readlines()


def extended_blank_lines(logical_line: str,
                         blank_lines: int,
                         blank_before: int,
                         indent_level: int,
                         previous_logical: str) -> collections.Iterator[tuple[int, str]]:
    """Check for missing blank lines after class declaration."""
    if previous_logical.startswith('def '):
        if blank_lines and pycodestyle.DOCSTRING_REGEX.match(logical_line):
            yield (0, 'E303 too many blank lines ({0})'.format(blank_lines))
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


def continued_indentation(logical_line: str, tokens: list[tuple[int, str, tuple[int, int], tuple[int, int], str]], indent_level: int, indent_char: str,
                          noqa: bool) -> collections.Iterator[tuple[tuple[int, int], str]:
    """Override pycodestyle's function to provide indentation information."""
    first_row = tokens[0][2][0]
    nrows = 1 + tokens[-1][2][0] - first_row
    if noqa or nrows == 1:
        return

    # indent_next tells us whether the next block is indented. Assuming
    # that it is indented by 4 spaces, then we should not allow 4-space
    # indents on the final continuation line. In turn, some other
    # indents are allowed to have an extra 4 spaces.
    indent_next = logical_line.endswith(':')

    row = depth = 0
    valid_hangs = (
        (DEFAULT_INDENT_SIZE,)
        if indent_char != '\t' else (DEFAULT_INDENT_SIZE,
                                     2 * DEFAULT_INDENT_SIZE)
    )

    # Remember how many brackets were opened on each line.
    parens = [0] * nrows

    # Relative indents of physical lines.
    rel_indent = [0] * nrows

    # For each depth, collect a list of opening rows.
    open_rows = [[0]]
    # For each depth, memorize the hanging indentation.
    hangs = [None]

    # Visual indents.
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
            newline = (not last_token_multiline and
                       token_type not in (tokenize.NL, tokenize.NEWLINE))
            last_line_begins_with_multiline = last_token_multiline

        if newline:
            # This is the beginning of a continuation line.
            last_indent = start

            # Record the initial indent.
            rel_indent[row] = pycodestyle.expand_indent(line) - indent_level

            # Identify closing bracket.
            close_bracket = (token_type == tokenize.OP and text in ']})')

            # Is the indent relative to an opening bracket line?
            for open_row in reversed(open_rows[depth]):
                hang = rel_indent[row] - rel_indent[open_row]
                hanging_indent = hang in valid_hangs
                if hanging_indent:
                    break
            if hangs[depth]:
                hanging_indent = (hang == hangs[depth])

            visual_indent = (not close_bracket and hang > 0 and
                             indent_chances.get(start[1]))

            if close_bracket and indent[depth]:
                # Closing bracket for visual indent.
                if start[1] != indent[depth]:
                    yield (start, 'E124 {0}'.format(indent[depth]))
            elif close_bracket and not hang:
                pass
            elif indent[depth] and start[1] < indent[depth]:
                # Visual indent is broken.
                yield (start, 'E128 {0}'.format(indent[depth]))
            elif (hanging_indent or
                  (indent_next and
                   rel_indent[row] == 2 * DEFAULT_INDENT_SIZE)):
                # Hanging indent is verified.
                if close_bracket:
                    yield (start, 'E123 {0}'.format(indent_level +
                                                    rel_indent[open_row]))
                hangs[depth] = hang
            elif visual_indent is True:
                # Visual indent is verified.
                indent[depth] = start[1]
            elif visual_indent in (text, unicode):
                # Ignore token lined up with matching one from a previous line.
                pass
            else:
                one_indented = (indent_level + rel_indent[open_row] +
                                DEFAULT_INDENT_SIZE)
                # Indent is broken.
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
        pos = (start[0], indent[0] + 4)
        desired_indent = indent_level + 2 * DEFAULT_INDENT_SIZE
        if visual_indent:
            yield (pos, 'E129 {0}'.format(desired_indent))
        else:
            yield (pos, 'E125 {0}'.format(desired_indent))


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

    def __init__(self, filename: str,
                 options: argparse.Namespace,
                 contents: str = None,
                 long_line_ignore_cache: set = None):
        self.filename = filename
        if contents is None:
            self.source = readlines_from_file(filename)
        else:
            sio = io.StringIO(contents)
            self.source = sio.readlines()
        self.options = options
        self.indent_word = _get_indentword(''.join(self.source))

        self.long_line_ignore_cache = (
            set() if long_line_ignore_cache is None
            else long_line_ignore_cache)

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
            self.fix_long_line_physically)
        self.fix_e703 = self.fix_e702
        self.fix_w293 = self.fix_w291

    def _fix_source(self, results: list[dict]) -> None:
        try:
            (logical_start, logical_end) = _find_logical(self.source)
            logical_support = True
        except (SyntaxError, tokenize.TokenError):  # pragma: no cover
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
                        logical = _get_logical(self.source,
                                               result,
                                               logical_start,
                                               logical_end)
                        if logical and set(range(
                            logical[0][0] + 1,
                            logical[1][0] + 1)).intersection(
                                completed_lines):
                            continue

                    modified_lines = fix(result, logical)
                else:
                    modified_lines = fix(result)

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
                            '--->  Not fixing {error} on line {line}'.format(
                                error=result['id'], line=result['line']),
                            file=sys.stderr)
                else:  # We assume one-line fix when None.
                    completed_lines.add(result['line'])
            else:
                if self.options.verbose >= 3:
                    print(
                        "--->  '{0}' is not defined.".format(fixed_methodname),
                        file=sys.stderr)

                    info = result['info'].strip()
                    print('--->  {0}:{1}:{2}:{3}'.format(self.filename,
                                                         result['line'],
                                                         result['column'],
                                                         info),
                          file=sys.stderr)

    def fix(self) -> str:
        """Return a version of the source code with PEP 8 violations fixed."""
        pep8_options = {
            'ignore': self.options.ignore,
            'select': self.options.select,
            'max_line_length': self.options.max_line_length,
        }
        results = _execute_pep8(pep8_options, self.source)

        if self.options.verbose:
            progress = {}
            for r in results:
                if r['id'] not in progress:
                    progress[r['id']] = set()
                progress[r['id']].add(r['line'])
            print('--->  {n} issue(s) to fix {progress}'.format(
                n=len(results), progress=progress), file=sys.stderr)

        if self.options.line_range:
            start, end = self.options.line_range
            results = [r for r in results
                       if start <= r['line'] <= end]

        self._fix_source(filter_results(source=''.join(self.source),
                                        results=results,
                                        aggressive=self.options.aggressive))

        if self.options.line_range:
            # If number of lines has changed then change line_range.
            count = sum(sline.count('\n')
                        for sline in self.source[start - 1:end])
            self.options.line_range[1] = start + count - 1

        return ''.join(self.source)

    def _fix_reindent(self, result: dict) -> None:
        """Fix a badly indented line.

        This is done by adding or removing from its initial indent only.

        """
        num_indent_spaces = int(result['info'].split()[1])
        line_index = result['line'] - 1
        target = self.source[line_index]

        self.source[line_index] = ' ' * num_indent_spaces + target.lstrip()

    def fix_e112(self, result: dict) -> None:
        """Fix under-indented comments."""
        line_index = result['line'] - 1
        target = self.source[line_index]

        if not target.lstrip().startswith('#'):
            # Don't screw with invalid syntax.
            return []

        self.source[line_index] = self.indent_word + target

    def fix_e113(self, result: dict) -> None:
        """Fix over-indented comments."""
        line_index = result['line'] - 1
        target = self.source[line_index]

        indent = _get_indentation(target)
        stripped = target.lstrip()

        if not stripped.startswith('#'):
            # Don't screw with invalid syntax.
            return []

        self.source[line_index] = indent[1:] + stripped

    def fix_e125(self, result: dict) -> list[int]:
        """Fix indentation undistinguish from the next logical line."""
        num_indent_spaces = int(result['info'].split()[1])
        line_index = result['line'] - 1
        target = self.source[line_index]

        spaces_to_add = num_indent_spaces - len(_get_indentation(target))
        indent = len(_get_indentation(target))
        modified_lines = []

        while len(_get_indentation(self.source[line_index])) >= indent:
            self.source[line_index] = (' ' * spaces_to_add +
                                       self.source[line_index])
            modified_lines.append(1 + line_index)  # Line indexed at 1.
            line_index -= 1

        return modified_lines

    def fix_e131(self, result: dict) -> None:
        """Fix indentation undistinguish from the next logical line."""
        num_indent_spaces = int(result['info'].split()[1])
        line_index = result['line'] - 1
        target = self.source[line_index]

        spaces_to_add = num_indent_spaces - len(_get_indentation(target))

        if spaces_to_add >= 0:
            self.source[line_index] = (' ' * spaces_to_add +
                                       self.source[line_index])
        else:
            offset = abs(spaces_to_add)
            self.source[line_index] = self.source[line_index][offset:]

    def fix_e201(self, result: dict) -> None:
        """Remove extraneous whitespace."""
        line_index = result['line'] - 1
        target = self.source[line_index]
        offset = result['column'] - 1

        fixed = fix_whitespace(target,
                               offset=offset,
                               replacement='')

        self.source[line_index] = fixed

    def fix_e224(self, result: dict) -> None:
        """Remove extraneous whitespace around operator."""
        target = self.source[result['line'] - 1]
        offset = result['column'] - 1
        fixed = target[:offset] + target[offset:].replace('\t', ' ')
        self.source[result['line'] - 1] = fixed

    def fix_e225(self, result: dict) -> None:
        """Fix missing whitespace around operator."""
        target = self.source[result['line'] - 1]
        offset = result['column'] - 1
        fixed = target[:offset] + ' ' + target[offset:]

        # Only proceed if non-whitespace characters match.
        # And make sure we don't break the indentation.
        if (
            fixed.replace(' ', '') == target.replace(' ', '') and
            _get_indentation(fixed) == _get_indentation(target)
        ):
            self.source[result['line'] - 1] = fixed
            error_code = result.get('id', 0)
            try:
                ts = generate_tokens(fixed)
            except tokenize.TokenError:
                return
            if not check_syntax(fixed.lstrip()):
                return
            errors = list(
                pycodestyle.missing_whitespace_around_operator(fixed, ts))
            for e in reversed(errors):
                if error_code != e[1].split()[0]:
                    continue
                offset = e[0][1]
                fixed = fixed[:offset] + ' ' + fixed[offset:]
            self.source[result['line'] - 1] = fixed
        else:
            return []

    def fix_e231(self, result: dict) -> None:
        """Add missing whitespace."""
        line_index = result['line'] - 1
        target = self.source[line_index]
        offset = result['column']
        fixed = target[:offset].rstrip() + ' ' + target[offset:].lstrip()
        self.source[line_index] = fixed

    def fix_e251(self, result: dict) -> None:
        """Remove whitespace around parameter '=' sign."""
        line_index = result['line'] - 1
        target = self.source[line_index]

        # This is necessary since pycodestyle sometimes reports columns that
        # goes past the end of the physical line. This happens in cases like,
        # foo(bar\n=None)
        c = min(result['column'] - 1,
                len(target) - 1)

        if target[c].strip():
            fixed = target
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

    def fix_e262(self, result: dict) -> None:
        """Fix spacing after comment hash."""
        target = self.source[result['line'] - 1]
        offset = result['column']

        code = target[:offset].rstrip(' \t#')
        comment = target[offset:].lstrip(' \t#')

        fixed = code + ('  # ' + comment if comment.strip() else '\n')

        self.source[result['line'] - 1] = fixed

    def fix_e271(self, result: dict) -> None:
        """Fix extraneous whitespace around keywords."""
        line_index = result['line'] - 1
        target = self.source[line_index]
        offset = result['column'] - 1

        fixed = fix_whitespace(target,
                               offset=offset,
                               replacement=' ')

        if fixed == target:
            return []
        else:
            self.source[line_index] = fixed

    def fix_e301(self, result: dict) -> None:
        """Add missing blank line."""
        cr = '\n'
        self.source[result['line'] - 1] = cr + self.source[result['line'] - 1]

    def fix_e302(self, result: dict) -> None:
        """Add missing 2 blank lines."""
        add_linenum = 2 - int(result['info'].split()[-1])
        cr = '\n' * add_linenum
        self.source[result['line'] - 1] = cr + self.source[result['line'] - 1]

    def fix_e303(self, result: dict) -> list[int]:
        """Remove extra blank lines."""
        delete_linenum = int(result['info'].split('(')[1].split(')')[0]) - 2
        delete_linenum = max(1, delete_linenum)

        # We need to count because pycodestyle reports an offset line number if
        # there are comments.
        cnt = 0
        line = result['line'] - 2
        modified_lines = []
        while cnt < delete_linenum and line >= 0:
            if not self.source[line].strip():
                self.source[line] = ''
                modified_lines.append(1 + line)  # Line indexed at 1
                cnt += 1
            line -= 1

        return modified_lines

    def fix_e304(self, result: dict) -> None:
        """Remove blank line following function decorator."""
        line = result['line'] - 2
        if not self.source[line].strip():
            self.source[line] = ''

    def fix_e305(self, result: dict) -> None:
        """Add missing 2 blank lines after end of function or class."""
        cr = '\n'
        # check comment line
        offset = result['line'] - 2
        while True:
            if offset < 0:
                break
            line = self.source[offset].lstrip()
            if not line:
                break
            if line[0] != '#':
                break
            offset -= 1
        offset += 1
        self.source[offset] = cr + self.source[offset]

    def fix_e401(self, result: dict) -> None:
        """Put imports on separate lines."""
        line_index = result['line'] - 1
        target = self.source[line_index]
        offset = result['column'] - 1

        if not target.lstrip().startswith('import'):
            return []

        indentation = re.split(pattern=r'\bimport\b',
                               string=target, maxsplit=1)[0]
        fixed = (target[:offset].rstrip('\t ,') + '\n' +
                 indentation + 'import ' + target[offset:].lstrip('\t ,'))
        self.source[line_index] = fixed

    def fix_long_line_logically(self, result: dict, logical: tuple) -> list[int]:
        """Try to make lines fit within --max-line-length characters."""
        if (
            not logical or
            len(logical[2]) == 1 or
            self.source[result['line'] - 1].lstrip().startswith('#')
        ):
            return self.fix_long_line_physically(result)

        start_line_index = logical[0][0]
        end_line_index = logical[1][0]
        logical_lines = logical[2]

        previous_line = get_item(self.source, start_line_index - 1, default='')
        next_line = get_item(self.source, end_line_index + 1, default='')

        single_line = join_logical_line(''.join(logical_lines))

        try:
            fixed = self.fix_long_line(
                target=single_line,
                previous_line=previous_line,
                next_line=next_line,
                original=''.join(logical_lines))
        except (SyntaxError, tokenize.TokenError):
            return self.fix_long_line_physically(result)

        if fixed:
            for line_index in range(start_line_index, end_line_index + 1):
                self.source[line_index] = ''
            self.source[start_line_index] = fixed
            return range(start_line_index + 1, end_line_index + 1)

        return []

    def fix_long_line_physically(self, result: dict) -> list[int]:
        """Try to make lines fit within --max-line-length characters."""
        line_index = result['line'] - 1
        target = self.source[line_index]

        previous_line = get_item(self.source, line_index - 1, default='')
        next_line = get_item(self.source, line_index + 1, default='')

        try:
            fixed = self.fix_long_line(
                target=target,
                previous_line=previous_line,
                next_line=next_line,
                original=target)
        except (SyntaxError, tokenize.TokenError):
            return []

        if fixed:
            self.source[line_index] = fixed
            return [line_index + 1]

        return []

    def fix_long_line(self, target: str, previous_line: str,
                      next_line: str, original: str) -> str:
        cache_entry = (target, previous_line, next_line)
        if cache_entry in self.long_line_ignore_cache:
            return []

        if target.lstrip().startswith('#'):
            if self.options.aggressive:
                # Wrap commented lines.
                return shorten_comment(
                    line=target,
                    max_line_length=self.options.max_line_length,
                    last_comment=not next_line.lstrip().startswith('#'))
            else:
                return []

        fixed = get_fixed_long_line(
            target=target,
            previous_line=previous_line,
            original=original,
            indent_word=self.indent_word,
            max_line_length=self.options.max_line_length,
            aggressive=self.options.aggressive,
            experimental=self.options.experimental,
            verbose=self.options.verbose)

        if fixed and not code_almost_equal(original, fixed):
            return fixed

        self.long_line_ignore_cache.add(cache_entry)
        return None

    def fix_e502(self, result: dict) -> None:
