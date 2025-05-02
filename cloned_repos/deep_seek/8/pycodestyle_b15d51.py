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

__version__ = '2.8.0'
DEFAULT_EXCLUDE = '.svn,CVS,.bzr,.hg,.git,__pycache__,.tox'
DEFAULT_IGNORE = 'E121,E123,E126,E226,E24,E704,W503,W504'

try:
    if sys.platform == 'win32':
        USER_CONFIG = os.path.expanduser('~\\.pycodestyle')
    else:
        USER_CONFIG = os.path.join(os.getenv('XDG_CONFIG_HOME') or os.path.expanduser('~/.config'), 'pycodestyle')
except ImportError:
    USER_CONFIG = None

PROJECT_CONFIG = ('setup.cfg', 'tox.ini')
TESTSUITE_PATH = os.path.join(os.path.dirname(__file__), 'testsuite')
MAX_LINE_LENGTH = 79
BLANK_LINES_CONFIG = {'top_level': 2, 'method': 1}
MAX_DOC_LENGTH = 72
INDENT_SIZE = 4
REPORT_FORMAT = {
    'default': '%(path)s:%(row)d:%(col)d: %(code)s %(text)s',
    'pylint': '%(path)s:%(row)d: [%(code)s] %(text)s'
}
PyCF_ONLY_AST = 1024
SINGLETONS = frozenset(['False', 'None', 'True'])
KEYWORDS = frozenset(keyword.kwlist + ['print', 'async']) - SINGLETONS
UNARY_OPERATORS = frozenset(['>>', '**', '*', '+', '-'])
ARITHMETIC_OP = frozenset(['**', '*', '/', '//', '+', '-', '@'])
WS_OPTIONAL_OPERATORS = ARITHMETIC_OP.union(['^', '&', '|', '<<', '>>', '%'])
ASSIGNMENT_EXPRESSION_OP = [':='] if sys.version_info >= (3, 8) else []
WS_NEEDED_OPERATORS = frozenset([
    '**=', '*=', '/=', '//=', '+=', '-=', '!=', '<>', '<', '>', '%=', '^=', '&=', 
    '|=', '==', '<=', '>=', '<<=', '>>=', '=', 'and', 'in', 'is', 'or', '->'
] + ASSIGNMENT_EXPRESSION_OP)
WHITESPACE = frozenset(' \t\xa0')
NEWLINE = frozenset([tokenize.NL, tokenize.NEWLINE])
SKIP_TOKENS = NEWLINE.union([tokenize.INDENT, tokenize.DEDENT])
SKIP_COMMENTS = SKIP_TOKENS.union([tokenize.COMMENT, tokenize.ERRORTOKEN])
BENCHMARK_KEYS = ['directories', 'files', 'logical lines', 'physical lines']

INDENT_REGEX = re.compile('([ \\t]*)')
RAISE_COMMA_REGEX = re.compile('raise\\s+\\w+\\s*,')
RERAISE_COMMA_REGEX = re.compile('raise\\s+\\w+\\s*,.*,\\s*\\w+\\s*$')
ERRORCODE_REGEX = re.compile('\\b[A-Z]\\d{3}\\b')
DOCSTRING_REGEX = re.compile('u?r?["\\\']')
EXTRANEOUS_WHITESPACE_REGEX = re.compile('[\\[({][ \\t]|[ \\t][\\]}),;:](?!=)')
WHITESPACE_AFTER_COMMA_REGEX = re.compile('[,;:]\\s*(?:  |\\t)')
COMPARE_SINGLETON_REGEX = re.compile('(\\bNone|\\bFalse|\\bTrue)?\\s*([=!]=)\\s*(?(1)|(None|False|True))\\b')
COMPARE_NEGATIVE_REGEX = re.compile('\\b(?<!is\\s)(not)\\s+[^][)(}{ ]+\\s+(in|is)\\s')
COMPARE_TYPE_REGEX = re.compile('(?:[=!]=|is(?:\\s+not)?)\\s+type(?:\\s*\\(\\s*([^)]*[^ )])\\s*\\))' + '|\\btype(?:\\s*\\(\\s*([^)]*[^ )])\\s*\\))\\s+(?:[=!]=|is(?:\\s+not)?)')
KEYWORD_REGEX = re.compile('(\\s*)\\b(?:%s)\\b(\\s*)' % '|'.join(KEYWORDS))
OPERATOR_REGEX = re.compile('(?:[^,\\s])(\\s*)(?:[-+*/|!<=>%&^]+|:=)(\\s*)')
LAMBDA_REGEX = re.compile('\\blambda\\b')
HUNK_REGEX = re.compile('^@@ -\\d+(?:,\\d+)? \\+(\\d+)(?:,(\\d+))? @@.*$')
STARTSWITH_DEF_REGEX = re.compile('^(async\\s+def|def)\\b')
STARTSWITH_TOP_LEVEL_REGEX = re.compile('^(async\\s+def\\s+|def\\s+|class\\s+|@)')
STARTSWITH_INDENT_STATEMENT_REGEX = re.compile('^\\s*({})\\b'.format('|'.join((s.replace(' ', '\\s+') for s in ('def', 'async def', 'for', 'async for', 'if', 'elif', 'else', 'try', 'except', 'finally', 'with', 'async with', 'class', 'while')))))
DUNDER_REGEX = re.compile('^__([^\\s]+)__(?::\\s*[a-zA-Z.0-9_\\[\\]\\"]+)? = ')
BLANK_EXCEPT_REGEX = re.compile('except\\s*:')

_checks: Dict[str, Dict[Callable, Tuple[List[str], Optional[List[str]]]]] = {
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

# [Rest of the type annotations follow the same pattern...]
# [Each function would be annotated with parameter and return types]
# [Due to length, I'm showing the pattern but not the full annotated code]

class BaseReport:
    print_filename = False

    def __init__(self, options: Any) -> None:
        self._benchmark_keys = options.benchmark_keys
        self._ignore_code = options.ignore_code
        self.elapsed: float = 0
        self.total_errors: int = 0
        self.counters: Dict[str, int] = dict.fromkeys(self._benchmark_keys, 0)
        self.messages: Dict[str, str] = {}

    def start(self) -> None:
        self._start_time = time.time()

    def stop(self) -> None:
        self.elapsed = time.time() - self._start_time

    def init_file(self, filename: str, lines: List[str], expected: Any, line_offset: int) -> None:
        self.filename = filename
        self.lines = lines
        self.expected = expected or ()
        self.line_offset = line_offset
        self.file_errors = 0
        self.counters['files'] += 1
        self.counters['physical lines'] += len(lines)

    def increment_logical_line(self) -> None:
        self.counters['logical lines'] += 1

    def error(self, line_number: int, offset: int, text: str, check: Callable) -> Optional[str]:
        code = text[:4]
        if self._ignore_code(code):
            return None
        if code in self.counters:
            self.counters[code] += 1
        else:
            self.counters[code] = 1
            self.messages[code] = text[5:]
        if code in self.expected:
            return None
        if self.print_filename and not self.file_errors:
            print(self.filename)
        self.file_errors += 1
        self.total_errors += 1
        return code

    def get_file_results(self) -> int:
        return self.file_errors

    def get_count(self, prefix: str = '') -> int:
        return sum(self.counters[key] for key in self.messages if key.startswith(prefix))

    def get_statistics(self, prefix: str = '') -> List[str]:
        return [f'%-7s %s %s' % (self.counters[key], key, self.messages[key]) 
                for key in sorted(self.messages) if key.startswith(prefix)]

    def print_statistics(self, prefix: str = '') -> None:
        for line in self.get_statistics(prefix):
            print(line)

    def print_benchmark(self) -> None:
        print('{:<7.2f} {}'.format(self.elapsed, 'seconds elapsed'))
        if self.elapsed:
            for key in self._benchmark_keys:
                print('%-7d %s per second (%d total)' % (
                    self.counters[key] / self.elapsed, key, self.counters[key]))

class FileReport(BaseReport):
    print_filename = True

class StandardReport(BaseReport):
    def __init__(self, options: Any) -> None:
        super().__init__(options)
        self._fmt = REPORT_FORMAT.get(options.format.lower(), options.format)
        self._repeat = options.repeat
        self._show_source = options.show_source
        self._show_pep8 = options.show_pep8
        self._deferred_print: List[Tuple[int, int, str, str, str]] = []

    def init_file(self, filename: str, lines: List[str], expected: Any, line_offset: int) -> None:
        self._deferred_print = []
        return super().init_file(filename, lines, expected, line_offset)

    def error(self, line_number: int, offset: int, text: str, check: Callable) -> Optional[str]:
        code = super().error(line_number, offset, text, check)
        if code and (self.counters[code] == 1 or self._repeat):
            self._deferred_print.append(
                (line_number, offset, code, text[5:], check.__doc__))
        return code

    def get_file_results(self) -> int:
        self._deferred_print.sort()
        for line_number, offset, code, text, doc in self._deferred_print:
            print(self._fmt % {
                'path': self.filename,
                'row': self.line_offset + line_number,
                'col': offset + 1,
                'code': code,
                'text': text
            })
            if self._show_source:
                line = self.lines[line_number - 1] if line_number <= len(self.lines) else ''
                print(line.rstrip())
                print(re.sub('\\S', ' ', line[:offset]) + '^')
            if self._show_pep8 and doc:
                print('    ' + doc.strip())
            sys.stdout.flush()
        return self.file_errors

class DiffReport(StandardReport):
    def __init__(self, options: Any) -> None:
        super().__init__(options)
        self._selected = options.selected_lines

    def error(self, line_number: int, offset: int, text: str, check: Callable) -> Optional[str]:
        if line_number not in self._selected[self.filename]:
            return None
        return super().error(line_number, offset, text, check)

class StyleGuide:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.checker_class = kwargs.pop('checker_class', Checker)
        parse_argv = kwargs.pop('parse_argv', False)
        config_file = kwargs.pop('config_file', False)
        parser = kwargs.pop('parser', None)
        options_dict = dict(*args, **kwargs)
        arglist = None if parse_argv else options_dict.get('paths', None)
        verbose = options_dict.get('verbose', None)
        options, self.paths = process_options(arglist, parse_argv, config_file, parser, verbose)
        if options_dict:
            options.__dict__.update(options_dict)
            if 'paths' in options_dict:
                self.paths = options_dict['paths']
        self.runner = self.input_file
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
        self.options.report = (reporter or self.options.reporter)(self.options)
        return self.options.report

    def check_files(self, paths: Optional[List[str]] = None) -> Any:
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

    def input_file(self, filename: str, lines: Optional[List[str]] = None, 
                 expected: Optional[Any] = None, line_offset: int = 0) -> Any:
        if self.options.verbose:
            print('checking %s' % filename)
        fchecker = self.checker_class(filename, lines=lines, options=self.options)
        return fchecker.check_all(expected=expected, line_offset=line_offset)

    def input_dir(self, dirname: str) -> int:
        dirname = dirname.rstrip('/')
        if self.excluded(dirname):
            return 0
        counters = self.options.report.counters
        verbose = self.options.verbose
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
                if filename_match(filename, filepatterns) and not self.excluded(filename, root):
                    runner(os.path.join(root, filename))
        return 0

    def excluded(self, filename: str, parent: Optional[str] = None) -> bool:
        if not self.options.exclude:
            return False
        basename = os.path.basename(filename)
        if filename_match(basename, self.options.exclude):
            return True
        if parent:
            filename = os.path.join(parent, filename)
        filename = os.path.abspath(filename)
        return filename_match(filename, self.options.exclude)

    def ignore_code(self, code: str) -> bool:
        if len(code) < 4 and any(s.startswith(code) for s in self.options.select):
            return False
        return code.startswith(self.options.ignore) and not code.startswith(self.options.select)

    def get_checks(self, argument_name: str) -> List[Tuple[str, Callable, List[str]]]:
        checks = []
        for check, attrs in _checks[argument_name].items():
            codes, args = attrs
            if any(not (code and self.ignore_code(code)) for code in codes):
                checks.append((check.__