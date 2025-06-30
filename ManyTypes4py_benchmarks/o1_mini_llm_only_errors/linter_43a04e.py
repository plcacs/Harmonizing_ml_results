"""
The majority of this code is based/inspired or directly taken from
SublimeLinter plugin. Because that, SublimeLinter license file has
been added to this package.

This doesn't meant that anaconda is a fork of SublimeLinter in any
way or that anaconda is going to be updated with latest SublimeLinter
updates. Anaconda and SublimeLinter are two completely separated
projects but they did a great job with SublimeLinter so we are reusing
part of their plugin to lint our Python scripts.

The main difference between SublimeLinter linting and anaconda one is
that the former lints always for Python3.3 even if we are coding in
a Python 2 project. Anaconda lints for the configured Python environment
"""
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

sys.path.insert(0, os.path.dirname(__file__))
import _ast
import pycodestyle as pep8
import pyflakes.checker as pyflakes

if sys.version_info < (2, 7):

    def cmp_to_key(mycmp: Callable[[Any, Any], int]) -> Callable[[Any], Any]:
        """Convert a cmp= function into a key= function
        """

        class K:
            __slots__ = ['obj']

            def __init__(self, obj: Any) -> None:
                self.obj = obj

            def __lt__(self, other: 'K') -> bool:
                return mycmp(self.obj, other.obj) < 0

            def __gt__(self, other: 'K') -> bool:
                return mycmp(self.obj, other.obj) > 0

            def __eq__(self, other: 'K') -> bool:
                return mycmp(self.obj, other.obj) == 0

            def __le__(self, other: 'K') -> bool:
                return mycmp(self.obj, other.obj) <= 0

            def __ge__(self, other: 'K') -> bool:
                return mycmp(self.obj, other.obj) >= 0

            def __ne__(self, other: 'K') -> bool:
                return mycmp(self.obj, other.obj) != 0

            def __hash__(self) -> None:
                raise TypeError('hash not implemented')
        return K
else:
    from functools import cmp_to_key

pyflakes.messages.Message.__str__ = lambda self: self.message % self.message_args


class LintError:
    """Lint error base class
    """

    lineno: int
    level: str
    message: str
    message_args: Tuple[Any, ...]
    offset: Optional[int]
    text: Optional[str]

    def __init__(
        self,
        filename: str,
        loc: Union[int, Any],
        level: str,
        message: str,
        message_args: Tuple[Any, ...],
        **kwargs: Any
    ) -> None:
        self.lineno = loc
        self.level = level
        self.message = message
        self.message_args = message_args
        self.offset = kwargs.get('offset')
        self.text = kwargs.get('text')
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self) -> str:
        """String representation of the error
        """
        return self.message % self.message_args


class Pep8Error(LintError):
    """
    Lint error class for PEP-8 errors
    PEP-8 errors are treated as Warnings
    """

    def __init__(
        self,
        filename: str,
        loc: int,
        offset: Optional[int],
        code: str,
        text: str
    ) -> None:
        super(Pep8Error, self).__init__(
            filename,
            loc,
            'W',
            '[W] PEP 8 (%s): %s',
            (code, text),
            offset=offset,
            text=text
        )


class Pep8Warning(LintError):
    """
    Lint error class for PEP-8 warnings
    PEP-8 warnings are treated as violations
    """

    def __init__(
        self,
        filename: str,
        loc: int,
        offset: Optional[int],
        code: str,
        text: str
    ) -> None:
        super(Pep8Warning, self).__init__(
            filename,
            loc,
            'V',
            '[V] PEP 8 (%s): %s',
            (code, text),
            offset=offset,
            text=text
        )


class PythonError(LintError):
    """Python errors class
    """

    def __init__(self, filename: str, loc: Any, text: str) -> None:
        super(PythonError, self).__init__(
            filename,
            loc,
            'E',
            '[E] %r',
            (text,),
            text=text
        )


class OffsetError(LintError):

    def __init__(
        self,
        filename: str,
        loc: Any,
        text: str,
        offset: int
    ) -> None:
        super(OffsetError, self).__init__(
            filename,
            loc,
            'E',
            '[E] %s',
            (text,),
            offset=offset + 1,
            text=text
        )


class Linter:
    """Linter class for Anaconda's Python linter
    """

    enabled: bool

    def __init__(self) -> None:
        self.enabled = False

    def pyflakes_check(
        self,
        code: str,
        filename: Optional[str],
        ignore: Optional[List[str]] = None
    ) -> List[pyflakes.messages.Message]:
        """Check the code with pyflakes to find errors
        """

        class FakeLoc:
            lineno: int = 0

        try:
            code_bytes = code.encode('utf8') + b'\n'
            tree = compile(code_bytes, filename or '', 'exec', _ast.PyCF_ONLY_AST)
        except (SyntaxError, IndentationError):
            return self._handle_syntactic_error(code, filename)
        except ValueError as error:
            return [PythonError(filename or '', FakeLoc(), error.args[0])]
        else:
            w = pyflakes.Checker(tree, filename, ignore)
            return w.messages

    def pep8_check(
        self,
        code: str,
        filename: Optional[str],
        rcfile: Optional[str],
        ignore: List[str],
        max_line_length: int
    ) -> List[Union[Pep8Error, Pep8Warning]]:
        """Check the code with pep8 to find PEP 8 errors
        """
        messages: List[Union[Pep8Error, Pep8Warning]] = []
        _lines = code.split('\n')
        if _lines:

            class FakeCol:
                """Fake class to represent a col object for PyFlakes
                """

                lineno: int

                def __init__(self, line_number: int) -> None:
                    self.lineno = line_number

            class SublimeLinterReport(pep8.BaseReport):
                """Helper class to report PEP 8 problems
                """

                counters: Dict[str, int]
                messages_set: Dict[str, str]
                file_errors: int
                total_errors: int

                def __init__(self, options: Any) -> None:
                    super().__init__(options)
                    self.counters = {}
                    self.messages_set = {}
                    self.file_errors = 0
                    self.total_errors = 0

                def error(
                    self,
                    line_number: int,
                    offset: int,
                    text: str,
                    check: Any
                ) -> Optional[str]:
                    """Report an error, according to options
                    """
                    col = FakeCol(line_number)
                    code = text[:4]
                    message = text[5:]
                    if self._ignore_code(code):
                        return None
                    if code in self.counters:
                        self.counters[code] += 1
                    else:
                        self.counters[code] = 1
                        self.messages_set[code] = message
                    if code in self.expected:
                        return code
                    self.file_errors += 1
                    self.total_errors += 1
                    pep8_error = code.startswith('E')
                    klass: Callable[..., Union[Pep8Error, Pep8Warning]] = Pep8Error if pep8_error else Pep8Warning
                    messages.append(klass(filename or '', line_number, offset, code, message))
                    return code

            params: Dict[str, Any] = {'reporter': SublimeLinterReport}
            if not rcfile:
                _ignore = ignore + pep8.DEFAULT_IGNORE.split(',')
                params['ignore'] = _ignore
            else:
                params['config_file'] = os.path.expanduser(rcfile)
            style_guide = pep8.StyleGuide(**params)
            options = style_guide.options
            if not rcfile:
                options.max_line_length = max_line_length
            good_lines = [l + '\n' for l in _lines]
            good_lines[-1] = good_lines[-1].rstrip('\n')
            if not good_lines[-1]:
                good_lines = good_lines[:-1]
            pep8.Checker(filename, good_lines, options=options).check_all()
        return messages

    def run_linter(
        self,
        settings: Dict[str, Any],
        code: str,
        filename: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Check the code to find errors
        """
        errors: List[Any] = []
        if settings.get('pep8', True):
            check_params: Dict[str, Any] = {
                'ignore': settings.get('pep8_ignore', []),
                'max_line_length': settings.get('pep8_max_line_length', pep8.MAX_LINE_LENGTH)
            }
            errors.extend(
                self.pep8_check(
                    code,
                    filename,
                    settings.get('pep8_rcfile'),
                    ignore=check_params['ignore'],
                    max_line_length=check_params['max_line_length']
                )
            )
        pyflakes_ignore: Optional[List[str]] = settings.get('pyflakes_ignore', None)
        pyflakes_disabled: bool = settings.get('pyflakes_disabled', False)
        explicit_ignore: List[str] = settings.get('pyflakes_explicit_ignore', [])
        if not pyflakes_disabled and (not settings.get('use_pylint', False)):
            errors.extend(
                self.pyflakes_check(
                    code,
                    filename,
                    ignore=pyflakes_ignore
                )
            )
        return self.parse_errors(errors, explicit_ignore)

    def sort_errors(self, errors: List[Dict[str, Any]]) -> None:
        """Sort errors by line number
        """
        errors.sort(key=cmp_to_key(lambda a, b: a['lineno'] - b['lineno']))

    def prepare_error_level(self, error: Any) -> str:
        """Prepare a common error level in case that the error doesn't define
        """
        return 'W' if not hasattr(error, 'level') else error.level

    def parse_errors(
        self,
        errors: Optional[List[Any]],
        explicit_ignore: List[str]
    ) -> List[Dict[str, Any]]:
        """Parse errors returned from the PyFlakes and pep8 libraries
        """
        errors_list: List[Dict[str, Any]] = []
        if errors is None:
            return errors_list
        errors.sort(key=cmp_to_key(lambda a, b: a.lineno - b.lineno))
        for error in errors:
            error_level: str = 'W' if not hasattr(error, 'level') else error.level
            message: str = error.message.capitalize()
            offset: Optional[int] = None
            if hasattr(error, 'offset'):
                offset = error.offset
            elif hasattr(error, 'col'):
                offset = error.col
            error_data: Dict[str, Any] = {
                'pep8': False,
                'level': error_level,
                'lineno': error.lineno,
                'offset': offset,
                'message': message,
                'raw_error': str(error)
            }
            if isinstance(error, (Pep8Error, Pep8Warning, OffsetError)):
                error_data['pep8'] = True
                errors_list.append(error_data)
            elif isinstance(
                error,
                (
                    pyflakes.messages.RedefinedWhileUnused,
                    pyflakes.messages.RedefinedInListComp,
                    pyflakes.messages.UndefinedName,
                    pyflakes.messages.UndefinedExport,
                    pyflakes.messages.UndefinedLocal,
                    pyflakes.messages.Redefined,
                    pyflakes.messages.UnusedVariable
                )
            ) and error.__class__.__name__ not in explicit_ignore:
                regex = (
                    r'((and|or|not|if|elif|while|in)\s+|[+\-*^%%<>=\(\{{])*\s*'
                    r'(?P<underline>[\w\.]*{0}[\w]*)'.format(
                        re.escape(str(error.message_args[0]))
                    )
                )
                error_data['len'] = len(str(error.message_args[0]))
                error_data['regex'] = regex
                errors_list.append(error_data)
            elif isinstance(error, pyflakes.messages.ImportShadowedByLoopVar):
                regex = (
                    r'for\s+(?P<underline>[\w]*{0}[\w*])'.format(
                        re.escape(str(error.message_args[0]))
                    )
                )
                error_data['regex'] = regex
                errors_list.append(error_data)
            elif isinstance(
                error,
                (
                    pyflakes.messages.UnusedImport,
                    pyflakes.messages.ImportStarUsed
                )
            ) and error.__class__.__name__ not in explicit_ignore:
                if isinstance(error, pyflakes.messages.ImportStarUsed):
                    word = '*'
                else:
                    word = str(error.message_args[0])
                linematch = r'(from\s+[\w_\.]+\s+)?import\s+[^#;]+'
                r = r'(^|\s+|,\s*|as\s+)(?P<underline>[\w]*{0}[\w]*)'.format(
                    re.escape(word)
                )
                error_data['regex'] = r
                error_data['linematch'] = linematch
                errors_list.append(error_data)
            elif isinstance(error, pyflakes.messages.DuplicateArgument) and error.__class__.__name__ not in explicit_ignore:
                regex = (
                    r'def [\w_]+\([^)]*(?P<underline>[\w]*{0}[\w]*)'.format(
                        re.escape(str(error.message_args[0]))
                    )
                )
                error_data['regex'] = regex
                errors_list.append(error_data)
            elif isinstance(error, pyflakes.messages.LateFutureImport):
                pass
            elif isinstance(error, PythonError):
                print(error)
            else:
                print('Oops, we missed an error type!', type(error))
        return errors_list

    def _handle_syntactic_error(
        self,
        code: str,
        filename: Optional[str]
    ) -> List[LintError]:
        """Handle PythonError and OffsetError
        """
        value = sys.exc_info()[1]
        msg: str = value.args[0]
        lineno: int
        offset: Optional[int]
        text: Optional[str]
        lineno, offset, text = (value.lineno, value.offset, value.text)
        if text is None:
            if msg.startswith('duplicate argument'):
                arg = msg.split('duplicate argument ', 1)[1].split(' ', 1)[0].strip('\'"')
                error = pyflakes.messages.DuplicateArgument(filename or '', lineno, arg)
            else:
                error = PythonError(filename or '', value, msg)
        else:
            line = text.splitlines()[-1]
            if offset is not None:
                offset = offset - (len(text) - len(line))
            if offset is not None:
                error = OffsetError(filename or '', value, msg, offset)
            else:
                error = PythonError(filename or '', value, msg)
            error.lineno = lineno
        return [error]
