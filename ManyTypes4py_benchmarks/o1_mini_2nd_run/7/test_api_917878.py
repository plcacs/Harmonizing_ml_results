"""
Tests for L{pyflakes.scripts.pyflakes}.
"""
import contextlib
import os
import sys
import shutil
import subprocess
import tempfile
from typing import Any, Callable, List, Optional, Tuple, Union, Generator
from pyflakes.messages import UnusedImport
from pyflakes.reporter import Reporter
from pyflakes.api import main, checkPath, checkRecursive, iterSourceCode
from pyflakes.test.harness import TestCase, skipIf

if sys.version_info < (3,):
    from cStringIO import StringIO
else:
    from io import StringIO
    unichr = chr

try:
    sys.pypy_version_info
    PYPY = True
except AttributeError:
    PYPY = False

try:
    WindowsError
    WIN = True
except NameError:
    WIN = False

ERROR_HAS_COL_NUM: bool
ERROR_HAS_LAST_LINE: bool
ERROR_HAS_COL_NUM = ERROR_HAS_LAST_LINE = sys.version_info >= (3, 2) or PYPY


def withStderrTo(
    stderr: Union[StringIO, Any],
    f: Callable[..., Any],
    *args: Any,
    **kwargs: Any
) -> Any:
    """
    Call C{f} with C{sys.stderr} redirected to C{stderr}.
    """
    outer, sys.stderr = (sys.stderr, stderr)
    try:
        return f(*args, **kwargs)
    finally:
        sys.stderr = outer


class Node(object):
    """
    Mock an AST node.
    """

    lineno: int
    col_offset: int

    def __init__(self, lineno: int, col_offset: int = 0) -> None:
        self.lineno = lineno
        self.col_offset = col_offset


class SysStreamCapturing(object):
    """
    Context manager capturing sys.stdin, sys.stdout and sys.stderr.

    The file handles are replaced with a StringIO object.
    On environments that support it, the StringIO object uses newlines
    set to os.linesep.  Otherwise newlines are converted from \\n to
    os.linesep during __exit__.
    """

    _newline: bool
    _stdin: StringIO
    _stdout_stringio: StringIO
    _stderr_stringio: StringIO
    output: str
    error: str

    def _create_StringIO(self, buffer: Optional[str] = None) -> StringIO:
        try:
            return StringIO(buffer, newline=os.linesep)  # type: ignore
        except TypeError:
            self._newline = True
            if buffer is None:
                return StringIO()
            else:
                return StringIO(buffer)

    def __init__(self, stdin: Optional[str]) -> None:
        self._newline = False
        self._stdin = self._create_StringIO(stdin or '')

    def __enter__(self) -> "SysStreamCapturing":
        self._orig_stdin: Any = sys.stdin
        self._orig_stdout: Any = sys.stdout
        self._orig_stderr: Any = sys.stderr
        sys.stdin = self._stdin
        sys.stdout = self._stdout_stringio = self._create_StringIO()
        sys.stderr = self._stderr_stringio = self._create_StringIO()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_value: Optional[BaseException],
        traceback: Optional[Any],
    ) -> None:
        self.output = self._stdout_stringio.getvalue()
        self.error = self._stderr_stringio.getvalue()
        if self._newline and os.linesep != '\n':
            self.output = self.output.replace('\n', os.linesep)
            self.error = self.error.replace('\n', os.linesep)
        sys.stdin = self._orig_stdin
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr


class LoggingReporter(object):
    """
    Implementation of Reporter that just appends any error to a list.
    """

    log: List[Tuple[str, ...]]

    def __init__(self, log: List[Tuple[str, ...]]) -> None:
        """
        Construct a C{LoggingReporter}.

        @param log: A list to append log messages to.
        """
        self.log = log

    def flake(self, message: Any) -> None:
        self.log.append(("flake", str(message)))

    def unexpectedError(self, filename: str, message: str) -> None:
        self.log.append(("unexpectedError", filename, message))

    def syntaxError(
        self,
        filename: str,
        msg: str,
        lineno: int,
        offset: Optional[int],
        line: str,
    ) -> None:
        self.log.append(("syntaxError", filename, msg, lineno, offset, line))


class TestIterSourceCode(TestCase):
    """
    Tests for L{iterSourceCode}.
    """

    tempdir: str

    def setUp(self) -> None:
        self.tempdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tempdir)

    def makeEmptyFile(self, *parts: str) -> str:
        assert parts
        fpath = os.path.join(self.tempdir, *parts)
        open(fpath, 'a').close()
        return fpath

    def test_emptyDirectory(self) -> None:
        """
        There are no Python files in an empty directory.
        """
        self.assertEqual(list(iterSourceCode([self.tempdir])), [])

    def test_singleFile(self) -> None:
        """
        If the directory contains one Python file, C{iterSourceCode} will find
        it.
        """
        childpath = self.makeEmptyFile('foo.py')
        self.assertEqual(list(iterSourceCode([self.tempdir])), [childpath])

    def test_onlyPythonSource(self) -> None:
        """
        Files that are not Python source files are not included.
        """
        self.makeEmptyFile('foo.pyc')
        self.assertEqual(list(iterSourceCode([self.tempdir])), [])

    def test_recurses(self) -> None:
        """
        If the Python files are hidden deep down in child directories, we will
        find them.
        """
        os.mkdir(os.path.join(self.tempdir, 'foo'))
        apath = self.makeEmptyFile('foo', 'a.py')
        self.makeEmptyFile('foo', 'a.py~')
        os.mkdir(os.path.join(self.tempdir, 'bar'))
        bpath = self.makeEmptyFile('bar', 'b.py')
        cpath = self.makeEmptyFile('c.py')
        self.assertEqual(
            sorted(iterSourceCode([self.tempdir])),
            sorted([apath, bpath, cpath]),
        )

    def test_shebang(self) -> None:
        """
        Find Python files that don't end with `.py`, but contain a Python
        shebang.
        """
        python = os.path.join(self.tempdir, 'a')
        with open(python, 'w') as fd:
            fd.write('#!/usr/bin/env python\n')
        self.makeEmptyFile('b')
        with open(os.path.join(self.tempdir, 'c'), 'w') as fd:
            fd.write('hello\nworld\n')
        python2 = os.path.join(self.tempdir, 'd')
        with open(python2, 'w') as fd:
            fd.write('#!/usr/bin/env python2\n')
        python3 = os.path.join(self.tempdir, 'e')
        with open(python3, 'w') as fd:
            fd.write('#!/usr/bin/env python3\n')
        pythonw = os.path.join(self.tempdir, 'f')
        with open(pythonw, 'w') as fd:
            fd.write('#!/usr/bin/env pythonw\n')
        python3args = os.path.join(self.tempdir, 'g')
        with open(python3args, 'w') as fd:
            fd.write('#!/usr/bin/python3 -u\n')
        python2u = os.path.join(self.tempdir, 'h')
        with open(python2u, 'w') as fd:
            fd.write('#!/usr/bin/python2u\n')
        python3d = os.path.join(self.tempdir, 'i')
        with open(python3d, 'w') as fd:
            fd.write('#!/usr/local/bin/python3d\n')
        python38m = os.path.join(self.tempdir, 'j')
        with open(python38m, 'w') as fd:
            fd.write('#! /usr/bin/env python3.8m\n')
        python27 = os.path.join(self.tempdir, 'k')
        with open(python27, 'w') as fd:
            fd.write('#!/usr/bin/python2.7   \n')
        notfirst = os.path.join(self.tempdir, 'l')
        with open(notfirst, 'w') as fd:
            fd.write('#!/bin/sh\n#!/usr/bin/python\n')
        self.assertEqual(
            sorted(iterSourceCode([self.tempdir])),
            sorted(
                [
                    python,
                    python2,
                    python3,
                    pythonw,
                    python3args,
                    python2u,
                    python3d,
                    python38m,
                    python27,
                ]
            ),
        )

    def test_multipleDirectories(self) -> None:
        """
        L{iterSourceCode} can be given multiple directories.  It will recurse
        into each of them.
        """
        foopath = os.path.join(self.tempdir, 'foo')
        barpath = os.path.join(self.tempdir, 'bar')
        os.mkdir(foopath)
        apath = self.makeEmptyFile('foo', 'a.py')
        os.mkdir(barpath)
        bpath = self.makeEmptyFile('bar', 'b.py')
        self.assertEqual(
            sorted(iterSourceCode([foopath, barpath])),
            sorted([apath, bpath]),
        )

    def test_explicitFiles(self) -> None:
        """
        If one of the paths given to L{iterSourceCode} is not a directory but
        a file, it will include that in its output.
        """
        epath = self.makeEmptyFile('e.py')
        self.assertEqual(list(iterSourceCode([epath])), [epath])


class TestReporter(TestCase):
    """
    Tests for L{Reporter}.
    """

    def test_syntaxError(self) -> None:
        """
        C{syntaxError} reports that there was a syntax error in the source
        file.  It reports to the error stream and includes the filename, line
        number, error message, actual line of source and a caret pointing to
        where the error is.
        """
        err = StringIO()
        reporter = Reporter(None, err)
        reporter.syntaxError(
            'foo.py',
            'a problem',
            3,
            8 if sys.version_info >= (3, 8) else 7,
            'bad line of source',
        )
        self.assertEqual(
            'foo.py:3:8: a problem\nbad line of source\n       ^\n',
            err.getvalue(),
        )

    def test_syntaxErrorNoOffset(self) -> None:
        """
        C{syntaxError} doesn't include a caret pointing to the error if
        C{offset} is passed as C{None}.
        """
        err = StringIO()
        reporter = Reporter(None, err)
        reporter.syntaxError(
            'foo.py', 'a problem', 3, None, 'bad line of source'
        )
        self.assertEqual(
            'foo.py:3: a problem\nbad line of source\n',
            err.getvalue(),
        )

    def test_multiLineSyntaxError(self) -> None:
        """
        If there's a multi-line syntax error, then we only report the last
        line.  The offset is adjusted so that it is relative to the start of
        the last line.
        """
        err = StringIO()
        lines = ['bad line of source', 'more bad lines of source']
        reporter = Reporter(None, err)
        reporter.syntaxError(
            'foo.py',
            'a problem',
            3,
            len(lines[0]) + 7,
            '\n'.join(lines),
        )
        column = 25 if sys.version_info >= (3, 8) else 7
        self.assertEqual(
            'foo.py:3:%d: a problem\n' % column
            + lines[-1]
            + '\n'
            + ' ' * (column - 1)
            + '^\n',
            err.getvalue(),
        )

    def test_unexpectedError(self) -> None:
        """
        C{unexpectedError} reports an error processing a source file.
        """
        err = StringIO()
        reporter = Reporter(None, err)
        reporter.unexpectedError('source.py', 'error message')
        self.assertEqual('source.py: error message\n', err.getvalue())

    def test_flake(self) -> None:
        """
        C{flake} reports a code warning from Pyflakes.  It is exactly the
        str() of a L{pyflakes.messages.Message}.
        """
        out = StringIO()
        reporter = Reporter(out, None)
        message = UnusedImport('foo.py', Node(42), 'bar')
        reporter.flake(message)
        self.assertEqual(out.getvalue(), f'{message}\n')


class CheckTests(TestCase):
    """
    Tests for L{check} and L{checkPath} which check a file for flakes.
    """

    @contextlib.contextmanager
    def makeTempFile(self, content: Union[str, bytes]) -> Generator[str, None, None]:
        """
        Make a temporary file containing C{content} and return a path to it.
        """
        fd, name = tempfile.mkstemp()
        try:
            with os.fdopen(fd, 'wb') as f:
                if not hasattr(content, 'decode'):
                    content = content.encode('ascii')
                f.write(content)
            yield name
        finally:
            os.remove(name)

    def assertHasErrors(self, path: str, errorList: List[str]) -> None:
        """
        Assert that C{path} causes errors.

        @param path: A path to a file to check.
        @param errorList: A list of errors expected to be printed to stderr.
        """
        err = StringIO()
        count = withStderrTo(err, checkPath, path)
        self.assertEqual((count, err.getvalue()), (len(errorList), ''.join(errorList)))

    def getErrors(self, path: str) -> Tuple[int, List[Tuple[str, ...]]]:
        """
        Get any warnings or errors reported by pyflakes for the file at C{path}.

        @param path: The path to a Python file on disk that pyflakes will check.
        @return: C{(count, log)}, where C{count} is the number of warnings or
            errors generated, and log is a list of those warnings, presented
            as structured data.  See L{LoggingReporter} for more details.
        """
        log: List[Tuple[str, ...]] = []
        reporter = LoggingReporter(log)
        count = checkPath(path, reporter)
        return (count, log)

    def test_legacyScript(self) -> None:
        from pyflakes.scripts import pyflakes as script_pyflakes

        self.assertIs(script_pyflakes.checkPath, checkPath)

    def test_missingTrailingNewline(self) -> None:
        """
        Source which doesn't end with a newline shouldn't cause any
        exception to be raised nor an error indicator to be returned by
        L{check}.
        """
        with self.makeTempFile('def foo():\n\tpass\n\t') as fName:
            self.assertHasErrors(fName, [])

    def test_checkPathNonExisting(self) -> None:
        """
        L{checkPath} handles non-existing files.
        """
        count, errors = self.getErrors('extremo')
        self.assertEqual(count, 1)
        self.assertEqual(
            errors, [('unexpectedError', 'extremo', 'No such file or directory')]
        )

    def test_multilineSyntaxError(self) -> None:
        """
        Source which includes a syntax error which results in the raised
        L{SyntaxError.text} containing multiple lines of source are reported
        with only the last line of that source.
        """
        source = (
            "def foo():\n"
            "    '''\n"
            "\n"
            "def bar():\n"
            "    pass\n"
            "\n"
            "def baz():\n"
            "    '''quux'''\n"
        )

        def evaluate(source: str) -> None:
            exec(source)

        try:
            evaluate(source)
        except SyntaxError:
            e = sys.exc_info()[1]
            if not PYPY and sys.version_info < (3, 10):
                self.assertTrue(e.text.count('\n') > 1)
        else:
            self.fail()

        with self.makeTempFile(source) as sourcePath:
            if PYPY:
                message = 'end of file (EOF) while scanning triple-quoted string literal'
            elif sys.version_info >= (3, 10):
                message = 'unterminated triple-quoted string literal (detected at line 8)'
            else:
                message = 'invalid syntax'
            if sys.version_info >= (3, 10):
                column = 12
            elif sys.version_info >= (3, 8):
                column = 8
            else:
                column = 11
            self.assertHasErrors(
                sourcePath,
                [
                    f"{sourcePath}:8:{column}: {message}\n"
                    "    '''quux'''\n"
                    f"{' ' * (column - 1)}^\n"
                ],
            )

    def test_eofSyntaxError(self) -> None:
        """
        The error reported for source files which end prematurely causing a
        syntax error reflects the cause for the syntax error.
        """
        with self.makeTempFile('def foo(') as sourcePath:
            if PYPY:
                msg = 'parenthesis is never closed'
            elif sys.version_info >= (3, 10):
                msg = "'(' was never closed"
            else:
                msg = 'unexpected EOF while parsing'
            if PYPY:
                column = 7
            elif sys.version_info >= (3, 10):
                column = 8
            else:
                column = 9
            spaces = ' ' * (column - 1)
            expected = f'{sourcePath}:1:{column}: {msg}\ndef foo(\n{spaces}^\n'
            self.assertHasErrors(sourcePath, [expected])

    def test_eofSyntaxErrorWithTab(self) -> None:
        """
        The error reported for source files which end prematurely causing a
        syntax error reflects the cause for the syntax error.
        """
        with self.makeTempFile('if True:\n\tfoo =') as sourcePath:
            column = 6 if PYPY else 7
            last_line = '\t    ^' if PYPY else '\t     ^'
            self.assertHasErrors(
                sourcePath,
                [f'{sourcePath}:2:{column}: invalid syntax\n\tfoo =\n{last_line}\n'],
            )

    def test_nonDefaultFollowsDefaultSyntaxError(self) -> None:
        """
        Source which has a non-default argument following a default argument
        should include the line number of the syntax error.  However these
        exceptions do not include an offset.
        """
        source = 'def foo(bar=baz, bax):\n    pass\n'
        with self.makeTempFile(source) as sourcePath:
            if ERROR_HAS_LAST_LINE:
                if PYPY:
                    column = 7
                elif sys.version_info >= (3, 10):
                    column = 18
                elif sys.version_info >= (3, 9):
                    column = 21
                elif sys.version_info >= (3, 8):
                    column = 9
                else:
                    column = 8
                last_line = ' ' * (column - 1) + '^\n'
                columnstr = f'{column}:'
            else:
                last_line = ''
                columnstr = ''
            self.assertHasErrors(
                sourcePath,
                [
                    f'{sourcePath}:1:{columnstr} non-default argument follows default argument\n'
                    'def foo(bar=baz, bax):\n'
                    f'{last_line}'
                ],
            )

    def test_nonKeywordAfterKeywordSyntaxError(self) -> None:
        """
        Source which has a non-keyword argument after a keyword argument should
        include the line number of the syntax error.  However these exceptions
        do not include an offset.
        """
        source = 'foo(bar=baz, bax)\n'
        with self.makeTempFile(source) as sourcePath:
            if ERROR_HAS_LAST_LINE:
                if PYPY:
                    column = 12
                elif sys.version_info >= (3, 9):
                    column = 17
                elif sys.version_info >= (3, 8):
                    column = 14
                else:
                    column = 13
                last_line = ' ' * (column - 1) + '^\n'
                columnstr = f'{column}:'
            else:
                last_line = ''
                columnstr = ''
            if sys.version_info >= (3, 5):
                message = 'positional argument follows keyword argument'
            else:
                message = 'non-keyword arg after keyword arg'
            self.assertHasErrors(
                sourcePath,
                [
                    f'{sourcePath}:1:{columnstr} {message}\n'
                    'foo(bar=baz, bax)\n'
                    f'{last_line}'
                ],
            )

    def test_invalidEscape(self) -> None:
        """
        The invalid escape syntax raises ValueError in Python 2
        """
        ver = sys.version_info
        with self.makeTempFile("foo = '\\xyz'") as sourcePath:
            if ver < (3,):
                decoding_error = f'{sourcePath}: problem decoding source\n'
            else:
                position_end = 1
                if PYPY:
                    column = 5
                elif ver >= (3, 9):
                    column = 13
                else:
                    column = 7
                    if ver < (3, 2, 4) or ver[:3] == (3, 3, 0):
                        position_end = 2
                if ERROR_HAS_LAST_LINE:
                    last_line = f'{" " * (column - 1)}^\n'
                else:
                    last_line = ''
                decoding_error = (
                    f"{sourcePath}:1:{column}: (unicode error) 'unicodeescape' codec can't decode bytes in position 0-{position_end}: truncated \\xXX escape\n"
                    'foo = \'\\xyz\'\n'
                    f'{last_line}'
                )
            self.assertHasErrors(sourcePath, [decoding_error])

    @skipIf(sys.platform == 'win32', 'unsupported on Windows')
    def test_permissionDenied(self) -> None:
        """
        If the source file is not readable, this is reported on standard
        error.
        """
        if os.getuid() == 0:
            self.skipTest('root user can access all files regardless of permissions')
        with self.makeTempFile('') as sourcePath:
            os.chmod(sourcePath, 0)
            count, errors = self.getErrors(sourcePath)
            self.assertEqual(count, 1)
            self.assertEqual(
                errors,
                [
                    ('unexpectedError', sourcePath, 'Permission denied'),
                ],
            )

    def test_pyflakesWarning(self) -> None:
        """
        If the source file has a pyflakes warning, this is reported as a
        'flake'.
        """
        with self.makeTempFile('import foo') as sourcePath:
            count, errors = self.getErrors(sourcePath)
            self.assertEqual(count, 1)
            self.assertEqual(
                errors,
                [('flake', str(UnusedImport(sourcePath, Node(1), 'foo')))],
            )

    def test_encodedFileUTF8(self) -> None:
        """
        If source file declares the correct encoding, no error is reported.
        """
        SNOWMAN = unichr(9731)
        source = f'# coding: utf-8\nx = "{SNOWMAN}"\n'.encode('utf-8')
        with self.makeTempFile(source) as sourcePath:
            self.assertHasErrors(sourcePath, [])

    def test_CRLFLineEndings(self) -> None:
        """
        Source files with Windows CR LF line endings are parsed successfully.
        """
        with self.makeTempFile('x = 42\r\n') as sourcePath:
            self.assertHasErrors(sourcePath, [])

    def test_misencodedFileUTF8(self) -> None:
        """
        If a source file contains bytes which cannot be decoded, this is
        reported on stderr.
        """
        SNOWMAN = unichr(9731)
        source = f'# coding: ascii\nx = "{SNOWMAN}"\n'.encode('utf-8')
        with self.makeTempFile(source) as sourcePath:
            if PYPY and sys.version_info < (3,):
                message = (
                    f"'{ 'ascii' }' codec can't decode byte 0xe2 in position 21: ordinal not in range(128)"
                )
                result = (
                    f'{sourcePath}:0:0: {message}\n'
                    'x = "â\x98\x83"\n'
                    '        ^\n'
                )
            else:
                message = 'problem decoding source'
                result = f'{sourcePath}: problem decoding source\n'
            self.assertHasErrors(sourcePath, [result])

    def test_misencodedFileUTF16(self) -> None:
        """
        If a source file contains bytes which cannot be decoded, this is
        reported on stderr.
        """
        SNOWMAN = unichr(9731)
        source = f'# coding: ascii\nx = "{SNOWMAN}"\n'.encode('utf-16')
        with self.makeTempFile(source) as sourcePath:
            self.assertHasErrors(sourcePath, [f'{sourcePath}: problem decoding source\n'])

    def test_checkRecursive(self) -> None:
        """
        L{checkRecursive} descends into each directory, finding Python files
        and reporting problems.
        """
        tempdir = tempfile.mkdtemp()
        try:
            os.mkdir(os.path.join(tempdir, 'foo'))
            file1 = os.path.join(tempdir, 'foo', 'bar.py')
            with open(file1, 'wb') as fd:
                fd.write('import baz\n'.encode('ascii'))
            file2 = os.path.join(tempdir, 'baz.py')
            with open(file2, 'wb') as fd:
                fd.write('import contraband'.encode('ascii'))
            log: List[Tuple[str, ...]] = []
            reporter = LoggingReporter(log)
            warnings = checkRecursive([tempdir], reporter)
            self.assertEqual(warnings, 2)
            self.assertEqual(
                sorted(log),
                sorted(
                    [
                        ('flake', str(UnusedImport(file1, Node(1), 'baz'))),
                        ('flake', str(UnusedImport(file2, Node(1), 'contraband'))),
                    ]
                ),
            )
        finally:
            shutil.rmtree(tempdir)


class IntegrationTests(TestCase):
    """
    Tests of the pyflakes script that actually spawn the script.
    """

    tempdir: str
    tempfilepath: str

    def setUp(self) -> None:
        self.tempdir = tempfile.mkdtemp()
        self.tempfilepath = os.path.join(self.tempdir, 'temp')

    def tearDown(self) -> None:
        shutil.rmtree(self.tempdir)

    def getPyflakesBinary(self) -> str:
        """
        Return the path to the pyflakes binary.
        """
        import pyflakes

        package_dir = os.path.dirname(pyflakes.__file__)
        return os.path.join(package_dir, '..', 'bin', 'pyflakes')

    def runPyflakes(
        self, paths: List[str], stdin: Optional[str] = None
    ) -> Tuple[str, str, int]:
        """
        Launch a subprocess running C{pyflakes}.

        @param paths: Command-line arguments to pass to pyflakes.
        @param stdin: Text to use as stdin.
        @return: C{(returncode, stdout, stderr)} of the completed pyflakes
            process.
        """
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join(sys.path)
        command = [sys.executable, self.getPyflakesBinary()]
        command.extend(paths)
        if stdin:
            p = subprocess.Popen(
                command,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = p.communicate(stdin.encode('ascii'))
        else:
            p = subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = p.communicate()
        rv = p.wait()
        if sys.version_info >= (3,):
            stdout = stdout.decode('utf-8')
            stderr = stderr.decode('utf-8')
        return (stdout, stderr, rv)

    def test_goodFile(self) -> None:
        """
        When a Python source file is all good, the return code is zero and no
        messages are printed to either stdout or stderr.
        """
        open(self.tempfilepath, 'a').close()
        d = self.runPyflakes([self.tempfilepath])
        self.assertEqual(d, ('', '', 0))

    def test_fileWithFlakes(self) -> None:
        """
        When a Python source file has warnings, the return code is non-zero
        and the warnings are printed to stdout.
        """
        with open(self.tempfilepath, 'wb') as fd:
            fd.write('import contraband\n'.encode('ascii'))
        d = self.runPyflakes([self.tempfilepath])
        expected = UnusedImport(self.tempfilepath, Node(1), 'contraband')
        self.assertEqual(d, (f'{expected}{os.linesep}', '', 1))

    def test_errors_io(self) -> None:
        """
        When pyflakes finds errors with the files it's given, (if they don't
        exist, say), then the return code is non-zero and the errors are
        printed to stderr.
        """
        d = self.runPyflakes([self.tempfilepath])
        error_msg = f'{self.tempfilepath}: No such file or directory{os.linesep}'
        self.assertEqual(d, ('', error_msg, 1))

    def test_errors_syntax(self) -> None:
        """
        When pyflakes finds errors with the files it's given, (if they don't
        exist, say), then the return code is non-zero and the errors are
        printed to stderr.
        """
        with open(self.tempfilepath, 'wb') as fd:
            fd.write('import'.encode('ascii'))
        d = self.runPyflakes([self.tempfilepath])
        if PYPY:
            column = 6
        elif sys.version_info >= (3, 8):
            column = 7
        else:
            column = 7 if PYPY else 8
        error_msg = (
            f"{self.tempfilepath}:1:{column}: invalid syntax{os.linesep}"
            "import\n"
            f"{' ' * (column - 1)}^\n"
        )
        self.assertEqual(d, ('', error_msg, 1))

    def test_readFromStdin(self) -> None:
        """
        If no arguments are passed to C{pyflakes} then it reads from stdin.
        """
        d = self.runPyflakes([], stdin='import contraband')
        expected = UnusedImport('<stdin>', Node(1), 'contraband')
        self.assertEqual(d, (f'{expected}{os.linesep}', '', 1))


class TestMain(IntegrationTests):
    """
    Tests of the pyflakes main function.
    """

    def runPyflakes(
        self, paths: List[str], stdin: Optional[str] = None
    ) -> Tuple[str, str, int]:
        try:
            with SysStreamCapturing(stdin) as capture:
                main(args=paths)
        except SystemExit as e:
            self.assertIsInstance(e.code, bool)
            rv = int(e.code)
            return (capture.output, capture.error, rv)
        else:
            raise RuntimeError('SystemExit not raised')
