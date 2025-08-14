#!/usr/bin/env python
"""
Tests for L{pyflakes.scripts.pyflakes}.
"""

import os
import sys
import shutil
import subprocess
import tempfile
from io import StringIO
from typing import List, Tuple, Callable, Any, TypeVar, IO, Optional, Union

from pyflakes.messages import UnusedImport
from pyflakes.reporter import Reporter
from pyflakes.api import (
    checkPath,
    checkRecursive,
    iterSourceCode,
)
from pyflakes.test.harness import TestCase, skipIf

T = TypeVar('T')
unichr = chr


def withStderrTo(stderr: IO[str], f: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    Call C{f} with C{sys.stderr} redirected to C{stderr}.
    """
    outer, sys.stderr = sys.stderr, stderr
    try:
        return f(*args, **kwargs)
    finally:
        sys.stderr = outer


class Node(object):
    """
    Mock an AST node.
    """
    def __init__(self, lineno: int, col_offset: int = 0) -> None:
        self.lineno: int = lineno
        self.col_offset: int = col_offset


class LoggingReporter(object):
    """
    Implementation of Reporter that just appends any error to a list.
    """

    def __init__(self, log: List[Tuple[Any, ...]]) -> None:
        """
        Construct a C{LoggingReporter}.

        @param log: A list to append log messages to.
        """
        self.log: List[Tuple[Any, ...]] = log

    def flake(self, message: Any) -> None:
        self.log.append(('flake', str(message)))

    def unexpectedError(self, filename: str, message: str) -> None:
        self.log.append(('unexpectedError', filename, message))

    def syntaxError(self, filename: str, msg: str, lineno: int,
                    offset: Optional[int], line: str) -> None:
        self.log.append(('syntaxError', filename, msg, lineno, offset, line))


class TestIterSourceCode(TestCase):
    """
    Tests for L{iterSourceCode}.
    """

    def setUp(self) -> None:
        self.tempdir: str = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tempdir)

    def makeEmptyFile(self, *parts: str) -> str:
        assert parts
        fpath: str = os.path.join(self.tempdir, *parts)
        with open(fpath, 'a'):
            pass
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
        childpath: str = self.makeEmptyFile('foo.py')
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
        apath: str = self.makeEmptyFile('foo', 'a.py')
        os.mkdir(os.path.join(self.tempdir, 'bar'))
        bpath: str = self.makeEmptyFile('bar', 'b.py')
        cpath: str = self.makeEmptyFile('c.py')
        self.assertEqual(
            sorted(iterSourceCode([self.tempdir])),
            sorted([apath, bpath, cpath])
        )

    def test_multipleDirectories(self) -> None:
        """
        L{iterSourceCode} can be given multiple directories.  It will recurse
        into each of them.
        """
        foopath: str = os.path.join(self.tempdir, 'foo')
        barpath: str = os.path.join(self.tempdir, 'bar')
        os.mkdir(foopath)
        apath: str = self.makeEmptyFile('foo', 'a.py')
        os.mkdir(barpath)
        bpath: str = self.makeEmptyFile('bar', 'b.py')
        self.assertEqual(
            sorted(iterSourceCode([foopath, barpath])),
            sorted([apath, bpath])
        )

    def test_explicitFiles(self) -> None:
        """
        If one of the paths given to L{iterSourceCode} is not a directory but
        a file, it will include that in its output.
        """
        epath: str = self.makeEmptyFile('e.py')
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
        err: StringIO = StringIO()
        reporter: Reporter = Reporter(None, err)
        reporter.syntaxError('foo.py', 'a problem', 3, 7, 'bad line of source')
        self.assertEqual(
            ("foo.py:3:8: a problem\n"
             "bad line of source\n"
             "       ^\n"),
            err.getvalue()
        )

    def test_syntaxErrorNoOffset(self) -> None:
        """
        C{syntaxError} doesn't include a caret pointing to the error if
        C{offset} is passed as C{None}.
        """
        err: StringIO = StringIO()
        reporter: Reporter = Reporter(None, err)
        reporter.syntaxError('foo.py', 'a problem', 3, None, 'bad line of source')
        self.assertEqual(
            ("foo.py:3: a problem\n"
             "bad line of source\n"),
            err.getvalue()
        )

    def test_multiLineSyntaxError(self) -> None:
        """
        If there's a multi-line syntax error, then we only report the last
        line.  The offset is adjusted so that it is relative to the start of
        the last line.
        """
        err: StringIO = StringIO()
        lines: List[str] = [
            'bad line of source',
            'more bad lines of source',
        ]
        reporter: Reporter = Reporter(None, err)
        reporter.syntaxError('foo.py', 'a problem', 3, len(lines[0]) + 7,
                             '\n'.join(lines))
        self.assertEqual(
            ("foo.py:3:7: a problem\n" +
             lines[-1] + "\n" +
             "      ^\n"),
            err.getvalue()
        )

    def test_unexpectedError(self) -> None:
        """
        C{unexpectedError} reports an error processing a source file.
        """
        err: StringIO = StringIO()
        reporter: Reporter = Reporter(None, err)
        reporter.unexpectedError('source.py', 'error message')
        self.assertEqual('source.py: error message\n', err.getvalue())

    def test_flake(self) -> None:
        """
        C{flake} reports a code warning from Pyflakes.  It is exactly the
        str() of a L{pyflakes.messages.Message}.
        """
        out: StringIO = StringIO()
        reporter: Reporter = Reporter(out, None)
        message: UnusedImport = UnusedImport('foo.py', Node(42), 'bar')
        reporter.flake(message)
        self.assertEqual(out.getvalue(), f"{message}\n")


class CheckTests(TestCase):
    """
    Tests for L{check} and L{checkPath} which check a file for flakes.
    """

    def makeTempFile(self, content: Union[str, bytes]) -> str:
        """
        Make a temporary file containing C{content} and return a path to it.
        """
        _, fpath = tempfile.mkstemp()
        if not hasattr(content, 'decode'):
            content = content.encode('ascii')
        with open(fpath, 'wb') as fd:
            fd.write(content)
        return fpath

    def assertHasErrors(self, path: str, errorList: List[str]) -> None:
        """
        Assert that C{path} causes errors.

        @param path: A path to a file to check.
        @param errorList: A list of errors expected to be printed to stderr.
        """
        err: StringIO = StringIO()
        count: int = withStderrTo(err, checkPath, path)
        self.assertEqual(
            (count, err.getvalue()), (len(errorList), ''.join(errorList))
        )

    def getErrors(self, path: str) -> Tuple[int, List[Tuple[Any, ...]]]:
        """
        Get any warnings or errors reported by pyflakes for the file at C{path}.

        @param path: The path to a Python file on disk that pyflakes will check.
        @return: C{(count, log)}, where C{count} is the number of warnings or
            errors generated, and log is a list of those warnings, presented
            as structured data.  See L{LoggingReporter} for more details.
        """
        log: List[Tuple[Any, ...]] = []
        reporter: LoggingReporter = LoggingReporter(log)
        count: int = checkPath(path, reporter)
        return count, log

    def test_legacyScript(self) -> None:
        from pyflakes.scripts import pyflakes as script_pyflakes
        self.assertIs(script_pyflakes.checkPath, checkPath)

    def test_missingTrailingNewline(self) -> None:
        """
        Source which doesn't end with a newline shouldn't cause any
        exception to be raised nor an error indicator to be returned by
        L{check}.
        """
        fName: str = self.makeTempFile("def foo():\n\tpass\n\t")
        self.assertHasErrors(fName, [])

    def test_checkPathNonExisting(self) -> None:
        """
        L{checkPath} handles non-existing files.
        """
        count, errors = self.getErrors('extremo')
        self.assertEqual(count, 1)
        self.assertEqual(
            errors,
            [('unexpectedError', 'extremo', 'No such file or directory')]
        )

    def test_multilineSyntaxError(self) -> None:
        """
        Source which includes a syntax error which results in the raised
        L{SyntaxError.text} containing multiple lines of source are reported
        with only the last line of that source.
        """
        source: str = """\
def foo():
    '''

def bar():
    pass

def baz():
    '''quux'''
"""
        # Sanity check - SyntaxError.text should be multiple lines, if it
        # isn't, something this test was unprepared for has happened.
        def evaluate(source: str) -> None:
            exec(source)
        try:
            evaluate(source)
        except SyntaxError:
            e: Exception = sys.exc_info()[1]  # type: ignore
            self.assertTrue(e.text.count('\n') > 1)  # type: ignore
        else:
            self.fail()

        sourcePath: str = self.makeTempFile(source)
        self.assertHasErrors(
            sourcePath,
            [f"""\
{sourcePath}:8:11: invalid syntax
    '''quux'''
          ^
"""]
        )

    def test_eofSyntaxError(self) -> None:
        """
        The error reported for source files which end prematurely causing a
        syntax error reflects the cause for the syntax error.
        """
        sourcePath: str = self.makeTempFile("def foo(")
        self.assertHasErrors(
            sourcePath,
            [f"""\
{sourcePath}:1:9: unexpected EOF while parsing
def foo(
        ^
"""]
        )

    def test_eofSyntaxErrorWithTab(self) -> None:
        """
        The error reported for source files which end prematurely causing a
        syntax error reflects the cause for the syntax error.
        """
        sourcePath: str = self.makeTempFile("if True:\n\tfoo =")
        self.assertHasErrors(
            sourcePath,
            [f"""\
{sourcePath}:2:7: invalid syntax
\tfoo =
\t     ^
"""]
        )

    def test_nonDefaultFollowsDefaultSyntaxError(self) -> None:
        """
        Source which has a non-default argument following a default argument
        should include the line number of the syntax error.  However these
        exceptions do not include an offset.
        """
        source: str = """\
def foo(bar=baz, bax):
    pass
"""
        sourcePath: str = self.makeTempFile(source)
        last_line: str = '       ^\n' if sys.version_info >= (3, 2) else ''
        column: str = '8:' if sys.version_info >= (3, 2) else ''
        self.assertHasErrors(
            sourcePath,
            [f"""\
{sourcePath}:1:{column} non-default argument follows default argument
def foo(bar=baz, bax):
{last_line}"""]
        )

    def test_nonKeywordAfterKeywordSyntaxError(self) -> None:
        """
        Source which has a non-keyword argument after a keyword argument should
        include the line number of the syntax error.  However these exceptions
        do not include an offset.
        """
        source: str = """\
foo(bar=baz, bax)
"""
        sourcePath: str = self.makeTempFile(source)
        last_line: str = '            ^\n' if sys.version_info >= (3, 2) else ''
        column: str = '13:' if sys.version_info >= (3, 2) else ''

        if sys.version_info >= (3, 5):
            message: str = 'positional argument follows keyword argument'
        else:
            message = 'non-keyword arg after keyword arg'

        self.assertHasErrors(
            sourcePath,
            [f"""\
{sourcePath}:1:{column} {message}
foo(bar=baz, bax)
{last_line}"""]
        )

    def test_invalidEscape(self) -> None:
        """
        The invalid escape syntax raises ValueError in Python 2
        """
        ver = sys.version_info
        sourcePath: str = self.makeTempFile(r"foo = '\xyz'")
        if ver < (3,):
            decoding_error: str = f"{sourcePath}: problem decoding source\n"
        else:
            last_line: str = '      ^\n' if ver >= (3, 2) else ''
            col: int = 1 if ver >= (3, 3, 1) or ((3, 2, 4) <= ver < (3, 3)) else 2
            decoding_error = f"""{sourcePath}:1:7: (unicode error) 'unicodeescape' codec can't decode bytes in position 0-{col}: truncated \\xXX escape
foo = '\\xyz'
{last_line}"""
        self.assertHasErrors(
            sourcePath, [decoding_error]
        )

    @skipIf(sys.platform == 'win32', 'unsupported on Windows')
    def test_permissionDenied(self) -> None:
        """
        If the source file is not readable, this is reported on standard
        error.
        """
        sourcePath: str = self.makeTempFile('')
        os.chmod(sourcePath, 0)
        count, errors = self.getErrors(sourcePath)
        self.assertEqual(count, 1)
        self.assertEqual(
            errors,
            [('unexpectedError', sourcePath, "Permission denied")]
        )

    def test_pyflakesWarning(self) -> None:
        """
        If the source file has a pyflakes warning, this is reported as a
        'flake'.
        """
        sourcePath: str = self.makeTempFile("import foo")
        count, errors = self.getErrors(sourcePath)
        self.assertEqual(count, 1)
        self.assertEqual(
            errors, [('flake', str(UnusedImport(sourcePath, Node(1), 'foo')))]
        )

    def test_encodedFileUTF8(self) -> None:
        """
        If source file declares the correct encoding, no error is reported.
        """
        SNOWMAN: str = unichr(0x2603)
        source: bytes = (f"""\
# coding: utf-8
x = "{SNOWMAN}"
""").encode('utf-8')
        sourcePath: str = self.makeTempFile(source)
        self.assertHasErrors(sourcePath, [])

    def test_CRLFLineEndings(self) -> None:
        """
        Source files with Windows CR LF line endings are parsed successfully.
        """
        sourcePath: str = self.makeTempFile("x = 42\r\n")
        self.assertHasErrors(sourcePath, [])

    def test_misencodedFileUTF8(self) -> None:
        """
        If a source file contains bytes which cannot be decoded, this is
        reported on stderr.
        """
        SNOWMAN: str = unichr(0x2603)
        source: bytes = (f"""\
# coding: ascii
x = "{SNOWMAN}"
""").encode('utf-8')
        sourcePath: str = self.makeTempFile(source)
        self.assertHasErrors(
            sourcePath, [f"{sourcePath}: problem decoding source\n"]
        )

    def test_misencodedFileUTF16(self) -> None:
        """
        If a source file contains bytes which cannot be decoded, this is
        reported on stderr.
        """
        SNOWMAN: str = unichr(0x2603)
        source: bytes = (f"""\
# coding: ascii
x = "{SNOWMAN}"
""").encode('utf-16')
        sourcePath: str = self.makeTempFile(source)
        self.assertHasErrors(
            sourcePath, [f"{sourcePath}: problem decoding source\n"]
        )

    def test_checkRecursive(self) -> None:
        """
        L{checkRecursive} descends into each directory, finding Python files
        and reporting problems.
        """
        tempdir: str = tempfile.mkdtemp()
        os.mkdir(os.path.join(tempdir, 'foo'))
        file1: str = os.path.join(tempdir, 'foo', 'bar.py')
        with open(file1, 'wb') as fd:
            fd.write("import baz\n".encode('ascii'))
        file2: str = os.path.join(tempdir, 'baz.py')
        with open(file2, 'wb') as fd:
            fd.write("import contraband".encode('ascii'))
        log: List[Tuple[Any, ...]] = []
        reporter: LoggingReporter = LoggingReporter(log)
        warnings: int = checkRecursive([tempdir], reporter)
        self.assertEqual(warnings, 2)
        self.assertEqual(
            sorted(log),
            sorted([
                ('flake', str(UnusedImport(file1, Node(1), 'baz'))),
                ('flake', str(UnusedImport(file2, Node(1), 'contraband')))
            ])
        )


class IntegrationTests(TestCase):
    """
    Tests of the pyflakes script that actually spawn the script.
    """

    def setUp(self) -> None:
        self.tempdir: str = tempfile.mkdtemp()
        self.tempfilepath: str = os.path.join(self.tempdir, 'temp')

    def tearDown(self) -> None:
        shutil.rmtree(self.tempdir)

    def getPyflakesBinary(self) -> str:
        """
        Return the path to the pyflakes binary.
        """
        import pyflakes
        package_dir: str = os.path.dirname(pyflakes.__file__)
        return os.path.join(package_dir, '..', 'bin', 'pyflakes')

    def runPyflakes(self, paths: List[str], stdin: Optional[bytes] = None) -> Tuple[str, str, int]:
        """
        Launch a subprocess running C{pyflakes}.

        @param paths: Command-line arguments to pass to pyflakes.
        @param stdin: Input to be passed to the subprocess.
        @return: C{(stdout, stderr, returncode)} of the completed pyflakes
            process.
        """
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join(sys.path)
        command: List[str] = [sys.executable, self.getPyflakesBinary()]
        command.extend(paths)
        if stdin:
            p = subprocess.Popen(command, env=env, stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout_bytes, stderr_bytes = p.communicate(stdin)
        else:
            p = subprocess.Popen(command, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout_bytes, stderr_bytes = p.communicate()
        rv: int = p.wait()
        stdout: str = stdout_bytes.decode('utf-8')
        stderr: str = stderr_bytes.decode('utf-8')
        return stdout, stderr, rv

    def test_goodFile(self) -> None:
        """
        When a Python source file is all good, the return code is zero and no
        messages are printed to either stdout or stderr.
        """
        with open(self.tempfilepath, 'a'):
            pass
        d: Tuple[str, str, int] = self.runPyflakes([self.tempfilepath])
        self.assertEqual(d, ('', '', 0))

    def test_fileWithFlakes(self) -> None:
        """
        When a Python source file has warnings, the return code is non-zero
        and the warnings are printed to stdout.
        """
        with open(self.tempfilepath, 'wb') as fd:
            fd.write("import contraband\n".encode('ascii'))
        d: Tuple[str, str, int] = self.runPyflakes([self.tempfilepath])
        expected = UnusedImport(self.tempfilepath, Node(1), 'contraband')
        self.assertEqual(d, (f"{expected}{os.linesep}", '', 1))

    def test_errors(self) -> None:
        """
        When pyflakes finds errors with the files it's given, (if they don't
        exist, say), then the return code is non-zero and the errors are
        printed to stderr.
        """
        d: Tuple[str, str, int] = self.runPyflakes([self.tempfilepath])
        error_msg: str = f"{self.tempfilepath}: No such file or directory{os.linesep}"
        self.assertEqual(d, ('', error_msg, 1))

    def test_readFromStdin(self) -> None:
        """
        If no arguments are passed to C{pyflakes} then it reads from stdin.
        """
        d: Tuple[str, str, int] = self.runPyflakes([], stdin="import contraband".encode('ascii'))
        expected = UnusedImport('<stdin>', Node(1), 'contraband')
        self.assertEqual(d, (f"{expected}{os.linesep}", '', 1))
