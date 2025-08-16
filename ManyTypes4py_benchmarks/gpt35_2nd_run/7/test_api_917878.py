import contextlib
import os
import sys
import shutil
import subprocess
import tempfile
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
ERROR_HAS_COL_NUM: bool = sys.version_info >= (3, 2) or PYPY

def withStderrTo(stderr: StringIO, f, *args, **kwargs):
    outer, sys.stderr = (sys.stderr, stderr)
    try:
        return f(*args, **kwargs)
    finally:
        sys.stderr = outer

class Node:
    def __init__(self, lineno: int, col_offset: int = 0):
        self.lineno = lineno
        self.col_offset = col_offset

class SysStreamCapturing:
    def _create_StringIO(self, buffer=None):
        try:
            return StringIO(buffer, newline=os.linesep)
        except TypeError:
            self._newline = True
            if buffer is None:
                return StringIO()
            else:
                return StringIO(buffer)

    def __init__(self, stdin: str):
        self._newline = False
        self._stdin = self._create_StringIO(stdin or '')

    def __enter__(self):
        self._orig_stdin = sys.stdin
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdin = self._stdin
        sys.stdout = self._stdout_stringio = self._create_StringIO()
        sys.stderr = self._stderr_stringio = self._create_StringIO()
        return self

    def __exit__(self, *args):
        self.output = self._stdout_stringio.getvalue()
        self.error = self._stderr_stringio.getvalue()
        if self._newline and os.linesep != '\n':
            self.output = self.output.replace('\n', os.linesep)
            self.error = self.error.replace('\n', os.linesep)
        sys.stdin = self._orig_stdin
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

class LoggingReporter:
    def __init__(self, log: list):
        self.log = log

    def flake(self, message):
        self.log.append(('flake', str(message)))

    def unexpectedError(self, filename: str, message: str):
        self.log.append(('unexpectedError', filename, message))

    def syntaxError(self, filename: str, msg: str, lineno: int, offset: int, line: str):
        self.log.append(('syntaxError', filename, msg, lineno, offset, line))

class TestIterSourceCode(TestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def makeEmptyFile(self, *parts) -> str:
        assert parts
        fpath = os.path.join(self.tempdir, *parts)
        open(fpath, 'a').close()
        return fpath

    def test_emptyDirectory(self):
        self.assertEqual(list(iterSourceCode([self.tempdir])), [])

    def test_singleFile(self):
        childpath = self.makeEmptyFile('foo.py')
        self.assertEqual(list(iterSourceCode([self.tempdir])), [childpath])

    def test_onlyPythonSource(self):
        self.makeEmptyFile('foo.pyc')
        self.assertEqual(list(iterSourceCode([self.tempdir])), [])

    def test_recurses(self):
        os.mkdir(os.path.join(self.tempdir, 'foo'))
        apath = self.makeEmptyFile('foo', 'a.py')
        self.makeEmptyFile('foo', 'a.py~')
        os.mkdir(os.path.join(self.tempdir, 'bar'))
        bpath = self.makeEmptyFile('bar', 'b.py')
        cpath = self.makeEmptyFile('c.py')
        self.assertEqual(sorted(iterSourceCode([self.tempdir])), sorted([apath, bpath, cpath]))

    def test_shebang(self):
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
        self.assertEqual(sorted(iterSourceCode([self.tempdir])), sorted([python, python2, python3, pythonw, python3args, python2u, python3d, python38m, python27]))

    def test_multipleDirectories(self):
        foopath = os.path.join(self.tempdir, 'foo')
        barpath = os.path.join(self.tempdir, 'bar')
        os.mkdir(foopath)
        apath = self.makeEmptyFile('foo', 'a.py')
        os.mkdir(barpath)
        bpath = self.makeEmptyFile('bar', 'b.py')
        self.assertEqual(sorted(iterSourceCode([foopath, barpath])), sorted([apath, bpath]))

    def test_explicitFiles(self):
        epath = self.makeEmptyFile('e.py')
        self.assertEqual(list(iterSourceCode([epath])), [epath])

class TestReporter(TestCase):
    def test_syntaxError(self):
        err = StringIO()
        reporter = Reporter(None, err)
        reporter.syntaxError('foo.py', 'a problem', 3, 8 if sys.version_info >= (3, 8) else 7, 'bad line of source')
        self.assertEqual('foo.py:3:8: a problem\nbad line of source\n       ^\n', err.getvalue())

    def test_syntaxErrorNoOffset(self):
        err = StringIO()
        reporter = Reporter(None, err)
        reporter.syntaxError('foo.py', 'a problem', 3, None, 'bad line of source')
        self.assertEqual('foo.py:3: a problem\nbad line of source\n', err.getvalue())

    def test_multiLineSyntaxError(self):
        err = StringIO()
        lines = ['bad line of source', 'more bad lines of source']
        reporter = Reporter(None, err)
        reporter.syntaxError('foo.py', 'a problem', 3, len(lines[0]) + 7, '\n'.join(lines))
        column = 25 if sys.version_info >= (3, 8) else 7
        self.assertEqual('foo.py:3:%d: a problem\n' % column + lines[-1] + '\n' + ' ' * (column - 1) + '^\n', err.getvalue())

    def test_unexpectedError(self):
        err = StringIO()
        reporter = Reporter(None, err)
        reporter.unexpectedError('source.py', 'error message')
        self.assertEqual('source.py: error message\n', err.getvalue())

    def test_flake(self):
        out = StringIO()
        reporter = Reporter(out, None)
        message = UnusedImport('foo.py', Node(42), 'bar')
        reporter.flake(message)
        self.assertEqual(out.getvalue(), '%s\n' % (message,))

class CheckTests(TestCase):
    @contextlib.contextmanager
    def makeTempFile(self, content: str):
        fd, name = tempfile.mkstemp()
        try:
            with os.fdopen(fd, 'wb') as f:
                if not hasattr(content, 'decode'):
                    content = content.encode('ascii')
                f.write(content)
            yield name
        finally:
            os.remove(name)

    def assertHasErrors(self, path: str, errorList: list):
        err = StringIO()
        count = withStderrTo(err, checkPath, path)
        self.assertEqual((count, err.getvalue()), (len(errorList), ''.join(errorList)))

    def getErrors(self, path: str):
        log = []
        reporter = LoggingReporter(log)
        count = checkPath(path, reporter)
        return (count, log)

    def test_legacyScript(self):
        from pyflakes.scripts import pyflakes as script_pyflakes
        self.assertIs(script_pyflakes.checkPath, checkPath)

    def test_missingTrailingNewline(self):
        with self.makeTempFile('def foo():\n\tpass\n\t') as fName:
            self.assertHasErrors(fName, [])

    def test_checkPathNonExisting(self):
        count, errors = self.getErrors('extremo')
        self.assertEqual(count, 1)
        self.assertEqual(errors, [('unexpectedError', 'extremo', 'No such file or directory')])

    def test_multilineSyntaxError(self):
        source = "def foo():\n    '''\n\ndef bar():\n    pass\n\ndef baz():\n    '''quux'''\n"

        def evaluate(source):
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
            self.assertHasErrors(sourcePath, ["%s:8:%d: %s\n    '''quux'''\n%s^\n" % (sourcePath, column, message, ' ' * (column - 1))])

    def test_eofSyntaxError(self):
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
            expected = '{}:1:{}: {}\ndef foo(\n{}^\n'.format(sourcePath, column, msg, spaces)
            self.assertHasErrors(sourcePath, [expected])

    def test_eofSyntaxErrorWithTab(self):
        with self.makeTempFile('if True:\n\tfoo =') as sourcePath:
            column = 6 if PYPY else 7
            last_line = '\t    ^' if PYPY else '\t     ^'
            self.assertHasErrors(sourcePath, ['%s:2:%s: invalid syntax\n\tfoo =\n%s\n' % (sourcePath, column, last_line)])

    def test_nonDefaultFollowsDefaultSyntaxError(self):
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
                columnstr = '%d:' % column
            else:
                last_line = columnstr = ''
            self.assertHasErrors(sourcePath, ['%s:1:%s non-default argument follows default argument\ndef foo(bar=baz, bax):\n%s' % (sourcePath, columnstr, last_line)])

    def test_nonKeywordAfterKeywordSyntaxError(self):
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
                columnstr = '%d:' % column
            else:
                last_line = columnstr = ''
            if sys.version_info >= (3, 5):
                message = 'positional argument follows keyword argument'
            else:
                message = 'non-keyword arg after keyword arg'
            self.assertHasErrors(sourcePath, ['%s:1:%s %s\nfoo(bar=baz, bax)\n%s' % (sourcePath, columnstr, message, last_line)])

    def test_invalidEscape(self):
        ver = sys.version_info
        with self.makeTempFile("foo = '\\xyz'") as sourcePath:
            if ver < (3,):
                decoding_error = '%s: problem decoding source\n' % (sourcePath,)
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
                    last_line = '%s^\n' % (' ' * (column - 1))
                else:
                    last_line = ''
                decoding_error = "%s:1:%d: (unicode error) 'unicodeescape' codec can't decode bytes in position 0-%d: truncated \\xXX escape\nfoo = '\\xyz'\n%s" % (sourcePath, column, position_end, last_line)
            self.assertHasErrors(sourcePath, [decoding_error])

    @skipIf(sys.platform == 'win32', 'unsupported on Windows')
    def test_permissionDenied(self):
        if os.getuid() == 0:
            self.skipTest('root user can access all files regardless of permissions')
        with self.makeTempFile('') as sourcePath:
            os.chmod(sourcePath, 0)
            count, errors = self.getErrors(sourcePath)
            self.assertEqual(count, 1)
            self.assertEqual(errors, [('unexpectedError', sourcePath, 'Permission denied')])

    def test_pyflakesWarning(self):
        with self.makeTempFile('import foo') as sourcePath:
            count, errors = self.getErrors(sourcePath)
            self.assertEqual(count, 1)
            self.assertEqual(errors, [('flake', str(UnusedImport(sourcePath, Node(1), 'foo'))])

    def test_encodedFileUTF8(self):
        SNOWMAN = unichr(9731)
        source = ('# coding: utf-8\nx = "%s"\n' % SNOWMAN).encode('utf-8')
        with self.makeTempFile(source) as sourcePath:
            self.assertHasErrors(sourcePath, [])

    def test_CRLFLineEndings(self):
        with self.makeTempFile('x = 42\r\n') as sourcePath:
            self.assertHasErrors(sourcePath, [])

    def test_misencodedFileUTF8(self):
        SNOWMAN = unichr(9731)
        source = ('# coding: ascii\nx = "%s"\n' % SNOWMAN).encode('utf-8')
        with self.makeTempFile(source) as sourcePath:
            if PYPY and sys.version_info < (3,):
                message = "'ascii' codec can't decode byte 0xe2 in position 21: ordinal not in range(128)"
                result = '%s:0:0: %s\nx = "â\x98\x83"\n        ^\n' % (sourcePath, message)
            else:
                message = 'problem decoding source'
                result = '%s: problem decoding source\n' % (sourcePath,)
            self.assertHasErrors(sourcePath, [result])

    def test_misencodedFileUTF16(self):
        SNOWMAN = unichr(9731)
        source = ('# coding: ascii\nx = "%s"\n' % SNOWMAN).encode('utf-16')
        with self.makeTempFile(source) as sourcePath:
            self.assertHasErrors(sourcePath, ['%s: problem decoding source\n' % (sourcePath,)])

    def test_checkRecursive(self):
        tempdir = tempfile.mkdtemp()
        try:
            os.mkdir(os.path.join(tempdir, 'foo'))
            file1 = os.path.join(tempdir, 'foo', 'bar.py')
            with open(file1, 'wb') as fd:
                fd.write('import baz\n'.encode('ascii'))
            file2 = os.path.join(tempdir, 'baz.py')
            with open(file2, 'wb') as fd:
                fd.write('import contraband