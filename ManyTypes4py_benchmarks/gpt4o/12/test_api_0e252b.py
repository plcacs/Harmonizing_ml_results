import os
import sys
import shutil
import subprocess
import tempfile
from typing import List, Tuple, Union
from pyflakes.messages import UnusedImport
from pyflakes.reporter import Reporter
from pyflakes.api import checkPath, checkRecursive, iterSourceCode
from pyflakes.test.harness import TestCase, skipIf
if sys.version_info < (3,):
    from cStringIO import StringIO
else:
    from io import StringIO
    unichr = chr

def withStderrTo(stderr: StringIO, f: callable, *args: object, **kwargs: object) -> object:
    outer, sys.stderr = (sys.stderr, stderr)
    try:
        return f(*args, **kwargs)
    finally:
        sys.stderr = outer

class Node(object):
    def __init__(self, lineno: int, col_offset: int = 0) -> None:
        self.lineno = lineno
        self.col_offset = col_offset

class LoggingReporter(object):
    def __init__(self, log: List[Tuple[str, ...]]) -> None:
        self.log = log

    def flake(self, message: object) -> None:
        self.log.append(('flake', str(message)))

    def unexpectedError(self, filename: str, message: str) -> None:
        self.log.append(('unexpectedError', filename, message))

    def syntaxError(self, filename: str, msg: str, lineno: int, offset: Union[int, None], line: str) -> None:
        self.log.append(('syntaxError', filename, msg, lineno, offset, line))

class TestIterSourceCode(TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tempdir)

    def makeEmptyFile(self, *parts: str) -> str:
        assert parts
        fpath = os.path.join(self.tempdir, *parts)
        fd = open(fpath, 'a')
        fd.close()
        return fpath

    def test_emptyDirectory(self) -> None:
        self.assertEqual(list(iterSourceCode([self.tempdir])), [])

    def test_singleFile(self) -> None:
        childpath = self.makeEmptyFile('foo.py')
        self.assertEqual(list(iterSourceCode([self.tempdir])), [childpath])

    def test_onlyPythonSource(self) -> None:
        self.makeEmptyFile('foo.pyc')
        self.assertEqual(list(iterSourceCode([self.tempdir])), [])

    def test_recurses(self) -> None:
        os.mkdir(os.path.join(self.tempdir, 'foo'))
        apath = self.makeEmptyFile('foo', 'a.py')
        os.mkdir(os.path.join(self.tempdir, 'bar'))
        bpath = self.makeEmptyFile('bar', 'b.py')
        cpath = self.makeEmptyFile('c.py')
        self.assertEqual(sorted(iterSourceCode([self.tempdir])), sorted([apath, bpath, cpath]))

    def test_multipleDirectories(self) -> None:
        foopath = os.path.join(self.tempdir, 'foo')
        barpath = os.path.join(self.tempdir, 'bar')
        os.mkdir(foopath)
        apath = self.makeEmptyFile('foo', 'a.py')
        os.mkdir(barpath)
        bpath = self.makeEmptyFile('bar', 'b.py')
        self.assertEqual(sorted(iterSourceCode([foopath, barpath])), sorted([apath, bpath]))

    def test_explicitFiles(self) -> None:
        epath = self.makeEmptyFile('e.py')
        self.assertEqual(list(iterSourceCode([epath])), [epath])

class TestReporter(TestCase):
    def test_syntaxError(self) -> None:
        err = StringIO()
        reporter = Reporter(None, err)
        reporter.syntaxError('foo.py', 'a problem', 3, 7, 'bad line of source')
        self.assertEqual('foo.py:3:8: a problem\nbad line of source\n       ^\n', err.getvalue())

    def test_syntaxErrorNoOffset(self) -> None:
        err = StringIO()
        reporter = Reporter(None, err)
        reporter.syntaxError('foo.py', 'a problem', 3, None, 'bad line of source')
        self.assertEqual('foo.py:3: a problem\nbad line of source\n', err.getvalue())

    def test_multiLineSyntaxError(self) -> None:
        err = StringIO()
        lines = ['bad line of source', 'more bad lines of source']
        reporter = Reporter(None, err)
        reporter.syntaxError('foo.py', 'a problem', 3, len(lines[0]) + 7, '\n'.join(lines))
        self.assertEqual('foo.py:3:7: a problem\n' + lines[-1] + '\n' + '      ^\n', err.getvalue())

    def test_unexpectedError(self) -> None:
        err = StringIO()
        reporter = Reporter(None, err)
        reporter.unexpectedError('source.py', 'error message')
        self.assertEqual('source.py: error message\n', err.getvalue())

    def test_flake(self) -> None:
        out = StringIO()
        reporter = Reporter(out, None)
        message = UnusedImport('foo.py', Node(42), 'bar')
        reporter.flake(message)
        self.assertEqual(out.getvalue(), '%s\n' % (message,))

class CheckTests(TestCase):
    def makeTempFile(self, content: Union[str, bytes]) -> str:
        _, fpath = tempfile.mkstemp()
        if not hasattr(content, 'decode'):
            content = content.encode('ascii')
        fd = open(fpath, 'wb')
        fd.write(content)
        fd.close()
        return fpath

    def assertHasErrors(self, path: str, errorList: List[str]) -> None:
        err = StringIO()
        count = withStderrTo(err, checkPath, path)
        self.assertEqual((count, err.getvalue()), (len(errorList), ''.join(errorList)))

    def getErrors(self, path: str) -> Tuple[int, List[Tuple[str, ...]]]:
        log = []
        reporter = LoggingReporter(log)
        count = checkPath(path, reporter)
        return (count, log)

    def test_legacyScript(self) -> None:
        from pyflakes.scripts import pyflakes as script_pyflakes
        self.assertIs(script_pyflakes.checkPath, checkPath)

    def test_missingTrailingNewline(self) -> None:
        fName = self.makeTempFile('def foo():\n\tpass\n\t')
        self.assertHasErrors(fName, [])

    def test_checkPathNonExisting(self) -> None:
        count, errors = self.getErrors('extremo')
        self.assertEqual(count, 1)
        self.assertEqual(errors, [('unexpectedError', 'extremo', 'No such file or directory')])

    def test_multilineSyntaxError(self) -> None:
        source = "def foo():\n    '''\n\ndef bar():\n    pass\n\ndef baz():\n    '''quux'''\n"

        def evaluate(source: str) -> None:
            exec(source)
        try:
            evaluate(source)
        except SyntaxError:
            e = sys.exc_info()[1]
            self.assertTrue(e.text.count('\n') > 1)
        else:
            self.fail()
        sourcePath = self.makeTempFile(source)
        self.assertHasErrors(sourcePath, ["%s:8:11: invalid syntax\n    '''quux'''\n          ^\n" % (sourcePath,)])

    def test_eofSyntaxError(self) -> None:
        sourcePath = self.makeTempFile('def foo(')
        self.assertHasErrors(sourcePath, ['%s:1:9: unexpected EOF while parsing\ndef foo(\n        ^\n' % (sourcePath,)])

    def test_eofSyntaxErrorWithTab(self) -> None:
        sourcePath = self.makeTempFile('if True:\n\tfoo =')
        self.assertHasErrors(sourcePath, ['%s:2:7: invalid syntax\n\tfoo =\n\t     ^\n' % (sourcePath,)])

    def test_nonDefaultFollowsDefaultSyntaxError(self) -> None:
        source = 'def foo(bar=baz, bax):\n    pass\n'
        sourcePath = self.makeTempFile(source)
        last_line = '       ^\n' if sys.version_info >= (3, 2) else ''
        column = '8:' if sys.version_info >= (3, 2) else ''
        self.assertHasErrors(sourcePath, ['%s:1:%s non-default argument follows default argument\ndef foo(bar=baz, bax):\n%s' % (sourcePath, column, last_line)])

    def test_nonKeywordAfterKeywordSyntaxError(self) -> None:
        source = 'foo(bar=baz, bax)\n'
        sourcePath = self.makeTempFile(source)
        last_line = '            ^\n' if sys.version_info >= (3, 2) else ''
        column = '13:' if sys.version_info >= (3, 2) else ''
        if sys.version_info >= (3, 5):
            message = 'positional argument follows keyword argument'
        else:
            message = 'non-keyword arg after keyword arg'
        self.assertHasErrors(sourcePath, ['%s:1:%s %s\nfoo(bar=baz, bax)\n%s' % (sourcePath, column, message, last_line)])

    def test_invalidEscape(self) -> None:
        ver = sys.version_info
        sourcePath = self.makeTempFile("foo = '\\xyz'")
        if ver < (3,):
            decoding_error = '%s: problem decoding source\n' % (sourcePath,)
        else:
            last_line = '      ^\n' if ver >= (3, 2) else ''
            col = 1 if ver >= (3, 3, 1) or (3, 2, 4) <= ver < (3, 3) else 2
            decoding_error = "%s:1:7: (unicode error) 'unicodeescape' codec can't decode bytes in position 0-%d: truncated \\xXX escape\nfoo = '\\xyz'\n%s" % (sourcePath, col, last_line)
        self.assertHasErrors(sourcePath, [decoding_error])

    @skipIf(sys.platform == 'win32', 'unsupported on Windows')
    def test_permissionDenied(self) -> None:
        sourcePath = self.makeTempFile('')
        os.chmod(sourcePath, 0)
        count, errors = self.getErrors(sourcePath)
        self.assertEqual(count, 1)
        self.assertEqual(errors, [('unexpectedError', sourcePath, 'Permission denied')])

    def test_pyflakesWarning(self) -> None:
        sourcePath = self.makeTempFile('import foo')
        count, errors = self.getErrors(sourcePath)
        self.assertEqual(count, 1)
        self.assertEqual(errors, [('flake', str(UnusedImport(sourcePath, Node(1), 'foo')))])

    def test_encodedFileUTF8(self) -> None:
        SNOWMAN = unichr(9731)
        source = ('# coding: utf-8\nx = "%s"\n' % SNOWMAN).encode('utf-8')
        sourcePath = self.makeTempFile(source)
        self.assertHasErrors(sourcePath, [])

    def test_CRLFLineEndings(self) -> None:
        sourcePath = self.makeTempFile('x = 42\r\n')
        self.assertHasErrors(sourcePath, [])

    def test_misencodedFileUTF8(self) -> None:
        SNOWMAN = unichr(9731)
        source = ('# coding: ascii\nx = "%s"\n' % SNOWMAN).encode('utf-8')
        sourcePath = self.makeTempFile(source)
        self.assertHasErrors(sourcePath, ['%s: problem decoding source\n' % (sourcePath,)])

    def test_misencodedFileUTF16(self) -> None:
        SNOWMAN = unichr(9731)
        source = ('# coding: ascii\nx = "%s"\n' % SNOWMAN).encode('utf-16')
        sourcePath = self.makeTempFile(source)
        self.assertHasErrors(sourcePath, ['%s: problem decoding source\n' % (sourcePath,)])

    def test_checkRecursive(self) -> None:
        tempdir = tempfile.mkdtemp()
        os.mkdir(os.path.join(tempdir, 'foo'))
        file1 = os.path.join(tempdir, 'foo', 'bar.py')
        fd = open(file1, 'wb')
        fd.write('import baz\n'.encode('ascii'))
        fd.close()
        file2 = os.path.join(tempdir, 'baz.py')
        fd = open(file2, 'wb')
        fd.write('import contraband'.encode('ascii'))
        fd.close()
        log = []
        reporter = LoggingReporter(log)
        warnings = checkRecursive([tempdir], reporter)
        self.assertEqual(warnings, 2)
        self.assertEqual(sorted(log), sorted([('flake', str(UnusedImport(file1, Node(1), 'baz'))), ('flake', str(UnusedImport(file2, Node(1), 'contraband')))]))

class IntegrationTests(TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.mkdtemp()
        self.tempfilepath = os.path.join(self.tempdir, 'temp')

    def tearDown(self) -> None:
        shutil.rmtree(self.tempdir)

    def getPyflakesBinary(self) -> str:
        import pyflakes
        package_dir = os.path.dirname(pyflakes.__file__)
        return os.path.join(package_dir, '..', 'bin', 'pyflakes')

    def runPyflakes(self, paths: List[str], stdin: Union[bytes, None] = None) -> Tuple[str, str, int]:
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join(sys.path)
        command = [sys.executable, self.getPyflakesBinary()]
        command.extend(paths)
        if stdin:
            p = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = p.communicate(stdin)
        else:
            p = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = p.communicate()
        rv = p.wait()
        if sys.version_info >= (3,):
            stdout = stdout.decode('utf-8')
            stderr = stderr.decode('utf-8')
        return (stdout, stderr, rv)

    def test_goodFile(self) -> None:
        fd = open(self.tempfilepath, 'a')
        fd.close()
        d = self.runPyflakes([self.tempfilepath])
        self.assertEqual(d, ('', '', 0))

    def test_fileWithFlakes(self) -> None:
        fd = open(self.tempfilepath, 'wb')
        fd.write('import contraband\n'.encode('ascii'))
        fd.close()
        d = self.runPyflakes([self.tempfilepath])
        expected = UnusedImport(self.tempfilepath, Node(1), 'contraband')
        self.assertEqual(d, ('%s%s' % (expected, os.linesep), '', 1))

    def test_errors(self) -> None:
        d = self.runPyflakes([self.tempfilepath])
        error_msg = '%s: No such file or directory%s' % (self.tempfilepath, os.linesep)
        self.assertEqual(d, ('', error_msg, 1))

    def test_readFromStdin(self) -> None:
        d = self.runPyflakes([], stdin='import contraband'.encode('ascii'))
        expected = UnusedImport('<stdin>', Node(1), 'contraband')
        self.assertEqual(d, ('%s%s' % (expected, os.linesep), '', 1))
