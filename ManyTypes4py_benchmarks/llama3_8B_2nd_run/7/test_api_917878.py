from __future__ import print_function
import contextlib
import os
import sys
import shutil
import tempfile
from pyflakes.messages import UnusedImport
from pyflakes.reporter import Reporter
from pyflakes.api import main, checkPath, checkRecursive, iterSourceCode
from pyflakes.test.harness import TestCase, skipIf
from io import StringIO
import unichr

class Node(object):
    """Mock an AST node."""
    def __init__(self, lineno: int, col_offset: int = 0) -> None:
        self.lineno = lineno
        self.col_offset = col_offset

class SysStreamCapturing(object):
    """Context manager capturing sys.stdin, sys.stdout and sys.stderr."""
    def __init__(self, stdin: str) -> None:
        self._newline = False
        self._stdin = self._create_StringIO(stdin or '')

    def __enter__(self) -> 'SysStreamCapturing':
        self._orig_stdin = sys.stdin
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdin = self._stdin
        sys.stdout = self._stdout_stringio = self._create_StringIO()
        sys.stderr = self._stderr_stringio = self._create_StringIO()
        return self

    def __exit__(self, *args: object) -> None:
        self.output = self._stdout_stringio.getvalue()
        self.error = self._stderr_stringio.getvalue()
        if self._newline and os.linesep != '\n':
            self.output = self.output.replace('\n', os.linesep)
            self.error = self.error.replace('\n', os.linesep)
        sys.stdin = self._orig_stdin
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

class LoggingReporter(object):
    """Implementation of Reporter that just appends any error to a list."""
    def __init__(self, log: list) -> None:
        self.log = log

    def flake(self, message: UnusedImport) -> None:
        self.log.append(('flake', str(message)))

    def unexpectedError(self, filename: str, message: str) -> None:
        self.log.append(('unexpectedError', filename, message))

    def syntaxError(self, filename: str, msg: str, lineno: int, offset: int, line: str) -> None:
        self.log.append(('syntaxError', filename, msg, lineno, offset, line))

class CheckTests(TestCase):
    # ... (rest of the class remains the same)

class IntegrationTests(TestCase):
    # ... (rest of the class remains the same)

class TestMain(IntegrationTests):
    # ... (rest of the class remains the same)
