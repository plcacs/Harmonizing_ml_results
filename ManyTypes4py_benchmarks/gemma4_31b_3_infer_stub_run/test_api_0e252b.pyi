"""
Tests for L{pyflakes.scripts.pyflakes}.
"""
import os
import sys
import subprocess
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union, TypeVar

from pyflakes.messages import UnusedImport
from pyflakes.reporter import Reporter
from pyflakes.api import checkPath, checkRecursive, iterSourceCode
from pyflakes.test.harness import TestCase, skipIf

T = TypeVar("T")

def withStderrTo(stderr: Any, f: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    Call C{f} with C{sys.stderr} redirected to C{stderr}.
    """
    ...

class Node:
    """
    Mock an AST node.
    """
    lineno: int
    col_offset: int

    def __init__(self, lineno: int, col_offset: int = 0) -> None: ...

class LoggingReporter:
    """
    Implementation of Reporter that just appends any error to a list.
    """
    log: List[Tuple[str, ...]]

    def __init__(self, log: List[Tuple[str, ...]]) -> None:
        """
        Construct a C{LoggingReporter}.

        @param log: A list to append log messages to.
        """
        ...

    def flake(self, message: Any) -> None: ...

    def unexpectedError(self, filename: str, message: str) -> None: ...

    def syntaxError(self, filename: str, msg: str, lineno: int, offset: Optional[int], line: str) -> None: ...

class TestIterSourceCode(TestCase):
    """
    Tests for L{iterSourceCode}.
    """
    tempdir: str

    def setUp(self) -> None: ...

    def tearDown(self) -> None: ...

    def makeEmptyFile(self, *parts: str) -> str: ...

    def test_emptyDirectory(self) -> None: ...

    def test_singleFile(self) -> None: ...

    def test_onlyPythonSource(self) -> None: ...

    def test_recurses(self) -> None: ...

    def test_multipleDirectories(self) -> None: ...

    def test_explicitFiles(self) -> None: ...

class TestReporter(TestCase):
    """
    Tests for L{Reporter}.
    """
    def test_syntaxError(self) -> None: ...

    def test_syntaxErrorNoOffset(self) -> None: ...

    def test_multiLineSyntaxError(self) -> None: ...

    def test_unexpectedError(self) -> None: ...

    def test_flake(self) -> None: ...

class CheckTests(TestCase):
    """
    Tests for L{check} and L{checkPath} which check a file for flakes.
    """
    def makeTempFile(self, content: Union[str, bytes]) -> str:
        """
        Make a temporary file containing C{content} and return a path to it.
        """
        ...

    def assertHasErrors(self, path: str, errorList: List[str]) -> None:
        """
        Assert that C{path} causes errors.

        @param path: A path to a file to check.
        @param errorList: A list of errors expected to be printed to stderr.
        """
        ...

    def getErrors(self, path: str) -> Tuple[int, List[Tuple[str, ...]]]:
        """
        Get any warnings or errors reported by pyflakes for the file at C{path}.

        @param path: The path to a Python file on disk that pyflakes will check.
        @return: C{(count, log)}, where C{count} is the number of warnings or
            errors generated, and log is a list of those warnings, presented
            as structured data.  See L{LoggingReporter} for more details.
        """
        ...

    def test_legacyScript(self) -> None: ...

    def test_missingTrailingNewline(self) -> None: ...

    def test_checkPathNonExisting(self) -> None: ...

    def test_multilineSyntaxError(self) -> None: ...

    def test_eofSyntaxError(self) -> None: ...

    def test_eofSyntaxErrorWithTab(self) -> None: ...

    def test_nonDefaultFollowsDefaultSyntaxError(self) -> None: ...

    def test_nonKeywordAfterKeywordSyntaxError(self) -> None: ...

    def test_invalidEscape(self) -> None: ...

    def test_permissionDenied(self) -> None: ...

    def test_pyflakesWarning(self) -> None: ...

    def test_encodedFileUTF8(self) -> None: ...

    def test_CRLFLineEndings(self) -> None: ...

    def test_misencodedFileUTF8(self) -> None: ...

    def test_misencodedFileUTF16(self) -> None: ...

    def test_checkRecursive(self) -> None: ...

class IntegrationTests(TestCase):
    """
    Tests of the pyflakes script that actually spawn the script.
    """
    tempdir: str
    tempfilepath: str

    def setUp(self) -> None: ...

    def tearDown(self) -> None: ...

    def getPyflakesBinary(self) -> str:
        """
        Return the path to the pyflakes binary.
        """
        ...

    def runPyflakes(self, paths: List[str], stdin: Optional[bytes] = ...) -> Tuple[str, str, int]:
        """
        Launch a subprocess running C{pyflakes}.

        @param args: Command-line arguments to pass to pyflakes.
        @param kwargs: Options passed on to C{subprocess.Popen}.
        @return: C{(returncode, stdout, stderr)} of the completed pyflakes
            process.
        """
        ...

    def test_goodFile(self) -> None: ...

    def test_fileWithFlakes(self) -> None: ...

    def test_errors(self) -> None: ...

    def test_readFromStdin(self) -> None: ...