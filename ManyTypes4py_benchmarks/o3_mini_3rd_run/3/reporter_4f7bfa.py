#!/usr/bin/env python3
"""
Provide the Reporter class.
"""
import re
import sys
from typing import IO, Optional
from pyflakes.messages import Message  # type: ignore

class Reporter(object):
    """
    Formats the results of pyflakes checks to users.
    """

    def __init__(self, warningStream: IO[str], errorStream: IO[str]) -> None:
        """
        Construct a Reporter.

        @param warningStream: A file-like object where warnings will be
            written to. The stream's write method must accept unicode.
            sys.stdout is a good value.
        @param errorStream: A file-like object where error output will be
            written to. The stream's write method must accept unicode.
            sys.stderr is a good value.
        """
        self._stdout = warningStream
        self._stderr = errorStream

    def unexpectedError(self, filename: str, msg: str) -> None:
        """
        An unexpected error occurred trying to process filename.

        @param filename: The path to a file that we could not process.
        @param msg: A message explaining the problem.
        """
        self._stderr.write('%s: %s\n' % (filename, msg))

    def syntaxError(self, filename: str, msg: str, lineno: int, offset: Optional[int], text: str) -> None:
        """
        There was a syntax error in filename.

        @param filename: The path to the file with the syntax error.
        @param msg: An explanation of the syntax error.
        @param lineno: The line number where the syntax error occurred.
        @param offset: The column on which the syntax error occurred, or None.
        @param text: The source code containing the syntax error.
        """
        line = text.splitlines()[-1]
        if offset is not None:
            if sys.version_info < (3, 8):
                offset = offset - (len(text) - len(line)) + 1
            self._stderr.write('%s:%d:%d: %s\n' % (filename, lineno, offset, msg))
        else:
            self._stderr.write('%s:%d: %s\n' % (filename, lineno, msg))
        self._stderr.write(line)
        self._stderr.write('\n')
        if offset is not None:
            self._stderr.write(re.sub('\\S', ' ', line[:offset - 1]) + '^\n')

    def flake(self, message: Message) -> None:
        """
        pyflakes found something wrong with the code.

        @param message: A pyflakes.messages.Message.
        """
        self._stdout.write(str(message))
        self._stdout.write('\n')

def _makeDefaultReporter() -> Reporter:
    """
    Make a reporter that can be used when no reporter is specified.
    """
    return Reporter(sys.stdout, sys.stderr)