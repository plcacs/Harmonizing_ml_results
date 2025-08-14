import re
import sys
from typing import TextIO, Optional, Any


class Reporter(object):
    """
    Formats the results of pyflakes checks to users.
    """

    def __init__(self, warningStream: TextIO, errorStream: TextIO) -> None:
        """
        Construct a Reporter.

        @param warningStream: A file-like object where warnings will be
            written to.  The stream's write method must accept unicode.
            sys.stdout is a good value.
        @param errorStream: A file-like object where error output will be
            written to.  The stream's write method must accept unicode.
            sys.stderr is a good value.
        """
        self._stdout: TextIO = warningStream
        self._stderr: TextIO = errorStream

    def unexpectedError(self, filename: str, msg: str) -> None:
        """
        An unexpected error occurred trying to process filename.

        @param filename: The path to a file that we could not process.
        @param msg: A message explaining the problem.
        """
        self._stderr.write("%s: %s\n" % (filename, msg))

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
            offset = offset - (len(text) - len(line))
            self._stderr.write('%s:%d:%d: %s\n' % (filename, lineno, offset + 1, msg))
        else:
            self._stderr.write('%s:%d: %s\n' % (filename, lineno, msg))
        self._stderr.write(line)
        self._stderr.write('\n')
        if offset is not None:
            self._stderr.write(re.sub(r'\S', ' ', line[:offset]) + "^\n")

    def flake(self, message: Any) -> None:
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