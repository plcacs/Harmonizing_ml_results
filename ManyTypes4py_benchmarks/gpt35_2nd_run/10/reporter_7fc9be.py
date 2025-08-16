import re
import sys

class Reporter:
    def __init__(self, warningStream: 'file', errorStream: 'file') -> None:
        self._stdout: 'file' = warningStream
        self._stderr: 'file' = errorStream

    def unexpectedError(self, filename: str, msg: str) -> None:
        self._stderr.write('%s: %s\n' % (filename, msg))

    def syntaxError(self, filename: str, msg: str, lineno: int, offset: int, text: str) -> None:
        line = text.splitlines()[-1]
        if offset is not None:
            offset = offset - (len(text) - len(line))
            self._stderr.write('%s:%d:%d: %s\n' % (filename, lineno, offset + 1, msg))
        else:
            self._stderr.write('%s:%d: %s\n' % (filename, lineno, msg))
        self._stderr.write(line)
        self._stderr.write('\n')
        if offset is not None:
            self._stderr.write(re.sub('\\S', ' ', line[:offset]) + '^\n')

    def flake(self, message: 'pyflakes.messages.Message') -> None:
        self._stdout.write(str(message))
        self._stdout.write('\n')

def _makeDefaultReporter() -> Reporter:
    return Reporter(sys.stdout, sys.stderr)
