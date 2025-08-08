import re
import sys

class Reporter:
    def __init__(self, warningStream: 'file', errorStream: 'file') -> None:
        self._stdout = warningStream
        self._stderr = errorStream

    def unexpectedError(self, filename: str, msg: str) -> None:
        self._stderr.write(f'{filename}: {msg}\n')

    def syntaxError(self, filename: str, msg: str, lineno: int, offset: int, text: str) -> None:
        line = text.splitlines()[-1]
        if offset is not None:
            if sys.version_info < (3, 8):
                offset = offset - (len(text) - len(line)) + 1
            self._stderr.write(f'{filename}:{lineno}:{offset}: {msg}\n')
        else:
            self._stderr.write(f'{filename}:{lineno}: {msg}\n')
        self._stderr.write(line + '\n')
        if offset is not None:
            self._stderr.write(re.sub('\\S', ' ', line[:offset - 1]) + '^\n')

    def flake(self, message: 'pyflakes.messages.Message') -> None:
        self._stdout.write(str(message) + '\n')

def _makeDefaultReporter() -> Reporter:
    return Reporter(sys.stdout, sys.stderr)
