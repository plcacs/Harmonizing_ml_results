import os
import re
import sys
from typing import List, Dict

sys.path.insert(0, os.path.dirname(__file__))
import _ast
import pycodestyle as pep8
import pyflakes.checker as pyflakes

if sys.version_info < (2, 7):
    def cmp_to_key(mycmp) -> 'K':
        ...

def parse_errors(errors: List, explicit_ignore: List) -> List[Dict]:
    ...

class LintError:
    def __init__(self, filename: str, loc: 'FakeLoc', level: str, message: str, message_args: tuple, **kwargs):
        ...

class Pep8Error(LintError):
    def __init__(self, filename: str, loc: 'FakeLoc', offset: int, code: str, text: str):
        ...

class Pep8Warning(LintError):
    def __init__(self, filename: str, loc: 'FakeLoc', offset: int, code: str, text: str):
        ...

class PythonError(LintError):
    def __init__(self, filename: str, loc: 'FakeLoc', text: str):
        ...

class OffsetError(LintError):
    def __init__(self, filename: str, loc: 'FakeLoc', text: str, offset: int):
        ...

class Linter:
    def __init__(self):
        ...

    def pyflakes_check(self, code: str, filename: str, ignore: List) -> List:
        ...

    def pep8_check(self, code: str, filename: str, rcfile: str, ignore: List, max_line_length: int) -> List:
        ...

    def run_linter(self, settings: Dict, code: str, filename: str) -> List:
        ...

    def sort_errors(self, errors: List) -> None:
        ...

    def prepare_error_level(self, error: LintError) -> str:
        ...

    def parse_errors(self, errors: List, explicit_ignore: List) -> List[Dict]:
        ...

    def _handle_syntactic_error(self, code: str, filename: str) -> List:
        ...
