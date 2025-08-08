import os
import sys
from functools import partial
from typing import List, Dict, Any, Union

sys.path.append(os.path.join(os.path.dirname(__file__), '../../anaconda_lib'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from import_validator import Validator
from linting.anaconda_pep8 import Pep8Linter
from lib.anaconda_handler import AnacondaHandler
from linting.anaconda_pyflakes import PyFlakesLinter
from linting.anaconda_mypy import MyPy as AnacondaMyPy
from linting.anaconda_pep257 import PEP257 as AnacondaPep257
from commands import PyFlakes, PEP257, PEP8, PyLint, ImportValidator, MyPy

try:
    from linting.anaconda_pylint import PyLinter
    from linting.anaconda_pylint import numversion
    PYLINT_AVAILABLE: bool = True
except ImportError:
    PYLINT_AVAILABLE: bool = False

class PythonLintHandler(AnacondaHandler):
    def __init__(self, command: str, data: Dict[str, Any], uid: int, vid: int, settings: Dict[str, Any], callback: Any, debug: bool = False) -> None:
        self.uid: int = uid
        self.vid: int = vid
        self.data: Dict[str, Any] = data
        self.debug: bool = debug
        self.callback: Any = callback
        self.command: str = command
        self.settings: Dict[str, Any] = settings
        self._linters: Dict[str, bool] = {'pyflakes': False, 'pylint': False, 'pep8': False, 'pep257': False, 'import_validator': False}
        self._errors: List[str] = []
        self._failures: List[str] = []

    def lint(self, code: str = None, filename: str = None) -> None:
        self._configure_linters()
        for linter_name, expected in self._linters.items():
            if expected is True:
                func = getattr(self, linter_name)
                func(code, filename)
        if len(self._errors) == 0 and len(self._failures) > 0:
            self.callback({'success': False, 'errors': '. '.join([str(e) for e in self._failures]), 'uid': self.uid, 'vid': self.vid})
            return
        self.callback({'success': True, 'errors': self._errors, 'uid': self.uid, 'vid': self.vid})

    def pyflakes(self, code: str = None, filename: str = None) -> None:
        lint: PyFlakesLinter = PyFlakesLinter()
        PyFlakes(self._merge, self.uid, self.vid, lint, self.settings, code, filename)

    def pep8(self, code: str = None, filename: str = None) -> None:
        lint: Pep8Linter = Pep8Linter()
        PEP8(self._merge, self.uid, self.vid, lint, self.settings, code, filename)

    def pep257(self, code: str = None, filename: str = None) -> None:
        lint: AnacondaPep257 = AnacondaPep257()
        ignore: List[str] = self.settings.get('pep257_ignore', [])
        PEP257(self._merge, self.uid, self.vid, lint, ignore, code, filename)

    def pylint(self, code: str = None, filename: str = None) -> None:
        if not PYLINT_AVAILABLE:
            errors: str = "Your configured python interpreter can't import pylint"
            self._failures.append(errors)
            return
        rcfile: str = self.settings.get('pylint_rcfile', False)
        if numversion < (2, 4, 4):
            PyLint(partial(self._normalize, self.settings), self.uid, self.vid, PyLinter, rcfile, filename)
        else:
            PyLint(self._normalize, self.uid, self.vid, PyLinter, rcfile, filename)

    def import_validator(self, code: str, filename: str = None) -> None:
        lint: Validator = Validator()
        ImportValidator(self._merge, self.uid, self.vid, lint, code, filename, self.settings)

    def mypy(self, code: str = None, filename: str = None) -> None:
        lint: AnacondaMyPy = AnacondaMyPy()
        MyPy(self._merge, self.uid, self.vid, lint, code, filename, self.mypypath, self.settings)

    def _normalize(self, data: Dict[str, Any]) -> None:
        normalized_errors: List[Dict[str, Union[bool, str, int]]] = []
        for error_level, error_data in data.get('errors', {}).items():
            pylint_ignores: List[str] = self.settings.get('pylint_ignore', [])
            pylint_rcfile: str = self.settings.get('pylint_rcfile')
            for error in error_data:
                try:
                    if error['code'] in pylint_ignores and (not pylint_rcfile):
                        continue
                except TypeError:
                    print('Anaconda: pylint_ignore option must be a list of strings but we got a {} '.format(type(pylint_ignores)))
                normalized_error: Dict[str, Union[bool, str, int]] = {'underline_range': True, 'level': error_level, 'message': error['message'], 'offset': int(error.get('offset', 0)), 'lineno': int(error['line'])}
                normalized_errors.append(normalized_error)
        if data.get('errors') is not None:
            data['errors'] = normalized_errors
        self._merge(data)

    def _configure_linters(self) -> None:
        self._linters['pyflakes'] = self.settings.get('use_pyflakes', True)
        self._linters['pylint'] = self.settings.get('use_pylint', False)
        self._linters['pep257'] = self.settings.get('use_pep257', False)
        self._linters['mypy'] = self.settings.get('use_mypy', False)
        self._linters['pep8'] = self.settings.get('pep8', True)
        self._linters['import_validator'] = self.settings.get('validate_imports', False)
        if self._linters['pylint'] is True:
            self._linters['pyflakes'] = False
        if self._linters['mypy']:
            self.mypypath: str = self.settings.get('mypypath')

    def _merge(self, lint_result: Dict[str, Union[bool, List[str], str]]) -> None:
        if lint_result['success'] is True:
            self._errors += lint_result['errors']
        else:
            self._failures.append(lint_result['error'])
