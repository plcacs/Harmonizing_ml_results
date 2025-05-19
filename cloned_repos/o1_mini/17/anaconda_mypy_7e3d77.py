"""
Anaconda MyPy wrapper
"""
import os
import sys
import shlex
import logging
import subprocess
from subprocess import PIPE, Popen, STARTUPINFO
from typing import Optional, Tuple, List, Dict, Any

def parse_mypy_version() -> Optional[Tuple[int, int, int]]:
    try:
        from mypy import main as mypy
        version = mypy.__version__
        if 'dev' in version:
            version = version.split('+dev.')[0]
            version = version.replace('-dev', '')
        tuple_version = tuple(int(i) for i in version.split('.'))
        while len(tuple_version) < 3:
            tuple_version += (0,)
        return tuple_version
    except ImportError:
        print('MyPy is enabled but we could not import it')
        logging.info('MyPy is enabled but we could not import it')
        return None

class MyPy:
    """MyPy class for Anaconda
    """
    VERSION: Optional[Tuple[int, int, int]] = parse_mypy_version()

    def __init__(self, code: str, filename: str, mypypath: Optional[str], settings: List[str]) -> None:
        self.code: str = code
        self.filename: str = filename
        self.mypypath: Optional[str] = mypypath
        self.settings: List[str] = settings

    @property
    def silent(self) -> bool:
        """Returns True if --silent-imports setting is present
        """
        return '--silent-imports' in self.settings

    def execute(self) -> List[Dict[str, Any]]:
        """Check the code with MyPy check types
        """
        if MyPy.VERSION is None:
            raise RuntimeError('MyPy was not found')
        errors: List[Dict[str, Any]] = []
        try:
            errors = self.check_source()
        except Exception as error:
            print(error)
            logging.error(error)
        return errors

    def check_source(self) -> List[Dict[str, Any]]:
        """Wrap calls to MyPy as a library
        """
        err_sum: str = '--no-error-summary'
        if MyPy.VERSION < (0, 761, 0):
            err_sum = ''
        err_ctx: str = '--hide-error-context'
        if MyPy.VERSION < (0, 4, 5):
            err_ctx = '--suppress-error-context'
        dont_follow_imports: str = '--follow-imports silent'
        args: List[str] = shlex.split(
            f"'{sys.executable}' -O -m mypy {err_sum} {err_ctx} {dont_follow_imports} {' '.join(self.settings[:-1])} '{self.filename}'"
        )
        env: Dict[str, str] = os.environ.copy()
        if self.mypypath:
            env['MYPYPATH'] = self.mypypath
        kwargs: Dict[str, Any] = {
            'cwd': os.path.dirname(os.path.abspath(__file__)),
            'bufsize': -1,
            'env': env
        }
        if os.name == 'nt':
            startupinfo: STARTUPINFO = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            kwargs['startupinfo'] = startupinfo
        proc: Popen = Popen(args, stdout=PIPE, stderr=PIPE, **kwargs)
        out: bytes
        err: bytes
        out, err = proc.communicate()
        if err and len(err) > 0:
            if sys.version_info >= (3,):
                err = err.decode('utf8')
            raise RuntimeError(err)
        if sys.version_info >= (3,):
            out = out.decode('utf8')
        errors = []
        for line in out.splitlines():
            if self.settings[-1] and not self.silent and 'stub' in line.lower():
                continue
            data: List[str] = line.split(':', maxsplit=3) if os.name != 'nt' else line[2:].split(':', maxsplit=3)
            errors.append({
                'level': 'W',
                'lineno': int(data[1]),
                'offset': 0,
                'code': ' ',
                'raw_error': f"[W] MyPy {data[2].strip()}: {data[3].strip()}",
                'message': '[W] MyPy%s: %s',
                'underline_range': True
            })
        return errors
