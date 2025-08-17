#!/usr/bin/env python3
# Copyright (C) 2013 - 2016 - Oscar Campos <oscar.campos@member.fsf.org>
# This program is Free Software see LICENSE file for details

"""
Anaconda MyPy wrapper
"""

import os
import sys
import shlex
import logging
import subprocess
from subprocess import PIPE, Popen
from typing import List, Tuple, Dict, Any, Optional

def parse_mypy_version() -> Optional[Tuple[int, int, int]]:
    try:
        from mypy import main as mypy  # type: ignore
        version: str = mypy.__version__
        if "dev" in version:
            # Handle when Mypy is installed directly from github:
            # eg: 0.730+dev.ddec163790d107f1fd9982f19cbfa0b6966c2eea
            version = version.split("+dev.")[0]
            # Handle old style Mypy version: 0.480-dev
            version = version.replace('-dev', '')

        tuple_version: Tuple[int, ...] = tuple(int(i) for i in version.split('.'))
        while len(tuple_version) < 3:
            tuple_version += (0,)
        return tuple_version  # type: ignore
    except ImportError:
        print('MyPy is enabled but we could not import it')
        logging.info('MyPy is enabled but we could not import it')
        return None

class MyPy:
    """MyPy class for Anaconda
    """
    VERSION: Optional[Tuple[int, int, int]] = parse_mypy_version()

    def __init__(self, code: str, filename: str, mypypath: str, settings: List[str]) -> None:
        self.code: str = code
        self.filename: str = filename
        self.mypypath: str = mypypath
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
            raise RuntimeError("MyPy was not found")

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
        if MyPy.VERSION is not None and MyPy.VERSION < (0, 761, 0):
            err_sum = ''

        err_ctx: str = '--hide-error-context'
        if MyPy.VERSION is not None and MyPy.VERSION < (0, 4, 5):
            err_ctx = '--suppress-error-context'

        dont_follow_imports: str = "--follow-imports silent"

        args_str: str = '\'{}\' -O -m mypy {} {} {} {} \'{}\''.format(
            sys.executable,
            err_sum,
            err_ctx,
            dont_follow_imports,
            ' '.join(self.settings[:-1]),
            self.filename
        )
        args: List[str] = shlex.split(args_str)
        env: Dict[str, str] = os.environ.copy()
        if self.mypypath is not None and self.mypypath != "":
            env['MYPYPATH'] = self.mypypath

        kwargs: Dict[str, Any] = {
            'cwd': os.path.dirname(os.path.abspath(__file__)),
            'bufsize': -1,
            'env': env
        }
        if os.name == 'nt':
            startupinfo: subprocess.STARTUPINFO = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            kwargs['startupinfo'] = startupinfo

        proc: Popen = Popen(args, stdout=PIPE, stderr=PIPE, **kwargs)
        out_bytes, err_bytes = proc.communicate()
        if err_bytes is not None and len(err_bytes) > 0:
            if sys.version_info >= (3,):
                err_str: str = err_bytes.decode('utf8')
            else:
                err_str = err_bytes
            raise RuntimeError(err_str)

        if sys.version_info >= (3,):
            out_str: str = out_bytes.decode('utf8')
        else:
            out_str = out_bytes

        errors: List[Dict[str, Any]] = []
        for line in out_str.splitlines():
            if (self.settings[-1] and not self.silent and 'stub' in line.lower()):
                continue

            # On Windows, skip the drive letter display
            data: List[str] = line[2:].split(':', maxsplit=3) if os.name == 'nt' else line.split(':', maxsplit=3)
            errors.append({
                'level': 'W',
                'lineno': int(data[1]),
                'offset': 0,
                'code': ' ',
                'raw_error': '[W] MyPy {}: {}'.format(data[2].strip(), data[3].strip()),
                'message': '[W] MyPy{}: {}'.format(data[2].strip(), data[3].strip()),
                'underline_range': True
            })

        return errors
