#!/usr/bin/env python3
from typing import List, Any
import glob
import os.path
import sys
import subprocess
import pyperf


def main() -> None:
    runner: pyperf.Runner = pyperf.Runner()
    runner.metadata['description'] = 'Performance of the Python 2to3 program'
    args: Any = runner.parse_args()
    datadir: str = os.path.join(os.path.dirname(__file__), 'data', '2to3')
    pyfiles: List[str] = glob.glob(os.path.join(datadir, '*.py.txt'))
    command: List[str] = [sys.executable, '-m', 'lib2to3', '-f', 'all'] + pyfiles

    try:
        import lib2to3  # type: ignore
    except ModuleNotFoundError:
        vendor: str = os.path.join(os.path.dirname(__file__), 'vendor')
        subprocess.run([sys.executable, '-m', 'pip', 'install', vendor], check=True)
    runner.bench_command('2to3', command)


if __name__ == '__main__':
    main()