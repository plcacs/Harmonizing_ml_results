import glob
import os.path
import sys
import subprocess
import pyperf
from typing import List

if (__name__ == '__main__'):
    runner: pyperf.Runner = pyperf.Runner()
    runner.metadata['description'] = 'Performance of the Python 2to3 program'
    args = runner.parse_args()
    datadir: str = os.path.join(os.path.dirname(__file__), 'data', '2to3')
    pyfiles: List[str] = glob.glob(os.path.join(datadir, '*.py.txt'))
    command: List[str] = [sys.executable, '-m', 'lib2to3', '-f', 'all'] + pyfiles
    try:
        import lib2to3
    except ModuleNotFoundError:
        vendor: str = os.path.join(os.path.dirname(__file__), 'vendor')
        subprocess.run([sys.executable, '-m', 'pip', 'install', vendor], check=True)
    runner.bench_command('2to3', command)
