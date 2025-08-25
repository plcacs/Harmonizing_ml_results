import sys
import subprocess
import pyperf
from pyperformance.venv import get_venv_program
from typing import Any

def get_hg_version(hg_bin: str) -> str:
    try:
        from mercurial.__version__ import version  # type: Any
        if isinstance(version, bytes):
            return version.decode('utf8')
        else:
            return version  # type: ignore
    except ImportError:
        pass
    proc: subprocess.Popen[Any] = subprocess.Popen(
        [sys.executable, hg_bin, '--version'],
        stdout=subprocess.PIPE,
        universal_newlines=True
    )
    stdout: str = proc.communicate()[0]
    if proc.returncode:
        print('ERROR: Mercurial command failed!')
        sys.exit(proc.returncode)
    return stdout.splitlines()[0]

if __name__ == '__main__':
    runner: pyperf.Runner = pyperf.Runner(values=25)
    runner.metadata['description'] = 'Performance of the Python startup'
    args = runner.parse_args()  # type: Any
    hg_bin: str = get_venv_program('hg')
    runner.metadata['hg_version'] = get_hg_version(hg_bin)
    command: list[str] = [sys.executable, hg_bin, 'help']
    runner.bench_command('hg_startup', command)