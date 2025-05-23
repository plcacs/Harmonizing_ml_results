'\nBenchmark Python startup.\n'
import sys
import pyperf
from typing import List
from argparse import Namespace

def add_cmdline_args(cmd: List[str], args: Namespace) -> None:
    if args.no_site:
        cmd.append('--no-site')
    if args.exit:
        cmd.append('--exit')

if __name__ == '__main__':
    runner: pyperf.Runner = pyperf.Runner(values=10, add_cmdline_args=add_cmdline_args)
    runner.argparser.add_argument('--no-site', action='store_true')
    runner.argparser.add_argument('--exit', action='store_true')
    runner.metadata['description'] = 'Performance of the Python startup'
    args: Namespace = runner.parse_args()
    name: str = 'python_startup'
    if args.no_site:
        name += '_no_site'
    if args.exit:
        name += '_exit'
    command: List[str] = [sys.executable]
    if args.no_site:
        command.append('-S')
    if args.exit:
        command.extend(('-c', 'import os; os._exit(0)'))
    else:
        command.extend(('-c', 'pass'))
    runner.bench_command(name, command)
