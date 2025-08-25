import json
import sys
import pyperf
from typing import Any, Dict, List, Tuple, Iterable
import argparse

EMPTY: Tuple[Dict[str, Any], int] = ({}, 2000)
SIMPLE_DATA: Dict[str, Any] = {
    'key1': 0,
    'key2': True,
    'key3': 'value',
    'key4': 'foo',
    'key5': 'string'
}
SIMPLE: Tuple[Dict[str, Any], int] = (SIMPLE_DATA, 1000)
NESTED_DATA: Dict[str, Any] = {
    'key1': 0,
    'key2': SIMPLE[0],
    'key3': 'value',
    'key4': SIMPLE[0],
    'key5': SIMPLE[0],
    'key': 'ąćż'
}
NESTED: Tuple[Dict[str, Any], int] = (NESTED_DATA, 1000)
HUGE: Tuple[List[Dict[str, Any]], int] = (([NESTED[0]] * 1000), 1)
CASES: List[str] = ['EMPTY', 'SIMPLE', 'NESTED', 'HUGE']

def bench_json_dumps(data: List[Tuple[Any, Iterable[int]]]) -> None:
    for (obj, count_it) in data:
        for _ in count_it:
            json.dumps(obj)

def add_cmdline_args(cmd: List[str], args: argparse.Namespace) -> None:
    if args.cases:
        cmd.extend(('--cases', args.cases))

def main() -> None:
    runner: pyperf.Runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.argparser.add_argument(
        '--cases',
        help=(
            'Comma separated list of cases. Available cases: %s. '
            'By default, run all cases.' % ', '.join(CASES)
        )
    )
    runner.metadata['description'] = 'Benchmark json.dumps()'
    args: argparse.Namespace = runner.parse_args()
    if args.cases:
        cases: List[str] = []
        for case in args.cases.split(','):
            case = case.strip()
            if case:
                cases.append(case)
        if not cases:
            print('ERROR: empty list of cases')
            sys.exit(1)
    else:
        cases = CASES
    data: List[Tuple[Any, Iterable[int]]] = []
    for case in cases:
        (obj, count) = globals()[case]
        data.append((obj, range(count)))
    runner.bench_func('json_dumps', bench_json_dumps, data)

if __name__ == '__main__':
    main()
