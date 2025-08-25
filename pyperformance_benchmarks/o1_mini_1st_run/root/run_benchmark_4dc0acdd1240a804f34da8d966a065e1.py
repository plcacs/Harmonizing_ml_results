"Microbenchmark for Python's sequence unpacking."
import pyperf
from typing import Sequence, List, Tuple, Dict, Callable
import argparse

def do_unpacking(loops: int, to_unpack: Sequence[int]) -> float:
    range_it = range(loops)
    t0 = pyperf.perf_counter()
    for _ in range_it:
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
        (a, b, c, d, e, f, g, h, i, j) = to_unpack
    return (pyperf.perf_counter() - t0)

def bench_tuple_unpacking(loops: int) -> float:
    x: Tuple[int, ...] = tuple(range(10))
    return do_unpacking(loops, x)

def bench_list_unpacking(loops: int) -> float:
    x: List[int] = list(range(10))
    return do_unpacking(loops, x)

def bench_all(loops: int) -> float:
    dt1: float = bench_tuple_unpacking(loops)
    dt2: float = bench_list_unpacking(loops)
    return (dt1 + dt2)

def add_cmdline_args(cmd: List[str], args: argparse.Namespace) -> None:
    if args.benchmark:
        cmd.append(args.benchmark)

if (__name__ == '__main__'):
    benchmarks: Dict[str, Callable[[int], float]] = {
        'tuple': bench_tuple_unpacking, 
        'list': bench_list_unpacking
    }
    runner: pyperf.Runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.metadata['description'] = "Microbenchmark for Python's sequence unpacking."
    runner.argparser.add_argument('benchmark', nargs='?', choices=sorted(benchmarks))
    options: argparse.Namespace = runner.parse_args()
    name: str = 'unpack_sequence'
    if options.benchmark:
        func: Callable[[int], float] = benchmarks[options.benchmark]
        name += f'_{options.benchmark}'
    else:
        func: Callable[[int], float] = bench_all
    runner.bench_time_func(name, func, inner_loops=400)
