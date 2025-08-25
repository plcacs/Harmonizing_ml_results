from typing import List
import collections
from pathlib import Path
import networkx
import pyperf

DATA_FILE: Path = ((Path(__file__).parent / 'data') / 'amazon0302.txt.gz')
graph: networkx.Graph = networkx.read_adjlist(DATA_FILE)

def bench_shortest_path() -> None:
    collections.deque(networkx.shortest_path_length(graph, '0'))

def bench_connected_components() -> int:
    return networkx.number_connected_components(graph)

def bench_k_core() -> networkx.Graph:
    return networkx.k_core(graph)

BENCHMARKS: dict = {'shortest_path': bench_shortest_path, 'connected_components': bench_connected_components, 'k_core': bench_k_core}

def add_cmdline_args(cmd: List[str], args: pyperf.BenchmarkOptions) -> None:
    cmd.append(args.benchmark)

def add_parser_args(parser: pyperf.ArgParser) -> None:
    parser.add_argument('benchmark', choices=BENCHMARKS, help='Which benchmark to run.')

if (__name__ == '__main__'):
    runner: pyperf.Runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.metadata['description'] = 'NetworkX benchmark'
    add_parser_args(runner.argparser)
    args: pyperf.BenchmarkOptions = runner.parse_args()
    benchmark: str = args.benchmark
    runner.bench_func(args.benchmark, BENCHMARKS[args.benchmark])
