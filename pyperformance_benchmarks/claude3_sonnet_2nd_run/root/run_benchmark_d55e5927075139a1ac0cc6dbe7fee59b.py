from typing import Dict, Callable, Any, Deque, List
import collections
from pathlib import Path
import networkx
import pyperf

DATA_FILE: Path = ((Path(__file__).parent / 'data') / 'amazon0302.txt.gz')
graph: networkx.Graph = networkx.read_adjlist(DATA_FILE)

def bench_shortest_path() -> None:
    collections.deque(networkx.shortest_path_length(graph, '0'))

def bench_connected_components() -> None:
    networkx.number_connected_components(graph)

def bench_k_core() -> None:
    networkx.k_core(graph)

BENCHMARKS: Dict[str, Callable[[], None]] = {'shortest_path': bench_shortest_path, 'connected_components': bench_connected_components, 'k_core': bench_k_core}

def add_cmdline_args(cmd: List[str], args: Any) -> None:
    cmd.append(args.benchmark)

def add_parser_args(parser: Any) -> None:
    parser.add_argument('benchmark', choices=BENCHMARKS, help='Which benchmark to run.')

if (__name__ == '__main__'):
    runner: pyperf.Runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.metadata['description'] = 'NetworkX benchmark'
    add_parser_args(runner.argparser)
    args: Any = runner.parse_args()
    benchmark: str = args.benchmark
    runner.bench_func(args.benchmark, BENCHMARKS[args.benchmark])
