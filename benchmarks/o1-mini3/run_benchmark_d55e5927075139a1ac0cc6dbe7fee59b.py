"""
Some graph algorithm benchmarks using networkx

This uses the public domain Amazon data set from the SNAP benchmarks:

    https://snap.stanford.edu/data/amazon0302.html

Choice of benchmarks inspired by Timothy Lin's work here:

    https://www.timlrx.com/blog/benchmark-of-popular-graph-network-packages
"""

import collections
from pathlib import Path
import networkx
import pyperf
import argparse
from typing import Callable, Dict, List, Any


DATA_FILE: Path = (Path(__file__).parent / 'data') / 'amazon0302.txt.gz'
graph: networkx.Graph = networkx.read_adjlist(DATA_FILE)


def bench_shortest_path() -> None:
    collections.deque(networkx.shortest_path_length(graph, '0'))


def bench_connected_components() -> None:
    networkx.number_connected_components(graph)


def bench_k_core() -> None:
    networkx.k_core(graph)


BENCHMARKS: Dict[str, Callable[[], None]] = {
    'shortest_path': bench_shortest_path,
    'connected_components': bench_connected_components,
    'k_core': bench_k_core
}


def add_cmdline_args(cmd: List[str], args: Any) -> None:
    cmd.append(args.benchmark)


def add_parser_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('benchmark', choices=BENCHMARKS, help='Which benchmark to run.')


if __name__ == '__main__':
    runner: pyperf.Runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.metadata['description'] = 'NetworkX benchmark'
    add_parser_args(runner.argparser)
    args: Any = runner.parse_args()
    benchmark: str = args.benchmark
    runner.bench_func(args.benchmark, BENCHMARKS[args.benchmark])
